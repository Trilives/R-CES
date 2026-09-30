"""Stratified (within-stratum) AUROC: does the score use the image, or the rule prior?

Motivation (reviewer-driven). Pooled AUROC over (rule, checkpoint) examples mixes
two sources of ranking signal:

  * *between-stratum* variance -- some rules/checkpoints are simply resolvable more
    often than others. A model that reads only the rule text recovers this, because
    the stratum identity is a deterministic function of the text.
  * *within-stratum* variance -- among pairs sharing the same rule (hence the same
    conditioning text and the same critical-checkpoint set), which images actually
    carry the evidence. Only the image can supply this.

R-CES's critical checkpoint set is a property of the *rule*, not of the pair
(``exp_io.resolvable_targets`` reads ``critical_checkpoints(rule)``), so a
text-only head emits a constant score inside a stratum and scores exactly 0.5
here. Within-stratum AUROC is therefore a shortcut-free test of image use:
chance is 0.5 by construction, not by assumption.

Two aggregations are reported. ``macro`` averages per-stratum AUROC unweighted;
``pooled`` weights each stratum by its number of positive-negative pairs, which is
the Mann-Whitney U estimator restricted to within-stratum comparisons.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence

import numpy as np

from metrics import _auroc


def stratified_auroc(
    scores: Sequence[float],
    targets: Sequence[int],
    strata: Sequence[Hashable],
) -> dict[str, float | int]:
    """AUROC computed inside each stratum, then aggregated.

    Strata with a single class contribute no comparable pairs and are skipped;
    the number retained is reported so a near-empty result is not mistaken for a
    confident one.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(targets, dtype=int)
    if not (len(s) == len(y) == len(strata)):
        raise ValueError("scores, targets and strata must be equal-length")
    if len(y) and not np.isin(y, (0, 1)).all():
        raise ValueError("targets must contain only 0 and 1")

    by_stratum: dict[Hashable, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        by_stratum[stratum].append(index)

    per_stratum: list[float] = []
    weights: list[float] = []
    for stratum in sorted(by_stratum, key=repr):
        index = np.asarray(by_stratum[stratum], dtype=int)
        positives = int((y[index] == 1).sum())
        negatives = int((y[index] == 0).sum())
        if not positives or not negatives:
            continue
        per_stratum.append(_auroc(s[index], y[index]))
        weights.append(float(positives * negatives))

    if not per_stratum:
        return {
            "macro": float("nan"),
            "pooled": float("nan"),
            "n_strata_used": 0,
            "n_strata_total": len(by_stratum),
            "n_comparable_pairs": 0,
        }
    values = np.asarray(per_stratum, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    return {
        "macro": float(values.mean()),
        "pooled": float((values * weight_array).sum() / weight_array.sum()),
        "n_strata_used": len(per_stratum),
        "n_strata_total": len(by_stratum),
        "n_comparable_pairs": int(weight_array.sum()),
    }


def bootstrap_stratified_auroc(
    scores: Sequence[float],
    targets: Sequence[int],
    strata: Sequence[Hashable],
    *,
    groups: Sequence[Hashable],
    aggregation: str = "pooled",
    n_bootstrap: int = 5_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Cluster bootstrap interval for :func:`stratified_auroc`.

    Whole groups (image IDs) are resampled, matching ``metrics.bootstrap_auroc``,
    so checkpoints sharing an image stay correlated. Replicates in which no
    stratum retains both classes are discarded and counted.
    """
    if aggregation not in {"macro", "pooled"}:
        raise ValueError(f"aggregation must be macro or pooled; got {aggregation!r}")
    s = np.asarray(scores, dtype=float)
    y = np.asarray(targets, dtype=int)
    stratum_array = np.asarray(strata, dtype=object)
    group_array = np.asarray(groups, dtype=object)

    index_by_group: dict[Hashable, list[int]] = defaultdict(list)
    for index, group in enumerate(group_array):
        index_by_group[group].append(index)
    unique_groups = sorted(index_by_group, key=repr)

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        picked = rng.integers(0, len(unique_groups), size=len(unique_groups))
        index = np.concatenate([index_by_group[unique_groups[i]] for i in picked])
        result = stratified_auroc(s[index], y[index], stratum_array[index])
        value = float(result[aggregation])
        if not np.isnan(value):
            values.append(value)

    point = stratified_auroc(s, y, stratum_array)
    if not values:
        return {**point, "ci_95_low": float("nan"), "ci_95_high": float("nan"),
                "ci_contains_0_5": False, "valid_bootstrap_samples": 0,
                "n_groups": len(unique_groups), "aggregation": aggregation}
    array = np.asarray(values, dtype=float)
    low = float(np.percentile(array, 2.5))
    high = float(np.percentile(array, 97.5))
    return {
        **point,
        "aggregation": aggregation,
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_contains_0_5": bool(low <= 0.5 <= high),
        "valid_bootstrap_samples": len(values),
        "n_groups": len(unique_groups),
    }


def mean_of_seed_aurocs(
    per_seed_frames,
    stratum_field: str,
    *,
    seed: int = 0,
    n_bootstrap: int = 3000,
) -> dict[str, float | int | bool | str]:
    """Two-level interval for the within-stratum AUROC of a randomly seeded head.

    The estimand is what a deployer gets: train one head with a random seed, run it
    on a fresh sample of test images. Each replicate therefore

      1. draws ONE image bootstrap sample -- shared across seeds, because a single
         test set is common to every training run;
      2. draws seeds with replacement;
      3. computes the within-stratum AUROC **separately for each drawn seed**;
      4. averages those per-seed values.

    Step 3 is what distinguishes this from pooling every seed's rows into one AUROC:
    pooling ranks scores emitted by *different* heads against each other, and each
    head has its own score scale, so the pooled figure estimates nothing anyone
    deploys. The point estimate returned here equals the mean of the per-seed
    AUROCs, which is the consistency check that the estimand is the intended one.

    `per_seed_frames` is a list of dicts with aligned "score", "target",
    "image_id" and `stratum_field` sequences -- one dict per training seed.
    """
    rng = np.random.default_rng(seed)
    prepared = []
    for frame in per_seed_frames:
        by_image: dict[Hashable, list[int]] = defaultdict(list)
        for index, image_id in enumerate(frame["image_id"]):
            by_image[image_id].append(index)
        prepared.append((
            np.asarray(frame["score"], dtype=float),
            np.asarray(frame["target"], dtype=int),
            np.asarray(frame[stratum_field], dtype=object),
            by_image,
        ))
    images = sorted({i for frame in per_seed_frames for i in frame["image_id"]}, key=repr)
    n_seeds = len(prepared)

    values: list[float] = []
    for _ in range(n_bootstrap):
        drawn = [images[i] for i in rng.integers(0, len(images), size=len(images))]
        per_seed: list[float] = []
        for seed_index in rng.integers(0, n_seeds, size=n_seeds):
            scores, targets, strata, by_image = prepared[seed_index]
            rows = np.concatenate([by_image[img] for img in drawn if img in by_image])
            value = float(stratified_auroc(scores[rows], targets[rows], strata[rows])["pooled"])
            if not np.isnan(value):
                per_seed.append(value)
        if per_seed:
            values.append(float(np.mean(per_seed)))

    point = float(np.mean([
        stratified_auroc(b[0], b[1], b[2])["pooled"] for b in prepared
    ]))
    array = np.asarray(values, dtype=float)
    low = float(np.percentile(array, 2.5)) if len(array) else float("nan")
    high = float(np.percentile(array, 97.5)) if len(array) else float("nan")
    return {
        "estimand": "mean over seeds of within-stratum AUROC (no cross-head score pooling)",
        "point_mean_of_seeds": point,
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_contains_0_5": bool(low <= 0.5 <= high) if len(array) else True,
        "n_seeds": n_seeds,
        "valid_bootstrap_samples": int(len(array)),
    }
