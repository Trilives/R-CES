"""Selective-prediction metrics for R-CES.

A selective predictor answers when a confidence score >= tau and abstains
otherwise. Conventions used throughout:

  score    : higher == more confident the pair is answerable (R-CES sufficiency
             or another offline method's score).
  correct  : 1 if the downstream answer (compliance / sufficiency decision) would
             be right if answered, else 0.

Reported:
  * risk-coverage curve + AURC (area under the risk-coverage curve, lower better)
  * coverage @ target risk
  * effective reliability (Whitehead et al. 2022): per answered item +1 if
    correct, -1 if wrong, abstain 0; normalised by N.
  * ECE over the confidence scores vs correctness.
  * Critical Missing Checkpoint Recall (CMCR) -- R-CES's custom metric: of the
    pairs that are truly undecidable because a critical checkpoint is missing
    (or the rule is non-visual-in-principle), the fraction the method abstains on.
    This is the failure mode where an answerer may remain overconfident despite
    unresolved critical evidence.

Pure numpy; no sklearn dependency.
"""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np


def _auroc(scores: np.ndarray, targets: np.ndarray) -> float:
    positive_count = int((targets == 1).sum())
    negative_count = int((targets == 0).sum())
    if not positive_count or not negative_count:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[targets == 1].sum())
    mann_whitney_u = positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(mann_whitney_u / (positive_count * negative_count))


def bootstrap_auroc(
    scores: Sequence[float],
    targets: Sequence[int],
    *,
    groups: Sequence[Hashable] | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int | bool | str]:
    """Percentile bootstrap interval for AUROC.

    When ``groups`` is provided, whole groups are sampled with replacement.
    R-CES callers supply original-image family IDs, keeping originals, crops
    and their checkpoints together. Replicates containing only one
    class are excluded and the valid count is reported explicitly.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(targets, dtype=int)
    if s.ndim != 1 or y.ndim != 1 or len(s) != len(y):
        raise ValueError("scores and targets must be equal-length one-dimensional sequences")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("targets must contain only 0 and 1")
    if not len(y):
        return {
            "auroc": float("nan"),
            "ci_95_low": float("nan"),
            "ci_95_high": float("nan"),
            "ci_contains_0_5": False,
            "positives": 0,
            "negatives": 0,
            "n": 0,
            "bootstrap_unit": "cluster" if groups is not None else "example",
            "n_groups": 0,
            "requested_bootstrap_samples": int(n_bootstrap),
            "valid_bootstrap_samples": 0,
        }

    rng = np.random.default_rng(seed)
    group_array = None if groups is None else np.asarray(groups, dtype=object)
    if group_array is not None and len(group_array) != len(y):
        raise ValueError("groups must have the same length as scores and targets")

    bootstrap_values: list[float] = []
    if group_array is None:
        for _ in range(n_bootstrap):
            indices = rng.integers(0, len(y), size=len(y))
            value = _auroc(s[indices], y[indices])
            if np.isfinite(value):
                bootstrap_values.append(value)
        bootstrap_unit = "example"
        n_groups = len(y)
    else:
        unique_groups = np.asarray(list(dict.fromkeys(group_array.tolist())), dtype=object)
        group_indices = {group: np.flatnonzero(group_array == group) for group in unique_groups}
        for _ in range(n_bootstrap):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            indices = np.concatenate([group_indices[group] for group in sampled_groups])
            value = _auroc(s[indices], y[indices])
            if np.isfinite(value):
                bootstrap_values.append(value)
        bootstrap_unit = "cluster"
        n_groups = len(unique_groups)

    if not bootstrap_values:
        low = high = float("nan")
    else:
        low, high = np.quantile(np.asarray(bootstrap_values), [0.025, 0.975]).tolist()
    return {
        "auroc": _auroc(s, y),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "ci_contains_0_5": bool(np.isfinite(low) and low <= 0.5 <= high),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "n": int(len(y)),
        "bootstrap_unit": bootstrap_unit,
        "n_groups": int(n_groups),
        "requested_bootstrap_samples": int(n_bootstrap),
        "valid_bootstrap_samples": int(len(bootstrap_values)),
    }


def binary_classification_metrics(
    scores: Sequence[float],
    targets: Sequence[int],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """AUROC, average precision, and F1 for a binary probabilistic output."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(targets, dtype=int)
    if len(s) != len(y):
        raise ValueError("scores and targets must have equal length")
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if not len(y):
        return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"), "n": 0}

    auroc = _auroc(s, y)

    if positives:
        order = np.argsort(-s, kind="stable")
        sorted_s, sorted_y = s[order], y[order]
        ends = np.flatnonzero(np.r_[sorted_s[:-1] != sorted_s[1:], True]) + 1
        cumulative_positive = np.cumsum(sorted_y == 1)[ends - 1]
        precision = cumulative_positive / ends
        recall_steps = np.diff(np.r_[0, cumulative_positive]) / positives
        auprc = float(np.sum(recall_steps * precision))
    else:
        auprc = float("nan")

    predicted = s >= threshold
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    return {"auroc": float(auroc), "auprc": float(auprc), "f1": float(f1), "n": int(len(y))}


def risk_coverage_curve(scores: Sequence[float], correct: Sequence[int]) -> dict[str, np.ndarray]:
    """Realizable ``score >= tau`` decisions, ordered by increasing coverage.

    Each point accepts an entire equal-score group. Splitting a tie would report
    a coverage/risk pair that no score threshold can implement. The zero-coverage
    abstain-all decision is implicit; its conditional risk is undefined.
    """
    s = np.asarray(scores, dtype=float)
    c = np.asarray(correct, dtype=float)
    if s.ndim != 1 or c.ndim != 1 or len(s) != len(c):
        raise ValueError("scores and correct must be equal-length one-dimensional sequences")
    if not np.isfinite(s).all() or not np.isin(c, (0, 1)).all():
        raise ValueError("scores must be finite and correct must contain only 0 and 1")
    if not len(s):
        return {key: np.array([], dtype=float) for key in ("coverage", "risk", "thresholds")}
    order = np.argsort(-s, kind="stable")
    sorted_s, sorted_c = s[order], c[order]
    answered = np.flatnonzero(np.r_[sorted_s[:-1] != sorted_s[1:], True]) + 1
    errors = np.cumsum(1 - sorted_c)[answered - 1]
    return {"coverage": answered / len(s), "risk": errors / answered,
            "thresholds": sorted_s[answered - 1]}


def aurc(scores: Sequence[float], correct: Sequence[int]) -> float:
    """Right-endpoint step integral over complete tie groups (lower is better).

    AURC = sum_g (coverage_g - coverage_{g-1}) * risk_g, coverage_0 = 0.
    With distinct scores this is the usual mean of prefix risks. Constant
    scores give the full-population risk, independently of input row order.
    """
    curve = risk_coverage_curve(scores, correct)
    if not len(curve["coverage"]):
        return float("nan")
    return float(np.sum(np.diff(np.r_[0.0, curve["coverage"]]) * curve["risk"]))


def coverage_at_risk(scores: Sequence[float], correct: Sequence[int], target_risk: float) -> float:
    """Largest realizable coverage with conditional risk <= target_risk."""
    if not 0 <= target_risk <= 1:
        raise ValueError("target_risk must lie in [0, 1]")
    curve = risk_coverage_curve(scores, correct)
    ok = curve["risk"] <= target_risk
    if not ok.any():
        return 0.0
    return float(curve["coverage"][ok].max())


def effective_reliability(scores: Sequence[float], correct: Sequence[int], tau: float) -> float:
    """Whitehead et al. effective reliability at threshold tau: answered-correct
    rewarded +1, answered-wrong penalised -1, abstentions 0; divided by N."""
    s = np.asarray(scores, dtype=float)
    c = np.asarray(correct, dtype=float)
    answered = s >= tau
    reward = np.where(c >= 0.5, 1.0, -1.0) * answered
    return float(reward.sum() / len(c)) if len(c) else 0.0


def expected_calibration_error(scores: Sequence[float], correct: Sequence[int], n_bins: int) -> float:
    """ECE between the confidence score (interpreted as P(correct)) and accuracy."""
    s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    c = np.asarray(correct, dtype=float)
    if len(c) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (s > lo) & (s <= hi) if lo > 0 else (s >= lo) & (s <= hi)
        if not in_bin.any():
            continue
        conf = s[in_bin].mean()
        acc = c[in_bin].mean()
        ece += (in_bin.mean()) * abs(conf - acc)
    return float(ece)


def critical_missing_checkpoint_recall(
    scores: Sequence[float], should_abstain: Sequence[int], tau: float
) -> float:
    """CMCR: among pairs that truly should abstain (a critical checkpoint is
    missing / rule non-visual-in-principle), the fraction the method abstains on
    (score < tau)."""
    s = np.asarray(scores, dtype=float)
    must = np.asarray(should_abstain, dtype=float)
    denom = must.sum()
    if denom == 0:
        return float("nan")
    abstained = (s < tau).astype(float)
    return float((abstained * must).sum() / denom)


def tune_tau_for_risk(scores: Sequence[float], correct: Sequence[int], target_risk: float) -> float:
    """Pick the most permissive admissible threshold on validation tie groups.

    If no group is admissible, return a threshold strictly above the maximum
    score (infinity for an empty validation set), accepting no samples.
    """
    if not 0 <= target_risk <= 1:
        raise ValueError("target_risk must lie in [0, 1]")
    curve = risk_coverage_curve(scores, correct)
    ok = curve["risk"] <= target_risk
    if not ok.any():
        # Keep the strong float64 scalar: NumPy otherwise rounds a Python float
        # back to float32 when compared with float32 scores, undoing nextafter.
        return np.nextafter(np.max(curve["thresholds"]), np.inf) if len(curve["thresholds"]) else float("inf")
    # thresholds are sorted descending alongside coverage ascending; the last OK
    # index is the most permissive admissible threshold.
    idx = np.where(ok)[0].max()
    return float(curve["thresholds"][idx])


def summarize(
    scores: Sequence[float],
    correct: Sequence[int],
    *,
    tau: float,
    target_risk: float,
    should_abstain: Sequence[int] | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    out = {
        "aurc": aurc(scores, correct),
        "coverage_at_target_risk": coverage_at_risk(scores, correct, target_risk),
        "effective_reliability": effective_reliability(scores, correct, tau),
        "ece": expected_calibration_error(scores, correct, n_bins),
        "tau": float(tau),
        "n": int(len(correct)),
        "answer_rate": float(np.mean(np.asarray(scores, dtype=float) >= tau)) if len(correct) else 0.0,
    }
    out.update(binary_classification_metrics(scores, correct, threshold=tau))
    if should_abstain is not None:
        out["cmcr"] = critical_missing_checkpoint_recall(scores, should_abstain, tau)
    return out
