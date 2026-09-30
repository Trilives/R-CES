"""Rule-disjoint (leave-rules-out) evaluation: does the gate transfer to unseen rules?

The paper's splits are image-disjoint, so every rule's text is seen during
training. Two reviewers observed that this cannot distinguish a general
image-rule sufficiency model from one that memorises a per-rule prior, and asked
for rule-disjoint evaluation. This driver supplies it.

K folds partition the rule library. For each fold the head is trained on the
other folds' rules and evaluated only on pairs of the held-out rules, whose text
the head has never seen. Two regimes are reported:

  strict -- rule-disjoint AND image-disjoint (train images x train rules ->
            test images x held-out rules). No shared rule and no shared image.
  rules  -- rule-disjoint only, using all images. Larger test support; retained
            because the strict regime leaves few negatives per fold.

Metrics mirror evidence_controls.py: the pooled AUROC the paper headlines and
the within-rule AUROC that holds the rule prior fixed. On held-out rules the
pooled number is the honest transfer measure, since no per-rule prior is
available to the model.

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_N_FOLDS=5 RCES_N_SEEDS=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/ablation/rule_disjoint.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import numpy as np  # noqa: E402

import config  # noqa: E402
import metrics  # noqa: E402
import stratified  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces import evaluate as rces_eval  # noqa: E402
from rces import train as rces_train  # noqa: E402

N_FOLDS = int(os.environ.get("RCES_N_FOLDS", "5"))
N_SEEDS = int(os.environ.get("RCES_N_SEEDS", "3"))
N_BOOTSTRAP = int(os.environ.get("RCES_N_BOOTSTRAP", "5000"))
TAG = config._CROPAUG_TAG


def all_rule_ids() -> list[str]:
    config.RULE_FILTER = None
    every_image = set(splits.load_split()["train"]) | set(splits.load_split()["val"]) \
        | set(splits.load_split()["test"])
    return sorted({r.rule_id for r in ds.build_pair_records(every_image)})


def make_folds(rule_ids: list[str], k: int) -> list[list[str]]:
    """Deterministic round-robin partition, so folds stay balanced in size."""
    folds: list[list[str]] = [[] for _ in range(k)]
    for index, rule_id in enumerate(rule_ids):
        folds[index % k].append(rule_id)
    return [f for f in folds if f]


def frames_for(head, image_ids, device) -> tuple[dict, dict]:
    records = ds.build_pair_records(set(image_ids))
    if not records:
        return {}, {}
    rows = rces_eval.build_eval_rows(head, set(image_ids), device)
    probs = rces_eval.checkpoint_probs(head, records, device)
    ckpt = {k: [] for k in ("score", "target", "image_id", "key")}
    for record in records:
        for key, target in record.resolvable_targets.items():
            if (record.pair_id, key) not in probs:
                continue
            ckpt["score"].append(probs[(record.pair_id, key)])
            ckpt["target"].append(int(target))
            ckpt["image_id"].append(record.image_id)
            ckpt["key"].append(f"{record.rule_id}::{key}")
    pair = {k: [] for k in ("score", "target", "image_id", "rule_id")}
    for row in rows:
        if row.decision_scope == "requires_external_evidence":
            continue
        pair["score"].append(row.visual_sufficiency_score)
        pair["target"].append(int(row.sufficiency_gold))
        pair["image_id"].append(row.image_id)
        pair["rule_id"].append(row.rule_id)
    return ckpt, pair


def main() -> None:
    device = rces_train._device()
    rule_ids = all_rule_ids()
    folds = make_folds(rule_ids, N_FOLDS)
    print(f"{len(rule_ids)} rules -> {len(folds)} folds: "
          + ", ".join(str(len(f)) for f in folds))

    split = splits.load_split()
    regimes = {
        "strict": (set(split["train"]), set(split["val"]), set(split["test"])),
        "rules": (set(split["train"]), set(split["val"]),
                  set(split["train"]) | set(split["val"]) | set(split["test"])),
    }
    # accumulated[regime] -> pooled score frames across folds and seeds
    accumulated: dict[str, dict] = {
        name: {"ckpt": {k: [] for k in ("score", "target", "image_id", "key")},
               "pair": {k: [] for k in ("score", "target", "image_id", "rule_id")}}
        for name in regimes
    }
    per_fold: list[dict] = []

    for seed in range(N_SEEDS):
        config.SEED = seed
        for fold_index, held_out in enumerate(folds):
            held = frozenset(held_out)
            config.HEAD_CKPT = config.ARTIFACT_DIR / f"rces_head_ruledisjoint_f{fold_index}{TAG}.pt"

            config.RULE_FILTER = ("exclude", held)
            rces_train.main()
            config.RULE_FILTER = ("include", held)
            head = rces_eval.load_head(device)

            row = {"seed": seed, "fold": fold_index, "held_out_rules": sorted(held)}
            for regime, (_, _, test_images) in regimes.items():
                ckpt, pair = frames_for(head, test_images, device)
                if not ckpt:
                    continue
                for level, frame in (("ckpt", ckpt), ("pair", pair)):
                    for k, v in frame.items():
                        accumulated[regime][level][k].extend(v)
                # Conditional (within-stratum) AUROC computed INSIDE the fold, so no
                # score is ever ranked against one from a head trained on a different
                # rule partition. This is the statistic that holds the rule text fixed
                # on rules the head has never seen.
                ckpt_within = stratified.stratified_auroc(
                    ckpt["score"], ckpt["target"], ckpt["key"])
                pair_within = stratified.stratified_auroc(
                    pair["score"], pair["target"], pair["rule_id"]) if pair["score"] else {}
                row[regime] = {
                    "n_checkpoint": len(ckpt["score"]),
                    "n_pair": len(pair["score"]),
                    "checkpoint_pooled_auroc": float(metrics._auroc(
                        np.asarray(ckpt["score"], float), np.asarray(ckpt["target"], int))),
                    "pair_pooled_auroc": float(metrics._auroc(
                        np.asarray(pair["score"], float), np.asarray(pair["target"], int)))
                    if pair["score"] else float("nan"),
                    "checkpoint_within_key": {
                        "pooled": ckpt_within.get("pooled", float("nan")),
                        "macro": ckpt_within.get("macro", float("nan")),
                        "n_strata_used": ckpt_within.get("n_strata_used", 0),
                        "n_strata_total": ckpt_within.get("n_strata_total", 0),
                        "n_comparable_pairs": ckpt_within.get("n_comparable_pairs", 0),
                    },
                    "pair_within_rule": {
                        "pooled": pair_within.get("pooled", float("nan")),
                        "macro": pair_within.get("macro", float("nan")),
                        "n_strata_used": pair_within.get("n_strata_used", 0),
                        "n_strata_total": pair_within.get("n_strata_total", 0),
                        "n_comparable_pairs": pair_within.get("n_comparable_pairs", 0),
                    },
                }
            per_fold.append(row)
            summary_bits = " | ".join(
                f"{r}: ckpt {row[r]['checkpoint_pooled_auroc']:.3f} "
                f"pair {row[r]['pair_pooled_auroc']:.3f}" for r in regimes if r in row)
            print(f"  seed {seed} fold {fold_index} ({len(held)} rules held out) {summary_bits}")
            config.RULE_FILTER = None

    summary: dict[str, object] = {
        "n_rules": len(rule_ids), "n_folds": len(folds), "n_seeds": N_SEEDS,
        "folds": [sorted(f) for f in folds], "per_fold": per_fold, "aggregate": {},
    }
    for regime, store in accumulated.items():
        ckpt, pair = store["ckpt"], store["pair"]
        if not ckpt["score"]:
            continue
        summary["aggregate"][regime] = {
            "checkpoint_pooled": metrics.bootstrap_auroc(
                ckpt["score"], ckpt["target"], groups=ckpt["image_id"],
                n_bootstrap=N_BOOTSTRAP),
            "checkpoint_within_key": stratified.bootstrap_stratified_auroc(
                ckpt["score"], ckpt["target"], ckpt["key"], groups=ckpt["image_id"],
                n_bootstrap=N_BOOTSTRAP),
            "pair_pooled": metrics.bootstrap_auroc(
                pair["score"], pair["target"], groups=pair["image_id"],
                n_bootstrap=N_BOOTSTRAP) if pair["score"] else None,
            "pair_within_rule": stratified.bootstrap_stratified_auroc(
                pair["score"], pair["target"], pair["rule_id"], groups=pair["image_id"],
                n_bootstrap=N_BOOTSTRAP) if pair["score"] else None,
        }

    # Fold-level summary of the within-fold conditional metric: average over seeds
    # within a fold, then describe across folds. Folds share training rules, so these
    # are descriptive counts, not an independent-sample test.
    fold_level: dict[str, object] = {}
    for regime in regimes:
        per_regime: dict[str, object] = {}
        for level in ("checkpoint_within_key", "pair_within_rule"):
            fold_means, fold_support = [], []
            for fold_index in range(len(folds)):
                vals = [r[regime][level]["pooled"] for r in per_fold
                        if r["fold"] == fold_index and regime in r
                        and np.isfinite(r[regime][level]["pooled"])]
                used = [r[regime][level]["n_strata_used"] for r in per_fold
                        if r["fold"] == fold_index and regime in r]
                if vals:
                    fold_means.append(float(np.mean(vals)))
                    fold_support.append(int(np.mean(used)) if used else 0)
            per_regime[level] = {
                "fold_means": fold_means,
                "mean_of_fold_means": float(np.mean(fold_means)) if fold_means else float("nan"),
                "folds_above_0_5": int(sum(1 for v in fold_means if v > 0.5)),
                "n_folds_with_signal": len(fold_means),
                "mean_strata_used_per_fold": fold_support,
            }
        fold_level[regime] = per_regime
    summary["fold_level_within_stratum"] = fold_level

    print("\n===== within-fold conditional AUROC (computed inside each fold) =====")
    for regime, entry in fold_level.items():
        for level, e in entry.items():
            means = ", ".join(f"{v:.3f}" for v in e["fold_means"])
            print(f"  [{regime}] {level:<22} fold means [{means}] -> "
                  f"{e['mean_of_fold_means']:.4f}  "
                  f"{e['folds_above_0_5']}/{e['n_folds_with_signal']} folds > 0.5  "
                  f"strata used/fold {e['mean_strata_used_per_fold']}")

    out_path = config.ARTIFACT_DIR / f"rule_disjoint{TAG}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
                        encoding="utf-8")
    print(f"\n===== rule-disjoint ({len(folds)} folds x {N_SEEDS} seeds, pooled) =====")
    for regime, entry in summary["aggregate"].items():
        print(f"\n[{regime}]")
        for label, key in (("checkpoint pooled", "checkpoint_pooled"),
                           ("checkpoint within-key", "checkpoint_within_key"),
                           ("pair pooled", "pair_pooled"),
                           ("pair within-rule", "pair_within_rule")):
            e = entry.get(key)
            if not e:
                continue
            point = e.get("auroc", e.get("pooled"))
            print(f"  {label:<24}{point:.4f} [{e['ci_95_low']:.4f}, {e['ci_95_high']:.4f}] "
                  f"n={e.get('n', e.get('n_comparable_pairs'))} "
                  f"contains_0.5={e['ci_contains_0_5']}")
    print(f"\nwrote {out_path.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
