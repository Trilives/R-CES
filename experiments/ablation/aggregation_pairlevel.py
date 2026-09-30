"""Aggregator comparison on pair-level metrics and on the subset where it can matter.

Reviewer S6tp observed, correctly, that the paper compared MIN against mean/product
using *checkpoint* AUROC -- a metric computed upstream of aggregation, which therefore
cannot test the aggregation rule at all. This driver replaces that comparison with:

  * pair-level metrics (AUROC, AURC, coverage@risk, CMCR), which aggregation does affect;
  * a stratification by how many critical checkpoints are actually missing.

MIN and mean assign different scores to every pair with >=2 critical checkpoints, since
the predicted probabilities differ; gold resolvability does not decide whether the
aggregation can change a ranking. The `exactly_one_missing` slice is where the soft-AND
prior is *decisive against gold* -- one critical checkpoint unresolved, the rest fine --
so reporting its support matters: an aggregator ablation targeted at a slice this small
is underpowered by construction.

Checkpoint probabilities are computed once per seed and re-aggregated three ways, so the
arms differ only in the aggregation rule.

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_N_SEEDS=10 CUBLAS_WORKSPACE_CONFIG=:4096:8 UV_CACHE_DIR=/tmp/uv-cache \
  uv run python experiments/ablation/aggregation_pairlevel.py
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
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces import aggregate  # noqa: E402
from rces import evaluate as rces_eval  # noqa: E402
from rces import train as rces_train  # noqa: E402

N_SEEDS = int(os.environ.get("RCES_N_SEEDS", "10"))
N_BOOTSTRAP = int(os.environ.get("RCES_N_BOOTSTRAP", "5000"))
TAG = config._CROPAUG_TAG
MODES = ("min", "mean", "product")


def slice_of(record) -> str:
    """Which aggregator-discriminative bucket a pair falls in."""
    if len(record.critical_keys) < 2:
        return "trivial_lt2_critical"
    missing = sum(1 for value in record.resolvable_targets.values() if value == 0)
    if missing == 0:
        return "all_resolvable"
    if missing == 1:
        return "exactly_one_missing"
    return "two_or_more_missing"


def main() -> None:
    device = rces_train._device()
    config.INPUT_MODE = "full"
    per_seed: dict[str, dict[str, list]] = {m: {} for m in MODES}
    support: dict[str, int] = {}

    for seed in range(N_SEEDS):
        config.SEED = seed
        config.HEAD_CKPT = config.ARTIFACT_DIR / f"rces_head_aggregation{TAG}.pt"
        config.AGGREGATION_MODE = "min"
        rces_train.main()

        split = splits.load_split()
        val_records = ds.build_pair_records(set(split["val"]))
        test_records = ds.build_pair_records(set(split["test"]))
        head = rces_eval.load_head(device)
        val_probs = rces_eval.checkpoint_probs(head, val_records, device)
        test_probs = rces_eval.checkpoint_probs(head, test_records, device)

        for mode in MODES:
            def pair_scores(records, probs):
                out = {}
                for record in records:
                    resolvable = {k: probs.get((record.pair_id, k), 1.0)
                                  for k in record.critical_keys}
                    visual = aggregate.aggregate_pair_score(record, resolvable, mode=mode)
                    out[record.pair_id] = aggregate.apply_external_gate(record, visual)
                return out

            val_scores = pair_scores(val_records, val_probs)
            test_scores = pair_scores(test_records, test_probs)
            tau = metrics.tune_tau_for_risk(
                [val_scores[r.pair_id] for r in val_records],
                [r.sufficiency_gold for r in val_records], config.TARGET_RISK,
            ) if val_records else 0.5

            buckets: dict[str, list] = {}
            for record in test_records:
                buckets.setdefault("all", []).append(record)
                buckets.setdefault(slice_of(record), []).append(record)
            for name, records in buckets.items():
                support[name] = len(records)
                scores = [test_scores[r.pair_id] for r in records]
                targets = [r.sufficiency_gold for r in records]
                store = per_seed[mode].setdefault(name, {})
                if len(set(targets)) > 1:
                    store.setdefault("auroc", []).append(
                        float(metrics._auroc(np.asarray(scores), np.asarray(targets))))
                    store.setdefault("aurc", []).append(
                        float(metrics.aurc(scores, targets)))
                    store.setdefault("coverage_at_risk", []).append(
                        float(metrics.coverage_at_risk(scores, targets, config.TARGET_RISK)))
                store.setdefault("mean_score", []).append(float(np.mean(scores)))
            # Discriminative comparison: can the aggregator separate a pair with one
            # missing critical checkpoint from a fully resolvable one?
            discriminative = [r for r in test_records
                              if slice_of(r) in {"exactly_one_missing", "all_resolvable"}]
            if discriminative:
                scores = [test_scores[r.pair_id] for r in discriminative]
                targets = [r.sufficiency_gold for r in discriminative]
                support["discriminative"] = len(discriminative)
                if len(set(targets)) > 1:
                    per_seed[mode].setdefault("discriminative", {}).setdefault(
                        "auroc", []).append(
                        float(metrics._auroc(np.asarray(scores), np.asarray(targets))))
        print(f"  seed {seed}: " + "  ".join(
            f"{m}={per_seed[m]['all']['auroc'][-1]:.4f}" for m in MODES))

    def agg(values):
        array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if not len(array):
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "n": int(len(array))}

    summary = {
        "seeds": N_SEEDS,
        "support": support,
        "note": ("MIN and mean differ in score on every pair with >=2 critical "
                 "checkpoints, since the predicted probabilities differ; "
                 "'exactly_one_missing' is where the soft-AND prior is decisive "
                 "against gold, not the only slice where aggregation can change a "
                 "ranking."),
        "per_mode": {m: {s: {k: agg(v) for k, v in d.items()}
                         for s, d in per_seed[m].items()} for m in MODES},
        "paired_delta_min_minus": {},
    }
    for mode in ("mean", "product"):
        for slice_name in per_seed["min"]:
            if "auroc" not in per_seed["min"][slice_name] or \
               "auroc" not in per_seed[mode].get(slice_name, {}):
                continue
            deltas = np.asarray(per_seed["min"][slice_name]["auroc"]) - \
                np.asarray(per_seed[mode][slice_name]["auroc"])
            summary["paired_delta_min_minus"][f"{mode}::{slice_name}"] = {
                "mean_delta": float(deltas.mean()),
                "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "seeds_min_better": int((deltas > 0).sum()),
                "n_seeds": int(len(deltas)),
            }

    out_path = config.ARTIFACT_DIR / f"aggregation_pairlevel{TAG}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
                        encoding="utf-8")

    print(f"\n===== aggregator comparison, PAIR-level ({N_SEEDS} seeds) =====")
    print("support:", json.dumps(support))
    print(f"\n{'slice':<24}{'MIN':>16}{'mean':>16}{'product':>16}")
    for slice_name in ("all", "discriminative", "exactly_one_missing",
                       "two_or_more_missing", "all_resolvable"):
        if slice_name not in per_seed["min"]:
            continue
        cells = []
        for mode in MODES:
            entry = summary["per_mode"][mode].get(slice_name, {}).get("auroc")
            cells.append(f"{entry['mean']:.4f}±{entry['std']:.3f}" if entry else "n/a")
        print(f"{slice_name:<24}" + "".join(f"{c:>16}" for c in cells)
              + f"   (n={support.get(slice_name, 0)})")
    print("\npaired delta (MIN - alternative), pair-level AUROC:")
    for key, d in summary["paired_delta_min_minus"].items():
        print(f"  {key:<44} {d['mean_delta']:+.4f}±{d['std_delta']:.4f} "
              f"MIN better {d['seeds_min_better']}/{d['n_seeds']}")
    print(f"\nwrote {out_path.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
