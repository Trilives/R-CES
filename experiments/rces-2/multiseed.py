"""Multi-seed R-CES v2 ablation matrix with confidence intervals (items 2 & 3).

Training is deterministic per seed (experiments/determinism.set_determinism +
math SDPA), so the spread here is genuine seed-init variance, not kernel noise.
For each seed in RCES_N_SEEDS (default 5) every ablation arm is retrained and
evaluated; the script reports, per arm:

  * mean +/- std across seeds for checkpoint AUROC, Task-1 AUROC, AURC, CMCR;
  * the full method's image-clustered test bootstrap AUROC 95% CI (median seed);
  * paired (full - arm) deltas across seeds: mean, std, and how many seeds keep
    the same sign -- the statistic that says whether an ablation actually hurts.

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_N_SEEDS=5 RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 UV_CACHE_DIR=/tmp/uv-cache \
  uv run python R-CES/experiments/rces-2/multiseed.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
RCES_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(RCES_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import config  # noqa: E402
import exp_io  # noqa: E402
import metrics  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces import evaluate as base_evaluate  # noqa: E402

from ablation import (  # noqa: E402
    ABLATIONS,
    _reset_ablations,
    evaluate_head,
    train_head,
)
from train import device  # noqa: E402

N_SEEDS = int(os.environ.get("RCES_N_SEEDS", "5"))
ARM_NAMES = [name for name, _ in ABLATIONS]
METRIC_KEYS = ("checkpoint_auroc", "task1_auroc", "aurc", "cmcr")


def checkpoint_bootstrap_ci(
    head, test_records, active_device
) -> dict[str, float]:
    """Image-clustered bootstrap 95% CI of the test checkpoint AUROC."""
    import evaluate as v2_evaluate

    probs = v2_evaluate.probabilities_for_records(head, test_records, active_device)
    scores: list[float] = []
    targets: list[int] = []
    groups: list[str] = []
    for record in test_records:
        for key, target in record.resolvable_targets.items():
            if (record.pair_id, key) in probs:
                scores.append(probs[(record.pair_id, key)])
                targets.append(target)
                groups.append(record.image_id)
    return metrics.bootstrap_auroc(
        np.asarray(scores), np.asarray(targets), groups=exp_io.original_image_families(groups), n_bootstrap=5000
    )


def arm_metrics(report: dict) -> dict[str, float]:
    t1 = report["task1_sufficiency_detection"]
    ck = report["checkpoint_resolvability"]
    return {
        "checkpoint_auroc": float(ck.get("auroc", float("nan"))),
        "task1_auroc": float(t1.get("auroc", float("nan"))),
        "aurc": float(t1.get("aurc", float("nan"))),
        "cmcr": float(t1.get("cmcr", float("nan"))),
    }


def agg(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": n,
    }


def main() -> None:
    active_device = device()
    seeds = list(range(N_SEEDS))
    # per arm -> per metric -> list over seeds
    collected: dict[str, dict[str, list[float]]] = {
        name: {k: [] for k in METRIC_KEYS} for name in ARM_NAMES
    }
    full_ci_per_seed: list[dict[str, float]] = []

    for seed in seeds:
        config.SEED = seed
        print(f"\n########## seed {seed} ##########")
        split = splits.load_split()
        train_records = ds.build_pair_records(set(split["train"]))
        val_records = ds.build_pair_records(set(split["val"]))
        test_records = ds.build_pair_records(set(split["test"]))
        full_head = None
        for name, overrides in ABLATIONS:
            _reset_ablations()
            for key, value in overrides.items():
                setattr(config, key, value)
            aggregation_only = name.startswith("aggregation_")
            if aggregation_only:
                head = full_head
            else:
                head, _ = train_head(active_device, train_records, val_records)
                if name == "full":
                    full_head = head
            report = evaluate_head(head, val_records, test_records, active_device)
            m = arm_metrics(report)
            for k in METRIC_KEYS:
                collected[name][k].append(m[k])
            print(f"  {name:<26} ckAUROC={m['checkpoint_auroc']:.4f} "
                  f"T1AUROC={m['task1_auroc']:.4f} CMCR={m['cmcr']:.4f}")
            if name == "full":
                full_ci_per_seed.append(
                    checkpoint_bootstrap_ci(head, test_records, active_device)
                )
        _reset_ablations()

    # Aggregate
    summary: dict[str, object] = {
        "architecture": "rces-2_text_context_then_image_cross_attention",
        "seeds": seeds,
        "deterministic": config.DETERMINISTIC,
        "n_test_pairs": len(test_records),
        "per_arm": {},
        "paired_delta_vs_full": {},
    }
    for name in ARM_NAMES:
        summary["per_arm"][name] = {k: agg(collected[name][k]) for k in METRIC_KEYS}
    full_ck = np.asarray(collected["full"]["checkpoint_auroc"])
    for name in ARM_NAMES:
        if name == "full":
            continue
        deltas = full_ck - np.asarray(collected[name]["checkpoint_auroc"])
        summary["paired_delta_vs_full"][name] = {
            "metric": "checkpoint_auroc",
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "seeds_full_better": int((deltas > 0).sum()),
            "n_seeds": len(deltas),
            "per_seed": [round(float(d), 4) for d in deltas],
        }
    # Median-seed bootstrap CI for the full method (image-clustered test sampling).
    median_idx = int(np.argsort(full_ck)[len(full_ck) // 2])
    summary["full_checkpoint_auroc_test_bootstrap_ci_median_seed"] = full_ci_per_seed[
        median_idx
    ]

    out_dir = Path(config.ARTIFACT_DIR) / "rces-2" / f"{config.VISION_ENCODER}_cropaug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multiseed_ablation_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n===== v2 multi-seed ablation ({len(seeds)} seeds) =====")
    print(f"{'arm':<26}{'ckAUROC mean±std':>22}{'T1AUROC mean':>14}{'CMCR mean':>11}")
    for name in ARM_NAMES:
        a = summary["per_arm"][name]
        print(f"{name:<26}{a['checkpoint_auroc']['mean']:>10.4f}±{a['checkpoint_auroc']['std']:<10.4f}"
              f"{a['task1_auroc']['mean']:>14.4f}{a['cmcr']['mean']:>11.4f}")
    print("\npaired delta (full - arm) on checkpoint AUROC:")
    for name, d in summary["paired_delta_vs_full"].items():
        print(f"  {name:<26} mean={d['mean_delta']:+.4f} std={d['std_delta']:.4f} "
              f"full_better {d['seeds_full_better']}/{d['n_seeds']}")
    ci = summary["full_checkpoint_auroc_test_bootstrap_ci_median_seed"]
    print(f"\nfull checkpoint AUROC (median seed) test bootstrap 95% CI: "
          f"{ci['auroc']:.4f} [{ci['ci_95_low']:.4f}, {ci['ci_95_high']:.4f}]")
    print(f"wrote {out_path.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
