"""Multi-seed R-CES v1 ablation matrix with confidence intervals (items 2 & 3).

Same contract as rces-2/multiseed.py but for the default checkpoint-resolvability
head. Training is deterministic per seed; RCES_N_SEEDS (default 5) seeds are run.
Per arm it reports mean +/- std across seeds for checkpoint AUROC, Task-1 AUROC,
AURC, CMCR; the full method's image-clustered test bootstrap AUROC CI; and the
paired (full - arm) checkpoint-AUROC deltas with sign consistency.

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_N_SEEDS=5 RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 UV_CACHE_DIR=/tmp/uv-cache \
  uv run python experiments/ablation/multiseed.py
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
from rces import evaluate as rces_eval  # noqa: E402
from rces import train as rces_train  # noqa: E402

N_SEEDS = int(os.environ.get("RCES_N_SEEDS", "5"))
TAG = config._CROPAUG_TAG

# is_visual ablation kept (v1-specific). Aggregation arms reuse the full head.
ABLATIONS: list[tuple[str, dict]] = [
    ("full", {}),
    ("no_checkpoint_supervision", {"ABLATE_NO_CHECKPOINT_SUPERVISION": True}),
    ("with_is_visual_legacy", {"USE_IS_VISUAL_FEATURE": True}),
    ("no_text_conditioning", {"ABLATE_NO_TEXT_CONDITIONING": True}),
    ("aggregation_mean", {"AGGREGATION_MODE": "mean"}),
    ("aggregation_product", {"AGGREGATION_MODE": "product"}),
]
ARM_NAMES = [n for n, _ in ABLATIONS]
METRIC_KEYS = ("checkpoint_auroc", "task1_auroc", "aurc", "cmcr")


def _reset() -> None:
    config.ABLATE_NO_CHECKPOINT_SUPERVISION = False
    config.ABLATE_NO_IS_VISUAL = False
    config.USE_IS_VISUAL_FEATURE = False
    config.ABLATE_NO_TEXT_CONDITIONING = False
    config.AGGREGATION_MODE = "min"


def evaluate_arm(device, split, test_records) -> tuple[dict, dict]:
    head = rces_eval.load_head(device)
    val_rows = rces_eval.build_eval_rows(head, set(split["val"]), device) if split["val"] else []
    test_rows = rces_eval.build_eval_rows(head, set(split["test"]), device)
    test_probs = rces_eval.checkpoint_probs(head, test_records, device)
    tau = (
        metrics.tune_tau_for_risk(
            [r.final_answerability_score for r in val_rows],
            [r.sufficiency_gold for r in val_rows],
            config.TARGET_RISK,
        )
        if val_rows
        else 0.5
    )
    ck = rces_eval.checkpoint_metrics(test_probs, test_records)
    t1 = rces_eval.task1_metrics(test_rows, tau)
    m = {
        "checkpoint_auroc": float(ck.get("auroc", float("nan"))),
        "task1_auroc": float(t1.get("auroc", float("nan"))),
        "aurc": float(t1.get("aurc", float("nan"))),
        "cmcr": float(t1.get("cmcr", float("nan"))),
    }
    # image-clustered bootstrap of checkpoint AUROC
    scores, targets, groups = [], [], []
    for record in test_records:
        for key, target in record.resolvable_targets.items():
            if (record.pair_id, key) in test_probs:
                scores.append(test_probs[(record.pair_id, key)])
                targets.append(target)
                groups.append(record.image_id)
    ci = metrics.bootstrap_auroc(
        np.asarray(scores), np.asarray(targets), groups=groups, n_bootstrap=5000
    )
    return m, ci


def agg(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half,
            "min": float(arr.min()), "max": float(arr.max()), "n": n}


def main() -> None:
    device = rces_train._device()
    seeds = list(range(N_SEEDS))
    collected = {name: {k: [] for k in METRIC_KEYS} for name in ARM_NAMES}
    full_ci_per_seed: list[dict] = []

    for seed in seeds:
        config.SEED = seed
        print(f"\n########## seed {seed} ##########")
        split = splits.load_split()
        test_records = ds.build_pair_records(set(split["test"]))
        full_head_path = config.ARTIFACT_DIR / f"rces_head_full{TAG}.pt"
        for name, overrides in ABLATIONS:
            _reset()
            for key, value in overrides.items():
                setattr(config, key, value)
            aggregation_only = name.startswith("aggregation_")
            if aggregation_only:
                config.HEAD_CKPT = full_head_path
            else:
                config.HEAD_CKPT = config.ARTIFACT_DIR / f"rces_head_{name}{TAG}.pt"
                rces_train.main()
            m, ci = evaluate_arm(device, split, test_records)
            for k in METRIC_KEYS:
                collected[name][k].append(m[k])
            print(f"  {name:<26} ckAUROC={m['checkpoint_auroc']:.4f} "
                  f"T1AUROC={m['task1_auroc']:.4f} CMCR={m['cmcr']:.4f}")
            if name == "full":
                full_ci_per_seed.append(ci)
        _reset()

    summary: dict[str, object] = {
        "architecture": "rces-1_checkpoint_resolvability_head",
        "seeds": seeds,
        "deterministic": config.DETERMINISTIC,
        "n_test_pairs": len(test_records),
        "per_arm": {n: {k: agg(collected[n][k]) for k in METRIC_KEYS} for n in ARM_NAMES},
        "paired_delta_vs_full": {},
    }
    # Paired full-minus-arm deltas on BOTH checkpoint AUROC (NaN for the
    # pair-level no_checkpoint_supervision head, which emits no per-checkpoint
    # score) and Task-1 AUROC (defined for every arm, so it is the universal
    # cross-arm comparison).
    for delta_metric in ("checkpoint_auroc", "task1_auroc"):
        full_vals = np.asarray(collected["full"][delta_metric])
        for name in ARM_NAMES:
            if name == "full":
                continue
            deltas = full_vals - np.asarray(collected[name][delta_metric])
            finite = deltas[np.isfinite(deltas)]
            summary["paired_delta_vs_full"][f"{name}::{delta_metric}"] = {
                "metric": delta_metric,
                "mean_delta": float(finite.mean()) if len(finite) else float("nan"),
                "std_delta": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                "seeds_full_better": int((finite > 0).sum()),
                "n_seeds": int(len(finite)),
                "per_seed": [round(float(d), 4) for d in deltas],
            }
    full_ck = np.asarray(collected["full"]["checkpoint_auroc"])
    median_idx = int(np.argsort(full_ck)[len(full_ck) // 2])
    summary["full_checkpoint_auroc_test_bootstrap_ci_median_seed"] = full_ci_per_seed[median_idx]

    out_path = config.ARTIFACT_DIR / f"multiseed_ablation_summary{TAG}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n===== v1 multi-seed ablation ({len(seeds)} seeds) =====")
    print(f"{'arm':<26}{'ckAUROC mean±std':>22}{'T1AUROC':>10}{'CMCR':>9}")
    for name in ARM_NAMES:
        a = summary["per_arm"][name]
        print(f"{name:<26}{a['checkpoint_auroc']['mean']:>10.4f}±{a['checkpoint_auroc']['std']:<10.4f}"
              f"{a['task1_auroc']['mean']:>10.4f}{a['cmcr']['mean']:>9.4f}")
    print("\npaired delta (full - arm), checkpoint AUROC then Task-1 AUROC:")
    for key, d in summary["paired_delta_vs_full"].items():
        print(f"  {key:<40} mean={d['mean_delta']:+.4f} std={d['std_delta']:.4f} "
              f"full_better {d['seeds_full_better']}/{d['n_seeds']}")
    ci = summary["full_checkpoint_auroc_test_bootstrap_ci_median_seed"]
    print(f"\nfull checkpoint AUROC (median seed) test bootstrap 95% CI: "
          f"{ci['auroc']:.4f} [{ci['ci_95_low']:.4f}, {ci['ci_95_high']:.4f}]")
    print(f"wrote {out_path.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
