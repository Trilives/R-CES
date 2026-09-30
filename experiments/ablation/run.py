"""Ablation driver (Readme step 9): prove it is checkpoint coverage doing the
work, not ordinary calibration.

Runs the full R-CES model and the required core ablations, each as its own
train+evaluate with isolated artifacts:
with isolated artifacts, then prints a comparison table:

  full                       the method
  no_checkpoint_supervision  train on pair sufficiency, not per-checkpoint
  with_is_visual_legacy       restore the old rule-side is_visual head feature
  no_text_conditioning       remove checkpoint text conditioning
  aggregation_mean           replace MIN with mean aggregation
  aggregation_product        replace MIN with product aggregation

  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/ablation/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import config  # noqa: E402
from data import splits  # noqa: E402
from rces import evaluate  # noqa: E402
from rces import train  # noqa: E402


def _reset_ablations() -> None:
    config.ABLATE_NO_CHECKPOINT_SUPERVISION = False
    config.ABLATE_NO_IS_VISUAL = False
    config.USE_IS_VISUAL_FEATURE = False
    config.ABLATE_NO_TEXT_CONDITIONING = False
    config.AGGREGATION_MODE = "min"


ABLATIONS: list[tuple[str, dict[str, Any]]] = [
    ("full", {}),
    ("no_checkpoint_supervision", {"ABLATE_NO_CHECKPOINT_SUPERVISION": True}),
    ("with_is_visual_legacy", {"USE_IS_VISUAL_FEATURE": True}),
    ("no_text_conditioning", {"ABLATE_NO_TEXT_CONDITIONING": True}),
    ("aggregation_mean", {"AGGREGATION_MODE": "mean"}),
    ("aggregation_product", {"AGGREGATION_MODE": "product"}),
]


def run_one(name: str, overrides: dict[str, Any]) -> dict[str, Any] | None:
    _reset_ablations()
    for key, value in overrides.items():
        setattr(config, key, value)
    tag = config._CROPAUG_TAG
    full_head = config.ARTIFACT_DIR / f"rces_head_full{tag}.pt"
    aggregation_only = name.startswith("aggregation_")
    config.HEAD_CKPT = full_head if aggregation_only else config.ARTIFACT_DIR / f"rces_head_{name}{tag}.pt"

    if not aggregation_only:
        try:
            train.main()
        except FileNotFoundError as exc:
            print(f"[ablation {name}] skipped: {exc}")
            return None
    elif not full_head.exists():
        print(f"[ablation {name}] skipped: full head missing at {full_head}")
        return None

    device = evaluate._device()
    head = evaluate.load_head(device)
    split = splits.load_split()
    val_rows = evaluate.build_eval_rows(head, set(split["val"]), device) if split["val"] else []
    test_rows = evaluate.build_eval_rows(head, set(split["test"]), device)

    if val_rows:
        tau = evaluate.metrics.tune_tau_for_risk(
            [r.final_answerability_score for r in val_rows],
            [r.sufficiency_gold for r in val_rows],
            config.TARGET_RISK,
        )
    else:
        tau = 0.5
    task2_val = [row for row in val_rows if row.compliance_correct is not None]
    if task2_val:
        task2_tau = evaluate.metrics.tune_tau_for_risk(
            [row.final_answerability_score for row in task2_val],
            [int(row.compliance_correct) for row in task2_val],
            config.TARGET_RISK,
        )
    else:
        task2_tau = tau
    report = {
        "ablation": name,
        "task1_tau": tau,
        "task2_tau": task2_tau,
        "task1_sufficiency_detection": evaluate.task1_metrics(test_rows, tau),
        "task2_selective_compliance": evaluate.task2_metrics(test_rows, task2_tau),
    }
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACT_DIR / f"ablation_{name}{config._CROPAUG_TAG}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    reports = []
    for name, overrides in ABLATIONS:
        print(f"\n===== ablation: {name} =====")
        report = run_one(name, overrides)
        if report is not None:
            reports.append(report)
    _reset_ablations()

    print("\n===== ablation comparison (task 1) =====")
    print(f"{'ablation':<28} {'AURC':>8} {'CMCR':>8} {'cov@risk':>9} {'ECE':>8}")
    for r in reports:
        t1 = r["task1_sufficiency_detection"]
        print(
            f"{r['ablation']:<28} {t1.get('aurc', float('nan')):>8.4f} "
            f"{t1.get('cmcr', float('nan')):>8.4f} {t1.get('coverage_at_target_risk', float('nan')):>9.4f} "
            f"{t1.get('ece', float('nan')):>8.4f}"
        )
    (config.ARTIFACT_DIR / f"ablation_summary{config._CROPAUG_TAG}.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
