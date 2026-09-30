"""Evaluate R-CES v2 with unchanged critical-MIN and tau policy."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
RCES_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(RCES_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import config  # noqa: E402
import metrics  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces import aggregate  # noqa: E402
from rces import evaluate as base_evaluate  # noqa: E402
from rces2_model import ContextualRCESHead  # noqa: E402
from rces2_scoring import (  # noqa: E402
    checkpoint_probabilities,
    visual_pair_scores,
)
from rces2_settings import (  # noqa: E402
    EVAL_REPORT_PATH,
    HEAD_CKPT,
    PREDICTIONS_PATH,
)
from train import device, make_loader  # noqa: E402


def load_head(active_device: torch.device) -> ContextualRCESHead:
    if not HEAD_CKPT.exists():
        raise FileNotFoundError(f"Train R-CES v2 first: {HEAD_CKPT} missing")
    checkpoint = torch.load(HEAD_CKPT, map_location=active_device)
    if checkpoint.get("vision_encoder") != config.VISION_ENCODER:
        raise ValueError(
            "Checkpoint vision encoder does not match active config: "
            f"{checkpoint.get('vision_encoder')!r} != {config.VISION_ENCODER!r}"
        )
    head = ContextualRCESHead(
        img_dim=checkpoint["img_dim"],
        txt_dim=checkpoint["txt_dim"],
        hidden=checkpoint["hidden"],
        dropout=checkpoint["dropout"],
        context_heads=checkpoint["context_heads"],
        image_heads=checkpoint["image_heads"],
        context_layers=checkpoint["context_layers"],
    )
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(active_device).eval()


def probabilities_for_records(
    head: ContextualRCESHead,
    records: list[ds.PairRecord],
    active_device: torch.device,
) -> dict[tuple[str, str], float]:
    if not records:
        return {}
    loader, _ = make_loader(records, shuffle=False)
    return checkpoint_probabilities(head, loader, active_device)


def build_eval_rows(
    head: ContextualRCESHead,
    records: list[ds.PairRecord],
    active_device: torch.device,
) -> list[base_evaluate.EvalRow]:
    probabilities = probabilities_for_records(head, records, active_device)
    visual_scores = visual_pair_scores(records, probabilities)
    local_predictions = base_evaluate.load_local_compliance()
    rows: list[base_evaluate.EvalRow] = []
    for record in records:
        predicted = local_predictions.get(record.pair_id)
        visual_score = visual_scores[record.pair_id]
        rows.append(
            base_evaluate.EvalRow(
                pair_id=record.pair_id,
                image_id=record.image_id,
                rule_id=record.rule_id,
                decision_scope=record.decision_scope,
                visual_sufficiency_score=visual_score,
                external_gate_passed=not record.external_evidence_required,
                final_answerability_score=aggregate.apply_external_gate(
                    record, visual_score
                ),
                sufficiency_gold=record.sufficiency_gold,
                should_abstain=1 - record.sufficiency_gold,
                decidable_gold=record.decidable_gold,
                compliance_gold=record.compliance_gold,
                predicted_compliance=predicted,
                raw_compliance_correct=(
                    None
                    if predicted is None
                    else int(predicted == record.compliance_gold)
                ),
                compliance_correct=(
                    None
                    if predicted is None
                    else int(
                        record.sufficiency_gold == 1
                        and predicted == record.compliance_gold
                    )
                ),
            )
        )
    return rows


def main() -> None:
    active_device = device()
    split = splits.load_split()
    head = load_head(active_device)
    val_records = ds.build_pair_records(set(split["val"]))
    test_records = ds.build_pair_records(set(split["test"]))
    if not test_records:
        raise RuntimeError("No locked-gold test pairs available")

    val_rows = build_eval_rows(head, val_records, active_device)
    test_rows = build_eval_rows(head, test_records, active_device)
    test_probabilities = probabilities_for_records(
        head, test_records, active_device
    )

    tau = (
        metrics.tune_tau_for_risk(
            [row.final_answerability_score for row in val_rows],
            [row.sufficiency_gold for row in val_rows],
            config.TARGET_RISK,
        )
        if val_rows
        else 0.5
    )
    visual_val_rows = [
        row for row in val_rows if row.external_gate_passed
    ]
    visual_test_rows = [
        row for row in test_rows if row.external_gate_passed
    ]
    visual_tau = (
        metrics.tune_tau_for_risk(
            [row.visual_sufficiency_score for row in visual_val_rows],
            [row.sufficiency_gold for row in visual_val_rows],
            config.TARGET_RISK,
        )
        if visual_val_rows
        else 0.5
    )
    task2_val_rows = [
        row for row in val_rows if row.compliance_correct is not None
    ]
    task2_tau = (
        metrics.tune_tau_for_risk(
            [row.final_answerability_score for row in task2_val_rows],
            [int(row.compliance_correct) for row in task2_val_rows],
            config.TARGET_RISK,
        )
        if task2_val_rows
        else tau
    )

    checkpoint = torch.load(HEAD_CKPT, map_location="cpu")
    report = {
        "architecture": checkpoint["architecture"],
        "vision_encoder": checkpoint["vision_encoder"],
        "best_epoch": checkpoint["best_epoch"],
        "aggregation": "critical_min",
        "threshold_policy": "unchanged_validation_risk_coverage",
        "task1_tau": tau,
        "visual_task1_tau": visual_tau,
        "task2_tau": task2_tau,
        "n_test": len(test_rows),
        "checkpoint_resolvability": base_evaluate.checkpoint_metrics(
            test_probabilities, test_records
        ),
        "visual_pair_sufficiency_detection": base_evaluate.task1_metrics(
            visual_test_rows,
            visual_tau,
            score_field="visual_sufficiency_score",
        ),
        "task1_sufficiency_detection": base_evaluate.task1_metrics(
            test_rows, tau
        ),
        "task2_selective_compliance": base_evaluate.task2_metrics(
            test_rows, task2_tau
        ),
    }

    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds.exp_io.rio.write_jsonl(
        PREDICTIONS_PATH, [asdict(row) for row in test_rows]
    )
    EVAL_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        "wrote v2 eval table -> "
        f"{PREDICTIONS_PATH.relative_to(config.RCES_ROOT)}"
    )


if __name__ == "__main__":
    main()
