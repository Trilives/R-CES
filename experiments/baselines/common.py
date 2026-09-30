"""Shared offline evaluation for local-Qwen selective baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import config
import metrics
from data import dataset as ds


@dataclass(frozen=True)
class BaselineRow:
    pair_id: str
    decision_scope: str
    n_critical: int
    confidence: float
    sufficiency_gold: int
    should_abstain: int
    compliance_gold: str
    predicted_compliance: str
    raw_compliance_correct: int
    compliance_correct: int


def load_local_predictions() -> dict[str, dict[str, Any]]:
    if not config.LOCAL_QWEN_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Local Qwen predictions are missing. Run "
            "experiments/baselines/local_qwen35.py first."
        )
    return {
        row["pair_id"]: row
        for row in ds.exp_io.rio.load_jsonl(config.LOCAL_QWEN_PREDICTIONS_PATH)
    }


def load_self_eval() -> dict[str, dict[str, Any]]:
    """Per-pair elicited-confidence baselines (P(True), verbalized), if present.

    Returns {} when the active VLM has no self_eval_predictions.jsonl, so cascade
    silently runs without these gates.
    """
    path = getattr(config, "SELF_EVAL_PREDICTIONS_PATH", None)
    if path is None or not path.exists():
        return {}
    return {row["pair_id"]: row for row in ds.exp_io.rio.load_jsonl(path)}


def build_rows(image_ids: set[str]) -> list[BaselineRow]:
    predictions = load_local_predictions()
    rows: list[BaselineRow] = []
    for record in ds.build_pair_records(image_ids):
        prediction = predictions.get(record.pair_id)
        if prediction is None:
            continue
        predicted = str(prediction["predicted_compliance"])
        confidence_label = str(prediction.get("confidence_label", ""))
        if confidence_label != predicted:
            raise ValueError(
                f"Prediction {record.pair_id} has confidence for {confidence_label!r} "
                f"but final label {predicted!r}"
            )
        rows.append(
            BaselineRow(
                pair_id=record.pair_id,
                decision_scope=record.decision_scope,
                n_critical=len(record.critical_keys),
                confidence=float(prediction["confidence"]),
                sufficiency_gold=record.sufficiency_gold,
                should_abstain=1 - record.sufficiency_gold,
                compliance_gold=record.compliance_gold,
                predicted_compliance=predicted,
                raw_compliance_correct=int(predicted == record.compliance_gold),
                compliance_correct=int(
                    record.sufficiency_gold == 1 and predicted == record.compliance_gold
                ),
            )
        )
    return rows


def maxprob_scores(rows: Sequence[BaselineRow]) -> list[float]:
    return [row.confidence for row in rows]


def _task_metrics(
    rows: Sequence[BaselineRow],
    scores: Sequence[float],
    *,
    task1_tau: float,
    task2_tau: float,
) -> dict[str, dict[str, float]]:
    return {
        "task1_sufficiency_detection": metrics.summarize(
            scores,
            [row.sufficiency_gold for row in rows],
            tau=task1_tau,
            target_risk=config.TARGET_RISK,
            should_abstain=[row.should_abstain for row in rows],
            n_bins=config.ECE_BINS,
        ),
        "task2_selective_compliance": metrics.summarize(
            scores,
            [row.compliance_correct for row in rows],
            tau=task2_tau,
            target_risk=config.TARGET_RISK,
            n_bins=config.ECE_BINS,
        ),
    }


def report(
    name: str,
    val_rows: list[BaselineRow],
    val_scores: Sequence[float],
    test_rows: list[BaselineRow],
    test_scores: Sequence[float],
    *,
    fixed_tau: float | None = None,
) -> dict[str, Any]:
    if not test_rows:
        raise RuntimeError("No local-Qwen predictions align with locked test gold")
    if fixed_tau is not None:
        task1_tau = fixed_tau
        task2_tau = fixed_tau
    elif val_rows:
        task1_tau = metrics.tune_tau_for_risk(
            val_scores,
            [row.sufficiency_gold for row in val_rows],
            config.TARGET_RISK,
        )
        task2_tau = metrics.tune_tau_for_risk(
            val_scores,
            [row.compliance_correct for row in val_rows],
            config.TARGET_RISK,
        )
    else:
        task1_tau = 0.5
        task2_tau = 0.5
    out: dict[str, Any] = {
        "baseline": name,
        "task1_tau": task1_tau,
        "task2_tau": task2_tau,
        "n_test": len(test_rows),
        "raw_label_accuracy_all_pairs": sum(
            row.raw_compliance_correct for row in test_rows
        ) / len(test_rows),
    }
    sufficient = [row for row in test_rows if row.sufficiency_gold == 1]
    out["label_accuracy_on_sufficient_pairs"] = (
        sum(row.raw_compliance_correct for row in sufficient) / len(sufficient)
        if sufficient
        else float("nan")
    )
    out.update(
        _task_metrics(
            test_rows,
            test_scores,
            task1_tau=task1_tau,
            task2_tau=task2_tau,
        )
    )
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (config.ARTIFACT_DIR / f"baseline_{name}{config._CROPAUG_TAG}_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
