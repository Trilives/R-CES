"""Compare isolated Qwen annotations with locked human gold.

This is a deterministic, offline analysis step. It does not call a model, does
not modify gold, and does not feed its output into R-CES experiments.

  uv run python Script/compare_qwen_to_gold.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rces_io as rio  # noqa: E402


JUDGMENT_FIELDS = (
    "decidability",
    "compliance_label",
    "image_quality",
    "evidence_sufficiency_label",
)
CHECKPOINT_FIELDS = (
    "observed_checkpoints",
    "missing_checkpoints",
    "critical_missing_checkpoints",
)
LOCKED_STATUSES = {"judged", "described", "reviewed"}
QWEN_PATH = rio.PAIRS_DIR / "rces_qwen_full_baseline.jsonl"
HIGH_CONFIDENCE_THRESHOLD = 0.8


@dataclass(frozen=True)
class RunConfig:
    gold_path: Path = rio.GOLD_PATH
    qwen_path: Path = QWEN_PATH
    output_root: Path = rio.CACHE_DIR / "qwen_comparison"
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD


def _normalized_set(row: dict[str, Any], field: str) -> list[str]:
    value = row.get(field, [])
    if not isinstance(value, list):
        return []
    return sorted({str(item) for item in value})


def compare_pair(
    human: dict[str, Any],
    qwen: dict[str, Any],
    *,
    high_confidence_threshold: float,
) -> dict[str, Any]:
    disagreed_fields = [
        field for field in JUDGMENT_FIELDS if human.get(field) != qwen.get(field)
    ]
    checkpoint_disagreements = {
        field: {
            "human": _normalized_set(human, field),
            "qwen": _normalized_set(qwen, field),
        }
        for field in CHECKPOINT_FIELDS
        if _normalized_set(human, field) != _normalized_set(qwen, field)
    }
    confidence = qwen.get("confidence")
    numeric_confidence = float(confidence) if isinstance(confidence, (int, float)) else None
    any_disagreement = bool(disagreed_fields or checkpoint_disagreements)

    return {
        "pair_id": human["pair_id"],
        "image_id": human.get("image_id"),
        "rule_id": human.get("rule_id"),
        "human_decidability": human.get("decidability"),
        "qwen_decidability": qwen.get("decidability"),
        "human_compliance_label": human.get("compliance_label"),
        "qwen_compliance_label": qwen.get("compliance_label"),
        "qwen_confidence": numeric_confidence,
        "disagreed_fields": disagreed_fields,
        "checkpoint_disagreements": checkpoint_disagreements,
        "any_disagreement": any_disagreement,
        "high_confidence_disagreement": bool(
            any_disagreement
            and numeric_confidence is not None
            and numeric_confidence >= high_confidence_threshold
        ),
    }


def build_report(
    human_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    *,
    high_confidence_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    human_by_id = {row["pair_id"]: row for row in human_rows}
    qwen_by_id = {row["pair_id"]: row for row in qwen_rows}
    aligned_ids = sorted(human_by_id.keys() & qwen_by_id.keys())
    comparisons = [
        compare_pair(
            human_by_id[pair_id],
            qwen_by_id[pair_id],
            high_confidence_threshold=high_confidence_threshold,
        )
        for pair_id in aligned_ids
    ]

    field_matches = {
        field: sum(
            human_by_id[pair_id].get(field) == qwen_by_id[pair_id].get(field)
            for pair_id in aligned_ids
        )
        for field in JUDGMENT_FIELDS
    }
    checkpoint_matches = {
        field: sum(
            _normalized_set(human_by_id[pair_id], field)
            == _normalized_set(qwen_by_id[pair_id], field)
            for pair_id in aligned_ids
        )
        for field in CHECKPOINT_FIELDS
    }
    denominator = len(aligned_ids)
    disagreements = [row for row in comparisons if row["any_disagreement"]]

    report = {
        "human_locked_count": len(human_rows),
        "qwen_count": len(qwen_rows),
        "aligned_count": len(aligned_ids),
        "missing_qwen_pair_ids": sorted(human_by_id.keys() - qwen_by_id.keys()),
        "extra_qwen_pair_ids": sorted(qwen_by_id.keys() - human_by_id.keys()),
        "field_agreement": {
            field: {
                "matches": matches,
                "rate": matches / denominator if denominator else None,
            }
            for field, matches in field_matches.items()
        },
        "checkpoint_exact_agreement": {
            field: {
                "matches": matches,
                "rate": matches / denominator if denominator else None,
            }
            for field, matches in checkpoint_matches.items()
        },
        "disagreement_count": len(disagreements),
        "high_confidence_threshold": high_confidence_threshold,
        "high_confidence_disagreement_count": sum(
            row["high_confidence_disagreement"] for row in comparisons
        ),
        "human_decidability_counts": dict(
            Counter(human_by_id[pair_id].get("decidability", "") for pair_id in aligned_ids)
        ),
        "qwen_decidability_counts": dict(
            Counter(qwen_by_id[pair_id].get("decidability", "") for pair_id in aligned_ids)
        ),
        "human_abstain_qwen_decidable_count": sum(
            human_by_id[pair_id].get("decidability") != "decidable"
            and qwen_by_id[pair_id].get("decidability") == "decidable"
            for pair_id in aligned_ids
        ),
    }
    return report, disagreements


def main() -> None:
    config = RunConfig()
    if not config.qwen_path.exists():
        raise FileNotFoundError(
            f"Run baseline_qwen_full.py first: {config.qwen_path} is missing"
        )

    human_rows = [
        row
        for row in rio.load_jsonl(config.gold_path)
        if row.get("annotation_status") in LOCKED_STATUSES
    ]
    qwen_rows = rio.load_jsonl(config.qwen_path)
    report, disagreements = build_report(
        human_rows,
        qwen_rows,
        high_confidence_threshold=config.high_confidence_threshold,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "qwen_vs_human_report.json"
    disagreement_path = output_dir / "qwen_vs_human_disagreements.jsonl"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rio.write_jsonl(disagreement_path, disagreements, sort_keys=True)

    summary = {
        "human_locked_count": report["human_locked_count"],
        "qwen_count": report["qwen_count"],
        "aligned_count": report["aligned_count"],
        "disagreement_count": report["disagreement_count"],
        "high_confidence_disagreement_count": report[
            "high_confidence_disagreement_count"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {report_path.relative_to(rio.PROJECT_ROOT)}")
    print(f"Disagreements: {disagreement_path.relative_to(rio.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
