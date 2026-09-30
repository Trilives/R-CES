"""Compare Qwen3-VL-8B and Qwen3.5-9B structured-v2 predictions."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import config  # noqa: E402
import rces_io as rio  # noqa: E402


LABELS = ("compliant", "non_compliant", "not_applicable")
BOOTSTRAP_SAMPLES = 10_000
GOLD_PATH = config.ARTIFACT_DIR / "local_qwen35_9b/input_snapshot/rces_pilot_pairs.jsonl"
PREDICTIONS_8B = config.ARTIFACT_DIR / "local_qwen3vl_8b/compliance_predictions_structured_v2.jsonl"
PREDICTIONS_9B = config.ARTIFACT_DIR / "local_qwen35_9b/compliance_predictions_structured_v2.jsonl"
OUTPUT_JSON = config.ARTIFACT_DIR / "local_vlm_8b_vs_9b.json"
OUTPUT_MD = config.ARTIFACT_DIR / "local_vlm_8b_vs_9b.md"


def classification_report(
    gold: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    pair_ids: list[str],
) -> dict[str, Any]:
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    correct = 0
    for pair_id in pair_ids:
        actual = gold[pair_id]["compliance_label"]
        predicted = predictions[pair_id]["predicted_compliance"]
        matrix[actual][predicted] += 1
        correct += int(actual == predicted)
    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in LABELS if actual != label)
        fn = sum(matrix[label][predicted] for predicted in LABELS if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }
    return {
        "n": len(pair_ids),
        "accuracy": correct / len(pair_ids) if pair_ids else float("nan"),
        "macro_f1": sum(float(per_class[label]["f1"]) for label in LABELS) / len(LABELS),
        "confusion_matrix": matrix,
        "per_class": per_class,
        "prediction_distribution": dict(
            Counter(predictions[pair_id]["predicted_compliance"] for pair_id in pair_ids)
        ),
    }


def exact_mcnemar_p_value(first_wins: int, second_wins: int) -> float:
    discordant = first_wins + second_wins
    if not discordant:
        return 1.0
    smaller = min(first_wins, second_wins)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def paired_image_bootstrap(
    gold: dict[str, dict[str, Any]],
    first: dict[str, dict[str, Any]],
    second: dict[str, dict[str, Any]],
    pair_ids: list[str],
) -> dict[str, float | int | bool]:
    by_image: dict[str, list[str]] = {}
    for pair_id in pair_ids:
        by_image.setdefault(gold[pair_id]["image_id"], []).append(pair_id)
    image_ids = list(by_image)
    rng = np.random.default_rng(config.SEED)
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_images = rng.choice(image_ids, size=len(image_ids), replace=True)
        sampled_pairs = [pair_id for image_id in sampled_images for pair_id in by_image[image_id]]
        first_accuracy = sum(
            first[pair_id]["predicted_compliance"] == gold[pair_id]["compliance_label"]
            for pair_id in sampled_pairs
        ) / len(sampled_pairs)
        second_accuracy = sum(
            second[pair_id]["predicted_compliance"] == gold[pair_id]["compliance_label"]
            for pair_id in sampled_pairs
        ) / len(sampled_pairs)
        deltas.append(second_accuracy - first_accuracy)
    low, high = np.quantile(np.asarray(deltas), [0.025, 0.975]).tolist()
    return {
        "delta_accuracy": float(np.mean(deltas)),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "ci_contains_zero": bool(low <= 0 <= high),
        "n_images": len(image_ids),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def structured_audit(rows: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        "n": len(rows),
        "structured_ok": sum(bool(row.get("structured_ok")) for row in rows.values()),
        "fallbacks": sum(row.get("reasoning_mode") != "structured" for row in rows.values()),
        "confidence_label_mismatches": sum(
            row.get("confidence_label") != row.get("predicted_compliance")
            for row in rows.values()
        ),
    }


def main() -> None:
    gold_rows = [
        row
        for row in rio.load_jsonl(GOLD_PATH)
        if row.get("annotation_status") in config.LOCKED_STATUSES
    ]
    gold = {row["pair_id"]: row for row in gold_rows}
    predictions_8b = {row["pair_id"]: row for row in rio.load_jsonl(PREDICTIONS_8B)}
    predictions_9b = {row["pair_id"]: row for row in rio.load_jsonl(PREDICTIONS_9B)}
    decidable = sorted(
        pair_id for pair_id, row in gold.items() if row.get("decidability") == "decidable"
    )
    common = [
        pair_id
        for pair_id in decidable
        if pair_id in predictions_8b and pair_id in predictions_9b
    ]
    wins_8b = sum(
        predictions_8b[pair_id]["predicted_compliance"] == gold[pair_id]["compliance_label"]
        and predictions_9b[pair_id]["predicted_compliance"] != gold[pair_id]["compliance_label"]
        for pair_id in common
    )
    wins_9b = sum(
        predictions_9b[pair_id]["predicted_compliance"] == gold[pair_id]["compliance_label"]
        and predictions_8b[pair_id]["predicted_compliance"] != gold[pair_id]["compliance_label"]
        for pair_id in common
    )
    report = {
        "gold_snapshot": str(GOLD_PATH.relative_to(config.RCES_ROOT)),
        "decidable_gold_count": len(decidable),
        "common_decidable_count": len(common),
        "qwen35_9b_all_decidable": classification_report(
            gold, predictions_9b, decidable
        ),
        "common_set": {
            "qwen3vl_8b": classification_report(gold, predictions_8b, common),
            "qwen35_9b": classification_report(gold, predictions_9b, common),
            "paired_accuracy": paired_image_bootstrap(
                gold, predictions_8b, predictions_9b, common
            ),
            "discordant": {
                "8b_correct_9b_wrong": wins_8b,
                "9b_correct_8b_wrong": wins_9b,
                "exact_mcnemar_p_value": exact_mcnemar_p_value(wins_8b, wins_9b),
            },
        },
        "structured_audit": {
            "qwen3vl_8b": structured_audit(predictions_8b),
            "qwen35_9b": structured_audit(predictions_9b),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    common_8b = report["common_set"]["qwen3vl_8b"]
    common_9b = report["common_set"]["qwen35_9b"]
    paired = report["common_set"]["paired_accuracy"]
    lines = [
        "# Local VLM 8B vs 9B",
        "",
        "| Model / set | n | Accuracy | Macro F1 |",
        "|---|---:|---:|---:|",
        f"| Qwen3-VL-8B, common | {common_8b['n']} | {common_8b['accuracy']:.4f} | {common_8b['macro_f1']:.4f} |",
        f"| Qwen3.5-9B, common | {common_9b['n']} | {common_9b['accuracy']:.4f} | {common_9b['macro_f1']:.4f} |",
        f"| Qwen3.5-9B, all current decidable | {report['qwen35_9b_all_decidable']['n']} | {report['qwen35_9b_all_decidable']['accuracy']:.4f} | {report['qwen35_9b_all_decidable']['macro_f1']:.4f} |",
        "",
        f"Paired common-set accuracy delta (9B - 8B): **{common_9b['accuracy'] - common_8b['accuracy']:+.4f}**.",
        f"Image-cluster bootstrap 95% CI: **[{paired['ci_95_low']:+.4f}, {paired['ci_95_high']:+.4f}]**.",
        f"Exact McNemar p-value: **{report['common_set']['discordant']['exact_mcnemar_p_value']:.6f}**.",
        "",
        "See the JSON report for confusion matrices and per-class precision/recall/F1.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
