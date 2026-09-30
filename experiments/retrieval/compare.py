"""Compare SigLIP vs hierarchical-VLM rule retrieval against human gold.

Reads the two retrieval artifacts (siglip_retrieve.py, vlm_retrieve.py) and the
locked human gold. The gold gives, per annotated (image, rule) pair, whether the
rule is APPLICABLE:

    gold_applicable = 0 iff compliance_label == not_applicable  (subject absent),
                      1 otherwise  (subject in play: compliant / non_compliant /
                                    unknown).

This is exactly the signal behind "some SigLIP-retrieved rules don't apply".
Each method supplies a per-(image, rule) score (SigLIP: cosine; VLM: Stage-B
applies-probability, 0 when the rule's category was not selected). We report how
well each score separates applicable from not-applicable rules (AUROC / average
precision) on the annotated pairs, the VLM threshold precision/recall/F1, the
SigLIP target (top-1) not-applicable rate, and method-vs-method agreement.

No model is loaded here; it only reads artifacts. Run after both retrievers:

  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/retrieval/compare.py
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
import exp_io  # noqa: E402


# --- Dependency-light ranking metrics ----------------------------------------
def _average_ranks(values: list[float]) -> list[float]:
    """1-based average ranks (ties share the mean rank), in the input order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney-U AUROC; None when only one class is present."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = _average_ranks(scores)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision(scores: list[float], labels: list[int]) -> float | None:
    """Area under precision-recall (interpolation-free), None without positives."""
    pos = sum(labels)
    if pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    cumulative = 0.0
    for seen, idx in enumerate(order, 1):
        if labels[idx] == 1:
            tp += 1
            cumulative += tp / seen
    return cumulative / pos


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# --- Comparison ---------------------------------------------------------------
def compute_report() -> dict[str, Any]:
    vlm_rows = {r["image_id"]: r for r in exp_io.rio.load_jsonl(config.RETRIEVAL_VLM_PATH)}
    siglip_rows = {r["image_id"]: r for r in exp_io.rio.load_jsonl(config.RETRIEVAL_SIGLIP_PATH)}
    if not vlm_rows:
        raise RuntimeError(f"No VLM retrieval rows at {config.RETRIEVAL_VLM_PATH}")
    if not siglip_rows:
        raise RuntimeError(f"No SigLIP retrieval rows at {config.RETRIEVAL_SIGLIP_PATH}")

    gold = exp_io.load_locked_gold()
    labelled = [g for g in gold if g.get("compliance_label") in exp_io.rio.COMPLIANCE]

    siglip_scores: list[float] = []
    vlm_scores: list[float] = []
    labels: list[int] = []
    target_total = 0
    target_not_applicable = 0
    siglip_topk_tp = siglip_topk_fp = siglip_topk_fn = 0
    vlm_tp = vlm_fp = vlm_fn = 0

    for pair in labelled:
        image_id, rule_id = pair["image_id"], pair["rule_id"]
        s_row, v_row = siglip_rows.get(image_id), vlm_rows.get(image_id)
        if s_row is None or v_row is None:
            continue
        applicable = 0 if pair["compliance_label"] == "not_applicable" else 1
        s = float(s_row["rule_scores"].get(rule_id, 0.0))
        v = float(v_row["rule_relevance"].get(rule_id, 0.0))
        siglip_scores.append(s)
        vlm_scores.append(v)
        labels.append(applicable)

        if rule_id in set(s_row.get("applicable_rules", [])):
            siglip_topk_tp += applicable
            siglip_topk_fp += 1 - applicable
        else:
            siglip_topk_fn += applicable
        if rule_id in set(v_row.get("applicable_rules", [])):
            vlm_tp += applicable
            vlm_fp += 1 - applicable
        else:
            vlm_fn += applicable

        if pair.get("rule_role") == "target":
            target_total += 1
            target_not_applicable += 1 - applicable

    # Method-vs-method agreement over images present in both artifacts.
    shared = sorted(set(vlm_rows) & set(siglip_rows))
    jaccards: list[float] = []
    siglip_top1_in_vlm = 0
    for image_id in shared:
        vlm_set = set(vlm_rows[image_id].get("applicable_rules", []))
        sig_set = set(siglip_rows[image_id].get("applicable_rules", []))
        union = vlm_set | sig_set
        jaccards.append(len(vlm_set & sig_set) / len(union) if union else 1.0)
        if siglip_rows[image_id].get("top1") in vlm_set:
            siglip_top1_in_vlm += 1

    mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else None
    return {
        "active_method": config.RETRIEVAL_METHOD,
        "rule_pool": "benchmark_rule_subset (23)",
        "annotated_pairs_scored": len(labels),
        "applicable_count": sum(labels),
        "not_applicable_count": len(labels) - sum(labels),
        "applicability_separation": {
            "siglip2_similarity": {
                "auroc": auroc(siglip_scores, labels),
                "average_precision": average_precision(siglip_scores, labels),
            },
            "vlm_hierarchical": {
                "auroc": auroc(vlm_scores, labels),
                "average_precision": average_precision(vlm_scores, labels),
            },
        },
        "applicable_set_quality_on_gold": {
            f"siglip2_top{config.RETRIEVAL_SIGLIP_TOP_K}": prf(
                siglip_topk_tp, siglip_topk_fp, siglip_topk_fn
            ),
            f"vlm_top{config.RETRIEVAL_TOP_K_PER_CATEGORY}_per_category": prf(
                vlm_tp, vlm_fp, vlm_fn
            ),
        },
        "siglip_target_not_applicable": {
            "target_pairs": target_total,
            "not_applicable": target_not_applicable,
            "rate": (target_not_applicable / target_total) if target_total else None,
            "note": "Fraction of SigLIP top-1 'target' rules the human marked not_applicable.",
        },
        "method_agreement": {
            "images_compared": len(shared),
            "mean_jaccard_applicable_sets": mean_jaccard,
            "siglip_top1_in_vlm_applicable_rate": (
                siglip_top1_in_vlm / len(shared) if shared else None
            ),
        },
        "thresholds": {
            "category_threshold": config.RETRIEVAL_CATEGORY_THRESHOLD,
            "top_k_per_category": config.RETRIEVAL_TOP_K_PER_CATEGORY,
            "siglip_top_k": config.RETRIEVAL_SIGLIP_TOP_K,
        },
    }


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    sep = report["applicability_separation"]
    setq = report["applicable_set_quality_on_gold"]
    lines = [
        "# Rule retrieval: SigLIP vs hierarchical VLM",
        "",
        f"- Active method (config.RETRIEVAL_METHOD): **{report['active_method']}**",
        f"- Rule pool: {report['rule_pool']}",
        f"- Annotated pairs scored: {report['annotated_pairs_scored']} "
        f"({report['applicable_count']} applicable / {report['not_applicable_count']} not_applicable)",
        "",
        "## Applicability separation (gold = applicable vs not_applicable)",
        "",
        "| method | AUROC | average precision |",
        "|---|---|---|",
        f"| SigLIP 2 similarity | {_fmt(sep['siglip2_similarity']['auroc'])} | "
        f"{_fmt(sep['siglip2_similarity']['average_precision'])} |",
        f"| VLM hierarchical | {_fmt(sep['vlm_hierarchical']['auroc'])} | "
        f"{_fmt(sep['vlm_hierarchical']['average_precision'])} |",
        "",
        "## Applicable-set quality on gold",
        "",
        "| method | precision | recall | F1 |",
        "|---|---|---|---|",
    ]
    for name, m in setq.items():
        lines.append(f"| {name} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} |")
    tgt = report["siglip_target_not_applicable"]
    agr = report["method_agreement"]
    lines += [
        "",
        "## SigLIP target mis-retrieval",
        f"- SigLIP top-1 'target' rules marked not_applicable: "
        f"{tgt['not_applicable']}/{tgt['target_pairs']} (rate {_fmt(tgt['rate'])})",
        "",
        "## Method agreement",
        f"- Images compared: {agr['images_compared']}",
        f"- Mean Jaccard of applicable sets: {_fmt(agr['mean_jaccard_applicable_sets'])}",
        f"- SigLIP top-1 inside VLM applicable set: {_fmt(agr['siglip_top1_in_vlm_applicable_rate'])}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    report = compute_report()
    config.RETRIEVAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.RETRIEVAL_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config.RETRIEVAL_REPORT_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    try:
        print(f"Wrote: {config.RETRIEVAL_REPORT_PATH.relative_to(config.RCES_ROOT)}")
    except ValueError:
        print(f"Wrote: {config.RETRIEVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
