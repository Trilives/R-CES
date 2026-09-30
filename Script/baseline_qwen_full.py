"""Qwen full-pipeline annotation baseline for human/model comparison.

Qwen judges every selected image-rule pair under the same neutral prompt used
for human gold (Stage 1 / the codebook decision procedure). Its output uses the
shared flat annotation schema but is written to an isolated file
(rces_qwen_full_baseline.jsonl, source=model_full -- the generic model-baseline
token, with the actual model recorded in `model`), so it can be compared with
human gold without ever becoming ground truth. This is the project's only
full-pipeline model baseline; the former GPT baseline has been retired.

This script owns all DashScope configuration and is the only Qwen cloud caller.
Experiment code consumes the generated JSONL as an offline input.

Resumable: each image's pairs are persisted (atomic full-file rewrite) the
moment that image is annotated. A re-run with RESUME=True skips any image whose
pairs are already present in the baseline file and continues where it stopped,
so an interrupted or failed run never re-spends on already-annotated images.

Run controls live in the constants below; edit them instead of passing flags.

  uv run python Script/baseline_qwen_full.py
"""

from __future__ import annotations

import html
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
import rces_io as rio  # noqa: E402

# --- Run controls -------------------------------------------------------------
MODEL = "qwen3.7-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ENV_KEY = "QWEN_API_KEY"
LIMIT_IMAGES = 0
LIMIT_PAIRS = 0
DRY_RUN = False
RESUME = True  # Skip images already present in the baseline file; resume the run.
DROP_INVALID_EXISTING_ROWS = True
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.0
ENABLE_THINKING = False
STREAM = True
THINKING_BUDGET = 8192
API_RETRIES = 3
API_RETRY_BASE_SECONDS = 2.0
PROMPT_PATH = rio.PROMPT_DIR / "preannotation_prompt.md"
BASELINE_PATH = rio.BASELINE_PATH  # data/pairs/rces_qwen_full_baseline.jsonl


@dataclass(frozen=True)
class RunConfig:
    gold_path: Path = rio.GOLD_PATH
    baseline_path: Path = BASELINE_PATH
    image_dir: Path = rio.IMAGE_DIR
    cache_dir: Path = rio.CACHE_DIR / "qwen_full"
    prompt_path: Path = PROMPT_PATH
    model: str = MODEL
    base_url: str = BASE_URL
    env_key: str = ENV_KEY
    limit_images: int = LIMIT_IMAGES
    limit_pairs: int = LIMIT_PAIRS
    dry_run: bool = DRY_RUN
    resume: bool = RESUME
    drop_invalid_existing_rows: bool = DROP_INVALID_EXISTING_ROWS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float = TEMPERATURE
    enable_thinking: bool = ENABLE_THINKING
    stream: bool = STREAM
    thinking_budget: int = THINKING_BUDGET
    api_retries: int = API_RETRIES
    api_retry_base_seconds: float = API_RETRY_BASE_SECONDS


def validate_config(config: RunConfig) -> None:
    if config.limit_images < 0 or config.limit_pairs < 0:
        raise ValueError("LIMIT_IMAGES and LIMIT_PAIRS must be non-negative")
    if config.max_output_tokens <= 0:
        raise ValueError("MAX_OUTPUT_TOKENS must be positive")
    if config.thinking_budget <= 0:
        raise ValueError("THINKING_BUDGET must be positive")


# --- Pair selection (deterministic, so RESUME picks the same images) ----------
def group_pairs_by_image(
    pairs: list[dict[str, Any]], limit_images: int
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["image_id"]].append(pair)
    all_ids = sorted(grouped)
    if 0 < limit_images < len(all_ids):
        n = len(all_ids)
        idx = sorted({int((i + 0.5) * n / limit_images) for i in range(limit_images)})
        return {all_ids[i]: grouped[all_ids[i]] for i in idx}
    return {image_id: grouped[image_id] for image_id in all_ids}


def apply_pair_limit(
    groups: dict[str, list[dict[str, Any]]], limit_pairs: int
) -> dict[str, list[dict[str, Any]]]:
    if limit_pairs <= 0:
        return groups
    remaining = limit_pairs
    limited: dict[str, list[dict[str, Any]]] = {}
    for image_id, pairs in groups.items():
        if remaining <= 0:
            break
        selected = pairs[:remaining]
        if selected:
            limited[image_id] = selected
            remaining -= len(selected)
    return limited


# --- Prompt payload / row building (shared flat schema) -----------------------
def build_prompt_payload(
    pairs: list[dict[str, Any]], rules: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    payload = []
    for pair in pairs:
        rule = rules.get(pair["rule_id"], {})
        required, critical, _ = rio.checkpoints_for(rule)
        payload.append(
            {
                "pair_id": pair["pair_id"],
                "rule_id": pair["rule_id"],
                "major_category": pair["major_category"],
                "subcategory": pair["subcategory"],
                "rule_name": pair["rule_name"],
                "rule_role": pair["rule_role"],
                "decision_scope": pair["decision_scope"],
                "required_checkpoints": required,
                "critical_checkpoints": critical,
            }
        )
    return payload


def dry_run_annotations(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": pair["pair_id"],
            "decidability": "undecidable",
            "compliance_label": "unknown",
            "observed_checkpoints": [],
            "missing_checkpoints": [],
            "image_quality": "usable",
            "evidence_sufficiency_label": "insufficient",
            "visual_evidence": [],
            "missing_information": ["dry_run_no_model_call"],
            "rationale": "Dry run placeholder; no API call was made.",
            "confidence": 0.0,
        }
        for pair in pairs
    ]


def coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def normalize_decidability(annotation: dict[str, Any]) -> str:
    raw_decidability = str(annotation.get("decidability", ""))
    if raw_decidability in rio.DECIDABILITY:
        return raw_decidability
    # Common model slip: it returns a compliance value in the decidability slot.
    if raw_decidability in ("compliant", "non_compliant", "not_applicable"):
        return "decidable"

    raw_sufficiency = str(annotation.get("evidence_sufficiency_label", ""))
    by_sufficiency = {
        "sufficient": "decidable",
        "partial": "partially_decidable",
        "insufficient": "undecidable",
    }
    if raw_sufficiency in by_sufficiency:
        return by_sufficiency[raw_sufficiency]

    raw_compliance = str(annotation.get("compliance_label", ""))
    if raw_compliance in ("compliant", "non_compliant", "not_applicable"):
        return "decidable"
    if raw_compliance == "unknown":
        return "undecidable"
    raise ValueError(f"Cannot normalize model decidability: {raw_decidability!r}")


def normalize_compliance(decidability: str, annotation: dict[str, Any]) -> str:
    raw_compliance = str(annotation.get("compliance_label", ""))
    if raw_compliance not in rio.COMPLIANCE:
        raw_decidability = str(annotation.get("decidability", ""))
        raw_compliance = raw_decidability if raw_decidability in rio.COMPLIANCE else "unknown"

    forced = rio.forced_compliance(decidability)
    if forced is not None:
        return forced
    if raw_compliance == "unknown":
        raise ValueError("Model returned compliance=unknown for decidability=decidable")
    return raw_compliance


def normalize_image_quality(annotation: dict[str, Any]) -> str:
    raw_quality = str(annotation.get("image_quality", ""))
    if raw_quality in rio.IMAGE_QUALITY:
        return raw_quality
    return "usable"


def reconcile_decision_with_checkpoints(
    decidability: str,
    compliance: str,
    missing: list[str],
    rule: dict[str, Any],
) -> tuple[str, str]:
    """Make the model's judgment obey checkpoint-driven codebook invariants."""
    if decidability == "decidable" and rio.critical_missing(missing, rule):
        return "partially_decidable", "unknown"
    return decidability, compliance


def baseline_row_errors(row: dict[str, Any], rule: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    decidability = row.get("decidability")
    compliance = row.get("compliance_label")
    image_quality = row.get("image_quality")
    evidence = row.get("evidence_sufficiency_label")
    if decidability not in rio.DECIDABILITY:
        errors.append(f"decidability {decidability!r} not in {rio.DECIDABILITY}")
    if compliance not in rio.COMPLIANCE:
        errors.append(f"compliance_label {compliance!r} not in {rio.COMPLIANCE}")
    if image_quality not in rio.IMAGE_QUALITY:
        errors.append(f"image_quality {image_quality!r} not in {rio.IMAGE_QUALITY}")
    if decidability in rio.DECIDABILITY:
        expected_evidence = rio.derive_evidence_sufficiency(str(decidability))
        if evidence != expected_evidence:
            errors.append(
                f"evidence_sufficiency_label {evidence!r} != derived {expected_evidence!r}"
            )

    required = set(rule.get("required_checkpoints", []))
    critical = set(rule.get("critical_checkpoints", []))
    observed = set(coerce_list(row.get("observed_checkpoints")))
    missing = set(coerce_list(row.get("missing_checkpoints")))
    critical_missing = set(coerce_list(row.get("critical_missing_checkpoints")))
    if observed - required:
        errors.append(f"observed_checkpoints outside rule: {sorted(observed - required)}")
    if missing - required:
        errors.append(f"missing_checkpoints outside rule: {sorted(missing - required)}")
    if observed & missing:
        errors.append(f"observed/missing overlap: {sorted(observed & missing)}")
    expected_critical_missing = missing & critical
    if critical_missing != expected_critical_missing:
        errors.append(
            "critical_missing_checkpoints "
            f"{sorted(critical_missing)} != missing INTERSECT critical "
            f"{sorted(expected_critical_missing)}"
        )
    if decidability in rio.DECIDABILITY and compliance in rio.COMPLIANCE:
        errors.extend(
            rio.consistency_errors(str(decidability), str(compliance), sorted(critical_missing))
        )
    return errors


def drop_invalid_rows(
    rows: list[dict[str, Any]],
    rules: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    kept: list[dict[str, Any]] = []
    dropped: dict[str, list[str]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        rule = rules.get(str(row.get("rule_id", "")), {})
        errors = baseline_row_errors(row, rule)
        if errors:
            dropped[pair_id] = errors
        else:
            kept.append(row)
    return kept, dropped


def build_baseline_row(
    pair: dict[str, Any],
    annotation: dict[str, Any],
    rule: dict[str, Any],
    model: str,
    dry_run: bool,
) -> dict[str, Any]:
    required, _, _ = rio.checkpoints_for(rule)
    required_set = set(required)
    observed = [k for k in coerce_list(annotation.get("observed_checkpoints")) if k in required_set]
    missing = [k for k in coerce_list(annotation.get("missing_checkpoints")) if k in required_set]
    # Keep observed/missing disjoint; missing wins ties (the model said it could not resolve it).
    observed = [k for k in observed if k not in set(missing)]

    confidence = annotation.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = None
    decidability = normalize_decidability(annotation)
    compliance = normalize_compliance(decidability, annotation)
    decidability, compliance = reconcile_decision_with_checkpoints(
        decidability, compliance, missing, rule
    )
    image_quality = normalize_image_quality(annotation)

    row = {
        # carried pair metadata (identical shape to the gold file)
        "pair_id": pair["pair_id"],
        "image_id": pair["image_id"],
        "image_split": pair["image_split"],
        "rule_id": pair["rule_id"],
        "rule_name": pair["rule_name"],
        "major_category": pair["major_category"],
        "subcategory": pair["subcategory"],
        "decision_scope": pair["decision_scope"],
        "rule_role": pair["rule_role"],
        "image_weak_label": pair["image_weak_label"],
        "weak_candidate_category": pair["weak_candidate_category"],
        # model judgment
        "decidability": decidability,
        "compliance_label": compliance,
        "image_quality": image_quality,
        "evidence_sufficiency_label": rio.derive_evidence_sufficiency(decidability),
        # checkpoints (critical_missing derived, never taken from the model)
        "observed_checkpoints": observed,
        "missing_checkpoints": missing,
        "critical_missing_checkpoints": rio.critical_missing(missing, rule),
        # descriptions
        "visual_evidence": coerce_list(annotation.get("visual_evidence")),
        "missing_information": coerce_list(annotation.get("missing_information")),
        # provenance / isolation
        "source": rio.SOURCE_MODEL_FULL,
        "annotation_status": "model_complete",
        "double_annotation": False,
        "model": model,
        "is_dry_run": dry_run,
        "confidence": confidence,
        "rationale": str(annotation.get("rationale", "")),
        "warning": rio.BASELINE_WARNING,
    }
    errors = baseline_row_errors(row, rule)
    if errors:
        raise ValueError(f"Invalid normalized model row {pair['pair_id']}: {'; '.join(errors)}")
    return row


def merge_rows(
    pairs: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    rules: dict[str, dict[str, Any]],
    model: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    by_id = {str(a.get("pair_id", "")): a for a in annotations}
    rows = []
    for pair in pairs:
        annotation = by_id.get(pair["pair_id"]) or {
            "pair_id": pair["pair_id"],
            "decidability": "undecidable",
            "compliance_label": "unknown",
            "observed_checkpoints": [],
            "missing_checkpoints": [],
            "image_quality": "poor",
            "evidence_sufficiency_label": "insufficient",
            "visual_evidence": [],
            "missing_information": ["model_response_missing_pair_id"],
            "rationale": "The model response did not include this pair_id.",
            "confidence": 0.0,
        }
        rows.append(build_baseline_row(pair, annotation, rules.get(pair["rule_id"], {}), model, dry_run))
    return rows


# --- DashScope client / call --------------------------------------------------
def make_client(config: RunConfig) -> OpenAI:
    load_dotenv(rio.PROJECT_ROOT / ".env")
    api_key = os.environ.get(config.env_key, "")
    if not api_key:
        raise RuntimeError(f"Missing {config.env_key} in .env")
    return OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=300.0,
        max_retries=0,
    )


def stream_answer_content(completion: Any) -> str:
    """Return final-answer content while discarding reasoning deltas."""
    parts: list[str] = []
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            parts.append(content)
    return "".join(parts)


def call_qwen_json(
    client: OpenAI,
    config: RunConfig,
    *,
    prompt_text: str,
    image_path: Path,
    log_label: str = "",
) -> dict[str, Any]:
    mime, encoded_image = rio.encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded_image}"},
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    extra_body: dict[str, Any] = {"enable_thinking": config.enable_thinking}
    if config.enable_thinking:
        extra_body["thinking_budget"] = config.thinking_budget

    attempts = max(1, config.api_retries + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            completion = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_output_tokens,
                stream=config.stream,
                extra_body=extra_body,
            )
            if config.stream:
                text = stream_answer_content(completion)
            else:
                text = completion.choices[0].message.content or ""
            if not text.strip():
                raise ValueError(
                    "Qwen returned empty answer content; lower THINKING_BUDGET "
                    "or raise MAX_OUTPUT_TOKENS"
                )
            return rio.extract_json_object(text)
        except Exception as exc:  # noqa: BLE001 - external API retry boundary
            last_error = exc
            if attempt >= attempts or not rio.is_retryable_api_error(exc):
                raise
            delay = config.api_retry_base_seconds * (2 ** (attempt - 1))
            print(
                f"[retry] {log_label}: attempt {attempt}/{attempts} failed: "
                f"{exc}; retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    raise RuntimeError(f"Qwen call failed for {log_label}") from last_error


# --- Stats / HTML report ------------------------------------------------------
def build_stats(rows: list[dict[str, Any]], config: RunConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "dry_run": config.dry_run,
        "source": rio.SOURCE_MODEL_FULL,
        "image_count": len({row["image_id"] for row in rows}),
        "pair_count": len(rows),
        "decidability_counts": dict(Counter(row.get("decidability", "") for row in rows)),
        "compliance_label_counts": dict(Counter(row.get("compliance_label", "") for row in rows)),
        "evidence_sufficiency_counts": dict(
            Counter(row.get("evidence_sufficiency_label", "") for row in rows)
        ),
        "image_quality_counts": dict(Counter(row.get("image_quality", "") for row in rows)),
        "human_review_required": True,
    }


def html_list(values: Any) -> str:
    values = values if isinstance(values, list) else [values]
    if not values:
        return "<span class='muted'>None</span>"
    return "<ul>" + "".join(f"<li>{html.escape(str(v))}</li>" for v in values) + "</ul>"


def write_html(
    path: Path,
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    output_dir: Path,
    *,
    title: str = "R-CES Qwen Full Annotation Comparison",
) -> None:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    cards = []
    for image_id, image_rows in sorted(by_image.items()):
        image_rel = os.path.relpath(rio.IMAGE_DIR / image_id, output_dir)
        blocks = []
        for row in image_rows:
            fields = "".join(
                f"<tr><th>{html.escape(k)}</th><td>{html_list(row.get(k)) if isinstance(row.get(k), list) else html.escape(str(row.get(k)))}</td></tr>"
                for k in (
                    "rule_name", "decision_scope", "decidability", "compliance_label",
                    "image_quality", "evidence_sufficiency_label", "observed_checkpoints",
                    "missing_checkpoints", "critical_missing_checkpoints", "confidence",
                    "visual_evidence", "missing_information", "rationale",
                )
            )
            blocks.append(f"<section class='pair'><h3>{html.escape(row['pair_id'])}</h3><table>{fields}</table></section>")
        cards.append(
            f"<article class='image-card'><h2>{html.escape(image_id)}</h2>"
            f"<img src='{html.escape(image_rel)}' alt='{html.escape(image_id)}'>" + "".join(blocks) + "</article>"
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#20242a}}
header,.image-card{{background:#fff;border:1px solid #dde1e7;border-radius:12px;padding:16px;margin-bottom:18px}}
img{{max-width:100%;max-height:720px;display:block;border-radius:8px;margin:12px 0}}
.pair{{border-top:1px solid #e6e9ee;padding-top:12px;margin-top:12px}}
table{{width:100%;border-collapse:collapse;font-size:.92rem}}
th{{text-align:left;width:34%;color:#586170;padding:4px 8px;border-bottom:1px solid #eef1f5;vertical-align:top}}
td{{padding:4px 8px;border-bottom:1px solid #eef1f5;vertical-align:top}}
ul{{margin:0 0 0 1.1rem;padding:0}}.muted{{color:#798292}}
</style></head><body><header><h1>{html.escape(title)} (source=model_full, never ground truth)</h1>
<p>Model: <code>{html.escape(str(stats['model']))}</code> | Dry run: <code>{html.escape(str(stats['dry_run']))}</code> | Images: {stats['image_count']} | Pairs: {stats['pair_count']}</p>
<p>Decidability: {html.escape(json.dumps(stats['decidability_counts'], ensure_ascii=False))}</p></header>
{''.join(cards)}</body></html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    config = RunConfig()
    validate_config(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.cache_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    rules = rio.load_rules()
    pairs = rio.load_jsonl(config.gold_path)
    groups = group_pairs_by_image(pairs, config.limit_images)
    groups = apply_pair_limit(groups, config.limit_pairs)
    if sum(len(group) for group in groups.values()) == 0:
        raise RuntimeError("No pairs selected for the Qwen baseline run")

    # Resume: rows already in the baseline file are kept and their images skipped.
    rows_by_id: dict[str, dict[str, Any]] = {}
    if config.resume:
        existing_rows = rio.load_jsonl(config.baseline_path)
        if config.drop_invalid_existing_rows and existing_rows:
            existing_rows, dropped = drop_invalid_rows(existing_rows, rules)
            if dropped:
                rio.write_jsonl(config.baseline_path, existing_rows, sort_keys=True)
                print(
                    f"[resume] dropped {len(dropped)} invalid existing rows from "
                    f"{config.baseline_path.name}"
                )
                for pair_id, errors in sorted(dropped.items())[:20]:
                    print(f"[resume]   {pair_id}: {'; '.join(errors)}")
                if len(dropped) > 20:
                    print(f"[resume]   ... {len(dropped) - 20} more")
        for row in existing_rows:
            rows_by_id[row["pair_id"]] = row
    done_pairs = set(rows_by_id)
    if done_pairs:
        print(f"[resume] {len(done_pairs)} pairs already annotated in {config.baseline_path.name}")

    def persist() -> None:
        ordered = sorted(rows_by_id.values(), key=lambda r: r["pair_id"])
        rio.write_jsonl(config.baseline_path, ordered, sort_keys=True)

    client = None if config.dry_run else make_client(config)
    template = rio.load_prompt_template(config.prompt_path)

    total = len(groups)
    processed = 0
    skipped = 0
    for index, (image_id, image_pairs) in enumerate(groups.items(), 1):
        if config.resume and all(pair["pair_id"] in done_pairs for pair in image_pairs):
            skipped += 1
            print(f"[{index}/{total}] skip {image_id} (already annotated, {len(image_pairs)} pairs)")
            continue
        image_path = config.image_dir / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        print(f"[{index}/{total}] Qwen annotation {image_id} ({len(image_pairs)} pairs)")
        if config.dry_run:
            annotations = dry_run_annotations(image_pairs)
        else:
            assert client is not None
            payload = build_prompt_payload(image_pairs, rules)
            prompt_text = rio.fill_prompt(template, image_id, payload)
            result = call_qwen_json(
                client,
                config,
                prompt_text=prompt_text,
                image_path=image_path,
                log_label=image_id,
            )
            annotations = result.get("annotations", [])
            if not isinstance(annotations, list):
                raise ValueError(f"Qwen response for {image_id} missing annotations list")
        for row in merge_rows(image_pairs, annotations, rules, config.model, config.dry_run):
            rows_by_id[row["pair_id"]] = row
        processed += 1
        # Persist after every image so an interruption never loses annotated work.
        persist()

    rows = sorted(rows_by_id.values(), key=lambda r: r["pair_id"])
    persist()  # normalise the canonical file even when nothing new was annotated

    stats = build_stats(rows, config)
    base = f"rces_qwen_full_baseline_{timestamp}"
    cache_jsonl = output_dir / f"{base}.jsonl"
    rio.write_jsonl(cache_jsonl, rows, sort_keys=True)
    (output_dir / f"{base}_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_html(output_dir / f"{base}.html", rows, stats, output_dir)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Annotated this run: {processed} images | skipped (resume): {skipped} | total pairs: {len(rows)}")
    print(f"Wrote baseline: {config.baseline_path.relative_to(rio.PROJECT_ROOT)}")
    print(f"Cache: {cache_jsonl.relative_to(rio.PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
