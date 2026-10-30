"""Shared helpers for the R-CES annotation pipeline (Stages 1-3).

This module centralises paths, the canonical label vocabulary, JSONL/rule IO,
image encoding, and the OpenAI client so the three stage scripts stay thin:

  annotate_web.py       Stage 1  human judgment, no model (LAN web UI)
  describe_with_gpt.py  optional GPT descriptive text under locked judgment
  baseline_qwen_full.py isolated Qwen full-pipeline model baseline (resumable)

Both the gold file (rces_pilot_pairs.jsonl, source=human) and the isolated
model baseline file (rces_qwen_full_baseline.jsonl, source=model_full -- the
generic model-baseline token, with the model recorded in `model`) share ONE
record shape, validated by rces_final_annotations.schema.json.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps

# R-CES is a self-contained, all-English line: every asset lives under R-CES/.
# PROJECT_ROOT therefore points at the R-CES root (this file is R-CES/src/<name>).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Paths ---
PAIRS_DIR = PROJECT_ROOT / "data" / "pairs"
GOLD_PATH = PAIRS_DIR / "rces_pilot_pairs.jsonl"
# Second annotator's independent pass over the ~20% double_annotation subset.
# Physically separate from gold so the second pass cannot see (and be anchored
# by) the first annotator's labels; aligned by pair_id for inter-annotator
# agreement (tmp/compute_iaa.py).
DOUBLE_PATH = PAIRS_DIR / "rces_double_annotation.jsonl"
BASELINE_PATH = PAIRS_DIR / "rces_qwen_full_baseline.jsonl"
# Human-screened crops of train/validation/test originals. Each crop inherits
# its parent's split; bootstrap inference clusters originals and crops together.
# The annotation file is kept separate from pilot gold and the model baseline.
CROP_AUG_PATH = PAIRS_DIR / "rces_crop_augmentation.jsonl"
# Test-expansion ("enhance") pairs: extra test-split images (image_split=
# test_expansion), human-screened with the full Stage-1 UI. Isolated from gold,
# the baseline, and the crop-augmentation pass; aligned to the same record shape.
ENHANCE_PATH = PROJECT_ROOT / "data" / "enhance" / "rces_enhance_pairs.jsonl"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
# Crop-augmentation images live in a SEPARATE directory from the original dataset
# images, so derived random crops never mix with the source images. The image_id
# namespace is still shared (crops use img_9NNNNN.jpg) because the SigLIP feature
# cache is a flat <image_id>.pt store.
CROP_IMAGE_DIR = PROJECT_ROOT / "data" / "images_crops"
CACHE_DIR = PROJECT_ROOT / "cache"
RULES_PATH = PROJECT_ROOT / "data" / "rule" / "rule.json"
PROMPT_DIR = PROJECT_ROOT / "Prompt"

# --- Canonical label vocabulary (one source of truth for all stages) ---
PLACEHOLDER = "to_be_annotated"
DECIDABILITY = ("decidable", "partially_decidable", "undecidable")
COMPLIANCE = ("compliant", "non_compliant", "not_applicable", "unknown")
IMAGE_QUALITY = ("good", "usable", "poor")
EVIDENCE_SUFFICIENCY = ("sufficient", "partial", "insufficient")
ANNOTATION_STATUS = ("pending", "judged", "described", "reviewed", "model_complete")
# Gold: human Stage-1 judgment. Any later free-text (visual_evidence /
# missing_information) is post-processing under the locked judgment -- never gold
# supervision and training-irrelevant -- so the source stays plain "human".
SOURCE_HUMAN = "human"
# Generic model-full-pipeline baseline token; the actual model (e.g. qwen3.7-plus)
# is recorded in each row's `model` field.
SOURCE_MODEL_FULL = "model_full"

BASELINE_WARNING = (
    "Model full-pipeline baseline; never ground truth. Produced without human "
    "judgment and isolated from the gold annotation pipeline; must never flow "
    "into rces_pilot_pairs.jsonl."
)

# --- Image encoding controls ---
IMAGE_LONGEST_EDGE = 1920  # Cap inputs at 1080p (1920 on the longest edge).
IMAGE_MAX_PIXELS = 1920 * 1080
IMAGE_JPEG_QUALITY = 85


# --- JSONL IO --------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, sort_keys: bool = False) -> None:
    """Atomic write: serialise to a temp file then replace, so an interrupted
    run never leaves a half-written gold file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n")
    os.replace(tmp, path)


# --- Rules -----------------------------------------------------------------
def load_rules(path: Path = RULES_PATH) -> dict[str, dict[str, Any]]:
    rules = json.loads(path.read_text(encoding="utf-8"))
    return {rule["rule_id"]: rule for rule in rules}


def checkpoints_for(rule: dict[str, Any]) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (required_checkpoints, critical_checkpoints, visual_checkpoints map)."""
    required = list(rule.get("required_checkpoints", []))
    critical = list(rule.get("critical_checkpoints", []))
    questions = dict(rule.get("visual_checkpoints", {}))
    return required, critical, questions


def critical_missing(missing: list[str], rule: dict[str, Any]) -> list[str]:
    """Derived field: never hand-filled. critical_missing = missing INTERSECT critical."""
    critical = set(rule.get("critical_checkpoints", []))
    return [key for key in missing if key in critical]


# The rule's *subject gate*: the presence checkpoint whose confirmed absence
# makes the pair `not_applicable`. For a decidable pair the gate is always
# `observed` (subject seen present, or seen absent), so the UI can pre-fill it.
# It is reliable only when a rule has exactly one critical `*_present` checkpoint;
# rules with several (a risk-side AND a protection-side, e.g. R-EDG-007's
# fall_risk_present + guardrail_present) need an explicit, human-confirmed gate —
# the protection side is NEVER the gate. Confirmed by the maintainer 2026-06-08.
SUBJECT_GATE_OVERRIDES = {
    "R-BHV-002-no-helmet": "person_present",
    "R-BHV-003-helmet-worn-incorrectly": "person_present",
    "R-EDG-003-roof-balcony-platform-edge": "roof_balcony_stair_platform_edge_present",
    "R-EDG-007-guardrail-height": "fall_risk_present",
    "R-OPN-009-service-shaft-opening": "shaft_opening_present",
}


def subject_gate(rule: dict[str, Any]) -> str | None:
    """The rule's subject-presence checkpoint, or None when it cannot be
    determined unambiguously (no `*_present`, or several critical ones without a
    confirmed override). Never returns a protection-side checkpoint by guesswork."""
    rule_id = rule.get("rule_id")
    if rule_id in SUBJECT_GATE_OVERRIDES:
        return SUBJECT_GATE_OVERRIDES[rule_id]
    critical = set(rule.get("critical_checkpoints", []))
    crit_present = [
        key for key in rule.get("visual_checkpoints", {})
        if key.endswith("_present") and key in critical
    ]
    return crit_present[0] if len(crit_present) == 1 else None


# --- Cross-field invariants (codebook §8; one source of truth) -------------
# Used by the annotation UI to gate inputs (annotate_web.py)
# and by R-CES/tmp/validate_rces_pairs.py to guard gold regardless of entry path.
TYPICAL_SUFFICIENCY = {
    "decidable": "sufficient",
    "partially_decidable": "partial",
    "undecidable": "insufficient",
}


def derive_evidence_sufficiency(decidability: str) -> str:
    """Derived, never hand-typed (codebook §7): evidence_sufficiency follows
    decidability one-to-one — decidable→sufficient, partially_decidable→partial,
    undecidable→insufficient. A not-yet-set decidability yields the placeholder so
    pending rows stay valid."""
    return TYPICAL_SUFFICIENCY.get(decidability, PLACEHOLDER)


def allowed_compliance(decidability: str) -> tuple[str, ...]:
    """compliance_label values consistent with a given decidability (codebook
    §8.1-8.3): an abstaining decidability forces `unknown`; a `decidable` one
    excludes it. A placeholder / not-yet-picked decidability imposes nothing."""
    if decidability == "decidable":
        return ("compliant", "non_compliant", "not_applicable")
    if decidability in ("partially_decidable", "undecidable"):
        return ("unknown",)
    return COMPLIANCE


def forced_compliance(decidability: str) -> str | None:
    """The single compliance value mandated by decidability, or None when the
    annotator still has a real choice (decidable) or hasn't picked yet."""
    allowed = allowed_compliance(decidability)
    return allowed[0] if len(allowed) == 1 else None


def consistency_errors(decidability: str, compliance: str, critical_missing: list[str]) -> list[str]:
    """Human-readable violations of the codebook §8 cross-field invariants.
    Empty == consistent. Placeholder values are treated as 'not yet set' and
    skipped so pending rows never trip it."""
    errs: list[str] = []
    if PLACEHOLDER in (decidability, compliance):
        return errs
    if compliance not in allowed_compliance(decidability):
        if decidability == "decidable":
            errs.append("decidable cannot pair with compliance=unknown (codebook §8.1)")
        else:
            errs.append(f"{decidability} requires compliance=unknown (codebook §8.2)")
    if decidability == "decidable" and critical_missing:
        errs.append(
            f"decidable forbidden while critical checkpoints are missing "
            f"{sorted(critical_missing)} (codebook §8.5)"
        )
    return errs


# --- Image -----------------------------------------------------------------
def encode_image(path: Path) -> tuple[str, str]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        longest = max(width, height)
        pixels = width * height
        scale = min(
            1.0,
            IMAGE_LONGEST_EDGE / longest,
            (IMAGE_MAX_PIXELS / pixels) ** 0.5,
        )
        if scale < 1.0:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
    return "image/jpeg", base64.standard_b64encode(buffer.getvalue()).decode("ascii")


# --- OpenAI ----------------------------------------------------------------
def make_client() -> OpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in project .env")
    return OpenAI(api_key=api_key, timeout=180.0, max_retries=0)


def is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    if status_code is not None:
        return False
    name = exc.__class__.__name__.lower()
    return any(token in name for token in ("timeout", "connection", "rate", "server"))


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_model_json(
    client: OpenAI,
    *,
    model: str,
    reasoning_effort: str | None,
    prompt_text: str,
    image_path: Path,
    max_output_tokens: int,
    retries: int,
    retry_base_seconds: float,
    log_label: str = "",
) -> dict[str, Any]:
    """Send one (text + image) request and return the parsed JSON object.

    Shared by Stage 2 (describe) and Stage 3 (full baseline); each caller pulls
    its own list (annotations / descriptions) out of the returned object.
    """
    mime, b64 = encode_image(image_path)
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": prompt_text},
        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
    ]
    attempts = max(1, retries + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": max_output_tokens,
            }
            if reasoning_effort:
                request_kwargs["reasoning"] = {"effort": reasoning_effort}
            response = client.responses.create(**cast(Any, request_kwargs))
            return extract_json_object(response.output_text)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts or not is_retryable_api_error(exc):
                raise
            delay = retry_base_seconds * (2 ** (attempt - 1))
            print(f"[retry] {log_label}: attempt {attempt}/{attempts} failed: {exc}; retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"OpenAI call failed for {log_label}") from last_error


# --- Prompt ----------------------------------------------------------------
def load_prompt_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8")
    for token in ("<<IMAGE_ID>>", "<<PAIRS_JSON>>"):
        if token not in template:
            raise ValueError(f"Prompt template {path} is missing placeholder {token}")
    return template


def fill_prompt(template: str, image_id: str, payload: list[dict[str, Any]]) -> str:
    return template.replace("<<IMAGE_ID>>", image_id).replace(
        "<<PAIRS_JSON>>", json.dumps(payload, ensure_ascii=False)
    )
