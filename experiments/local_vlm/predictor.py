"""Prompt and output adapter for local compliance prediction.

Structured mode uses two linked calls:
  - `generate_text` produces a complete checkpoint analysis;
  - `predict_option` reads that validated analysis and emits restricted A/B/C odds.
The selected option is both `predicted_compliance` and `confidence_label`, so the
reported confidence and final label always come from the same model decision.
The compliance VLM never abstains (no `unknown`); the R-CES evidence-sufficiency
gate, which runs independently and does NOT read this JSON, owns abstention.
`config.LOCAL_USE_STRUCTURED=False` uses the direct A/B/C/D option ablation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config
import rces_io as rio

from local_vlm.runtime import OptionRuntime


LABEL_BY_OPTION = {
    "A": "compliant",
    "B": "non_compliant",
    "C": "not_applicable",
    "D": "unknown",
}
# The compliance VLM commits to one of three labels (abstention is the gate's job).
STRUCTURED_COMPLIANCE = ("compliant", "non_compliant", "not_applicable")
CHECKPOINT_JUDGMENTS = ("satisfied", "violated", "unresolved")

_PROMPT_TEMPLATE: str = ""
_STRUCTURED_TEMPLATE: str = ""
_STRUCTURED_DECISION_TEMPLATE: str = ""
PREDICTION_SCHEMA_VERSION = 3


def prediction_record_is_compatible(record: dict[str, Any]) -> bool:
    """Whether a cached prediction satisfies the corrected v2 contract."""
    predicted = record.get("predicted_compliance")
    probabilities = record.get("label_probabilities")
    if (
        record.get("prediction_schema_version") != PREDICTION_SCHEMA_VERSION
        or record.get("model_input_policy") != config.LOCAL_VLM_INPUT_POLICY
        or record.get("confidence_label") != predicted
        or not isinstance(probabilities, dict)
        or predicted not in probabilities
    ):
        return False
    try:
        return abs(float(probabilities[predicted]) - float(record["confidence"])) <= 1e-8
    except (KeyError, TypeError, ValueError):
        return False


def _prompt_template() -> str:
    """Load (once) the neutral 4-state A/B/C/D compliance prompt (confidence path)."""
    global _PROMPT_TEMPLATE
    if not _PROMPT_TEMPLATE:
        template = config.LOCAL_COMPLIANCE_PROMPT_PATH.read_text(encoding="utf-8")
        if "<<RULE_JSON>>" not in template:
            raise ValueError(
                f"{config.LOCAL_COMPLIANCE_PROMPT_PATH} must contain the <<RULE_JSON>> placeholder"
            )
        _PROMPT_TEMPLATE = template
    return _PROMPT_TEMPLATE


def _structured_template() -> str:
    """Load (once) the checkpoint-guided structured JSON prompt (label path)."""
    global _STRUCTURED_TEMPLATE
    if not _STRUCTURED_TEMPLATE:
        template = config.LOCAL_STRUCTURED_PROMPT_PATH.read_text(encoding="utf-8")
        if "<<RULE_JSON>>" not in template:
            raise ValueError(
                f"{config.LOCAL_STRUCTURED_PROMPT_PATH} must contain the <<RULE_JSON>> placeholder"
            )
        _STRUCTURED_TEMPLATE = template
    return _STRUCTURED_TEMPLATE


def _structured_decision_template() -> str:
    global _STRUCTURED_DECISION_TEMPLATE
    if not _STRUCTURED_DECISION_TEMPLATE:
        template = config.LOCAL_STRUCTURED_DECISION_PROMPT_PATH.read_text(encoding="utf-8")
        required = ("<<RULE_JSON>>", "<<CHECKPOINT_JSON>>")
        if any(placeholder not in template for placeholder in required):
            raise ValueError(
                f"{config.LOCAL_STRUCTURED_DECISION_PROMPT_PATH} must contain {required}"
            )
        _STRUCTURED_DECISION_TEMPLATE = template
    return _STRUCTURED_DECISION_TEMPLATE


def _rule_block(pair: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_name": pair.get("rule_name", rule.get("rule_name", "")),
        "visual_screening_rule": rule.get("visual_screening_rule", ""),
        "checkpoint_questions": rule.get("visual_checkpoints", {}),
    }


def build_compliance_prompt(pair: dict[str, Any], rule: dict[str, Any]) -> str:
    return _prompt_template().replace(
        "<<RULE_JSON>>", json.dumps(_rule_block(pair, rule), ensure_ascii=False, indent=2)
    )


def build_structured_prompt(pair: dict[str, Any], rule: dict[str, Any]) -> str:
    required, critical, questions = rio.checkpoints_for(rule)
    critical_set = set(critical)
    rule_block = {
        "rule_name": pair.get("rule_name", rule.get("rule_name", "")),
        "visual_screening_rule": rule.get("visual_screening_rule", ""),
        "checkpoints": [
            {"key": key, "question": questions.get(key, ""), "critical": key in critical_set}
            for key in required
        ],
    }
    return _structured_template().replace(
        "<<RULE_JSON>>", json.dumps(rule_block, ensure_ascii=False, indent=2)
    )


def build_structured_decision_prompt(
    pair: dict[str, Any],
    rule: dict[str, Any],
    checkpoint_analysis: list[dict[str, str]],
) -> str:
    return (
        _structured_decision_template()
        .replace("<<RULE_JSON>>", json.dumps(_rule_block(pair, rule), ensure_ascii=False, indent=2))
        .replace(
            "<<CHECKPOINT_JSON>>",
            json.dumps(checkpoint_analysis, ensure_ascii=False, indent=2),
        )
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort: parse the first {...} object in a free-form decode."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_structured(text: str, rule: dict[str, Any]) -> dict[str, Any]:
    """Coerce the structured decode into canonical fields.

    Returns {compliance, checkpoint_analysis, structured_ok}. compliance is None
    when the decode is unusable (caller falls back to the A/B/C/D option label).
    checkpoint_analysis is filtered to the rule's own checkpoint keys, with the
    judgment coerced to satisfied/violated/unresolved.
    """
    required_keys = rio.checkpoints_for(rule)[0]
    allowed_keys = set(required_keys)
    parsed = _extract_json(text)
    if parsed is None:
        return {
            "compliance": None,
            "checkpoint_analysis": [],
            "structured_ok": False,
            "missing_checkpoints": required_keys,
            "duplicate_checkpoints": [],
        }

    compliance = parsed.get("compliance")
    if compliance not in STRUCTURED_COMPLIANCE:
        compliance = None

    analysis_by_key: dict[str, dict[str, str]] = {}
    duplicate_keys: set[str] = set()
    raw_analysis = parsed.get("checkpoint_analysis")
    if isinstance(raw_analysis, list):
        for item in raw_analysis:
            if not isinstance(item, dict):
                continue
            key = item.get("checkpoint")
            if key not in allowed_keys:
                continue
            if key in analysis_by_key:
                duplicate_keys.add(str(key))
                continue
            judgment = item.get("judgment")
            if judgment not in CHECKPOINT_JUDGMENTS:
                judgment = "unresolved"
            observation = item.get("observation")
            analysis_by_key[str(key)] = {
                "checkpoint": str(key),
                "observation": str(observation) if observation is not None else "",
                "judgment": str(judgment),
            }

    missing_keys = [key for key in required_keys if key not in analysis_by_key]
    analysis = [analysis_by_key[key] for key in required_keys if key in analysis_by_key]
    structured_ok = compliance is not None and not missing_keys and not duplicate_keys
    return {
        "compliance": compliance,
        "checkpoint_analysis": analysis,
        "structured_ok": structured_ok,
        "missing_checkpoints": missing_keys,
        "duplicate_checkpoints": sorted(duplicate_keys),
    }


def _retry_structured_prompt(
    base_prompt: str,
    structured: dict[str, Any],
) -> str:
    problems = []
    if structured["missing_checkpoints"]:
        problems.append(f"missing={structured['missing_checkpoints']}")
    if structured["duplicate_checkpoints"]:
        problems.append(f"duplicates={structured['duplicate_checkpoints']}")
    if structured["compliance"] is None:
        problems.append("invalid compliance label")
    problem_text = "; ".join(problems) or "invalid JSON"
    return (
        f"{base_prompt}\n\nYour previous response was rejected: {problem_text}. "
        "Regenerate the entire JSON object. Include every listed checkpoint exactly once."
    )


def predict_pair(
    runtime: OptionRuntime,
    *,
    pair: dict[str, Any],
    rule: dict[str, Any],
    image_path: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "pair_id": pair["pair_id"],
        "image_id": pair["image_id"],
        "rule_id": pair.get("rule_id", ""),
        "model": runtime.model_name,
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "model_input_policy": config.LOCAL_VLM_INPUT_POLICY,
    }

    if config.LOCAL_USE_STRUCTURED:
        if config.LOCAL_STRUCTURED_MAX_ATTEMPTS < 1:
            raise ValueError("LOCAL_STRUCTURED_MAX_ATTEMPTS must be at least 1")
        base_prompt = build_structured_prompt(pair, rule)
        structured: dict[str, Any] | None = None
        for attempt in range(1, config.LOCAL_STRUCTURED_MAX_ATTEMPTS + 1):
            if attempt == 1:
                prompt = base_prompt
            else:
                assert structured is not None
                prompt = _retry_structured_prompt(base_prompt, structured)
            text = runtime.generate_text(
                image_path,
                prompt,
                max_new_tokens=config.LOCAL_STRUCTURED_MAX_NEW_TOKENS,
            )
            structured = parse_structured(text, rule)
            if structured["structured_ok"]:
                break
        assert structured is not None
        record["structured_attempts"] = attempt
        record["structured_ok"] = structured["structured_ok"]
        record["generated_compliance"] = structured["compliance"]
        record["structured_missing_checkpoints"] = structured["missing_checkpoints"]
        record["structured_duplicate_checkpoints"] = structured["duplicate_checkpoints"]
        if structured["structured_ok"]:
            record["checkpoint_analysis"] = structured["checkpoint_analysis"]
            decision_prompt = build_structured_decision_prompt(
                pair, rule, structured["checkpoint_analysis"]
            )
            prediction = runtime.predict_option(
                image_path,
                decision_prompt,
                allowed_options=("A", "B", "C"),
            )
            record["reasoning_mode"] = "structured"
        else:
            record["checkpoint_analysis"] = []
            prediction = runtime.predict_option(
                image_path,
                build_compliance_prompt(pair, rule),
                allowed_options=("A", "B", "C"),
            )
            record["reasoning_mode"] = "structured_incomplete_fallback_direct"
    else:
        prediction = runtime.predict_option(
            image_path,
            build_compliance_prompt(pair, rule),
            allowed_options=("A", "B", "C", "D"),
        )
        record["reasoning_mode"] = "direct_option"

    if prediction.option not in LABEL_BY_OPTION:
        raise ValueError(f"Unexpected local-VLM option: {prediction.option}")
    label_probabilities = {
        LABEL_BY_OPTION[option]: float(probability)
        for option, probability in prediction.probabilities.items()
    }
    option_label = LABEL_BY_OPTION[prediction.option]
    record["predicted_compliance"] = option_label
    record["confidence_label"] = option_label
    record["confidence"] = float(prediction.confidence)
    record["label_probabilities"] = label_probabilities
    record["option_label"] = option_label

    return record
