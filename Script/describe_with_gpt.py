"""Stage 2 - GPT fills descriptive text under LOCKED human judgment.

Reads the gold file (rces_pilot_pairs.jsonl), takes every row whose
annotation_status == "judged" (human judgment already locked in Stage 1), and
asks GPT-5.5 to write ONLY descriptive fields (visual_evidence,
missing_information) consistent with that judgment. The model never touches a
judgment field; if the image contradicts the judgment it raises FLAG_CONFLICT
for a human to adjudicate in Stage 3 review.

  uv run python Script/describe_with_gpt.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rces_io as rio  # noqa: E402

# --- Run controls ---
MODEL = "gpt-5.5"
REASONING_EFFORT: str | None = "low"
DRY_RUN = False  # True writes deterministic placeholder descriptions, no API call.
MAX_OUTPUT_TOKENS = 4096
API_RETRIES = 3
API_RETRY_BASE_SECONDS = 2.0
PROMPT_PATH = rio.PROMPT_DIR / "describe_prompt.md"

# The model may only ever write these keys back into a gold row.
DESCRIPTION_FIELDS = ("visual_evidence", "missing_information")


@dataclass(frozen=True)
class RunConfig:
    gold_path: Path = rio.GOLD_PATH
    image_dir: Path = rio.IMAGE_DIR
    prompt_path: Path = PROMPT_PATH
    model: str = MODEL
    reasoning_effort: str | None = REASONING_EFFORT
    dry_run: bool = DRY_RUN
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    api_retries: int = API_RETRIES
    api_retry_base_seconds: float = API_RETRY_BASE_SECONDS


def locked_payload(row: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    """The frozen human judgment handed to the model as ground truth."""
    _, _, questions = rio.checkpoints_for(rule)
    return {
        "pair_id": row["pair_id"],
        "rule_name": row["rule_name"],
        "decision_scope": row["decision_scope"],
        "decidability": row["decidability"],
        "compliance_label": row["compliance_label"],
        "image_quality": row["image_quality"],
        "evidence_sufficiency_label": row["evidence_sufficiency_label"],
        "observed_checkpoints": row.get("observed_checkpoints", []),
        "missing_checkpoints": row.get("missing_checkpoints", []),
        "critical_missing_checkpoints": row.get("critical_missing_checkpoints", []),
        "checkpoint_questions": questions,
    }


def coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def dry_run_descriptions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        row["pair_id"]: {
            "pair_id": row["pair_id"],
            "visual_evidence": [],
            "missing_information": ["dry_run_no_model_call"],
            "flag_conflict": False,
            "conflict_note": "",
        }
        for row in rows
    }


def apply_descriptions(
    row: dict[str, Any],
    description: dict[str, Any],
    model: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Return a copy of the gold row with ONLY descriptive fields and Stage-2
    provenance changed. Judgment fields are passed through untouched."""
    updated = dict(row)
    updated["visual_evidence"] = coerce_list(description.get("visual_evidence"))
    updated["missing_information"] = coerce_list(description.get("missing_information"))
    updated["description_model"] = None if dry_run else model
    updated["description_is_dry_run"] = dry_run
    updated["description_flag_conflict"] = bool(description.get("flag_conflict", False))
    updated["description_conflict_note"] = str(description.get("conflict_note", ""))
    updated["annotation_status"] = "described"
    return updated


def main() -> None:
    config = RunConfig()
    if config.reasoning_effort not in {None, "low", "medium", "high"}:
        raise ValueError('REASONING_EFFORT must be None, "low", "medium", or "high"')

    rules = rio.load_rules()
    rows = rio.load_jsonl(config.gold_path)
    index_by_pair = {row["pair_id"]: i for i, row in enumerate(rows)}

    todo = [row for row in rows if row.get("annotation_status") == "judged"]
    if not todo:
        print("No rows with annotation_status='judged'; nothing to describe.")
        return

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in todo:
        by_image[row["image_id"]].append(row)

    client = None if config.dry_run else rio.make_client()
    template = rio.load_prompt_template(config.prompt_path)

    flagged = 0
    described = 0
    for index, (image_id, image_rows) in enumerate(sorted(by_image.items()), 1):
        image_path = config.image_dir / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        print(f"[{index}/{len(by_image)}] describing {image_id} ({len(image_rows)} pairs)")

        if config.dry_run:
            descriptions = dry_run_descriptions(image_rows)
        else:
            assert client is not None
            payload = [locked_payload(row, rules.get(row["rule_id"], {})) for row in image_rows]
            prompt_text = rio.fill_prompt(template, image_id, payload)
            result = rio.call_model_json(
                client,
                model=config.model,
                reasoning_effort=config.reasoning_effort,
                prompt_text=prompt_text,
                image_path=image_path,
                max_output_tokens=config.max_output_tokens,
                retries=config.api_retries,
                retry_base_seconds=config.api_retry_base_seconds,
                log_label=image_id,
            )
            items = result.get("descriptions", [])
            if not isinstance(items, list):
                raise ValueError(f"Model response for {image_id} missing descriptions list")
            descriptions = {str(d.get("pair_id", "")): d for d in items}

        for row in image_rows:
            description = descriptions.get(row["pair_id"], {})
            updated = apply_descriptions(row, description, config.model, config.dry_run)
            rows[index_by_pair[row["pair_id"]]] = updated
            described += 1
            if updated["description_flag_conflict"]:
                flagged += 1
                print(f"  FLAG_CONFLICT {row['pair_id']}: {updated['description_conflict_note']}")

    rio.write_jsonl(config.gold_path, rows)
    print(f"Described {described} pairs ({flagged} FLAG_CONFLICT). Wrote {config.gold_path.relative_to(rio.PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
