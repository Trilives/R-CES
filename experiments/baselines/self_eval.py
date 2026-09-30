"""Elicited-confidence Task-2 baselines for the active local/remote VLM.

Two signals, computed AFTER the compliance answer and tied to the model's OWN
proposed label (read from the compliance predictions, never from gold):

  P(True)   -- Kadavath et al. (2022): re-show image+rule+proposed label and ask
               "is this conclusion correct? A) Correct B) Incorrect"; take P(A).
  verbalized -- Tian et al. (2023): ask for an integer 0-100 confidence; /100.

Isolated output at config.SELF_EVAL_PREDICTIONS_PATH, never ground truth. Resumable
(persists after each pair; re-runs skip done pairs). Model is selected exactly like
the compliance driver, via RCES_LOCAL_QWEN_MODEL (local or remote vLLM):

  RCES_INCLUDE_CROP_AUG=1 RCES_LOCAL_QWEN_MODEL=Qwen3.5-4B \
  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/baselines/self_eval.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import config  # noqa: E402
import exp_io  # noqa: E402
from local_vlm.predictor import _rule_block  # noqa: E402
from local_vlm.runtime import (  # noqa: E402
    DryRunOptionRuntime,
    VLLMOptionRuntime,
    build_option_runtime,
)

SELF_EVAL_SCHEMA_VERSION = 1
LABEL_DISPLAY = {
    "compliant": "Compliant",
    "non_compliant": "Non-compliant",
    "not_applicable": "Not applicable / cannot be assessed",
}
VERBALIZED_MAX_NEW_TOKENS = 8


def _prompt(path: Path, rule: dict, pair: dict, proposed_label: str) -> str:
    rule_json = json.dumps(_rule_block(pair, rule), ensure_ascii=False, indent=2)
    return (
        path.read_text(encoding="utf-8")
        .replace("<<RULE_JSON>>", rule_json)
        .replace("<<PROPOSED_LABEL>>", LABEL_DISPLAY.get(proposed_label, proposed_label))
    )


def _parse_verbalized(text: str) -> tuple[float | None, str]:
    """First integer 0..100 in the decode -> [0,1]; None if unparseable."""
    match = re.search(r"\d{1,3}", text)
    if not match:
        return None, text.strip()
    value = max(0, min(100, int(match.group())))
    return value / 100.0, text.strip()


def _build_runtime():
    if config.LOCAL_QWEN_DRY_RUN:
        return DryRunOptionRuntime()
    if config.LOCAL_VLM_IS_REMOTE:
        return VLLMOptionRuntime(
            config.REMOTE_VLM_BASE_URL,
            config.REMOTE_VLM_MODEL_ID,
            model_name=config.LOCAL_QWEN_MODEL_NAME,
        )
    return build_option_runtime(
        config.LOCAL_QWEN_MODEL_DIR,
        dtype=config.LOCAL_QWEN_DTYPE,
        device_map=config.LOCAL_QWEN_DEVICE_MAP,
    )


def main() -> None:
    pairs = exp_io.load_locked_gold()
    if config.LOCAL_QWEN_LIMIT_PAIRS > 0:
        pairs = pairs[: config.LOCAL_QWEN_LIMIT_PAIRS]
    if not pairs:
        raise RuntimeError("No locked human pairs for self-evaluation")
    if not config.LOCAL_QWEN_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Compliance predictions missing: {config.LOCAL_QWEN_PREDICTIONS_PATH}. "
            "Run local_qwen35.py for this model first."
        )
    proposed = {
        row["pair_id"]: row["predicted_compliance"]
        for row in exp_io.rio.load_jsonl(config.LOCAL_QWEN_PREDICTIONS_PATH)
    }

    runtime = _build_runtime()
    rules = exp_io.rio.load_rules()
    out_path = config.SELF_EVAL_PREDICTIONS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_id: dict[str, dict] = {}
    if out_path.exists() and not config.LOCAL_QWEN_OVERWRITE:
        rows_by_id = {
            row["pair_id"]: row
            for row in exp_io.rio.load_jsonl(out_path)
            if row.get("model") == runtime.model_name
            and row.get("self_eval_schema_version") == SELF_EVAL_SCHEMA_VERSION
        }

    for index, pair in enumerate(pairs, 1):
        pid = pair["pair_id"]
        if pid in rows_by_id:
            print(f"[{index}/{len(pairs)}] {pid} cached")
            continue
        label = proposed.get(pid)
        if label is None:
            print(f"[{index}/{len(pairs)}] {pid} no proposed label -> skip")
            continue
        image_path = exp_io.rio.IMAGE_DIR / pair["image_id"]
        if not image_path.is_file():
            image_path = exp_io.rio.CROP_IMAGE_DIR / pair["image_id"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        rule = rules.get(pair["rule_id"], {})

        ptrue_pred = runtime.predict_option(
            image_path,
            _prompt(config.PTRUE_PROMPT_PATH, rule, pair, label),
            allowed_options=("A", "B"),
        )
        verbalized_text = runtime.generate_text(
            image_path,
            _prompt(config.VERBALIZED_PROMPT_PATH, rule, pair, label),
            max_new_tokens=VERBALIZED_MAX_NEW_TOKENS,
        )
        verbalized, verbalized_raw = _parse_verbalized(verbalized_text)

        rows_by_id[pid] = {
            "pair_id": pid,
            "image_id": pair["image_id"],
            "rule_id": pair.get("rule_id", ""),
            "model": runtime.model_name,
            "self_eval_schema_version": SELF_EVAL_SCHEMA_VERSION,
            "proposed_compliance": label,
            "ptrue": float(ptrue_pred.probabilities.get("A", 0.0)),
            "ptrue_option": ptrue_pred.option,
            "verbalized": verbalized if verbalized is not None else 0.5,
            "verbalized_parsed": verbalized is not None,
            "verbalized_raw": verbalized_raw,
        }
        selected = [rows_by_id[p["pair_id"]] for p in pairs if p["pair_id"] in rows_by_id]
        exp_io.rio.write_jsonl(out_path, selected, sort_keys=True)
        print(
            f"[{index}/{len(pairs)}] {pid} ptrue={rows_by_id[pid]['ptrue']:.3f} "
            f"verbalized={rows_by_id[pid]['verbalized']:.2f}"
            f"{'' if verbalized is not None else ' (unparsed)'}"
        )

    rows = [rows_by_id[p["pair_id"]] for p in pairs if p["pair_id"] in rows_by_id]
    exp_io.rio.write_jsonl(out_path, rows, sort_keys=True)
    print(json.dumps({
        "model": runtime.model_name,
        "pair_count": len(rows),
        "path": str(out_path.relative_to(config.RCES_ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
