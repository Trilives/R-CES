"""Generate isolated compliance predictions with the configured local Qwen VLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import config  # noqa: E402
import exp_io  # noqa: E402
from local_vlm.predictor import prediction_record_is_compatible, predict_pair  # noqa: E402
from local_vlm.runtime import (  # noqa: E402
    DryRunOptionRuntime,
    VLLMOptionRuntime,
    build_option_runtime,
)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.RCES_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    pairs = exp_io.load_locked_gold()
    if config.LOCAL_QWEN_LIMIT_PAIRS > 0:
        pairs = pairs[: config.LOCAL_QWEN_LIMIT_PAIRS]
    if not pairs:
        raise RuntimeError("No locked human pairs available for local Qwen prediction")

    if config.LOCAL_QWEN_DRY_RUN:
        runtime = DryRunOptionRuntime()
    elif config.LOCAL_VLM_IS_REMOTE:
        runtime = VLLMOptionRuntime(
            config.REMOTE_VLM_BASE_URL,
            config.REMOTE_VLM_MODEL_ID,
            model_name=config.LOCAL_QWEN_MODEL_NAME,
        )
    else:
        runtime = build_option_runtime(
            config.LOCAL_QWEN_MODEL_DIR,
            dtype=config.LOCAL_QWEN_DTYPE,
            device_map=config.LOCAL_QWEN_DEVICE_MAP,
        )
    rules = exp_io.rio.load_rules()
    config.LOCAL_VLM_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    existing_by_id: dict[str, dict] = {}
    if config.LOCAL_QWEN_PREDICTIONS_PATH.exists() and not config.LOCAL_QWEN_OVERWRITE:
        existing_by_id = {
            row["pair_id"]: row
            for row in exp_io.rio.load_jsonl(config.LOCAL_QWEN_PREDICTIONS_PATH)
            if row.get("model") == runtime.model_name
            and prediction_record_is_compatible(row)
        }
    rows_by_id = dict(existing_by_id)
    for index, pair in enumerate(pairs, 1):
        if pair["pair_id"] in rows_by_id:
            print(f"[{index}/{len(pairs)}] {pair['pair_id']} cached")
            continue
        image_path = exp_io.rio.IMAGE_DIR / pair["image_id"]
        if not image_path.is_file():
            # Crop-augmentation images live in a separate directory.
            image_path = exp_io.rio.CROP_IMAGE_DIR / pair["image_id"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        prediction = predict_pair(
            runtime,
            pair=pair,
            rule=rules.get(pair["rule_id"], {}),
            image_path=image_path,
        )
        rows_by_id = {**rows_by_id, pair["pair_id"]: prediction}
        selected_rows = [rows_by_id[row["pair_id"]] for row in pairs if row["pair_id"] in rows_by_id]
        exp_io.rio.write_jsonl(config.LOCAL_QWEN_PREDICTIONS_PATH, selected_rows, sort_keys=True)
        print(
            f"[{index}/{len(pairs)}] {pair['pair_id']} -> "
            f"{prediction['predicted_compliance']} ({prediction['confidence']:.4f})"
        )

    rows = [rows_by_id[pair["pair_id"]] for pair in pairs if pair["pair_id"] in rows_by_id]
    exp_io.rio.write_jsonl(config.LOCAL_QWEN_PREDICTIONS_PATH, rows, sort_keys=True)
    summary = {
        "model": runtime.model_name,
        "dry_run": config.LOCAL_QWEN_DRY_RUN,
        "pair_count": len(rows),
        "prediction_path": display_path(config.LOCAL_QWEN_PREDICTIONS_PATH),
    }
    (config.LOCAL_VLM_ARTIFACT_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
