"""Dump v2 (ContextualRCESHead) per-pair sufficiency scores for val and test.

Writes {split: {pair_id: {final_answerability_score, sufficiency_gold,
compliance_gold, image_id}}} so the Task-2 cascade gates can be recomputed offline
with the v2 head swapped in for v1. Reuses the rces-2 evaluate path verbatim.

  RCES_INCLUDE_CROP_AUG=1 RCES_VISION_ENCODER=siglip2_512 \
  RCES_LOCAL_QWEN_MODEL=InternVL3_5-8B-HF UV_CACHE_DIR=/tmp/uv-cache \
  uv run python experiments/rces-2/dump_v2_scores.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
RCES_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(RCES_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(HERE))

import evaluate as v2_eval  # noqa: E402  (rces-2/evaluate.py)
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402

OUT = RCES_ROOT / "experiments" / "artifacts" / "rces-2" / "v2_scores_val_test.json"


def main() -> None:
    dev = v2_eval.device()
    head = v2_eval.load_head(dev)
    split = splits.load_split()
    out: dict[str, dict] = {}
    for name in ("val", "test"):
        records = ds.build_pair_records(set(split[name]))
        rows = v2_eval.build_eval_rows(head, records, dev)
        out[name] = {
            r.pair_id: {
                "final_answerability_score": float(r.final_answerability_score),
                "sufficiency_gold": int(r.sufficiency_gold),
                "compliance_gold": r.compliance_gold,
                "image_id": r.image_id,
            }
            for r in rows
        }
        print(f"{name}: {len(out[name])} pairs")
    OUT.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
