"""Isolated run controls and artifact paths for R-CES v2."""

from __future__ import annotations

import config


ARCHITECTURE = "text_context_then_image_cross_attention"
HIDDEN_DIM = 256
CONTEXT_HEADS = 4
IMAGE_HEADS = 4
CONTEXT_LAYERS = 1
BATCH_SIZE = 32

ARTIFACT_DIR = (
    config.ARTIFACT_DIR / "rces-2" / f"{config.VISION_ENCODER}{config._CROPAUG_TAG}"
)
HEAD_CKPT = ARTIFACT_DIR / "head.pt"
PREDICTIONS_PATH = ARTIFACT_DIR / "predictions.jsonl"
EVAL_REPORT_PATH = ARTIFACT_DIR / "eval_report.json"
