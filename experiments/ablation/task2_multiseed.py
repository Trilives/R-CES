"""R-CES Task-2 AUROC against each of the seven compliance VLMs, over 10 seeds.

The seven-VLM decoupling table was single-seed for the R-CES column. Each seed's
v1 head is trained once (deterministically -- the same heads camera_ready.py
trains) and scored against every VLM's cached compliance predictions, so the
cross-VLM spread and the seed spread can be reported separately. Answer-side
gates (MaxProb etc.) are properties of the VLM, not of an R-CES seed, and stay as
reported by experiments/baselines/cascade.py.

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_N_SEEDS=10 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  uv run python experiments/ablation/task2_multiseed.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import numpy as np  # noqa: E402

import config  # noqa: E402
import metrics  # noqa: E402
from data import splits  # noqa: E402
from rces import evaluate as rces_eval  # noqa: E402
from rces import train as rces_train  # noqa: E402

N_SEEDS = int(os.environ.get("RCES_N_SEEDS", "10"))
TAG = config._CROPAUG_TAG
VLMS = ("InternVL3_5-8B-HF", "gemma-4-E2B-it", "Qwen3.5-4B", "gemma-4-E4B-it",
        "Qwen3.6-35B-A3B", "Qwen3-VL-8B-Instruct", "Qwen3.5-9B")


def predictions_path(vlm: str) -> Path:
    directory = config.ARTIFACT_DIR / config._LOCAL_QWEN_ARTIFACT_NAMES[vlm]
    return directory / config._LOCAL_QWEN_PRED_FILE[config.LOCAL_USE_STRUCTURED]


def task2_auroc(head, test_ids: set[str], device, vlm: str) -> float:
    config.LOCAL_QWEN_PREDICTIONS_PATH = predictions_path(vlm)
    rows = [r for r in rces_eval.build_eval_rows(head, test_ids, device)
            if r.compliance_correct is not None]
    if len(rows) != 267:
        raise RuntimeError(f"{vlm}: expected 267 scored test pairs, got {len(rows)}")
    return float(metrics._auroc(np.asarray([r.final_answerability_score for r in rows]),
                                np.asarray([int(r.compliance_correct) for r in rows])))


def agg(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)),
            "min": float(array.min()), "max": float(array.max()), "n": int(len(array))}


def main() -> None:
    device = rces_train._device()
    config.INPUT_MODE = "full"
    config.HEAD_CKPT = config.ARTIFACT_DIR / f"rces_head_task2ms_{config.VISION_ENCODER}{TAG}.pt"
    # auroc[seed][vlm]
    auroc: list[dict[str, float]] = []
    for seed in range(N_SEEDS):
        config.SEED = seed
        split = splits.load_split()
        rces_train.main()
        head = rces_eval.load_head(device)
        test_ids = set(split["test"])
        auroc.append({vlm: task2_auroc(head, test_ids, device, vlm) for vlm in VLMS})
        print(f"seed {seed}: " + " ".join(f"{v}={a:.3f}" for v, a in auroc[-1].items()))

    matrix = np.asarray([[row[v] for v in VLMS] for row in auroc])  # seeds x vlms
    summary = {
        "per_vlm": {v: agg(matrix[:, j]) for j, v in enumerate(VLMS)},
        # Cross-VLM std inside each seed, then averaged: the spread a deployer
        # sees when swapping the answerer under one fixed R-CES head.
        "cross_vlm_std_per_seed": agg(matrix.std(axis=1, ddof=1)),
        "cross_vlm_mean_per_seed": agg(matrix.mean(axis=1)),
        "below_chance_cells": int((matrix < 0.5).sum()),
    }
    out = {"encoder": config.VISION_ENCODER, "vlms": list(VLMS), "seeds": list(range(N_SEEDS)),
           "std_convention": "sample (ddof=1)", "summary": summary,
           "per_seed": auroc}
    path = config.ARTIFACT_DIR / f"task2_multiseed{TAG}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {path.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
