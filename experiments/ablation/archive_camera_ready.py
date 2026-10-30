"""Persist camera-ready per-seed inputs and recompute corrected summaries.

Uses the original frozen features, never the resized public photograph inputs.
Every regenerated AUROC is checked against its stored original seed estimate.
The score archive, validation inputs and all seed heads are retained. Run once
per encoder with the same environment as camera_ready.py; outputs are isolated
under experiments/artifacts/camera_ready_audit/ and can resume after each seed.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

import config
import torch
from ablation import camera_ready as cr


def main():
    torch.set_num_threads(int(os.environ.get("RCES_CPU_THREADS", "2")))
    original_dir = config.ARTIFACT_DIR
    original = json.loads((original_dir / f"camera_ready_{config.VISION_ENCODER}{config._CROPAUG_TAG}.json").read_text())
    config.ARTIFACT_DIR = original_dir / "camera_ready_audit"
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    archive = config.ARTIFACT_DIR / f"camera_ready_scores_{config.VISION_ENCODER}{config._CROPAUG_TAG}"
    device = cr.rces_train._device()
    gold = {r["pair_id"]: r for r in cr.exp_io.load_locked_gold()}
    per_seed, frames = [], []
    for seed in range(cr.N_SEEDS):
        path = archive / f"seed_{seed:02d}.json.gz"
        if path.exists():
            with gzip.open(path, "rt") as stream:
                saved = json.load(stream)
            result, frame = saved["result"], saved["frames"]
        else:
            result, frame = cr.run_seed(device, seed, gold)
        for metric in ("checkpoint_pooled_auroc", "checkpoint_within_key", "pair_visual_pooled_auroc", "pair_within_rule", "task1_gate_on_auroc"):
            if abs(result[metric] - original["per_seed"][seed][metric]) > 1e-12:
                raise RuntimeError(f"{config.VISION_ENCODER} seed {seed}: regenerated {metric} differs from original; audit before replacing results")
        per_seed.append(result)
        frames.append(frame)
        print(f"{config.VISION_ENCODER} seed {seed}: original AUROCs match; raw validation/test scores archived", flush=True)
    summary = cr.summarise(per_seed, frames)
    for result in per_seed:
        for block in result["calibration"].values():
            block.pop("_scores", None)
    out = dict(original, summary=summary, per_seed=per_seed,
               metric_revision="threshold_tie_groups_and_original_image_family_bootstrap",
               original_aurocs_verified=True)
    path = config.ARTIFACT_DIR / f"camera_ready_{config.VISION_ENCODER}{config._CROPAUG_TAG}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Completed {config.VISION_ENCODER}: {path.name}", flush=True)


if __name__ == "__main__":
    main()
