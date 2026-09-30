"""v2 head under pooled and within-stratum AUROC, over 10 seeds.

The author response committed to reporting both heads on both metrics; the
v1 figures come from experiments/ablation/evidence_controls.py and camera_ready.py.
Same estimators (src/stratified.py), same test population (checkpoint level: all
test checkpoints; pair level: visual+normative pairs, external gate excluded).

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_N_SEEDS=10 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  uv run python experiments/rces-2/within_stratum.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
RCES_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(RCES_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import config  # noqa: E402
import metrics  # noqa: E402
import stratified  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402

import evaluate as v2_evaluate  # noqa: E402  (rces-2/evaluate.py)
from ablation import _reset_ablations, train_head  # noqa: E402  (rces-2/ablation.py)
from train import device  # noqa: E402

N_SEEDS = int(os.environ.get("RCES_N_SEEDS", "10"))
N_BOOTSTRAP = int(os.environ.get("RCES_N_BOOTSTRAP", "3000"))
EXTERNAL = "requires_external_evidence"


def frames(head, test_records, active_device) -> tuple[dict, dict]:
    probs = v2_evaluate.probabilities_for_records(head, test_records, active_device)
    ckpt = {k: [] for k in ("score", "target", "image_id", "key")}
    for record in test_records:
        for key, target in record.resolvable_targets.items():
            if (record.pair_id, key) in probs:
                ckpt["score"].append(probs[(record.pair_id, key)])
                ckpt["target"].append(int(target))
                ckpt["image_id"].append(record.image_id)
                ckpt["key"].append(f"{record.rule_id}::{key}")
    pair = {k: [] for k in ("score", "target", "image_id", "rule_id")}
    for row in v2_evaluate.build_eval_rows(head, test_records, active_device):
        if row.decision_scope == EXTERNAL:
            continue
        pair["score"].append(row.visual_sufficiency_score)
        pair["target"].append(int(row.sufficiency_gold))
        pair["image_id"].append(row.image_id)
        pair["rule_id"].append(row.rule_id)
    return ckpt, pair


def pooled(frame) -> float:
    return float(metrics._auroc(np.asarray(frame["score"], dtype=float),
                                np.asarray(frame["target"], dtype=int)))


def agg(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)),
            "min": float(array.min()), "max": float(array.max()), "n": int(len(array))}


def main() -> None:
    active_device = device()
    per_seed, ckpt_frames, pair_frames = [], [], []
    for seed in range(N_SEEDS):
        config.SEED = seed
        _reset_ablations()
        split = splits.load_split()
        train_records = ds.build_pair_records(set(split["train"]))
        val_records = ds.build_pair_records(set(split["val"]))
        test_records = ds.build_pair_records(set(split["test"]))
        head, _ = train_head(active_device, train_records, val_records)
        ckpt, pair = frames(head, test_records, active_device)
        ckpt_frames.append(ckpt)
        pair_frames.append(pair)
        result = {
            "checkpoint_pooled": pooled(ckpt),
            "checkpoint_within_key": float(stratified.stratified_auroc(
                ckpt["score"], ckpt["target"], ckpt["key"])["pooled"]),
            "pair_visual_pooled": pooled(pair),
            "pair_within_rule": float(stratified.stratified_auroc(
                pair["score"], pair["target"], pair["rule_id"])["pooled"]),
        }
        per_seed.append(result)
        print(f"seed {seed}: " + " ".join(f"{k}={v:.4f}" for k, v in result.items()))

    summary = {k: agg([s[k] for s in per_seed]) for k in per_seed[0]}
    summary["checkpoint_within_key_ci"] = stratified.mean_of_seed_aurocs(
        ckpt_frames, "key", n_bootstrap=N_BOOTSTRAP)
    summary["pair_within_rule_ci"] = stratified.mean_of_seed_aurocs(
        pair_frames, "rule_id", n_bootstrap=N_BOOTSTRAP)
    out = {"head": "v2", "encoder": config.VISION_ENCODER, "seeds": list(range(N_SEEDS)),
           "std_convention": "sample (ddof=1)", "summary": summary, "per_seed": per_seed}
    path = config.ARTIFACT_DIR / f"v2_within_stratum{config._CROPAUG_TAG}.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
