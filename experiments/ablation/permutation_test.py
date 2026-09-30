"""Multi-permutation shuffled-image test.

The single-permutation control (hierarchical_ci.py) pairs real vs shuffled across
training seeds, but every seed shares ONE image permutation, so its sign test is
conditional on that permutation. Here each seed is scored against K independent
permutations, giving a per-seed randomization null.

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_N_SEEDS=10 RCES_N_PERM=20 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  uv run python experiments/ablation/permutation_test.py
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
import stratified  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces import evaluate as rces_eval  # noqa: E402
from rces import train as rces_train  # noqa: E402
from ablation.evidence_controls import checkpoint_frame, pair_frame  # noqa: E402

N_SEEDS = int(os.environ.get("RCES_N_SEEDS", "10"))
N_PERM = int(os.environ.get("RCES_N_PERM", "20"))
TAG = config._CROPAUG_TAG


def _score(device, split, test_records):
    head = rces_eval.load_head(device)
    rows = rces_eval.build_eval_rows(head, set(split["test"]), device)
    probs = rces_eval.checkpoint_probs(head, test_records, device)
    ckpt = checkpoint_frame(probs, test_records)
    pair = pair_frame(rows)
    return (
        stratified.stratified_auroc(ckpt["score"], ckpt["target"], ckpt["key"])["pooled"],
        stratified.stratified_auroc(pair["score"], pair["target"], pair["rule_id"])["pooled"],
    )


def main() -> None:
    device = rces_train._device()
    config.INPUT_MODE = "full"
    out = {"n_seeds": N_SEEDS, "n_permutations": N_PERM, "per_seed": []}

    for seed in range(N_SEEDS):
        config.SEED = seed
        config.HEAD_CKPT = config.ARTIFACT_DIR / f"rces_head_perm{TAG}.pt"
        rces_train.main()
        split = splits.load_split()
        test_records = ds.build_pair_records(set(split["test"]))

        config.SHUFFLE_IMAGE_EVAL = False
        real_ckpt, real_pair = _score(device, split, test_records)

        perm_ckpt, perm_pair = [], []
        config.SHUFFLE_IMAGE_EVAL = True
        for p in range(N_PERM):
            config.SHUFFLE_IMAGE_SEED = 100000 + 997 * p
            c, pr = _score(device, split, test_records)
            perm_ckpt.append(c)
            perm_pair.append(pr)
        config.SHUFFLE_IMAGE_EVAL = False

        rec = {
            "seed": seed,
            "real_ckpt": real_ckpt,
            "real_pair": real_pair,
            "perm_ckpt_mean": float(np.mean(perm_ckpt)),
            "perm_pair_mean": float(np.mean(perm_pair)),
            "perm_ckpt_max": float(np.max(perm_ckpt)),
            "perm_pair_max": float(np.max(perm_pair)),
            # randomization p: (#perm >= real + 1) / (K + 1)
            "p_ckpt": (int(sum(1 for v in perm_ckpt if v >= real_ckpt)) + 1) / (N_PERM + 1),
            "p_pair": (int(sum(1 for v in perm_pair if v >= real_pair)) + 1) / (N_PERM + 1),
            "perm_ckpt": perm_ckpt,
            "perm_pair": perm_pair,
        }
        out["per_seed"].append(rec)
        print(f"  seed {seed}: real ckpt={real_ckpt:.4f} perm_mean={rec['perm_ckpt_mean']:.4f} "
              f"perm_max={rec['perm_ckpt_max']:.4f} p={rec['p_ckpt']:.3f} | "
              f"pair real={real_pair:.4f} perm_mean={rec['perm_pair_mean']:.4f} p={rec['p_pair']:.3f}",
              flush=True)

    for field in ("ckpt", "pair"):
        reals = [r[f"real_{field}"] for r in out["per_seed"]]
        allperm = [v for r in out["per_seed"] for v in r[f"perm_{field}"]]
        beat_all = sum(1 for r in out["per_seed"] if r[f"real_{field}"] > r[f"perm_{field}_max"])
        out[f"summary_{field}"] = {
            "real_mean": float(np.mean(reals)),
            "perm_mean": float(np.mean(allperm)),
            "perm_std": float(np.std(allperm, ddof=1)),
            "seeds_real_above_every_permutation": beat_all,
            "n_seeds": N_SEEDS,
            "min_attainable_p": 1 / (N_PERM + 1),
            "n_permutation_evaluations": len(allperm),
        }
        print(f"\n{field}: real {np.mean(reals):.4f} vs permuted "
              f"{np.mean(allperm):.4f}±{np.std(allperm, ddof=1):.4f} "
              f"({len(allperm)} permutation evaluations); real above EVERY permutation in "
              f"{beat_all}/{N_SEEDS} seeds")

    path = config.ARTIFACT_DIR / f"permutation_test{TAG}.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
