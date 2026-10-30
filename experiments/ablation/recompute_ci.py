"""Recompute the within-stratum interval with the correct estimand.

The previous replicate pooled rows from several sampled seeds and computed ONE
AUROC over the union. That ranks scores produced by *different* heads against each
other, which is not a quantity anyone deploys: each head has its own score scale.
The mismatch is visible directly -- pooled 0.627 versus a per-seed mean of 0.640.

The estimand we want is "the within-stratum AUROC of an R-CES head trained with a
random seed, evaluated on a fresh sample of test images". So each replicate:
  1. draws ONE original-image-family bootstrap sample (shared -- the test family set is common to
     every seed),
  2. draws seeds with replacement,
  3. computes the within-stratum AUROC *separately per drawn seed* on those images,
  4. averages the per-seed AUROCs.

Reads the cached frames written by hierarchical_ci.py, so no retraining is needed.

  UV_CACHE_DIR=/tmp/uv-cache uv run python R-CES/experiments/ablation/recompute_ci.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import numpy as np  # noqa: E402

import config  # noqa: E402
import stratified  # noqa: E402

N_BOOTSTRAP = int(os.environ.get("RCES_N_BOOTSTRAP", "3000"))
TAG = config._CROPAUG_TAG


def main() -> None:
    frames_path = config.ARTIFACT_DIR / f"hierarchical_frames{TAG}.json"
    collected = json.loads(frames_path.read_text(encoding="utf-8"))
    summary = {}
    for label in ("real", "shuffled"):
        summary[f"{label}::checkpoint_within_key"] = stratified.mean_of_seed_aurocs(
            collected[label]["ckpt"], "key", n_bootstrap=N_BOOTSTRAP)
        summary[f"{label}::pair_within_rule"] = stratified.mean_of_seed_aurocs(
            collected[label]["pair"], "rule_id", n_bootstrap=N_BOOTSTRAP)

    # Paired per-seed difference (real - shuffled), the threshold-free headline test.
    paired = {}
    for level, field in (("checkpoint_within_key", "key"), ("pair_within_rule", "rule_id")):
        key = "ckpt" if level.startswith("checkpoint") else "pair"
        real = [stratified.stratified_auroc(f["score"], f["target"], f[field])["pooled"]
                for f in collected["real"][key]]
        shuf = [stratified.stratified_auroc(f["score"], f["target"], f[field])["pooled"]
                for f in collected["shuffled"][key]]
        deltas = np.asarray(real) - np.asarray(shuf)
        paired[level] = {
            "seeds_real_better": int((deltas > 0).sum()), "n_seeds": len(deltas),
            "mean_delta": float(deltas.mean()), "std_delta": float(deltas.std(ddof=1)),
        }
    summary["paired_real_minus_shuffled"] = paired

    out = config.ARTIFACT_DIR / f"within_stratum_ci{TAG}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
                   encoding="utf-8")
    print("=== within-stratum AUROC, mean-over-seeds estimand ===")
    for key, entry in summary.items():
        if key == "paired_real_minus_shuffled":
            continue
        marker = "contains 0.5" if entry["ci_contains_0_5"] else "EXCLUDES 0.5"
        print(f"  {key:34s} {entry['point_mean_of_seeds']:.4f} "
              f"[{entry['ci_95_low']:.4f}, {entry['ci_95_high']:.4f}]  {marker}")
    print("\npaired real - shuffled, per seed:")
    for key, entry in paired.items():
        print(f"  {key:34s} {entry['seeds_real_better']}/{entry['n_seeds']} seeds, "
              f"delta {entry['mean_delta']:+.4f}±{entry['std_delta']:.4f}")
    print(f"\nwrote {out.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
