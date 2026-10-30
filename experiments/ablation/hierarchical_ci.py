"""Train R-CES across seeds, cache per-seed score frames, and interval the controls.

The interval comes from `stratified.mean_of_seed_aurocs`, which resamples training
seeds and one shared original-image-family sample and averages per-seed AUROCs; see that
function for why the per-seed step matters. Per-seed frames are cached to
`hierarchical_frames*.json` so the interval can be recomputed (see
`recompute_ci.py`) without retraining.

Only the `full` arm is retrained (once per seed); it is then scored twice, on the
real image->pair assignment and on the shuffled one.

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_N_SEEDS=10 CUBLAS_WORKSPACE_CONFIG=:4096:8 UV_CACHE_DIR=/tmp/uv-cache \
  uv run python R-CES/experiments/ablation/hierarchical_ci.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
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
N_BOOTSTRAP = int(os.environ.get("RCES_N_BOOTSTRAP", "5000"))
TAG = config._CROPAUG_TAG


def main() -> None:
    device = rces_train._device()
    config.INPUT_MODE = "full"
    collected: dict[str, dict[str, list]] = {"real": {"ckpt": [], "pair": []},
                                             "shuffled": {"ckpt": [], "pair": []}}
    per_seed_point: dict[str, list[float]] = defaultdict(list)

    for seed in range(N_SEEDS):
        config.SEED = seed
        config.HEAD_CKPT = config.ARTIFACT_DIR / f"rces_head_hier{TAG}.pt"
        rces_train.main()
        split = splits.load_split()
        test_records = ds.build_pair_records(set(split["test"]))
        for label, shuffle in (("real", False), ("shuffled", True)):
            previous = config.SHUFFLE_IMAGE_EVAL
            config.SHUFFLE_IMAGE_EVAL = shuffle
            try:
                head = rces_eval.load_head(device)
                rows = rces_eval.build_eval_rows(head, set(split["test"]), device)
                probs = rces_eval.checkpoint_probs(head, test_records, device)
            finally:
                config.SHUFFLE_IMAGE_EVAL = previous
            ckpt = checkpoint_frame(probs, test_records)
            pair = pair_frame(rows)
            collected[label]["ckpt"].append(ckpt)
            collected[label]["pair"].append(pair)
            per_seed_point[f"{label}::ckpt"].append(
                stratified.stratified_auroc(ckpt["score"], ckpt["target"], ckpt["key"])["pooled"])
            per_seed_point[f"{label}::pair"].append(
                stratified.stratified_auroc(pair["score"], pair["target"], pair["rule_id"])["pooled"])
        print(f"  seed {seed}: real ckpt={per_seed_point['real::ckpt'][-1]:.4f} "
              f"pair={per_seed_point['real::pair'][-1]:.4f} | "
              f"shuffled ckpt={per_seed_point['shuffled::ckpt'][-1]:.4f} "
              f"pair={per_seed_point['shuffled::pair'][-1]:.4f}")

    summary: dict[str, object] = {"n_seeds": N_SEEDS, "hierarchical": {}, "per_seed": {}}
    for label in ("real", "shuffled"):
        summary["hierarchical"][f"{label}::checkpoint_within_key"] = stratified.mean_of_seed_aurocs(
            collected[label]["ckpt"], "key", n_bootstrap=N_BOOTSTRAP)
        summary["hierarchical"][f"{label}::pair_within_rule"] = stratified.mean_of_seed_aurocs(
            collected[label]["pair"], "rule_id", n_bootstrap=N_BOOTSTRAP)
    for key, values in per_seed_point.items():
        array = np.asarray(values, dtype=float)
        summary["per_seed"][key] = {
            "mean": float(array.mean()), "std": float(array.std(ddof=1)),
            "min": float(array.min()), "max": float(array.max()),
            "seeds_above_0_5": int((array > 0.5).sum()), "n": len(array),
        }

    frames_out = config.ARTIFACT_DIR / f"hierarchical_frames{TAG}.json"
    frames_out.write_text(json.dumps(collected, ensure_ascii=False, default=float) + "\n",
                          encoding="utf-8")
    print(f"cached per-seed frames -> {frames_out.name}")

    out = config.ARTIFACT_DIR / f"within_stratum_ci{TAG}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
                   encoding="utf-8")
    print(f"\n===== hierarchical (seed x original-image-family) bootstrap, {N_SEEDS} seeds =====")
    for key, entry in summary["hierarchical"].items():
        marker = "contains 0.5" if entry["ci_contains_0_5"] else "EXCLUDES 0.5"
        print(f"  {key:38s} {entry['point_mean_of_seeds']:.4f} "
              f"[{entry['ci_95_low']:.4f}, {entry['ci_95_high']:.4f}]  {marker}")
    print("\nper-seed spread:")
    for key, entry in summary["per_seed"].items():
        print(f"  {key:24s} {entry['mean']:.4f}±{entry['std']:.4f} "
              f"[{entry['min']:.3f}, {entry['max']:.3f}]  above 0.5 in "
              f"{entry['seeds_above_0_5']}/{entry['n']} seeds")
    print(f"\nwrote {out.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
