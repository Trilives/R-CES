"""Channel controls: is the R-CES score image-conditioned, or a rule prior?

Three reviewers independently asked the same question -- whether the learned
signal reflects image evidence or is recoverable from the rule text alone -- and
asked for a text-only baseline, a shuffled-image test, and rule-disjoint
evaluation. This driver runs the first two, and reports them on *both* the
pooled metric the paper headlines and the within-stratum metric that can
actually separate the two hypotheses (see src/stratified.py).

Arms (each trained from scratch, once per seed):
  full        -- image patches + checkpoint text
  text_only   -- patch tokens zeroed inside the head; the rule prior alone
  image_only  -- conditioning text zeroed; the image alone

Evaluations:
  every arm on the real test set, plus `full` re-scored against a deterministic
  permutation of the image->pair assignment (the shuffled-image control).

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_N_SEEDS=10 CUBLAS_WORKSPACE_CONFIG=:4096:8 UV_CACHE_DIR=/tmp/uv-cache \
  uv run python R-CES/experiments/ablation/evidence_controls.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import gzip  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
import exp_io  # noqa: E402
import metrics  # noqa: E402
import stratified  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces import evaluate as rces_eval  # noqa: E402
from rces import train as rces_train  # noqa: E402

N_SEEDS = int(os.environ.get("RCES_N_SEEDS", "10"))
N_BOOTSTRAP = int(os.environ.get("RCES_N_BOOTSTRAP", "3000"))
TAG = config._CROPAUG_TAG
ARMS = tuple(os.environ.get("RCES_CONTROL_ARMS", "full,text_only,image_only").split(","))
if not set(ARMS) <= {"full", "text_only", "image_only"}:
    raise ValueError("Unknown channel-control arm")
# Crop-augmented pairs carry this pair_id prefix; originals are the pilot gold.
CROP_PAIR_PREFIX = "rces_crop"


def _is_crop(pair_id: str) -> bool:
    return pair_id.startswith(CROP_PAIR_PREFIX)


def checkpoint_frame(probs: dict, records: list) -> dict[str, list]:
    """Flatten per-(pair, checkpoint) scores into aligned metric inputs."""
    frame: dict[str, list] = {k: [] for k in
                              ("score", "target", "image_id", "rule_id", "key", "pair_id")}
    for record in records:
        for key, target in record.resolvable_targets.items():
            if (record.pair_id, key) not in probs:
                continue
            frame["score"].append(probs[(record.pair_id, key)])
            frame["target"].append(int(target))
            frame["image_id"].append(record.image_id)
            frame["rule_id"].append(record.rule_id)
            # Stratum = the atomic supervised unit: a specific checkpoint of a
            # specific rule. Its text is identical across every pair in it.
            frame["key"].append(f"{record.rule_id}::{key}")
            frame["pair_id"].append(record.pair_id)
    frame["family_id"] = exp_io.original_image_families(frame["image_id"])
    return frame


def pair_frame(rows: list) -> dict[str, list]:
    """Pair-level scores, excluding the deterministic external-evidence gate.

    The gate zeroes the score from rule metadata alone, so including those pairs
    would credit the visual head for a rule-side decision.
    """
    frame: dict[str, list] = {k: [] for k in
                              ("score", "target", "image_id", "rule_id", "pair_id")}
    for row in rows:
        if row.decision_scope == "requires_external_evidence":
            continue
        frame["score"].append(row.visual_sufficiency_score)
        frame["target"].append(int(row.sufficiency_gold))
        frame["image_id"].append(row.image_id)
        frame["rule_id"].append(row.rule_id)
        frame["pair_id"].append(row.pair_id)
    frame["family_id"] = exp_io.original_image_families(frame["image_id"])
    return frame


def _subset(frame: dict[str, list], keep) -> dict[str, list]:
    index = [i for i, pid in enumerate(frame["pair_id"]) if keep(pid)]
    return {k: [v[i] for i in index] for k, v in frame.items()}


def score_frame(frame: dict[str, list], stratum_field: str) -> dict[str, object]:
    """Pooled AUROC and within-stratum AUROC for one (arm, slice)."""
    if not frame["score"] or len(set(frame["target"])) < 2:
        return {"n": len(frame["score"]), "pooled_auroc": float("nan"),
                "within_stratum": {"macro": float("nan"), "pooled": float("nan"),
                                   "n_strata_used": 0}}
    pooled = metrics._auroc(np.asarray(frame["score"], dtype=float),
                            np.asarray(frame["target"], dtype=int))
    within = stratified.stratified_auroc(frame["score"], frame["target"], frame[stratum_field])
    return {
        "n": len(frame["score"]),
        "positives": int(sum(frame["target"])),
        "negatives": int(len(frame["target"]) - sum(frame["target"])),
        "pooled_auroc": float(pooled),
        "within_stratum": within,
    }


def evaluate_arm(device, split, test_records, *, shuffle: bool) -> dict[str, object]:
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
    result = {
        "checkpoint": {
            "all": score_frame(ckpt, "key"),
            "originals": score_frame(_subset(ckpt, lambda p: not _is_crop(p)), "key"),
            "crops": score_frame(_subset(ckpt, _is_crop), "key"),
        },
        "pair_visual_normative": {
            "all": score_frame(pair, "rule_id"),
            "originals": score_frame(_subset(pair, lambda p: not _is_crop(p)), "rule_id"),
            "crops": score_frame(_subset(pair, _is_crop), "rule_id"),
        },
    }
    result["_frames"] = {"checkpoint": ckpt, "pair": pair}
    return result


def agg(values: list[float]) -> dict[str, float]:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(array):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {"mean": mean, "std": std, "min": float(array.min()),
            "max": float(array.max()), "n": int(len(array))}


def main() -> None:
    torch.set_num_threads(int(os.environ.get("RCES_CPU_THREADS", "2")))
    device = rces_train._device()
    seeds = list(range(N_SEEDS))
    # collected[arm][slice][metric] -> per-seed values
    collected: dict[str, dict] = {}
    seed_frames: dict[str, dict[str, list]] = {}

    for seed in seeds:
        config.SEED = seed
        split = splits.load_split()
        test_records = ds.build_pair_records(set(split["test"]))
        print(f"\n########## seed {seed} ##########")

        for arm in ARMS:
            config.INPUT_MODE = arm
            config.HEAD_CKPT = config.ARTIFACT_DIR / f"rces_head_channel_{arm}{TAG}.pt"
            archive_dir = config.ARTIFACT_DIR / f"channel_score_frames{TAG}" / arm
            archive_dir.mkdir(parents=True, exist_ok=True)
            seed_path = archive_dir / f"seed_{seed:02d}.json.gz"
            cached = None
            if seed_path.exists():
                with gzip.open(seed_path, "rt") as stream:
                    cached = json.load(stream)
            else:
                rces_train.main()
            for shuffle in (False, True) if arm == "full" else (False,):
                name = "full_shuffled_image" if shuffle else arm
                if cached is not None and not shuffle:
                    result = cached["result"]
                else:
                    result = evaluate_arm(device, split, test_records, shuffle=shuffle)
                frames = result["_frames"]
                output_dir = config.ARTIFACT_DIR / f"channel_score_frames{TAG}" / name
                output_dir.mkdir(parents=True, exist_ok=True)
                with gzip.open(output_dir / f"seed_{seed:02d}.json.gz", "wt") as stream:
                    json.dump({"seed": seed, "arm": name, "result": result}, stream)
                result = {k: v for k, v in result.items() if k != "_frames"}
                bucket = collected.setdefault(name, {})
                for level, slices in result.items():
                    for slice_name, values in slices.items():
                        target = bucket.setdefault(f"{level}::{slice_name}", {})
                        target.setdefault("pooled_auroc", []).append(values["pooled_auroc"])
                        target.setdefault("within_macro", []).append(
                            values["within_stratum"]["macro"])
                        target.setdefault("within_pooled", []).append(
                            values["within_stratum"]["pooled"])
                # Preserve per-seed rankings and resample shared original-image
                # families; scores from different heads are never pooled.
                store = seed_frames.setdefault(name, {})
                for level, frame in frames.items():
                    store.setdefault(level, []).append(frame)
                ck = result["checkpoint"]["all"]
                pr = result["pair_visual_normative"]["all"]
                print(f"  {name:<20} ckpt pooled={ck['pooled_auroc']:.4f} "
                      f"within={ck['within_stratum']['pooled']:.4f} | "
                      f"pair pooled={pr['pooled_auroc']:.4f} "
                      f"within={pr['within_stratum']['pooled']:.4f}")
        config.INPUT_MODE = "full"

    summary: dict[str, object] = {
        "seeds": seeds,
        "encoder": config.VISION_ENCODER,
        "include_crop_augmentation": config.INCLUDE_CROP_AUGMENTATION,
        "per_arm": {arm: {slice_name: {m: agg(v) for m, v in metrics_.items()}
                          for slice_name, metrics_ in bucket.items()}
                    for arm, bucket in collected.items()},
        "family_bootstrap": {},
    }
    for name, store in seed_frames.items():
        ck, pr = store["checkpoint"], store["pair"]
        summary["family_bootstrap"][name] = {
            "checkpoint_within_key": stratified.mean_of_seed_aurocs(ck, "key", n_bootstrap=N_BOOTSTRAP),
            "pair_within_rule": stratified.mean_of_seed_aurocs(pr, "rule_id", n_bootstrap=N_BOOTSTRAP),
            "checkpoint_pooled": stratified.mean_of_seed_aurocs(
                [dict(f, all_comparisons=["all"] * len(f["score"])) for f in ck],
                "all_comparisons", n_bootstrap=N_BOOTSTRAP),
        }

    out_path = config.ARTIFACT_DIR / f"evidence_controls{TAG}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
                        encoding="utf-8")

    print(f"\n===== channel controls ({len(seeds)} seeds) =====")
    header = f"{'arm':<22}{'ckpt pooled':>14}{'ckpt within-key':>18}{'pair pooled':>14}{'pair within-rule':>18}"
    print(header)
    for name in collected:
        c = summary["per_arm"][name]["checkpoint::all"]
        p = summary["per_arm"][name]["pair_visual_normative::all"]
        print(f"{name:<22}{c['pooled_auroc']['mean']:>9.4f}±{c['pooled_auroc']['std']:<4.3f}"
              f"{c['within_pooled']['mean']:>13.4f}±{c['within_pooled']['std']:<4.3f}"
              f"{p['pooled_auroc']['mean']:>9.4f}±{p['pooled_auroc']['std']:<4.3f}"
              f"{p['within_pooled']['mean']:>13.4f}±{p['within_pooled']['std']:<4.3f}")
    print("\nseed x original-image-family bootstrap 95% CI (per-seed AUROC):")
    for name, entry in summary["family_bootstrap"].items():
        for label, key in (("ckpt within-key", "checkpoint_within_key"),
                           ("pair within-rule", "pair_within_rule")):
            e = entry[key]
            print(f"  {name:<22}{label:<18}{e['point_mean_of_seeds']:.4f} "
                  f"[{e['ci_95_low']:.4f}, {e['ci_95_high']:.4f}] "
                  f"contains_0.5={e['ci_contains_0_5']}")
    print(f"\nwrote {out_path.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
