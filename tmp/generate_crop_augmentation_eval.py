"""Crop-augmentation for the validation and test experiment splits.

Mirrors the train crop-augmentation line (generate_crop_augmentation.py +
regenerate_crop_rules.py) but builds crops for the held-out splits so val/test
can be expanded by ~1/3, like train (train: 600 gold pairs + 200 crops). Each
split's crops come ONLY from that split's mother images, and the experiment
split (experiments/artifacts/rces_image_split.json) is image-disjoint, so val
and test crops can never overlap each other or train -- isolation is structural,
no cross-contamination.

What it does (one pass, append-only):
  1. pick eligible mother images per split (those with visual_screening_only
     gold pairs that have critical checkpoints -- the clean, vision-decidable
     crop targets, same filter as the train line);
  2. cut one random crop per mother into data/images_crops/ with a per-split id
     block (val: img_910001+, test: img_920001+; train owns 900001-90xxxx);
  3. bind each crop to its TOP-2 rules by FRESH retrieval ON THE CROP (reusing
     the train line's hybrid router: SigLIP-224, 9B VLM for non-opening top-1),
     rank0=target / rank1=related, capped per split (~1/3 of the split);
  4. APPEND the records to rces_crop_augmentation.jsonl with image_split=val/test.
     Existing rows (train, and any already-written val/test) are never rewritten,
     so the 70 already-judged train crops are untouched.

Judgment fields stay placeholders -- a human screens them with annotate_web 3
(pass `augment` reads the whole file, so val/test rows are picked up with no
wiring change). The crops are NOT yet consumed by the experiments (build_pair_records
reads only locked gold today), matching train's staged state.

Run controls are module constants (the repo avoids CLI flags). DRY_RUN previews
the plan and loads no model; flip it to write crops + run retrieval (GPU):

    UV_CACHE_DIR=/tmp/uv-cache RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B \
        uv run python tmp/generate_crop_augmentation_eval.py

Validate: uv run python tmp/validate_rces_pairs.py data/pairs/rces_crop_augmentation.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
import rces_io as rio  # noqa: E402
import exp_io as eio  # noqa: E402  (critical_checkpoints)
import regenerate_crop_rules as RG  # noqa: E402  (retrieval router + SigLIP scoring, reused)

# ---- Run controls (edit these) ----------------------------------------------
DRY_RUN = False                      # True: preview only; write no images, rows, or model inputs.
TARGET_SPLITS = ("val", "test")      # experiment splits to crop (image-disjoint).
SPLIT_IMAGE_ID_START = {"val": 910001, "test": 920001}  # per-split crop id block.
SPLIT_TARGET_RECORDS = {"val": 67, "test": 67}          # ~1/3 of each split's 200 gold pairs.
N_CROPS_PER_IMAGE = 1                 # random crops per mother (max mother diversity).
CROP_SCALE_RANGE = (0.30, 0.90)       # per-axis fraction of the mother edge (random aspect).
RULE_DECISION_SCOPES = ("visual_screening_only",)  # crop-target rules (same as train).
RANDOM_SEED = 20260617                # mother visiting order + crop boxes (reproducible).
JPEG_QUALITY = 92
SKIP_VLM = RG.SKIP_VLM                # honour the train router's default (hybrid SigLIP->VLM).
SPLIT_PATH = rio.PROJECT_ROOT / "experiments" / "artifacts" / "rces_image_split.json"
# -----------------------------------------------------------------------------

PLACEHOLDER_JUDGMENT = dict(RG.PLACEHOLDER_JUDGMENT)
ID_BLOCK = 10000  # each split owns a 10k id block (900k train, 910k val, 920k test).


def load_split(split_name: str) -> list[str]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if split_name not in split:
        raise KeyError(f"Split {split_name!r} absent from {SPLIT_PATH}")
    return sorted(split[split_name])


def eligible_mothers(image_ids: list[str], rules: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Mother image_id -> its weak label, for images with at least one
    visual_screening_only gold pair that has critical checkpoints."""
    wanted = set(image_ids)
    out: dict[str, str] = {}
    for pair in rio.load_jsonl(rio.GOLD_PATH):
        if pair["image_id"] not in wanted:
            continue
        if pair["decision_scope"] not in RULE_DECISION_SCOPES:
            continue
        rule = rules.get(pair["rule_id"])
        if rule is None or not eio.critical_checkpoints(rule):
            continue
        out.setdefault(pair["image_id"], pair["image_weak_label"])
    return out


def random_box(width: int, height: int, rng: random.Random) -> tuple[int, int, int, int]:
    lo, hi = CROP_SCALE_RANGE
    crop_w = max(1, round(width * rng.uniform(lo, hi)))
    crop_h = max(1, round(height * rng.uniform(lo, hi)))
    left = rng.randint(0, max(0, width - crop_w))
    top = rng.randint(0, max(0, height - crop_h))
    return left, top, left + crop_w, top + crop_h


def existing_state(rows: list[dict[str, Any]]) -> tuple[dict[int, set[str]], int]:
    """Per-block set of mother ids already cropped, and the next global pair index
    (rces_crop_* ids are one global sequence; crop image ids are per-split blocks)."""
    done_by_block: dict[int, set[str]] = {}
    next_pair = 1
    for r in rows:
        n = int(r["image_id"][len("img_"):].split(".")[0])
        done_by_block.setdefault(n // ID_BLOCK, set()).add(r.get("base_image_id", ""))
        p = int(r["pair_id"][len("rces_crop_"):])
        if p >= next_pair:
            next_pair = p + 1
    return done_by_block, next_pair


def make_record(
    pair_idx: int,
    image_id: str,
    image_split: str,
    rule: dict[str, Any],
    role: str,
    method: str,
    score: float,
    prov: dict[str, Any],
    target_category: str,
) -> dict[str, Any]:
    return {
        "pair_id": f"rces_crop_{pair_idx:06d}",
        "image_id": image_id,
        "image_split": image_split,
        "rule_id": rule["rule_id"],
        "rule_name": rule["rule_name"],
        "major_category": rule["major_category"],
        "subcategory": rule["subcategory"],
        "decision_scope": rule["decision_scope"],
        "rule_role": role,
        "image_weak_label": prov["image_weak_label"],
        "weak_candidate_category": target_category,
        **PLACEHOLDER_JUDGMENT,
        "source": rio.SOURCE_HUMAN,
        "double_annotation": False,
        "base_image_id": prov["base_image_id"],
        "crop_bbox": prov["crop_bbox"],
        "augmentation": prov["augmentation"],
        "retrieval_method": method,
        "retrieval_score": round(score, 6),
    }


def main() -> None:
    rules = rio.load_rules()
    existing = rio.load_jsonl(rio.CROP_AUG_PATH)
    done_by_block, next_pair = existing_state(existing)

    # ---- Plan + cut crops per split (cheap, deterministic) ----
    plans: dict[str, dict[str, Any]] = {}
    print("=" * 64)
    print("R-CES crop-augmentation: VAL/TEST" + ("  [DRY RUN]" if DRY_RUN else ""))
    print("=" * 64)
    for split_name in TARGET_SPLITS:
        start = SPLIT_IMAGE_ID_START[split_name]
        block = start // ID_BLOCK
        done = done_by_block.get(block, set())
        mothers = eligible_mothers(load_split(split_name), rules)
        todo = sorted(m for m in mothers if m not in done)
        random.Random(f"{RANDOM_SEED}:{split_name}").shuffle(todo)
        plans[split_name] = {
            "start": start, "mothers": mothers, "todo": todo,
            "target": SPLIT_TARGET_RECORDS[split_name],
        }
        print(f"{split_name:5s}: eligible mothers={len(mothers):3d} | already done={len(done):3d} | "
              f"to crop={len(todo):3d} | id block img_{start}+ | target records≈{SPLIT_TARGET_RECORDS[split_name]}")
    print("-" * 64)

    if DRY_RUN:
        print("[dry-run] no crops written, no model loaded, nothing appended.")
        print("Set DRY_RUN=False to cut crops + run retrieval (GPU) + append.")
        return

    if not any(plans[s]["todo"] for s in TARGET_SPLITS):
        print("Nothing to do (all eligible mothers already cropped).")
        return

    # Cut one random crop per mother into the split's id block.
    rio.CROP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    prov_by_crop: dict[str, dict[str, Any]] = {}
    crops_by_split: dict[str, list[str]] = {s: [] for s in TARGET_SPLITS}
    for split_name in TARGET_SPLITS:
        img_idx = plans[split_name]["start"]
        mothers = plans[split_name]["mothers"]
        for mother_id in plans[split_name]["todo"]:
            mother_path = rio.IMAGE_DIR / mother_id
            if not mother_path.is_file():
                print(f"  WARN missing mother image, skipped: {mother_id}")
                continue
            with Image.open(mother_path) as raw:
                im = ImageOps.exif_transpose(raw).convert("RGB")
                width, height = im.size
                rng = random.Random(f"{RANDOM_SEED}:{mother_id}")
                for _ in range(N_CROPS_PER_IMAGE):
                    box = random_box(width, height, rng)
                    crop_id = f"img_{img_idx}.jpg"
                    img_idx += 1
                    im.crop(box).save(rio.CROP_IMAGE_DIR / crop_id, format="JPEG",
                                      quality=JPEG_QUALITY, optimize=True)
                    prov_by_crop[crop_id] = {
                        "base_image_id": mother_id,
                        "crop_bbox": list(box),
                        "augmentation": "random_crop",
                        "image_weak_label": mothers[mother_id],
                        "image_split": split_name,
                    }
                    crops_by_split[split_name].append(crop_id)

    all_new_crops = [cid for s in TARGET_SPLITS for cid in crops_by_split[s]]
    print(f"crops cut: {len(all_new_crops)}  ({', '.join(f'{s}={len(crops_by_split[s])}' for s in TARGET_SPLITS)})")

    # ---- Bind each crop to its top-2 rules by fresh retrieval on the crop ----
    pool = RG.visual_pool(rules)
    options = RG.H.category_options(pool)
    sig = RG.siglip_scores(all_new_crops, pool)

    runtime = None
    if not SKIP_VLM:
        if RG.config.LOCAL_QWEN_MODEL_NAME != "Qwen3.5-9B":
            print(f"WARN: RCES_LOCAL_QWEN_MODEL={RG.config.LOCAL_QWEN_MODEL_NAME!r}, expected Qwen3.5-9B")
        runtime = RG.Qwen35OptionRuntime(
            RG.config.LOCAL_QWEN35_DIR, dtype=RG.config.LOCAL_QWEN_DTYPE,
            device_map=RG.config.LOCAL_QWEN_DEVICE_MAP,
        )
    prompts = (
        RG.H.load_prompt(RG.config.RETRIEVAL_CATEGORY_PROMPT_PATH),
        RG.H.load_prompt(RG.config.RETRIEVAL_RULES_PROMPT_PATH),
    )

    routed: dict[str, tuple[str, list[tuple[str, float]]]] = {}
    for i, cid in enumerate(all_new_crops, 1):
        method, chosen = RG.route_crop(cid, sig[cid], rules, pool, runtime, prompts, options)
        routed[cid] = (method, chosen)
        print(f"[{i}/{len(all_new_crops)}] {cid} ({prov_by_crop[cid]['image_split']}) -> "
              f"{method:16s} {[rid for rid, _ in chosen]}")

    # ---- Emit records: rank0 (target) for every crop first, then rank1 (related),
    # per split, until that split's cap. Mirrors the train emission order. ----
    pair_idx = next_pair
    new_rows: list[dict[str, Any]] = []
    for split_name in TARGET_SPLITS:
        crops = crops_by_split[split_name]
        cap = plans[split_name]["target"]
        emitted = 0
        for rank in range(RG.TOP_K_PER_CROP):
            role = "target" if rank == 0 else "related"
            for cid in crops:
                if emitted >= cap:
                    break
                method, chosen = routed[cid]
                if rank >= len(chosen):
                    continue
                rule_id, score = chosen[rank]
                target_category = rules[chosen[0][0]]["major_category"]
                new_rows.append(make_record(
                    pair_idx, cid, split_name, rules[rule_id], role, method, score,
                    prov_by_crop[cid], target_category,
                ))
                pair_idx += 1
                emitted += 1
            if emitted >= cap:
                break
        print(f"{split_name}: emitted {emitted} records (cap {cap})")

    rio.write_jsonl(rio.CROP_AUG_PATH, existing + new_rows)
    print(f"\nappended {len(new_rows)} rows -> {rio.CROP_AUG_PATH.relative_to(rio.PROJECT_ROOT)} "
          f"(total {len(existing) + len(new_rows)})")
    print("Validate: uv run python tmp/validate_rces_pairs.py data/pairs/rces_crop_augmentation.jsonl")
    print("Annotate: uv run python Script/annotate_web.py 3")


if __name__ == "__main__":
    main()
