"""Generate random-crop augmentation candidates for the R-CES training set.

Crop-augmentation line. Diagnosis showed (a) too few negative
checkpoint samples and (b) the head degenerating into a text base-rate lookup
(text-only ~= full, blackout barely drops the score). Random crops of TRAIN-split
mother images attack both: the same checkpoint text is `resolvable=1` on the full
mother image but often `0` on a crop that cut the subject away, so a text-only
solution cannot hold; and crops are cheap negatives.

This script ONLY generates the crop images and the *pending* candidate pair
records — no judgment is written here. A human screens them with the full Stage-1
UI (`annotate_web.py 3`), so the "no model writes a judgment field" invariant
holds. The records are isolated in `rces_crop_augmentation.jsonl` (source=human,
augmentation=random_crop), train-only, never gold, never evaluated.

Design (see plan): mother images come ONLY from the experiment train split
(`rces_image_split.json`); each crop is bound to the mother's DETERMINISTIC rules only
(RULE_DECISION_SCOPES, default visual_screening_only, with critical checkpoints),
and the total is capped (TARGET_PAIR_RECORDS) so it does not swamp the original
train set; crops are pure random size +
position (random aspect ratio), so positives/negatives arise naturally like a
random video-frame grab. The mother's own pilot pair (already in train, mostly
resolvable=1) is the same-checkpoint counterfactual, so we do not re-label it.

Run controls are module constants (the repo avoids CLI flags). Start with
DRY_RUN=True to preview, then flip it:

    UV_CACHE_DIR=/tmp/uv-cache uv run python tmp/generate_crop_augmentation.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
import rces_io as rio  # noqa: E402
import exp_io as eio  # noqa: E402  (is_visual_rule, critical_checkpoints)

# ---- Run controls (edit these) ----------------------------------------------
DRY_RUN = True                  # True: report only, write nothing.
N_CROPS_PER_IMAGE = 1           # random crops per mother image (1 -> max mother diversity).
CROP_SCALE_RANGE = (0.30, 0.90)  # per-axis fraction of the mother edge (random aspect).
# Cap total candidate pair records so the augmentation does not swamp the original
# train set (~617 checkpoint examples). 0 = no cap. Mother images are shuffled, so
# the cap is reached across many different mothers (diversity + anti-shortcut).
TARGET_PAIR_RECORDS = 200
LIMIT_IMAGES = 0                # 0 = all eligible train mother images (debug cap).
# Only bind crops to rules with these decision scopes. Default = the most
# DETERMINISTIC scope (a single image settles the visual screening), so a crop's
# resolvable label is clean. Add "normative_decision_available" to widen.
RULE_DECISION_SCOPES = ("visual_screening_only",)
RANDOM_SEED = 20260612          # base seed; per-image seeding keeps it reproducible.
IMAGE_ID_START = 900001         # crop image ids: img_900001.jpg ... (schema-legal, no clash).
JPEG_QUALITY = 92
SPLIT_PATH = rio.PROJECT_ROOT / "experiments" / "artifacts" / "rces_image_split.json"
# -----------------------------------------------------------------------------

# Judgment fields seeded as placeholders (a human fills them in the UI).
PLACEHOLDER_FIELDS: dict[str, Any] = {
    "decidability": rio.PLACEHOLDER,
    "compliance_label": rio.PLACEHOLDER,
    "image_quality": rio.PLACEHOLDER,
    "evidence_sufficiency_label": rio.PLACEHOLDER,
    "observed_checkpoints": [],
    "missing_checkpoints": [],
    "critical_missing_checkpoints": [],
    "visual_evidence": [],
    "missing_information": [],
}
# Candidate metadata carried from the mother pair (rule-side + weak hints).
INHERITED_FIELDS = (
    "image_split", "rule_id", "rule_name", "major_category", "subcategory",
    "decision_scope", "rule_role", "image_weak_label", "weak_candidate_category",
)


def load_train_mothers() -> list[str]:
    """Train-split mother image ids (experiment pilot split), sorted."""
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    return sorted(split["train"])


def visual_pairs_by_image(rules: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Mother image_id -> its pilot pairs whose rule is in RULE_DECISION_SCOPES
    (deterministic, vision-decidable) and has critical checkpoints -- the only
    rules a crop can be screened against."""
    by_image: dict[str, list[dict[str, Any]]] = {}
    for pair in rio.load_jsonl(rio.GOLD_PATH):
        rule = rules.get(pair["rule_id"])
        if rule is None:
            continue
        if pair["decision_scope"] not in RULE_DECISION_SCOPES:
            continue
        if not eio.critical_checkpoints(rule):
            continue
        by_image.setdefault(pair["image_id"], []).append(pair)
    return by_image


def random_box(width: int, height: int, rng: random.Random) -> tuple[int, int, int, int]:
    """A random-size, random-position crop box (independent per-axis scale, so the
    aspect ratio is random too)."""
    lo, hi = CROP_SCALE_RANGE
    crop_w = max(1, round(width * rng.uniform(lo, hi)))
    crop_h = max(1, round(height * rng.uniform(lo, hi)))
    left = rng.randint(0, max(0, width - crop_w))
    top = rng.randint(0, max(0, height - crop_h))
    return left, top, left + crop_w, top + crop_h


def resume_state(rows: list[dict[str, Any]]) -> tuple[set[str], int, int]:
    """From existing crop records: mother images already done, and the next free
    crop image-id index and pair-id index (idempotent re-runs append only)."""
    done_mothers = {r["base_image_id"] for r in rows if "base_image_id" in r}
    next_img = IMAGE_ID_START
    next_pair = 1
    for r in rows:
        n = int(r["image_id"][len("img_"):].split(".")[0])
        if n >= next_img:
            next_img = n + 1
        p = int(r["pair_id"][len("rces_crop_"):])
        if p >= next_pair:
            next_pair = p + 1
    return done_mothers, next_img, next_pair


def main() -> None:
    rules = rio.load_rules()
    mothers = load_train_mothers()
    by_image = visual_pairs_by_image(rules)
    eligible = [m for m in mothers if by_image.get(m)]
    random.Random(RANDOM_SEED).shuffle(eligible)  # random mothers -> diverse coverage under the cap
    if LIMIT_IMAGES:
        eligible = eligible[:LIMIT_IMAGES]

    existing = rio.load_jsonl(rio.CROP_AUG_PATH)
    done_mothers, img_idx, pair_idx = resume_state(existing)
    todo = [m for m in eligible if m not in done_mothers]

    print("=" * 60)
    print("R-CES crop-augmentation generation" + ("  [DRY RUN]" if DRY_RUN else ""))
    print("=" * 60)
    print(f"train mother images        : {len(mothers)}")
    print(f"  eligible (has visual rule): {len(eligible)}")
    print(f"  already generated         : {len(eligible) - len(todo)}")
    print(f"  candidate mother pool     : {len(todo)}  (capped by target)")
    print(f"target pair records        : {TARGET_PAIR_RECORDS or 'unlimited'}   scopes={RULE_DECISION_SCOPES}")
    print(f"crops/image={N_CROPS_PER_IMAGE}  scale={CROP_SCALE_RANGE}  -> next img_{img_idx}, pair {pair_idx}")
    print("-" * 60)

    if not DRY_RUN and todo:
        rio.CROP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict[str, Any]] = []
    n_crops = 0
    for mother_id in todo:
        if TARGET_PAIR_RECORDS and len(existing) + len(new_rows) >= TARGET_PAIR_RECORDS:
            break
        mother_pairs = by_image[mother_id]
        mother_path = rio.IMAGE_DIR / mother_id
        if not mother_path.is_file():
            print(f"  WARN missing mother image, skipped: {mother_id}")
            continue
        with Image.open(mother_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            width, height = im.size
            rng = random.Random(f"{RANDOM_SEED}:{mother_id}")
            for _ in range(N_CROPS_PER_IMAGE):
                box = random_box(width, height, rng)
                crop_id = f"img_{img_idx}.jpg"
                img_idx += 1
                n_crops += 1
                if not DRY_RUN:
                    im.crop(box).save(rio.CROP_IMAGE_DIR / crop_id, format="JPEG", quality=JPEG_QUALITY, optimize=True)
                for mp in mother_pairs:
                    row: dict[str, Any] = {"pair_id": f"rces_crop_{pair_idx:06d}", "image_id": crop_id}
                    pair_idx += 1
                    row.update({k: mp[k] for k in INHERITED_FIELDS})
                    row.update(PLACEHOLDER_FIELDS)
                    row.update({
                        "source": rio.SOURCE_HUMAN,
                        "annotation_status": "pending",
                        "double_annotation": False,
                        "base_image_id": mother_id,
                        "crop_bbox": list(box),
                        "augmentation": "random_crop",
                    })
                    new_rows.append(row)

    print(f"crops to create            : {n_crops}")
    print(f"candidate pair records     : {len(new_rows)}")
    if DRY_RUN:
        print("\n[dry-run] no images written, no records appended. Set DRY_RUN=False to apply.")
        if new_rows:
            print("sample record:")
            print(json.dumps(new_rows[0], ensure_ascii=False))
        return

    rio.write_jsonl(rio.CROP_AUG_PATH, existing + new_rows)
    print(f"\nappended -> {rio.CROP_AUG_PATH.relative_to(rio.PROJECT_ROOT)} (total {len(existing) + len(new_rows)})")
    print("Next: encode crops, then screen with  uv run python Script/annotate_web.py 3")
    print(f"Validate: uv run python tmp/validate_rces_pairs.py {rio.CROP_AUG_PATH}")


if __name__ == "__main__":
    main()
