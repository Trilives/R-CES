"""Generate R-CES test-set-expansion image-rule pairs with local SigLIP 2.

Companion to ``generate_rces_pilot_pairs.py``. It builds candidate annotation
pairs for the images picked with ``Script/select_test_images_web.py`` (recorded
by name in ``data/splits/test_expansion_selected_images.txt``) so the test set
can be expanded. The images are NOT moved -- they stay in ``data/images`` and
are referenced in place; only the new pair records land under ``data/enhance``.

Differences from the pilot generator (the rest is reused from it):

  - Image pool is the explicit selection file, not a sample of development_subset.
  - ``image_split`` is ``test_expansion`` (a dedicated schema enum value).
  - Exactly TWO pairs per image: roles ``target`` + ``related`` (the two
    highest-SigLIP2 visual/normative rules; same subcategory/category preferred
    for ``related``). No not_applicable / possibly_undecidable role here.
  - ``image_weak_label`` is baked once from the repo-root anonymized mapping
    ``data/annotations/image_mapping.csv`` (the only provenance source for these
    images; the pilot copies its labels from the same original mapping). This is
    the single generation-time read outside R-CES; the emitted file is then
    self-contained.

All judgment fields stay placeholders (a human fills them with annotate_web.py).
SigLIP 2 is used only for image-rule candidate construction.

Run:
    UV_CACHE_DIR=/tmp/uv-cache uv run python tmp/generate_enhance_pairs.py

Validate:
    uv run python tmp/validate_rces_pairs.py data/enhance/rces_enhance_pairs.jsonl
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import generate_rces_pilot_pairs as pilot  # noqa: E402  (SigLIP2 + rule machinery)

RCES_ROOT = pilot.RCES_ROOT
PROJECT_ROOT = pilot.PROJECT_ROOT  # repo root (one above R-CES)

SELECTION_PATH = RCES_ROOT / "data" / "splits" / "test_expansion_selected_images.txt"
IMAGE_MAPPING_PATH = PROJECT_ROOT / "data" / "annotations" / "image_mapping.csv"
OUTPUT_DIR = RCES_ROOT / "data" / "enhance"
PAIRS_PATH = OUTPUT_DIR / "rces_enhance_pairs.jsonl"
STATS_PATH = OUTPUT_DIR / "rces_enhance_pairs_stats.json"

IMAGE_SPLIT = "test_expansion"
PAIR_ROLES = ("target", "related")
RANDOM_SEED = 20260617
WEAK_LABELS = frozenset({"compliant", "non_compliant"})


def load_selected_image_ids() -> list[str]:
    if not SELECTION_PATH.is_file():
        raise FileNotFoundError(f"Selection file not found: {SELECTION_PATH}")
    ids: list[str] = []
    seen: set[str] = set()
    for line in SELECTION_PATH.read_text(encoding="utf-8").splitlines():
        image_id = line.strip()
        if not image_id or image_id in seen:
            continue
        if Path(image_id).name != image_id:
            raise ValueError(f"Unsafe image_id in selection file: {image_id!r}")
        ids.append(image_id)
        seen.add(image_id)
    if not ids:
        raise ValueError(f"Selection file is empty: {SELECTION_PATH}")
    return sorted(ids)


def load_weak_labels(image_ids: list[str]) -> dict[str, str]:
    """Map each selected image to its compliant/non_compliant weak label from the
    repo-root anonymized mapping. Raises if any image is uncovered."""
    if not IMAGE_MAPPING_PATH.is_file():
        raise FileNotFoundError(f"Image mapping not found: {IMAGE_MAPPING_PATH}")
    wanted = set(image_ids)
    labels: dict[str, str] = {}
    with IMAGE_MAPPING_PATH.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("anonymous_name", "")
            if name in wanted:
                label = row.get("label", "")
                if label not in WEAK_LABELS:
                    raise ValueError(f"Unexpected label {label!r} for {name} in {IMAGE_MAPPING_PATH}")
                labels[name] = label
    missing = wanted - labels.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} selected image(s) missing from {IMAGE_MAPPING_PATH}: "
            f"{sorted(missing)[:10]}"
        )
    return labels


def check_images_exist(image_ids: list[str]) -> None:
    missing = [image_id for image_id in image_ids if not (pilot.IMAGE_DIR / image_id).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected image(s) absent from {pilot.IMAGE_DIR}: {missing[:10]}"
        )


def make_pair(
    pair_number: int,
    image: dict[str, str],
    role: str,
    rule: dict[str, Any],
    target_rule: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pair_id": f"rces_enhance_{pair_number:06d}",
        "image_id": image["image_id"],
        "image_split": IMAGE_SPLIT,
        "rule_id": rule["rule_id"],
        "rule_name": rule["rule_name"],
        "major_category": rule["major_category"],
        "subcategory": rule["subcategory"],
        "decision_scope": rule["decision_scope"],
        "rule_role": role,
        "image_weak_label": image["image_weak_label"],
        "weak_candidate_category": target_rule["major_category"],
        **pilot.PLACEHOLDERS,
        "source": "human",
        "annotation_status": "pending",
        "double_annotation": False,
    }


def build_pairs(
    selected_images: list[dict[str, str]],
    rules: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    visual_rules = [rule for rule in rules if rule["decision_scope"] != "requires_external_evidence"]
    if not visual_rules:
        raise ValueError("Benchmark subset has no visual/normative candidate rules.")

    scores, metadata = pilot.compute_siglip2_scores(selected_images, rules)
    usage = {"target": Counter(), "related": Counter()}
    pairs: list[dict[str, Any]] = []

    for image in selected_images:
        image_scores = scores[image["image_id"]]
        used: set[str] = set()

        target_rule = pilot.choose_by_similarity(
            visual_rules, used, image_scores, rng,
            prefer_high=True, usage=usage["target"], pool_size=pilot.TARGET_TOP_K,
        )
        used.add(target_rule["rule_id"])

        same_subcategory = [
            r for r in visual_rules
            if r["subcategory"] == target_rule["subcategory"] and r["rule_id"] not in used
        ]
        same_category = [
            r for r in visual_rules
            if r["major_category"] == target_rule["major_category"] and r["rule_id"] not in used
        ]
        related_rule = pilot.choose_by_similarity(
            same_subcategory or same_category or [r for r in visual_rules if r["rule_id"] not in used] or visual_rules,
            used, image_scores, rng,
            prefer_high=True, usage=usage["related"], pool_size=pilot.SIMILARITY_TOP_K,
        )

        for role, rule in (("target", target_rule), ("related", related_rule)):
            pairs.append(make_pair(len(pairs) + 1, image, role, rule, target_rule))

    metadata["role_rule_usage"] = {role: dict(counter) for role, counter in usage.items()}
    return pairs, metadata


def build_stats(
    selected_images: list[dict[str, str]],
    pairs: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    image_label_counts = Counter(image["image_weak_label"] for image in selected_images)
    role_counts = Counter(pair["rule_role"] for pair in pairs)
    category_counts = Counter(pair["major_category"] for pair in pairs)
    return {
        "random_seed": RANDOM_SEED,
        "image_split": IMAGE_SPLIT,
        **metadata,
        "image_count": len(selected_images),
        "pair_count": len(pairs),
        "pairs_per_image": len(PAIR_ROLES),
        "distinct_rule_count": len({pair["rule_id"] for pair in pairs}),
        "image_label_counts": {
            "compliant": image_label_counts["compliant"],
            "non_compliant": image_label_counts["non_compliant"],
        },
        "major_category_pair_counts": dict(sorted(category_counts.items())),
        "rule_role_counts": {role: role_counts[role] for role in PAIR_ROLES},
        "selected_images": [image["image_id"] for image in selected_images],
        "image_source_policy": "Images stay in data/images and are referenced in place (not copied).",
        "weak_label_policy": "image_weak_label is baked once from the repo-root image_mapping.csv (provenance only); it is not used for rule selection.",
        "annotation_policy": "Judgment fields are placeholders; no visual annotation is performed during pair generation.",
    }


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    rules = pilot.load_rules()
    image_ids = load_selected_image_ids()
    check_images_exist(image_ids)
    labels = load_weak_labels(image_ids)
    selected_images = [
        {"image_id": image_id, "image_weak_label": labels[image_id]}
        for image_id in image_ids
    ]

    pairs, metadata = build_pairs(selected_images, rules, rng)
    stats = build_stats(selected_images, pairs, metadata)

    pilot.write_jsonl_atomic(PAIRS_PATH, pairs)
    pilot.write_json_atomic(STATS_PATH, stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote pairs: {PAIRS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote stats: {STATS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        sys.exit(str(exc))
