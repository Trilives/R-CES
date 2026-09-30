"""Generate R-CES pilot image-rule annotation pairs with local SigLIP 2.

This is the self-contained R-CES pair generator. It reads only R-CES assets:

  - data/rule/rule.json
  - data/pairs/benchmark_rule_subset.json
  - data/splits/development_subset_images.txt
  - data/images/
  - models/siglip2-base-patch16-224 (auto-downloaded from Hugging Face
    using HF_TOKEN in .env when the local copy is missing)

The output is the Stage-1 gold task file:

  - data/pairs/rces_pilot_pairs.jsonl
  - data/pairs/rces_pilot_pairs_stats.json

All judgment fields remain placeholders. SigLIP 2 is used only for image-rule
candidate construction via rule.visual_retrieval_text.

Run:
    UV_CACHE_DIR=/tmp/uv-cache uv run python tmp/generate_rces_pilot_pairs.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


RCES_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RCES_ROOT.parent

ENV_PATH = RCES_ROOT / ".env"
RULES_PATH = RCES_ROOT / "data" / "rule" / "rule.json"
SUBSET_PATH = RCES_ROOT / "data" / "pairs" / "benchmark_rule_subset.json"
DEV_SPLIT_PATH = RCES_ROOT / "data" / "splits" / "development_subset_images.txt"
IMAGE_DIR = RCES_ROOT / "data" / "images"
PAIRS_PATH = RCES_ROOT / "data" / "pairs" / "rces_pilot_pairs.jsonl"
STATS_PATH = RCES_ROOT / "data" / "pairs" / "rces_pilot_pairs_stats.json"
CACHE_DIR = RCES_ROOT / "tmp" / "siglip2_pair_generation"
MODEL_DIR = RCES_ROOT / "models" / "siglip2-base-patch16-224"

RANDOM_SEED = 20260528
TARGET_IMAGE_COUNT = 250
PAIR_ROLES = ("target", "related", "not_applicable", "possibly_undecidable")
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP2_IMAGE_SIZE = 512
SIGLIP2_IMAGE_BATCH_SIZE = 16
SIGLIP2_TEXT_BATCH_SIZE = 32
SIMILARITY_TOP_K = 6
TARGET_TOP_K = 1
NOT_APPLICABLE_BOTTOM_K = 8

PLACEHOLDERS = {
    "decidability": "to_be_annotated",
    "compliance_label": "to_be_annotated",
    "image_quality": "to_be_annotated",
    "evidence_sufficiency_label": "to_be_annotated",
    "observed_checkpoints": [],
    "missing_checkpoints": [],
    "critical_missing_checkpoints": [],
    "visual_evidence": [],
    "missing_information": [],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_rules() -> list[dict[str, Any]]:
    selected_ids = set(load_json(SUBSET_PATH)["selected_rule_ids"])
    rules = [rule for rule in load_json(RULES_PATH) if rule["rule_id"] in selected_ids]
    missing = selected_ids - {rule["rule_id"] for rule in rules}
    if missing:
        raise KeyError(f"Subset rule_ids missing from {RULES_PATH}: {sorted(missing)}")
    return rules


def load_development_image_ids() -> list[str]:
    return [
        line.strip()
        for line in DEV_SPLIT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_existing_image_labels() -> dict[str, str]:
    """Compatibility provenance for the current schema.

    R-CES pair construction does not use weak labels. Existing labels are copied
    only because rces_final_annotations.schema.json still requires
    image_weak_label as source provenance.
    """
    labels: dict[str, str] = {}
    for row in load_jsonl(PAIRS_PATH):
        image_id = str(row.get("image_id", ""))
        label = str(row.get("image_weak_label", ""))
        if image_id and label in {"compliant", "non_compliant"}:
            previous = labels.setdefault(image_id, label)
            if previous != label:
                raise ValueError(f"Conflicting weak labels for {image_id}: {previous} vs {label}")
    return labels


def load_previous_selected_image_ids() -> list[str]:
    if not STATS_PATH.exists():
        return []
    stats = load_json(STATS_PATH)
    selected = stats.get("selected_images", [])
    if not isinstance(selected, list):
        return []
    return [str(image_id) for image_id in selected if str(image_id).strip()]


def select_images(dev_ids: list[str], labels: dict[str, str], rng: random.Random) -> list[dict[str, str]]:
    previous = load_previous_selected_image_ids()
    if len(previous) == TARGET_IMAGE_COUNT and set(previous) <= set(dev_ids):
        selected_ids = previous
    else:
        pool = dev_ids.copy()
        rng.shuffle(pool)
        selected_ids = sorted(pool[:TARGET_IMAGE_COUNT])

    missing_labels = [image_id for image_id in selected_ids if image_id not in labels]
    if missing_labels:
        raise ValueError(
            "Selected images are missing image_weak_label provenance required by the current schema: "
            f"{missing_labels[:10]}{'...' if len(missing_labels) > 10 else ''}. "
            "The labels are not used for SigLIP2 selection; they are copied only for schema compatibility."
        )

    return [
        {"image_id": image_id, "image_weak_label": labels[image_id]}
        for image_id in sorted(selected_ids)
    ]


def rule_text(rule: dict[str, Any]) -> str:
    text = str(rule.get("visual_retrieval_text", "")).strip()
    if text:
        return text
    checkpoints = "; ".join(str(value) for value in rule.get("visual_checkpoints", {}).values())
    return (
        f"{rule['major_category']}; {rule['subcategory']}; {rule['rule_name']}. "
        f"Visual checkpoints: {checkpoints}. Decision scope: {rule.get('decision_scope', '')}."
    )


def cache_path(images: list[dict[str, str]], rules: list[dict[str, Any]]) -> Path:
    payload = {
        "siglip2_model_id": SIGLIP2_MODEL_ID,
        "siglip2_image_size": SIGLIP2_IMAGE_SIZE,
        "model_dir": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
        "images": [image["image_id"] for image in images],
        "rules": [{"rule_id": rule["rule_id"], "text": rule_text(rule)} for rule in rules],
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"similarities_{digest}.json"


def require_siglip2_model() -> Path:
    if MODEL_DIR.is_dir() and any(MODEL_DIR.iterdir()):
        return MODEL_DIR
    return download_siglip2_model()


def download_siglip2_model() -> Path:
    """Fetch SigLIP 2 from Hugging Face into MODEL_DIR using HF_TOKEN in .env."""
    from huggingface_hub import snapshot_download

    load_dotenv(ENV_PATH)
    token = os.environ.get("HF_TOKEN") or None
    print(
        f"Local SigLIP 2 model not found at {MODEL_DIR.relative_to(PROJECT_ROOT)}; "
        f"downloading {SIGLIP2_MODEL_ID} from Hugging Face"
        f"{'' if token else ' (no HF_TOKEN set, anonymous access)'}...",
        file=sys.stderr,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=SIGLIP2_MODEL_ID, local_dir=str(MODEL_DIR), token=token)
    except Exception as exc:  # noqa: BLE001 - surface any download failure as a clean exit
        raise RuntimeError(
            f"Failed to download {SIGLIP2_MODEL_ID} from Hugging Face: {exc}. "
            f"Check HF_TOKEN in {ENV_PATH.relative_to(PROJECT_ROOT)} (and HF_ENDPOINT / network access)."
        ) from exc
    if not (MODEL_DIR.is_dir() and any(MODEL_DIR.iterdir())):
        raise RuntimeError(f"Download produced no files in {MODEL_DIR}.")
    print(f"Downloaded SigLIP 2 to {MODEL_DIR.relative_to(PROJECT_ROOT)}.", file=sys.stderr)
    return MODEL_DIR


def normalize_features(tensor: Any) -> Any:
    return tensor / tensor.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def pooled_feature_tensor(output: Any) -> Any:
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def encode_text_features(model: Any, processor: Any, texts: list[str], device: Any) -> Any:
    import torch

    batches = []
    for start in range(0, len(texts), SIGLIP2_TEXT_BATCH_SIZE):
        batch = texts[start : start + SIGLIP2_TEXT_BATCH_SIZE]
        inputs = processor(text=batch, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            if not hasattr(model, "get_text_features"):
                raise RuntimeError("Selected SigLIP model does not expose get_text_features().")
            features = pooled_feature_tensor(model.get_text_features(**inputs))
        batches.append(normalize_features(features).detach().cpu())
    return torch.cat(batches, dim=0)


def encode_image_features(model: Any, processor: Any, image_paths: list[Path], device: Any) -> Any:
    import torch
    from PIL import Image, ImageOps

    batches = []
    for start in range(0, len(image_paths), SIGLIP2_IMAGE_BATCH_SIZE):
        paths = image_paths[start : start + SIGLIP2_IMAGE_BATCH_SIZE]
        images = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Image file missing for SigLIP pair generation: {path}")
            with Image.open(path) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            if not hasattr(model, "get_image_features"):
                raise RuntimeError("Selected SigLIP model does not expose get_image_features().")
            features = pooled_feature_tensor(
                model.get_image_features(**inputs, interpolate_pos_encoding=True)
            )
        batches.append(normalize_features(features).detach().cpu())
    return torch.cat(batches, dim=0)


def compute_siglip2_scores(
    selected_images: list[dict[str, str]],
    rules: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    path = cache_path(selected_images, rules)
    if path.exists():
        cached = load_json(path)
        return cached["scores"], {
            "pair_selection_method": "siglip2_similarity",
            "siglip2_model_id": cached["siglip2_model_id"],
            "siglip2_model_dir": cached["siglip2_model_dir"],
            "siglip2_image_size": cached["siglip2_image_size"],
            "similarity_cache": str(path.relative_to(PROJECT_ROOT)),
            "similarity_cache_hit": True,
        }

    model_dir = require_siglip2_model()

    import torch
    from transformers import AutoModel, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.image_processor.size = {
        "height": SIGLIP2_IMAGE_SIZE,
        "width": SIGLIP2_IMAGE_SIZE,
    }
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()

    image_paths = [IMAGE_DIR / image["image_id"] for image in selected_images]
    rule_ids = [rule["rule_id"] for rule in rules]
    texts = [rule_text(rule) for rule in rules]

    image_features = encode_image_features(model, processor, image_paths, device)
    text_features = encode_text_features(model, processor, texts, device)
    matrix = image_features @ text_features.T

    scores = {
        image["image_id"]: {
            rule_id: float(matrix[image_index, rule_index].item())
            for rule_index, rule_id in enumerate(rule_ids)
        }
        for image_index, image in enumerate(selected_images)
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        path,
        {
            "siglip2_model_id": SIGLIP2_MODEL_ID,
            "siglip2_image_size": SIGLIP2_IMAGE_SIZE,
            "siglip2_model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
            "image_ids": [image["image_id"] for image in selected_images],
            "rule_ids": rule_ids,
            "scores": scores,
        },
    )
    return scores, {
        "pair_selection_method": "siglip2_similarity",
        "siglip2_model_id": SIGLIP2_MODEL_ID,
        "siglip2_model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
        "siglip2_image_size": SIGLIP2_IMAGE_SIZE,
        "similarity_cache": str(path.relative_to(PROJECT_ROOT)),
        "similarity_cache_hit": False,
    }


def choose_by_similarity(
    candidates: list[dict[str, Any]],
    used_rule_ids: set[str],
    scores: dict[str, float],
    rng: random.Random,
    *,
    prefer_high: bool,
    usage: Counter,
    pool_size: int,
) -> dict[str, Any]:
    available = [rule for rule in candidates if rule["rule_id"] not in used_rule_ids] or candidates
    ranked = sorted(
        available,
        key=lambda rule: (scores.get(rule["rule_id"], float("-inf")), rule["rule_id"]),
        reverse=prefer_high,
    )
    pool = ranked[: max(1, min(pool_size, len(ranked)))]
    min_use = min(usage[rule["rule_id"]] for rule in pool)
    pool = [rule for rule in pool if usage[rule["rule_id"]] == min_use]
    chosen = rng.choice(pool)
    usage[chosen["rule_id"]] += 1
    return chosen


def make_pair(pair_number: int, image: dict[str, str], role: str, rule: dict[str, Any], target_rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": f"rces_pilot_{pair_number:06d}",
        "image_id": image["image_id"],
        "image_split": "development_subset",
        "rule_id": rule["rule_id"],
        "rule_name": rule["rule_name"],
        "major_category": rule["major_category"],
        "subcategory": rule["subcategory"],
        "decision_scope": rule["decision_scope"],
        "rule_role": role,
        "image_weak_label": image["image_weak_label"],
        "weak_candidate_category": target_rule["major_category"],
        **PLACEHOLDERS,
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
    external_rules = [rule for rule in rules if rule["decision_scope"] == "requires_external_evidence"]
    if not visual_rules:
        raise ValueError("Benchmark subset has no visual/normative candidate rules.")
    if not external_rules:
        raise ValueError("Benchmark subset has no requires_external_evidence rules.")

    scores, metadata = compute_siglip2_scores(selected_images, rules)
    pairs: list[dict[str, Any]] = []
    usage = {
        "target": Counter(),
        "related": Counter(),
        "not_applicable": Counter(),
        "possibly_undecidable": Counter(),
    }

    for image in selected_images:
        image_scores = scores[image["image_id"]]
        used_rule_ids: set[str] = set()

        target_rule = choose_by_similarity(
            visual_rules,
            used_rule_ids,
            image_scores,
            rng,
            prefer_high=True,
            usage=usage["target"],
            pool_size=TARGET_TOP_K,
        )
        used_rule_ids.add(target_rule["rule_id"])

        same_subcategory = [
            rule
            for rule in visual_rules
            if rule["subcategory"] == target_rule["subcategory"] and rule["rule_id"] not in used_rule_ids
        ]
        same_category = [
            rule
            for rule in visual_rules
            if rule["major_category"] == target_rule["major_category"] and rule["rule_id"] not in used_rule_ids
        ]
        related_rule = choose_by_similarity(
            same_subcategory or same_category or visual_rules,
            used_rule_ids,
            image_scores,
            rng,
            prefer_high=True,
            usage=usage["related"],
            pool_size=SIMILARITY_TOP_K,
        )
        used_rule_ids.add(related_rule["rule_id"])

        different_category = [
            rule
            for rule in visual_rules
            if rule["major_category"] != target_rule["major_category"] and rule["rule_id"] not in used_rule_ids
        ]
        not_applicable_rule = choose_by_similarity(
            different_category or visual_rules,
            used_rule_ids,
            image_scores,
            rng,
            prefer_high=False,
            usage=usage["not_applicable"],
            pool_size=NOT_APPLICABLE_BOTTOM_K,
        )
        used_rule_ids.add(not_applicable_rule["rule_id"])

        undecidable_rule = choose_by_similarity(
            external_rules,
            used_rule_ids,
            image_scores,
            rng,
            prefer_high=True,
            usage=usage["possibly_undecidable"],
            pool_size=len(external_rules),
        )
        used_rule_ids.add(undecidable_rule["rule_id"])

        for role, rule in (
            ("target", target_rule),
            ("related", related_rule),
            ("not_applicable", not_applicable_rule),
            ("possibly_undecidable", undecidable_rule),
        ):
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
        **metadata,
        "image_count": len(selected_images),
        "pair_count": len(pairs),
        "distinct_rule_count": len({pair["rule_id"] for pair in pairs}),
        "image_label_counts": {
            "compliant": image_label_counts["compliant"],
            "non_compliant": image_label_counts["non_compliant"],
        },
        "major_category_pair_counts": dict(sorted(category_counts.items())),
        "rule_role_counts": {role: role_counts[role] for role in PAIR_ROLES},
        "selected_images": [image["image_id"] for image in selected_images],
        "image_sampling_policy": "Selected images are fixed by prior stats when available, otherwise sampled from development_subset without using weak labels.",
        "weak_label_policy": "image_weak_label is copied only for schema/provenance compatibility; it is not used for image sampling or rule selection.",
        "annotation_policy": "Judgment fields are placeholders; no visual annotation is performed during pair generation.",
    }


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    rules = load_rules()
    dev_ids = load_development_image_ids()
    labels = load_existing_image_labels()
    selected_images = select_images(dev_ids, labels, rng)
    pairs, metadata = build_pairs(selected_images, rules, rng)
    stats = build_stats(selected_images, pairs, metadata)

    write_jsonl_atomic(PAIRS_PATH, pairs)
    write_json_atomic(STATS_PATH, stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote pairs: {PAIRS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote stats: {STATS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        sys.exit(str(exc))
