"""SigLIP 2 rule retrieval: per-image image<->rule-text similarity ranking.

The comparison baseline for vlm_retrieve.py. For every image in the gold pair
file it ranks the 23 benchmark-subset rules by cosine similarity between the
pooled SigLIP 2 image embedding and the rule's subject text (the same
visual_retrieval_text the VLM retriever shows as the subject cue, so the two
methods see the same rule description). One row per image:

  {image_id, method, model, rule_scores: {rule_id: cosine},
   ranked_rules: [rule_id ...], top1, applicable_rules: [top-K rule_id ...]}

Offline, local SigLIP 2 weights only (models/siglip2-base-patch16-224).
Run controls are constants in src/config.py.

  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/retrieval/siglip_retrieve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import config  # noqa: E402
import exp_io  # noqa: E402
import hierarchy as H  # noqa: E402
from data.encode_siglip2 import configure_image_processor, image_model_kwargs  # noqa: E402

IMAGE_BATCH_SIZE = 16
TEXT_BATCH_SIZE = 32


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.RCES_ROOT))
    except ValueError:
        return str(path)


def _normalize(tensor: Any) -> Any:
    return tensor / tensor.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def _pooled(output: Any) -> Any:
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, attr, None)
        if value is not None:
            return value
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def _encode_images(model, processor, image_paths, device):
    import torch
    from PIL import Image, ImageOps

    batches = []
    for start in range(0, len(image_paths), IMAGE_BATCH_SIZE):
        chunk = image_paths[start : start + IMAGE_BATCH_SIZE]
        images = []
        for path in chunk:
            if not path.is_file():
                raise FileNotFoundError(f"Missing image for SigLIP retrieval: {path}")
            with Image.open(path) as src:
                images.append(ImageOps.exif_transpose(src).convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            features = _pooled(model.get_image_features(**inputs, **image_model_kwargs(model)))
        batches.append(_normalize(features).detach().cpu())
    return torch.cat(batches, dim=0)


def _encode_texts(model, processor, texts, device):
    import torch

    batches = []
    for start in range(0, len(texts), TEXT_BATCH_SIZE):
        chunk = texts[start : start + TEXT_BATCH_SIZE]
        inputs = processor(text=chunk, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            features = _pooled(model.get_text_features(**inputs))
        batches.append(_normalize(features).detach().cpu())
    return torch.cat(batches, dim=0)


def main() -> None:
    rules = H.load_subset_rules(exp_io.rio.RULES_PATH, config.RETRIEVAL_SUBSET_PATH)
    rule_ids = [r["rule_id"] for r in rules]
    texts = [H.rule_subject_text(r) for r in rules]

    image_ids = exp_io.all_pair_image_ids()
    if config.RETRIEVAL_LIMIT_IMAGES > 0:
        image_ids = image_ids[: config.RETRIEVAL_LIMIT_IMAGES]
    if not image_ids:
        raise RuntimeError("No images in the gold pair file for SigLIP retrieval")
    if not config.SIGLIP2_DIR.is_dir() or not any(config.SIGLIP2_DIR.iterdir()):
        raise FileNotFoundError(
            f"Local SigLIP 2 weights not found at {display_path(config.SIGLIP2_DIR)}; "
            "run tmp/generate_rces_pilot_pairs.py once to download them."
        )

    import torch
    from transformers import AutoModel, AutoProcessor

    device = torch.device("cuda" if (config.DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    processor = AutoProcessor.from_pretrained(config.SIGLIP2_DIR, local_files_only=True)
    configure_image_processor(processor)
    model = AutoModel.from_pretrained(config.SIGLIP2_DIR, local_files_only=True).to(device).eval()

    image_paths = [exp_io.rio.IMAGE_DIR / image_id for image_id in image_ids]
    image_features = _encode_images(model, processor, image_paths, device)
    text_features = _encode_texts(model, processor, texts, device)
    matrix = image_features @ text_features.T  # [n_images, n_rules]

    rows: list[dict[str, Any]] = []
    for i, image_id in enumerate(image_ids):
        scores = {rule_ids[j]: float(matrix[i, j].item()) for j in range(len(rule_ids))}
        ranked = sorted(rule_ids, key=lambda rid: (-scores[rid], rid))
        rows.append(
            {
                "image_id": image_id,
                "method": "siglip2_similarity",
                "model": config.SIGLIP2_DIR.name,
                "image_size": config.SIGLIP2_IMAGE_SIZE,
                "rule_scores": scores,
                "ranked_rules": ranked,
                "top1": ranked[0],
                "applicable_rules": ranked[: config.RETRIEVAL_SIGLIP_TOP_K],
            }
        )

    config.RETRIEVAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    exp_io.rio.write_jsonl(config.RETRIEVAL_SIGLIP_PATH, rows, sort_keys=True)
    summary = {
        "method": "siglip2_similarity",
        "model": config.SIGLIP2_DIR.name,
        "image_size": config.SIGLIP2_IMAGE_SIZE,
        "image_count": len(rows),
        "rule_pool": len(rule_ids),
        "top_k": config.RETRIEVAL_SIGLIP_TOP_K,
        "device": str(device),
        "output_path": display_path(config.RETRIEVAL_SIGLIP_PATH),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {display_path(config.RETRIEVAL_SIGLIP_PATH)}")


if __name__ == "__main__":
    main()
