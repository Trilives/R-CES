"""Extract & cache R-CES's own frozen SigLIP 2 features (Readme step 7 input).

Image patches are stored under cache/experiments/siglip2_512/ (gitignored),
while encoder-independent checkpoint text stays under siglip2/:

  image_patches/<image_id>.pt   dense patch tokens [N_patch, D_img] per image
  checkpoint_text.pt            {"<rule_id>::<key>": [D_txt]} checkpoint questions

The head conditions image patches on checkpoint text, so we keep the dense
vision tokens (vision_model.last_hidden_state), NOT the pooled image embedding.
SigLIP 2 is frozen: features are computed once here and never backprop'd through.
Resumable: images already cached are skipped.

  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/data/encode_siglip2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

import config  # noqa: E402
import exp_io  # noqa: E402


def _device() -> torch.device:
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_encoder(device: torch.device) -> tuple[Any, Any]:
    # Same loading convention as the existing SigLIP 2 user
    # tmp/generate_rces_pilot_pairs.py (local files only, AutoModel +
    # AutoProcessor). The deliberate difference: pair generation pools via
    # get_image_features for retrieval similarity, whereas the R-CES head needs
    # the DENSE patch tokens (vision_model.last_hidden_state) for text-conditioned
    # attention -- see encode_image_patches below.
    model = AutoModel.from_pretrained(config.SIGLIP2_DIR, local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(config.SIGLIP2_DIR, local_files_only=True)
    configure_image_processor(processor)
    return model, processor


def configure_image_processor(processor: Any) -> Any:
    """Apply the active R-CES SigLIP 2 image resolution."""
    processor.image_processor.size = {
        "height": config.VISION_IMAGE_SIZE,
        "width": config.VISION_IMAGE_SIZE,
    }
    return processor


def image_model_kwargs(model: Any) -> dict[str, bool]:
    native_size = int(model.vision_model.config.image_size)
    return {"interpolate_pos_encoding": config.VISION_IMAGE_SIZE != native_size}


@torch.no_grad()
def encode_image_patches(model: Any, processor: Any, image_path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
    vision = model.vision_model(
        pixel_values=inputs["pixel_values"],
        **image_model_kwargs(model),
    )
    # last_hidden_state: [1, N_patch, D_img] -> drop batch, move to cpu float32.
    return vision.last_hidden_state.squeeze(0).float().cpu()


@torch.no_grad()
def encode_texts(model: Any, processor: Any, texts: list[str], device: torch.device) -> torch.Tensor:
    inputs = processor(
        text=texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
    ).to(device)
    feats = model.get_text_features(**inputs)  # [B, D_txt], projected joint space.
    # transformers 5.x SiglipModel.get_text_features returns a
    # BaseModelOutputWithPooling whose pooler_output is the projected embedding;
    # older versions return the tensor directly. Accept both.
    if not isinstance(feats, torch.Tensor):
        feats = feats.pooler_output
    return feats.float().cpu()


def encode_all_images(model: Any, processor: Any, device: torch.device) -> dict[str, Any]:
    config.IMG_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    # Gold pilot images plus, when the crop-augmentation arm is on, the crop
    # images (resolved from the separate images_crops/ dir below).
    image_ids = exp_io.all_pair_image_ids() + exp_io.crop_image_ids()
    d_img = n_patch = 0
    done = skipped = 0
    for index, image_id in enumerate(image_ids, 1):
        out_path = config.IMG_FEATURE_DIR / f"{image_id}.pt"
        if out_path.exists():
            skipped += 1
            continue
        image_path = exp_io.rio.IMAGE_DIR / image_id
        if not image_path.is_file():
            image_path = exp_io.rio.CROP_IMAGE_DIR / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        patches = encode_image_patches(model, processor, image_path, device)
        n_patch, d_img = patches.shape
        torch.save(patches, out_path)
        done += 1
        if index % 25 == 0 or index == len(image_ids):
            print(f"[images {index}/{len(image_ids)}] encoded {done}, skipped {skipped}")
    if image_ids and not d_img:
        sample = torch.load(config.IMG_FEATURE_DIR / f"{image_ids[0]}.pt", map_location="cpu")
        n_patch, d_img = sample.shape
    return {"n_images": len(image_ids), "encoded": done, "skipped": skipped, "n_patch": n_patch, "img_dim": d_img}


def encode_checkpoint_texts(model: Any, processor: Any, device: torch.device) -> dict[str, Any]:
    """Encode every required-checkpoint question across the rules the pairs use
    (a superset of the critical ones, harmless and reusable)."""
    rules = exp_io.rio.load_rules()
    used_rule_ids = sorted({r["rule_id"] for r in exp_io.rio.load_jsonl(exp_io.rio.GOLD_PATH)})
    ids: list[str] = []
    texts: list[str] = []
    for rule_id in used_rule_ids:
        rule = rules.get(rule_id, {})
        required, _, _ = exp_io.rio.checkpoints_for(rule)
        for key in required:
            ids.append(exp_io.checkpoint_text_id(rule_id, key))
            texts.append(exp_io.checkpoint_question(rule, key))
    if not texts:
        raise RuntimeError("No checkpoint texts to encode (no rules referenced by pairs).")
    feats = encode_texts(model, processor, texts, device)
    table = {cid: feats[i].clone() for i, cid in enumerate(ids)}
    config.CKPT_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, config.CKPT_TEXT_PATH)
    return {"n_checkpoint_texts": len(ids), "txt_dim": int(feats.shape[1])}


def main() -> None:
    if not config.VISION_ENCODER.startswith("siglip2_"):
        raise SystemExit(
            f"config.VISION_ENCODER is {config.VISION_ENCODER!r}; "
            "select siglip2_224 or siglip2_512"
        )
    device = _device()
    print(
        f"device={device}, encoder={config.SIGLIP2_DIR.name}, "
        f"profile={config.VISION_ENCODER}, image_size={config.VISION_IMAGE_SIZE}"
    )
    model, processor = load_encoder(device)

    manifest: dict[str, Any] = {
        "encoder": config.SIGLIP2_DIR.name,
        "profile": config.VISION_ENCODER,
        "image_size": config.VISION_IMAGE_SIZE,
        "patch_size": 16,
        "interpolate_pos_encoding": config.VISION_IMAGE_SIZE != config.SIGLIP2_NATIVE_IMAGE_SIZE,
    }
    manifest.update(encode_all_images(model, processor, device))
    manifest.update(encode_checkpoint_texts(model, processor, device))

    config.FEATURE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"wrote features under {config.FEATURE_DIR.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
