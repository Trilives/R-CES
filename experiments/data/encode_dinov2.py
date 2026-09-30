"""Cache R-CES's frozen DINOv2 image patch tokens (image-encoder ablation).

Companion to `encode_siglip2.py`, selected by `config.VISION_ENCODER == "dinov2"`.
DINOv2 is vision-only, so this script writes ONLY the per-image patch cache under
`cache/experiments/dinov2/image_patches/`; the checkpoint TEXT conditioning stays
SigLIP 2 (DINOv2 has no text tower), so the shared SigLIP 2 text cache is built
here via SigLIP 2 if it is missing. Holding the text encoder fixed makes
VISION_ENCODER a clean image-backbone ablation: DINOv2 patch tokens vs SigLIP 2
patch tokens, same head, same text.

The head conditions image patches on checkpoint text, so we keep the dense patch
tokens (`last_hidden_state` with the leading CLS token dropped), NOT the pooled
CLS embedding -- the same "patches, not CLS" choice the SigLIP 2 cache makes.
DINOv2 is frozen: features are computed once here and never backprop'd through.
Resumable: images already cached are skipped.

  # set VISION_ENCODER = "dinov2" in src/config.py first
  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/data/encode_dinov2.py
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
from transformers import AutoImageProcessor, AutoModel  # noqa: E402

import config  # noqa: E402
import exp_io  # noqa: E402
from data import encode_siglip2  # noqa: E402  (reuse the SigLIP 2 text encoder)


def _device() -> torch.device:
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_encoder(device: torch.device) -> tuple[Any, Any]:
    if not config.DINOV2_DIR.is_dir() or not any(config.DINOV2_DIR.iterdir()):
        raise FileNotFoundError(
            f"Local DINOv2 weights not found at {config.DINOV2_DIR}. Download e.g. "
            f"facebook/dinov2-large into that directory (local_files_only)."
        )
    model = AutoModel.from_pretrained(config.DINOV2_DIR, local_files_only=True).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(config.DINOV2_DIR, local_files_only=True)
    # Feed DINOv2 its native high (dense-task) resolution, not the 224 default --
    # construction-safety detail is lost at 224. Resize+crop to a square SIZE that
    # is a multiple of the patch; the forward pass interpolates pos-embeddings.
    size = config.DINOV2_IMAGE_SIZE
    processor.size = {"shortest_edge": size}
    processor.crop_size = {"height": size, "width": size}
    return model, processor


@torch.no_grad()
def encode_image_patches(model: Any, processor: Any, image_path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
    out = model(pixel_values=inputs["pixel_values"], interpolate_pos_encoding=True)
    # last_hidden_state: [1, 1 + N_patch, D]; index 0 is the CLS token. Drop it so
    # the cache holds dense PATCH tokens only, matching the SigLIP 2 patch cache.
    patches = out.last_hidden_state[:, 1:, :]
    return patches.squeeze(0).float().cpu()


def encode_all_images(model: Any, processor: Any, device: torch.device) -> dict[str, Any]:
    config.IMG_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    # Gold pilot images plus, when the crop-augmentation arm is on, the crop
    # images (resolved from images_crops/ below).
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
    return {"n_images": len(image_ids), "encoded": done, "skipped": skipped, "n_patch": n_patch, "img_dim": d_img}


def ensure_checkpoint_texts(device: torch.device) -> dict[str, Any]:
    """The text tower is always SigLIP 2; build its checkpoint-text cache once if
    this DINOv2 run is the first to need it."""
    if config.CKPT_TEXT_PATH.exists():
        return {"checkpoint_text": "reused", "path": str(config.CKPT_TEXT_PATH)}
    print(f"checkpoint-text cache missing; building it via SigLIP 2 ({config.SIGLIP2_DIR.name})")
    siglip_model, siglip_processor = encode_siglip2.load_encoder(device)
    stats = encode_siglip2.encode_checkpoint_texts(siglip_model, siglip_processor, device)
    stats["checkpoint_text"] = "built_via_siglip2"
    return stats


def main() -> None:
    if config.VISION_ENCODER != "dinov2":
        raise SystemExit(
            f"config.VISION_ENCODER is {config.VISION_ENCODER!r}; set it to 'dinov2' before "
            f"running this script (the cache dir is derived from it)."
        )
    device = _device()
    print(f"device={device}, image_encoder={config.DINOV2_DIR.name}, text_encoder={config.SIGLIP2_DIR.name}")
    model, processor = load_encoder(device)

    manifest: dict[str, Any] = {
        "image_encoder": config.DINOV2_DIR.name,
        "text_encoder": config.SIGLIP2_DIR.name,
        "image_size": config.VISION_IMAGE_SIZE,
    }
    manifest.update(encode_all_images(model, processor, device))
    manifest.update(ensure_checkpoint_texts(device))

    if manifest.get("img_dim") and manifest["img_dim"] != config.IMG_FEATURE_DIM:
        print(
            f"WARNING: cached DINOv2 img_dim={manifest['img_dim']} != config.IMG_FEATURE_DIM="
            f"{config.IMG_FEATURE_DIM}; update _ENCODER_IMG_DIM['dinov2'] so the head matches."
        )

    config.FEATURE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"wrote features under {config.FEATURE_DIR.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
