"""Regenerate the existing annotated crops from released parent photographs.

Uses fixed EXIF-transposed original-parent coordinates, scaled to released
parent pixels. It never draws random crops or changes annotations. The output
preserves the annotated field of view but differs from original experiment pixels.
See DATA_AVAILABILITY.md for access terms, rounding and privacy-review limits.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_images import ROOT, sha256  # noqa: E402

IMAGE_DIR = ROOT / "data" / "images"
OUTPUT_DIR = ROOT / "data" / "images_crops"
MANIFEST_DIR = ROOT / "data" / "upstream"


def mapped_box(box: list[int], original_size: tuple[int, int], release_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Outward rounding retains the complete annotated field of view."""
    ow, oh = original_size
    rw, rh = release_size
    left, top, right, bottom = box
    if min(ow, oh, rw, rh) <= 0 or not (0 <= left < right <= ow and 0 <= top < bottom <= oh):
        raise ValueError("Invalid original dimensions or crop bounds")
    return (
        max(0, math.floor(left * rw / ow)),
        max(0, math.floor(top * rh / oh)),
        min(rw, math.ceil(right * rw / ow)),
        min(rh, math.ceil(bottom * rh / oh)),
    )


def fixed_crop_specs(annotation_path: Path) -> dict[str, tuple[str, list[int]]]:
    specs: dict[str, tuple[str, list[int]]] = {}
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        spec = (row["base_image_id"], row["crop_bbox"])
        if row["image_id"] in specs and specs[row["image_id"]] != spec:
            raise ValueError(f"Conflicting parent or box for {row['image_id']}")
        specs[row["image_id"]] = spec
    return specs


def generate_one(parent_path: Path, output: Path, row: dict[str, str], box: list[int]) -> dict:
    if sha256(parent_path) != row["release_sha256"]:
        raise ValueError(f"Parent checksum differs from the fixed upstream release: {parent_path.name}")
    original_size = (int(row["original_width"]), int(row["original_height"]))
    release_size = (int(row["release_width"]), int(row["release_height"]))
    with Image.open(parent_path) as source:
        parent = ImageOps.exif_transpose(source)
        if parent.size != release_size:
            raise ValueError(f"Unexpected EXIF-transposed release dimensions: {parent_path.name}")
        bounds = mapped_box(box, original_size, parent.size)
        crop = parent.crop(bounds).convert("RGB")
        # Strip metadata inherited from the parent; only JPEG pixels are written.
        crop.info.clear()
        crop.save(output, format="JPEG", quality=95, subsampling=0, exif=b"")
    return {"image_id": output.name, "parent_image_id": parent_path.name,
            "original_bbox": box, "release_bbox": list(bounds),
            "width": crop.width, "height": crop.height, "sha256": sha256(output)}


def main() -> None:
    with (MANIFEST_DIR / "images.csv").open(newline="", encoding="utf-8") as stream:
        images = {row["image_id"]: row for row in csv.DictReader(stream)}
    specs = fixed_crop_specs(ROOT / "data" / "pairs" / "rces_crop_augmentation.jsonl")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Keep original experimental crops safe if this is run in a research checkout.
    existing = [image_id for image_id in specs if (OUTPUT_DIR / image_id).exists()]
    if existing:
        raise ValueError("Crop output already exists; use an empty data/images_crops directory")
    results = []
    for image_id, (parent_id, box) in sorted(specs.items()):
        if Path(image_id).name != image_id or Path(parent_id).name != parent_id:
            raise ValueError("Image IDs must be filenames")
        results.append(generate_one(IMAGE_DIR / parent_id, OUTPUT_DIR / image_id, images[parent_id], box))
    (OUTPUT_DIR / "generated_manifest.json").write_text(
        json.dumps({"transform": "EXIF transpose, scale x/y, round outward, JPEG quality 95, 4:4:4",
                    "exact_original_pixels": False, "crops": results}, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(results)} fixed annotated crops; original experiment pixels are not reproduced.")


if __name__ == "__main__":
    main()
