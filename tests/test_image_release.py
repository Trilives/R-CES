"""Fixed-crop geometry and acquisition integrity, without network access."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "data"))
from acquire_images import acquire_one, sha256  # noqa: E402
from generate_fixed_crops import fixed_crop_specs, generate_one, mapped_box  # noqa: E402


class ImageReleaseTest(unittest.TestCase):
    def test_nonuniform_release_scaling_rounds_outward(self):
        self.assertEqual(mapped_box([3, 5, 89, 73], (100, 80), (51, 39)), (1, 2, 46, 36))
        with self.assertRaises(ValueError):
            mapped_box([0, 0, 101, 80], (100, 80), (51, 39))

    def test_exif_orientation_precedes_cropping_and_metadata_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, output = Path(directory) / "parent.jpg", Path(directory) / "crop.jpg"
            image = Image.new("RGB", (40, 20), "red")
            for x in range(20, 40):
                for y in range(20):
                    image.putpixel((x, y), (0, 0, 255))
            exif = Image.Exif()
            exif[274] = 6  # 90 degrees clockwise: blue becomes the bottom half.
            image.save(parent, exif=exif)
            row = {"release_sha256": sha256(parent), "original_width": "40", "original_height": "80",
                   "release_width": "20", "release_height": "40"}
            generated = generate_one(parent, output, row, [0, 40, 40, 80])
            self.assertEqual(generated["release_bbox"], [0, 20, 20, 40])
            with Image.open(output) as result:
                self.assertEqual(result.size, (20, 20))
                self.assertFalse(result.getexif())
                red, _, blue = result.getpixel((10, 10))
                self.assertGreater(blue, red + 100)
            row["release_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                generate_one(parent, output, row, [0, 40, 40, 80])

    def test_existing_original_is_never_overwritten_by_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "img.jpg"
            path.write_bytes(b"original experimental input")
            with self.assertRaises(ValueError):
                acquire_one({"image_id": path.name, "release_sha256": "0" * 64}, "https://example.com", path.parent)
            self.assertEqual(path.read_bytes(), b"original experimental input")

    def test_shipped_crop_annotations_have_one_fixed_box_per_image(self):
        specs = fixed_crop_specs(ROOT / "data" / "pairs" / "rces_crop_augmentation.jsonl")
        self.assertEqual(len(specs), 230)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflicting.jsonl"
            path.write_text("\n".join(json.dumps({"image_id": "crop.jpg", "base_image_id": "parent.jpg",
                                                   "crop_bbox": box}) for box in ([0, 0, 4, 4], [1, 1, 4, 4])))
            with self.assertRaises(ValueError):
                fixed_crop_specs(path)


if __name__ == "__main__":
    unittest.main()
