"""Acquire checksum-verified photographs from the frozen public ARGus snapshot.

Run from the repository root: uv run python experiments/data/acquire_images.py
Access terms and reproduction limits are documented in DATA_AVAILABILITY.md.
Existing files with different bytes are never overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "data" / "upstream"
DESTINATION = ROOT / "data" / "images"
WORKERS = 8


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquire_one(row: dict[str, str], base_url: str, destination: Path) -> str:
    image_id = row["image_id"]
    if Path(image_id).name != image_id:
        raise ValueError("Image ID must be a filename")
    target = destination / image_id
    if target.exists():
        if sha256(target) != row["release_sha256"]:
            raise ValueError(f"Refusing to overwrite {image_id}: existing bytes differ from the release")
        return image_id
    url = base_url.rstrip("/") + "/" + row["upstream_path"]
    for attempt in range(3):
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                with urllib.request.urlopen(url, timeout=45) as response:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            if sha256(temporary) != row["release_sha256"]:
                raise ValueError(f"Published-file checksum mismatch: {image_id}")
            os.replace(temporary, target)
            return image_id
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not acquire {image_id}")


def main() -> None:
    release = json.loads((MANIFEST_DIR / "release.json").read_text(encoding="utf-8"))
    with (MANIFEST_DIR / "images.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = [pool.submit(acquire_one, row, release["raw_base_url"], DESTINATION) for row in rows]
        for count, job in enumerate(as_completed(jobs), 1):
            job.result()
            if count % 100 == 0 or count == len(rows):
                print(f"Verified {count}/{len(rows)} upstream photographs", flush=True)


if __name__ == "__main__":
    main()
