"""Build the image-disjoint R-CES train/val/test split (Readme step 7).

Two modes, selected by config.USE_DATASET_SPLITS:

  False (pilot, default): split over the FIXED universe of images referenced by
    the pair file (currently 250 development_subset images), deterministic on
    SPLIT_SEED, SPLIT_RATIOS-wise. Stable as annotation progresses. Use while
    only the development_subset is annotated -- those images are all train-side
    in the dataset split, so the dataset split alone yields no val/test.
  True (official run): read the dataset-level lists under data/splits/ instead
    (train <- development_subset, val <- val_images, test <- realistic_test);
    flip to this once annotation reaches the val / realistic-test splits.

Either way train/val/test are image-disjoint: no image (hence no pair) crosses
splits, the only leakage guard that matters for a per-image visual head. (The
dataset primary splits are disjoint by construction, and development_subset is
sampled from train, so it cannot overlap val / realistic_test.)

  uv run python experiments/data/splits.py   # build + write artifact
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import config  # noqa: E402
import exp_io  # noqa: E402


def _read_image_list(name: str) -> list[str]:
    path = config.DATASET_SPLIT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split list missing: {path}")
    return sorted({line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()})


def build_pilot_split() -> dict[str, Any]:
    """Pilot mode: deterministic SPLIT_RATIOS re-split of the pair file's images."""
    image_ids = exp_io.all_pair_image_ids()
    rng = random.Random(config.SPLIT_SEED)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    r_train, r_val, _ = config.SPLIT_RATIOS
    n_train = int(round(n * r_train))
    n_val = int(round(n * r_val))
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val :])
    split = {
        "mode": "pilot",
        "seed": config.SPLIT_SEED,
        "ratios": list(config.SPLIT_RATIOS),
        "n_images": n,
        "train": train,
        "val": val,
        "test": test,
    }
    return _append_crop_augmentation(split)


def _append_crop_augmentation(split: dict[str, Any]) -> dict[str, Any]:
    """Append crop-augmentation image_ids to their carried split bucket without
    disturbing the original (deterministic / dataset-level) assignment. A crop's
    `image_split` is its mother image's split, so this stays image-disjoint."""
    crops = exp_io.crop_image_ids_by_split()
    if not crops:
        return split
    counts: dict[str, int] = {}
    for name in ("train", "val", "test"):
        present = set(split[name])
        extra = [i for i in crops.get(name, []) if i not in present]
        split[name] = sorted(split[name] + extra)
        counts[name] = len(extra)
    split["crop_augmentation"] = {
        "included": True,
        "added_images": counts,
        "n_crop_images": sum(counts.values()),
    }
    split["n_images"] = len(split["train"]) + len(split["val"]) + len(split["test"])
    return split


def build_dataset_split() -> dict[str, Any]:
    """Official mode: read the dataset-level lists under data/splits/."""
    files = config.DATASET_SPLIT_FILES
    train = _read_image_list(files["train"])
    val = _read_image_list(files["val"])
    test = _read_image_list(files["test"])
    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        raise ValueError(f"Dataset splits are not image-disjoint; overlap e.g. {sorted(overlap)[:5]}")
    split = {
        "mode": "dataset",
        "source_files": {k: files[k] for k in ("train", "val", "test")},
        "n_images": len(train) + len(val) + len(test),
        "train": train,
        "val": val,
        "test": test,
    }
    return _append_crop_augmentation(split)


def build_split() -> dict[str, Any]:
    return build_dataset_split() if config.USE_DATASET_SPLITS else build_pilot_split()


def load_split() -> dict[str, list[str]]:
    """Load the split artifact, (re)building it when absent or when the recorded
    mode no longer matches config.USE_DATASET_SPLITS."""
    want_mode = "dataset" if config.USE_DATASET_SPLITS else "pilot"
    if not config.SPLIT_PATH.exists():
        write_split(build_split())
    data = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))
    if data.get("mode") != want_mode:
        data = build_split()
        write_split(data)
    return {k: data[k] for k in ("train", "val", "test")}


def split_of_image() -> dict[str, str]:
    """image_id -> split name lookup."""
    splits = load_split()
    return {img: name for name, ids in splits.items() for img in ids}


def write_split(split: dict[str, Any]) -> None:
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.SPLIT_PATH.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    split = build_split()
    write_split(split)
    origin = f"seed={split['seed']}" if split["mode"] == "pilot" else "data/splits/"
    print(
        f"[{split['mode']}] split ({origin}) over {split['n_images']} images -> "
        f"train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}"
    )
    print(f"wrote {Path(config.SPLIT_PATH).relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
