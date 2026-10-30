"""Download or verify the fixed model snapshots in configs/models.lock.json.

RCES_MODELS is a comma-separated list of manifest names (default: the two
encoders). RCES_MODEL_ACTION is plan (default), download or verify. Gated models
require accepting the provider's terms and authenticating with Hugging Face.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(model: dict, directory: Path) -> None:
    for entry in model["files"]:
        path = directory / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or wrong-sized model file: {model['name']}/{entry['path']}")
        algorithm, expected = ("sha256", entry["sha256"]) if "sha256" in entry else ("sha1", entry["git_sha1"])
        if file_digest(path, algorithm) != expected:
            raise ValueError(f"Model checksum mismatch: {model['name']}/{entry['path']}")
    if not model["files"]:
        record = directory / "rces_snapshot.json"
        if not record.is_file() or json.loads(record.read_text()).get("revision") != model["revision"]:
            raise ValueError(f"No pinned download receipt for {model['name']}")


def main() -> None:
    manifest = json.loads((ROOT / "configs/models.lock.json").read_text())
    models = {entry["name"]: entry for entry in manifest["models"]}
    selected = os.environ.get("RCES_MODELS", "siglip2-base-patch16-224,dinov2-large").split(",")
    action = os.environ.get("RCES_MODEL_ACTION", "plan")
    if action not in {"plan", "download", "verify"} or any(name not in models for name in selected):
        raise ValueError("Use a manifest model name and RCES_MODEL_ACTION=plan, download or verify")
    for name in selected:
        model = models[name]
        directory = ROOT / "models" / name
        print(f"{model['repository_id']} @ {model['revision']} -> models/{name}/", flush=True)
        if action == "plan":
            continue
        if action == "download":
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model["repository_id"], revision=model["revision"], local_dir=directory,
                              allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt", "*.py", "*.tiktoken"])
            (directory / "rces_snapshot.json").write_text(json.dumps({
                "repository_id": model["repository_id"], "revision": model["revision"]}, indent=2) + "\n")
        verify(model, directory)
        print(f"Verified {len(model['files'])} recorded experimental files for {name}", flush=True)


if __name__ == "__main__":
    main()
