"""Train the R-CES head on cached SigLIP 2 features (Readme step 7).

Trains ONLY the light head; SigLIP 2 stays frozen (features are precomputed by
encode_siglip2.py). Supervision is per-critical-checkpoint resolvable with BCE;
under the no-checkpoint-supervision ablation it degrades to one pair-level
target. Train/val are image-disjoint (splits.py). Saves the head + the feature
dims + the resolved ablation flags so evaluate.py reconstructs the exact model.

  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/rces/train.py
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
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
import determinism  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces.model import RCESHead  # noqa: E402


def _device() -> torch.device:
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(image_ids: set[str], *, shuffle: bool) -> tuple[DataLoader, list[ds.PairRecord], list[ds.Example]]:
    records = ds.build_pair_records(image_ids)
    examples = ds.build_examples(records)
    if not examples:
        raise RuntimeError(
            "No training examples. Either no gold is locked yet for this split, or "
            "features are not encoded. Annotate / run encode_siglip2.py first."
        )
    data = ds.RCESDataset(examples, records)
    loader = DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=shuffle, drop_last=False)
    return loader, records, examples


def infer_dims(loader: DataLoader) -> tuple[int, int]:
    batch = next(iter(loader))
    return int(batch["patches"].shape[-1]), int(batch["text"].shape[-1])


def run_epoch(head: RCESHead, loader: DataLoader, device: torch.device, optimizer: Any | None) -> float:
    training = optimizer is not None
    head.train(training)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total = 0.0
    count = 0
    for batch in loader:
        patches = batch["patches"].to(device)
        text = batch["text"].to(device)
        is_visual = batch["is_visual"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(training):
            logit = head(patches, text, is_visual)
            loss = loss_fn(logit, target)
            if training:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), config.GRAD_CLIP)
                optimizer.step()
        total += float(loss.detach()) * len(target)
        count += len(target)
    return total / max(1, count)


def main() -> None:
    determinism.set_determinism(config.SEED, deterministic=config.DETERMINISTIC)
    device = _device()
    split = splits.load_split()
    train_ids = set(split["train"])
    val_ids = set(split["val"])

    train_loader, _, train_examples = make_loader(train_ids, shuffle=True)
    img_dim, txt_dim = infer_dims(train_loader)
    use_is_visual = config.USE_IS_VISUAL_FEATURE and not config.ABLATE_NO_IS_VISUAL
    head = RCESHead(img_dim=img_dim, txt_dim=txt_dim, use_is_visual=use_is_visual).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    has_val = bool(val_ids)
    val_loader = make_loader(val_ids, shuffle=False)[0] if has_val else None

    print(
        f"device={device} | train examples={len(train_examples)} | dims img={img_dim} txt={txt_dim} | "
        f"is_visual={use_is_visual} | ablations: ckpt={config.ABLATE_NO_CHECKPOINT_SUPERVISION} "
        f"is_visual_off={config.ABLATE_NO_IS_VISUAL} "
        f"text_off={config.ABLATE_NO_TEXT_CONDITIONING} aggregation={config.AGGREGATION_MODE}"
    )
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = run_epoch(head, train_loader, device, optimizer)
        msg = f"epoch {epoch:3d}/{config.EPOCHS} | train_bce {train_loss:.4f}"
        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(head, val_loader, device, None)
            msg += f" | val_bce {val_loss:.4f}"
        if epoch % 5 == 0 or epoch == config.EPOCHS:
            print(msg)

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "img_dim": img_dim,
            "txt_dim": txt_dim,
            "use_is_visual": use_is_visual,
            "input_mode": head.input_mode,
            "ablations": {
                "no_checkpoint_supervision": config.ABLATE_NO_CHECKPOINT_SUPERVISION,
                "no_is_visual": config.ABLATE_NO_IS_VISUAL,
                "no_text_conditioning": config.ABLATE_NO_TEXT_CONDITIONING,
                "aggregation_mode": config.AGGREGATION_MODE,
            },
        },
        config.HEAD_CKPT,
    )
    print(f"saved head -> {config.HEAD_CKPT.relative_to(config.RCES_ROOT)}")
    print(json.dumps({"train_examples": len(train_examples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
