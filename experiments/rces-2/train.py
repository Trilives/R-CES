"""Train the isolated text-context-first R-CES v2 head."""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
RCES_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(RCES_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
import determinism  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces2_dataset import (  # noqa: E402
    RuleCheckpointDataset,
    build_rule_examples,
    collate_rule_batch,
)
from rces2_model import ContextualRCESHead  # noqa: E402
from rces2_settings import (  # noqa: E402
    ARCHITECTURE,
    BATCH_SIZE,
    CONTEXT_HEADS,
    CONTEXT_LAYERS,
    HEAD_CKPT,
    HIDDEN_DIM,
    IMAGE_HEADS,
)
from rces2_training import run_epoch  # noqa: E402


def device() -> torch.device:
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    determinism.set_determinism(seed, deterministic=config.DETERMINISTIC)


def make_loader(
    records: list[ds.PairRecord],
    *,
    shuffle: bool,
) -> tuple[DataLoader, int]:
    examples = build_rule_examples(records)
    if not examples:
        raise RuntimeError(
            "No rule-grouped examples. Lock gold and encode features first."
        )
    generator = torch.Generator().manual_seed(config.SEED)
    loader = DataLoader(
        RuleCheckpointDataset(examples),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_rule_batch,
        generator=generator,
    )
    return loader, len(examples)


def main() -> None:
    seed_everything(config.SEED)
    active_device = device()
    split = splits.load_split()
    train_records = ds.build_pair_records(set(split["train"]))
    val_records = ds.build_pair_records(set(split["val"]))
    train_loader, train_pairs = make_loader(train_records, shuffle=True)
    val_loader = make_loader(val_records, shuffle=False)[0] if val_records else None

    sample = next(iter(train_loader))
    img_dim = int(sample["patches"].shape[-1])
    txt_dim = int(sample["checkpoint_text"].shape[-1])
    head = ContextualRCESHead(
        img_dim=img_dim,
        txt_dim=txt_dim,
        hidden=HIDDEN_DIM,
        dropout=config.DROPOUT,
        context_heads=CONTEXT_HEADS,
        image_heads=IMAGE_HEADS,
        context_layers=CONTEXT_LAYERS,
    ).to(active_device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    print(
        f"device={active_device} | train_pairs={train_pairs} | "
        f"img_dim={img_dim} txt_dim={txt_dim} | architecture={ARCHITECTURE}"
    )
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = run_epoch(head, train_loader, active_device, optimizer)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_bce": train_loss,
        }
        message = (
            f"epoch {epoch:3d}/{config.EPOCHS} | train_bce {train_loss:.4f}"
        )
        if val_loader is not None:
            val_loss = run_epoch(head, val_loader, active_device, None)
            row["val_bce"] = val_loss
            message += f" | val_bce {val_loss:.4f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(head.state_dict())
        else:
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == config.EPOCHS:
            print(message)

    HEAD_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "state_dict": best_state,
            "img_dim": img_dim,
            "txt_dim": txt_dim,
            "hidden": HIDDEN_DIM,
            "dropout": config.DROPOUT,
            "context_heads": CONTEXT_HEADS,
            "image_heads": IMAGE_HEADS,
            "context_layers": CONTEXT_LAYERS,
            "vision_encoder": config.VISION_ENCODER,
            "aggregation": "critical_min",
            "threshold_policy": "validation_risk_coverage",
            "best_epoch": best_epoch,
            "history": history,
        },
        HEAD_CKPT,
    )
    print(f"saved v2 head -> {HEAD_CKPT.relative_to(config.RCES_ROOT)}")
    print(
        json.dumps(
            {"train_pairs": train_pairs, "best_epoch": best_epoch},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
