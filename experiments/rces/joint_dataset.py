"""Pair-level dataset for the isolated joint-checkpoint experiment.

Each sample contains one image-rule pair and every required checkpoint in the
rule. Every required checkpoint receives the existing annotation-derived
resolvability target (missing -> 0, observed/not-applicable -> 1). Only critical
checkpoints are used by the final sufficiency aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import config
import exp_io
from data.dataset import PairRecord, load_checkpoint_text, load_image_features


@dataclass(frozen=True)
class JointExample:
    pair_id: str
    image_id: str
    rule_id: str
    checkpoint_keys: tuple[str, ...]
    critical_mask: tuple[bool, ...]
    targets: tuple[float, ...]
    supervision_mask: tuple[bool, ...]


def build_joint_example(
    record: PairRecord,
    rule: dict[str, Any],
    missing_checkpoints: Sequence[str] = (),
) -> JointExample:
    required = tuple(rule.get("required_checkpoints", ()))
    critical = set(record.critical_keys)
    missing = set(missing_checkpoints)
    return JointExample(
        pair_id=record.pair_id,
        image_id=record.image_id,
        rule_id=record.rule_id,
        checkpoint_keys=required,
        critical_mask=tuple(key in critical for key in required),
        targets=tuple(0.0 if key in missing else 1.0 for key in required),
        supervision_mask=tuple(True for _ in required),
    )


def build_joint_examples(records: Sequence[PairRecord]) -> list[JointExample]:
    rules = exp_io.rio.load_rules()
    missing_by_pair = {
        pair["pair_id"]: tuple(pair.get("missing_checkpoints", ()))
        for pair in exp_io.load_locked_gold()
    }
    examples = [
        build_joint_example(
            record,
            rules.get(record.rule_id, {}),
            missing_by_pair.get(record.pair_id, ()),
        )
        for record in records
    ]
    return [example for example in examples if example.checkpoint_keys]


class JointCheckpointDataset(Dataset):
    def __init__(self, examples: Sequence[JointExample]):
        self.examples = list(examples)
        self.images = load_image_features({example.image_id for example in self.examples})
        self.checkpoint_text = load_checkpoint_text()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        text_ids = [exp_io.checkpoint_text_id(example.rule_id, key) for key in example.checkpoint_keys]
        try:
            text = torch.stack([self.checkpoint_text[text_id] for text_id in text_ids])
        except KeyError as exc:
            raise KeyError(f"Missing checkpoint text feature {exc.args[0]!r}; rerun encode_siglip2.py") from exc
        if config.ABLATE_NO_TEXT_CONDITIONING:
            text = torch.zeros_like(text)
        return {
            "patches": self.images[example.image_id],
            "checkpoint_text": text,
            "critical_mask": torch.tensor(example.critical_mask, dtype=torch.bool),
            "target": torch.tensor(example.targets, dtype=torch.float32),
            "supervision_mask": torch.tensor(example.supervision_mask, dtype=torch.bool),
            "pair_id": example.pair_id,
            "checkpoint_keys": example.checkpoint_keys,
        }


def collate_joint_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty joint-checkpoint batch")
    lengths = torch.tensor([item["checkpoint_text"].shape[0] for item in items])
    max_checkpoints = int(lengths.max())
    checkpoint_mask = torch.arange(max_checkpoints).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "patches": torch.stack([item["patches"] for item in items]),
        "checkpoint_text": pad_sequence(
            [item["checkpoint_text"] for item in items], batch_first=True
        ),
        "checkpoint_mask": checkpoint_mask,
        "critical_mask": pad_sequence(
            [item["critical_mask"] for item in items], batch_first=True
        ),
        "target": pad_sequence([item["target"] for item in items], batch_first=True),
        "supervision_mask": pad_sequence(
            [item["supervision_mask"] for item in items], batch_first=True
        ),
        "pair_id": [item["pair_id"] for item in items],
        "checkpoint_keys": [item["checkpoint_keys"] for item in items],
    }
