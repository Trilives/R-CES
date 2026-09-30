"""Rule-grouped inputs for R-CES v2.

One sample is one image-rule pair with every required checkpoint. The model
therefore sees the complete rule-local checkpoint set before any query attends
to image patches. Only the critical subset is used by pair-level aggregation.
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
class RuleExample:
    pair_id: str
    image_id: str
    rule_id: str
    checkpoint_keys: tuple[str, ...]
    critical_mask: tuple[bool, ...]
    targets: tuple[float, ...]


def build_rule_example(
    record: PairRecord,
    rule: dict[str, Any],
    missing_checkpoints: Sequence[str] = (),
) -> RuleExample:
    required = tuple(rule.get("required_checkpoints", ()))
    critical = frozenset(record.critical_keys)
    missing = frozenset(missing_checkpoints)
    required_set = frozenset(required)
    unknown_critical = critical - required_set
    unknown_missing = missing - required_set
    if unknown_critical or unknown_missing:
        raise ValueError(
            f"Rule {record.rule_id!r} has out-of-contract checkpoints: "
            f"critical={sorted(unknown_critical)}, missing={sorted(unknown_missing)}"
        )
    return RuleExample(
        pair_id=record.pair_id,
        image_id=record.image_id,
        rule_id=record.rule_id,
        checkpoint_keys=required,
        critical_mask=tuple(key in critical for key in required),
        targets=tuple(0.0 if key in missing else 1.0 for key in required),
    )


def build_rule_examples(records: Sequence[PairRecord]) -> list[RuleExample]:
    rules = exp_io.rio.load_rules()
    missing_by_pair = {
        pair["pair_id"]: tuple(pair.get("missing_checkpoints", ()))
        for pair in exp_io.load_locked_gold()
    }
    examples = [
        build_rule_example(
            record,
            rules.get(record.rule_id, {}),
            missing_by_pair.get(record.pair_id, ()),
        )
        for record in records
    ]
    return [example for example in examples if example.checkpoint_keys]


class RuleCheckpointDataset(Dataset):
    def __init__(self, examples: Sequence[RuleExample]):
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("RuleCheckpointDataset requires at least one example")
        self.images = load_image_features(
            {example.image_id for example in self.examples}
        )
        self.checkpoint_text = load_checkpoint_text()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        text_ids = [
            exp_io.checkpoint_text_id(example.rule_id, key)
            for key in example.checkpoint_keys
        ]
        try:
            checkpoint_text = torch.stack(
                [self.checkpoint_text[text_id] for text_id in text_ids]
            )
        except KeyError as exc:
            raise KeyError(
                f"Missing checkpoint text feature {exc.args[0]!r}; "
                "rerun experiments/data/encode_siglip2.py"
            ) from exc
        if config.ABLATE_NO_TEXT_CONDITIONING:
            # Readme step-9 ablation: zero the rule text so the head is image-only.
            checkpoint_text = torch.zeros_like(checkpoint_text)
        return {
            "patches": self.images[example.image_id],
            "checkpoint_text": checkpoint_text,
            "critical_mask": torch.tensor(
                example.critical_mask, dtype=torch.bool
            ),
            "target": torch.tensor(example.targets, dtype=torch.float32),
            "pair_id": example.pair_id,
            "checkpoint_keys": example.checkpoint_keys,
        }


def collate_rule_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty R-CES v2 batch")
    lengths = torch.tensor(
        [item["checkpoint_text"].shape[0] for item in items]
    )
    max_checkpoints = int(lengths.max())
    checkpoint_mask = (
        torch.arange(max_checkpoints).unsqueeze(0) < lengths.unsqueeze(1)
    )
    return {
        "patches": torch.stack([item["patches"] for item in items]),
        "checkpoint_text": pad_sequence(
            [item["checkpoint_text"] for item in items], batch_first=True
        ),
        "checkpoint_mask": checkpoint_mask,
        "critical_mask": pad_sequence(
            [item["critical_mask"] for item in items], batch_first=True
        ),
        "target": pad_sequence(
            [item["target"] for item in items], batch_first=True
        ),
        "pair_id": [item["pair_id"] for item in items],
        "checkpoint_keys": [item["checkpoint_keys"] for item in items],
    }
