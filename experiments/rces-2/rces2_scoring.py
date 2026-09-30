"""Inference and fixed critical-MIN aggregation for R-CES v2."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

import config
from data.dataset import PairRecord
from rces2_model import ContextualRCESHead


def aggregate_critical(values: list[float]) -> float:
    """Pair-level aggregation over a rule's critical checkpoint probabilities.

    Default is the structural MIN (soft-AND). `config.AGGREGATION_MODE` swaps in
    `mean` / `product` for the Readme step-9 aggregation ablation; an empty
    critical set (checkpoint-less rule) scores 1.0 in every mode."""
    if not values:
        return 1.0
    mode = config.AGGREGATION_MODE
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "product":
        product = 1.0
        for value in values:
            product *= value
        return product
    return min(values)


def critical_min_score(
    record: PairRecord,
    checkpoint_probabilities: Mapping[str, float],
) -> float:
    values = [
        float(checkpoint_probabilities.get(key, 1.0))
        for key in record.critical_keys
    ]
    return aggregate_critical(values)


@torch.no_grad()
def checkpoint_probabilities(
    head: ContextualRCESHead,
    loader: Any,
    device: torch.device,
) -> dict[tuple[str, str], float]:
    head.eval()
    output: dict[tuple[str, str], float] = {}
    for batch in loader:
        probabilities = head.resolvable_prob(
            batch["patches"].to(device),
            batch["checkpoint_text"].to(device),
            batch["checkpoint_mask"].to(device),
        ).cpu()
        checkpoint_mask = batch["checkpoint_mask"]
        for row, pair_id in enumerate(batch["pair_id"]):
            for column, key in enumerate(batch["checkpoint_keys"][row]):
                if bool(checkpoint_mask[row, column]):
                    output[(pair_id, key)] = float(
                        probabilities[row, column]
                    )
    return output


def visual_pair_scores(
    records: list[PairRecord],
    probabilities: Mapping[tuple[str, str], float],
) -> dict[str, float]:
    return {
        record.pair_id: critical_min_score(
            record,
            {
                key: probabilities.get((record.pair_id, key), 1.0)
                for key in record.critical_keys
            },
        )
        for record in records
    }
