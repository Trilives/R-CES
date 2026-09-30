"""Mechanical aggregation of per-checkpoint resolvability into pair sufficiency.

The network predicts per-checkpoint resolvability, and visual sufficiency is
computed by a fixed AND/min over the rule's critical checkpoints. External
evidence is applied separately so it cannot erase the visual head's score.

    visual_sufficiency(x, r) = MIN over critical k of P(resolvable(k,x,r))

  * MIN is the soft AND: one unresolved critical checkpoint sinks the pair.
  * a checkpoint-less rule has an empty MIN and visual sufficiency 1.0.
  * apply_external_gate() converts the visual score into final answerability.

The abstention threshold tau is tuned on val by risk-coverage (evaluate.py); here
we only produce the continuous sufficiency score per pair.
"""

from __future__ import annotations

from math import prod
from typing import Mapping

import config

from data.dataset import PairRecord


def aggregate_pair_score(
    record: PairRecord,
    resolvable_probs: Mapping[str, float],
    *,
    mode: str | None = None,
) -> float:
    """Fold per-critical-checkpoint probabilities into one sufficiency score.

    `resolvable_probs` maps checkpoint_key -> P(resolvable) for this pair. Missing
    keys default to 1.0 (treated as resolvable) so a partial prediction never
    fabricates abstention.
    """
    resolved_mode = mode or config.AGGREGATION_MODE
    values = [float(resolvable_probs.get(k, 1.0)) for k in record.critical_keys]
    if values and resolved_mode == "min":
        aggregate = min(values)
    elif values and resolved_mode == "mean":
        aggregate = sum(values) / len(values)
    elif values and resolved_mode == "product":
        aggregate = prod(values)
    elif values:
        raise ValueError(f"Unknown aggregation mode: {resolved_mode}")
    else:
        aggregate = 1.0
    return aggregate


def apply_external_gate(record: PairRecord, visual_score: float) -> float:
    """Apply the rule-side external-evidence gate after visual scoring."""
    if record.external_evidence_required:
        return config.EXTERNAL_GATE_SCORE
    return float(visual_score)


def aggregate_pair_score_from_pairlevel(record: PairRecord, pair_prob: float) -> float:
    """No-checkpoint-supervision ablation: the head emitted a single pair-level
    resolvability, so visual aggregation degenerates to that probability."""
    return float(pair_prob)
