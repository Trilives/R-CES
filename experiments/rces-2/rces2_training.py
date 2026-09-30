"""Reusable training helpers for R-CES v2."""

from __future__ import annotations

from typing import Any

import torch

import config
from rces2_model import ContextualRCESHead


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    checkpoint_mask: torch.Tensor,
) -> torch.Tensor:
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    weights = checkpoint_mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def pair_sufficiency_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    critical_mask: torch.Tensor,
) -> torch.Tensor:
    """Readme step-9 ablation: drop dense per-checkpoint supervision and train
    only on the pair-level sufficiency label. The pair probability is the
    differentiable soft-AND (product) of the critical checkpoint probabilities;
    the pair target is the AND of the critical resolvable targets (1 iff no
    critical checkpoint is missing). Rules with no critical checkpoint contribute
    a degenerate (prob 1, target 1) pair and add ~0 loss."""
    probs = torch.sigmoid(logits)
    cm = critical_mask.to(probs.dtype)
    log_p = torch.log(probs.clamp_min(1e-6))
    pair_prob = torch.exp((cm * log_p).sum(dim=1)).clamp(1e-6, 1.0 - 1e-6)
    critical_missing = (cm * (1.0 - targets)).sum(dim=1)
    pair_target = (critical_missing == 0).to(probs.dtype)
    return torch.nn.functional.binary_cross_entropy(pair_prob, pair_target)


def run_epoch(
    head: ContextualRCESHead,
    loader: Any,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    head.train(training)
    total_loss = 0.0
    total_checkpoints = 0
    for batch in loader:
        patches = batch["patches"].to(device)
        checkpoint_text = batch["checkpoint_text"].to(device)
        checkpoint_mask = batch["checkpoint_mask"].to(device)
        targets = batch["target"].to(device)
        critical_mask = batch["critical_mask"].to(device)
        with torch.set_grad_enabled(training):
            logits = head(patches, checkpoint_text, checkpoint_mask)
            if config.ABLATE_NO_CHECKPOINT_SUPERVISION:
                loss = pair_sufficiency_loss(logits, targets, critical_mask)
            else:
                loss = masked_bce_loss(logits, targets, checkpoint_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), config.GRAD_CLIP
                )
                optimizer.step()
        checkpoint_count = int(checkpoint_mask.sum())
        total_loss += float(loss.detach()) * checkpoint_count
        total_checkpoints += checkpoint_count
    return total_loss / max(1, total_checkpoints)
