"""Joint checkpoint-resolvability head used only by the experimental arm."""

from __future__ import annotations

import torch
from torch import nn

import config


class JointRCESHead(nn.Module):
    """Predict every checkpoint jointly for one image-rule pair.

    Checkpoint text tokens first retrieve checkpoint-specific visual context.
    A small Transformer encoder then lets those tokens exchange information
    before producing one resolvability logit per checkpoint.
    """

    def __init__(
        self,
        img_dim: int = config.IMG_FEATURE_DIM,
        txt_dim: int = config.TXT_FEATURE_DIM,
        hidden: int = config.HIDDEN_DIM,
        dropout: float = config.DROPOUT,
        relation_layers: int = 1,
        relation_heads: int = 4,
        max_checkpoints: int = 32,
    ) -> None:
        super().__init__()
        if hidden % relation_heads:
            raise ValueError("hidden must be divisible by relation_heads")
        self.img_proj = nn.Linear(img_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)
        self.critical_proj = nn.Linear(1, hidden, bias=False)
        self.position = nn.Embedding(max_checkpoints, hidden)
        self.cross_norm = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=relation_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.relation_encoder = nn.TransformerEncoder(
            layer, num_layers=relation_layers, enable_nested_tensor=False
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.scale = hidden**-0.5
        self.max_checkpoints = max_checkpoints
        self.relation_layers = relation_layers
        self.relation_heads = relation_heads

    def forward(
        self,
        patches: torch.Tensor,
        checkpoint_text: torch.Tensor,
        checkpoint_mask: torch.Tensor,
        critical_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits shaped ``[batch, checkpoints]``; padded logits are zero."""
        batch_size, checkpoint_count, _ = checkpoint_text.shape
        if checkpoint_count > self.max_checkpoints:
            raise ValueError(
                f"checkpoint_count={checkpoint_count} exceeds max_checkpoints={self.max_checkpoints}"
            )
        positions = torch.arange(checkpoint_count, device=checkpoint_text.device)
        queries = (
            self.txt_proj(checkpoint_text)
            + self.critical_proj(critical_mask.unsqueeze(-1).to(checkpoint_text.dtype))
            + self.position(positions).unsqueeze(0)
        )
        image_tokens = self.img_proj(patches)
        attention_logits = torch.einsum("bkh,bnh->bkn", queries, image_tokens) * self.scale
        visual_attention = torch.softmax(attention_logits, dim=-1)
        visual_context = torch.einsum("bkn,bnh->bkh", visual_attention, image_tokens)
        tokens = self.cross_norm(queries + visual_context)
        tokens = self.relation_encoder(tokens, src_key_padding_mask=~checkpoint_mask)
        logits = self.output(tokens).squeeze(-1)
        return logits.masked_fill(~checkpoint_mask, 0.0)

    @torch.no_grad()
    def resolvable_prob(
        self,
        patches: torch.Tensor,
        checkpoint_text: torch.Tensor,
        checkpoint_mask: torch.Tensor,
        critical_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward(patches, checkpoint_text, checkpoint_mask, critical_mask)
        ).masked_fill(~checkpoint_mask, 0.0)
