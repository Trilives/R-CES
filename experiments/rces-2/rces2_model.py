"""Text-context-first R-CES v2 head over frozen image and text features."""

from __future__ import annotations

import torch
from torch import nn


class ContextualRCESHead(nn.Module):
    """Contextualize a rule's checkpoints, then query image patches independently.

    The checkpoint Transformer runs before image cross-attention. A rule token
    derived from the masked mean of all checkpoint embeddings gives each
    checkpoint access to a rule-level summary as well as its sibling texts.
    Critical membership is intentionally absent from the model input.
    """

    def __init__(
        self,
        img_dim: int,
        txt_dim: int,
        hidden: int = 256,
        dropout: float = 0.1,
        context_heads: int = 4,
        image_heads: int = 4,
        context_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden % context_heads:
            raise ValueError("hidden must be divisible by context_heads")
        if hidden % image_heads:
            raise ValueError("hidden must be divisible by image_heads")
        if context_layers != 1:
            raise ValueError("R-CES v2 requires exactly one context layer")

        self.text_projection = nn.Linear(txt_dim, hidden)
        self.image_projection = nn.Linear(img_dim, hidden)
        self.rule_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.rule_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.checkpoint_type = nn.Parameter(torch.zeros(1, 1, hidden))

        context_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=context_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.image_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=image_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_norm = nn.LayerNorm(hidden)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.hidden = hidden
        self.context_heads = context_heads
        self.image_heads = image_heads
        self.context_layers = context_layers
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.rule_token, std=0.02)
        nn.init.normal_(self.rule_type, std=0.02)
        nn.init.normal_(self.checkpoint_type, std=0.02)

    def contextual_queries(
        self,
        checkpoint_text: torch.Tensor,
        checkpoint_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return contextual checkpoint queries shaped ``[B, K, hidden]``."""
        checkpoint_tokens = (
            self.text_projection(checkpoint_text) + self.checkpoint_type
        )
        weights = checkpoint_mask.unsqueeze(-1).to(checkpoint_tokens.dtype)
        rule_summary = (checkpoint_tokens * weights).sum(dim=1, keepdim=True)
        rule_summary = rule_summary / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        rule_token = rule_summary + self.rule_token + self.rule_type
        tokens = torch.cat([rule_token, checkpoint_tokens], dim=1)
        rule_mask = torch.ones(
            checkpoint_mask.shape[0],
            1,
            dtype=torch.bool,
            device=checkpoint_mask.device,
        )
        token_mask = torch.cat([rule_mask, checkpoint_mask], dim=1)
        contextualized = self.context_encoder(
            tokens,
            src_key_padding_mask=~token_mask,
        )
        return contextualized[:, 1:]

    def forward(
        self,
        patches: torch.Tensor,
        checkpoint_text: torch.Tensor,
        checkpoint_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one logit per checkpoint; padded positions are exactly zero."""
        queries = self.contextual_queries(checkpoint_text, checkpoint_mask)
        image_tokens = self.image_projection(patches)
        visual_context, _ = self.image_attention(
            query=queries,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )
        fused = self.image_norm(queries + visual_context)
        logits = self.output(fused).squeeze(-1)
        return logits.masked_fill(~checkpoint_mask, 0.0)

    @torch.no_grad()
    def resolvable_prob(
        self,
        patches: torch.Tensor,
        checkpoint_text: torch.Tensor,
        checkpoint_mask: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(
            self.forward(patches, checkpoint_text, checkpoint_mask)
        )
        return probabilities.masked_fill(~checkpoint_mask, 0.0)
