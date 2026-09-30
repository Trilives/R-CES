"""The R-CES head: a light, text-conditioned head over frozen SigLIP 2 patches.

Per Readme step 7 this is the ONLY trainable form (a light head on frozen
features, the origin of "lightweight"). Its atomic task is

    resolvable(k, x, r) = f(dense image patches, checkpoint-question text)

i.e. it predicts whether checkpoint k can be *resolved* in image x -- visibility,
NOT the checkpoint's value and NOT the compliance answer. Conditioning is built
in: the checkpoint text is the attention query that pools the image patches.

Architecture (a handful of linears; ~sub-million params):
    text  --proj-->  q  [H]
    patch --proj-->  K, V  [N, H]
    attn = softmax(q . K^T / sqrt(H));  ctx = attn . V        # text-conditioned pool
    head([ctx, q]) -> resolvable logit

The legacy `is_visual` scalar remains an optional compatibility input for old
checkpoints, but corrected E0 does not feed it. External evidence is gated after
visual scoring in aggregate.py. Diagnostic input modes can zero the image or
text channel before projection, producing explicit text-only and image-only
baselines without changing the parameter structure of existing checkpoints.
"""

from __future__ import annotations

import torch
from torch import nn

import config


class RCESHead(nn.Module):
    def __init__(
        self,
        img_dim: int = config.IMG_FEATURE_DIM,
        txt_dim: int = config.TXT_FEATURE_DIM,
        hidden: int = config.HIDDEN_DIM,
        dropout: float = config.DROPOUT,
        use_is_visual: bool = False,
        input_mode: str = "full",
    ):
        super().__init__()
        if input_mode not in {"full", "text_only", "image_only"}:
            raise ValueError(f"input_mode must be full, text_only, or image_only; got {input_mode!r}")
        self.use_is_visual = use_is_visual
        self.input_mode = input_mode
        self.img_proj = nn.Linear(img_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)
        self.scale = hidden ** -0.5
        head_in = hidden * 2 + (1 if use_is_visual else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, patches: torch.Tensor, text: torch.Tensor, is_visual: torch.Tensor) -> torch.Tensor:
        """patches [B, N, img_dim], text [B, txt_dim], is_visual [B, 1] -> logit [B, 1]."""
        if self.input_mode == "text_only":
            patches = torch.zeros_like(patches)
        elif self.input_mode == "image_only":
            text = torch.zeros_like(text)
        q = self.txt_proj(text)  # [B, H]
        kv = self.img_proj(patches)  # [B, N, H]
        attn = torch.softmax((kv @ q.unsqueeze(-1)).squeeze(-1) * self.scale, dim=-1)  # [B, N]
        ctx = (attn.unsqueeze(-1) * kv).sum(dim=1)  # [B, H], text-conditioned pool
        feats = [ctx, q]
        if self.use_is_visual:
            feats.append(is_visual)
        return self.head(torch.cat(feats, dim=-1))

    @torch.no_grad()
    def resolvable_prob(self, patches: torch.Tensor, text: torch.Tensor, is_visual: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(patches, text, is_visual)).squeeze(-1)


def build_head() -> RCESHead:
    """Construct the head with the currently-resolved ablation flags."""
    use_is_visual = config.USE_IS_VISUAL_FEATURE and not config.ABLATE_NO_IS_VISUAL
    return RCESHead(use_is_visual=use_is_visual)
