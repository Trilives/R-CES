# R-CES v2

This directory is an isolated second architecture. It reuses the frozen image
patch cache, frozen SigLIP 2 checkpoint-text cache, human labels, image split,
external-evidence gate, critical-checkpoint MIN aggregation, and validation
risk-coverage threshold policy from the current R-CES experiment.

## Architecture

For one image-rule pair, let the rule have all required checkpoint text
embeddings `T = {t_1, ..., t_K}`:

1. Project every checkpoint text into the trainable hidden space.
2. Build a rule token from the masked mean of all checkpoint tokens plus a
   learned rule-token embedding.
3. Run exactly one Transformer encoder layer over
   `[rule_token, checkpoint_1, ..., checkpoint_K]`.
4. Use each contextualized checkpoint output as an independent query over the
   frozen image patch tokens.
5. Emit one resolvability logit per checkpoint.
6. Select the rule's critical checkpoint outputs and compute their fixed MIN.
7. Tune and apply `tau` exactly as in the existing evaluator.

The model never receives `critical_mask`. Critical membership only controls the
mechanical pair-level MIN, so the checkpoint predictor cannot use the critical
designation as a shortcut.

The checkpoint set has no positional embedding. Reordering a rule's checkpoints
therefore only reorders the corresponding outputs; it does not change their
meaning.

## Commands

From the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run python experiments/rces-2/train.py

UV_CACHE_DIR=/tmp/uv-cache \
uv run python experiments/rces-2/evaluate.py
```

For the explicit 224 control and Qwen3.5-9B Task-2 arm:

```bash
RCES_VISION_ENCODER=siglip2_224 \
UV_CACHE_DIR=/tmp/uv-cache \
uv run python experiments/data/encode_siglip2.py

RCES_VISION_ENCODER=siglip2_224 \
UV_CACHE_DIR=/tmp/uv-cache \
uv run python experiments/rces-2/train.py

RCES_VISION_ENCODER=siglip2_224 \
RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B \
UV_CACHE_DIR=/tmp/uv-cache \
uv run python experiments/rces-2/evaluate.py
```

Repeat with `RCES_VISION_ENCODER=siglip2_512` for the default-resolution arm.
The compliance table is shared across both arms. Qwen images are capped by the
`max_1080p_v1` policy; SigLIP 2 still receives the fixed profile resolution.

Artifacts are written under:

```text
experiments/artifacts/rces-2/<vision_encoder>/
```
