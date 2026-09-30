# R-CES — supplementary code and data (anonymized)

Anonymized supplement for the submission introducing **R-CES** (Rule-conditioned
Checkpoint Evidence Sufficiency Scoring): a checkpoint-level evidence-sufficiency
benchmark for rule-conditioned visual compliance, and a light decoupled head that
predicts *P(evidence sufficient to decide rule r from image x)* rather than
*P(answer correct)*.

The included code, schemas, and compact annotation files are self-contained and
run against each other without external dependencies. No author-identifying
information is included; large binary
assets (images, model weights, feature caches) are not bundled -- see
`DATA_AVAILABILITY.md`. Reviewer-facing documentation and prompts are in English;
model-output baselines are preserved verbatim as experimental records.

## Layout

```
src/                 core library (paths/config, IO, metrics, label vocab)
  config.py          all run-controls (module constants; env-var overrides)
  rces_io.py         canonical label vocab, JSONL/rule IO, image encoding hooks
  exp_io.py          experiment data loading (gold + crop augmentation)
  metrics.py         AUROC / AURC / ECE / image-clustered bootstrap CIs
experiments/         offline experiments (human gold only)
  rces/              v1 head (checkpoint-resolvability) — train/evaluate
  rces-2/            v2 head (text-context-first) — train/evaluate/ablation/multiseed
  ablation/          v1 ablation + multi-seed drivers
  baselines/         answer-side confidence (MaxProb, entropy, margin, temperature
                     scaling, Kamath calibrator), self-evaluation (P(True),
                     verbalized), local-VLM driver, Task-2 cascade + 7-VLM decoupling
  retrieval/         SigLIP2 vs hierarchical-VLM rule retrieval comparison
  data/              feature encoding (SigLIP2 / DINOv2) + split construction
  determinism.py     deterministic-training helper (math SDPA, seeded)
Script/              annotation pipeline (LAN web UI, GPT description, model baseline)
Prompt/              English prompts (annotation, description, baselines, self-eval)
tests/               syntax/interface/boundary checks (not a pytest suite)
tmp/                 reproduction helpers (pair generators, validator)
data/
  pairs/             gold, isolated model baseline, double-annotation, crop aug,
                     23-rule benchmark subset, unified schema, field docs
  rule/              English rule library (54 rules) + schema + field docs
  splits/            image-disjoint split lists
  enhance/           held-out test-expansion pairs
codebook.md          authoritative Stage-1 annotation manual
```

## Environment

- Python 3.13, dependencies managed by `uv`. Run everything as `uv run python ...`.
- The repository convention is **no CLI flags**: edit the module constants at the
  top of `src/config.py` (and the per-script constants), or set the documented
  environment variables.

## Key environment variables

| Variable | Meaning |
|---|---|
| `RCES_VISION_ENCODER` | `siglip2_512` (default) \| `siglip2_224` \| `dinov2` |
| `RCES_INCLUDE_CROP_AUG` | `1` adds the human-screened crop-augmentation pairs |
| `RCES_DETERMINISTIC` | `1` (default) forces deterministic training (math SDPA) |
| `RCES_SEED` / `RCES_N_SEEDS` | base seed / number of seeds for multi-seed drivers |
| `RCES_LOCAL_QWEN_MODEL` | downstream compliance VLM for Task-2 (model dir name) |
| `RCES_REMOTE_VLM_URL` / `RCES_REMOTE_VLM_ID` | OpenAI-compatible (vLLM) endpoint + served model id for an off-box compliance VLM |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | required for deterministic cuBLAS |

## Reproduction sketch (after providing images + weights, see DATA_AVAILABILITY.md)

```bash
ENV="RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 UV_CACHE_DIR=/tmp/uv-cache"

# 1. encode frozen image features (writes to cache/, gitignored)
env $ENV uv run python experiments/data/encode_siglip2.py
# 2. train + evaluate the R-CES head (v1)
env $ENV uv run python experiments/rces/train.py
env $ENV uv run python experiments/rces/evaluate.py
# 3. multi-seed ablations (v1 and v2)
env $ENV RCES_N_SEEDS=10 uv run python experiments/ablation/multiseed.py
env $ENV RCES_N_SEEDS=10 uv run python experiments/rces-2/multiseed.py
# 4. downstream compliance predictions + self-evaluation gates, per VLM
env $ENV RCES_LOCAL_QWEN_MODEL=InternVL3_5-8B-HF uv run python experiments/baselines/local_qwen35.py
env $ENV RCES_LOCAL_QWEN_MODEL=InternVL3_5-8B-HF uv run python experiments/baselines/self_eval.py
# 5. Task-2 selective-compliance cascade (all gates, per downstream VLM)
env $ENV RCES_LOCAL_QWEN_MODEL=InternVL3_5-8B-HF uv run python experiments/baselines/cascade.py
```

The Task-2 study sweeps seven open-weight compliance VLMs (Gemma-4 E2B/E4B,
Qwen3.5-4B/9B, Qwen3.6-35B-A3B, Qwen3-VL-8B, InternVL3.5-8B); `RCES_LOCAL_QWEN_MODEL`
selects one. A remote OpenAI-compatible (vLLM) endpoint is supported via
`RCES_REMOTE_VLM_URL` / `RCES_REMOTE_VLM_ID` for models served off-box.

## Data integrity

`uv run python tmp/validate_rces_pairs.py` checks rule-id existence,
observed/missing ⊆ a rule's required checkpoints (and disjoint), the
critical-missing relation, schema consistency, and a model→gold leak guard. It
exits non-zero on any error. The shipped data passes with 0 errors / 0 warnings.

## Ground-truth protocol (annotation isolation)

No model ever writes a judgment field. The annotation order is **human locks
judgment → model adds description → human reviews**, so model narration can never
anchor the gold labels; `codebook.md` is the authoritative manual. Gold (`source=human`) and the isolated model baseline
(`source=model_full`) share one record shape (`rces_final_annotations.schema.json`)
but live in two physical files so the baseline can never contaminate gold.
