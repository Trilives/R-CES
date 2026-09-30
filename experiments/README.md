# R-CES offline experiments

This directory contains the offline training, evaluation, and ablation pipeline
for R-CES. It never calls a cloud model and never treats model-generated labels
or descriptions as gold.

The only supervision source is the locked human annotation in
`data/pairs/rces_pilot_pairs.jsonl`. Rows with `annotation_status` in
`judged`, `described`, or `reviewed` are usable because their judgment and
checkpoint fields were already locked by the human annotator. Optional GPT text
does not enter the feature cache, targets, training loop, or metrics.

## Boundary

```text
Script/
  annotate_web.py             human judgment -> gold
  describe_with_gpt.py        optional free-text enrichment; not an experiment input
  baseline_qwen_full.py       Qwen3.7-Plus annotation comparison; never gold
  compare_qwen_to_gold.py     offline disagreement/confidence analysis

experiments/            offline only
  reads: human gold + rules + local images + local model weights
  writes: feature caches + heads + predictions + reports
```

Qwen3.7-Plus output exists only to compare a frontier vision model with human
annotation and study disagreement / overconfidence. It is not used for R-CES
training, threshold tuning, evaluation labels, or ablation inputs.

The locally stored `models/Qwen3.5-4B` is the fixed downstream compliance
VLM for Task 2 and the source of MaxProb probabilities. It never supplies R-CES
training labels: human annotation remains the only supervision and gold.

## Layout

```text
src/
  config.py                   experiment paths and hyperparameters
  exp_io.py                   locked-human-gold loading and target derivation
  metrics.py                  risk-coverage, AUROC bootstrap, calibration, threshold tuning

experiments/
  data/
    splits.py                 image-disjoint pilot or official dataset split
    encode_siglip2.py         local frozen SigLIP 2 512 cache (patches + checkpoint text)
    encode_dinov2.py          local frozen DINOv2 image-patch cache (image-encoder ablation)
    dataset.py                checkpoint-level train / pair-level eval views
  local_vlm/
    runtime.py                supported local Qwen runtimes; restricted-label logits
    predictor.py              complete structured analysis + aligned label/confidence
    compare_models.py         paired 8B/9B accuracy and class-level comparison
  baselines/
    local_qwen35.py           build one isolated local prediction table
    direct.py                 Qwen answers every pair
    maxprob.py                gate by local Qwen restricted-label probability
    kamath.py                 validation-trained correctness calibrator
    recoverr.py               fixed reliability gate (ReCoVERR-reduced)
  rces/
    model.py                  lightweight checkpoint-resolvability head
    aggregate.py              visual MIN aggregation + separate external gate
    train.py                  train on human checkpoint supervision
    evaluate.py               Task 1 + local-Qwen selective-compliance Task 2
    diagnose.py               CI, polarity, learning curves, blackout, channel baselines
    compare_siglip_resolutions.py historical same-snapshot 224/512 result analysis
  ablation/
    run.py                    required core ablation matrix
  rces-2/                     isolated second architecture (see rces-2/README.md)
    rces2_settings.py         isolated run controls + per-encoder artifact paths
    rces2_dataset.py          rule-grouped inputs (all required checkpoints per pair)
    rces2_model.py            text-context-first head: checkpoint Transformer then image cross-attention
    rces2_scoring.py          inference + fixed critical-checkpoint MIN aggregation
    rces2_training.py         masked checkpoint BCE training helpers
    train.py / evaluate.py    train v2 / evaluate with unchanged tau + MIN policy
  retrieval/
    hierarchy.py              category->rule hierarchy + prompt rendering (pure)
    siglip_retrieve.py        SigLIP 2 image<->rule-text similarity ranking
    vlm_retrieve.py           hierarchical local-VLM retriever (Stage A + Stage B)
    compare.py                score both methods vs human-gold applicability
  artifacts/                  split, heads, predictions, reports (gitignored)
```

## Rule retrieval: SigLIP vs hierarchical VLM

Which rules is a given image a candidate for? The committed pairs were built with
SigLIP 2 image<->rule-text similarity, but similarity sometimes retrieves rules
whose subject is not in the image (a `target` rule the human then marks
`not_applicable`). This experiment compares that against a hierarchical local-VLM
retriever and scores both against human gold.

- **SigLIP** (`siglip_retrieve.py`): pooled SigLIP 2 image embedding vs each
  rule's `visual_retrieval_text`; per image it ranks the 23 benchmark-subset
  rules and keeps the top-K as the applicable set.
- **Hierarchical VLM** (`vlm_retrieve.py`): coarse-to-fine, so the model is never
  shown all rules at once and is **not** queried once per rule.
  - **Stage A** — one A/B/C/D call picks the visible hazard category (the 4
    categories map to A/B/C/D); categories above `RETRIEVAL_CATEGORY_THRESHOLD`
    (plus the argmax) are kept.
  - **Stage B** — for each selected category, **one generation call** lists that
    category's rules (numbered, grouped by sub-type) and returns the top-K most
    relevant rule numbers in a single answer.
  - Prompts are templates under `Prompt/` (`retrieval_category_prompt.md`,
    `retrieval_rules_prompt.md`); the runtime is the same local Qwen3.5-4B
    (`local_vlm/runtime.py`, with `generate_text` for Stage B). Resumable: rows
    flush after each image.
- **Comparison** (`compare.py`): gold applicability is `compliance_label !=
  not_applicable`. On the annotated pairs it reports each method's AUROC / average
  precision at separating applicable from not-applicable rules, the applicable-set
  precision/recall/F1, the SigLIP top-1 (`target`) not-applicable rate, and
  method-vs-method agreement (Jaccard, SigLIP-top1-in-VLM-set).

`config.RETRIEVAL_METHOD ∈ {siglip2, vlm_hierarchical}` selects which retriever is
authoritative for the downstream pipeline (`config.active_retrieval_path()`);
both artifacts are still produced so the comparison runs regardless. All other
controls (`RETRIEVAL_*`) are in `src/config.py`.

## Method

For each critical checkpoint `k`, the head predicts whether that checkpoint is
resolvable from image `x` under rule `r`:

```text
resolvable(k, x, r) = can checkpoint k be resolved in this image?
```

Human checkpoint annotation supplies the dense target:

```text
resolvable(k) = 0 if k is in missing_checkpoints, else 1
```

Visual pair sufficiency is mechanical rather than learned, and external evidence
is applied afterward as a separate rule-side gate:

```text
visual_sufficiency(x, r) = MIN_k P(resolvable(k, x, r))
final_answerability(x, r) = external_gate(r, visual_sufficiency(x, r))
```

The validation split tunes abstention thresholds; the test split reports
checkpoint and pair AUROC/AUPRC/F1, risk-coverage, AURC, coverage at target risk,
ECE, effective reliability, and Critical Missing Checkpoint Recall (CMCR).

### Experimental joint-checkpoint arm

`rces/compare_joint_checkpoints.py` tests whether checkpoints from one rule
should be scored jointly. It leaves corrected E0 unchanged and trains two
isolated, same-snapshot arms:

- independent control: one image plus one critical-checkpoint text per example;
- joint candidate: one image plus all ordered required-checkpoint texts, a known
  critical/context role embedding, checkpoint-to-image attention, one
  checkpoint self-attention layer, and one output per required checkpoint.

Every required output uses `missing -> 0, observed/not-applicable -> 1`. Final
pair sufficiency still uses only critical outputs and the unchanged `MIN`
aggregation. The recorded comparison used the then-current 224px cache, seed 0,
40 epochs, AdamW,
and best validation checkpoint AUROC selection. The candidate has 558,273
parameters versus 526,337 for the control.

On the June 10 snapshot, checkpoint AUROC increased from `0.6163` to `0.6739`
and visual-pair AUROC from `0.5672` to `0.6470`. Paired image-cluster bootstrap
intervals still include zero (`[-0.0960, 0.1998]` and `[-0.0769, 0.2467]`). The
result is suggestive but does not justify replacing the default head; test has
only 18 negative checkpoint examples and 11 negative visual pairs.

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run python experiments/rces/compare_joint_checkpoints.py
```

Artifacts are isolated under `experiments/artifacts/joint_checkpoints/`.

### Second architecture (`rces-2/`)

`rces-2/` is a fully isolated second architecture rather than an arm bolted onto
the default head. It reuses everything fixed about the experiment — the frozen
image-patch and SigLIP 2 checkpoint-text caches, human labels, image split,
external-evidence gate, critical-checkpoint MIN, and the validation
risk-coverage threshold — and only changes how a rule's checkpoints are scored.

The head is **text-context-first**: it builds a rule token from the masked mean
of a rule's required-checkpoint embeddings, runs one Transformer encoder layer
over `[rule_token, checkpoint_1, ..., checkpoint_K]`, then uses each
contextualized checkpoint as an independent query over the frozen image patches
and emits one resolvability logit per checkpoint. The model never sees
`critical_mask` (critical membership only drives the mechanical pair-level MIN)
and uses no checkpoint positional embedding, so reordering a rule's checkpoints
only reorders outputs. Artifacts are isolated under
`experiments/artifacts/rces-2/<vision_encoder>/`. See `rces-2/README.md` for the
full architecture description and the 224/512 + Qwen3.5-9B run commands
(summarized in run order step 8 below).

## Required comparison methods

All methods reuse the same active compliance-prediction file; only the gate
changes. The configured local Qwen compliance VLM is the EXPERIMENT's predictor
(method + ablation), kept apart from the frontier baseline
`data/pairs/rces_qwen_full_baseline.jsonl` (qwen3.7-plus), which the experiments
never write. `config.LOCAL_USE_STRUCTURED` selects the reasoning-mode arm and the
file each consumer reads: `compliance_predictions_structured_v2.jsonl`
(complete checkpoint analysis followed by aligned A/B/C label/confidence) or
`compliance_predictions_direct.jsonl` (direct A/B/C/D label, the ablation arm).
The two arms persist side by side so neither overwrites the other.

| Method | Current implementation |
|---|---|
| `local_qwen35_direct` | The configured local Qwen model answers every pair. |
| `maxprob` | Restricted softmax probability over the predicted A/B/C/D label. |
| `kamath` | Numpy logistic correctness calibrator trained on validation data. |
| `recoverr_reduced` | Fixed reliability threshold `1 - TARGET_RISK`; no iterative evidence gathering. |
| `R-CES` | Checkpoint resolvability and structural aggregation gate the same Qwen answers. |

`ReCoVERR-reduced` must not be described as the full ReCoVERR method in results.

## Run order

Run from the repository root with Python managed by `uv`:

```bash
# 0. Build or inspect the fixed image-disjoint split.
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/data/splits.py

# 1. Cache local frozen image patches + checkpoint text for the active backbone.
#    config.VISION_ENCODER selects which one ("siglip2_512" default |
#    "siglip2_224" resolution control | "dinov2"); the
#    cache dir is per-encoder, so both can coexist. Checkpoint TEXT is always
#    SigLIP 2 (DINOv2 has no text tower), so encode_dinov2.py builds it via SigLIP 2
#    once if missing -- the DINOv2 run is a pure image-backbone ablation.
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/data/encode_siglip2.py
# (image-encoder ablation) set VISION_ENCODER="dinov2" in config, then:
# UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/data/encode_dinov2.py

# 2. Run the configured local Qwen once; every Task-2 method shares this table.
RCES_LOCAL_QWEN_MODEL=Qwen3-VL-8B-Instruct \
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/baselines/local_qwen35.py

# Model-size comparison arm; keeps the same dynamic image resolution and writes
# to artifacts/local_qwen35_9b/ without replacing 8B outputs.
RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B \
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/baselines/local_qwen35.py
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/local_vlm/compare_models.py

# 3. Train and evaluate R-CES against human gold.
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/rces/train.py
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/rces/evaluate.py

# 4. Run required comparison methods.
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/baselines/direct.py
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/baselines/maxprob.py
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/baselines/kamath.py
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/baselines/recoverr.py

# 5. Run the core ablations.
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/ablation/run.py

# 6. Rule-retrieval comparison (independent of steps 1-5; needs local SigLIP 2
#    weights and local Qwen3.5-4B). Run both retrievers, then compare.
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/retrieval/siglip_retrieve.py
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/retrieval/vlm_retrieve.py
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/retrieval/compare.py

# 7. Statistical/channel diagnostics. The script snapshots mutable annotation
#    inputs before reading them, then writes reports and curves under
#    experiments/artifacts/diagnostics/.
UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/rces/diagnose.py

# 8. Optional same-code resolution control:
# RCES_VISION_ENCODER=siglip2_224 UV_CACHE_DIR=/tmp/uv-cache \
#   uv run python experiments/data/encode_siglip2.py
# RCES_VISION_ENCODER=siglip2_224 UV_CACHE_DIR=/tmp/uv-cache \
#   uv run python experiments/rces-2/train.py
# RCES_VISION_ENCODER=siglip2_224 RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B \
#   UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/rces-2/evaluate.py
```

All run controls are module constants in `src/config.py`; experiment
scripts intentionally have no CLI flags.

`RCES_VISION_ENCODER` is the exception used to select an existing encoder
profile without editing source. Supported values are `siglip2_512` (512 with
interpolated position encoding, default), `siglip2_224` (explicit resolution
control with its own cache/artifacts), and `dinov2`. The ambiguous legacy value
`siglip2` remains unsupported.
`RCES_E0_ARTIFACT_TAG` can isolate a same-profile control run without replacing
the default E0 artifact.

Local Qwen inputs use `model_input_policy=max_1080p_v1`: after EXIF orientation,
images keep their aspect ratio and are downscaled only when needed so the
longest edge is at most 1920 and total area is at most `1920 x 1080` pixels.
This policy does not replace the fixed 224/512 input sizes used by the R-CES
SigLIP 2 feature profiles.

`local_qwen35.py` is resumable: compatible predictions are flushed after each
pair and reused on the next run. Set `LOCAL_QWEN_OVERWRITE = True` after changing
the compliance prompt or when a clean rerun is required.

## Splits

`USE_DATASET_SPLITS = False` is the pilot mode. It deterministically re-splits
the pair file's images 60/20/20 into image-disjoint train, validation, and test
sets.

`USE_DATASET_SPLITS = True` reads the official lists under `data/splits/`:

```text
train <- development_subset_images.txt
val   <- val_images.txt
test  <- realistic_test_images.txt
```

Use the official mode only after those splits have enough locked human labels.

## Paper analyses (10 seeds)

Every multi-seed number in the paper comes from one of these scripts, run with

```bash
RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
RCES_N_SEEDS=10 RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B CUBLAS_WORKSPACE_CONFIG=:4096:8 \
UV_CACHE_DIR=/tmp/uv-cache uv run python <script>
```

(for the backbone table, also set `RCES_VISION_ENCODER` to each encoder). Outputs land in
`experiments/artifacts/` with the `_cropaug` tag.

| paper | script | artifact |
|---|---|---|
| Tab 1 (IAA) | `tmp/compute_iaa.py` | `cache/iaa_report.json` (κ per field + disagreement list) |
| Tab 2: v1 and shuffled-image CIs | `ablation/hierarchical_ci.py` → `ablation/recompute_ci.py` | `hierarchical_frames_cropaug.json` → `within_stratum_ci_cropaug.json` |
| Tab 2: text-only / image-only; crops vs originals | `ablation/evidence_controls.py` | `evidence_controls_cropaug.json` |
| Tab 2: v2 | `rces-2/within_stratum.py` | `v2_within_stratum_cropaug.json` |
| §6.1 permutation test | `ablation/permutation_test.py` | `permutation_test_cropaug.json` |
| §6.1 rule-disjoint CV | `ablation/rule_disjoint.py` | `rule_disjoint_cropaug.json` |
| §6.1 causes; Tab 3; Fig 3 | `ablation/camera_ready.py`; figure: `figures/make_figures.py` | `camera_ready_siglip2_512_cropaug.json` |
| §6.2 gate on / off / visual subset; §6.5 error decomposition | `rces/evaluate.py` | `rces_eval_report_e0_corrected_siglip2_512_cropaug.json` |
| §6.3 ablation prose | `ablation/multiseed.py` | `multiseed_ablation_summary_cropaug.json` |
| Tab 4 (aggregation) | `ablation/aggregation_pairlevel.py` | `aggregation_pairlevel_cropaug.json` |
| Tab 5 (backbone) | `ablation/camera_ready.py` per encoder | `camera_ready_{siglip2_512,siglip2_224,dinov2}_cropaug.json` |
| Tab 6: answer-side gates | `baselines/cascade.py` per VLM | `cascade_task2_<VLM>.json` |
| Tab 6: R-CES column | `ablation/task2_multiseed.py` | `task2_multiseed_cropaug.json` |

The within-stratum estimator lives only in `src/stratified.py`. It computes AUROC per seed and
then averages, and bootstraps with one shared test-image resample per replicate.

## Ablations

`ablation/run.py` trains and evaluates:

| Name | Change |
|---|---|
| `full` | Corrected R-CES method with no is_visual input and a separate external gate. |
| `no_checkpoint_supervision` | Replaces dense checkpoint targets with one pair-level human sufficiency target. |
| `with_is_visual_legacy` | Restores the old is_visual head feature for comparison. |
| `no_text_conditioning` | Zeros checkpoint text so the head becomes image-only. |
| `aggregation_mean` | Reuses the full head but replaces `MIN` with mean. |
| `aggregation_product` | Reuses the full head but replaces `MIN` with product. |

### Visual-backbone ablation (DINOv2 vs SigLIP 2)

`config.VISION_ENCODER ∈ {siglip2_512, dinov2}` swaps only the **image** backbone the
head attends over; the checkpoint-text encoder stays SigLIP 2 (DINOv2 is
vision-only), so this isolates the visual representation. Image-patch caches are
per-encoder (`cache/experiments/<encoder>/image_patches/`) and the text cache is
shared, so both backbones coexist. The head reads `IMG_FEATURE_DIM` /
`N_PATCH` for the active encoder; `img_proj` and `txt_proj` project to a shared
hidden dim, so DINOv2's 1024-d patches and SigLIP 2's 768-d text mix cleanly. To
run the comparison: cache `siglip2_512`, train/evaluate; flip `VISION_ENCODER="dinov2"`,
run `encode_dinov2.py`, retrain/re-evaluate; compare the two `rces_eval_report`s.
Requires local `models/dinov2-large` weights (e.g. `facebook/dinov2-large`).

Additional ideas such as image-quality-only gates, rule-prior-only baselines,
critical-vs-all checkpoints, description-text inputs, and region-proposal tokens
(Grounding DINO / Florence-2) are future optional work and are not part of the
current scripts.

## Artifacts

All outputs are regenerable and gitignored:

- `experiments/artifacts/rces_image_split.json`
- `experiments/artifacts/local_qwen*/compliance_predictions_{structured_v2,direct}.jsonl`
- `cache/experiments/siglip2_512/image_patches/<image_id>.pt`
- `cache/experiments/siglip2/checkpoint_text.pt`
- `cache/experiments/siglip2_512/manifest.json`
- `experiments/artifacts/rces_head*.pt`
- `experiments/artifacts/rces_predictions_e0_corrected.jsonl`
- `experiments/artifacts/rces_eval_report_e0_corrected.json`
- `experiments/artifacts/ablation_*_report.json`
- `experiments/artifacts/retrieval/siglip_retrieval.jsonl`
- `experiments/artifacts/retrieval/vlm_retrieval.jsonl`
- `experiments/artifacts/retrieval/retrieval_comparison_report.{json,md}`

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests/test_offline_experiment_boundary.py
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests/test_local_vlm_interface.py tests/test_experiment_matrix.py
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests/test_retrieval.py
UV_CACHE_DIR=/tmp/uv-cache uv run python -m py_compile src/*.py Script/*.py experiments/**/*.py
```

These checks do not replace the GPU-backed feature, training, and evaluation
workflow. Do not claim experiment results until that workflow has actually run.
