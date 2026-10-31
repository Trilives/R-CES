# R-CES — checkpoint-level decidability benchmark and evidence-sufficiency scoring

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23827805.svg)](https://doi.org/10.5281/zenodo.23827805)

Code and data for *When Is an Image Enough? A Checkpoint-Level Decidability
Benchmark and Rule-Conditioned Evidence-Sufficiency Analysis for Visual Compliance*
(ACML 2026). **R-CES** (Rule-conditioned Checkpoint Evidence Sufficiency Scoring)
asks whether an image contains enough evidence to decide a rule, *P(evidence
sufficient to decide rule r from image x)*, rather than whether an answer is
likely correct.

The code, annotations and an image-free results archive are included. Verify all
six paper tables from the archived per-seed scores and Task-2 predictions on a
CPU, without downloading model weights or running VLMs:

```bash
uv sync --locked --python 3.13
uv run --locked python experiments/verify_paper_results.py
```

[results/acml2026/README.md](results/acml2026/README.md) describes the score schema,
configurations, hashes and table mapping. For photograph acquisition and crop
generation, see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md); for fixed model
snapshots and hardware/software, see [ENVIRONMENT.md](ENVIRONMENT.md).

## Benchmark at a glance

| | |
|---|---|
| Gold pairs | 1000 image–rule pairs over 250 images, 23 rules |
| Rule buckets (`decision_scope`) | 8 visual screening / 8 normative / 7 external evidence |
| Decidability | decidable 825 / undecidable 108 / partially_decidable 67 |
| Crop augmentation | 334 human-screened crop pairs over 230 crop images |
| Double annotation | 200 pairs; Cohen's κ 0.942 (decidability), 0.862 (critical-missing checkpoints) |
| Split | image-disjoint 60/20/20; crops inherit their source image's split |

Labels are human-only: no model output is shown during annotation or used as gold.
The annotation protocol is `codebook.md`; the κ report and the 19-pair
disagreement list are `data/pairs/iaa_report.json`.

## Evaluate your own predictor

No images or GPU needed to score predictions. List the 267 test pairs, add a score per pair
(higher = more likely the image suffices to decide the rule), and score them:

```bash
uv run python experiments/evaluate_scores.py          # writes experiments/artifacts/test_pairs.jsonl
RCES_SCORES=my_scores.jsonl uv run python experiments/evaluate_scores.py
```

It reports pooled AUROC, AURC and within-rule AUROC on all test pairs and on the 217
visual/normative pairs. Within-rule AUROC compares pairs only inside a rule, so a scorer that
reads only the rule text gets 0.5; report it next to pooled AUROC.

## Layout

```
src/                 core library (paths/config, IO, metrics, label vocab)
  config.py          all run-controls (module constants; env-var overrides)
  rces_io.py         canonical label vocab, JSONL/rule IO, image encoding hooks
  exp_io.py          experiment data loading (gold + crop augmentation)
  metrics.py         AUROC / AURC / ECE / original-image-family bootstrap CIs
  stratified.py      within-stratum AUROC + two-level (seed x original-image-family) bootstrap
experiments/         offline experiments (human gold only)
  rces/              v1 head (checkpoint-resolvability) — train/evaluate
  rces-2/            v2 head (text-context-first) — train/evaluate/ablation/multiseed,
                     within-stratum
  ablation/          v1 ablation + multi-seed drivers; 10-seed paper analyses
                     (channel controls, permutation test, rule-disjoint CV,
                     pair-level aggregation, calibration/backbone, Task-2)
  baselines/         answer-side confidence (MaxProb, entropy, margin, temperature
                     scaling, Kamath calibrator), self-evaluation (P(True),
                     verbalized), local-VLM driver, Task-2 cascade + 7-VLM decoupling
  retrieval/         SigLIP2 vs hierarchical-VLM rule retrieval comparison
  local_vlm/         local/remote VLM runtime and model comparisons
  data/              feature encoding (SigLIP2 / DINOv2) + split construction;
                     fixed upstream image acquisition + annotated-crop generation
  figures/           data-driven paper figure (calibration reliability diagrams)
  determinism.py     deterministic-training helper (math SDPA, seeded)
  evaluate_scores.py score any predictor's pair scores on the test split
  verify_paper_results.py CPU verification of all six paper tables
Script/              annotation pipeline (web UI, descriptions, model baseline)
Prompt/              English prompts (annotation, description, baselines, self-eval)
tests/               syntax/interface/boundary checks (not a pytest suite)
tmp/                 reproduction helpers (pair generators, validator, IAA)
data/
  pairs/             gold, isolated model baseline, double-annotation, crop aug,
                     23-rule benchmark subset, unified schema, field docs
  rule/              English rule library (54 rules) + schema + field docs
  upstream/          pinned ARGus image mapping, hashes, licences and evidence review
  SHA256SUMS         checksums of every shipped data file
  splits/            image-disjoint split lists
  enhance/           held-out test-expansion pairs
configs/             pinned model snapshots and validated runtime
results/acml2026/    per-seed scores, Task-2 predictions, reports, configs and hashes
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
| `OMP_NUM_THREADS=2` / `OPENBLAS_NUM_THREADS=1` | CPU thread limits used for the validated head archival runs |
| `RCES_LOCAL_QWEN_MODEL` | downstream compliance VLM for Task-2 (model dir name) |
| `RCES_REMOTE_VLM_URL` / `RCES_REMOTE_VLM_ID` | OpenAI-compatible (vLLM) endpoint + served model id for an off-box compliance VLM |
| `RCES_INPUT_MODE` | `full` (default) \| `text_only` \| `image_only` channel control |
| `RCES_SHUFFLE_IMAGE_EVAL` / `RCES_SHUFFLE_IMAGE_SEED` | shuffled image->pair assignment at evaluation |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | required for deterministic cuBLAS |

## Reproduction sketch

Set up with [ENVIRONMENT.md](ENVIRONMENT.md); obtain photographs as described in
[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) and pinned weights with
`RCES_MODEL_ACTION=download uv run --locked python experiments/data/download_models.py`.
The results archive verifies the original numbers independently of public pixels. Every paper float is mapped
to its command and artifact in [experiments/README.md](experiments/README.md),
including the separate manuscript-source build for the illustrative figures.

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
# 6. 10-seed paper analyses (script -> paper float: experiments/README.md)
env $ENV RCES_N_SEEDS=10 RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B uv run python experiments/ablation/camera_ready.py
# 7. inter-annotator agreement (CPU only, no images needed)
uv run python tmp/compute_iaa.py
```

The Task-2 study sweeps seven open-weight compliance VLMs (Gemma-4 E2B/E4B,
Qwen3.5-4B/9B, Qwen3.6-35B-A3B, Qwen3-VL-8B, InternVL3.5-8B); `RCES_LOCAL_QWEN_MODEL`
selects one. A remote OpenAI-compatible (vLLM) endpoint is supported via
`RCES_REMOTE_VLM_URL` / `RCES_REMOTE_VLM_ID` for models served off-box.

## Data integrity

`sha256sum -c data/SHA256SUMS` checks every shipped data file. The annotation,
rule and split files are byte-identical to the inputs of the reported results. `uv run python tmp/validate_rces_pairs.py` checks rule-id existence,
observed/missing ⊆ a rule's required checkpoints (and disjoint), the
critical-missing relation, schema consistency, and a model→gold leak guard. It
exits non-zero on any error. The shipped data passes with 0 errors / 0 warnings.

## Ground-truth protocol (annotation isolation)

No model ever writes a judgment field. The annotation order is **human locks
judgment → model adds description → human reviews**, so model narration can never
anchor the gold labels; `codebook.md` is the authoritative manual. Gold (`source=human`) and the isolated model baseline
(`source=model_full`) share one record shape (`rces_final_annotations.schema.json`)
but live in two physical files so the baseline can never contaminate gold.

## Citation

Software and results archive, version `v1.0.1-acml2026`:
[10.5281/zenodo.23827805](https://doi.org/10.5281/zenodo.23827805).
[All versions](https://doi.org/10.5281/zenodo.23827804).

```bibtex
@inproceedings{jiang2026rces,
  title     = {When Is an Image Enough? A Checkpoint-Level Decidability Benchmark and
               Rule-Conditioned Evidence-Sufficiency Analysis for Visual Compliance},
  author    = {Jiang, Xingpeng and Wang, Boran},
  booktitle = {Proceedings of the Asian Conference on Machine Learning (ACML)},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  volume    = {353},
  year      = {2026}
}
```

## License

Code and documentation: [MIT](LICENSE). Annotations, rule library, splits, IAA
report (`data/`), frozen results (`results/`) and the annotation codebook (`codebook.md`):
[CC BY 4.0](LICENSE-DATA). Source images are not distributed here and are not
covered by either license.
