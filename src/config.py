"""Central run-controls for the R-CES Phase-2 experiments (Readme steps 7-9).

Repo convention: NO CLI flags. Edit the module constants below, then run a
script, e.g.

  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/data/encode_siglip2.py

All code under `experiments/` is offline and uses human gold only. Cloud
annotation comparisons live under `Script/` and are not experiment inputs.

R-CES design recap this config wires up (see README.md):
  - atomic unit = checkpoint; the head predicts resolvable(k, x, r), not the
    compliance answer, from R-CES's OWN frozen SigLIP 2 features (decoupled from
    the compliance VLM);
  - dense supervision = per-critical-checkpoint resolvable (observed/na->1,
    missing->0); pair sufficiency is MECHANICALLY aggregated (AND/min) and is
    derived gold, never hand-typed.
"""

from __future__ import annotations

import os
from pathlib import Path

# This module lives in src alongside rces_io.py; experiment scripts live in
# experiments/<category>/.
SRC_DIR = Path(__file__).resolve().parent
RCES_ROOT = SRC_DIR.parent
EXP_DIR = RCES_ROOT / "experiments"
ARTIFACT_DIR = EXP_DIR / "artifacts"

# --- Visual backbone selection (image-encoder ablation) -----------------------
# R-CES's IMAGE features come from ONE frozen visual backbone; the checkpoint
# TEXT conditioning is always SigLIP 2 (DINOv2 is vision-only, no text tower).
# So VISION_ENCODER is a pure image-encoder ablation: it isolates the visual
# backbone's effect while holding the rule-text encoder fixed. The head projects
# image and text to a shared hidden dim independently (model.py), so differing
# image / text feature dims are fine. (This realises the README "second visual
# backbone" item: DINOv2 patch tokens vs SigLIP 2 patch tokens.)
VISION_ENCODER = os.environ.get(
    "RCES_VISION_ENCODER", "siglip2_512"
)  # "siglip2_224" | "siglip2_512" | "dinov2".
SIGLIP2_DIR = RCES_ROOT / "models" / "siglip2-base-patch16-224"
SIGLIP2_NATIVE_IMAGE_SIZE = 224
# Retrieval and pair generation retain the canonical 512px input. R-CES image
# feature experiments use VISION_IMAGE_SIZE from the selected profile instead.
SIGLIP2_IMAGE_SIZE = 512
DINOV2_DIR = RCES_ROOT / "models" / "dinov2-large"
DEVICE = "cuda"  # "cuda" | "cpu"; scripts fall back to cpu when cuda is absent.

# DINOv2 input resolution. Construction-safety detail (guardrail height, cover
# fixing, helmet straps) is lost at 224; DINOv2 supports variable resolution, so
# we feed its native dense-task size. 518/14 = 37 -> 37x37 = 1369 patch tokens.
# Must be a multiple of the patch size; the encode script sets the processor to
# this and passes interpolate_pos_encoding=True.
DINOV2_PATCH = 14
DINOV2_IMAGE_SIZE = 518

# Per-encoder image-feature geometry. The encode script auto-detects the true
# values at runtime and records them in the cache manifest. IMG_FEATURE_DIM is
# what the head is CONSTRUCTED with, so it must match the active encoder; N_PATCH
# is informational only (the head's attention pools over any token count).
ENCODER_IMAGE_SIZE = {
    "siglip2_224": SIGLIP2_NATIVE_IMAGE_SIZE,
    "siglip2_512": SIGLIP2_IMAGE_SIZE,
    "dinov2": DINOV2_IMAGE_SIZE,
}
ENCODER_IMG_DIM = {
    "siglip2_224": 768,
    "siglip2_512": 768,
    "dinov2": 1024,
}  # dinov2-large hidden size.
ENCODER_N_PATCH = {
    "siglip2_224": (ENCODER_IMAGE_SIZE["siglip2_224"] // 16) ** 2,
    "siglip2_512": (ENCODER_IMAGE_SIZE["siglip2_512"] // 16) ** 2,
    "dinov2": (DINOV2_IMAGE_SIZE // DINOV2_PATCH) ** 2,
}
if VISION_ENCODER not in ENCODER_IMG_DIM:
    raise ValueError(f"VISION_ENCODER must be one of {tuple(ENCODER_IMG_DIM)}; got {VISION_ENCODER!r}")
IMG_FEATURE_DIM = ENCODER_IMG_DIM[VISION_ENCODER]
N_PATCH = ENCODER_N_PATCH[VISION_ENCODER]
VISION_IMAGE_SIZE = ENCODER_IMAGE_SIZE[VISION_ENCODER]
TXT_FEATURE_DIM = 768  # checkpoint text is always SigLIP 2 (768-d), independent of VISION_ENCODER.


def active_vision_dir() -> Path:
    """Local weights dir for the currently selected image backbone."""
    return SIGLIP2_DIR if VISION_ENCODER.startswith("siglip2") else DINOV2_DIR


# --- Feature cache (gitignored under cache/) ----------------------------------
# Image patches live under a PER-ENCODER dir so the two backbones' caches never
# collide; the checkpoint-text cache is encoder-INDEPENDENT (always SigLIP 2).
FEATURE_DIR = RCES_ROOT / "cache" / "experiments" / VISION_ENCODER
IMG_FEATURE_DIR = FEATURE_DIR / "image_patches"  # one <image_id>.pt per image
FEATURE_MANIFEST = FEATURE_DIR / "manifest.json"
CKPT_TEXT_PATH = RCES_ROOT / "cache" / "experiments" / "siglip2" / "checkpoint_text.pt"  # {"<rule_id>::<key>": vec}

# --- Local downstream compliance VLM -----------------------------------------
# Keep the 4B model as the default experiment arm. A one-off model-size run can
# select another locally downloaded Qwen checkpoint without editing this file:
#   RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B uv run python .../local_qwen35.py
LOCAL_QWEN_MODEL_NAME = os.environ.get("RCES_LOCAL_QWEN_MODEL", "Qwen3.5-4B")
_LOCAL_QWEN_ARTIFACT_NAMES = {
    "Qwen3.5-4B": "local_qwen35",
    "Qwen3.5-9B": "local_qwen35_9b",
    "Qwen3-VL-8B-Instruct": "local_qwen3vl_8b",
    # Supplementary non-Qwen edge backbones (generic AutoModelForImageTextToText).
    "gemma-4-E2B-it": "local_gemma_e2b",
    "gemma-4-E4B-it": "local_gemma_e4b",
    "InternVL3_5-8B": "local_internvl_8b",
    # HF-native repackaging of InternVL3.5-8B (model_type internvl); loads through
    # the generic runtime, unlike the OpenGVLab custom-code build above.
    "InternVL3_5-8B-HF": "local_internvl_8b_hf",
    # Remote server-class anchor served by vLLM on another host (runs in parallel
    # with the local GPU; reached over the OpenAI-compatible HTTP API).
    "Qwen3.6-35B-A3B": "local_qwen36_35b",
}
# Models whose weights are NOT local: selected by RCES_LOCAL_QWEN_MODEL but served
# remotely. base_url is the OpenAI-compatible root; model_id is the served name.
_REMOTE_VLM_ENDPOINTS = {
    "Qwen3.6-35B-A3B": {
        "base_url": os.environ.get("RCES_REMOTE_VLM_URL", "http://localhost:8000/v1"),
        "model_id": os.environ.get("RCES_REMOTE_VLM_ID", "Qwen/Qwen3.6-35B-A3B-FP8"),
    },
}
LOCAL_VLM_IS_REMOTE = LOCAL_QWEN_MODEL_NAME in _REMOTE_VLM_ENDPOINTS
REMOTE_VLM_BASE_URL = (
    _REMOTE_VLM_ENDPOINTS[LOCAL_QWEN_MODEL_NAME]["base_url"] if LOCAL_VLM_IS_REMOTE else ""
)
REMOTE_VLM_MODEL_ID = (
    _REMOTE_VLM_ENDPOINTS[LOCAL_QWEN_MODEL_NAME]["model_id"] if LOCAL_VLM_IS_REMOTE else ""
)
if LOCAL_QWEN_MODEL_NAME not in _LOCAL_QWEN_ARTIFACT_NAMES:
    raise ValueError(
        "RCES_LOCAL_QWEN_MODEL must be one of "
        f"{tuple(_LOCAL_QWEN_ARTIFACT_NAMES)}; got {LOCAL_QWEN_MODEL_NAME!r}"
    )
LOCAL_QWEN_MODEL_DIR = RCES_ROOT / "models" / LOCAL_QWEN_MODEL_NAME
LOCAL_VLM_ARTIFACT_DIR = ARTIFACT_DIR / _LOCAL_QWEN_ARTIFACT_NAMES[LOCAL_QWEN_MODEL_NAME]
# Compatibility alias for callers not yet renamed to the model-generic constant.
LOCAL_QWEN35_DIR = LOCAL_QWEN_MODEL_DIR
LOCAL_QWEN_DTYPE = "auto"
LOCAL_QWEN_DEVICE_MAP = "auto"
LOCAL_VLM_INPUT_POLICY = "max_1080p_v1"
LOCAL_VLM_MAX_LONGEST_EDGE = 1920
LOCAL_VLM_MAX_PIXELS = 1920 * 1080
LOCAL_QWEN_LIMIT_PAIRS = 0  # 0 = every locked human pair.
LOCAL_QWEN_DRY_RUN = False
LOCAL_QWEN_OVERWRITE = False  # True ignores compatible cached predictions.
# Neutral 4-state (A/B/C/D) compliance prompt for the local VLM, loaded as a
# template (placeholder <<RULE_JSON>>); lives with the other prompts so it is
# editable without touching code. Balanced framing avoids the over-abstention
# (everything -> D/unknown) the earlier hardcoded prompt produced.
LOCAL_COMPLIANCE_PROMPT_PATH = RCES_ROOT / "Prompt" / "local_compliance_prompt.md"
# Checkpoint-guided STRUCTURED reasoning prompt (single pair -> one JSON object:
# per-checkpoint observation+judgment, then a forced 3-state compliance label).
# The JSON supplies a validated, complete checkpoint analysis. A restricted A/B/C
# decision over that analysis produces both `predicted_compliance` and confidence.
# The compliance VLM never abstains in this structured arm; R-CES owns abstention.
LOCAL_STRUCTURED_PROMPT_PATH = RCES_ROOT / "Prompt" / "local_structured_prompt.md"
LOCAL_STRUCTURED_DECISION_PROMPT_PATH = RCES_ROOT / "Prompt" / "local_structured_decision_prompt.md"
LOCAL_STRUCTURED_MAX_NEW_TOKENS = 768
LOCAL_STRUCTURED_MAX_ATTEMPTS = 2
# Ablation toggle (Readme step 9 / 必要消融 conditions 1-2 vs 3-4): True routes the
# label through checkpoint-guided JSON reasoning followed by A/B/C classification;
# False uses direct A/B/C/D classification without checkpoint reasoning.
LOCAL_USE_STRUCTURED = True

# Local Qwen3.5-4B is the EXPERIMENT's compliance VLM (method + ablation data),
# NOT the Stage-3 frontier model baseline (a cloud model, source=model_full),
# which lives in data/pairs/ and is never read or written by this offline line.
# The two reasoning-mode arms persist in SEPARATE files so flipping
# LOCAL_USE_STRUCTURED never overwrites the other arm; every consumer
# (baselines/common.py, rces/evaluate.py) reads whichever arm is active.
_LOCAL_QWEN_PRED_FILE = {
    True: "compliance_predictions_structured_v2.jsonl",  # complete + confidence-aligned
    False: "compliance_predictions_direct.jsonl",       # direct A/B/C/D label (ablation arm)
}
LOCAL_QWEN_PREDICTIONS_PATH = LOCAL_VLM_ARTIFACT_DIR / _LOCAL_QWEN_PRED_FILE[LOCAL_USE_STRUCTURED]

# --- Self-evaluation Task-2 baselines (P(True), verbalized confidence) -----------
# Elicited confidence signals computed AFTER the compliance answer: the model is
# re-shown the image+rule and its own proposed label and asked whether it is
# correct (P(True), Kadavath et al. 2022) and how confident it is 0-100
# (verbalized, Tian et al. 2023). Written per VLM next to its compliance file.
SELF_EVAL_PREDICTIONS_PATH = LOCAL_VLM_ARTIFACT_DIR / "self_eval_predictions.jsonl"
PTRUE_PROMPT_PATH = RCES_ROOT / "Prompt" / "ptrue_prompt.md"
VERBALIZED_PROMPT_PATH = RCES_ROOT / "Prompt" / "verbalized_prompt.md"

# --- Crop-augmentation arm (RCES_INCLUDE_CROP_AUG=1) ---------------------------
# When on, the experiment pipeline additionally loads the human-screened crop
# augmentation pairs (data/pairs/rces_crop_augmentation.jsonl). Crops are random
# crops of the ORIGINAL pilot images; each crop carries its mother image's split
# in `image_split`, so adding them never crosses the train/val/test boundary the
# deterministic re-split of the 250 originals already drew (verified leakage-free:
# crop.image_split == mother pilot split for every crop). The 250-image
# deterministic split is left untouched; crops are only appended to their own
# split bucket. All cropaug artifacts (split file, head, predictions, report) get
# a distinct path so the no-aug baseline artifacts are never overwritten.
INCLUDE_CROP_AUGMENTATION = os.environ.get("RCES_INCLUDE_CROP_AUG", "0") == "1"
_CROPAUG_TAG = "_cropaug" if INCLUDE_CROP_AUGMENTATION else ""

# --- Image-disjoint split over the 250 development_subset images ---------------
# Built once from the pair file's image set (deterministic on SPLIT_SEED), so the
# split is fixed regardless of how far annotation has progressed.
SPLIT_PATH = ARTIFACT_DIR / f"rces_image_split{_CROPAUG_TAG}.json"
SPLIT_SEED = 20260527
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train / val / test, image-disjoint.

# --- Split mode: pilot internal re-split vs. dataset-level official split -------
# False (pilot, default): re-split the pair file's OWN images SPLIT_RATIOS-wise,
#   image-disjoint, deterministic on SPLIT_SEED. Use this while only the
#   development_subset is annotated -- those 250 images are all train-side in the
#   dataset split, so the dataset split alone gives no val/test to work with.
# True (official run): read the dataset-level lists under data/splits/ instead --
#   train the head on the annotated train-side images (development_subset), tune
#   tau on val_images, report on realistic_test_images. Flip this once annotation
#   has reached the val / realistic-test splits. load_split() rebuilds the split
#   artifact automatically when this flag (the recorded "mode") changes.
USE_DATASET_SPLITS = False
DATASET_SPLIT_DIR = RCES_ROOT / "data" / "splits"
# Which dataset list file backs each experiment split when USE_DATASET_SPLITS=True.
# train <- development_subset (sampled from the dataset train split); val/test are
# the disjoint primary val / realistic-test splits, so the three stay image-disjoint.
DATASET_SPLIT_FILES = {
    "train": "development_subset_images.txt",
    "val": "val_images.txt",
    "test": "realistic_test_images.txt",
}

# --- Which gold rows carry a locked human judgment ----------------------------
# Stage 1 -> judged; Stage 2 -> described; Stage 3 review -> reviewed. All three
# have judgment fields locked, so all three are usable as supervision/eval gold.
LOCKED_STATUSES = ("judged", "described", "reviewed")

# --- R-CES head ---------------------------------------------------------------
HIDDEN_DIM = 256
DROPOUT = 0.1
USE_IS_VISUAL_FEATURE = False  # external-evidence handling is a post-head gate.
EXTERNAL_GATE_SCORE = 0.0

# --- Training -----------------------------------------------------------------
EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
# Base seed; RCES_SEED overrides it so a multi-seed driver can vary it per run
# without editing source. Determinism (cudnn deterministic + deterministic algos)
# is requested by experiments.determinism.set_determinism, gated by DETERMINISTIC.
SEED = int(os.environ.get("RCES_SEED", "0"))
DETERMINISTIC = os.environ.get("RCES_DETERMINISTIC", "1") == "1"
_E0_ARTIFACT_TAG = os.environ.get("RCES_E0_ARTIFACT_TAG")
if _E0_ARTIFACT_TAG is None:
    _E0_ARTIFACT_TAG = f"_{VISION_ENCODER}"
elif _E0_ARTIFACT_TAG and not _E0_ARTIFACT_TAG.startswith("_"):
    _E0_ARTIFACT_TAG = f"_{_E0_ARTIFACT_TAG}"
HEAD_CKPT = ARTIFACT_DIR / f"rces_head_e0_corrected{_E0_ARTIFACT_TAG}.pt"
PREDICTIONS_PATH = ARTIFACT_DIR / f"rces_predictions_e0_corrected{_E0_ARTIFACT_TAG}.jsonl"
RCES_EVAL_REPORT_PATH = ARTIFACT_DIR / f"rces_eval_report_e0_corrected{_E0_ARTIFACT_TAG}.json"

# --- Ablations (Readme step 9). ablations.py flips these around train/evaluate.-
ABLATE_NO_CHECKPOINT_SUPERVISION = False  # train on pair sufficiency, not per-k.
ABLATE_NO_IS_VISUAL = False  # legacy compatibility; corrected E0 already omits it.
ABLATE_NO_TEXT_CONDITIONING = False  # zero checkpoint text -> image-only head.
AGGREGATION_MODE = "min"  # "min" | "mean" | "product".

# --- Selective prediction / evaluation ----------------------------------------
TARGET_RISK = 0.10  # coverage@risk and effective-reliability report point.
ECE_BINS = 10

# --- Rule-retrieval comparison experiment (experiments/retrieval) -------------
# Compares two ways to decide which rules a construction image is a candidate for:
#   (1) SigLIP 2 image<->rule-text similarity (the current pair generator), and
#   (2) a hierarchical local-VLM retriever (Stage A: which of the 4 hazard
#       CATEGORIES is visible, mapped to the A/B/C/D runtime; Stage B: per-rule
#       yes/no applicability over ONLY the rules in the selected categories, so
#       the model is never shown all rules at once).
# Both are scored against the human-gold applicability (compliance_label !=
# not_applicable) on the annotated pairs. Offline, local model only.
# Active retrieval method for the downstream pipeline / the Readme final
# multi-method comparison: which retriever's candidate rules are treated as the
# authoritative source. compare.py records it as "active_method"; any consumer
# picks its artifact via active_retrieval_path(). Both artifacts are still
# produced so the comparison can run regardless of which one is active.
RETRIEVAL_METHOD = "siglip2"  # "siglip2" | "vlm_hierarchical"
RETRIEVAL_METHODS = ("siglip2", "vlm_hierarchical")

RETRIEVAL_ARTIFACT_DIR = ARTIFACT_DIR / "retrieval"
RETRIEVAL_VLM_PATH = RETRIEVAL_ARTIFACT_DIR / "vlm_retrieval.jsonl"
RETRIEVAL_SIGLIP_PATH = RETRIEVAL_ARTIFACT_DIR / "siglip_retrieval.jsonl"
RETRIEVAL_REPORT_PATH = RETRIEVAL_ARTIFACT_DIR / "retrieval_comparison_report.json"
RETRIEVAL_REPORT_MD_PATH = RETRIEVAL_ARTIFACT_DIR / "retrieval_comparison_report.md"
# Rule pool: the 23-rule benchmark subset (matches rces_pilot_pairs / gold), so
# the VLM retriever and SigLIP rank over the same set the pairs were built from.
RETRIEVAL_SUBSET_PATH = RCES_ROOT / "data" / "pairs" / "benchmark_rule_subset.json"
# Stage prompts live with the other prompts under Prompt/.
RETRIEVAL_CATEGORY_PROMPT_PATH = RCES_ROOT / "Prompt" / "retrieval_category_prompt.md"
RETRIEVAL_RULES_PROMPT_PATH = RCES_ROOT / "Prompt" / "retrieval_rules_prompt.md"
RETRIEVAL_CATEGORY_THRESHOLD = 0.15  # Stage A: keep categories with prob >= this (argmax always kept).
# Stage B: one generation call per selected category returns the top-K most
# relevant rules (NOT one query per rule -- that was too slow).
RETRIEVAL_TOP_K_PER_CATEGORY = 3
RETRIEVAL_GEN_MAX_NEW_TOKENS = 48
RETRIEVAL_SIGLIP_TOP_K = 4           # SigLIP "applicable" set size for agreement / top-K headline.
RETRIEVAL_LIMIT_IMAGES = 0           # 0 = every image in the gold pair file.
RETRIEVAL_VLM_DRY_RUN = False        # True uses DryRunOptionRuntime (no model/GPU).
RETRIEVAL_VLM_OVERWRITE = False      # True ignores compatible cached VLM rows.


def active_retrieval_path() -> Path:
    """Artifact of the retrieval method selected by RETRIEVAL_METHOD."""
    if RETRIEVAL_METHOD not in RETRIEVAL_METHODS:
        raise ValueError(f"RETRIEVAL_METHOD must be one of {RETRIEVAL_METHODS}; got {RETRIEVAL_METHOD!r}")
    return RETRIEVAL_SIGLIP_PATH if RETRIEVAL_METHOD == "siglip2" else RETRIEVAL_VLM_PATH
