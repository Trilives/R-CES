# R-CES Image-Rule Pair Fields

Image-rule pair task files and the single schema for the R-CES evidence-sufficiency
annotation workflow.

## Human-gold and model-comparison boundary

**Human judgment and checkpoints are the only gold.** Models never prefill those
fields and their outputs never become R-CES training or evaluation targets.
Scripts live in `Script/`, shared helpers in `src/rces_io.py`:

The annotation decision rules (per-field boundaries, hard-case adjudication,
cross-field invariants, IAA protocol) live in `codebook.md` — the
authoritative manual for Stage-1 gold; this file is the schema-level reference.

- `annotate_web.py` (Stage 1): human fills only the judgment fields via the LAN web UI; no AI text shown.
- `describe_with_gpt.py` (optional): GPT drafts `visual_evidence` / `missing_information` under locked human judgment. Experiments do not read these fields.
- `baseline_qwen_full.py`: isolated, resumable Qwen3.7-Plus full-pipeline baseline, never ground truth.
- `compare_qwen_to_gold.py`: offline agreement and high-confidence disagreement report; never modifies either input.

## Files

- `rces_pilot_pairs.jsonl`: the GOLD file (`source=human`). Judgment and checkpoint supervision are human; any later free-text is post-processing, not supervision.
- `rces_double_annotation.jsonl`: the second annotator's independent pass over the ~20% `double_annotation=true` subset (same schema, separate file so the second pass cannot see the first annotator's labels). Written when `annotate_web.py` runs with `ANNOTATION_PASS="second"`; aligned to gold by `pair_id` for inter-annotator agreement.
- `rces_qwen_full_baseline.jsonl`: the isolated model baseline (`source=model_full` generic token, model in `model`), filled by `baseline_qwen_full.py`. Same record shape as gold, separate file; aligned to gold by `pair_id` and never used as supervision.
- `rces_final_annotations.schema.json`: the ONE schema both files validate against (`source` discriminates them).
- `benchmark_rule_subset.json`: the 23-rule benchmark subset (8 visual / 8 normative / 7 external) the pairs are built over; editable, regenerate pairs after changing it.
- `rces_pilot_pairs_stats.json`: generation statistics for the task file (1000 pairs = 250 development_subset images × 4 roles over the 23 subset rules), including the pair-selection method, SigLIP2 model id, similarity-cache path, and per-role rule usage.
- `tmp/siglip2_pair_generation/`: local similarity cache for SigLIP2 image-rule retrieval. It is regenerable diagnostic/cache material, not annotation ground truth.

Retired local development material is intentionally not included in this
anonymous supplement. The current files above are the release surface.

## Core metadata (immutable, carried into both files)

- `pair_id`: stable image-rule id, e.g. `rces_pilot_000001`.
- `image_id`: anonymized filename under `data/images/`.
- `image_split`: dataset split, e.g. `development_subset`.
- `rule_id`: stable rule id from `data/rule/rule.json`.
- `major_category`: `Opening Hazard`, `Edge Hazard`, `Unsafe Behavior`, `Civilized Construction Hazard`.
- `subcategory`, `rule_name`: from the rules library.
- `decision_scope`: `visual_screening_only` | `requires_external_evidence` | `normative_decision_available`.

## Candidate-construction provenance (not visual labels)

- `rule_role`: `target` | `related` | `not_applicable` | `possibly_undecidable`. Selection provenance only; never an R-CES feature. Current pilot pairs are generated with SigLIP2 image-rule similarity over `visual_retrieval_text`; legacy weak-text generation remains only as a fallback script mode.
  - `target` / `related` / `not_applicable`: selected by similarity over visual or normative-available rules, excluding `requires_external_evidence`.
  - `possibly_undecidable`: balanced over `requires_external_evidence` rules so the non-visual-evidence bucket is present by construction.
  - `visual_retrieval_text` should stay short and iconic: a few visible object/state noun phrases, not long rule prose, not source-file names, and not weak labels.
- `image_weak_label`: `compliant` | `non_compliant` from the source folder.
- `weak_candidate_category`: legacy field name. Under SigLIP2 pair generation it stores the target rule category, not a weak-label-derived category.

## Judgment fields (human gold; model values only in isolated comparison files)

Each accepts the placeholder `to_be_annotated` only while `annotation_status=pending`.

### `decidability` (3-state)
- `decidable`: this single image settles the rule-conditioned judgment, INCLUDING "the rule clearly does not apply".
- `partially_decidable`: the target is partly visible but at least one critical checkpoint is unresolved in this image (occlusion, angle, distance, crop, lighting).
- `undecidable`: the rule fundamentally needs non-visual evidence (mm, load, certificate, TSP/noise, duration); `decision_scope=requires_external_evidence` is the prior.

### `compliance_label` (4-state)
- `compliant`: positive visual evidence a PRESENT target meets the rule.
- `non_compliant`: visible violation.
- `not_applicable`: the rule's subject is absent (a decidable decision, not abstention).
- `unknown`: pairs with `partially_decidable` / `undecidable`.

### `image_quality`
- `good` | `usable` | `poor` — intrinsic pixel clarity of the image itself (focus, exposure, noise), **rated once per image, independent of the rule**. Framing — whether a given rule's subject is in frame — is **not** quality (it is decidability's, codebook §6). Judged at the image's **original resolution** (the annotator may zoom freely; labels are decoupled from any encoder's input size).

### `evidence_sufficiency_label` (R-CES's named gold — DERIVED, not hand-typed)
- `sufficient` | `partial` | `insufficient` — **derived from `decidability`** (codebook §7: decidable→sufficient, partially_decidable→partial, undecidable→insufficient), like `critical_missing_checkpoints`. The tools compute it; `tmp/validate_rces_pairs.py` enforces the derivation. Not asked of the annotator.

## Checkpoint layer (observation side; definitions live in the rules library)

- `observed_checkpoints`: checkpoint keys readable from this image. Subset of the rule's `required_checkpoints`, disjoint from `missing_checkpoints`.
- `missing_checkpoints`: checkpoint keys unresolved in this image. Subset of `required_checkpoints`.
- `critical_missing_checkpoints`: DERIVED, never hand-filled — `missing_checkpoints` INTERSECT the rule's `critical_checkpoints`.

`tmp/validate_rces_pairs.py` enforces the subset/disjoint/derivation invariants
and warns if a gold field equals its baseline twin while still pending.

## Evidence (description) fields

- `visual_evidence`: optional directly visible facts supporting the annotation.
- `missing_information`: optional wording of information missing for the visual-screening conclusion.

These fields are presentation metadata. R-CES training and evaluation do not
consume them. If generated for a release or qualitative analysis, they must be
conditioned on already-locked human judgment and reviewed before publication.

## Workflow / provenance

- `source`: `human` (gold) | `model_full` (baseline).
- `annotation_status`: `pending` -> `judged` for required gold; optional text enrichment may add `described` / `reviewed`; `model_complete` is comparison-only.
- `double_annotation`: true for the ~20% double-annotated gold subset (prioritising `partially_decidable` and `requires_external_evidence`). The first pass assigns it; the second annotator re-labels exactly this subset into `rces_double_annotation.jsonl`. `annotator_id` is derived from the pass (`annotator_1` / `annotator_2`), not typed by hand.
- Stage-1 gold extras: `annotator_id`, `annotation_seconds`.
- Stage-2 gold extras: `description_model`, `description_is_dry_run`, `description_flag_conflict`, `description_conflict_note`.
- Optional review: `review_status` (`final` | `needs_review` | `adjudicated`), `annotation_notes`.
- Stage-3 baseline extras: `model`, `is_dry_run`, `confidence` (0-1, MaxProb signal), `rationale`, `warning`.

## Labeling principles

- Do not infer hidden dimensions, fixed state, material, certificates, records, duration, weather, or process from a single image.
- `partially_decidable` when a critical checkpoint is unresolved in this image; `undecidable` when the rule principally needs non-visual evidence.
- `not_applicable` when the rule's subject is absent.
- Keep `visual_evidence` factual and visible; keep labels within the schema enums so R-CES targets stay clean.
