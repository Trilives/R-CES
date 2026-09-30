# Data availability — what is and isn't in this supplement

This supplement bundles all **code** and all **annotations / labels / rules /
splits** needed to understand and re-run the analysis. Large binary assets are
omitted to keep the package small and because they are either regenerable or
released separately; the pipeline writes regenerable assets back into the
omitted directories at the paths the code already expects.

## Included (the benchmark itself)

- **Annotations** — `data/pairs/`:
  - `rces_pilot_pairs.jsonl` — human gold (`source=human`), 1000 image–rule pairs.
  - `rces_double_annotation.jsonl` — independent second-annotator subset (IAA).
  - `rces_crop_augmentation.jsonl` — 334 human-screened crop pairs.
  - `rces_qwen_full_baseline.jsonl` — isolated model baseline (`source=model_full`),
    never ground truth.
  - `benchmark_rule_subset.json`, `rces_final_annotations.schema.json`, `fields.md`.
- **Rules** — `data/rule/rule.json` (54-rule English library) + schema + field docs.
- **Splits** — `data/splits/` image-disjoint split lists.
- **Held-out** — `data/enhance/` test-expansion pairs.
- All annotation identifiers are pseudonymous (`annotator_1`, `annotator_2`); image
  identifiers are anonymized (`img_NNNNNN`).

## Not included (regenerable or released separately)

| Path (created/used by the code) | Why omitted | How to obtain |
|---|---|---|
| `data/images/` | ~6.8 GB anonymized source images | released separately upon publication / on request |
| `data/images_crops/` | ~122 MB crop images for the augmentation pairs | released separately upon publication / on request |
| `models/` | frozen public backbones (SigLIP 2, DINOv2) + local VLMs | download from public model hubs into `models/<name>/` |
| `cache/` | regenerable frozen-feature cache | rebuilt by `experiments/data/encode_*.py` |
| `experiments/artifacts/` | regenerable run outputs (heads, predictions, reports) | rebuilt by the train/evaluate/ablation scripts |

The code resolves every path relative to the package root (`src/config.py`), so
re-providing `data/images/`, `data/images_crops/`, and the public weights under
`models/` makes the pipeline runnable end-to-end without edits.

## Note on the model baseline file

`rces_qwen_full_baseline.jsonl` is verbatim model output and a small number of its
`rationale` strings contain a stray non-English token produced by the model. These
are left unaltered on purpose — the baseline is an experimental record and is never
used as ground truth.
