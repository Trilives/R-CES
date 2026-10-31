# Data availability

R-CES contributes decidability and checkpoint annotations, an annotation protocol,
rules, splits, agreement reports and evaluation code. The photographs are reused
from upstream ARGus/RCASR and hosted there. No image binaries are distributed in
this repository or its supplementary ZIP. Repository setup is in
[ENVIRONMENT.md](ENVIRONMENT.md).

## Included (the benchmark itself)

- **Annotations** — `data/pairs/`:
  - `rces_pilot_pairs.jsonl` — human gold (`source=human`), 1000 image–rule pairs.
  - `rces_double_annotation.jsonl` — independent second-annotator subset (IAA).
  - `iaa_report.json` — Cohen's κ per field over the 200 double-annotated pairs,
    plus the 19-pair disagreement list (regenerate with `tmp/compute_iaa.py`).
  - `rces_crop_augmentation.jsonl` — 334 human-screened crop pairs.
  - `rces_qwen_full_baseline.jsonl` — isolated model baseline (`source=model_full`),
    never ground truth.
  - `benchmark_rule_subset.json`, `rces_final_annotations.schema.json`, `fields.md`.
- **Rules** — `data/rule/rule.json` (54-rule English library) + schema + field docs.
  The benchmark's 23 rules were selected from the 55-rule source library; in the
  released library its stair-edge rule is merged into `R-EDG-003` (roof, balcony,
  stair or platform edge). All gold pairs are annotated against the released
  library, which `tmp/validate_rces_pairs.py` enforces.
- **Splits** — `data/splits/` image-disjoint split lists.
- **Held-out** — `data/enhance/` test-expansion pairs.
- All annotation identifiers are pseudonymous (`annotator_1`, `annotator_2`); image
  identifiers are anonymized (`img_NNNNNN`).

Everything under `data/` and `codebook.md` is licensed CC BY 4.0 (`LICENSE-DATA`);
the code is MIT (`LICENSE`).

## Fixed upstream photograph snapshot

Use ARGus/RCASR **2.4.0**, commit
[`bc47272afea00b52456f36e6ffc96deda8c76aaa`](https://github.com/Trilives/argus/tree/bc47272afea00b52456f36e6ffc96deda8c76aaa).
The commit's public manifest and citation metadata were retrieved without credentials
on 2026-10-30. The frozen commit is the version identifier; the upstream all-versions
Zenodo DOI is not a pin for this image pool.

[data/upstream/release.json](data/upstream/release.json) records the commit, download
base URL and upstream manifest hash. [data/upstream/images.csv](data/upstream/images.csv)
maps every original R-CES `image_id` directly to its upstream file, released-file
SHA-256, original-source SHA-256, EXIF-transposed original dimensions, released
dimensions, source-byte aliases and per-file licence. The original-source hash
identifies the experimental input; acquisition verifies the **released-file** hash.
The mapping covers the main benchmark, all crop parents, the held-out expansion
used in this release. The historical full-pool split lists also reference
unpublished, unannotated photographs outside this snapshot; they are not the
paper's experiment split. The default experiment split is rebuilt from the
annotated pilot images and inherited by the crops. Crop IDs and boxes are owned by
`data/pairs/rces_crop_augmentation.jsonl`, not duplicated in another mapping.

Image licences differ from the R-CES annotation licence. Follow the `license` and
`license_url` fields for each file: upstream gold photographs and specified prior
gold-image grants retain CC BY 4.0; prior demonstration grants retain Apache-2.0;
the other additional photographs use CC BY-NC 4.0. Commercial use of those
CC BY-NC files requires separate permission from the upstream image owner.
Cropping does not change a parent's licence. Cite both the upstream manuscript
and the frozen image snapshot using its public
[CITATION.cff](https://github.com/Trilives/argus/blob/bc47272afea00b52456f36e6ffc96deda8c76aaa/CITATION.cff).

## Acquire photographs and generate the existing crops

Run from the root of a fresh checkout after setup:

```bash
uv run --locked python experiments/data/acquire_images.py
uv run --locked python experiments/data/generate_fixed_crops.py
```

The first command downloads every mapped photograph from the fixed public commit
to `data/images/`, checks its SHA-256, and can resume. It refuses to overwrite a
different existing file, including an original experimental photograph. No local
private image directory, credentials or model weights are needed.

The second command generates only the existing annotated crops under
`data/images_crops/`, using their existing image IDs, parent IDs and bounding boxes.
It requires an empty crop output directory to protect original experimental crops.
It checks parent hashes, applies EXIF orientation, and maps original-parent pixel
coordinates to released-parent pixels by multiplying horizontal coordinates by
`release_width/original_width` and vertical coordinates by
`release_height/original_height`. Left/top bounds round down; exclusive right/bottom
bounds round up. This retains the annotated field of view, with less than one extra
released pixel on each edge. It writes JPEGs at quality 95 with 4:4:4 subsampling
and no EXIF, plus `data/images_crops/generated_manifest.json` containing the actual
transformed boxes, output sizes and hashes. There are no new random crops and no
separate upstream crop download.

**Result verification and fresh training.** The archived scores and Task-2
predictions in [results/acml2026/](results/acml2026/README.md) reproduce the paper's
numbers on a CPU. They were generated from the original experimental photographs
and crops. The upstream image release applies privacy blurring, resizing and JPEG
encoding. The author compared all 230 annotated crops (covering 334 annotation
pairs) with their public-parent reconstructions and found no loss of checkpoint
evidence: all 230 were marked retained, with none changed or uncertain. The
[review summary](data/upstream/evidence_review.json) records the scope, crop IDs
and input hashes. Released pixels support fresh experiments; exact numerical
verification uses the frozen predictions, since the image bytes differ.

## Other omitted assets

| Path (created/used by the code) | Why omitted | How to obtain |
|---|---|---|
| `models/` | frozen public backbones (SigLIP 2, DINOv2) + local VLMs | fixed repository IDs, revisions and download commands in [ENVIRONMENT.md](ENVIRONMENT.md) |
| `cache/` | regenerable frozen-feature cache | rebuilt by `experiments/data/encode_*.py` |
| `experiments/artifacts/` | full training outputs and heads | rebuilt by the train/evaluate/ablation scripts; paper scores, Task-2 predictions and reports are included under `results/acml2026/` |

The code resolves every path relative to the package root (`src/config.py`), so
re-providing `data/images/`, `data/images_crops/`, and the public weights under
`models/` provides the inputs for fresh runs. The results archive checks the reported numbers without these assets.

## Note on the model baseline file

`rces_qwen_full_baseline.jsonl` is verbatim model output and a small number of its
`rationale` strings contain a stray non-English token produced by the model. These
are left unaltered on purpose — the baseline is an experimental record and is never
used as ground truth.
