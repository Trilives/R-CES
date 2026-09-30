# Dataset Splits

The anonymized construction-safety image pool — **1578 images** — is partitioned
into the splits the R-CES benchmark draws from. Each list file holds one
anonymized `image_id` (e.g. `img_0002.jpg`) per line, matching a file under
`data/images/`.

```text
1578 images
│
├── Train          stratified primary split; R-CES train image-rule pairs are drawn here
│   └── Development subset   small slice sampled FROM train, used for the pilot
│
├── Validation     stratified primary split; threshold / prompt / model selection
│
├── Realistic test stratified primary split; final evaluation on the realistic class distribution
│
└── Balanced test  independently sampled, class-balanced; for category-bias analysis only
```

## Files

| File | Contents |
|---|---|
| `image_split.json` | Full split definition and per-split statistics (source of truth). |
| `train_images.txt` | Train image list. |
| `val_images.txt` | Validation image list. |
| `realistic_test_images.txt` | Realistic-distribution test image list. |
| `development_subset_images.txt` | Pilot subset sampled from the train split. |
| `test_balanced_images.txt` | Class-balanced list for bias analysis. |
| `test_expansion_selected_images.txt` | Human-selected unused-image pool for the next annotation batch; generated on first selection and not yet a locked test split. |

## Test-expansion image selection

The active 1,000-pair pilot uses 250 images. To browse the remaining image pool
one image at a time and select high-value cases for annotation, run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python Script/select_test_images_web.py
```

Open the printed localhost URL. **Select and next** immediately adds the current
image name to `test_expansion_selected_images.txt`; **deselect** removes it.
The browser cursor is stored under `cache/`, so stopping and restarting the
script resumes from the last image.

These are field-inspection photos, so the inspector's primary photographed
subject should be clear and judgeable; do not select an image merely because
that primary subject is unclear. Select it when the image edge or background
also contains another inspection subject from at least one of the four rule
categories -- Opening Hazard, Edge Hazard, Unsafe Behavior, or Civilized
Construction Hazard -- but that secondary subject is cropped, occluded, or
touching the frame boundary, so its rule-relevant state cannot be judged. If the
image contains only a judgeable primary subject and no such secondary edge case,
do not select it. This file is an annotation candidate pool only. After labels
are available, stratify by whether an image contains a missing checkpoint, then
rebuild and lock the final train/validation/test split.

## Counts

| Split | Compliant | Non-compliant | Total |
|---|---:|---:|---:|
| Train | 73 | 1033 | 1106 |
| Validation | 15 | 221 | 236 |
| Realistic test | 15 | 221 | 236 |
| Development subset | 49 | 300 | 349 |
| Balanced test | 40 | 40 | 80 |

## Split rules

- **Train / Validation / Realistic test** are stratified primary splits at an
  approximate **70 / 15 / 15** ratio; together they cover every image exactly
  once (disjoint).
- **Development subset** is sampled *from* train, so it overlaps train by design.
  The pilot benchmark — the **1000 image-rule pairs** in
  `data/pairs/rces_pilot_pairs.jsonl` — is built from **250 images** sampled
  from this subset (× 4 rule roles over the 23-rule benchmark subset).
- **Balanced test** is sampled independently for class-bias analysis and may
  overlap the primary splits.

> These split lists are committed dataset-definition metadata. No code in this
> repository reads them at annotation/inference time (each pair already carries
> its `image_split`); they document how the benchmark is partitioned and back
> the planned expansion of annotation onto the val / realistic-test splits.
