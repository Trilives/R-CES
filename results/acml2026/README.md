# Frozen paper results — ACML 2026

This image-free archive contains the inputs for verifying all six paper tables
without training heads, calling VLMs, or obtaining original photographs.
Run from the repository root:

```bash
uv run --locked python experiments/verify_paper_results.py
```

The full check verifies hashes and input fingerprints, then recomputes point
estimates and the 3,000-replicate seed × original-image-family intervals. For a
quick point-estimate check, set `RCES_VERIFY_INTERVALS=0`. The output is
`experiments/artifacts/paper_verification.json`.

| Path | Contents |
|---|---|
| `scores/v1_{siglip2_512,siglip2_224,dinov2}/seed_XX.json.gz` | Ten seeds per backbone: checkpoint probabilities, pair scores, targets, IDs, validation/test rows and validation-fit calibration outputs |
| `scores/{v2,shuffled,text_only,image_only}/seed_XX.json.gz` | Ten seeds per arm: aligned checkpoint and visual/normative pair scores, targets, IDs and original-image families |
| `inputs/answerers/*.json.gz` | Seven answerers: all 267 validation and 267 test predictions, gold labels, class probabilities and six answer-side confidence scores |
| `reports/paper_tables.json` | Canonical unrounded table values, recomputed by the verifier |
| `reports/` | Corrected run summaries, IAA report and supporting AUROC/count reports for the prose |
| `configs/` | Run/split configuration, pinned model snapshots and validated runtime |
| `manifest.json` | Metric convention, bootstrap settings, source/prompt hashes and semantic annotation fingerprints |
| `SHA256SUMS` | SHA-256 of every archive payload file |

JSON gzip files use deterministic compression timestamps. `seed_XX` runs are
seeds 0–9. A checkpoint frame has aligned `score`, `target`, `image_id`,
`family_id`, `pair_id` and `key` (`rule_id::checkpoint`) arrays. Pair frames use
`rule_id` in place of `key`. Higher scores mean greater evidence sufficiency;
checkpoint targets are resolvability, and pair targets are sufficiency. For v1,
these frames are nested under `frames`; other arms store `ckpt` and `pair` at
the top level. `validation_rows` and `test_rows` additionally retain the composed
external-gate score and compliance correctness. JSON `null` denotes an undefined
quantity, such as risk when no pair is accepted.

Each answerer row identifies its split, predicted/gold compliance class,
`sufficiency_gold`, `label_correct`, and `correct` (gold-sufficient **and** label
correct). `scores` holds MaxProb, temperature-scaled MaxProb, P(True), verbalized
confidence, entropy confidence and margin. Temperatures use validation labels.
No generated narration is required for these numeric checks.

| Paper float | Fields recomputed from scores/labels |
|---|---|
| Table 1 | `table_1`: independent labels, checkpoint agreement and κ |
| Table 2 | `table_2`: both heads and channel controls, pooled/within-stratum AUROC and family intervals |
| Table 3 | `table_3`: validation-selected operating points and test ECE |
| Table 4 | `table_4`: MIN/mean/product from the same checkpoint probabilities, including paired seed differences |
| Table 5 | `table_5`: three visual-backbone profiles |
| Table 6 | `table_6`: seven answerers' confidence gates and the same ten R-CES heads |
| Figure 3 | `experiments/figures/make_figures.py`: reliability bins from archived calibration scores |

Risk–coverage points accept entire equal-score groups using `score >= threshold`.
AURC is the sum of each group's coverage increment multiplied by the risk after
accepting that group. Original images and their crops share one bootstrap family
(100 test image IDs, 50 families). AUROC is computed separately per drawn seed
before averaging; heads' score scales are never pooled for an AUROC.

The v1/backbone and v2 archival runs reproduce every stored per-seed AUROC.
They use the original cached experimental features, rather than the blurred and
resized upstream release. These frozen outputs provide exact numeric verification;
[DATA_AVAILABILITY.md](../../DATA_AVAILABILITY.md) describes photograph access for
fresh experiments. Heads, feature tensors, model weights and image binaries are
not needed by the verifier. The source export script consumes original private
run artifacts; it is a release-maintenance tool, not a fresh-clone prerequisite.

Code is MIT; score/prediction/report data are CC BY 4.0. Model weight licences
remain with the model providers.
