# Changes

## v1.0.2-acml2026 — final camera-ready polish — 2026-10-31

- Match the submitted Figure 3 layout: one shared legend, readable text, distinct
  line styles and complete endpoint markers. Reliability bins, scores and result
  values are unchanged.
- Describe the aggregate as a sufficiency ranking score and link the manuscript
  packages, checksums and change history from the README.
- Attach the final camera-ready PDF, LaTeX source, supplement and checksums.
  The PDF and source link this version's Zenodo archive; the supplement includes
  the current version metadata, DOI and documentation. Bibliographic links and
  local editorial refinements are included; experimental scores and paper table
  values are unchanged.

## Evidence-review completion — 2026-10-31

- Recorded the author's completed comparison of all 230 annotated crops, covering
  334 annotation pairs: every crop retained its checkpoint evidence.
- Added the review summary and input hashes, and updated the access documentation.
  Labels, experimental scores and paper results are unchanged.

## ACML 2026 camera-ready release — 2026-10-31

- Risk–coverage curves and validation thresholds use complete equal-score groups,
  matching the implemented `score >= threshold` decision. AURC uses a
  right-endpoint step integral; average precision also handles complete ties.
- Bootstrap intervals cluster each original image with all its crops, using a
  shared family resample across seeds and computing AUROC per seed before averaging.
- The image-free `results/acml2026/` archive provides per-seed scores, Task-2
  predictions, validation inputs, configurations, reports and hashes. A CPU-only
  verifier recomputes all six paper tables.
- Model repository IDs and commit revisions, checked weight/config hashes,
  download commands, dependency pins and validated hardware/runtime are supplied.
- The fixed upstream photograph manifest and crop generator provide fresh-run
  image inputs. Original paper numbers are verified through frozen predictions.
