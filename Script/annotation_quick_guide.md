# Annotation Quick Guide

Protocol version v1.1, 2026-06-10. This page summarizes the fields shown in
the local annotation UI. If this guide and an on-screen neutral prompt disagree,
follow `codebook.md`.

## Field Overview

| Field | Values | Meaning |
|---|---|---|
| `image_quality` | `good` / `usable` / `poor` | Intrinsic pixel quality of the image, independent of the rule. |
| checkpoint mark | `observed` / `missing` / `n/a` | Whether each required checkpoint is readable, unreadable, or not applicable. |
| `decidability` | `decidable` / `partially_decidable` / `undecidable` | Whether this single image can settle the rule-conditioned judgment. |
| `compliance_label` | `compliant` / `non_compliant` / `not_applicable` / `unknown` | The compliance conclusion, or abstention. |
| `evidence_sufficiency_label` | derived | Read-only; derived from `decidability`. |

Recommended order: mark checkpoints first, then `decidability`, then
`compliance_label`. The UI enforces cross-field constraints.

## Image Quality

Judge only the image itself:

- `good`: clear, with enough detail.
- `usable`: usable but somewhat blurry, dark, overexposed, or noisy.
- `poor`: too blurry, dark, overexposed, or noisy for reliable inspection.

Image quality is set once per image and reused across all rule pairs for that
image. Framing is not image quality: if a rule subject is cropped, occluded, or
out of frame, mark the relevant checkpoint or decidability field instead.
Gold labels are judged at the original image resolution; do not adapt labels to
a model input size such as 224 px.

## Checkpoints

Each rule lists `required_checkpoints`. For each checkpoint:

- `observed`: the state is readable from the image, whether it satisfies or
  violates the rule.
- `missing`: the checkpoint is in scope, but this image cannot resolve it
  because of occlusion, angle, crop, ambiguity, or poor visibility.
- `n/a`: the checkpoint is not applicable because the subject is absent or a
  conditional branch was not triggered.

Important distinctions:

- `observed` means resolved, not compliant. A visible violation is observed.
- `missing` means "should be assessable, but not from this image."
- `n/a` means "there is no subject or branch to assess."
- For subject-presence checkpoints such as `person_present` or
  `horizontal_opening_present`, a clearly absent subject is still `observed`
  because the presence state was resolved as false.

`critical_missing_checkpoints` is derived automatically as
`missing_checkpoints ∩ critical_checkpoints`.

## Decidability

Ask whether this image alone can settle the rule-conditioned judgment.

1. Is the rule subject present and identifiable?
   - Clearly absent: `decidable`; use `compliance_label=not_applicable`.
   - Presence cannot be determined: usually `partially_decidable`.
2. Does the rule fundamentally require non-visual evidence, such as exact
   dimensions, load, certificate, measurements, or duration?
   - Yes: `undecidable`.
3. Are all critical checkpoints readable from this image?
   - Yes: `decidable`.
   - No: `partially_decidable`.

Use this test for `partially_decidable` versus `undecidable`: if a better photo
of the same scene could resolve the missing evidence, use
`partially_decidable`; if no single photo could provide the required evidence,
use `undecidable`.

For rules with `decision_scope=requires_external_evidence`, unresolved subject
presence normally remains `undecidable`, because confirming the subject would
still leave the rule dependent on external evidence.

## Compliance Label

- `compliant`: positive visual evidence shows a present target meets the rule.
- `non_compliant`: a visible violation is present.
- `not_applicable`: the rule subject is clearly absent; this pairs with
  `decidability=decidable`.
- `unknown`: abstention; this pairs with `partially_decidable` or
  `undecidable`.

Do not infer `compliant` from subject absence. Absence is `not_applicable`, not
compliance.

## Derived Evidence Sufficiency

The tool derives this field from `decidability`:

- `decidable` -> `sufficient`
- `partially_decidable` -> `partial`
- `undecidable` -> `insufficient`

Annotators do not edit this field directly.

## Cross-Field Constraints

1. `decidable` requires `compliant`, `non_compliant`, or `not_applicable`.
2. `partially_decidable` and `undecidable` require `unknown`.
3. `not_applicable` requires `decidable`.
4. Any missing critical checkpoint prevents `decidable`.
5. `decidable + not_applicable` should have critical checkpoints marked
   `observed` or `n/a`, not `missing`.
6. `evidence_sufficiency_label` is always derived and must not be hand-edited.
