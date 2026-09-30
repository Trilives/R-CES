# R-CES Annotation Codebook

**Protocol version: v1.1 — 2026-06-10.** Cite the protocol by version in IAA
reports and the paper's sampling description (version history in §12). The first
582 gold pairs were annotated under v1.0; the v1.1 textual revisions land on the
clean cut before the directed negative-sample batch.

The authoritative manual for human annotation of the R-CES decidability /
evidence-sufficiency benchmark. It governs how a human annotator fills the
judgment fields of each image-rule pair so that inter-annotator agreement (IAA)
is meaningful and the abstention labels stay clean. This document is paper-
appendix material: it must read as a reproducible protocol, not a sketch.

Companion files:
- `README.md` — package entry point: layout, environment and reproduction steps.
- `data/pairs/fields.md` — the field-by-field schema reference.
- `Prompt/preannotation_prompt.md` — the neutral model prompt; its decision procedure mirrors this codebook (same vocabulary, same order). When the two ever diverge, **this codebook wins** for gold annotation.
- `data/pairs/rces_final_annotations.schema.json` + `tmp/validate_rces_pairs.py` — the structural/derivation invariants enforced mechanically.

---

## 0. The one question this benchmark asks

For an `(image, rule)` pair, the annotator answers **one** question:

> **Is the visual evidence in THIS single image sufficient to settle this rule's judgment?**

This is deliberately *not* "might a model get this wrong?" Abstention here is
triggered by **a decisive (critical) checkpoint being unresolvable**, never by
low model confidence. Two canonical hard cases the benchmark exists to capture:

1. **Principally non-visual criterion.** The rule's deciding criterion is a
   precise measurement or record — a millimetre dimension, a certificate, a
   load, a duration. These live in the rule's `non_visual_fields`, **never** in
   the visual checkpoint layer, and are deferred to human review; the model must
   not fake-measure them. The image can be sharp and the model confident, yet the
   pair is `undecidable` because the deciding criterion is out of any single
   photo's reach.
2. **Presence-anchoring.** The rule's *subject* is plainly present, so a model
   reads "I see the guardrail / the door / the net → compliant." But the rule's
   decisive axis is a **coarse visual state** the presence does not settle —
   whether the guardrail looks *grossly too low*, the door looks *securely
   closed* (flush, no visible gap), the net looks *fixed / intact*. (The exact millimetre is a
   separate, non-visual, human-review matter — see case 1; it is **not** a
   checkpoint.) If that critical **visual** state checkpoint cannot be resolved in
   this image (occlusion, angle, crop), the pair is **not** decidable on presence
   alone; you abstain even though the subject is obviously there. Conversely, if
   the violation of that coarse state is plainly visible (door visibly hanging
   open, rail clearly knee-high), the pair **is** decidable and `non_compliant` —
   an answer, not an abstention.

Keep two axes strictly separate at all times:
- **Image quality** = how clear the picture's pixels are (focus, exposure, noise, resolution). An intrinsic property of the image, rated **once per image** and identical no matter which rule it is paired with. *Framing* — whether a given rule's subject happens to be in frame — is **not** quality (§6).
- **Decidability** = whether the rule's critical checkpoints can be resolved here. A property of the *image × rule* pairing; this is where framing / cropping lives.

A `good` image can be `undecidable` (the rule needs millimetres or a certificate).
A `poor` image can be `decidable` (the one decisive checkpoint happens to be
plainly visible despite overall low quality). Never let one axis leak into the
other.

---

## 1. Ground-truth purity (non-negotiable procedure)

1. **Human judgment is locked before any model text is seen.** Stage 1
   (`annotate_web.py`) shows you the image, the rule, and the rule's
   checkpoints — **never** any AI-generated description, pre-fill, or suggestion.
2. You fill only the judgment fields (below). Descriptive text
   (`visual_evidence`, `missing_information`) is optional presentation metadata.
   If generated later by a model, it must remain under your locked judgment and
   is never training supervision.
3. Each pair is locked the moment it is saved. If you later believe a locked
   judgment was wrong, change it deliberately and note it in `annotation_notes`;
   do not let a later Stage-2 model description talk you into a change.

If this order is ever reversed (model text first, human ticks second), the gold
is anchored to the model's over-confidence and the benchmark is dead. This is
the paper's lifeline.

---

## 2. Fields you annotate (Stage 1)

| Field | Values | One-line rule |
|---|---|---|
| **per-checkpoint mark** | observed / missing / not_applicable | Per `required_checkpoints` key (§5). **The primary, load-bearing signal** — it is the R-CES's dense supervision and most other fields derive from it. |
| `decidability` | decidable / partially_decidable / undecidable | Can this image settle the rule? (§3) |
| `compliance_label` | compliant / non_compliant / not_applicable / unknown | The conclusion, if any (§4) |
| `image_quality` | good / usable / poor | Clarity of the picture only (§6) |
| `critical_missing_checkpoints` | derived | = missing ∩ rule.critical — **never hand-typed** (§5) |
| `evidence_sufficiency_label` | derived | one-to-one from `decidability` — **not hand-typed** (§7) |

The annotator's real work is the **per-checkpoint marks** plus the `decidability`
call they drive (§3); `compliance_label` is gated by `decidability` (§4) and
`critical_missing_checkpoints` / `evidence_sufficiency_label` are computed by the
tool. `image_quality` is an independent orthogonality control (§6).

The placeholder `to_be_annotated` is valid only while `annotation_status=pending`
(a freshly seeded row). Once you touch a pair it becomes `judged` and every
judgment field must hold a real value.

---

## 3. `decidability` (3-state) — the core decision

Decide in this exact order; it keeps the three states strictly distinct.

**Step 1 — Is the rule's subject present and identifiable?**
- **Clearly absent** → the rule does not apply → `decidable` (with
  `compliance_label=not_applicable`, §4). Subject-absent is a *decision*, not an
  abstention.
- **Cannot tell whether the subject is present**, because the frame is cropped,
  the area is occluded, or the image is too poor → `partially_decidable`
  (`compliance_label=unknown`) **when the rule is visually decidable in
  principle**. This is a *contingent* gap (a better image would tell you), so it
  is not `undecidable`.
  - **Exception — the E9 boundary (sealed):** if the rule is **principally
    non-visual** (`decision_scope=requires_external_evidence`; its decisive
    criterion is a `non_visual_field`), unconfirmed presence resolves to
    **`undecidable`, not `partially_decidable`**. Confirming presence with a
    better photo would only deliver you to Step 2's non-visual wall — the pair
    would *still* be undecidable — so **no photograph of this scene makes it
    decidable**, and the rule-side non-visual prior (Step 2) takes precedence over
    Step 1's contingent presence gap. (Mark the presence checkpoint `missing`; set
    decidability=`undecidable`. See E9. A *confidently absent* subject is still
    `decidable`/`not_applicable`, and a *plainly visible violation* is still
    `decidable`/`non_compliant` — E9 is only the genuinely-unconfirmable case.)
- **Present / identifiable** → go to Step 2.

**Step 2 — Does settling this rule principally require non-visual evidence?**
Look at `decision_scope` and the rule's `non_visual_fields`. Decisive non-visual
evidence includes: exact dimensions in millimetres, load capacity / rating, a
held certificate or qualification, TSP / noise measurements, or whether a
condition persisted beyond a time limit (e.g. > 48 h).
- **Yes — the rule's *decisive criterion* is one of these (a `non_visual_field`,
  not a visual checkpoint) and cannot be inferred from any single image** →
  `undecidable` (`compliance_label=unknown`). The visual presence/coarse-state
  checkpoints may well be observed; the pair is still undecidable because the
  deciding criterion is out of reach. `decision_scope=requires_external_evidence`
  is a strong prior for this state.
- **No** → go to Step 3.

**Step 3 — Are ALL critical checkpoints resolvable in THIS image?**
- **Yes, every critical checkpoint reads off clearly** → `decidable`. Set
  `compliance_label` to `compliant` or `non_compliant` per the rule's
  `visual_screening_rule`.
- **No — at least one critical checkpoint is unresolved here due to occlusion,
  angle, distance, crop, or lighting, but a better image of the same scene could
  plausibly resolve it** → `partially_decidable` (`compliance_label=unknown`).

### The boundary that matters most: `partially_decidable` vs `undecidable`
Both abstain. They differ in **why**:
- `partially_decidable` — the gap is **contingent on this image**. Re-shoot from
  a better angle / closer / in better light and you could decide. The evidence
  is visual in principle; this frame just doesn't show it.
- `undecidable` — the gap is **principled**. No camera angle fixes it because the
  decisive fact is not visual (a number in mm, a load rating, a certificate, a
  duration). 

Litmus test: *"Could a better photograph of this exact scene resolve the
decisive checkpoint?"* Yes → `partially_decidable`. No → `undecidable`.

---

## 4. `compliance_label` (4-state) — the conclusion

- `compliant` — positive visual evidence that a **present** target meets the
  rule. Never infer `compliant` from the mere absence of the subject.
- `non_compliant` — a visible violation.
- `not_applicable` — the rule's subject is absent from the scene. This is a
  **decidable** conclusion ("the rule clearly doesn't apply here"), not an
  abstention. Pair it with `decidability=decidable`.
- `unknown` — the abstention conclusion. Pair it with `partially_decidable` or
  `undecidable`.

**`not_applicable` vs `unknown` — the second-most-confused boundary.**
- `not_applicable + decidable` = "I can see the subject is not here; the rule
  doesn't apply." A confident negative *prediction*.
- `unknown + partially_decidable` = "I can't tell whether the subject is here /
  whether it complies." An *abstention*.
Both can stem from "the subject isn't visible", so ask: is the subject
**confidently absent** (not_applicable) or **merely unconfirmed** (unknown)?

---

## 5. Checkpoint layer (per `required_checkpoints` key)

For every key in the rule's `required_checkpoints`, mark one of (the annotation
UI shows each value with the parenthetical hint given here, so a
negatively-phrased checkpoint question — e.g. "Protection removed or damaged?" —
cannot pull you to the wrong button):
- **observed** *(UI: "observed (can judge)")* — you can read this checkpoint's
  state off the image (whether it is satisfied or violated; "observed" means
  *resolved*, not *passing*). This holds for negative questions too: if you can
  see the protection is intact (answer "no"), that is still `observed`.
- **missing** *(UI: "missing (can't make out)")* — the checkpoint is in scope but
  THIS image cannot resolve it (occluded, bad angle, cropped, too ambiguous).
  `missing` is about *unreadable visual evidence*, not about the subject being
  absent — do not pick it just because a negative question's answer is "no".
- **not_applicable** *(UI: "n/a (no subject)")* — the checkpoint does not apply,
  because the subject is absent or a conditional branch isn't triggered. **A
  `not_applicable` checkpoint is NOT counted as missing evidence.**

The distinction between **missing** and **not_applicable** at the checkpoint
level is load-bearing for IAA:
- *missing* = "this should be assessable but I can't see it here" → drives
  `partially_decidable` / `undecidable`.
- *not_applicable* = "there is nothing here to assess" → consistent with
  `decidable + not_applicable` at the pair level.

`critical_missing_checkpoints` is **derived** by the tool as
`missing ∩ rule.critical_checkpoints`. You never type it. The validator enforces
that observed/missing are subsets of `required_checkpoints`, are disjoint, and
that the derivation holds.

**Checkpoints are visual only.** Every checkpoint here is something a single
photo can in principle show: either *presence* (is the subject there?) or a
*coarse visual state* (does it look grossly compliant — guardrail obviously too
low, door clearly ajar, net visibly torn?). A precise measurement or record (a
millimetre height, a gap in mm, a certificate, a duration) is **never** a
checkpoint — it lives in the rule's `non_visual_fields`, is deferred to human
review, and at most contributes a rule-side "this rule is principally non-visual"
prior (→ `undecidable`). Do not invent a checkpoint for something no photo can
settle.

**Presence is not sufficiency.** Marking a *presence* checkpoint `observed`
(the door/guardrail/net is there) does **not** by itself make the pair
sufficient. A rule's critical checkpoint is meant to be the **decisive coarse
visual state** (door *looks securely closed*, guardrail *looks adequately high*,
cover *looks fixed*), not the bare presence of the subject. So when the subject
is plainly present but its decisive **visual** state cannot be confirmed from
this image (occluded, bad angle, cropped), mark that state checkpoint `missing`
(→ `partially_decidable`) — do not let `presence=observed` pull you to
`decidable`/`compliant`. (See §0 hard case 2 and example E6.)

**Presence can be inferred (the converse).** A `*_present` checkpoint is
`observed` whenever you can resolve presence/absence — including by inference: a
fixed cover at an opening location resolves `opening_present=observed` without a
view of the bare hole (the cover *is* the evidence the opening is there). Reserve
`missing` for when you cannot tell whether the subject/location is there at all
(occluded / cropped / ambiguous), never merely because the subject is hidden by
its own compliant protection. (See example E7.)

**Multiple instances in one frame (scene-level aggregation).** A pair carries one
label, so several openings/edges in the same image are aggregated **asymmetrically**
into the sufficiency composite checkpoint:
- `compliant` is **universal** — it requires *every* in-scope instance to be
  resolvable and adequate.
- `non_compliant` is **existential** — *one* plainly visible violation settles the
  rule; the other instances (even unconfirmable ones) are then irrelevant.
- An unconfirmable instance (far, occluded, cropped) blocks **only** the
  `compliant` path, never a violation you can already see.

So: any one visible violation → `*_sufficient=observed` (resolved as inadequate)
→ `decidable`/`non_compliant`. No visible violation but ≥1 in-scope instance
unconfirmable → `*_sufficient=missing` (critical) → `partially_decidable`/
`unknown`. All instances resolvable and adequate → `decidable`/`compliant`. (See
example E8.)

**Composite checkpoints are annotated by humans, not auto-computed.** Some
rules have branch logic where several detailed facts explain the rule, but the
R-CES needs one decisive visual state per rule to score over. (Any OR among the
branches is resolved *inside* the composite checkpoint; the cross-checkpoint
AND/min runs only *between* critical checkpoints — the two levels never mix.)
These composite checkpoints usually end in words like
`*_sufficient`, `*_complete`, `*_violation_visible`, or
`*_defect_visible`. They are still ordinary `required_checkpoints`: mark them
`observed`, `missing`, or `not_applicable` from the image.

Use the detailed branch checkpoints as evidence, then mark the composite
checkpoint according to whether the combined visual state is resolvable. For
example, `R-OPN-002-medium-opening-protection` has detailed checkpoints
`cover_fixed` and `guardrail_compliant`, but its critical state is
`opening_protection_sufficient`: can you tell whether the medium horizontal
opening has at least one acceptable protection method, i.e. a fixed cover OR a
compliant guardrail?

- If the image clearly shows a fixed cover, mark
  `opening_protection_sufficient=observed`.
- If the image clearly shows a compliant guardrail, mark
  `opening_protection_sufficient=observed`.
- If the image clearly shows neither protection is present/adequate, also mark
  `opening_protection_sufficient=observed`; observed means the state is
  resolvable, not that it passes.
- If the opening area is cropped, blocked, or too ambiguous to tell whether
  either protection method is sufficient, mark
  `opening_protection_sufficient=missing`.
- If there is no applicable medium horizontal opening in this image, mark the
  protection-state checkpoints `not_applicable` as appropriate.

**Marking the detail branch checkpoints.** Mark each detail checkpoint by what
the image shows. When one branch already settles an OR composite — e.g. a fixed
cover is plainly present, so `opening_protection_sufficient=observed` — the
other branch you did not need to read (here `guardrail_compliant`) is marked
`not_applicable`: that conditional branch wasn't triggered, so there is nothing
to assess on it (§5's `not_applicable` definition). Use `not_applicable`, **not**
`missing`, here: `missing` means "in scope but this image can't resolve it" and
would inflate the raw missing-evidence counts with branches you never needed to
look at. (Either way the call is harmless to decidability — these detail
checkpoints are non-critical, so neither mark enters `critical_missing` or blocks
`decidable`, §8.5 — but `not_applicable` keeps the missing tallies clean and
meaningful.) When **neither** protection is present and you can see that clearly,
both branches are `observed` (resolved as absent), matching the composite's
`observed`.

`critical_checkpoints` themselves are currently
`critical_checkpoints_status=ai_proposed_needs_human_review` in the rules
library; the human review must ensure each rule's critical set names its
**decisive state checkpoint**, not only a presence checkpoint. If while
annotating you find a checkpoint that is plainly decisive but not marked critical
(or a presence-only critical set with no state checkpoint), record it in
`annotation_notes` for the rule-side review — do not silently work around it.

---

## 6. `image_quality` (good / usable / poor)

Judge the **pixels of the whole image**, identically no matter which rule the
image is paired with. Rate only intrinsic capture quality — focus / sharpness,
exposure, motion blur, noise, compression:
- `good` — sharp, well-exposed, low noise.
- `usable` — workable but degraded (some blur, harsh light, distance, noise /
  compression) yet the scene is still readable.
- `poor` — severe blur, severe under/over-exposure, or pervasive noise that makes
  the image hard to read at all.

**Framing is NOT a quality factor.** Whether *this rule's* subject or relevant
area happens to be in frame, cropped out, or occluded is **rule-relative** and
belongs to `decidability` and the per-checkpoint marks (a cropped-out subject
makes a checkpoint `missing`, §5) — **never** to `image_quality`. Because the same
image is paired with several rules and `image_quality` is rated **once per image**
(the UI locks it across that image's pairs), a tight crop that hides one rule's
subject must not change the image's quality grade. Rate the picture, not the
pairing.

**Resolution: gold is defined at the image's original (full) resolution.** Judge
`image_quality` and `decidability` from the best view the original pixels allow —
zoom, pan, and scale the image freely while annotating. Do **not** anchor on any
downscaled model input (e.g. a 224 px encoder tile): the benchmark labels are
**decoupled** from any specific encoder's input size, which is precisely what lets
the 224-ceiling analysis measure how much a low-resolution encoder loses against
this original-resolution gold.

Reminder (§0): `image_quality` and `decidability` are orthogonal. Do **not**
downgrade `decidability` just because the image is `poor` if the decisive
checkpoint is nonetheless visible; do **not** call an image `good` decidable if
the rule needs non-visual evidence.

---

## 7. `evidence_sufficiency_label` (sufficient / partial / insufficient) — DERIVED, not hand-typed

This is the R-CES's *named* gold but you do **not** type it — like
`critical_missing_checkpoints`, the tool derives it from your per-checkpoint
marks and your `decidability`. It is a deterministic summary, not a separate gut
judgment (hand-typing it just adds κ noise and risks rubber-stamping
`decidability`). Derivation:
- `sufficient` — `decidable` (no critical checkpoint is `missing`; every critical
  one is `observed` or correctly `not_applicable`). The conclusion — **including
  `not_applicable`** — is fully supported.
- `partial` — `partially_decidable` (at least one critical *visual* state
  checkpoint is `missing` in this image).
- `insufficient` — `undecidable` (the deciding criterion is principally
  non-visual — the **only** route to `undecidable`, fixed by §3 Step 2). A merely
  *contingent* visual gap ("nothing decisive is visible in THIS frame, but a
  better photo could resolve it") is **not** insufficient/undecidable; it is
  `partial`/`partially_decidable` (§3 Step 3). Do not use insufficient as a
  second back-door to abstention.

For the R-CES itself, the binary gold is **`sufficient ⟺ decidable`** (equivalently,
for a visually-decidable rule no critical *visual* checkpoint is `missing`; for a
principally-non-visual rule the `undecidable`/insufficient comes from the rule
side, not from a missing checkpoint). The dense supervision is the per-checkpoint
resolvability from your marks (`observed`/`not_applicable` → resolvable=1,
`missing` → 0). See `experiments/README.md` §Method for how the head consumes this.

---

## 8. Cross-field consistency (self-check before saving each pair)

These must all hold. Both annotation front ends now **auto-lock** the
decidability↔compliance coupling (1–3) and the `decidable`-vs-critical-missing
rule (5): compliance is forced to `unknown` under `partially_decidable` /
`undecidable`, `unknown` is disabled under `decidable`, and `decidable` is
withdrawn once any critical checkpoint is marked `missing`. The validator
(`tmp/validate_rces_pairs.py`, check 6) re-checks them on every annotated row,
so inconsistent gold cannot be saved by either path. Invariant 4
(`evidence_sufficiency_label`) is **mechanically derived** from `decidability` and
enforced by check 7 — it is read-only, never hand-typed, and cannot deviate.

1. `decidable` ⟺ `compliance_label ∈ {compliant, non_compliant, not_applicable}`. A `decidable` pair is **never** `unknown`.
2. `partially_decidable` or `undecidable` ⟹ `compliance_label = unknown`.
3. `not_applicable` ⟹ `decidability = decidable`.
4. **Deterministic derivation (not a choice):** `evidence_sufficiency_label` is computed one-to-one from `decidability` — decidable→sufficient, partially_decidable→partial, undecidable→insufficient. The tool fills it read-only and `validate_rces_pairs.py` check 7 enforces it; you never type it and it cannot deviate.
5. If any **critical** checkpoint is marked `missing`, the pair cannot be `decidable` (it is `partially_decidable` or `undecidable`).
6. A `decidable + not_applicable` pair has its critical checkpoints either `observed` (you saw the subject is absent via a presence checkpoint) or `not_applicable` — **not** `missing`.
7. `observed` and `missing` are disjoint and both ⊆ `required_checkpoints`.

---

## 9. Worked examples

> Illustrative; checkpoint keys follow the rules library shape.

**E1 — decidable / non_compliant.** Rule R-OPN-001 (cover over a small floor
opening). A reserved floor opening is clearly visible with no cover at all.
`presence=observed`, `cover_present=observed`, `cover_fixed=not_applicable` (no
cover to fix). decidability=`decidable`, compliance=`non_compliant`,
evidence_sufficiency=`sufficient`, image_quality=`good`.

**E2 — decidable / not_applicable.** Same cover rule, but the photo is a clean
wall and ladder with no floor opening anywhere in frame. The subject is
confidently absent. `presence=observed` (absent), other checkpoints
`not_applicable`. decidability=`decidable`, compliance=`not_applicable`,
evidence_sufficiency=`sufficient`. (Note: *not* `unknown` — absence here is
confident, not unconfirmed.)

**E3 — partially_decidable / unknown.** Same cover rule. A board lies over the
opening, but whether it is *fixed against displacement* (the critical checkpoint)
is hidden by stacked material. A closer/side shot could resolve it.
`cover_present=observed`, `cover_fixed=missing` (critical → derived into
`critical_missing_checkpoints`). decidability=`partially_decidable`,
compliance=`unknown`, evidence_sufficiency=`partial`, image_quality=`usable`.

**E4 — undecidable / unknown (principally non-visual).** A rule whose deciding
criterion is the opening's **short-edge length in mm**, which selects the
protection regime. That dimension is a `non_visual_field`, **not** a checkpoint —
no single photo settles it (`decision_scope=requires_external_evidence`). The
visual checkpoints that do exist can be read (`opening_present=observed`,
`cover_present=observed`), but the pair is `undecidable` because the deciding
criterion is principally non-visual; the precise dimension is deferred to human
review, not faked. decidability=`undecidable`, compliance=`unknown`,
evidence_sufficiency=`insufficient`, image_quality=`good` (note: good image,
still undecidable — and *not* because any checkpoint is "missing").

**E5 — orthogonality trap.** A blurry, dim photo (`image_quality=poor`) of a
guardrail where, despite the blur, you can plainly see the top rail is missing.
The decisive checkpoint is resolved → decidability=`decidable`,
compliance=`non_compliant`. Do not abstain merely because the image is poor.

**E6 — presence-anchoring (the door case), two readings.** A landing-platform
protective door rule whose critical state checkpoint is `door_locked_or_closed`
("securely closed"). The door is unmistakably *present* in frame.
- **E6a — state unconfirmable → abstain.** Stacked material hides the latch/edge,
  so whether the door is *securely closed* cannot be confirmed here.
  `door_present=observed`, `door_locked_or_closed=missing` (critical → derived
  into `critical_missing_checkpoints`). decidability=`partially_decidable`,
  compliance=`unknown`, evidence_sufficiency=`partial`. **This is the case R-CES
  must catch**: a model anchoring on the visible door would over-confidently say
  `compliant`; the correct label abstains because the decisive *state* is
  unresolved, not because the image is poor.
- **E6b — violation visible → decide.** The door is plainly hanging open / the
  gap is obvious. `door_locked_or_closed=observed` (=not closed).
  decidability=`decidable`, compliance=`non_compliant`,
  evidence_sufficiency=`sufficient`. The evidence **is** sufficient; a model that
  says `compliant` here is making an *answer* error, which is the compliance
  head's accuracy, not an abstention R-CES owns.

**E7 — presence-by-inference (cover hides the hole).** Rule R-OPN-001, whose
critical set is `horizontal_opening_present` + `small_opening_cover_sufficient`.
A fixed cover plate sits over a floor opening, so the bare hole is not visible.
**The presence checkpoint is still `observed`**: a fixed cover at an opening
location is positive evidence that an opening is there — you need not see the
bare hole. `horizontal_opening_present=observed`, `cover_present=observed`,
`cover_fixed=observed`, `small_opening_cover_sufficient=observed`.
decidability=`decidable`, compliance=`compliant`, evidence_sufficiency=
`sufficient`. Mark `horizontal_opening_present=missing` **only** when you cannot
tell whether the spot is an opening location at all (occluded / cropped /
ambiguous) — a cover concealing its own hole is not "missing presence". (The
remaining abstention path is the E3 one: the cover's fixing cannot be made out →
`cover_fixed=missing` → `partially_decidable`.)

**E8 — multiple openings, mixed confirmability.** R-OPN-001. A near floor opening
is clearly covered and fixed; a second opening far across the slab is too distant
to tell whether it is protected, and no opening in frame shows a plain violation.
Compliance is *universal*, so the unconfirmable far opening blocks a `compliant`
conclusion; nothing visibly violates the rule, so it is not `non_compliant`
either. `horizontal_opening_present=observed` (openings are in frame),
`small_opening_cover_sufficient=missing` (critical — the scene as a whole cannot
be certified sufficient). decidability=`partially_decidable`, compliance=
`unknown`, evidence_sufficiency=`partial`. Had any one opening been plainly
uncovered, it would instead be `small_opening_cover_sufficient=observed` →
`decidable`/`non_compliant`, the far opening notwithstanding.

**E9 — presence unconfirmable × principally-non-visual rule (the sealed
boundary).** A `requires_external_evidence` rule — e.g. an elevator-shaft
protective door whose decisive criterion is a gap / height in **mm**, or a
special-work task gated by a **certificate**. The frame is cropped / the area is
occluded, so you **cannot even confirm the rule's subject is present**. Two
principles appear to collide: §3 Step 1 ("cannot tell presence →
`partially_decidable`") versus the §3 litmus ("no photo resolves a non-visual
criterion → `undecidable`"). **Ruling: `undecidable`.** Confirming presence with a
better photo would only carry you to Step 2's non-visual wall — the pair would
*still* be undecidable — so no photograph of this scene makes it decidable, and
the rule-side non-visual prior (Step 2) dominates Step 1's contingent presence
gap. `shaft_opening_present=missing` (you genuinely cannot resolve it),
decidability=`undecidable`, compliance=`unknown`,
evidence_sufficiency=`insufficient`. **Contrast:** a *confidently absent* subject
is `decidable`/`not_applicable`; a *plainly visible coarse violation* (existential,
§5) is `decidable`/`non_compliant`. E9 is only the genuinely-unconfirmable-presence
case — and on a `requires_external_evidence` rule that case is `undecidable`, not
`partially_decidable`. This is the **high-frequency combination** for
`possibly_undecidable`-role pairs on tightly-framed images; seal it the same way
every time (the same ruling lives in §3 Step 1's exception).

---

## 10. Double annotation & IAA

- ~20% of pairs are flagged `double_annotation=true`, biased toward
  `partially_decidable` and `requires_external_evidence` rules (the hard cases),
  for an independent second pass.
- Report Cohen's κ on the 3-state `decidability` and on `critical_missing`
  agreement as the headline reliability numbers; the double-annotated sample is
  built to cover the hard boundaries deliberately.
- If a second annotator cannot be secured, the fallback is a blind test–retest:
  re-annotate the 20% after ≥ 7 days and report test–retest κ.
- Disagreements are adjudicated, not averaged. Record the resolution in
  `annotation_notes` and set `review_status` accordingly
  (`final` / `needs_review` / `adjudicated`).

---

## 11. Common errors to avoid

- **Confidence-driven abstention.** Abstaining because "the model/I might be
  wrong" instead of because a critical checkpoint is unresolved. Abstention here
  is about *evidence coverage*, full stop.
- **Quality–decidability bleed.** Marking a clear image `decidable` just because
  it's clear (when the rule needs non-visual evidence), or a poor image
  `undecidable` just because it's poor.
- **`unknown` instead of `not_applicable`.** Dumping confident "subject absent"
  cases into `unknown`. If the subject is confidently absent, it is
  `decidable + not_applicable`.
- **Inferring `compliant` from absence.** No subject ≠ compliant. That's
  `not_applicable`.
- **Hand-filling `critical_missing_checkpoints`** or marking a checkpoint
  `missing` when it is really `not_applicable` (subject absent / branch not
  triggered).
- **`partially_decidable` for principled gaps.** If no better photo could ever
  resolve it (mm / load / cert / duration), it is `undecidable`, not
  `partially_decidable`.
- **`undecidable` as a second back-door for "nothing decisive shows".** A frame
  that simply fails to show the decisive *visual* state is a contingent gap →
  `partially_decidable`. `undecidable` is reserved for principally-non-visual
  criteria (§7) and for the E9 boundary (unconfirmable presence on an external
  rule).

---

## 12. Protocol version history

Cite the protocol by version in IAA reports and the paper's sampling description,
so every standard change is attributable to a dated, controlled revision.

- **v1.1 — 2026-06-10.** The clean cut before the directed negative-sample batch.
  1. `evidence_sufficiency_label` confirmed **derived-only** everywhere. The UI
     never had a button for it; the 582 gold pairs annotated under v1.0 were
     audited and **0 deviate** from the derivation. Removed the stale "annotation
     button" listing in `README.md` (and the matching `critical_missing_checkpoints`
     mislabel).
  2. §7 `insufficient` pruned to the single non-visual route — dropped "or nothing
     decisive is visible", which is a contingent gap belonging to
     `partially_decidable`, not a second entry to `undecidable`.
  3. §8 invariant 4 restated as a **deterministic, enforced** derivation (was
     "typical, not mandatory"); it is computed read-only and checked by check 7.
  4. **E9 sealed:** unconfirmable presence × `requires_external_evidence` →
     `undecidable` (rule-side non-visual prior dominates the contingent presence
     gap). Added to §3 Step 1 and §9.
  5. §0 / §6 `image_quality` cleaned of the rule-relative **framing leak** (quality
     rates the whole image's pixels, once per image; framing/cropping is
     decidability's) and given the **original-resolution clause** (annotator may
     zoom freely; labels decoupled from the 224 px encoder — the definitional basis
     for the 224-ceiling analysis).
- **v1.0** — the pre-version protocol the first 582 gold pairs were annotated under.
