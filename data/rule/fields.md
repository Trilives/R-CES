# R-CES Rule Fields

This file documents the fields in `data/rule/rule.json`, the self-contained
English rule library this benchmark conditions on. Each rule is reduced to the
fields the R-CES line actually consumes; auxiliary normative fields used by other
compliance pipelines (e.g. `priority`, `evidence_types`, `thresholds`,
`machine_rule`, `source_quote`, `rectification_advice`, `normative_rule`,
`non_visual_fields`) are intentionally omitted from this projection.

## Field Reference

| Field | Purpose |
|---|---|
| `rule_id` | Stable unique identifier for the rule. Pair files and annotations refer to rules through this value, so it should not be renamed casually. |
| `major_category` | Top-level hazard family, such as opening hazards, edge hazards, unsafe behavior, or civilized-construction hazards. Used for coverage analysis and coarse grouping, not as an annotation result. |
| `subcategory` | More specific scene or hazard object group within the major category. Prefer object-oriented wording, such as floor reserved opening or fire access equipment, rather than violation-status wording. |
| `rule_name` | Human-readable rule title. It describes the normative concern, but should not be treated as image-retrieval text by itself. |
| `visual_checkpoints` | Named visual observations that an annotator or model may check from an image. These should describe observable facts, not final compliance conclusions. |
| `required_checkpoints` | Checkpoints that are normally needed to evaluate the rule. They define the expected evidence surface for annotation and review. |
| `critical_checkpoints` | The minimal or high-impact checkpoints most important for applying the rule. Missing or uncertain critical checkpoints often make the rule hard to decide from one image. |
| `critical_checkpoints_status` | Provenance or review status for `critical_checkpoints`, such as whether they were AI-proposed and still need human review. |
| `visual_screening_rule` | Single-image screening expression. It should use only facts that can reasonably be seen or weakly inferred from one image. It may be less complete than the shared-library `normative_rule`. |
| `decision_scope` | Declares how far a single image can support the rule. Current values distinguish visual screening, normative decisions available from the image, and rules requiring external evidence. |
| `visual_retrieval_text` | Scene-anchor text for image-rule similarity retrieval with SigLIP2 or RAG-style candidate generation. It should name visible construction-site objects, protective elements, PPE, passages, openings, edges, equipment, and coarse visual states. It must not encode the final compliance decision, exact thresholds, or external-evidence requirements. |

## Notes

- `visual_retrieval_text` is for candidate recall only. It is not annotation
  gold, not a weak label, and not a substitute for `visual_checkpoints`.
- If `visual_retrieval_text` is regenerated, verify it stays scene-anchor
  oriented. A bad pattern is text dominated by rule names, violation wording,
  exact thresholds, or `decision_scope`; a good pattern is text dominated by
  visible scene nouns such as cover plate, guardrail, safety net, PPE, passage,
  opening, edge, fire extinguisher, cable, machinery, or signage.
- Name only objects that are *present and visible* in the image. Avoid
  absent-object phrasing — "missing X", "person without harness", "head without
  helmet", "removed cover" — because an absence fits an enormous range of
  unrelated images and hurts retrieval precision. Describe the present subject
  instead (the head, the edge, the fire box), keeping both the violation-state
  and the compliant-state present-object terms so compliant scenes still
  retrieve (e.g. `floor opening, opening cover plate`, not `uncovered opening`).
