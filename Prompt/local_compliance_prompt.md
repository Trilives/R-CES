You are screening ONE construction-site image against ONE safety rule, using only what is visible in this image.

Choose exactly one option:
A = compliant: the rule's subject IS present, and the visible evidence shows it MEETS the rule.
B = non_compliant: a violation of the rule is visibly present.
C = not_applicable: the rule's subject is not present in this scene at all.
D = unknown: the subject may be present but a decisive checkpoint cannot be resolved from THIS image (occlusion, angle, distance, crop, blur), OR the rule needs non-visual evidence (exact millimetres, load capacity, a certificate, records, TSP/noise).

Decide in this order:
1. Is the rule's subject present? Clearly absent -> C. Cannot even tell whether it is present -> D.
2. Does the conclusion require decisive non-visual evidence (mm, load, certificate, records, TSP/noise)? Yes -> D.
3. Otherwise judge from the image: every critical checkpoint is visible and the target MEETS the rule -> A; a critical checkpoint is visible and the target VIOLATES the rule -> B; a critical checkpoint cannot be resolved here -> D.

Calibration:
- Do NOT force A or B when a critical checkpoint is genuinely unobservable.
- But do NOT default to D when the visible evidence already settles it — when the scene is clear, commit to A, B, or C.
- "compliant" needs POSITIVE visible evidence that a present target meets the rule; never infer compliant merely from the subject's absence (that is C).

Rule:
<<RULE_JSON>>

Answer with one letter (A, B, C, or D):
