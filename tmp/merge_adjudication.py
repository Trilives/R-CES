"""Fold the web-adjudicated disagreement decisions back into the gold file.

After Script/adjudicate_web.py has written one resolved record per
disagreeing pair to rces_adjudication.jsonl, this step applies those decisions
to the first-pass gold (rces_pilot_pairs.jsonl) to form the final gold:

  - For each adjudicated pair, overwrite the gold row's judgment fields with the
    adjudicated values and mark review_status=adjudicated (with annotation_notes
    recording the choice). The transient `adjudication_choice` key is dropped.
  - image_quality is a per-IMAGE property: when a decision changes it, the new
    value is propagated to every other gold pair of the same image so the image
    stays internally consistent (the same rule annotate_web.py enforces live).

No model judgment is involved -- adjudication is a human pass, so the gold's
"no model writes a judgment field" invariant is preserved. Resumable / idempotent:
re-running applies the same decisions again with no further change.

    uv run python tmp/merge_adjudication.py            # apply
    uv run python tmp/merge_adjudication.py --dry-run  # report only
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
import rces_io as rio  # noqa: E402

ADJUDICATION_PATH = rio.PAIRS_DIR / "rces_adjudication.jsonl"

# Fields an adjudication decision is allowed to write back into gold.
JUDGMENT_FIELDS = (
    "decidability", "compliance_label", "image_quality", "evidence_sufficiency_label",
    "observed_checkpoints", "missing_checkpoints", "critical_missing_checkpoints",
)
PROVENANCE_FIELDS = ("annotator_id", "annotation_seconds", "review_status", "annotation_notes")


def main() -> None:
    dry_run = "--dry-run" in sys.argv[1:]

    decisions = {r["pair_id"]: r for r in rio.load_jsonl(ADJUDICATION_PATH)}
    if not decisions:
        print(f"No adjudication decisions at {ADJUDICATION_PATH.relative_to(rio.PROJECT_ROOT)} — nothing to merge.")
        return

    gold = rio.load_jsonl(rio.GOLD_PATH)
    gold_by_id = {r["pair_id"]: r for r in gold}

    missing = sorted(set(decisions) - set(gold_by_id))
    if missing:
        raise SystemExit(f"Adjudicated pair_ids absent from gold: {missing}")

    changed_fields = 0
    quality_propagated = 0
    by_image: dict[str, list[dict]] = {}
    for row in gold:
        by_image.setdefault(row["image_id"], []).append(row)

    for pid, dec in decisions.items():
        row = gold_by_id[pid]
        before = {f: row.get(f) for f in JUDGMENT_FIELDS}
        for f in JUDGMENT_FIELDS:
            row[f] = dec.get(f)
        for f in PROVENANCE_FIELDS:
            if f in dec:
                row[f] = dec[f]
        row.pop("adjudication_choice", None)  # transient UI bookkeeping, not gold
        diff = [f for f in JUDGMENT_FIELDS if before[f] != row.get(f)]
        if diff:
            changed_fields += 1
        # image_quality is a per-IMAGE property: the adjudicated value is
        # authoritative for the whole image, so force every other pair of the
        # same image to it (unconditionally, not just when this pair's value
        # changed): image quality is image-level, so the adjudicated value wins.
        new_q = row["image_quality"]
        for sib in by_image[row["image_id"]]:
            if sib["pair_id"] != pid and sib.get("image_quality") != new_q:
                sib["image_quality"] = new_q
                quality_propagated += 1
        note = dec.get("annotation_notes", "")
        marker = "" if diff else "  (no judgment change — confirmed)"
        print(f"  {pid}: {note}{marker}")

    print("-" * 60)
    print(f"adjudicated pairs        : {len(decisions)}")
    print(f"  with judgment changes  : {changed_fields}")
    print(f"  image_quality propagated to siblings: {quality_propagated}")

    if dry_run:
        print("\n[dry-run] gold NOT written. Re-run without --dry-run to apply.")
        return

    rio.write_jsonl(rio.GOLD_PATH, gold)
    print(f"\nWrote final gold -> {rio.GOLD_PATH.relative_to(rio.PROJECT_ROOT)}")
    print("Next: uv run python tmp/validate_rces_pairs.py")


if __name__ == "__main__":
    main()
