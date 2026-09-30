#!/usr/bin/env python3
"""Validate the R-CES pilot two-layer checkpoint structure.

Checks the rule definition layer (data/rule/rule.json) against the
observation layer (data/pairs/rces_pilot_pairs.jsonl) and guards the
ground-truth purity invariants that the paper depends on:

  1. Every pair's rule_id exists in rule.json.
  2. observed_checkpoints and missing_checkpoints are subsets of the rule's
     required_checkpoints, and the two are disjoint.
  3. critical_missing_checkpoints == missing_checkpoints INTERSECT
     rule.critical_checkpoints. (In the pending stage all three are empty, so
     this holds trivially; it becomes meaningful once a pair is annotated.)
  4. WARNING if any gold judgment field is byte-for-byte equal to the
     isolated model baseline (rces_qwen_full_baseline.jsonl, source=model_full)
     value for the same pair_id while annotation_status is still "pending"
     (guards against model baseline values leaking back into gold).
  5. required_checkpoints equals the key set of visual_checkpoints, per rule.
  6. For annotated rows, the codebook §8 cross-field invariants hold:
     decidable <-> compliance coupling (§8.1-8.3) and no `decidable` while a
     critical checkpoint is missing (§8.5). Pending placeholders are skipped.
  7. For annotated gold rows, evidence_sufficiency_label equals the value
     DERIVED from decidability (codebook §7: decidable->sufficient,
     partially_decidable->partial, undecidable->insufficient).

Exit code is non-zero if any ERROR is found; WARNINGs do not fail the run.

Run:  uv run python tmp/validate_rces_pairs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RCES_ROOT = Path(__file__).resolve().parents[1]  # tmp/<file> -> package root
sys.path.insert(0, str(RCES_ROOT / "src"))
import rces_io as rio  # noqa: E402  (codebook §8 cross-field invariants)
RULES_PATH = RCES_ROOT / "data" / "rule" / "rule.json"
PAIRS_PATH = RCES_ROOT / "data" / "pairs" / "rces_pilot_pairs.jsonl"
# Isolated model full-pipeline baseline (source=model_full). Cross-checked only to
# WARN if a gold value equals its baseline twin while still pending (leak guard).
SIDECAR_PATH = RCES_ROOT / "data" / "pairs" / "rces_qwen_full_baseline.jsonl"

# Gold judgment fields whose value, if it matches the preannotation verbatim
# while still pending, indicates a possible model->gold leak.
GOLD_VS_PREANNOTATION = [
    "decidability",
    "compliance_label",
    "evidence_sufficiency_label",
]


def load_rules(path: Path) -> dict[str, dict]:
    rules = json.loads(path.read_text(encoding="utf-8"))
    return {r["rule_id"]: r for r in rules}


def load_pairs(path: Path) -> list[dict]:
    rows = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise SystemExit(f"ERROR: pair line {ln} is not valid JSON: {exc}")
    return rows


def load_sidecar(path: Path) -> dict[str, dict]:
    """Load the isolated model baseline sidecar, keyed by pair_id."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["pair_id"]] = d
    return out


def main() -> int:
    # Default validates gold; pass a path to validate another pair file with the
    # same checks (e.g. the crop-augmentation file rces_crop_augmentation.jsonl).
    pairs_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else PAIRS_PATH
    rules = load_rules(RULES_PATH)
    pairs = load_pairs(pairs_path)
    sidecar = load_sidecar(SIDECAR_PATH)

    errors: list[str] = []
    warnings: list[str] = []

    # ---- Check 5: required_checkpoints == keys(visual_checkpoints) ----
    rules_checked = 0
    for rid, rule in rules.items():
        required = set(rule.get("required_checkpoints", []))
        vc_keys = set(rule.get("visual_checkpoints", {}).keys())
        if required != vc_keys:
            errors.append(
                f"[rule {rid}] required_checkpoints {sorted(required)} != "
                f"visual_checkpoints keys {sorted(vc_keys)}"
            )
        critical = set(rule.get("critical_checkpoints", []))
        if not critical <= required:
            errors.append(
                f"[rule {rid}] critical_checkpoints not a subset of "
                f"required_checkpoints: extra {sorted(critical - required)}"
            )
        rules_checked += 1

    # ---- Pair-level checks ----
    leak_warns = 0
    for row in pairs:
        pid = row.get("pair_id", "<no pair_id>")
        rid = row.get("rule_id")

        # Check 1: rule_id exists
        rule = rules.get(rid)
        if rule is None:
            errors.append(f"[pair {pid}] rule_id {rid!r} not found in rule.json")
            continue

        required = set(rule.get("required_checkpoints", []))
        critical = set(rule.get("critical_checkpoints", []))
        observed = row.get("observed_checkpoints", []) or []
        missing = row.get("missing_checkpoints", []) or []
        crit_missing = row.get("critical_missing_checkpoints", []) or []

        # Check 2: subset + disjoint
        obs_extra = set(observed) - required
        mis_extra = set(missing) - required
        if obs_extra:
            errors.append(
                f"[pair {pid}] observed_checkpoints not in rule.required_checkpoints: "
                f"{sorted(obs_extra)}"
            )
        if mis_extra:
            errors.append(
                f"[pair {pid}] missing_checkpoints not in rule.required_checkpoints: "
                f"{sorted(mis_extra)}"
            )
        overlap = set(observed) & set(missing)
        if overlap:
            errors.append(
                f"[pair {pid}] observed_checkpoints and missing_checkpoints overlap: "
                f"{sorted(overlap)}"
            )

        # Check 3: critical_missing == missing INTERSECT critical
        expected_crit_missing = set(missing) & critical
        if set(crit_missing) != expected_crit_missing:
            errors.append(
                f"[pair {pid}] critical_missing_checkpoints {sorted(crit_missing)} != "
                f"missing INTERSECT rule.critical {sorted(expected_crit_missing)}"
            )

        # Check 6: codebook §8 cross-field invariants on annotated rows
        # (decidability <-> compliance coupling, and §8.5 decidable-vs-critical-missing).
        if row.get("annotation_status") in ("judged", "described", "reviewed", "model_complete"):
            for msg in rio.consistency_errors(
                row.get("decidability", ""), row.get("compliance_label", ""), list(crit_missing)
            ):
                errors.append(f"[pair {pid}] {msg}")

        # Check 7: evidence_sufficiency_label is DERIVED from decidability
        # (codebook §7), never hand-typed. Enforced on annotated GOLD rows; the
        # isolated baseline (model_full) is left as the model produced it.
        dec = row.get("decidability", "")
        if (
            row.get("source") == rio.SOURCE_HUMAN
            and row.get("annotation_status") in ("judged", "described", "reviewed")
            and dec != rio.PLACEHOLDER
        ):
            want = rio.derive_evidence_sufficiency(dec)
            got = row.get("evidence_sufficiency_label", "")
            if got != want:
                errors.append(
                    f"[pair {pid}] evidence_sufficiency_label {got!r} != derived {want!r} "
                    f"from decidability={dec} (codebook §7)"
                )

        # Check 4: model->gold leak guard (only while pending), against sidecar
        if row.get("annotation_status") == "pending":
            pre = sidecar.get(pid, {}) or {}
            for field in GOLD_VS_PREANNOTATION:
                gold_val = row.get(field)
                pre_val = pre.get(field)
                # A pending gold field must still be the placeholder; if it is
                # instead the verbatim preannotation value, flag a likely leak.
                if (
                    gold_val != "to_be_annotated"
                    and pre_val is not None
                    and gold_val == pre_val
                ):
                    warnings.append(
                        f"[pair {pid}] gold {field}={gold_val!r} equals preannotation "
                        f"value while status=pending (possible model->gold leak)"
                    )
                    leak_warns += 1
            for field in ("visual_evidence", "missing_information"):
                gold_list = row.get(field) or []
                pre_list = pre.get(field) or []
                if gold_list and gold_list == pre_list:
                    warnings.append(
                        f"[pair {pid}] gold {field} equals preannotation list while "
                        f"status=pending (possible model->gold leak)"
                    )
                    leak_warns += 1

    # ---- Report ----
    print("=" * 60)
    print("R-CES pair / rule validation report")
    print("=" * 60)
    print(f"rules loaded         : {len(rules)}  (checkpoint structure checked: {rules_checked})")
    print(f"pairs loaded         : {len(pairs)}  ({pairs_path.name})")
    print(f"sidecar preann loaded: {len(sidecar)}  (isolated; leak-guard only)")
    print(f"ERRORS               : {len(errors)}")
    print(f"WARNINGS             : {len(warnings)}  (model->gold leak guard: {leak_warns})")
    print("-" * 60)
    for e in errors:
        print("ERROR  ", e)
    for w in warnings:
        print("WARNING", w)
    if not errors and not warnings:
        print("OK: all checks passed, no warnings.")
    print("=" * 60)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
