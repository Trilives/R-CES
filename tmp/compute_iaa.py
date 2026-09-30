"""Inter-annotator agreement for the R-CES gold over the double-annotated subset.

Aligns the first annotator's gold (rces_pilot_pairs.jsonl) with the second
annotator's independent pass (rces_double_annotation.jsonl) by pair_id, over the
pairs both have annotated, and reports Cohen's kappa per judgment field plus a
checkpoint-level kappa for critical-missing agreement. Disagreements are listed
for adjudication. No CLI flags -- edit the constants below.

    uv run python tmp/compute_iaa.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rces_io as aio  # noqa: E402

# ---- Run controls ------------------------------------------------------------
CATEGORICAL_FIELDS = ("decidability", "compliance_label", "image_quality", "evidence_sufficiency_label")
REPORT_PATH = aio.PAIRS_DIR / "iaa_report.json"
MAX_DISAGREEMENTS_SHOWN = 40
# ------------------------------------------------------------------------------


def cohen_kappa(observations: list[tuple[str, str]]) -> tuple[float | None, float]:
    """Return (kappa, observed_agreement) for paired categorical labels.
    kappa is None when it is undefined (single category on both sides)."""
    n = len(observations)
    if n == 0:
        return None, 0.0
    po = sum(1 for a, b in observations if a == b) / n
    count_a = Counter(a for a, _ in observations)
    count_b = Counter(b for _, b in observations)
    categories = set(count_a) | set(count_b)
    pe = sum((count_a[c] / n) * (count_b[c] / n) for c in categories)
    if pe >= 1.0:  # degenerate: everything in one category
        return (1.0 if po == 1.0 else None), po
    return (po - pe) / (1 - pe), po


def critical_checkpoint_observations(
    common: list[str],
    a_by_id: dict[str, dict[str, Any]],
    b_by_id: dict[str, dict[str, Any]],
    rules: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    """Expand each pair into one binary label per critical checkpoint of its rule
    ('1' if in critical_missing, else '0'), pooled across pairs."""
    obs: list[tuple[str, str]] = []
    for pid in common:
        rule = rules.get(a_by_id[pid]["rule_id"], {})
        _, critical, _ = aio.checkpoints_for(rule)
        a_cm = set(a_by_id[pid].get("critical_missing_checkpoints", []))
        b_cm = set(b_by_id[pid].get("critical_missing_checkpoints", []))
        for ckpt in critical:
            obs.append(("1" if ckpt in a_cm else "0", "1" if ckpt in b_cm else "0"))
    return obs


def main() -> None:
    rules = aio.load_rules()
    gold = aio.load_jsonl(aio.GOLD_PATH)
    double = aio.load_jsonl(aio.DOUBLE_PATH)

    annotated = lambda r: r.get("annotation_status") in ("judged", "described", "reviewed")
    a_by_id = {r["pair_id"]: r for r in gold if annotated(r)}
    b_by_id = {r["pair_id"]: r for r in double if annotated(r)}
    common = sorted(set(a_by_id) & set(b_by_id))

    print("=" * 60)
    print("R-CES inter-annotator agreement (double-annotated subset)")
    print("=" * 60)
    print(f"gold annotated      : {len(a_by_id)}")
    print(f"second-pass annotated: {len(b_by_id)}")
    print(f"compared (both)     : {len(common)}")
    if not common:
        print("\nNo overlapping annotated pairs yet. Run both passes first.")
        return
    print("-" * 60)

    report: dict[str, Any] = {"n_compared": len(common), "fields": {}, "disagreements": []}

    for field in CATEGORICAL_FIELDS:
        obs = [(a_by_id[p][field], b_by_id[p][field]) for p in common]
        kappa, po = cohen_kappa(obs)
        report["fields"][field] = {"kappa": kappa, "observed_agreement": po, "n": len(obs)}
        ks = "n/a " if kappa is None else f"{kappa:+.3f}"
        print(f"{field:28s} kappa={ks}  agreement={po:.3f}  n={len(obs)}")

    cm_obs = critical_checkpoint_observations(common, a_by_id, b_by_id, rules)
    kappa, po = cohen_kappa(cm_obs)
    report["fields"]["critical_missing_checkpoint"] = {"kappa": kappa, "observed_agreement": po, "n": len(cm_obs)}
    ks = "n/a " if kappa is None else f"{kappa:+.3f}"
    print(f"{'critical_missing (ckpt-level)':28s} kappa={ks}  agreement={po:.3f}  n={len(cm_obs)}")

    # ---- Disagreement list (for adjudication) ----
    print("-" * 60)
    for pid in common:
        diffs = {f: (a_by_id[pid][f], b_by_id[pid][f]) for f in CATEGORICAL_FIELDS if a_by_id[pid][f] != b_by_id[pid][f]}
        a_cm = sorted(a_by_id[pid].get("critical_missing_checkpoints", []))
        b_cm = sorted(b_by_id[pid].get("critical_missing_checkpoints", []))
        if a_cm != b_cm:
            diffs["critical_missing_checkpoints"] = (a_cm, b_cm)
        if diffs:
            report["disagreements"].append({"pair_id": pid, "rule_id": a_by_id[pid]["rule_id"], "diffs": diffs})

    n_dis = len(report["disagreements"])
    print(f"pairs with >=1 disagreement: {n_dis} / {len(common)}")
    for entry in report["disagreements"][:MAX_DISAGREEMENTS_SHOWN]:
        print(f"  {entry['pair_id']} ({entry['rule_id']}):")
        for field, (a_val, b_val) in entry["diffs"].items():
            print(f"      {field}: annotator_1={a_val!r}  vs  annotator_2={b_val!r}")
    if n_dis > MAX_DISAGREEMENTS_SHOWN:
        print(f"  ... and {n_dis - MAX_DISAGREEMENTS_SHOWN} more (see {REPORT_PATH.name})")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        shown = REPORT_PATH.relative_to(aio.PROJECT_ROOT)
    except ValueError:
        shown = REPORT_PATH
    print("-" * 60)
    print(f"report -> {shown}")
    print("=" * 60)


if __name__ == "__main__":
    main()
