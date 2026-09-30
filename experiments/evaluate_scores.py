"""Score any pair-level evidence-sufficiency predictor on the R-CES test split.

Input is a JSONL file with one ``{"pair_id": ..., "score": ...}`` line per test pair,
where a higher score means the image is more likely to suffice to decide the rule.
The split is rebuilt exactly as in the paper (image-disjoint 60/20/20 with crop
augmentation), and the predictor is reported on two populations:

  all     all test pairs (the paper's Task-1 "gate on" population)
  visual  test pairs whose rule is not requires_external_evidence (the population
          the visual head is meant for)

For each: pooled AUROC and AURC (positive = evidence sufficient), and within-rule
AUROC -- AUROC inside each rule, pooled by comparable pairs -- on which a scorer
that reads only the rule text gets 0.5 by construction.

  RCES_SCORES=my_scores.jsonl uv run python experiments/evaluate_scores.py

With RCES_SCORES unset, it writes the list of test pairs to score to
experiments/artifacts/test_pairs.jsonl and exits.
"""

from __future__ import annotations

import os

os.environ.setdefault("RCES_INCLUDE_CROP_AUG", "1")  # the paper's test split

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import numpy as np  # noqa: E402

import config  # noqa: E402
import metrics  # noqa: E402
import stratified  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402

SCORES_PATH = os.environ.get("RCES_SCORES")
TEMPLATE_PATH = config.ARTIFACT_DIR / "test_pairs.jsonl"


def load_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "pair_id" not in row or "score" not in row:
            raise ValueError(f"{path}:{line_no}: each line needs pair_id and score")
        score = float(row["score"])
        if not np.isfinite(score):
            raise ValueError(f"{path}:{line_no}: score must be finite, got {row['score']!r}")
        scores[row["pair_id"]] = score
    return scores


def evaluate(records: list[ds.PairRecord], scores: dict[str, float]) -> dict[str, float | int]:
    s = np.array([scores[r.pair_id] for r in records], dtype=float)
    y = np.array([r.sufficiency_gold for r in records], dtype=int)
    within = stratified.stratified_auroc(s, y, [r.rule_id for r in records])
    return {
        "n_pairs": len(records),
        "n_insufficient": int((y == 0).sum()),
        "auroc": metrics._auroc(s, y),
        "aurc": metrics.aurc(s, y),
        "within_rule_auroc": within["pooled"],
        "n_rules_used": within["n_strata_used"],
    }


def main() -> None:
    split = splits.build_split()
    records = ds.build_pair_records(set(split["test"]))

    if not SCORES_PATH:
        config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with TEMPLATE_PATH.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({"pair_id": r.pair_id, "image_id": r.image_id, "rule_id": r.rule_id}) + "\n")
        print(f"Wrote {len(records)} test pairs to {TEMPLATE_PATH.relative_to(config.RCES_ROOT)}.")
        print("Add a 'score' to each line, then run with RCES_SCORES=<that file>.")
        return

    scores = load_scores(Path(SCORES_PATH))
    missing = [r.pair_id for r in records if r.pair_id not in scores]
    if missing:
        raise SystemExit(f"{len(missing)} test pairs have no score, e.g. {missing[:3]}")

    populations = {
        "all": records,
        "visual": [r for r in records if not r.external_evidence_required],
    }
    print(f"{'population':<10} {'n':>5} {'insuff':>7} {'AUROC':>7} {'AURC':>7} {'within-rule':>12}")
    for name, recs in populations.items():
        m = evaluate(recs, scores)
        print(
            f"{name:<10} {m['n_pairs']:>5} {m['n_insufficient']:>7} {m['auroc']:>7.3f} "
            f"{m['aurc']:>7.3f} {m['within_rule_auroc']:>12.3f}"
        )


if __name__ == "__main__":
    main()
