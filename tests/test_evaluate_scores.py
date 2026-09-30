"""End-to-end checks for experiments/evaluate_scores.py on the shipped test split."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

RCES_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = RCES_ROOT / "experiments" / "evaluate_scores.py"
GOLD_FILES = (
    RCES_ROOT / "data" / "pairs" / "rces_pilot_pairs.jsonl",
    RCES_ROOT / "data" / "pairs" / "rces_crop_augmentation.jsonl",
)


def run_scorer(scores_path: Path | None) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("RCES_")}
    if scores_path is not None:
        env["RCES_SCORES"] = str(scores_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=RCES_ROOT, env=env,
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def parse_table(stdout: str) -> dict[str, dict[str, float]]:
    rows = {}
    for line in stdout.splitlines()[1:]:
        name, n, insuff, auroc, aurc, within = line.split()
        rows[name] = {"n": int(n), "insufficient": int(insuff), "auroc": float(auroc),
                      "aurc": float(aurc), "within": float(within)}
    return rows


class EvaluateScoresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tmp_dir = Path(cls.tmp.name)
        run_scorer(None)
        template = RCES_ROOT / "experiments" / "artifacts" / "test_pairs.jsonl"
        cls.pairs = [json.loads(line) for line in template.read_text().splitlines()]
        cls.gold = {}
        for path in GOLD_FILES:
            for line in path.read_text().splitlines():
                row = json.loads(line)
                cls.gold[row["pair_id"]] = int(row.get("evidence_sufficiency_label") == "sufficient")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def score_with(self, name: str, fn) -> dict[str, dict[str, float]]:
        path = self.tmp_dir / f"{name}.jsonl"
        path.write_text("".join(
            json.dumps({"pair_id": p["pair_id"], "score": fn(p)}) + "\n" for p in self.pairs
        ))
        return parse_table(run_scorer(path))

    def test_template_lists_the_paper_test_split(self) -> None:
        self.assertEqual(len(self.pairs), 267)

    def test_populations_match_the_paper(self) -> None:
        table = self.score_with("const", lambda p: 0.5)
        self.assertEqual(table["all"]["n"], 267)
        self.assertEqual(table["all"]["insufficient"], 59)
        self.assertEqual(table["visual"]["n"], 217)

    def test_constant_score_is_chance(self) -> None:
        table = self.score_with("const", lambda p: 0.5)
        for population in ("all", "visual"):
            self.assertAlmostEqual(table[population]["auroc"], 0.5)
            self.assertAlmostEqual(table[population]["within"], 0.5)

    def test_oracle_score_is_perfect(self) -> None:
        table = self.score_with("oracle", lambda p: self.gold[p["pair_id"]])
        for population in ("all", "visual"):
            self.assertAlmostEqual(table[population]["auroc"], 1.0)
            self.assertAlmostEqual(table[population]["within"], 1.0)

    def test_rule_only_score_is_chance_within_rule(self) -> None:
        table = self.score_with("rule_only", lambda p: zlib.crc32(p["rule_id"].encode()) % 1000)
        for population in ("all", "visual"):
            self.assertAlmostEqual(table[population]["within"], 0.5)

    def test_missing_pair_is_rejected(self) -> None:
        path = self.tmp_dir / "partial.jsonl"
        path.write_text(json.dumps({"pair_id": self.pairs[0]["pair_id"], "score": 1.0}) + "\n")
        with self.assertRaises(subprocess.CalledProcessError):
            run_scorer(path)


if __name__ == "__main__":
    unittest.main()
