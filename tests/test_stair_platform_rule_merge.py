"""Contracts for the merged stair/platform edge rule."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


RCES_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = RCES_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import rces_io as rio  # noqa: E402


MERGED_RULE_ID = "R-EDG-003-roof-balcony-platform-edge"
REMOVED_RULE_ID = "R-EDG-002-stair-edge"
PRESENCE_KEY = "roof_balcony_stair_platform_edge_present"
SUFFICIENCY_KEY = "roof_balcony_stair_platform_edge_protection_sufficient"


class StairPlatformRuleMergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = rio.load_rules()

    def test_stair_rule_is_removed_and_merged_rule_is_retained(self) -> None:
        self.assertNotIn(REMOVED_RULE_ID, self.rules)
        self.assertIn(MERGED_RULE_ID, self.rules)
        self.assertEqual(54, len(self.rules))

    def test_merged_rule_uses_stair_platform_checkpoint_contract(self) -> None:
        rule = self.rules[MERGED_RULE_ID]
        self.assertIn(PRESENCE_KEY, rule["visual_checkpoints"])
        self.assertIn(SUFFICIENCY_KEY, rule["visual_checkpoints"])
        self.assertIn(PRESENCE_KEY, rule["critical_checkpoints"])
        self.assertIn(SUFFICIENCY_KEY, rule["critical_checkpoints"])
        self.assertNotIn("roof_balcony_platform_edge_present", rule["visual_checkpoints"])
        self.assertIn("stair", rule["rule_name"].lower())

    def test_subject_gate_uses_merged_presence_checkpoint(self) -> None:
        self.assertEqual(PRESENCE_KEY, rio.subject_gate(self.rules[MERGED_RULE_ID]))

    def test_current_human_pairs_do_not_use_removed_checkpoint_keys(self) -> None:
        old_keys = {
            "roof_balcony_platform_edge_present",
            "roof_balcony_platform_edge_protection_sufficient",
        }
        path = RCES_ROOT / "data" / "pairs" / "rces_pilot_pairs.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("rule_id") != MERGED_RULE_ID:
                continue
            used = set(row.get("observed_checkpoints", ()))
            used.update(row.get("missing_checkpoints", ()))
            used.update(row.get("critical_missing_checkpoints", ()))
            self.assertTrue(used.isdisjoint(old_keys), path)


if __name__ == "__main__":
    unittest.main()
