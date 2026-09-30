"""Tests for deterministic human/Qwen annotation comparison."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


RCES_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = RCES_ROOT / "Script" / "compare_qwen_to_gold.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_qwen_to_gold", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QwenHumanComparisonTest(unittest.TestCase):
    def test_compare_pair_reports_field_and_checkpoint_disagreement(self) -> None:
        module = load_module()
        human = {
            "pair_id": "p1",
            "decidability": "undecidable",
            "compliance_label": "unknown",
            "image_quality": "good",
            "evidence_sufficiency_label": "insufficient",
            "observed_checkpoints": ["presence"],
            "missing_checkpoints": ["state"],
            "critical_missing_checkpoints": ["state"],
        }
        qwen = {
            "pair_id": "p1",
            "decidability": "decidable",
            "compliance_label": "compliant",
            "image_quality": "good",
            "evidence_sufficiency_label": "sufficient",
            "observed_checkpoints": ["presence", "state"],
            "missing_checkpoints": [],
            "critical_missing_checkpoints": [],
            "confidence": 0.93,
        }

        result = module.compare_pair(human, qwen, high_confidence_threshold=0.8)

        self.assertEqual(
            ["decidability", "compliance_label", "evidence_sufficiency_label"],
            result["disagreed_fields"],
        )
        self.assertIn("missing_checkpoints", result["checkpoint_disagreements"])
        self.assertTrue(result["high_confidence_disagreement"])

    def test_empty_alignment_has_no_agreement_rate(self) -> None:
        module = load_module()

        report, disagreements = module.build_report(
            [{"pair_id": "human-only"}],
            [{"pair_id": "qwen-only"}],
            high_confidence_threshold=0.8,
        )

        self.assertEqual(0, report["aligned_count"])
        self.assertIsNone(report["field_agreement"]["decidability"]["rate"])
        self.assertEqual([], disagreements)


if __name__ == "__main__":
    unittest.main()
