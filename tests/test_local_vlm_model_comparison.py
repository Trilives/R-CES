"""Tests for deterministic local-VLM model comparison metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RCES_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = RCES_ROOT / "experiments"
SRC_DIR = RCES_ROOT / "src"
for path in (SRC_DIR, EXPERIMENTS_DIR):
    sys.path.insert(0, str(path))

from local_vlm.compare_models import classification_report, exact_mcnemar_p_value  # noqa: E402


class LocalVLMModelComparisonTest(unittest.TestCase):
    def test_classification_report_builds_confusion_matrix(self) -> None:
        gold = {
            "a": {"compliance_label": "compliant"},
            "b": {"compliance_label": "non_compliant"},
            "c": {"compliance_label": "not_applicable"},
        }
        predictions = {
            "a": {"predicted_compliance": "compliant"},
            "b": {"predicted_compliance": "compliant"},
            "c": {"predicted_compliance": "not_applicable"},
        }

        report = classification_report(gold, predictions, ["a", "b", "c"])

        self.assertAlmostEqual(2 / 3, report["accuracy"])
        self.assertEqual(1, report["confusion_matrix"]["non_compliant"]["compliant"])

    def test_exact_mcnemar_is_one_when_models_never_disagree(self) -> None:
        self.assertEqual(1.0, exact_mcnemar_p_value(0, 0))

    def test_exact_mcnemar_detects_one_sided_disagreement(self) -> None:
        self.assertLess(exact_mcnemar_p_value(0, 10), 0.01)


if __name__ == "__main__":
    unittest.main()
