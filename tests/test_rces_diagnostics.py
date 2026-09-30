"""Regression tests for R-CES statistical and input-channel diagnostics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import torch


RCES_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = RCES_ROOT / "experiments"
SRC_DIR = RCES_ROOT / "src"
for path in (SRC_DIR, EXPERIMENTS_DIR):
    sys.path.insert(0, str(path))

import exp_io  # noqa: E402
import metrics  # noqa: E402
from rces.model import RCESHead  # noqa: E402


class BootstrapAurocTest(unittest.TestCase):
    def test_bootstrap_reports_class_counts_and_reproducible_interval(self) -> None:
        result = metrics.bootstrap_auroc(
            [0.9, 0.8, 0.7, 0.2, 0.1, 0.0],
            [1, 1, 1, 0, 0, 0],
            n_bootstrap=200,
            seed=7,
        )

        self.assertEqual(3, result["positives"])
        self.assertEqual(3, result["negatives"])
        self.assertEqual(1.0, result["auroc"])
        self.assertGreater(result["valid_bootstrap_samples"], 0)
        self.assertLessEqual(result["valid_bootstrap_samples"], 200)
        self.assertEqual(1.0, result["ci_95_low"])
        self.assertEqual(1.0, result["ci_95_high"])
        self.assertFalse(result["ci_contains_0_5"])

    def test_cluster_bootstrap_resamples_whole_groups(self) -> None:
        result = metrics.bootstrap_auroc(
            [0.9, 0.8, 0.2, 0.1],
            [1, 1, 0, 0],
            groups=["a", "a", "b", "b"],
            n_bootstrap=200,
            seed=3,
        )

        self.assertEqual("cluster", result["bootstrap_unit"])
        self.assertEqual(2, result["n_groups"])
        self.assertGreater(result["valid_bootstrap_samples"], 0)

    def test_empty_bootstrap_input_is_reported_without_sampling_error(self) -> None:
        result = metrics.bootstrap_auroc([], [], n_bootstrap=10)

        self.assertEqual(0, result["n"])
        self.assertEqual(0, result["valid_bootstrap_samples"])


class ResolvabilityPolarityTest(unittest.TestCase):
    def test_negative_checkpoint_wording_does_not_flip_resolvability_target(self) -> None:
        rule = {
            "critical_checkpoints": ["protection_removed_or_damaged"],
            "required_checkpoints": ["protection_removed_or_damaged"],
            "visual_checkpoints": {
                "protection_removed_or_damaged": "Protection removed or damaged?"
            },
        }

        self.assertEqual(
            {"protection_removed_or_damaged": 1},
            exp_io.resolvable_targets(
                {"observed_checkpoints": ["protection_removed_or_damaged"], "missing_checkpoints": []},
                rule,
            ),
        )
        self.assertEqual(
            {"protection_removed_or_damaged": 0},
            exp_io.resolvable_targets(
                {"observed_checkpoints": [], "missing_checkpoints": ["protection_removed_or_damaged"]},
                rule,
            ),
        )


class HeadInputModeTest(unittest.TestCase):
    def test_text_only_predictions_ignore_image_patches(self) -> None:
        torch.manual_seed(0)
        head = RCESHead(img_dim=4, txt_dim=3, hidden=5, dropout=0.0, input_mode="text_only").eval()
        text = torch.randn(2, 3)
        is_visual = torch.zeros(2, 1)

        first = head(torch.randn(2, 6, 4), text, is_visual)
        second = head(torch.randn(2, 6, 4), text, is_visual)

        torch.testing.assert_close(first, second)

    def test_image_only_predictions_ignore_checkpoint_text(self) -> None:
        torch.manual_seed(0)
        head = RCESHead(img_dim=4, txt_dim=3, hidden=5, dropout=0.0, input_mode="image_only").eval()
        patches = torch.randn(2, 6, 4)
        is_visual = torch.zeros(2, 1)

        first = head(patches, torch.randn(2, 3), is_visual)
        second = head(patches, torch.randn(2, 3), is_visual)

        torch.testing.assert_close(first, second)

    def test_invalid_input_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_mode"):
            RCESHead(img_dim=4, txt_dim=3, input_mode="invalid")


if __name__ == "__main__":
    unittest.main()
