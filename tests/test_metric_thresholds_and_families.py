"""Regression checks for deployable score thresholds and dependent crops."""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import exp_io
import metrics
import stratified


class ThresholdTieTest(unittest.TestCase):
    def test_mixed_constant_scores_cannot_meet_zero_risk(self):
        for labels in ([1, 0], [0, 1]):
            curve = metrics.risk_coverage_curve([0.5, 0.5], labels)
            np.testing.assert_allclose(curve["coverage"], [1.0])
            np.testing.assert_allclose(curve["risk"], [0.5])
            self.assertEqual(metrics.coverage_at_risk([0.5, 0.5], labels, 0), 0)
            self.assertGreater(metrics.tune_tau_for_risk([0.5, 0.5], labels, 0), 0.5)
            self.assertEqual(metrics.aurc([0.5, 0.5], labels), 0.5)
        scores = np.asarray([0.5, 0.5], dtype=np.float32)
        tau = metrics.tune_tau_for_risk(scores, [1, 0], 0)
        self.assertFalse((scores >= tau).any())

    def test_each_curve_point_equals_the_actual_threshold_decision(self):
        scores = np.array([0.9, 0.5, 0.5, 0.2, 0.2])
        labels = np.array([1, 1, 0, 1, 0])
        curve = metrics.risk_coverage_curve(scores, labels)
        for tau, coverage, risk in zip(curve["thresholds"], curve["coverage"], curve["risk"]):
            accepted = scores >= tau
            self.assertEqual(coverage, accepted.mean())
            self.assertEqual(risk, (1 - labels[accepted]).mean())
        for target in [0, 0.2, 0.4, 1]:
            tau = metrics.tune_tau_for_risk(scores, labels, target)
            accepted = scores >= tau
            self.assertEqual(accepted.mean(), metrics.coverage_at_risk(scores, labels, target))
            if accepted.any():
                self.assertLessEqual((1 - labels[accepted]).mean(), target)

    def test_tie_permutations_leave_aurc_and_average_precision_unchanged(self):
        expected = None
        for permutation in itertools.permutations(range(4)):
            scores = np.array([0.8, 0.8, 0.2, 0.2])[list(permutation)]
            labels = np.array([1, 0, 1, 0])[list(permutation)]
            actual = (metrics.aurc(scores, labels), metrics.binary_classification_metrics(scores, labels)["auprc"])
            if expected is None:
                expected = actual
            self.assertEqual(expected, actual)

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(metrics.coverage_at_risk([], [], 0), 0)
        self.assertEqual(metrics.tune_tau_for_risk([], [], 0), float("inf"))
        self.assertTrue(np.isnan(metrics.aurc([], [])))
        for scores, labels in [([float("nan")], [1]), ([1], [2]), ([1, 2], [1])]:
            with self.assertRaises(ValueError):
                metrics.risk_coverage_curve(scores, labels)


class ImageFamilyBootstrapTest(unittest.TestCase):
    def test_real_test_images_collapse_to_fifty_original_families(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "experiments"))
        from data import splits
        import config
        with patch.object(config, "INCLUDE_CROP_AUGMENTATION", True):
            split = splits.build_split()
        self.assertEqual(len(split["test"]), 100)
        self.assertEqual(len(set(exp_io.original_image_families(split["test"]))), 50)

    def test_family_resampling_matches_explicit_parent_ids(self):
        frame = {"score": [0.9, 0.1, 0.3, 0.7], "target": [1, 0, 1, 0],
                 "image_id": ["a", "crop_a", "b", "crop_b"], "key": ["k"] * 4}
        explicit = dict(frame, image_id=["a", "a", "b", "b"])
        with patch.object(exp_io, "crop_parent_map", return_value={"crop_a": "a", "crop_b": "b"}):
            actual = stratified.mean_of_seed_aurocs([frame, frame], "key", n_bootstrap=100)
            expected = stratified.mean_of_seed_aurocs([explicit, explicit], "key", n_bootstrap=100)
        self.assertEqual(actual["n_families"], 2)
        self.assertEqual(actual["n_image_ids"], 4)
        self.assertEqual(actual["ci_95_low"], expected["ci_95_low"])
        self.assertEqual(actual["ci_95_high"], expected["ci_95_high"])
        self.assertEqual(actual["point_mean_of_seeds"], expected["point_mean_of_seeds"])


if __name__ == "__main__":
    unittest.main()
