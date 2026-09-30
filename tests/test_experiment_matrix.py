"""Tests for the required comparison and ablation experiment matrix."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path


RCES_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = RCES_ROOT / "experiments"
SRC_DIR = RCES_ROOT / "src"
for path in (SRC_DIR, EXPERIMENTS_DIR):
    sys.path.insert(0, str(path))

import metrics  # noqa: E402
import config  # noqa: E402
from ablation.run import ABLATIONS  # noqa: E402
from baselines.common import BaselineRow, maxprob_scores  # noqa: E402
from data.dataset import PairRecord, validate_image_feature_cache  # noqa: E402
from rces.aggregate import (  # noqa: E402
    aggregate_pair_score,
    apply_external_gate,
)


def record() -> PairRecord:
    return PairRecord(
        pair_id="p1",
        image_id="img.jpg",
        rule_id="r1",
        decision_scope="visual_screening_only",
        critical_keys=("a", "b"),
        resolvable_targets={"a": 1, "b": 0},
        external_evidence_required=False,
        sufficiency_gold=0,
        decidable_gold=0,
        compliance_gold="non_compliant",
        critical_missing_gold=("b",),
    )


class ExperimentMatrixTest(unittest.TestCase):
    def test_siglip_profiles_keep_224_and_512_artifacts_separate(self) -> None:
        self.assertEqual("siglip2_512", config.VISION_ENCODER)
        self.assertEqual(224, config.ENCODER_IMAGE_SIZE["siglip2_224"])
        self.assertEqual(512, config.ENCODER_IMAGE_SIZE["siglip2_512"])
        self.assertEqual(196, config.ENCODER_N_PATCH["siglip2_224"])
        self.assertEqual(1024, config.ENCODER_N_PATCH["siglip2_512"])
        self.assertEqual(768, config.ENCODER_IMG_DIM["siglip2_224"])
        self.assertEqual(768, config.ENCODER_IMG_DIM["siglip2_512"])

    def test_feature_manifest_rejects_cache_from_another_profile(self) -> None:
        original_manifest = config.FEATURE_MANIFEST
        original_dir = config.FEATURE_DIR
        original_size = config.VISION_IMAGE_SIZE
        original_patches = config.N_PATCH
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = root / "manifest.json"
                manifest.write_text(
                    json.dumps({"image_size": 224, "n_patch": 196}),
                    encoding="utf-8",
                )
                config.FEATURE_DIR = root
                config.FEATURE_MANIFEST = manifest
                config.VISION_IMAGE_SIZE = 512
                config.N_PATCH = 1024
                with self.assertRaisesRegex(ValueError, "image_size=224"):
                    validate_image_feature_cache()
        finally:
            config.FEATURE_DIR = original_dir
            config.FEATURE_MANIFEST = original_manifest
            config.VISION_IMAGE_SIZE = original_size
            config.N_PATCH = original_patches

    def test_required_ablation_matrix_is_present(self) -> None:
        names = {name for name, _ in ABLATIONS}
        self.assertEqual(
            {
                "full",
                "no_checkpoint_supervision",
                "with_is_visual_legacy",
                "no_text_conditioning",
                "aggregation_mean",
                "aggregation_product",
            },
            names,
        )

    def test_aggregation_modes(self) -> None:
        probs = {"a": 0.8, "b": 0.5}
        self.assertAlmostEqual(0.5, aggregate_pair_score(record(), probs, mode="min"))
        self.assertAlmostEqual(0.65, aggregate_pair_score(record(), probs, mode="mean"))
        self.assertAlmostEqual(0.4, aggregate_pair_score(record(), probs, mode="product"))

    def test_visual_aggregation_is_not_zeroed_by_external_rule_gate(self) -> None:
        external = PairRecord(
            **{
                **record().__dict__,
                "decision_scope": "requires_external_evidence",
                "external_evidence_required": True,
            }
        )

        visual_score = aggregate_pair_score(external, {"a": 0.8, "b": 0.5})

        self.assertAlmostEqual(0.5, visual_score)
        self.assertEqual(0.0, apply_external_gate(external, visual_score))

    def test_checkpoint_metrics_include_required_statistics(self) -> None:
        result = metrics.binary_classification_metrics(
            [0.9, 0.8, 0.2, 0.1],
            [1, 1, 0, 0],
            threshold=0.5,
        )
        self.assertEqual(1.0, result["auroc"])
        self.assertEqual(1.0, result["auprc"])
        self.assertEqual(1.0, result["f1"])

    def test_maxprob_uses_local_model_confidence(self) -> None:
        rows = [
            BaselineRow(
                pair_id="p1",
                decision_scope="visual_screening_only",
                n_critical=2,
                confidence=0.73,
                sufficiency_gold=1,
                should_abstain=0,
                compliance_gold="compliant",
                predicted_compliance="compliant",
                raw_compliance_correct=1,
                compliance_correct=1,
            )
        ]
        self.assertEqual([0.73], maxprob_scores(rows))

    def test_codebook_keeps_load_bearing_experiment_statements(self) -> None:
        text = (RCES_ROOT / "codebook.md").read_text(encoding="utf-8")
        self.assertIn("The primary, load-bearing signal", text)
        self.assertIn("dense supervision", text)
        self.assertIn("For the R-CES itself, the binary gold", text)


if __name__ == "__main__":
    unittest.main()
