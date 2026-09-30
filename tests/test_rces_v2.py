"""Contract tests for the isolated R-CES v2 architecture."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


RCES_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = RCES_ROOT / "experiments"
SRC_DIR = RCES_ROOT / "src"
V2_DIR = EXPERIMENTS_DIR / "rces-2"
for path in (SRC_DIR, EXPERIMENTS_DIR, V2_DIR):
    sys.path.insert(0, str(path))

from data.dataset import PairRecord  # noqa: E402
from rces2_dataset import build_rule_example, collate_rule_batch  # noqa: E402
from rces2_model import ContextualRCESHead  # noqa: E402
from rces2_scoring import critical_min_score  # noqa: E402
from rces2_training import masked_bce_loss  # noqa: E402


def record() -> PairRecord:
    return PairRecord(
        pair_id="p1",
        image_id="img.jpg",
        rule_id="r1",
        decision_scope="visual_screening_only",
        critical_keys=("subject_present", "protection_intact"),
        resolvable_targets={"subject_present": 1, "protection_intact": 0},
        external_evidence_required=False,
        sufficiency_gold=0,
        decidable_gold=0,
        compliance_gold="unknown",
        critical_missing_gold=("protection_intact",),
    )


class RuleDatasetTest(unittest.TestCase):
    def test_example_contains_every_required_checkpoint(self) -> None:
        example = build_rule_example(
            record(),
            {
                "required_checkpoints": [
                    "subject_present",
                    "protection_present",
                    "protection_intact",
                ]
            },
            missing_checkpoints=("protection_present", "protection_intact"),
        )

        self.assertEqual(
            ("subject_present", "protection_present", "protection_intact"),
            example.checkpoint_keys,
        )
        self.assertEqual((True, False, True), example.critical_mask)
        self.assertEqual((1.0, 0.0, 0.0), example.targets)

    def test_example_rejects_checkpoint_outside_rule_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "out-of-contract"):
            build_rule_example(
                record(),
                {"required_checkpoints": ["subject_present"]},
                missing_checkpoints=("unknown_checkpoint",),
            )

    def test_collate_pads_text_targets_and_masks_together(self) -> None:
        batch = collate_rule_batch(
            [
                {
                    "patches": torch.ones(2, 4),
                    "checkpoint_text": torch.ones(2, 3),
                    "critical_mask": torch.tensor([True, False]),
                    "target": torch.tensor([1.0, 0.0]),
                    "pair_id": "p1",
                    "checkpoint_keys": ("a", "b"),
                },
                {
                    "patches": torch.zeros(2, 4),
                    "checkpoint_text": torch.zeros(1, 3),
                    "critical_mask": torch.tensor([True]),
                    "target": torch.tensor([1.0]),
                    "pair_id": "p2",
                    "checkpoint_keys": ("c",),
                },
            ]
        )

        self.assertEqual((2, 2, 3), tuple(batch["checkpoint_text"].shape))
        self.assertEqual([[True, True], [True, False]], batch["checkpoint_mask"].tolist())
        self.assertEqual([[True, False], [True, False]], batch["critical_mask"].tolist())


class ContextualHeadTest(unittest.TestCase):
    def make_head(self) -> ContextualRCESHead:
        torch.manual_seed(0)
        return ContextualRCESHead(
            img_dim=4,
            txt_dim=3,
            hidden=8,
            dropout=0.0,
            context_heads=2,
            image_heads=2,
        ).eval()

    def test_returns_one_masked_logit_per_checkpoint(self) -> None:
        logits = self.make_head()(
            patches=torch.randn(2, 5, 4),
            checkpoint_text=torch.randn(2, 3, 3),
            checkpoint_mask=torch.tensor([[True, True, True], [True, True, False]]),
        )

        self.assertEqual((2, 3), tuple(logits.shape))
        self.assertEqual(0.0, float(logits[1, 2].detach()))

    def test_valid_outputs_ignore_padded_checkpoint_text(self) -> None:
        head = self.make_head()
        patches = torch.randn(1, 5, 4)
        text = torch.randn(1, 2, 3)
        base = head(patches, text, torch.tensor([[True, True]]))
        padded = head(
            patches,
            torch.cat([text, torch.randn(1, 1, 3) * 100], dim=1),
            torch.tensor([[True, True, False]]),
        )

        torch.testing.assert_close(base, padded[:, :2])

    def test_checkpoint_query_uses_sibling_context_before_image_attention(self) -> None:
        head = self.make_head()
        patches = torch.randn(1, 5, 4)
        text = torch.randn(1, 3, 3)
        mask = torch.ones(1, 3, dtype=torch.bool)
        first = head(patches, text, mask)
        changed = text.clone()
        changed[:, 1] += 10.0
        second = head(patches, changed, mask)

        self.assertFalse(torch.allclose(first[:, 0], second[:, 0]))

    def test_checkpoint_order_only_reorders_outputs(self) -> None:
        head = self.make_head()
        patches = torch.randn(1, 5, 4)
        text = torch.randn(1, 3, 3)
        mask = torch.ones(1, 3, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 1])

        original = head(patches, text, mask)
        reordered = head(patches, text[:, permutation], mask[:, permutation])

        torch.testing.assert_close(reordered, original[:, permutation])


class TrainingAndScoringTest(unittest.TestCase):
    def test_masked_bce_ignores_padding(self) -> None:
        loss = masked_bce_loss(
            torch.tensor([[0.0, 100.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[True, False]]),
        )

        self.assertAlmostEqual(0.693147, float(loss), places=5)

    def test_pair_score_is_min_over_critical_subset_only(self) -> None:
        score = critical_min_score(
            record(),
            {
                "subject_present": 0.8,
                "protection_present": 0.01,
                "protection_intact": 0.3,
            },
        )

        self.assertEqual(0.3, score)


if __name__ == "__main__":
    unittest.main()
