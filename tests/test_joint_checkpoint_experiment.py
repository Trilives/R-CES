"""Tests for the isolated joint-checkpoint R-CES experiment."""

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

from data.dataset import PairRecord  # noqa: E402
from rces.compare_joint_checkpoints import masked_bce_loss  # noqa: E402
from rces.joint_dataset import build_joint_example, collate_joint_batch  # noqa: E402
from rces.joint_model import JointRCESHead  # noqa: E402


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


class JointDatasetTest(unittest.TestCase):
    def test_example_contains_and_supervises_all_required_checkpoints(self) -> None:
        example = build_joint_example(
            record(),
            {
                "required_checkpoints": [
                    "subject_present",
                    "protection_present",
                    "protection_intact",
                ],
                "critical_checkpoints": ["subject_present", "protection_intact"],
            },
            missing_checkpoints=("protection_present", "protection_intact"),
        )

        self.assertEqual(
            ("subject_present", "protection_present", "protection_intact"),
            example.checkpoint_keys,
        )
        self.assertEqual((True, False, True), example.critical_mask)
        self.assertEqual((1.0, 0.0, 0.0), example.targets)
        self.assertEqual((True, True, True), example.supervision_mask)

    def test_collate_pads_variable_checkpoint_counts(self) -> None:
        batch = collate_joint_batch(
            [
                {
                    "patches": torch.ones(2, 4),
                    "checkpoint_text": torch.ones(2, 3),
                    "critical_mask": torch.tensor([True, True]),
                    "target": torch.tensor([1.0, 0.0]),
                    "supervision_mask": torch.tensor([True, True]),
                    "pair_id": "p1",
                    "checkpoint_keys": ("a", "b"),
                },
                {
                    "patches": torch.zeros(2, 4),
                    "checkpoint_text": torch.zeros(1, 3),
                    "critical_mask": torch.tensor([True]),
                    "target": torch.tensor([1.0]),
                    "supervision_mask": torch.tensor([True]),
                    "pair_id": "p2",
                    "checkpoint_keys": ("c",),
                },
            ]
        )

        self.assertEqual((2, 2, 3), tuple(batch["checkpoint_text"].shape))
        self.assertEqual([[True, True], [True, False]], batch["checkpoint_mask"].tolist())
        self.assertEqual([[True, True], [True, False]], batch["supervision_mask"].tolist())


class JointHeadTest(unittest.TestCase):
    def test_masked_bce_ignores_padded_outputs(self) -> None:
        loss = masked_bce_loss(
            torch.tensor([[0.0, 100.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[True, False]]),
        )

        self.assertAlmostEqual(0.693147, float(loss), places=5)

    def test_head_returns_one_logit_per_checkpoint_and_masks_padding(self) -> None:
        torch.manual_seed(0)
        head = JointRCESHead(
            img_dim=4,
            txt_dim=3,
            hidden=8,
            dropout=0.0,
            relation_layers=1,
            relation_heads=2,
        ).eval()

        logits = head(
            patches=torch.randn(2, 5, 4),
            checkpoint_text=torch.randn(2, 3, 3),
            checkpoint_mask=torch.tensor([[True, True, True], [True, True, False]]),
            critical_mask=torch.tensor([[True, False, True], [True, True, False]]),
        )

        self.assertEqual((2, 3), tuple(logits.shape))
        self.assertEqual(0.0, float(logits[1, 2].detach()))

    def test_valid_outputs_ignore_masked_padding_checkpoint(self) -> None:
        torch.manual_seed(1)
        head = JointRCESHead(
            img_dim=4,
            txt_dim=3,
            hidden=8,
            dropout=0.0,
            relation_layers=1,
            relation_heads=2,
        ).eval()
        patches = torch.randn(1, 5, 4)
        text = torch.randn(1, 2, 3)
        base = head(
            patches,
            text,
            torch.tensor([[True, True]]),
            torch.tensor([[True, True]]),
        )
        padded = head(
            patches,
            torch.cat([text, torch.randn(1, 1, 3) * 100], dim=1),
            torch.tensor([[True, True, False]]),
            torch.tensor([[True, True, False]]),
        )

        torch.testing.assert_close(base, padded[:, :2])

    def test_checkpoint_outputs_use_other_checkpoint_context(self) -> None:
        torch.manual_seed(2)
        head = JointRCESHead(
            img_dim=4,
            txt_dim=3,
            hidden=8,
            dropout=0.0,
            relation_layers=1,
            relation_heads=2,
        ).eval()
        patches = torch.randn(1, 5, 4)
        text = torch.randn(1, 3, 3)
        mask = torch.ones(1, 3, dtype=torch.bool)
        critical = torch.tensor([[True, False, True]])
        first = head(patches, text, mask, critical)
        changed = text.clone()
        changed[:, 1] += 10.0
        second = head(patches, changed, mask, critical)

        self.assertFalse(torch.allclose(first[:, 0], second[:, 0]))


if __name__ == "__main__":
    unittest.main()
