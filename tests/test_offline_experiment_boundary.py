"""Architecture checks for the offline experiment boundary."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


RCES_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = RCES_ROOT / "experiments"
QWEN_SCRIPT = RCES_ROOT / "Script" / "baseline_qwen_full.py"


class OfflineExperimentBoundaryTest(unittest.TestCase):
    def test_experiment_python_files_do_not_contain_cloud_api_calls(self) -> None:
        forbidden = (
            "QWEN_API_KEY",
            "dashscope.aliyuncs.com",
            "make_qwen_client",
            "call_qwen_json",
            "from openai import OpenAI",
            "BASELINE_QWEN_PATH",
            "rces_qwen_full_baseline",
            "qwen_pred_compliance",
            "ABLATE_TEXT_ONLY_INPUT",
        )

        experiment_files = list(EXPERIMENTS_DIR.rglob("*.py")) + [
            RCES_ROOT / "src" / "config.py",
            RCES_ROOT / "src" / "exp_io.py",
            RCES_ROOT / "src" / "metrics.py",
        ]
        violations: list[str] = []
        for path in experiment_files:
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                if marker in text:
                    violations.append(f"{path.relative_to(RCES_ROOT)}: {marker}")

        self.assertEqual([], violations)

    def test_qwen_annotation_script_has_local_run_config(self) -> None:
        module = self._load_qwen_script()

        config = module.RunConfig()
        self.assertEqual("QWEN_API_KEY", config.env_key)
        self.assertEqual(
            RCES_ROOT / "data" / "pairs" / "rces_qwen_full_baseline.jsonl",
            config.baseline_path,
        )

    def test_qwen_baseline_row_normalizes_model_field_slips(self) -> None:
        module = self._load_qwen_script()
        pair = {
            "pair_id": "p1",
            "image_id": "img.jpg",
            "image_split": "development_subset",
            "rule_id": "R-TEST",
            "rule_name": "Opening rule",
            "major_category": "Opening Hazard",
            "subcategory": "Opening",
            "decision_scope": "visual_screening_only",
            "rule_role": "target",
            "image_weak_label": "non_compliant",
            "weak_candidate_category": "Opening Hazard",
        }
        rule = {
            "required_checkpoints": ["opening_present"],
            "critical_checkpoints": ["opening_present"],
            "visual_checkpoints": {"opening_present": "Opening present?"},
        }
        annotation = {
            "pair_id": "p1",
            "decidability": "not_applicable",
            "compliance_label": "not_applicable",
            "image_quality": "good",
            "evidence_sufficiency_label": "sufficient",
            "observed_checkpoints": ["opening_present", "old_checkpoint_name"],
            "missing_checkpoints": [],
            "confidence": 0.95,
        }

        row = module.build_baseline_row(pair, annotation, rule, "qwen-test", False)

        self.assertEqual("decidable", row["decidability"])
        self.assertEqual("not_applicable", row["compliance_label"])
        self.assertEqual("sufficient", row["evidence_sufficiency_label"])
        self.assertEqual(["opening_present"], row["observed_checkpoints"])
        self.assertEqual([], module.baseline_row_errors(row, rule))

    def test_qwen_baseline_row_downgrades_decidable_with_critical_missing(self) -> None:
        module = self._load_qwen_script()
        pair = {
            "pair_id": "p1",
            "image_id": "img.jpg",
            "image_split": "development_subset",
            "rule_id": "R-TEST",
            "rule_name": "Helmet rule",
            "major_category": "Unsafe Behavior",
            "subcategory": "Helmet",
            "decision_scope": "visual_screening_only",
            "rule_role": "target",
            "image_weak_label": "non_compliant",
            "weak_candidate_category": "Unsafe Behavior",
        }
        rule = {
            "required_checkpoints": ["person_present", "helmet_present"],
            "critical_checkpoints": ["person_present", "helmet_present"],
            "visual_checkpoints": {
                "person_present": "Person present?",
                "helmet_present": "Helmet present?",
            },
        }
        annotation = {
            "pair_id": "p1",
            "decidability": "decidable",
            "compliance_label": "non_compliant",
            "image_quality": "usable",
            "evidence_sufficiency_label": "sufficient",
            "observed_checkpoints": ["person_present"],
            "missing_checkpoints": ["helmet_present"],
            "confidence": 0.9,
        }

        row = module.build_baseline_row(pair, annotation, rule, "qwen-test", False)

        self.assertEqual("partially_decidable", row["decidability"])
        self.assertEqual("unknown", row["compliance_label"])
        self.assertEqual("partial", row["evidence_sufficiency_label"])
        self.assertEqual(["helmet_present"], row["critical_missing_checkpoints"])
        self.assertEqual([], module.baseline_row_errors(row, rule))

    def test_qwen_baseline_drop_invalid_existing_rows(self) -> None:
        module = self._load_qwen_script()
        rule = {
            "required_checkpoints": ["opening_present"],
            "critical_checkpoints": ["opening_present"],
            "visual_checkpoints": {"opening_present": "Opening present?"},
        }
        valid = {
            "pair_id": "valid",
            "rule_id": "R-TEST",
            "decidability": "decidable",
            "compliance_label": "not_applicable",
            "image_quality": "good",
            "evidence_sufficiency_label": "sufficient",
            "observed_checkpoints": ["opening_present"],
            "missing_checkpoints": [],
            "critical_missing_checkpoints": [],
        }
        invalid = {**valid, "pair_id": "invalid", "decidability": "not_applicable"}

        kept, dropped = module.drop_invalid_rows([valid, invalid], {"R-TEST": rule})

        self.assertEqual(["valid"], [row["pair_id"] for row in kept])
        self.assertIn("invalid", dropped)

    def test_experiment_config_has_no_cloud_baseline_input(self) -> None:
        config_text = (RCES_ROOT / "src" / "config.py").read_text(encoding="utf-8")
        self.assertNotIn("BASELINE_QWEN_PATH", config_text)
        self.assertNotIn("ABLATE_TEXT_ONLY_INPUT", config_text)

    def _load_qwen_script(self):
        self.assertTrue(QWEN_SCRIPT.is_file())
        spec = importlib.util.spec_from_file_location("baseline_qwen_full", QWEN_SCRIPT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader if spec else None)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
