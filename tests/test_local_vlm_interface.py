"""Contract tests for the offline local-Qwen experiment interface."""

from __future__ import annotations

import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


RCES_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = RCES_ROOT / "experiments"
SRC_DIR = RCES_ROOT / "src"
for path in (SRC_DIR, EXPERIMENTS_DIR):
    sys.path.insert(0, str(path))

import config  # noqa: E402
import rces_io  # noqa: E402

from local_vlm.predictor import (  # noqa: E402
    LABEL_BY_OPTION,
    build_compliance_prompt,
    build_structured_prompt,
    parse_structured,
    prediction_record_is_compatible,
    predict_pair,
)
from local_vlm.runtime import (  # noqa: E402
    OptionPrediction,
    resize_for_model_input,
    qwen_model_class_name,
    validate_local_qwen_model,
)


# The structured decode the fake compliance VLM emits: its committed label
# ("compliant") is DELIBERATELY different from the A/B/C/D option ("B" ->
# non_compliant) so a test can prove the label comes from the JSON, not the option.
FAKE_STRUCTURED_JSON = (
    '{"checkpoint_analysis": ['
    '{"checkpoint": "opening_present", "observation": "cover seen", "judgment": "satisfied"},'
    '{"checkpoint": "bogus_key", "observation": "ignored", "judgment": "satisfied"}'
    '], "compliance": "compliant"}'
)


class FakeRuntime:
    model_name = "fake-local-vlm"

    def __init__(self, generated: list[str] | None = None) -> None:
        self.generated = list(generated or [FAKE_STRUCTURED_JSON])
        self.option_prompts: list[str] = []
        self.allowed_options: list[tuple[str, ...]] = []
        self.generation_calls = 0

    def predict_option(
        self,
        image_path: Path,
        prompt: str,
        *,
        allowed_options: tuple[str, ...] = ("A", "B", "C", "D"),
    ) -> OptionPrediction:
        self.image_path = image_path
        self.prompt = prompt
        self.option_prompts.append(prompt)
        self.allowed_options.append(allowed_options)
        probabilities = {"A": 0.2, "B": 0.7, "C": 0.1}
        if "D" in allowed_options:
            probabilities = {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}
        return OptionPrediction(
            option="B",
            confidence=0.7,
            probabilities=probabilities,
        )

    def generate_text(self, image_path: Path, prompt: str, *, max_new_tokens: int = 48) -> str:
        del image_path, prompt, max_new_tokens
        value = self.generated[min(self.generation_calls, len(self.generated) - 1)]
        self.generation_calls += 1
        return value


class LocalVLMInterfaceTest(unittest.TestCase):
    def _require_model_config(self, model_name: str) -> Path:
        model_dir = RCES_ROOT / "models" / model_name
        if not (model_dir / "config.json").exists():
            self.skipTest(f"optional local model not bundled: {model_name}")
        return model_dir

    def test_cloud_model_image_encoding_is_capped_at_1080p_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.jpg"
            Image.new("RGB", (4000, 3000)).save(path)

            _, payload = rces_io.encode_image(path)
            with Image.open(io.BytesIO(base64.b64decode(payload))) as encoded:
                self.assertLessEqual(max(encoded.size), 1920)
                self.assertLessEqual(encoded.width * encoded.height, 1920 * 1080)

    def test_model_input_is_capped_at_1080p_budget(self) -> None:
        source = Image.new("RGB", (4000, 3000))

        resized = resize_for_model_input(source)

        self.assertLessEqual(max(resized.size), 1920)
        self.assertLessEqual(resized.width * resized.height, 1920 * 1080)
        self.assertAlmostEqual(source.width / source.height, resized.width / resized.height, places=2)

    def test_small_model_input_is_not_upscaled(self) -> None:
        source = Image.new("RGB", (1280, 720))

        resized = resize_for_model_input(source)

        self.assertIs(source, resized)

    def test_downloaded_qwen35_model_has_expected_multimodal_architecture(self) -> None:
        config = validate_local_qwen_model(self._require_model_config("Qwen3.5-4B"))
        self.assertIn("vision_config", config)
        self.assertEqual("Qwen3_5ForConditionalGeneration", qwen_model_class_name(config))

    def test_downloaded_qwen3vl_8b_model_has_supported_multimodal_architecture(self) -> None:
        config = validate_local_qwen_model(self._require_model_config("Qwen3-VL-8B-Instruct"))
        self.assertIn("vision_config", config)
        self.assertEqual("Qwen3VLForConditionalGeneration", qwen_model_class_name(config))

    def test_downloaded_qwen35_9b_model_has_supported_multimodal_architecture(self) -> None:
        model_config = validate_local_qwen_model(self._require_model_config("Qwen3.5-9B"))
        self.assertIn("vision_config", model_config)
        self.assertEqual(
            "Qwen3_5ForConditionalGeneration", qwen_model_class_name(model_config)
        )

    def test_prompt_uses_rule_metadata_without_human_answer(self) -> None:
        pair = {
            "pair_id": "p1",
            "rule_name": "Opening cover missing",
            "decision_scope": "visual_screening_only",
            "compliance_label": "non_compliant",
            "decidability": "decidable",
        }
        rule = {
            "visual_screening_rule": "opening_present == yes AND cover_sufficient == no",
            "visual_checkpoints": {"opening_present": "Opening present?"},
            "required_checkpoints": ["opening_present"],
            "critical_checkpoints": ["opening_present"],
        }

        prompt = build_compliance_prompt(pair, rule)

        self.assertIn("Opening cover missing", prompt)
        self.assertIn("A = compliant", prompt)
        self.assertNotIn('"compliance_label"', prompt)
        self.assertNotIn('"decidability"', prompt)

    def _predict(self, *, structured: bool, runtime: FakeRuntime | None = None) -> tuple[dict, FakeRuntime]:
        runtime = runtime or FakeRuntime()
        original = config.LOCAL_USE_STRUCTURED
        config.LOCAL_USE_STRUCTURED = structured
        try:
            prediction = predict_pair(
                runtime,
                pair={"pair_id": "p1", "image_id": "img.jpg", "rule_name": "Rule"},
                rule={
                    "visual_checkpoints": {"opening_present": "Opening present?"},
                    "required_checkpoints": ["opening_present"],
                    "critical_checkpoints": ["opening_present"],
                },
                image_path=Path("img.jpg"),
            )
            return prediction, runtime
        finally:
            config.LOCAL_USE_STRUCTURED = original

    def test_structured_label_and_confidence_come_from_same_decision_call(self) -> None:
        prediction, runtime = self._predict(structured=True)

        self.assertEqual("non_compliant", prediction["predicted_compliance"])
        self.assertEqual(LABEL_BY_OPTION["B"], prediction["option_label"])
        self.assertEqual(prediction["predicted_compliance"], prediction["confidence_label"])
        self.assertEqual(0.7, prediction["confidence"])
        self.assertEqual(1.0, sum(prediction["label_probabilities"].values()))
        self.assertEqual(prediction["confidence"], prediction["label_probabilities"][prediction["predicted_compliance"]])
        self.assertTrue(prediction["structured_ok"])
        self.assertEqual("structured", prediction["reasoning_mode"])
        self.assertEqual(("A", "B", "C"), runtime.allowed_options[-1])
        self.assertIn("opening_present", runtime.option_prompts[-1])
        self.assertIn("cover seen", runtime.option_prompts[-1])
        # checkpoint_analysis is filtered to the rule's own keys.
        keys = [item["checkpoint"] for item in prediction["checkpoint_analysis"]]
        self.assertEqual(["opening_present"], keys)
        self.assertEqual("fake-local-vlm", prediction["model"])

    def test_direct_option_mode_uses_option_label(self) -> None:
        prediction, runtime = self._predict(structured=False)

        self.assertEqual(LABEL_BY_OPTION["B"], prediction["predicted_compliance"])
        self.assertEqual(prediction["predicted_compliance"], prediction["confidence_label"])
        self.assertEqual(0.7, prediction["confidence"])
        self.assertEqual("direct_option", prediction["reasoning_mode"])
        self.assertEqual(("A", "B", "C", "D"), runtime.allowed_options[-1])
        self.assertNotIn("checkpoint_analysis", prediction)

    def test_parse_structured_falls_back_on_unusable_decode(self) -> None:
        rule = {
            "visual_checkpoints": {"opening_present": "Opening present?"},
            "required_checkpoints": ["opening_present"],
            "critical_checkpoints": ["opening_present"],
        }
        bad = parse_structured("the model rambled with no json", rule)
        self.assertIsNone(bad["compliance"])
        self.assertFalse(bad["structured_ok"])

        # A present-but-invalid label is rejected (compliance VLM never abstains).
        unknown = parse_structured('{"compliance": "unknown"}', rule)
        self.assertIsNone(unknown["compliance"])

        # An out-of-rule checkpoint judgment is coerced to "unresolved".
        coerced = parse_structured(
            '{"checkpoint_analysis": [{"checkpoint": "opening_present", '
            '"observation": "x", "judgment": "maybe"}], "compliance": "non_compliant"}',
            rule,
        )
        self.assertEqual("non_compliant", coerced["compliance"])
        self.assertEqual("unresolved", coerced["checkpoint_analysis"][0]["judgment"])

    def test_parse_structured_rejects_missing_or_duplicate_required_checkpoints(self) -> None:
        rule = {
            "visual_checkpoints": {
                "opening_present": "Opening present?",
                "cover_fixed": "Cover fixed?",
            },
            "required_checkpoints": ["opening_present", "cover_fixed"],
            "critical_checkpoints": ["opening_present"],
        }
        incomplete = parse_structured(
            '{"checkpoint_analysis": [{"checkpoint": "opening_present", '
            '"observation": "x", "judgment": "satisfied"}], "compliance": "compliant"}',
            rule,
        )
        self.assertFalse(incomplete["structured_ok"])
        self.assertEqual(["cover_fixed"], incomplete["missing_checkpoints"])

        duplicate = parse_structured(
            '{"checkpoint_analysis": ['
            '{"checkpoint": "opening_present", "observation": "x", "judgment": "satisfied"},'
            '{"checkpoint": "opening_present", "observation": "y", "judgment": "violated"},'
            '{"checkpoint": "cover_fixed", "observation": "z", "judgment": "satisfied"}'
            '], "compliance": "compliant"}',
            rule,
        )
        self.assertFalse(duplicate["structured_ok"])
        self.assertEqual(["opening_present"], duplicate["duplicate_checkpoints"])

    def test_incomplete_structured_generation_retries_before_decision(self) -> None:
        incomplete = '{"checkpoint_analysis": [], "compliance": "compliant"}'
        runtime = FakeRuntime([incomplete, FAKE_STRUCTURED_JSON])

        prediction, runtime = self._predict(structured=True, runtime=runtime)

        self.assertTrue(prediction["structured_ok"])
        self.assertEqual(2, prediction["structured_attempts"])
        self.assertEqual(2, runtime.generation_calls)

    def test_prediction_cache_requires_aligned_label_and_confidence(self) -> None:
        prediction, _ = self._predict(structured=True)
        self.assertTrue(prediction_record_is_compatible(prediction))
        self.assertEqual("max_1080p_v1", prediction["model_input_policy"])

        wrong_label = {**prediction, "confidence_label": "compliant"}
        self.assertFalse(prediction_record_is_compatible(wrong_label))

        wrong_value = {**prediction, "confidence": 0.6}
        self.assertFalse(prediction_record_is_compatible(wrong_value))

        old_input_policy = {**prediction, "model_input_policy": "unbounded"}
        self.assertFalse(prediction_record_is_compatible(old_input_policy))

    def test_structured_prompt_carries_checkpoints_and_no_human_answer(self) -> None:
        pair = {"pair_id": "p1", "rule_name": "Opening cover missing", "compliance_label": "non_compliant"}
        rule = {
            "visual_screening_rule": "opening_present == yes AND cover_sufficient == no",
            "visual_checkpoints": {"opening_present": "Opening present?"},
            "required_checkpoints": ["opening_present"],
            "critical_checkpoints": ["opening_present"],
        }
        prompt = build_structured_prompt(pair, rule)
        self.assertIn("opening_present", prompt)
        self.assertIn("Opening cover missing", prompt)
        self.assertNotIn('"compliance_label"', prompt)


if __name__ == "__main__":
    unittest.main()
