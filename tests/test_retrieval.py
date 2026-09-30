"""Contract tests for the rule-retrieval comparison experiment (no GPU)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

RCES_ROOT = Path(__file__).resolve().parents[1]
for path in (RCES_ROOT / "src", RCES_ROOT / "experiments", RCES_ROOT / "experiments" / "retrieval"):
    sys.path.insert(0, str(path))

import config  # noqa: E402
import rces_io as rio  # noqa: E402
import hierarchy as H  # noqa: E402
import vlm_retrieve  # noqa: E402
from local_vlm.runtime import OptionPrediction  # noqa: E402
from compare import auroc, average_precision, prf  # noqa: E402


class FixedRuntime:
    """Fixed A/B/C/D distribution for Stage A; fixed top-K answer for Stage B."""

    model_name = "fixed-test-runtime"

    def __init__(self, probabilities: dict[str, float], generation: str = "1, 2") -> None:
        self.probabilities = probabilities
        self.generation = generation

    def predict_option(self, image_path: Path, prompt: str) -> OptionPrediction:
        option = max(self.probabilities, key=self.probabilities.__getitem__)
        return OptionPrediction(
            option=option,
            confidence=self.probabilities[option],
            probabilities=dict(self.probabilities),
        )

    def generate_text(self, image_path: Path, prompt: str, *, max_new_tokens: int = 48) -> str:
        return self.generation


class HierarchyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = H.load_subset_rules(rio.RULES_PATH, config.RETRIEVAL_SUBSET_PATH)

    def test_subset_loads_and_spans_four_categories(self) -> None:
        self.assertEqual(len(self.rules), 23)
        options = H.category_options(self.rules)
        self.assertEqual([letter for letter, _ in options], ["A", "B", "C", "D"])
        # Categories are sorted, so A maps to the alphabetically-first category.
        self.assertEqual(options[0][1], "Civilized Construction Hazard")
        covered = {category for _, category in options}
        self.assertEqual(len(covered), 4)

    def test_rules_in_category_partitions_the_subset(self) -> None:
        options = H.category_options(self.rules)
        total = sum(len(H.rules_in_category(self.rules, c)) for _, c in options)
        self.assertEqual(total, len(self.rules))

    def test_category_prompt_fills_options(self) -> None:
        options = H.category_options(self.rules)
        template = H.load_prompt(config.RETRIEVAL_CATEGORY_PROMPT_PATH)
        cat_prompt = H.render_category_prompt(template, options)
        self.assertNotIn("<<CATEGORY_OPTIONS>>", cat_prompt)
        # Stage A sorts by scene, so options show the neutral scene description,
        # not the hazard-named category.
        self.assertIn(H.CATEGORY_SCENES["Opening Hazard"], cat_prompt)
        self.assertNotIn("Hazard", cat_prompt)

    def test_rules_prompt_numbers_rules_and_maps_back(self) -> None:
        template = H.load_prompt(config.RETRIEVAL_RULES_PROMPT_PATH)
        cat_rules = H.rules_in_category(self.rules, "Opening Hazard")
        prompt, idx = H.render_rules_prompt(template, "Opening Hazard", cat_rules, top_k=3)
        self.assertNotIn("<<RULE_LIST>>", prompt)
        self.assertIn("Opening Hazard", prompt)
        self.assertEqual(set(idx.values()), {r["rule_id"] for r in cat_rules})
        self.assertEqual(sorted(idx), list(range(1, len(cat_rules) + 1)))

    def test_parse_selected_indices(self) -> None:
        idx = {1: "r-a", 2: "r-b", 3: "r-c"}
        self.assertEqual(H.parse_selected_indices("3, 1", idx), ["r-c", "r-a"])
        self.assertEqual(H.parse_selected_indices("none", idx), [])
        # out-of-range and duplicates are dropped.
        self.assertEqual(H.parse_selected_indices("2 2 9", idx), ["r-b"])


class VlmRetrieveTest(unittest.TestCase):
    def test_retrieve_image_selects_category_then_top_k_rules(self) -> None:
        rules = H.load_subset_rules(rio.RULES_PATH, config.RETRIEVAL_SUBSET_PATH)
        options = H.category_options(rules)
        prompts = (
            H.load_prompt(config.RETRIEVAL_CATEGORY_PROMPT_PATH),
            H.load_prompt(config.RETRIEVAL_RULES_PROMPT_PATH),
        )
        # Stage A favours category A (Civilized); Stage B returns rules 1 and 2.
        runtime = FixedRuntime({"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}, generation="1, 2")
        body = vlm_retrieve.retrieve_image(runtime, Path("dummy.jpg"), rules, options, prompts)

        self.assertEqual(body["selected_categories"], ["Civilized Construction Hazard"])
        civ_rules = H.rules_in_category(rules, "Civilized Construction Hazard")
        # Numbers map through the SAME subcategory-grouped order the prompt used.
        _, idx = H.render_rules_prompt(prompts[1], "Civilized Construction Hazard", civ_rules, 3)
        # Only the top-2 the model returned ("1, 2") are applicable, not the whole category.
        self.assertEqual(body["applicable_rules"], [idx[1], idx[2]])
        self.assertAlmostEqual(body["rule_relevance"][idx[1]], 1.0)
        self.assertAlmostEqual(body["rule_relevance"][idx[2]], 1.0 - 1 / 3)
        self.assertAlmostEqual(body["category_probs"]["Civilized Construction Hazard"], 0.7)


class CompareMetricsTest(unittest.TestCase):
    def test_auroc_separable_and_inverted(self) -> None:
        self.assertEqual(auroc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]), 1.0)
        self.assertEqual(auroc([0.9, 0.8, 0.1, 0.2], [0, 0, 1, 1]), 0.0)

    def test_auroc_single_class_is_none(self) -> None:
        self.assertIsNone(auroc([0.1, 0.2], [1, 1]))

    def test_auroc_handles_ties(self) -> None:
        # All scores tied -> AUROC 0.5 (no separation).
        self.assertAlmostEqual(auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5)

    def test_average_precision_and_prf(self) -> None:
        self.assertEqual(average_precision([0.9, 0.1], [1, 0]), 1.0)
        scores = prf(tp=2, fp=1, fn=1)
        self.assertAlmostEqual(scores["precision"], 2 / 3)
        self.assertAlmostEqual(scores["recall"], 2 / 3)
        self.assertAlmostEqual(scores["f1"], 2 / 3)


class ConfigSwitchTest(unittest.TestCase):
    def test_active_retrieval_path_follows_method(self) -> None:
        self.assertIn(config.RETRIEVAL_METHOD, config.RETRIEVAL_METHODS)
        # Default ships as siglip2.
        self.assertEqual(config.active_retrieval_path(), config.RETRIEVAL_SIGLIP_PATH)


if __name__ == "__main__":
    unittest.main()
