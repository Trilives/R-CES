"""Hierarchical local-VLM rule retrieval (Stage A category -> Stage B rules).

For each image in the gold pair file, the local Qwen3.5 runtime first picks the
visible hazard CATEGORY (Stage A, decoded from the A/B/C/D token odds), then
then, for each selected category, makes ONE generation call that returns the
top-K most relevant rules of that category (Stage B). The model is therefore
never shown all 23 rules at once and is not queried once per rule. Output is one
row per image:

  {image_id, method, model, category_probs, selected_categories,
   rule_relevance: {rule_id: rank_score}, applicable_rules: [rule_id ...],
   generations: {category: raw_text}}

Offline, local model only. Resumable: rows are flushed after every image and
reused on the next run (set RETRIEVAL_VLM_OVERWRITE=True to ignore the cache).
Run controls are constants in src/config.py.

  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/retrieval/vlm_retrieve.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import config  # noqa: E402
import exp_io  # noqa: E402
import hierarchy as H  # noqa: E402
from local_vlm.runtime import DryRunOptionRuntime, Qwen35OptionRuntime  # noqa: E402


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.RCES_ROOT))
    except ValueError:
        return str(path)


def retrieve_image(runtime, image_path: Path, rules, options, prompts) -> dict:
    """Run Stage A + Stage B for one image; return its retrieval row body.

    Stage A: one A/B/C/D call -> category probabilities -> selected categories.
    Stage B: ONE generation call per selected category returns the top-K most
    relevant rules of that category (the model ranks them; we don't query per
    rule). `rule_relevance` is a rank score in (0, 1]: 1.0 for the top pick,
    decreasing with position; 0 for rules the model did not return.
    """
    category_template, rules_template = prompts
    cat_pred = runtime.predict_option(image_path, H.render_category_prompt(category_template, options))
    category_probs = {
        category: float(cat_pred.probabilities.get(letter, 0.0))
        for letter, category in options
    }
    # The model may pick 1 or 2 scenes: always the top category, plus the single
    # runner-up when it is also clearly present (prob >= threshold). Capped at 2.
    ranked_categories = sorted(category_probs, key=category_probs.__getitem__, reverse=True)
    selected = [ranked_categories[0]]
    if len(ranked_categories) > 1 and category_probs[ranked_categories[1]] >= config.RETRIEVAL_CATEGORY_THRESHOLD:
        selected.append(ranked_categories[1])
    selected_categories = sorted(selected)

    top_k = config.RETRIEVAL_TOP_K_PER_CATEGORY
    rule_relevance: dict[str, float] = {}
    generations: dict[str, str] = {}
    for category in selected_categories:
        category_rules = H.rules_in_category(rules, category)
        prompt, index_to_rule = H.render_rules_prompt(rules_template, category, category_rules, top_k)
        text = runtime.generate_text(
            image_path, prompt, max_new_tokens=config.RETRIEVAL_GEN_MAX_NEW_TOKENS
        )
        generations[category] = text
        selected = H.parse_selected_indices(text, index_to_rule)[:top_k]
        for rank, rule_id in enumerate(selected):
            # Higher score for earlier picks; keep the best if a rule recurs.
            score = 1.0 - rank / max(top_k, 1)
            rule_relevance[rule_id] = max(rule_relevance.get(rule_id, 0.0), score)

    applicable_rules = sorted(
        rule_relevance, key=lambda rid: (-rule_relevance[rid], rid)
    )
    return {
        "category_probs": category_probs,
        "selected_categories": selected_categories,
        "rule_relevance": rule_relevance,
        "applicable_rules": applicable_rules,
        "generations": generations,
    }


def main() -> None:
    rules = H.load_subset_rules(exp_io.rio.RULES_PATH, config.RETRIEVAL_SUBSET_PATH)
    options = H.category_options(rules)
    image_ids = exp_io.all_pair_image_ids()
    if config.RETRIEVAL_LIMIT_IMAGES > 0:
        image_ids = image_ids[: config.RETRIEVAL_LIMIT_IMAGES]
    if not image_ids:
        raise RuntimeError("No images in the gold pair file for VLM retrieval")

    runtime = (
        DryRunOptionRuntime()
        if config.RETRIEVAL_VLM_DRY_RUN
        else Qwen35OptionRuntime(
            config.LOCAL_QWEN35_DIR,
            dtype=config.LOCAL_QWEN_DTYPE,
            device_map=config.LOCAL_QWEN_DEVICE_MAP,
        )
    )
    prompts = (
        H.load_prompt(config.RETRIEVAL_CATEGORY_PROMPT_PATH),
        H.load_prompt(config.RETRIEVAL_RULES_PROMPT_PATH),
    )

    config.RETRIEVAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_id: dict[str, dict] = {}
    if config.RETRIEVAL_VLM_PATH.exists() and not config.RETRIEVAL_VLM_OVERWRITE:
        rows_by_id = {
            row["image_id"]: row
            for row in exp_io.rio.load_jsonl(config.RETRIEVAL_VLM_PATH)
            if row.get("model") == runtime.model_name
        }

    for index, image_id in enumerate(image_ids, 1):
        if image_id in rows_by_id:
            print(f"[{index}/{len(image_ids)}] {image_id} cached")
            continue
        image_path = exp_io.rio.IMAGE_DIR / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        body = retrieve_image(runtime, image_path, rules, options, prompts)
        rows_by_id[image_id] = {
            "image_id": image_id,
            "method": "vlm_hierarchical",
            "model": runtime.model_name,
            **body,
        }
        ordered = [rows_by_id[i] for i in image_ids if i in rows_by_id]
        exp_io.rio.write_jsonl(config.RETRIEVAL_VLM_PATH, ordered, sort_keys=True)
        print(
            f"[{index}/{len(image_ids)}] {image_id} -> "
            f"{body['selected_categories']} | {len(body['applicable_rules'])} applicable rules"
        )

    rows = [rows_by_id[i] for i in image_ids if i in rows_by_id]
    exp_io.rio.write_jsonl(config.RETRIEVAL_VLM_PATH, rows, sort_keys=True)
    summary = {
        "method": "vlm_hierarchical",
        "model": runtime.model_name,
        "dry_run": config.RETRIEVAL_VLM_DRY_RUN,
        "image_count": len(rows),
        "rule_pool": len(rules),
        "category_threshold": config.RETRIEVAL_CATEGORY_THRESHOLD,
        "top_k_per_category": config.RETRIEVAL_TOP_K_PER_CATEGORY,
        "output_path": display_path(config.RETRIEVAL_VLM_PATH),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {display_path(config.RETRIEVAL_VLM_PATH)}")


if __name__ == "__main__":
    main()
