"""Coarse-to-fine rule hierarchy and the prompt rendering for VLM retrieval.

Pure logic (no model; only reads the rule files and the prompt templates the
caller points it at), so it can be unit-tested without a GPU. The hierarchy is
the means by which we avoid showing the VLM all rules at once AND avoid a slow
per-rule query:

  Stage A  image -> which of the 4 hazard CATEGORIES is visible  (A/B/C/D)
  Stage B  for each selected category, ONE generation call returns the top-K
           most relevant rules among that category's subset rules (listed,
           grouped by sub-type) -- not one query per rule.

The prompt text lives as templates under Prompt/ (retrieval_category_prompt.md,
retrieval_rules_prompt.md); this module only fills their placeholders.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

OPTIONS = ("A", "B", "C", "D")

# Stage A sorts photos by SCENE/SUBJECT, never by hazard. The four benchmark
# major categories are hazard-named ("... Hazard"); showing those names makes the
# VLM hunt for a violation and miss compliant scenes (a properly covered opening
# is still an "opening" scene and must be matched). So each category is shown as a
# neutral scene description that names the subject in EITHER state -- open or
# covered, railed or not, helmeted or bare -- so compliant subjects match too.
# Decoding still uses the real major_category; only the displayed label changes.
CATEGORY_SCENES: dict[str, str] = {
    "Opening Hazard": (
        "Floor openings and shafts — a reserved opening in a floor or slab "
        "(left open OR covered by a board), an elevator hoistway opening, or a "
        "service/vent/utility shaft opening"
    ),
    "Edge Hazard": (
        "Edges of a raised surface — the perimeter of a floor, roof, balcony or "
        "platform with a drop, with OR without an edge guardrail/railing"
    ),
    "Unsafe Behavior": (
        "People at work — a worker up high (with OR without a harness), heads "
        "with OR without helmets, someone operating a crane/excavator, or "
        "electrical wiring work"
    ),
    "Civilized Construction Hazard": (
        "Site grounds and facilities — boundary hoarding/fencing, stacked "
        "materials, site roads and dust, fire access, gate signboards, "
        "dormitories/offices, and warning/reflective markings (present OR absent)"
    ),
}


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_subset_rules(rules_path: Path, subset_path: Path) -> list[dict[str, Any]]:
    """The benchmark-subset rules, in their order inside rule.json."""
    selected = set(json.loads(subset_path.read_text(encoding="utf-8"))["selected_rule_ids"])
    rules = [r for r in json.loads(rules_path.read_text(encoding="utf-8")) if r["rule_id"] in selected]
    missing = selected - {r["rule_id"] for r in rules}
    if missing:
        raise KeyError(f"Subset rule_ids missing from {rules_path}: {sorted(missing)}")
    return rules


def category_options(rules: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """[(letter, major_category)] for the (exactly 4) hazard categories, sorted.

    The sort makes the letter<->category mapping deterministic so Stage A can be
    decoded with the runtime's fixed A/B/C/D token odds.
    """
    categories = sorted({r["major_category"] for r in rules})
    if len(categories) != len(OPTIONS):
        raise ValueError(
            f"Stage A maps categories to A/B/C/D and needs exactly {len(OPTIONS)} "
            f"categories; got {len(categories)}: {categories}"
        )
    return list(zip(OPTIONS, categories, strict=True))


def rules_in_category(rules: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return [r for r in rules if r["major_category"] == category]


def rule_subject_text(rule: dict[str, Any]) -> str:
    """Short subject cue for a rule: the (subject-anchored) retrieval text,
    falling back to the visual-checkpoint questions."""
    text = str(rule.get("visual_retrieval_text", "")).strip()
    if text:
        return text
    return "; ".join(str(v) for v in rule.get("visual_checkpoints", {}).values())


def render_category_prompt(template: str, options: list[tuple[str, str]]) -> str:
    """Stage A: pick the SCENE the photo shows. Each option is shown as a neutral
    scene/subject description (CATEGORY_SCENES), not the hazard-named category, so
    the model sorts by setting rather than hunting for a violation. The letter ->
    category mapping (used to decode the answer) is unchanged."""
    lines = "\n".join(
        f"{letter} = {CATEGORY_SCENES.get(category, category)}" for letter, category in options
    )
    return template.replace("<<CATEGORY_OPTIONS>>", lines)


def render_rules_prompt(
    template: str,
    category: str,
    category_rules: list[dict[str, Any]],
    top_k: int,
) -> tuple[str, dict[int, str]]:
    """Stage B: list a category's rules (numbered, grouped by sub-type) and ask
    for the top-K most relevant in one call.

    Returns (prompt, index_to_rule_id) so the caller can map the model's returned
    numbers back to rule_ids.
    """
    by_sub: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for rule in category_rules:
        by_sub.setdefault(rule.get("subcategory", ""), []).append(rule)

    lines: list[str] = []
    index_to_rule: dict[int, str] = {}
    n = 1
    for sub, sub_rules in by_sub.items():
        if sub:
            lines.append(f"[{sub}]")
        for rule in sub_rules:
            lines.append(f"{n}. {rule.get('rule_name', '')} — {rule_subject_text(rule)}")
            index_to_rule[n] = rule["rule_id"]
            n += 1

    prompt = (
        template.replace("<<CATEGORY>>", category)
        .replace("<<TOP_K>>", str(top_k))
        .replace("<<RULE_LIST>>", "\n".join(lines))
    )
    return prompt, index_to_rule


def parse_selected_indices(text: str, index_to_rule: dict[int, str]) -> list[str]:
    """Parse the model's '3, 1' style answer into rule_ids, in returned order.

    Keeps only valid, in-range indices; de-duplicates preserving order; an empty
    list (e.g. the model said 'none') is a valid 'no relevant rule' answer.
    """
    selected: list[str] = []
    seen: set[int] = set()
    for match in re.findall(r"\d+", text):
        idx = int(match)
        if idx in index_to_rule and idx not in seen:
            seen.add(idx)
            selected.append(index_to_rule[idx])
    return selected
