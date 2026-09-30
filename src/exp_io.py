"""Shared experiment-side IO for the R-CES Phase-2 pipeline.

Centralises the experiment-side data derivations that the annotation pipeline
does not need:

  1. loading the *locked* gold and deriving per-checkpoint `resolvable`
     supervision plus the independent external-evidence gate;
  2. the image-disjoint split lookup.

It reuses the annotation line's `rces_io` (paths, label vocab, rule loaders,
image encoding) so the two stay consistent; nothing here writes gold.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# This module lives in src; make src importable so `config` and the
# annotation line's `rces_io` resolve even if a caller didn't set the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import rces_io as rio  # noqa: E402  (paths, vocab, rule IO)


# --- Gold loading -------------------------------------------------------------
def _locked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("annotation_status") in config.LOCKED_STATUSES]


def load_locked_crop() -> list[dict[str, Any]]:
    """Locked crop-augmentation rows (random crops of the original pilot images).

    Same record shape as gold; each row carries `image_split` (its mother
    image's split) so downstream split assignment stays leakage-free. Empty when
    the file is absent.
    """
    if not rio.CROP_AUG_PATH.exists():
        return []
    return _locked(rio.load_jsonl(rio.CROP_AUG_PATH))


def load_locked_gold() -> list[dict[str, Any]]:
    """Gold rows whose human judgment is locked (judged/described/reviewed).

    These are the only rows usable as supervision or evaluation gold. Pending
    rows (still `to_be_annotated`) are skipped, so the pipeline runs cleanly on a
    partially annotated file and simply grows as annotation completes. When
    config.INCLUDE_CROP_AUGMENTATION is set, the locked crop-augmentation rows
    are appended (their split is honoured by the split builder, not here).
    """
    rows = _locked(rio.load_jsonl(rio.GOLD_PATH))
    if config.INCLUDE_CROP_AUGMENTATION:
        rows = rows + load_locked_crop()
    return rows


def all_pair_image_ids() -> list[str]:
    """Every image_id referenced by the gold pair file (annotated or not) -- the
    fixed universe the deterministic split is built over. Crop-augmentation
    images are deliberately EXCLUDED here so adding crops never perturbs the
    deterministic re-split of the 250 originals; they are appended to their own
    split bucket by the split builder instead."""
    rows = rio.load_jsonl(rio.GOLD_PATH)
    return sorted({r["image_id"] for r in rows})


def crop_image_ids_by_split() -> dict[str, list[str]]:
    """Crop image_ids grouped by their carried `image_split`. Empty when the
    crop file is absent or the augmentation arm is off."""
    if not config.INCLUDE_CROP_AUGMENTATION:
        return {}
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in load_locked_crop():
        grouped[row.get("image_split", "")].add(row["image_id"])
    return {split: sorted(ids) for split, ids in grouped.items()}


def crop_image_ids() -> list[str]:
    """Flat sorted list of all crop image_ids in the augmentation arm."""
    ids: set[str] = set()
    for split_ids in crop_image_ids_by_split().values():
        ids.update(split_ids)
    return sorted(ids)


def group_by_image(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["image_id"]].append(row)
    return dict(grouped)


# --- Per-checkpoint supervision + independent external-evidence gate -----------
def critical_checkpoints(rule: dict[str, Any]) -> list[str]:
    _, critical, _ = rio.checkpoints_for(rule)
    return critical


def resolvable_targets(pair: dict[str, Any], rule: dict[str, Any]) -> dict[str, int]:
    """Per-critical-checkpoint resolvable gold (Readme step 7 dense supervision).

    resolvable(k, x, r) = 0 iff k is in the human-annotated missing_checkpoints,
    else 1 (observed or not-applicable -> 1; na->1 confirmed by the maintainer).
    Returned only over the rule's critical checkpoints -- the atomic units R-CES
    aggregates. A checkpoint-less rule (e.g. the TSP/dust external rules) yields
    an empty dict; its (in)sufficiency comes from the rule-side prior instead.
    """
    missing = set(pair.get("missing_checkpoints", []))
    return {k: (0 if k in missing else 1) for k in critical_checkpoints(rule)}


def is_visual_rule(rule: dict[str, Any]) -> bool:
    """Rule-side is_visual prior: whether the rule is decidable from vision in
    principle. requires_external_evidence rules (mm / certificate / records) are
    non-visual-in-principle -> undecidable regardless of any checkpoint."""
    return rule.get("decision_scope") != "requires_external_evidence"


def external_evidence_required(rule: dict[str, Any]) -> bool:
    """Whether final answerability needs a post-head external-evidence gate."""
    return not is_visual_rule(rule)


def pair_sufficiency_gold(pair: dict[str, Any]) -> int:
    """R-CES binary gold: 1 iff evidence is sufficient (== decidable). Derived,
    never hand-typed; here we read the already-derived evidence_sufficiency_label."""
    return 1 if pair.get("evidence_sufficiency_label") == "sufficient" else 0


def pair_decidable_gold(pair: dict[str, Any]) -> int:
    return 1 if pair.get("decidability") == "decidable" else 0


def checkpoint_question(rule: dict[str, Any], key: str) -> str:
    """The human-readable checkpoint question used as the text-conditioning input
    (falls back to a spaced key if a question text is missing)."""
    _, _, questions = rio.checkpoints_for(rule)
    return questions.get(key) or key.replace("_", " ")


def checkpoint_text_id(rule_id: str, key: str) -> str:
    return f"{rule_id}::{key}"
