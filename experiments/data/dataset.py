"""Datasets bridging cached SigLIP 2 features and R-CES supervision.

Two views of the same locked gold:

  * checkpoint-level (training): one example per (pair, critical checkpoint),
    target = resolvable(k, x, r) in {0, 1}. This is the dense supervision.
  * pair-level (evaluation/aggregation): one record per pair, carrying the
    rule's critical keys, external-evidence requirement, and human-derived gold used by
    aggregate.py / evaluate.py.

Ablation hooks (config flags resolved at build time):
  ABLATE_NO_CHECKPOINT_SUPERVISION -> pair-level training examples (one per pair,
    target = pair sufficiency, text = mean of the rule's critical checkpoint texts).
"""

from __future__ import annotations

import random

import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

import config
import exp_io


# --- Feature tables -----------------------------------------------------------
def validate_image_feature_cache() -> None:
    if not config.FEATURE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Run the active encoder first: {config.FEATURE_MANIFEST} missing"
        )
    manifest = json.loads(config.FEATURE_MANIFEST.read_text(encoding="utf-8"))
    actual_size = manifest.get("image_size")
    if actual_size != config.VISION_IMAGE_SIZE:
        raise ValueError(
            f"Feature cache image_size={actual_size!r} does not match active "
            f"image_size={config.VISION_IMAGE_SIZE}; rebuild {config.FEATURE_DIR}"
        )
    actual_patches = manifest.get("n_patch")
    if actual_patches != config.N_PATCH:
        raise ValueError(
            f"Feature cache n_patch={actual_patches!r} does not match active "
            f"n_patch={config.N_PATCH}; rebuild {config.FEATURE_DIR}"
        )


def load_checkpoint_text() -> dict[str, torch.Tensor]:
    if not config.CKPT_TEXT_PATH.exists():
        raise FileNotFoundError(f"Run encode_siglip2.py first: {config.CKPT_TEXT_PATH} missing")
    return torch.load(config.CKPT_TEXT_PATH)


def image_feature_path(image_id: str):
    return config.IMG_FEATURE_DIR / f"{image_id}.pt"


def load_image_features(image_ids: set[str]) -> dict[str, torch.Tensor]:
    validate_image_feature_cache()
    table: dict[str, torch.Tensor] = {}
    for image_id in image_ids:
        path = image_feature_path(image_id)
        if not path.exists():
            raise FileNotFoundError(f"Run encode_siglip2.py first: {path} missing")
        table[image_id] = torch.load(path)
    return table


# --- Pair-level records -------------------------------------------------------
@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    image_id: str
    rule_id: str
    decision_scope: str
    critical_keys: tuple[str, ...]
    resolvable_targets: dict[str, int]
    external_evidence_required: bool
    sufficiency_gold: int
    decidable_gold: int
    compliance_gold: str
    critical_missing_gold: tuple[str, ...]


def _rule_allowed(rule_id: str) -> bool:
    """Honour config.RULE_FILTER so a fold can train and test on disjoint rules."""
    rule_filter = config.RULE_FILTER
    if rule_filter is None:
        return True
    mode, rule_ids = rule_filter
    if mode == "exclude":
        return rule_id not in rule_ids
    if mode == "include":
        return rule_id in rule_ids
    raise ValueError(f"RULE_FILTER mode must be exclude or include; got {mode!r}")


def build_pair_records(image_ids: set[str]) -> list[PairRecord]:
    """Locked-gold pairs whose image is in `image_ids`, as pair-level records."""
    rules = exp_io.rio.load_rules()
    records: list[PairRecord] = []
    for pair in exp_io.load_locked_gold():
        if pair["image_id"] not in image_ids:
            continue
        if not _rule_allowed(pair["rule_id"]):
            continue
        rule = rules.get(pair["rule_id"], {})
        targets = exp_io.resolvable_targets(pair, rule)
        records.append(
            PairRecord(
                pair_id=pair["pair_id"],
                image_id=pair["image_id"],
                rule_id=pair["rule_id"],
                decision_scope=pair.get("decision_scope", ""),
                critical_keys=tuple(targets.keys()),
                resolvable_targets=targets,
                external_evidence_required=exp_io.external_evidence_required(rule),
                sufficiency_gold=exp_io.pair_sufficiency_gold(pair),
                decidable_gold=exp_io.pair_decidable_gold(pair),
                compliance_gold=pair.get("compliance_label", "unknown"),
                critical_missing_gold=tuple(pair.get("critical_missing_checkpoints", [])),
            )
        )
    return records


# --- Checkpoint-level training examples ---------------------------------------
@dataclass(frozen=True)
class Example:
    pair_id: str
    image_id: str
    rule_id: str
    checkpoint_key: str  # "__pair__" for the no-checkpoint-supervision ablation
    text_id: str  # key into the text feature table
    target: float
    is_visual: float


def build_examples(records: list[PairRecord]) -> list[Example]:
    """Per-(pair, critical checkpoint) examples, or per-pair examples under the
    no-checkpoint-supervision ablation."""
    examples: list[Example] = []
    if config.ABLATE_NO_CHECKPOINT_SUPERVISION:
        for rec in records:
            examples.append(
                Example(
                    pair_id=rec.pair_id,
                    image_id=rec.image_id,
                    rule_id=rec.rule_id,
                    checkpoint_key="__pair__",
                    text_id="__pair__",  # mean of the rule's critical checkpoint texts
                    target=float(rec.sufficiency_gold),
                    is_visual=float(not rec.external_evidence_required),
                )
            )
        return examples
    for rec in records:
        for key, target in rec.resolvable_targets.items():
            examples.append(
                Example(
                    pair_id=rec.pair_id,
                    image_id=rec.image_id,
                    rule_id=rec.rule_id,
                    checkpoint_key=key,
                    text_id=exp_io.checkpoint_text_id(rec.rule_id, key),
                    target=float(target),
                    is_visual=float(not rec.external_evidence_required),
                )
            )
    return examples


# --- torch Dataset ------------------------------------------------------------
class RCESDataset(Dataset):
    """Assembles (patches, conditioning text, is_visual) -> target tensors.

    Holds the in-memory feature tables (image set is <= 250, so eager loading is
    appropriate for the pilot).
    """

    def __init__(self, examples: list[Example], records: list[PairRecord]):
        self.examples = examples
        self.records_by_pair = {r.pair_id: r for r in records}
        image_ids = {ex.image_id for ex in examples}
        self.images = load_image_features(image_ids)
        self.ckpt_text = load_checkpoint_text()
        # Shuffled-image control: deterministically re-assign each pair's image to
        # another image in the same evaluation set. Preserves the image marginal
        # and every rule-side field, so any AUROC drop is attributable to the
        # image->pair correspondence alone.
        self.image_permutation: dict[str, str] = {}
        if config.SHUFFLE_IMAGE_EVAL:
            ordered = sorted(image_ids)
            rng = random.Random(config.SHUFFLE_IMAGE_SEED)
            shuffled = ordered[:]
            for _ in range(100):
                rng.shuffle(shuffled)
                if all(a != b for a, b in zip(ordered, shuffled)) or len(ordered) < 2:
                    break
            self.image_permutation = dict(zip(ordered, shuffled))

    def __len__(self) -> int:
        return len(self.examples)

    def _text_vec(self, ex: Example) -> torch.Tensor:
        if config.ABLATE_NO_TEXT_CONDITIONING:
            return torch.zeros_like(next(iter(self.ckpt_text.values())))
        if ex.text_id == "__pair__":
            rec = self.records_by_pair[ex.pair_id]
            keys = [exp_io.checkpoint_text_id(rec.rule_id, k) for k in rec.critical_keys]
            vecs = [self.ckpt_text[k] for k in keys if k in self.ckpt_text]
            if vecs:
                return torch.stack(vecs).mean(0)
            # Checkpoint-less rule: fall back to a zero conditioning vector.
            return torch.zeros_like(next(iter(self.ckpt_text.values())))
        return self.ckpt_text[ex.text_id]

    def _patches(self, ex: Example) -> torch.Tensor:
        image_id = self.image_permutation.get(ex.image_id, ex.image_id)
        return self.images[image_id]  # [N_patch, D_img]

    def __getitem__(self, index: int) -> dict[str, Any]:
        ex = self.examples[index]
        return {
            "patches": self._patches(ex),
            "text": self._text_vec(ex),
            "is_visual": torch.tensor([ex.is_visual], dtype=torch.float32),
            "target": torch.tensor([ex.target], dtype=torch.float32),
            "pair_id": ex.pair_id,
            "checkpoint_key": ex.checkpoint_key,
        }
