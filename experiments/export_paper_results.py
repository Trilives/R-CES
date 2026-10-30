"""Prepare the image-free, CPU-verifiable archive from original run artifacts.

Run after archive_camera_ready.py, within_stratum.py and the channel controls.
Outputs experiments/artifacts/paper_release/ for a scrubbed copy into the public
repository's results/acml2026/. No photographs, narration or weights are exported.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
import config
import exp_io
from baselines import cascade
from data import splits
from local_vlm.predictor import prediction_record_is_compatible
from verify_paper_results import compute_tables, annotation_fingerprint

ART = ROOT / "experiments/artifacts"
AUDIT = ART / "camera_ready_audit"
OUTPUT = ART / "paper_release"
VLMS = ("InternVL3_5-8B-HF", "gemma-4-E2B-it", "Qwen3.5-4B", "gemma-4-E4B-it",
        "Qwen3.6-35B-A3B", "Qwen3-VL-8B-Instruct", "Qwen3.5-9B")


def normalise(value):
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(normalise(value), ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
    path.write_bytes(gzip.compress(content, mtime=0) if str(path).endswith(".gz") else content)


def add_ids(frame, pairs):
    frame = dict(frame)
    frame["family_id"] = exp_io.original_image_families(frame["image_id"])
    if "pair_id" not in frame:
        rule_ids = frame.get("rule_id") or [key.split("::", 1)[0] for key in frame["key"]]
        frame["pair_id"] = [pairs[(image, rule)]["pair_id"] for image, rule in zip(frame["image_id"], rule_ids)]
    return frame


def export_answerer(vlm, gold, split):
    directory = ART / config._LOCAL_QWEN_ARTIFACT_NAMES[vlm]
    predictions = {r["pair_id"]: r for r in exp_io.rio.load_jsonl(directory / "compliance_predictions_structured_v2.jsonl") if prediction_record_is_compatible(r)}
    self_eval = {r["pair_id"]: r for r in exp_io.rio.load_jsonl(directory / "self_eval_predictions.jsonl")}
    records = []
    for subset in ("val", "test"):
        ids = set(split[subset])
        for pair in gold:
            if pair["image_id"] not in ids:
                continue
            prediction = predictions[pair["pair_id"]]
            dist = np.asarray([prediction["label_probabilities"].get(c, 0) for c in cascade.CLASS_ORDER], dtype=float)
            dist /= dist.sum()
            se = self_eval.get(pair["pair_id"], {})
            correct = int(pair["evidence_sufficiency_label"] == "sufficient" and prediction["predicted_compliance"] == pair["compliance_label"])
            records.append({"pair_id": pair["pair_id"], "image_id": pair["image_id"],
                "family_id": exp_io.original_image_family(pair["image_id"]), "rule_id": pair["rule_id"], "split": subset,
                "predicted_compliance": prediction["predicted_compliance"], "gold_compliance": pair["compliance_label"],
                "sufficiency_gold": int(pair["evidence_sufficiency_label"] == "sufficient"), "correct": correct,
                "label_correct": int(prediction["predicted_compliance"] == pair["compliance_label"]),
                "label_probabilities": dict(zip(cascade.CLASS_ORDER, dist.tolist())),
                "scores": {"maxprob": prediction["confidence"], "entropy": cascade._entropy_confidence(dist),
                           "margin": cascade._margin_confidence(dist), "ptrue": float(se.get("ptrue", 0.0)),
                           "verbalized": float(se.get("verbalized", 0.0))}})
    val = [r for r in records if r["split"] == "val" and r["gold_compliance"] in cascade.CLASS_ORDER]
    logits = np.log(np.clip(np.asarray([list(r["label_probabilities"].values()) for r in val]), 1e-12, 1))
    labels = np.asarray([cascade.CLASS_ORDER.index(r["gold_compliance"]) for r in val])
    temperature = cascade.fit_temperature(logits, labels)
    for record in records:
        record["scores"]["temp_scale"] = cascade._temperature_confidence(np.asarray(list(record["label_probabilities"].values())), temperature)
    original = read(ART / f"cascade_task2_{vlm}.json")
    test = [r for r in records if r["split"] == "test"]
    import metrics
    for method in ("maxprob", "temp_scale", "ptrue", "verbalized"):
        actual = metrics._auroc(np.asarray([r["scores"][method] for r in test]), np.asarray([r["correct"] for r in test]))
        if abs(actual - original["methods"][method]["auroc"]) > 1e-12:
            raise RuntimeError(f"{vlm}: archived {method} does not reproduce the original table input")
    return {"vlm": vlm, "temperature_fit_on": "validation_gold_compliance_classes", "temperature": temperature,
            "prediction_schema_version": 3, "model_input_policy": "max_1080p_v1", "rows": records}


def unaffected_report(value):
    """Retain AUROC/count inputs; do not republish obsolete risk or CI fields."""
    if isinstance(value, dict):
        return {k: unaffected_report(v) for k, v in value.items()
                if not any(part in k.lower() for part in ("aurc", "coverage", "cmcr", "ece", "brier", "bootstrap", "ci_", "risk", "n_groups"))}
    if isinstance(value, list):
        return [unaffected_report(v) for v in value]
    return value


def main():
    if not config.INCLUDE_CROP_AUGMENTATION:
        raise ValueError("Export the fixed paper population with RCES_INCLUDE_CROP_AUG=1")
    gold = exp_io.load_locked_gold()
    pair_map = {(r["image_id"], r["rule_id"]): r for r in gold}
    if len(pair_map) != len(gold):
        raise RuntimeError("Image/rule keys are not unique; supply explicit pair IDs")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for encoder in ("siglip2_512", "siglip2_224", "dinov2"):
        paths = sorted((AUDIT / f"camera_ready_scores_{encoder}_cropaug").glob("seed_*.json.gz"))
        if len(paths) != 10:
            raise RuntimeError(f"Waiting for ten {encoder} score archives")
        for path in paths:
            write(OUTPUT / "scores" / ("v1_" + encoder) / path.name, read(path))
    v2 = sorted((AUDIT / "v2_score_frames_cropaug").glob("seed_*.json.gz"))
    if len(v2) != 10:
        raise RuntimeError("Waiting for ten v2 score archives")
    for path in v2:
        value = read(path)
        for level in ("ckpt", "pair"):
            value[level] = add_ids(value[level], pair_map)
        write(OUTPUT / "scores/v2" / path.name, value)
    frames = read(ART / "hierarchical_frames_cropaug.json")
    for seed, (ckpt, pair) in enumerate(zip(frames["shuffled"]["ckpt"], frames["shuffled"]["pair"])):
        write(OUTPUT / f"scores/shuffled/seed_{seed:02d}.json.gz",
              {"seed": seed, "ckpt": add_ids(ckpt, pair_map), "pair": add_ids(pair, pair_map)})
    for arm in ("text_only", "image_only"):
        paths = sorted((AUDIT / "channel_score_frames_cropaug" / arm).glob("seed_*.json.gz"))
        if len(paths) != 10:
            raise RuntimeError(f"Waiting for ten {arm} score archives")
        for path in paths:
            value = read(path)
            frame = value["result"]["_frames"]
            write(OUTPUT / "scores" / arm / path.name,
                  {"seed": value["seed"], "ckpt": add_ids(frame["checkpoint"], pair_map), "pair": add_ids(frame["pair"], pair_map)})
    split = splits.load_split()
    for vlm in VLMS:
        write(OUTPUT / "inputs/answerers" / (vlm + ".json.gz"), export_answerer(vlm, gold, split))
    write(OUTPUT / "reports/iaa.json", read(ROOT / "data/pairs/iaa_report.json"))
    for filename in ("multiseed_ablation_summary_cropaug.json", "permutation_test_cropaug.json", "rule_disjoint_cropaug.json"):
        write(OUTPUT / "reports" / filename, unaffected_report(read(ART / filename)))
    write(OUTPUT / "reports/v2_ablation_summary.json", unaffected_report(read(ART / "rces-2/siglip2_512_cropaug/multiseed_ablation_summary.json")))
    for name in ("camera_ready_siglip2_512_cropaug.json", "camera_ready_siglip2_224_cropaug.json",
                 "camera_ready_dinov2_cropaug.json", "v2_within_stratum_cropaug.json", "evidence_controls_cropaug.json"):
        write(OUTPUT / "reports" / name, read(AUDIT / name))
    for name in ("models.lock.json", "validated_runtime.json"):
        write(OUTPUT / "configs" / name, read(ROOT / "configs" / name))
    write(OUTPUT / "configs/run.json", {
        "seeds": list(range(10)), "include_crop_augmentation": True,
        "split_seed": config.SPLIT_SEED, "split_ratios": config.SPLIT_RATIOS,
        "deterministic": True, "input_mode": "full", "aggregation": "min",
        "external_gate_score": 0.0, "shuffle_image_seed": 12345,
        "target_risk": 0.1, "ece_bins": 10,
        "v1_head": {k: getattr(config, k) for k in
                    ("HIDDEN_DIM", "DROPOUT", "EPOCHS", "BATCH_SIZE", "LR", "WEIGHT_DECAY", "GRAD_CLIP")},
        "calibration_answerer": "Qwen3.5-9B", "answerer_aliases": {"Qwen3.6-35B-A3B": "Qwen3.6-35B-A3B-FP8"},
        "split": split})
    source_files = [ROOT / "src/metrics.py", ROOT / "src/stratified.py", ROOT / "src/exp_io.py"]
    input_files = [p for p in (ROOT / "Prompt").glob("*.md")] + [ROOT / "data/rule/rule.json"]
    label_files = [ROOT / "data/pairs" / name for name in
                   ("rces_pilot_pairs.jsonl", "rces_crop_augmentation.jsonl")]
    write(OUTPUT / "manifest.json", {"archive_version": 1, "metric_revision": "tie_groups_and_original_image_families",
        "n_seeds": 10, "n_bootstrap": 3000, "bootstrap_seed": 0, "test_image_ids": 100, "test_original_image_families": 50,
        "risk_coverage": {"acceptance": "score >= threshold", "ties": "complete_equal_score_groups", "aurc": "right_endpoint_step_integral"},
        "pixels": "original_experimental_inputs", "images_included": False, "weights_required_for_verification": False,
        "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "input_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
        "annotation_semantic_hashes": {p.relative_to(ROOT).as_posix(): annotation_fingerprint(p) for p in label_files}})
    write(OUTPUT / "reports/paper_tables.json", compute_tables(OUTPUT))
    paths = sorted(p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUTPUT / "SHA256SUMS").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(OUTPUT).as_posix() + "\n" for p in paths))
    print(f"Prepared {len(paths)} image-free result files in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
