"""Verify the paper tables from released per-sample scores on a CPU.

No image pixels, model weights, VLM calls or training are used. The default
archive is results/acml2026; RCES_RESULTS_DIR may select another directory.
RCES_VERIFY_INTERVALS=0 skips confidence-interval recomputation for a quick check.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import metrics
import stratified
from exp_io import original_image_families

ARCHIVE = Path(os.environ.get("RCES_RESULTS_DIR", str(ROOT / "results/acml2026")))


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def annotation_fingerprint(path):
    fields = ("pair_id", "image_id", "rule_id", "decidability", "compliance_label",
              "evidence_sufficiency_label", "observed_checkpoints", "missing_checkpoints",
              "critical_missing_checkpoints", "base_image_id", "crop_bbox")
    records = [{k: r[k] for k in fields if k in r}
               for line in path.read_text().splitlines() if (r := json.loads(line))]
    records.sort(key=lambda r: r["pair_id"])
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def check_inputs(archive):
    manifest = load(archive / "manifest.json")
    for category in ("source_hashes", "input_hashes", "annotation_semantic_hashes"):
        for relative, expected in manifest[category].items():
            path = ROOT / relative
            actual = annotation_fingerprint(path) if category == "annotation_semantic_hashes" else hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Verification input has changed: {relative}")


def check_score_labels(archive):
    gold = {}
    for name in ("rces_pilot_pairs.jsonl", "rces_crop_augmentation.jsonl"):
        for line in (ROOT / "data/pairs" / name).read_text().splitlines():
            record = json.loads(line)
            gold[record["pair_id"]] = record
    split = load(archive / "configs/run.json")["split"]
    for path in (archive / "scores").glob("*/seed_*.json.gz"):
        run = load(path)
        frames = run.get("frames", run)
        for level in ("ckpt", "pair"):
            frame = frames[level]
            size = len(frame["score"])
            if any(len(values) != size for values in frame.values()):
                raise ValueError(f"Misaligned score arrays: {path.parent.name}/{path.name}")
            if frame["family_id"] != original_image_families(frame["image_id"]):
                raise ValueError("Archived family IDs disagree with crop ancestry")
            for i, pair_id in enumerate(frame["pair_id"]):
                record = gold[pair_id]
                if frame["image_id"][i] != record["image_id"] or record["image_id"] not in split["test"]:
                    raise ValueError("Archived score ID is not in the stated test population")
                if level == "ckpt":
                    rule, checkpoint = frame["key"][i].split("::", 1)
                    target = int(checkpoint not in record["missing_checkpoints"])
                else:
                    rule = frame["rule_id"][i]
                    target = int(record["evidence_sufficiency_label"] == "sufficient")
                if rule != record["rule_id"] or frame["target"][i] != target:
                    raise ValueError("Archived score target disagrees with the released human label")
        for field, subset in (("validation_rows", "val"), ("test_rows", "test")):
            if field not in frames:
                continue
            expected_ids = {p for p, r in gold.items() if r["image_id"] in split[subset]}
            if len(frames[field]) != len(expected_ids) or {r["pair_id"] for r in frames[field]} != expected_ids:
                raise ValueError("Incomplete validation/test operating-point population")
            for row in frames[field]:
                record = gold[row["pair_id"]]
                sufficient = int(record["evidence_sufficiency_label"] == "sufficient")
                if row["sufficiency_gold"] != sufficient or row["compliance_gold"] != record["compliance_label"]:
                    raise ValueError("Operating-point target disagrees with the released label")
    for path in (archive / "inputs/answerers").glob("*.json.gz"):
        for row in load(path)["rows"]:
            record = gold[row["pair_id"]]
            sufficient = int(record["evidence_sufficiency_label"] == "sufficient")
            label_correct = int(row["predicted_compliance"] == record["compliance_label"])
            if (row["image_id"] != record["image_id"] or row["rule_id"] != record["rule_id"] or
                row["image_id"] not in split[row["split"]] or
                row["sufficiency_gold"] != sufficient or row["label_correct"] != label_correct or
                row["correct"] != sufficient * label_correct):
                raise ValueError("Task-2 prediction target disagrees with the released human label")


def aggregate(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {"mean": float(array.mean()) if len(array) else None,
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "n": len(array)}


def json_ready(value):
    """Represent undefined numeric quantities as JSON null."""
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def auroc(frame):
    return float(metrics._auroc(np.asarray(frame["score"]), np.asarray(frame["target"])))


def within(frame, key):
    return float(stratified.stratified_auroc(frame["score"], frame["target"], frame[key])["pooled"])


def score_runs(folder):
    paths = sorted(folder.glob("seed_*.json.gz"))
    if len(paths) != 10:
        raise ValueError(f"Expected ten seeds in {folder.name}, found {len(paths)}")
    runs = [load(path) for path in paths]
    if [r["seed"] for r in runs] != list(range(10)):
        raise ValueError("Seed archive must contain seeds 0 through 9 exactly once")
    return runs


def interval(frames, field, n_bootstrap):
    if os.environ.get("RCES_VERIFY_INTERVALS", "1") == "0":
        return {"point_mean_of_seeds": float(np.mean([within(f, field) for f in frames]))}
    return stratified.mean_of_seed_aurocs(frames, field, n_bootstrap=n_bootstrap)


def iaa_summary(archive):
    spec = importlib.util.spec_from_file_location("_rces_iaa_verification", ROOT / "tmp/compute_iaa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def annotations(filename):
        return {row["pair_id"]: row for line in (ROOT / "data/pairs" / filename).read_text().splitlines()
                if (row := json.loads(line)).get("decidability") not in {None, "to_be_annotated"}}
    first = annotations("rces_pilot_pairs.jsonl")
    second = annotations("rces_double_annotation.jsonl")
    common = sorted(first.keys() & second.keys())
    rules = {r["rule_id"]: r for r in load(ROOT / "data/rule/rule.json")}
    fields = {}
    for field in module.CATEGORICAL_FIELDS:
        observations = [(first[p][field], second[p][field]) for p in common]
        kappa, agreement = module.cohen_kappa(observations)
        fields[field] = {"kappa": kappa, "observed_agreement": agreement, "n": len(observations)}
    observations = module.critical_checkpoint_observations(common, first, second, rules)
    kappa, agreement = module.cohen_kappa(observations)
    fields["critical_missing_checkpoint"] = {"kappa": kappa, "observed_agreement": agreement, "n": len(observations)}
    n_disagreements = sum(any(first[p][f] != second[p][f] for f in module.CATEGORICAL_FIELDS) or
                          set(first[p].get("critical_missing_checkpoints", [])) != set(second[p].get("critical_missing_checkpoints", [])) for p in common)
    expected = load(archive / "reports/iaa.json")
    compare(expected["fields"], fields)
    if len(common) != expected["n_compared"] or n_disagreements != len(expected["disagreements"]):
        raise ValueError("IAA support or disagreement count differs from the frozen report")
    return {"n_compared": len(common), "fields": fields, "n_disagreements": n_disagreements}


def population(rows, field="final_answerability_score"):
    return {"score": [r[field] for r in rows], "target": [r["sufficiency_gold"] for r in rows]}


def operating_point(validation, test, field="final_answerability_score"):
    val = population(validation, field)
    tau = metrics.tune_tau_for_risk(val["score"], val["target"], 0.1)
    accepted = [r for r in test if r[field] >= tau]
    return {"tau": tau, "coverage": len(accepted) / len(test),
            "risk": sum(1 - r["sufficiency_gold"] for r in accepted) / len(accepted) if accepted else None,
            "n_answered": len(accepted)}


def calibration_summary(runs):
    operations = {"gate_on": [], "visual": []}
    ece = {block: {method: [] for method in ("raw", "temperature", "isotonic")}
           for block in ("task1_gate_on", "task1_visual", "task2")}
    task2 = []
    for run in runs:
        frames, result = run["frames"], run["result"]
        val, test = frames["validation_rows"], frames["test_rows"]
        vv = [r for r in val if r["external_gate_passed"]]
        vt = [r for r in test if r["external_gate_passed"]]
        operations["gate_on"].append(operating_point(val, test))
        operations["visual"].append(operating_point(vv, vt, "visual_sufficiency_score"))
        for block in ece:
            for method in ece[block]:
                values = result["calibration"][block]["_scores"][method]
                ece[block][method].append(metrics.expected_calibration_error(values, frames["targets"][block], 10))
        t2v = [r for r in val if r["compliance_correct"] is not None]
        t2t = [r for r in test if r["compliance_correct"] is not None]
        tau = metrics.tune_tau_for_risk([r["final_answerability_score"] for r in t2v], [r["compliance_correct"] for r in t2v], 0.1)
        s = [r["final_answerability_score"] for r in t2t]
        y = [r["compliance_correct"] for r in t2t]
        task2.append({"coverage": float(np.mean(np.asarray(s) >= tau)),
                      "oracle_coverage": metrics.coverage_at_risk(s, y, 0.1), "aurc": metrics.aurc(s, y)})
    summary = {name: {"coverage": aggregate([r["coverage"] for r in rows]),
                      "risk": aggregate([r["risk"] if r["risk"] is not None else np.nan for r in rows]),
                      "seeds_at_target": sum(r["risk"] is not None and r["risk"] <= .1 for r in rows),
                      "answered_range": [min(r["n_answered"] for r in rows), max(r["n_answered"] for r in rows)]}
               for name, rows in operations.items()}
    summary["ece"] = {block: {m: aggregate(v) for m, v in methods.items()} for block, methods in ece.items()}
    summary["task2"] = {key: aggregate([r[key] for r in task2]) for key in task2[0]}
    summary["task2"]["seeds_answering_nothing"] = sum(r["coverage"] == 0 for r in task2)
    return summary


def aggregation_summary(runs):
    per_mode = {mode: defaultdict(list) for mode in ("min", "mean", "product")}
    for run in runs:
        frame = run["frames"]["ckpt"]
        probabilities, missing = defaultdict(list), defaultdict(int)
        for pair_id, score, target in zip(frame["pair_id"], frame["score"], frame["target"]):
            probabilities[pair_id].append(float(score))
            missing[pair_id] += int(target == 0)
        for mode in per_mode:
            all_scores, disc_scores, all_y, disc_y = [], [], [], []
            for row in run["frames"]["test_rows"]:
                values = probabilities[row["pair_id"]]
                if not row["external_gate_passed"]:
                    score = 0.0
                elif not values:
                    score = 1.0
                elif mode == "min":
                    score = min(values)
                elif mode == "mean":
                    score = sum(values) / len(values)
                else:
                    score = math.prod(values)
                all_scores.append(score)
                all_y.append(row["sufficiency_gold"])
                if len(values) >= 2 and missing[row["pair_id"]] <= 1:
                    disc_scores.append(score)
                    disc_y.append(row["sufficiency_gold"])
            per_mode[mode]["all"].append(float(metrics._auroc(np.asarray(all_scores), np.asarray(all_y))))
            per_mode[mode]["discriminative"].append(float(metrics._auroc(np.asarray(disc_scores), np.asarray(disc_y))))
    deltas = {}
    for mode in ("mean", "product"):
        for subset in ("all", "discriminative"):
            values = np.asarray(per_mode["min"][subset]) - np.asarray(per_mode[mode][subset])
            deltas[mode + "::" + subset] = dict(aggregate(values), seeds_min_better=int((values > 0).sum()))
    return {"per_mode": {m: {s: aggregate(v) for s, v in buckets.items()} for m, buckets in per_mode.items()},
            "paired_delta_min_minus": deltas}


def task2_summary(archive, runs):
    per_vlm, seed_matrix = {}, []
    rows_by_seed = [{r["pair_id"]: r for r in run["frames"]["test_rows"]} for run in runs]
    for path in sorted((archive / "inputs/answerers").glob("*.json.gz")):
        data = load(path)
        val = [r for r in data["rows"] if r["split"] == "val"]
        test = [r for r in data["rows"] if r["split"] == "test"]
        if len(val) != 267 or len(test) != 267:
            raise ValueError("Task-2 archive must include all validation and test pairs")
        methods = {}
        for name in ("maxprob", "temp_scale", "ptrue", "verbalized", "entropy", "margin"):
            tau = metrics.tune_tau_for_risk([r["scores"][name] for r in val], [r["correct"] for r in val], .1)
            methods[name] = metrics.summarize([r["scores"][name] for r in test], [r["correct"] for r in test], tau=tau, target_risk=.1)
        estimates = []
        for seed_rows in rows_by_seed:
            s = [seed_rows[r["pair_id"]]["final_answerability_score"] for r in test]
            estimates.append(float(metrics._auroc(np.asarray(s), np.asarray([r["correct"] for r in test]))))
        per_vlm[data["vlm"]] = {"accuracy": float(np.mean([r["label_correct"] for r in test])),
                                "answer_side": methods, "rces": aggregate(estimates), "per_seed_rces": estimates}
    if len(per_vlm) != 7:
        raise ValueError("Expected seven archived answerers")
    matrix = np.asarray([v["per_seed_rces"] for v in per_vlm.values()]).T
    return {"per_vlm": per_vlm, "cross_vlm_std_per_seed": aggregate(matrix.std(axis=1, ddof=1)),
            "cross_vlm_mean_per_seed": aggregate(matrix.mean(axis=1)), "below_chance_cells": int((matrix < .5).sum())}


def compute_tables(archive=ARCHIVE):
    metadata = load(archive / "manifest.json")
    n_bootstrap = metadata["n_bootstrap"]
    runs_by_encoder = {name: score_runs(archive / "scores" / ("v1_" + name))
                       for name in ("siglip2_512", "siglip2_224", "dinov2")}
    backbone = {}
    for encoder, runs in runs_by_encoder.items():
        ck, pairs = [r["frames"]["ckpt"] for r in runs], [r["frames"]["pair"] for r in runs]
        backbone[encoder] = {"checkpoint_pooled": aggregate([auroc(f) for f in ck]),
                             "pair_pooled": aggregate([auroc(f) for f in pairs]),
                             "checkpoint_within": interval(ck, "key", n_bootstrap),
                             "pair_within": interval(pairs, "rule_id", n_bootstrap),
                             "gate_on": aggregate([auroc(population(r["frames"]["test_rows"])) for r in runs])}
    channel = {"v1": backbone["siglip2_512"]}
    for name in ("v2", "shuffled", "text_only", "image_only"):
        runs = score_runs(archive / "scores" / name)
        ck, pairs = [r["ckpt"] for r in runs], [r["pair"] for r in runs]
        channel[name] = {"checkpoint_pooled": aggregate([auroc(f) for f in ck]),
                         "pair_pooled": aggregate([auroc(f) for f in pairs]),
                         "checkpoint_within": interval(ck, "key", n_bootstrap),
                         "pair_within": interval(pairs, "rule_id", n_bootstrap)}
        if name == "v2":
            # External-evidence scores are zero; all visual scores are positive.
            expected_rows = runs_by_encoder["siglip2_512"][0]["frames"]["test_rows"]
            gate_values = []
            for pair in pairs:
                values = dict(zip(pair["pair_id"], pair["score"]))
                scores = [values.get(r["pair_id"], 0.0) for r in expected_rows]
                gate_values.append(float(metrics._auroc(np.asarray(scores), np.asarray([r["sufficiency_gold"] for r in expected_rows]))))
            channel[name]["gate_on"] = aggregate(gate_values)
    main = runs_by_encoder["siglip2_512"]
    return {"table_1": iaa_summary(archive), "table_2": channel,
            "table_3": calibration_summary(main), "table_4": aggregation_summary(main),
            "table_5": backbone, "table_6": task2_summary(archive, main)}


def check_hashes(archive):
    checked = 0
    for line in (archive / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        path = (archive / relative).resolve()
        if not path.is_relative_to(archive.resolve()):
            raise ValueError("Invalid archive checksum path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        checked += 1
    return checked


def compare(expected, actual, path="tables"):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if os.environ.get("RCES_VERIFY_INTERVALS", "1") == "0" and key in {
                "estimand", "ci_95_low", "ci_95_high", "ci_contains_0_5", "n_seeds", "bootstrap_unit", "n_families", "n_image_ids", "requested_bootstrap_samples", "valid_bootstrap_samples"}:
                continue
            compare(value, actual[key], path + "." + key)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(f"Length mismatch: {path}")
        for index, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f"{path}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isclose(expected, actual, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"Value mismatch: {path}: expected {expected}, got {actual}")
    elif expected != actual:
        raise ValueError(f"Value mismatch: {path}")


def main():
    started = time.perf_counter()
    checked = check_hashes(ARCHIVE)
    check_inputs(ARCHIVE)
    check_score_labels(ARCHIVE)
    tables = json_ready(compute_tables(ARCHIVE))
    compare(load(ARCHIVE / "reports/paper_tables.json"), tables)
    output = ROOT / "experiments/artifacts/paper_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tables, indent=2) + "\n")
    elapsed = time.perf_counter() - started
    output.with_name("paper_verification_run.json").write_text(json.dumps({
        "elapsed_seconds": elapsed, "python": platform.python_version(), "numpy": np.__version__,
        "intervals_recomputed": os.environ.get("RCES_VERIFY_INTERVALS", "1") != "0",
        "archive_files_checked": checked, "gpu_required": False}, indent=2) + "\n")
    print(f"Verified {checked} archived files and all six paper tables without images, weights or VLM calls.")
    for name in ("v1", "v2", "shuffled", "text_only", "image_only"):
        entry = tables["table_2"][name]["pair_within"]
        print(f"{name}: mean within-rule AUROC {entry['point_mean_of_seeds']:.6f}")
    print("Risk–coverage points include complete tie groups; CIs resample original-image families.")
    print(f"CPU verification completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
