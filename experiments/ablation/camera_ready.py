"""Camera-ready measurements promised in the author response, over 10 seeds.

The response left five items as camera-ready work; this driver produces all of
them from one v1 training per seed, so every number shares the same heads:

  * 10-seed Task-1 figures (pooled, within-stratum, gate-on, visual subset);
  * the deployable operating point -- tau selected on validation, applied to
    test -- with a Wilson interval on the realised risk, per seed and pooled;
  * post-hoc calibration (temperature scaling on the logit, isotonic/PAV) fit on validation,
    reporting test ECE before and after, plus reliability-diagram bins;
  * stratification of the insufficient pairs by *cause* (partial vs undecidable,
    crop vs original, degraded vs good image quality);
  * multi-seed backbone/resolution robustness: run once per RCES_VISION_ENCODER.

The Task-2 answerer is whatever RCES_LOCAL_QWEN_MODEL selects; the paper's
error decomposition uses Qwen3.5-9B, so set it explicitly.

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_N_SEEDS=10 RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  uv run python experiments/ablation/camera_ready.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import numpy as np  # noqa: E402

import config  # noqa: E402
import exp_io  # noqa: E402
import metrics  # noqa: E402
import stratified  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces import evaluate as rces_eval  # noqa: E402
from rces import train as rces_train  # noqa: E402

N_SEEDS = int(os.environ.get("RCES_N_SEEDS", "10"))
N_BOOTSTRAP = int(os.environ.get("RCES_N_BOOTSTRAP", "3000"))
TAG = config._CROPAUG_TAG
CROP_PAIR_PREFIX = "rces_crop"
LOGIT_EPS = 1e-7
WILSON_Z = 1.96
EXTERNAL = "requires_external_evidence"


# --------------------------------------------------------------------------- stats

def wilson(errors: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = errors / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p) - np.log1p(-p)


def _nll(logits, targets, temperature: float) -> float:
    z = logits / temperature
    # log(1 + e^z) computed stably; targets are 0/1.
    return float(np.mean(np.logaddexp(0, z) - targets * z))


def fit_temperature(scores, targets, lo: float = 0.02, hi: float = 200.0) -> float:
    """Temperature scaling on the logit: one scalar T, fit by NLL on validation.

    A 1-D bounded search (log grid, then golden-section) instead of Newton on a
    two-parameter logistic: the scores saturate near 1 and the gate emits exact
    zeros that separate perfectly, which makes an unregularised Platt fit diverge.
    """
    x = _logit(np.asarray(scores, dtype=float))
    y = np.asarray(targets, dtype=float)
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 200))
    best = int(np.argmin([_nll(x, y, t) for t in grid]))
    a, b = np.log(grid[max(best - 1, 0)]), np.log(grid[min(best + 1, len(grid) - 1)])
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        if _nll(x, y, math.exp(c)) < _nll(x, y, math.exp(d)):
            b = d
        else:
            a = c
    return float(math.exp((a + b) / 2))


def apply_temperature(temperature: float, scores) -> np.ndarray:
    return 1 / (1 + np.exp(-_logit(np.asarray(scores, dtype=float)) / temperature))


def fit_isotonic(scores, targets) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators; returns block upper edges and block values."""
    order = np.argsort(scores, kind="mergesort")
    x = np.asarray(scores, dtype=float)[order]
    y = np.asarray(targets, dtype=float)[order]
    blocks: list[list[float]] = []  # [value, weight, max_x]
    for xi, yi in zip(x, y):
        blocks.append([yi, 1.0, xi])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, x2 = blocks.pop()
            v1, w1, _ = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, x2])
    edges = np.asarray([b[2] for b in blocks])
    values = np.asarray([b[0] for b in blocks])
    return edges, values


def apply_isotonic(model, scores) -> np.ndarray:
    edges, values = model
    index = np.searchsorted(edges, np.asarray(scores, dtype=float), side="left")
    return values[np.clip(index, 0, len(values) - 1)]


def reliability_bins(scores, targets, n_bins: int) -> dict[str, list[float]]:
    s = np.clip(np.asarray(scores, dtype=float), 0, 1)
    y = np.asarray(targets, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    conf, acc, count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s >= lo) & (s <= hi) if lo == 0 else (s > lo) & (s <= hi)
        count.append(int(mask.sum()))
        conf.append(float(s[mask].mean()) if mask.any() else float("nan"))
        acc.append(float(y[mask].mean()) if mask.any() else float("nan"))
    return {"confidence": conf, "accuracy": acc, "count": count}


def calibration_block(val_s, val_y, test_s, test_y) -> dict[str, object]:
    temperature = fit_temperature(val_s, val_y)
    iso = fit_isotonic(val_s, val_y)
    calibrated = {
        "raw": np.asarray(test_s, dtype=float),
        "temperature": apply_temperature(temperature, test_s),
        "isotonic": apply_isotonic(iso, test_s),
    }
    bins = config.ECE_BINS
    return {
        "ece": {k: metrics.expected_calibration_error(v, test_y, bins) for k, v in calibrated.items()},
        "brier": {k: float(np.mean((v - np.asarray(test_y)) ** 2)) for k, v in calibrated.items()},
        "temperature": temperature,
        "_scores": {k: v.tolist() for k, v in calibrated.items()},
    }


def agg(values) -> dict[str, float]:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(array):
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(len(array)),
    }


# --------------------------------------------------------------------- per seed

def operating_point(test_rows, tau: float, score_field: str) -> dict[str, object]:
    answered = [r for r in test_rows if getattr(r, score_field) >= tau]
    errors = sum(1 - r.sufficiency_gold for r in answered)
    base_errors = sum(1 - r.sufficiency_gold for r in test_rows)
    n = len(answered)
    low, high = wilson(errors, n)
    return {
        "tau": float(tau),
        "n_total": len(test_rows),
        "n_answered": n,
        "errors": int(errors),
        "coverage": n / len(test_rows),
        "risk": errors / n if n else float("nan"),
        "risk_wilson95": [low, high],
        "base_risk": base_errors / len(test_rows),
    }


def cause_strata(test_rows, gold_by_pair) -> dict[str, object]:
    """Detection of each insufficiency cause against all sufficient pairs.

    Restricted to the visual+normative population, where the visual head is the
    instrument; each cause's negatives are ranked against the same positives.
    """
    rows = [r for r in test_rows if r.decision_scope != EXTERNAL]
    positives = [r.visual_sufficiency_score for r in rows if r.sufficiency_gold == 1]
    causes = {
        "partially_decidable": lambda r, g: g["decidability"] == "partially_decidable",
        "undecidable": lambda r, g: g["decidability"] == "undecidable",
        "crop_truncation": lambda r, g: r.pair_id.startswith(CROP_PAIR_PREFIX),
        "natural_original": lambda r, g: not r.pair_id.startswith(CROP_PAIR_PREFIX),
        "degraded_image_quality": lambda r, g: g.get("image_quality") in {"usable", "poor"},
        "good_image_quality": lambda r, g: g.get("image_quality") == "good",
    }
    out: dict[str, object] = {}
    for name, test in causes.items():
        negatives = [r.visual_sufficiency_score for r in rows
                     if r.sufficiency_gold == 0 and test(r, gold_by_pair[r.pair_id])]
        if not negatives or not positives:
            out[name] = {"n_negatives": len(negatives), "auroc": float("nan")}
            continue
        scores = np.asarray(positives + negatives, dtype=float)
        targets = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=int)
        out[name] = {"n_negatives": len(negatives), "auroc": float(metrics._auroc(scores, targets))}
    return out


def checkpoint_frame(probs, records) -> dict[str, list]:
    frame = {k: [] for k in ("score", "target", "image_id", "key")}
    for record in records:
        for key, target in record.resolvable_targets.items():
            if (record.pair_id, key) in probs:
                frame["score"].append(probs[(record.pair_id, key)])
                frame["target"].append(int(target))
                frame["image_id"].append(record.image_id)
                frame["key"].append(f"{record.rule_id}::{key}")
    return frame


def pair_frame(rows) -> dict[str, list]:
    frame = {k: [] for k in ("score", "target", "image_id", "rule_id")}
    for row in rows:
        if row.decision_scope == EXTERNAL:
            continue
        frame["score"].append(row.visual_sufficiency_score)
        frame["target"].append(int(row.sufficiency_gold))
        frame["image_id"].append(row.image_id)
        frame["rule_id"].append(row.rule_id)
    return frame


def run_seed(device, seed: int, gold_by_pair) -> tuple[dict, dict]:
    config.SEED = seed
    config.INPUT_MODE = "full"
    config.HEAD_CKPT = config.ARTIFACT_DIR / f"rces_head_camready_{config.VISION_ENCODER}{TAG}.pt"
    split = splits.load_split()
    rces_train.main()

    head = rces_eval.load_head(device)
    test_records = ds.build_pair_records(set(split["test"]))
    val_rows = rces_eval.build_eval_rows(head, set(split["val"]), device)
    test_rows = rces_eval.build_eval_rows(head, set(split["test"]), device)
    probs = rces_eval.checkpoint_probs(head, test_records, device)

    ckpt = checkpoint_frame(probs, test_records)
    pair = pair_frame(test_rows)
    vis_val = [r for r in val_rows if r.external_gate_passed]
    vis_test = [r for r in test_rows if r.external_gate_passed]

    tau = metrics.tune_tau_for_risk([r.final_answerability_score for r in val_rows],
                                    [r.sufficiency_gold for r in val_rows], config.TARGET_RISK)
    vis_tau = metrics.tune_tau_for_risk([r.visual_sufficiency_score for r in vis_val],
                                        [r.sufficiency_gold for r in vis_val], config.TARGET_RISK)
    gate_on = rces_eval.task1_metrics(test_rows, tau)
    visual = rces_eval.task1_metrics(vis_test, vis_tau, score_field="visual_sufficiency_score")

    t2_val = [r for r in val_rows if r.compliance_correct is not None]
    t2_test = [r for r in test_rows if r.compliance_correct is not None]
    t2_tau = metrics.tune_tau_for_risk([r.final_answerability_score for r in t2_val],
                                       [int(r.compliance_correct) for r in t2_val], config.TARGET_RISK)
    task2 = rces_eval.task2_metrics(test_rows, t2_tau)

    result = {
        "checkpoint_pooled_auroc": float(metrics._auroc(np.asarray(ckpt["score"]), np.asarray(ckpt["target"]))),
        "checkpoint_within_key": float(stratified.stratified_auroc(ckpt["score"], ckpt["target"], ckpt["key"])["pooled"]),
        "pair_visual_pooled_auroc": float(visual["auroc"]),
        "pair_within_rule": float(stratified.stratified_auroc(pair["score"], pair["target"], pair["rule_id"])["pooled"]),
        "task1_gate_on_auroc": float(gate_on["auroc"]),
        "task1_gate_on_aurc": float(gate_on["aurc"]),
        "task1_gate_on_cmcr": float(gate_on["cmcr"]),
        "visual_aurc": float(visual["aurc"]),
        "visual_cmcr": float(visual["cmcr"]),
        "oracle_coverage_at_risk_gate_on": float(gate_on["coverage_at_target_risk"]),
        "oracle_coverage_at_risk_visual": float(visual["coverage_at_target_risk"]),
        "op_gate_on_267": operating_point(test_rows, tau, "final_answerability_score"),
        # Same tau and answered set, reported against the visual population: the
        # gate zeroes every external pair, so none of them is ever answered.
        "op_gate_on_on_217": operating_point(vis_test, tau, "visual_sufficiency_score"),
        "op_visual_tau_217": operating_point(vis_test, vis_tau, "visual_sufficiency_score"),
        "task2": {
            "tau": float(t2_tau),
            "auroc": float(task2.get("auroc", float("nan"))),
            "answer_rate": float(task2.get("answer_rate", float("nan"))),
            "oracle_coverage_at_risk": float(task2.get("coverage_at_target_risk", float("nan"))),
            "raw_label_accuracy_all_pairs": float(task2.get("raw_label_accuracy_all_pairs", float("nan"))),
        },
        "calibration": {
            "task1_gate_on": calibration_block(
                [r.final_answerability_score for r in val_rows], [r.sufficiency_gold for r in val_rows],
                [r.final_answerability_score for r in test_rows], [r.sufficiency_gold for r in test_rows]),
            "task1_visual": calibration_block(
                [r.visual_sufficiency_score for r in vis_val], [r.sufficiency_gold for r in vis_val],
                [r.visual_sufficiency_score for r in vis_test], [r.sufficiency_gold for r in vis_test]),
            "task2": calibration_block(
                [r.final_answerability_score for r in t2_val], [int(r.compliance_correct) for r in t2_val],
                [r.final_answerability_score for r in t2_test], [int(r.compliance_correct) for r in t2_test]),
        },
        "causes": cause_strata(test_rows, gold_by_pair),
    }
    targets = {
        "task1_gate_on": [r.sufficiency_gold for r in test_rows],
        "task1_visual": [r.sufficiency_gold for r in vis_test],
        "task2": [int(r.compliance_correct) for r in t2_test],
    }
    return result, {"ckpt": ckpt, "pair": pair, "targets": targets}


# ------------------------------------------------------------------------- main

def summarise(per_seed: list[dict], frames: list[dict]) -> dict[str, object]:
    scalar_keys = [k for k, v in per_seed[0].items() if isinstance(v, float)]
    summary: dict[str, object] = {k: agg([s[k] for s in per_seed]) for k in scalar_keys}

    summary["checkpoint_within_key_ci"] = stratified.mean_of_seed_aurocs(
        [f["ckpt"] for f in frames], "key", n_bootstrap=N_BOOTSTRAP)
    summary["pair_within_rule_ci"] = stratified.mean_of_seed_aurocs(
        [f["pair"] for f in frames], "rule_id", n_bootstrap=N_BOOTSTRAP)

    for op in ("op_gate_on_267", "op_gate_on_on_217", "op_visual_tau_217"):
        entries = [s[op] for s in per_seed]
        answered = sum(e["n_answered"] for e in entries)
        errors = sum(e["errors"] for e in entries)
        summary[op] = {
            "coverage": agg([e["coverage"] for e in entries]),
            "risk": agg([e["risk"] for e in entries]),
            "base_risk": entries[0]["base_risk"],
            "seeds_with_risk_le_target": int(sum(
                1 for e in entries if e["n_answered"] and e["risk"] <= config.TARGET_RISK)),
            "seeds_answering_nothing": int(sum(1 for e in entries if not e["n_answered"])),
            "pooled_over_seeds": {
                "answered": answered, "errors": errors,
                "risk": errors / answered if answered else float("nan"),
                "risk_wilson95": list(wilson(errors, answered)),
            },
            "per_seed": entries,
        }

    summary["task2"] = {k: agg([s["task2"][k] for s in per_seed]) for k in per_seed[0]["task2"]}

    calibration: dict[str, object] = {}
    for block in per_seed[0]["calibration"]:
        calibration[block] = {
            "ece": {m: agg([s["calibration"][block]["ece"][m] for s in per_seed])
                    for m in ("raw", "temperature", "isotonic")},
            "brier": {m: agg([s["calibration"][block]["brier"][m] for s in per_seed])
                      for m in ("raw", "temperature", "isotonic")},
            # Reliability bins pool every seed's test predictions; a diagram of
            # counts, not an estimator, so pooling across heads is harmless here.
            "reliability_pooled": {
                m: reliability_bins(
                    [x for s in per_seed for x in s["calibration"][block]["_scores"][m]],
                    [y for f in frames for y in f["targets"][block]],
                    config.ECE_BINS)
                for m in ("raw", "temperature", "isotonic")
            },
        }
    summary["calibration"] = calibration

    summary["causes"] = {
        name: {"n_negatives": per_seed[0]["causes"][name]["n_negatives"],
               "auroc": agg([s["causes"][name]["auroc"] for s in per_seed])}
        for name in per_seed[0]["causes"]
    }
    return summary


def main() -> None:
    device = rces_train._device()
    gold_by_pair = {g["pair_id"]: g for g in exp_io.load_locked_gold()}
    per_seed, frames = [], []
    for seed in range(N_SEEDS):
        print(f"\n########## {config.VISION_ENCODER} seed {seed} ##########")
        result, frame = run_seed(device, seed, gold_by_pair)
        per_seed.append(result)
        frames.append(frame)
        print(f"  ckpt pooled={result['checkpoint_pooled_auroc']:.4f} "
              f"within={result['checkpoint_within_key']:.4f} | pair217={result['pair_visual_pooled_auroc']:.4f} "
              f"within={result['pair_within_rule']:.4f} | gate-on={result['task1_gate_on_auroc']:.4f} | "
              f"op cov={result['op_gate_on_267']['coverage']:.3f} risk={result['op_gate_on_267']['risk']:.3f}")

    summary = summarise(per_seed, frames)
    for seed_result in per_seed:
        for block in seed_result["calibration"].values():
            block.pop("_scores")
    out = {
        "encoder": config.VISION_ENCODER,
        "compliance_vlm": config.LOCAL_QWEN_MODEL_NAME,
        "include_crop_augmentation": config.INCLUDE_CROP_AUGMENTATION,
        "target_risk": config.TARGET_RISK,
        "seeds": list(range(N_SEEDS)),
        "std_convention": "sample (ddof=1)",
        "summary": summary,
        "per_seed": per_seed,
    }
    path = config.ARTIFACT_DIR / f"camera_ready_{config.VISION_ENCODER}{TAG}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nwrote {path.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
