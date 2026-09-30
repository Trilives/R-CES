"""Does R-CES sufficiency add signal to VLM confidence? (Task-2 cascade study)

R-CES loses to MaxProb head-to-head on Task-2 selective compliance, but it
answers a different question (is the evidence sufficient?) than VLM confidence
(is the model sure?). This script tests whether the two are COMPLEMENTARY: it
compares, on the same Qwen3.5-9B answers and the same image-disjoint val/test
split, six gates for "answer only when the compliance label is likely correct":

  maxprob            gate by the VLM's restricted-label confidence
  rces               gate by R-CES final answerability (sufficiency) score
  min_combo          min(maxprob, rces)        -- answer only if both are high
  product_combo      maxprob * rces
  logistic_maxprob   logistic P(correct | maxprob)            (Kamath-style)
  logistic_combo     logistic P(correct | maxprob, rces)      (+ R-CES feature)

The decisive comparison is logistic_combo vs logistic_maxprob: if adding the
R-CES feature lowers Task-2 AURC / raises AUROC, R-CES carries information beyond
VLM confidence -- a positive result that reframes the head-to-head loss.

Needs a trained R-CES head at config.HEAD_CKPT (set RCES_E0_ARTIFACT_TAG) and the
Qwen predictions for the active model.

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 RCES_DETERMINISTIC=1 \
  RCES_E0_ARTIFACT_TAG=siglip2_512_cropaug RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 UV_CACHE_DIR=/tmp/uv-cache \
  uv run python experiments/baselines/cascade.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import numpy as np  # noqa: E402

import config  # noqa: E402
import metrics  # noqa: E402
from baselines import common  # noqa: E402
from data import splits  # noqa: E402
from rces import evaluate as rces_eval  # noqa: E402
from rces import train as rces_train  # noqa: E402


CLASS_ORDER = ("compliant", "non_compliant", "not_applicable")


def _entropy_confidence(probs: np.ndarray) -> float:
    """Negative Shannon entropy of the label distribution (higher = more certain)."""
    p = np.clip(probs, 1e-12, 1.0)
    return float(np.sum(p * np.log(p)))  # = -entropy


def _margin_confidence(probs: np.ndarray) -> float:
    """Top-1 minus top-2 probability."""
    s = np.sort(probs)[::-1]
    return float(s[0] - (s[1] if len(s) > 1 else 0.0))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Guo et al. (2017) temperature scaling: T>0 minimising val cross-entropy."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate([np.linspace(0.25, 1.0, 16), np.linspace(1.05, 8.0, 60)]):
        scaled = logits / t
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        logp = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        nll = -logp[np.arange(len(labels)), labels].mean()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def _temperature_confidence(probs: np.ndarray, temperature: float) -> float:
    """Max softmax probability after temperature scaling (logits := log probs)."""
    z = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    z = z - z.max()
    p = np.exp(z)
    return float((p / p.sum()).max())


def aligned_rows(image_ids, head, device, predictions_by_id):
    """Per pair with a usable compliance label.

    Returns parallel arrays/lists, all in the same pair order:
      base   -- (maxprob, rces_score, correct) used by the logistic baselines,
      probs  -- the model's 3-class label distribution (CLASS_ORDER),
      golds  -- gold compliance class index (0..2) or -1 when not one of the
                three classes (used only to fit temperature scaling).
    """
    rows = rces_eval.build_eval_rows(head, image_ids, device)
    base, probs, golds, pair_ids = [], [], [], []
    for r in rows:
        if r.compliance_correct is None:
            continue
        record = predictions_by_id.get(r.pair_id)
        if record is None:
            continue
        mp = float(record["confidence"])
        dist = record.get("label_probabilities") or {}
        p_vec = np.array([float(dist.get(c, 0.0)) for c in CLASS_ORDER], dtype=float)
        total = p_vec.sum()
        if total > 0:
            p_vec = p_vec / total
        base.append((mp, float(r.final_answerability_score), int(r.compliance_correct)))
        probs.append(p_vec)
        gold = getattr(r, "compliance_gold", None)
        golds.append(CLASS_ORDER.index(gold) if gold in CLASS_ORDER else -1)
        pair_ids.append(r.pair_id)
    return np.asarray(base), np.asarray(probs), np.asarray(golds, dtype=int), pair_ids


def fit_logistic(x: np.ndarray, y: np.ndarray, *, steps: int = 4000, lr: float = 0.2):
    """Standardised logistic regression; returns (weights, mu, sigma)."""
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0] = 1.0
    xs = np.hstack([(x - mu) / sigma, np.ones((len(x), 1))])
    w = np.zeros(xs.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(xs @ w, -30, 30)))
        w -= lr * (xs.T @ (p - y) / len(y))
    return w, mu, sigma


def logistic_scores(x: np.ndarray, fit) -> np.ndarray:
    w, mu, sigma = fit
    xs = np.hstack([(x - mu) / sigma, np.ones((len(x), 1))])
    return 1.0 / (1.0 + np.exp(-np.clip(xs @ w, -30, 30)))


def task2(val_scores, val_correct, test_scores, test_correct) -> dict:
    tau = metrics.tune_tau_for_risk(
        list(val_scores), list(val_correct), config.TARGET_RISK
    )
    return metrics.summarize(
        list(test_scores), list(test_correct), tau=tau,
        target_risk=config.TARGET_RISK, n_bins=config.ECE_BINS,
    )


def main() -> None:
    device = rces_train._device()
    head = rces_eval.load_head(device)
    predictions_by_id = common.load_local_predictions()
    split = splits.load_split()
    val, v_probs, v_gold, v_ids = aligned_rows(set(split["val"]), head, device, predictions_by_id)
    test, t_probs, t_gold, t_ids = aligned_rows(set(split["test"]), head, device, predictions_by_id)
    if not len(val) or not len(test):
        raise RuntimeError("No aligned val/test pairs with maxprob + compliance label")

    v_mp, v_rc, v_y = val[:, 0], val[:, 1], val[:, 2]
    t_mp, t_rc, t_y = test[:, 0], test[:, 1], test[:, 2]

    fit_mp = fit_logistic(val[:, :1], v_y)
    fit_combo = fit_logistic(val[:, :2], v_y)

    # --- Distributional confidence baselines (from the saved 3-class distribution).
    v_ent = np.array([_entropy_confidence(p) for p in v_probs])
    t_ent = np.array([_entropy_confidence(p) for p in t_probs])
    v_marg = np.array([_margin_confidence(p) for p in v_probs])
    t_marg = np.array([_margin_confidence(p) for p in t_probs])
    fit_mask = v_gold >= 0
    temperature = (
        fit_temperature(np.log(np.clip(v_probs[fit_mask], 1e-12, 1.0)), v_gold[fit_mask])
        if fit_mask.any() else 1.0
    )
    v_temp = np.array([_temperature_confidence(p, temperature) for p in v_probs])
    t_temp = np.array([_temperature_confidence(p, temperature) for p in t_probs])

    methods = {
        "maxprob": (v_mp, t_mp),
        "entropy": (v_ent, t_ent),
        "margin": (v_marg, t_marg),
        "temp_scale": (v_temp, t_temp),
        "rces": (v_rc, t_rc),
        "min_combo": (np.minimum(v_mp, v_rc), np.minimum(t_mp, t_rc)),
        "product_combo": (v_mp * v_rc, t_mp * t_rc),
        "logistic_maxprob": (logistic_scores(val[:, :1], fit_mp), logistic_scores(test[:, :1], fit_mp)),
        "logistic_combo": (logistic_scores(val[:, :2], fit_combo), logistic_scores(test[:, :2], fit_combo)),
    }

    # --- Optional self-evaluation baselines (P(True), verbalized) if precomputed.
    self_eval = common.load_self_eval() if hasattr(common, "load_self_eval") else {}
    if self_eval:
        def _se(ids, key):
            return np.array([float(self_eval.get(pid, {}).get(key, 0.0)) for pid in ids])
        for key in ("ptrue", "verbalized"):
            if any(key in v for v in self_eval.values()):
                methods[key] = (_se(v_ids, key), _se(t_ids, key))
    report = {
        "n_val": int(len(val)), "n_test": int(len(test)),
        "test_label_accuracy": float(t_y.mean()),
        "methods": {},
    }
    for name, (vs, ts) in methods.items():
        report["methods"][name] = task2(vs, v_y, ts, t_y)

    out_path = config.ARTIFACT_DIR / f"cascade_task2_report{config._CROPAUG_TAG}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== Task-2 cascade (n_test={len(test)}, label_acc={t_y.mean():.4f}) ===")
    print(f"{'method':<18}{'AUROC':>9}{'AURC':>9}{'cov@risk':>10}{'ans_rate':>10}{'ECE':>8}")
    for name in methods:
        m = report["methods"][name]
        print(f"{name:<18}{m['auroc']:>9.4f}{m['aurc']:>9.4f}"
              f"{m.get('coverage_at_target_risk', float('nan')):>10.4f}"
              f"{m.get('answer_rate', float('nan')):>10.4f}{m.get('ece', float('nan')):>8.4f}")
    mp, cb = report["methods"]["logistic_maxprob"], report["methods"]["logistic_combo"]
    print(f"\n+R-CES feature effect (logistic_combo - logistic_maxprob): "
          f"AUROC {cb['auroc']-mp['auroc']:+.4f}, AURC {cb['aurc']-mp['aurc']:+.4f}")
    print(f"wrote {out_path.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
