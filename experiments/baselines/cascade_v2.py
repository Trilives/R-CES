"""Task-2 cascade with the R-CES **v2** (ContextualRCESHead) sufficiency score.

Identical gate math to baselines/cascade.py (same task2/fit_logistic/temperature/
self-eval code, reused by import) -- only the R-CES head is swapped from v1 to v2.
Writes cascade_task2_report{tag}.json; copy to cascade_task2_<MODEL>_v2.json per VLM.

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 \
  RCES_LOCAL_QWEN_MODEL=InternVL3_5-8B-HF UV_CACHE_DIR=/tmp/uv-cache \
  uv run python experiments/baselines/cascade_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_EXP / "rces-2"))

import numpy as np  # noqa: E402

import config  # noqa: E402
from baselines import common  # noqa: E402
from baselines import cascade as casc  # noqa: E402  (reuse gate helpers)
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
import evaluate as v2_eval  # noqa: E402  (rces-2/evaluate.py, self-bootstraps path)

CLASS_ORDER = casc.CLASS_ORDER


def aligned_rows_v2(image_ids, head, device, predictions_by_id):
    """Mirror cascade.aligned_rows but score with the v2 head."""
    records = ds.build_pair_records(set(image_ids))
    rows = v2_eval.build_eval_rows(head, records, device)
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


def main() -> None:
    device = v2_eval.device()
    head = v2_eval.load_head(device)
    predictions_by_id = common.load_local_predictions()
    split = splits.load_split()
    val, v_probs, v_gold, v_ids = aligned_rows_v2(set(split["val"]), head, device, predictions_by_id)
    test, t_probs, t_gold, t_ids = aligned_rows_v2(set(split["test"]), head, device, predictions_by_id)
    if not len(val) or not len(test):
        raise RuntimeError("No aligned val/test pairs with maxprob + compliance label")

    v_mp, v_rc, v_y = val[:, 0], val[:, 1], val[:, 2]
    t_mp, t_rc, t_y = test[:, 0], test[:, 1], test[:, 2]

    fit_mp = casc.fit_logistic(val[:, :1], v_y)
    fit_combo = casc.fit_logistic(val[:, :2], v_y)

    v_ent = np.array([casc._entropy_confidence(p) for p in v_probs])
    t_ent = np.array([casc._entropy_confidence(p) for p in t_probs])
    v_marg = np.array([casc._margin_confidence(p) for p in v_probs])
    t_marg = np.array([casc._margin_confidence(p) for p in t_probs])
    fit_mask = v_gold >= 0
    temperature = (
        casc.fit_temperature(np.log(np.clip(v_probs[fit_mask], 1e-12, 1.0)), v_gold[fit_mask])
        if fit_mask.any() else 1.0
    )
    v_temp = np.array([casc._temperature_confidence(p, temperature) for p in v_probs])
    t_temp = np.array([casc._temperature_confidence(p, temperature) for p in t_probs])

    methods = {
        "maxprob": (v_mp, t_mp),
        "entropy": (v_ent, t_ent),
        "margin": (v_marg, t_marg),
        "temp_scale": (v_temp, t_temp),
        "rces": (v_rc, t_rc),
        "min_combo": (np.minimum(v_mp, v_rc), np.minimum(t_mp, t_rc)),
        "product_combo": (v_mp * v_rc, t_mp * t_rc),
        "logistic_maxprob": (casc.logistic_scores(val[:, :1], fit_mp), casc.logistic_scores(test[:, :1], fit_mp)),
        "logistic_combo": (casc.logistic_scores(val[:, :2], fit_combo), casc.logistic_scores(test[:, :2], fit_combo)),
    }

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
        "head_version": "v2",
        "methods": {},
    }
    for name, (vs, ts) in methods.items():
        report["methods"][name] = casc.task2(vs, v_y, ts, t_y)

    out_path = config.ARTIFACT_DIR / f"cascade_task2_report{config._CROPAUG_TAG}_v2.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== Task-2 cascade V2 (n_test={len(test)}, label_acc={t_y.mean():.4f}) ===")
    print(f"{'method':<18}{'AUROC':>9}{'AURC':>9}")
    for name in methods:
        m = report["methods"][name]
        print(f"{name:<18}{m['auroc']:>9.4f}{m['aurc']:>9.4f}")
    print(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
