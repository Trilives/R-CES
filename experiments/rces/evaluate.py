"""Evaluate the trained R-CES head against human sufficiency gold.

R-CES's score is evaluated as a selective predictor of whether the image-rule
pair is sufficiently evidenced / decidable. The headline custom metric is
Critical Missing Checkpoint Recall. No cloud-model annotation is read here.

tau is tuned on val by risk-coverage (target risk = config.TARGET_RISK), then
applied to test.

  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/rces/evaluate.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
import metrics  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from local_vlm.predictor import prediction_record_is_compatible  # noqa: E402
from rces import aggregate  # noqa: E402
from rces.model import RCESHead  # noqa: E402


def _device() -> torch.device:
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_head(device: torch.device) -> RCESHead:
    if not config.HEAD_CKPT.exists():
        raise FileNotFoundError(f"Train first: {config.HEAD_CKPT} missing")
    ckpt = torch.load(config.HEAD_CKPT, map_location=device)
    head = RCESHead(
        img_dim=ckpt["img_dim"],
        txt_dim=ckpt["txt_dim"],
        use_is_visual=ckpt["use_is_visual"],
        input_mode=ckpt.get("input_mode", "full"),
    )
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval()


@torch.no_grad()
def checkpoint_probs(head: RCESHead, records: list[ds.PairRecord], device: torch.device) -> dict[tuple[str, str], float]:
    """P(resolvable) for every (pair, checkpoint) example in `records`."""
    examples = ds.build_examples(records)
    loader = DataLoader(ds.RCESDataset(examples, records), batch_size=config.BATCH_SIZE, shuffle=False)
    out: dict[tuple[str, str], float] = {}
    for batch in loader:
        probs = head.resolvable_prob(
            batch["patches"].to(device), batch["text"].to(device), batch["is_visual"].to(device)
        )
        for pair_id, key, prob in zip(batch["pair_id"], batch["checkpoint_key"], probs.tolist()):
            out[(pair_id, key)] = float(prob)
    return out


def visual_pair_scores(
    head: RCESHead, records: list[ds.PairRecord], device: torch.device
) -> dict[str, float]:
    probs = checkpoint_probs(head, records, device)
    scores: dict[str, float] = {}
    for rec in records:
        if config.ABLATE_NO_CHECKPOINT_SUPERVISION:
            scores[rec.pair_id] = aggregate.aggregate_pair_score_from_pairlevel(
                rec, probs.get((rec.pair_id, "__pair__"), 1.0)
            )
        else:
            resolvable = {k: probs.get((rec.pair_id, k), 1.0) for k in rec.critical_keys}
            scores[rec.pair_id] = aggregate.aggregate_pair_score(rec, resolvable)
    return scores


@dataclass
class EvalRow:
    pair_id: str
    image_id: str
    rule_id: str
    decision_scope: str
    visual_sufficiency_score: float
    external_gate_passed: bool
    final_answerability_score: float
    sufficiency_gold: int
    should_abstain: int
    decidable_gold: int
    compliance_gold: str
    predicted_compliance: str | None
    raw_compliance_correct: int | None
    compliance_correct: int | None


def load_local_compliance() -> dict[str, str]:
    if not config.LOCAL_QWEN_PREDICTIONS_PATH.exists():
        return {}
    rows = ds.exp_io.rio.load_jsonl(config.LOCAL_QWEN_PREDICTIONS_PATH)
    valid = {}
    for row in rows:
        predicted = row.get("predicted_compliance")
        if not prediction_record_is_compatible(row):
            continue
        valid[row["pair_id"]] = str(predicted)
    return valid


def build_eval_rows(head: RCESHead, image_ids: set[str], device: torch.device) -> list[EvalRow]:
    records = ds.build_pair_records(image_ids)
    visual_scores = visual_pair_scores(head, records, device)
    local_predictions = load_local_compliance()
    rows: list[EvalRow] = []
    for rec in records:
        predicted = local_predictions.get(rec.pair_id)
        rows.append(
            EvalRow(
                pair_id=rec.pair_id,
                image_id=rec.image_id,
                rule_id=rec.rule_id,
                decision_scope=rec.decision_scope,
                visual_sufficiency_score=visual_scores[rec.pair_id],
                external_gate_passed=not rec.external_evidence_required,
                final_answerability_score=aggregate.apply_external_gate(
                    rec, visual_scores[rec.pair_id]
                ),
                sufficiency_gold=rec.sufficiency_gold,
                should_abstain=1 - rec.sufficiency_gold,
                decidable_gold=rec.decidable_gold,
                compliance_gold=rec.compliance_gold,
                predicted_compliance=predicted,
                raw_compliance_correct=(
                    None if predicted is None else int(predicted == rec.compliance_gold)
                ),
                compliance_correct=(
                    None
                    if predicted is None
                    else int(rec.sufficiency_gold == 1 and predicted == rec.compliance_gold)
                ),
            )
        )
    return rows


def task1_metrics(
    rows: list[EvalRow], tau: float, *, score_field: str = "final_answerability_score"
) -> dict[str, float]:
    scores = [float(getattr(r, score_field)) for r in rows]
    correct = [r.sufficiency_gold for r in rows]
    should_abstain = [r.should_abstain for r in rows]
    return metrics.summarize(
        scores, correct, tau=tau, target_risk=config.TARGET_RISK,
        should_abstain=should_abstain, n_bins=config.ECE_BINS,
    )


def checkpoint_metrics(
    probabilities: dict[tuple[str, str], float], records: list[ds.PairRecord]
) -> dict[str, float]:
    scores: list[float] = []
    targets: list[int] = []
    for record in records:
        for key, target in record.resolvable_targets.items():
            if (record.pair_id, key) in probabilities:
                scores.append(probabilities[(record.pair_id, key)])
                targets.append(target)
    return metrics.binary_classification_metrics(scores, targets, threshold=0.5)


def task2_metrics(rows: list[EvalRow], tau: float) -> dict[str, float | str]:
    usable = [row for row in rows if row.compliance_correct is not None]
    if not usable:
        return {"skipped": "run experiments/baselines/local_qwen35.py first"}
    summary = metrics.summarize(
        [row.final_answerability_score for row in usable],
        [int(row.compliance_correct) for row in usable],
        tau=tau,
        target_risk=config.TARGET_RISK,
        n_bins=config.ECE_BINS,
    )
    summary["raw_label_accuracy_all_pairs"] = sum(
        int(row.raw_compliance_correct) for row in usable
    ) / len(usable)
    sufficient = [row for row in usable if row.sufficiency_gold == 1]
    summary["label_accuracy_on_sufficient_pairs"] = (
        sum(int(row.raw_compliance_correct) for row in sufficient) / len(sufficient)
        if sufficient
        else float("nan")
    )
    return summary


def main() -> None:
    device = _device()
    split = splits.load_split()
    head = load_head(device)

    test_records = ds.build_pair_records(set(split["test"]))
    test_checkpoint_probs = checkpoint_probs(head, test_records, device)

    val_rows = build_eval_rows(head, set(split["val"]), device) if split["val"] else []
    test_rows = build_eval_rows(head, set(split["test"]), device)
    if not test_rows:
        raise RuntimeError("No locked-gold test pairs yet; annotate more or check the split.")

    if val_rows:
        tau = metrics.tune_tau_for_risk(
            [r.final_answerability_score for r in val_rows],
            [r.sufficiency_gold for r in val_rows],
            config.TARGET_RISK,
        )
    else:
        tau = 0.5
        print("[warn] no val pairs locked yet; falling back to tau=0.5")

    visual_val_rows = [row for row in val_rows if row.external_gate_passed]
    visual_test_rows = [row for row in test_rows if row.external_gate_passed]
    if visual_val_rows:
        visual_tau = metrics.tune_tau_for_risk(
            [row.visual_sufficiency_score for row in visual_val_rows],
            [row.sufficiency_gold for row in visual_val_rows],
            config.TARGET_RISK,
        )
    else:
        visual_tau = 0.5

    task2_usable_val = [row for row in val_rows if row.compliance_correct is not None]
    if task2_usable_val:
        task2_tau = metrics.tune_tau_for_risk(
            [row.final_answerability_score for row in task2_usable_val],
            [int(row.compliance_correct) for row in task2_usable_val],
            config.TARGET_RISK,
        )
    else:
        task2_tau = tau

    report = {
        "task1_tau": tau,
        "visual_task1_tau": visual_tau,
        "task2_tau": task2_tau,
        "n_test": len(test_rows),
        "checkpoint_resolvability": checkpoint_metrics(test_checkpoint_probs, test_records),
        "visual_pair_sufficiency_detection": task1_metrics(
            visual_test_rows,
            visual_tau,
            score_field="visual_sufficiency_score",
        ),
        "task1_sufficiency_detection": task1_metrics(test_rows, tau),
        "task2_selective_compliance": task2_metrics(test_rows, task2_tau),
        "ablations": {
            "no_checkpoint_supervision": config.ABLATE_NO_CHECKPOINT_SUPERVISION,
            "no_is_visual": config.ABLATE_NO_IS_VISUAL,
            "head_uses_is_visual": head.use_is_visual,
            "external_gate_separate": True,
            "no_text_conditioning": config.ABLATE_NO_TEXT_CONDITIONING,
            "aggregation_mode": config.AGGREGATION_MODE,
        },
    }

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ds.exp_io.rio.write_jsonl(config.PREDICTIONS_PATH, [asdict(r) for r in test_rows])
    config.RCES_EVAL_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote eval table -> {config.PREDICTIONS_PATH.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
