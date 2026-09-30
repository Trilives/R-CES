"""R-CES v2 ablation matrix (Readme step 9) for the text-context-first head.

Mirrors experiments/ablation/run.py but for the isolated v2 architecture. Each
arm is its own train+evaluate with isolated artifacts, then a comparison table is
printed. The is_visual ablation is intentionally absent: v2 never takes an
is_visual input (external evidence is a post-head gate), so there is nothing to
remove.

  full                       the v2 method (dense per-checkpoint BCE + MIN)
  no_checkpoint_supervision  train on pair sufficiency (soft-AND), not per-k
  no_text_conditioning       zero the checkpoint text -> image-only head
  aggregation_mean           reuse the full head, replace MIN with mean
  aggregation_product        reuse the full head, replace MIN with product

  RCES_VISION_ENCODER=siglip2_512 RCES_INCLUDE_CROP_AUG=1 \
  RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B UV_CACHE_DIR=/tmp/uv-cache \
  uv run python experiments/rces-2/ablation.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
RCES_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(RCES_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import config  # noqa: E402
import metrics  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces import evaluate as base_evaluate  # noqa: E402
from rces2_model import ContextualRCESHead  # noqa: E402
from rces2_settings import (  # noqa: E402
    ARCHITECTURE,
    ARTIFACT_DIR,
    CONTEXT_HEADS,
    CONTEXT_LAYERS,
    HIDDEN_DIM,
    IMAGE_HEADS,
)
from rces2_training import run_epoch  # noqa: E402
import evaluate as v2_evaluate  # noqa: E402
from train import device, make_loader, seed_everything  # noqa: E402


ABLATIONS: list[tuple[str, dict[str, Any]]] = [
    ("full", {}),
    ("no_checkpoint_supervision", {"ABLATE_NO_CHECKPOINT_SUPERVISION": True}),
    ("no_text_conditioning", {"ABLATE_NO_TEXT_CONDITIONING": True}),
    ("aggregation_mean", {"AGGREGATION_MODE": "mean"}),
    ("aggregation_product", {"AGGREGATION_MODE": "product"}),
]


def _reset_ablations() -> None:
    config.ABLATE_NO_CHECKPOINT_SUPERVISION = False
    config.ABLATE_NO_TEXT_CONDITIONING = False
    config.AGGREGATION_MODE = "min"


def train_head(
    active_device: torch.device,
    train_records: list[ds.PairRecord],
    val_records: list[ds.PairRecord],
) -> tuple[ContextualRCESHead, int]:
    seed_everything(config.SEED)
    train_loader, _ = make_loader(train_records, shuffle=True)
    val_loader = make_loader(val_records, shuffle=False)[0] if val_records else None
    sample = next(iter(train_loader))
    head = ContextualRCESHead(
        img_dim=int(sample["patches"].shape[-1]),
        txt_dim=int(sample["checkpoint_text"].shape[-1]),
        hidden=HIDDEN_DIM,
        dropout=config.DROPOUT,
        context_heads=CONTEXT_HEADS,
        image_heads=IMAGE_HEADS,
        context_layers=CONTEXT_LAYERS,
    ).to(active_device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    best_val = float("inf")
    for epoch in range(1, config.EPOCHS + 1):
        run_epoch(head, train_loader, active_device, optimizer)
        if val_loader is not None:
            val_loss = run_epoch(head, val_loader, active_device, None)
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(head.state_dict())
        else:
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    return head.eval(), best_epoch


def evaluate_head(
    head: ContextualRCESHead,
    val_records: list[ds.PairRecord],
    test_records: list[ds.PairRecord],
    active_device: torch.device,
) -> dict[str, Any]:
    val_rows = (
        v2_evaluate.build_eval_rows(head, val_records, active_device)
        if val_records
        else []
    )
    test_rows = v2_evaluate.build_eval_rows(head, test_records, active_device)
    test_probs = v2_evaluate.probabilities_for_records(
        head, test_records, active_device
    )
    tau = (
        metrics.tune_tau_for_risk(
            [r.final_answerability_score for r in val_rows],
            [r.sufficiency_gold for r in val_rows],
            config.TARGET_RISK,
        )
        if val_rows
        else 0.5
    )
    task2_val = [r for r in val_rows if r.compliance_correct is not None]
    task2_tau = (
        metrics.tune_tau_for_risk(
            [r.final_answerability_score for r in task2_val],
            [int(r.compliance_correct) for r in task2_val],
            config.TARGET_RISK,
        )
        if task2_val
        else tau
    )
    return {
        "aggregation_mode": config.AGGREGATION_MODE,
        "no_checkpoint_supervision": config.ABLATE_NO_CHECKPOINT_SUPERVISION,
        "no_text_conditioning": config.ABLATE_NO_TEXT_CONDITIONING,
        "task1_tau": tau,
        "task2_tau": task2_tau,
        "n_test": len(test_rows),
        "checkpoint_resolvability": base_evaluate.checkpoint_metrics(
            test_probs, test_records
        ),
        "task1_sufficiency_detection": base_evaluate.task1_metrics(test_rows, tau),
        "task2_selective_compliance": base_evaluate.task2_metrics(
            test_rows, task2_tau
        ),
    }


def main() -> None:
    active_device = device()
    split = splits.load_split()
    train_records = ds.build_pair_records(set(split["train"]))
    val_records = ds.build_pair_records(set(split["val"]))
    test_records = ds.build_pair_records(set(split["test"]))
    if not test_records:
        raise RuntimeError("No locked-gold test pairs available")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    full_head: ContextualRCESHead | None = None
    reports: list[dict[str, Any]] = []
    for name, overrides in ABLATIONS:
        print(f"\n===== v2 ablation: {name} =====")
        _reset_ablations()
        for key, value in overrides.items():
            setattr(config, key, value)
        aggregation_only = name.startswith("aggregation_")
        if aggregation_only:
            if full_head is None:
                print(f"[ablation {name}] skipped: full head missing")
                continue
            head, best_epoch = full_head, -1
        else:
            head, best_epoch = train_head(active_device, train_records, val_records)
            torch.save(
                {"architecture": ARCHITECTURE, "state_dict": head.state_dict()},
                ARTIFACT_DIR / f"ablation_{name}_head.pt",
            )
            if name == "full":
                full_head = head
        report = {"ablation": name, "best_epoch": best_epoch}
        report.update(evaluate_head(head, val_records, test_records, active_device))
        (ARTIFACT_DIR / f"ablation_{name}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(report)
    _reset_ablations()

    print("\n===== v2 ablation comparison (task 1) =====")
    header = f"{'ablation':<28}{'ckpt_AUROC':>11}{'AURC':>8}{'CMCR':>8}{'cov@risk':>9}{'ECE':>8}"
    print(header)
    for report in reports:
        t1 = report["task1_sufficiency_detection"]
        ck = report["checkpoint_resolvability"]
        print(
            f"{report['ablation']:<28}{ck.get('auroc', float('nan')):>11.4f}"
            f"{t1.get('aurc', float('nan')):>8.4f}{t1.get('cmcr', float('nan')):>8.4f}"
            f"{t1.get('coverage_at_target_risk', float('nan')):>9.4f}"
            f"{t1.get('ece', float('nan')):>8.4f}"
        )
    (ARTIFACT_DIR / "ablation_summary.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote v2 ablation reports -> {ARTIFACT_DIR.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
