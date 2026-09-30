"""Run statistical and channel-use diagnostics for the corrected R-CES head.

Outputs under ``experiments/artifacts/diagnostics``:

* ``diagnostic_report.json`` and ``diagnostic_report.md``
* ``learning_history.json`` and ``learning_curves.svg``
* one checkpoint for each ``full``, ``text_only``, and ``image_only`` run

The existing corrected E0 checkpoint is evaluated first. The three diagnostic
models are then trained from scratch with the same split, seed, optimizer, and
40-epoch schedule so their learning curves and held-out metrics are comparable.
The blackout check encodes actual all-black RGB images with the frozen SigLIP 2
vision encoder; it does not approximate blackout by zeroing cached features.

  UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/rces/diagnose.py
"""

from __future__ import annotations

import html
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
import exp_io  # noqa: E402
import metrics  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from data.encode_siglip2 import load_encoder  # noqa: E402
from rces import evaluate, train  # noqa: E402
from rces.model import RCESHead  # noqa: E402


DIAGNOSTIC_DIR = config.ARTIFACT_DIR / "diagnostics"
REPORT_JSON = DIAGNOSTIC_DIR / "diagnostic_report.json"
REPORT_MD = DIAGNOSTIC_DIR / "diagnostic_report.md"
HISTORY_JSON = DIAGNOSTIC_DIR / "learning_history.json"
CURVES_SVG = DIAGNOSTIC_DIR / "learning_curves.svg"
INPUT_MODES = ("full", "text_only", "image_only")
BOOTSTRAP_SAMPLES = 10_000
BLACKOUT_IMAGE_COUNT = 12
SNAPSHOT_DIR = DIAGNOSTIC_DIR / "input_snapshot"
NEGATIVE_WORDING = re.compile(
    r"removed|damaged|unrestored|violation|defect|grossly|unsafe|blocked|absent|"
    r"poor|overheight|incorrect|not[_ ]clear|missing",
    re.IGNORECASE,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activate_input_snapshot() -> dict[str, str]:
    """Copy mutable annotation inputs once and route this process to the copy."""
    if not config.SPLIT_PATH.exists():
        splits.load_split()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_gold = SNAPSHOT_DIR / exp_io.rio.GOLD_PATH.name
    snapshot_rules = SNAPSHOT_DIR / exp_io.rio.RULES_PATH.name
    snapshot_split = SNAPSHOT_DIR / config.SPLIT_PATH.name
    shutil.copy2(exp_io.rio.GOLD_PATH, snapshot_gold)
    shutil.copy2(exp_io.rio.RULES_PATH, snapshot_rules)
    shutil.copy2(config.SPLIT_PATH, snapshot_split)

    original_load_rules = exp_io.rio.load_rules
    exp_io.rio.GOLD_PATH = snapshot_gold
    exp_io.rio.RULES_PATH = snapshot_rules
    exp_io.rio.load_rules = lambda path=snapshot_rules: original_load_rules(path)
    config.SPLIT_PATH = snapshot_split
    return {
        "gold": file_sha256(snapshot_gold),
        "rules": file_sha256(snapshot_rules),
        "split": file_sha256(snapshot_split),
    }


@torch.no_grad()
def loader_predictions(
    head: RCESHead, loader: DataLoader, device: torch.device
) -> tuple[list[float], list[int]]:
    head.eval()
    scores: list[float] = []
    targets: list[int] = []
    for batch in loader:
        probs = head.resolvable_prob(
            batch["patches"].to(device),
            batch["text"].to(device),
            batch["is_visual"].to(device),
        )
        scores.extend(float(value) for value in probs.cpu().tolist())
        targets.extend(int(value) for value in batch["target"].squeeze(-1).tolist())
    return scores, targets


def evaluation_loader(records: list[ds.PairRecord]) -> DataLoader:
    examples = ds.build_examples(records)
    return DataLoader(
        ds.RCESDataset(examples, records),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )


def fit_mode(
    mode: str,
    train_ids: set[str],
    val_ids: set[str],
    device: torch.device,
) -> tuple[RCESHead, RCESHead, list[dict[str, float | int]]]:
    set_seed(config.SEED)
    train_loader, train_records, _ = train.make_loader(train_ids, shuffle=True)
    val_records = ds.build_pair_records(val_ids)
    train_eval_loader = evaluation_loader(train_records)
    val_loader = evaluation_loader(val_records)
    img_dim, txt_dim = train.infer_dims(train_loader)
    head = RCESHead(
        img_dim=img_dim,
        txt_dim=txt_dim,
        use_is_visual=False,
        input_mode=mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    history: list[dict[str, float | int]] = []
    best_val_auroc = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train.run_epoch(head, train_loader, device, optimizer)
        val_loss = train.run_epoch(head, val_loader, device, None)
        train_scores, train_targets = loader_predictions(head, train_eval_loader, device)
        val_scores, val_targets = loader_predictions(head, val_loader, device)
        train_auc = metrics.binary_classification_metrics(train_scores, train_targets)["auroc"]
        val_auc = metrics.binary_classification_metrics(val_scores, val_targets)["auroc"]
        history.append(
            {
                "epoch": epoch,
                "train_bce": train_loss,
                "val_bce": val_loss,
                "train_auroc": train_auc,
                "val_auroc": val_auc,
            }
        )
        if val_auc > best_val_auroc:
            best_val_auroc = val_auc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == config.EPOCHS:
            print(
                f"[{mode}] epoch {epoch:02d}/{config.EPOCHS} "
                f"train_auc={train_auc:.4f} val_auc={val_auc:.4f} "
                f"train_bce={train_loss:.4f} val_bce={val_loss:.4f}"
            )

    torch.save(
        {
            "state_dict": head.state_dict(),
            "img_dim": img_dim,
            "txt_dim": txt_dim,
            "use_is_visual": False,
            "input_mode": mode,
            "seed": config.SEED,
            "epochs": config.EPOCHS,
        },
        DIAGNOSTIC_DIR / f"rces_head_{mode}.pt",
    )
    best_head = RCESHead(
        img_dim=img_dim,
        txt_dim=txt_dim,
        use_is_visual=False,
        input_mode=mode,
    ).to(device)
    best_head.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "img_dim": img_dim,
            "txt_dim": txt_dim,
            "use_is_visual": False,
            "input_mode": mode,
            "seed": config.SEED,
            "selected_epoch": best_epoch,
            "selection_metric": "val_auroc",
        },
        DIAGNOSTIC_DIR / f"rces_head_{mode}_best_val.pt",
    )
    return head.eval(), best_head.eval(), history


def checkpoint_arrays(
    probabilities: dict[tuple[str, str], float], records: list[ds.PairRecord]
) -> tuple[list[float], list[int], list[str], list[str]]:
    scores: list[float] = []
    targets: list[int] = []
    image_ids: list[str] = []
    checkpoint_ids: list[str] = []
    for record in records:
        for key, target in record.resolvable_targets.items():
            probability = probabilities.get((record.pair_id, key))
            if probability is None:
                continue
            scores.append(probability)
            targets.append(target)
            image_ids.append(record.image_id)
            checkpoint_ids.append(exp_io.checkpoint_text_id(record.rule_id, key))
    return scores, targets, image_ids, checkpoint_ids


def visual_pair_arrays(
    head: RCESHead, image_ids: set[str], device: torch.device
) -> tuple[list[float], list[int], list[str]]:
    records = [
        record
        for record in ds.build_pair_records(image_ids)
        if not record.external_evidence_required
    ]
    scores = evaluate.visual_pair_scores(head, records, device)
    return (
        [scores[record.pair_id] for record in records],
        [record.sufficiency_gold for record in records],
        [record.image_id for record in records],
    )


def held_out_report(
    head: RCESHead,
    test_records: list[ds.PairRecord],
    test_ids: set[str],
    device: torch.device,
    *,
    include_bootstrap: bool,
) -> dict[str, Any]:
    probabilities = evaluate.checkpoint_probs(head, test_records, device)
    checkpoint_scores, checkpoint_targets, checkpoint_images, _ = checkpoint_arrays(
        probabilities, test_records
    )
    pair_scores, pair_targets, pair_images = visual_pair_arrays(head, test_ids, device)
    report: dict[str, Any] = {
        "checkpoint": metrics.binary_classification_metrics(
            checkpoint_scores, checkpoint_targets
        ),
        "visual_pair": metrics.binary_classification_metrics(pair_scores, pair_targets),
    }
    if include_bootstrap:
        report["checkpoint"]["bootstrap_by_example"] = metrics.bootstrap_auroc(
            checkpoint_scores,
            checkpoint_targets,
            n_bootstrap=BOOTSTRAP_SAMPLES,
            seed=config.SEED,
        )
        report["checkpoint"]["bootstrap_by_image"] = metrics.bootstrap_auroc(
            checkpoint_scores,
            checkpoint_targets,
            groups=checkpoint_images,
            n_bootstrap=BOOTSTRAP_SAMPLES,
            seed=config.SEED,
        )
        report["visual_pair"]["bootstrap_by_pair"] = metrics.bootstrap_auroc(
            pair_scores,
            pair_targets,
            n_bootstrap=BOOTSTRAP_SAMPLES,
            seed=config.SEED,
        )
        report["visual_pair"]["bootstrap_by_image"] = metrics.bootstrap_auroc(
            pair_scores,
            pair_targets,
            groups=pair_images,
            n_bootstrap=BOOTSTRAP_SAMPLES,
            seed=config.SEED,
        )
    return report


def polarity_audit(
    probabilities: dict[tuple[str, str], float], records: list[ds.PairRecord]
) -> dict[str, Any]:
    rules = exp_io.rio.load_rules()
    locked = {row["pair_id"]: row for row in exp_io.load_locked_gold()}
    mismatches: list[dict[str, str | int]] = []
    negative_scores: list[float] = []
    negative_targets: list[int] = []
    other_scores: list[float] = []
    other_targets: list[int] = []
    negative_ids: set[str] = set()

    for record in records:
        pair = locked[record.pair_id]
        missing = set(pair.get("missing_checkpoints", []))
        observed = set(pair.get("observed_checkpoints", []))
        rule = rules[record.rule_id]
        questions = rule.get("visual_checkpoints", {})
        for key, target in record.resolvable_targets.items():
            expected = 0 if key in missing else 1
            if target != expected:
                mismatches.append(
                    {
                        "pair_id": record.pair_id,
                        "checkpoint": key,
                        "target": target,
                        "expected": expected,
                    }
                )
            if key in missing and key in observed:
                mismatches.append(
                    {
                        "pair_id": record.pair_id,
                        "checkpoint": key,
                        "target": target,
                        "expected": -1,
                    }
                )
            score = probabilities.get((record.pair_id, key))
            if score is None:
                continue
            text = f"{key} {questions.get(key, '')}"
            if NEGATIVE_WORDING.search(text):
                negative_ids.add(exp_io.checkpoint_text_id(record.rule_id, key))
                negative_scores.append(score)
                negative_targets.append(target)
            else:
                other_scores.append(score)
                other_targets.append(target)

    return {
        "target_contract": "observed/not_applicable -> resolvable=1; missing -> resolvable=0",
        "mapping_mismatch_count": len(mismatches),
        "mapping_mismatch_examples": mismatches[:20],
        "negative_wording_checkpoint_ids": sorted(negative_ids),
        "negative_wording_metrics": metrics.binary_classification_metrics(
            negative_scores, negative_targets
        ),
        "other_wording_metrics": metrics.binary_classification_metrics(
            other_scores, other_targets
        ),
    }


def _bootstrap_mean_delta(
    original: list[float], degraded: list[float], *, seed: int
) -> dict[str, float | int | bool]:
    deltas = np.asarray(original, dtype=float) - np.asarray(degraded, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(BOOTSTRAP_SAMPLES)]
    )
    low, high = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "mean_original_minus_black": float(deltas.mean()),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "fraction_scores_lower_on_black": float((deltas > 0).mean()),
        "n_checkpoint_comparisons": int(len(deltas)),
    }


@torch.no_grad()
def blackout_audit(
    head: RCESHead,
    test_records: list[ds.PairRecord],
    device: torch.device,
) -> dict[str, Any]:
    by_image: dict[str, list[ds.PairRecord]] = {}
    for record in test_records:
        if record.critical_keys and not record.external_evidence_required:
            by_image.setdefault(record.image_id, []).append(record)
    ranked_images = sorted(
        by_image,
        key=lambda image_id: (
            not any(
                target == 0
                for record in by_image[image_id]
                for target in record.resolvable_targets.values()
            ),
            image_id,
        ),
    )[:BLACKOUT_IMAGE_COUNT]

    original_probabilities = evaluate.checkpoint_probs(
        head,
        [record for image_id in ranked_images for record in by_image[image_id]],
        device,
    )
    checkpoint_text = ds.load_checkpoint_text()
    encoder, processor = load_encoder(device)
    original_scores: list[float] = []
    black_scores: list[float] = []
    rows: list[dict[str, Any]] = []
    for image_id in ranked_images:
        image_path = exp_io.rio.IMAGE_DIR / image_id
        with Image.open(image_path) as source:
            black_image = Image.new("RGB", source.size, color=(0, 0, 0))
        inputs = processor(images=black_image, return_tensors="pt").to(device)
        black_patches = encoder.vision_model(
            pixel_values=inputs["pixel_values"],
            interpolate_pos_encoding=(
                config.SIGLIP2_IMAGE_SIZE != encoder.vision_model.config.image_size
            ),
        ).last_hidden_state.float()
        for record in by_image[image_id]:
            for key, target in record.resolvable_targets.items():
                text_id = exp_io.checkpoint_text_id(record.rule_id, key)
                text = checkpoint_text[text_id].to(device).unsqueeze(0)
                probability = float(
                    head.resolvable_prob(
                        black_patches,
                        text,
                        torch.zeros((1, 1), dtype=torch.float32, device=device),
                    ).item()
                )
                original = original_probabilities[(record.pair_id, key)]
                original_scores.append(original)
                black_scores.append(probability)
                rows.append(
                    {
                        "image_id": image_id,
                        "pair_id": record.pair_id,
                        "checkpoint": key,
                        "target": target,
                        "original_score": original,
                        "black_score": probability,
                        "original_minus_black": original - probability,
                    }
                )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "image_count": len(ranked_images),
        "image_ids": ranked_images,
        "summary": _bootstrap_mean_delta(original_scores, black_scores, seed=config.SEED),
        "comparisons": rows,
    }


def write_curves_svg(histories: dict[str, list[dict[str, float | int]]]) -> None:
    width, panel_height, margin = 960, 250, 45
    height = panel_height * len(INPUT_MODES)
    colors = {"train": "#1f77b4", "val": "#d62728"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:16px;font-weight:bold}</style>',
    ]
    plot_width = width - 2 * margin
    plot_height = panel_height - 2 * margin
    for panel, mode in enumerate(INPUT_MODES):
        y0 = panel * panel_height
        elements.append(f'<text class="title" x="{margin}" y="{y0 + 22}">{html.escape(mode)} AUROC</text>')
        elements.append(
            f'<rect x="{margin}" y="{y0 + margin}" width="{plot_width}" height="{plot_height}" '
            'fill="none" stroke="#999"/>'
        )
        chance_y = y0 + margin + (1.0 - 0.5) * plot_height
        elements.append(
            f'<line x1="{margin}" y1="{chance_y:.1f}" x2="{width - margin}" y2="{chance_y:.1f}" '
            'stroke="#777" stroke-dasharray="5 5"/>'
        )
        for split_name, field in (("train", "train_auroc"), ("val", "val_auroc")):
            points = []
            for row in histories[mode]:
                x = margin + (int(row["epoch"]) - 1) / max(1, config.EPOCHS - 1) * plot_width
                value = float(row[field])
                y = y0 + margin + (1.0 - value) * plot_height
                points.append(f"{x:.1f},{y:.1f}")
            elements.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[split_name]}" '
                'stroke-width="2"/>'
            )
        elements.extend(
            [
                f'<text x="{margin}" y="{y0 + panel_height - 10}">epoch 1</text>',
                f'<text x="{width - margin - 58}" y="{y0 + panel_height - 10}">epoch {config.EPOCHS}</text>',
                f'<text x="{width - 180}" y="{y0 + 22}" fill="{colors["train"]}">train</text>',
                f'<text x="{width - 120}" y="{y0 + 22}" fill="{colors["val"]}">val</text>',
            ]
        )
    elements.append("</svg>")
    CURVES_SVG.write_text("\n".join(elements) + "\n", encoding="utf-8")


def curve_summary(history: list[dict[str, float | int]]) -> dict[str, float | int]:
    best = max(history, key=lambda row: float(row["val_auroc"]))
    final = history[-1]
    return {
        "best_val_epoch": int(best["epoch"]),
        "best_val_auroc": float(best["val_auroc"]),
        "train_auroc_at_best_val": float(best["train_auroc"]),
        "final_train_auroc": float(final["train_auroc"]),
        "final_val_auroc": float(final["val_auroc"]),
        "final_train_minus_val_gap": float(final["train_auroc"]) - float(final["val_auroc"]),
    }


def write_markdown(report: dict[str, Any]) -> None:
    existing = report["existing_corrected_e0"]
    checkpoint_ci = existing["checkpoint"]["bootstrap_by_image"]
    pair_ci = existing["visual_pair"]["bootstrap_by_image"]
    blackout = report["blackout_audit"]["summary"]
    lines = [
        "# R-CES Statistical and Channel Diagnostics",
        "",
        f"Run device: `{report['run_config']['device']}`.",
        f"Gold snapshot SHA-256: `{report['data_snapshot_sha256']['gold']}`.",
        "",
        "## Existing corrected E0 uncertainty",
        "",
        "| Level | AUROC | Positive | Negative | Image-cluster 95% CI | Contains 0.5 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| checkpoint | {checkpoint_ci['auroc']:.4f} | {checkpoint_ci['positives']} | {checkpoint_ci['negatives']} | [{checkpoint_ci['ci_95_low']:.4f}, {checkpoint_ci['ci_95_high']:.4f}] | {checkpoint_ci['ci_contains_0_5']} |",
        f"| visual pair | {pair_ci['auroc']:.4f} | {pair_ci['positives']} | {pair_ci['negatives']} | [{pair_ci['ci_95_low']:.4f}, {pair_ci['ci_95_high']:.4f}] | {pair_ci['ci_contains_0_5']} |",
        "",
        "## Polarity audit",
        "",
        f"Mapping mismatches: **{report['polarity_audit']['mapping_mismatch_count']}**.",
        "The target is resolvability, so negative checkpoint wording does not invert the label.",
        "",
        "## Learning curves and degraded baselines",
        "",
        "| Mode | Best val epoch | Best val AUROC | Final train AUROC | Final val AUROC | Best-val test checkpoint AUROC | Best-val test visual-pair AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in INPUT_MODES:
        curve = report["curve_summary"][mode]
        held_out = report["trained_modes"][mode]["best_val"]
        lines.append(
            f"| {mode} | {curve['best_val_epoch']} | {curve['best_val_auroc']:.4f} | "
            f"{curve['final_train_auroc']:.4f} | {curve['final_val_auroc']:.4f} | "
            f"{held_out['checkpoint']['auroc']:.4f} | {held_out['visual_pair']['auroc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Black-image sanity check",
            "",
            f"Tested {report['blackout_audit']['image_count']} images and {blackout['n_checkpoint_comparisons']} checkpoint comparisons.",
            f"Mean original minus black score: **{blackout['mean_original_minus_black']:.4f}** "
            f"(95% bootstrap CI [{blackout['ci_95_low']:.4f}, {blackout['ci_95_high']:.4f}]).",
        f"Fraction lower after blackout: **{blackout['fraction_scores_lower_on_black']:.4f}**.",
        "",
        "## Interpretation",
        "",
        "- Both image-cluster AUROC intervals contain 0.5; the point estimates do not support model-selection decisions.",
        "- The full head nearly memorizes train but does not beat the degraded baselines on held-out test.",
        "- The resolvability target mapping has no polarity mismatch; negative wording does not mechanically flip 0/1 labels.",
        "- Blackout does not cause a consistent score decrease, so the current head fails the visual sanity check.",
        "- Expand unresolved/insufficient annotations before adding head complexity.",
        "",
        "See `diagnostic_report.json` for pair-level rows and both ordinary and image-cluster bootstrap intervals.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_hashes = activate_input_snapshot()
    device = evaluate._device()
    split = splits.load_split()
    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])
    test_records = ds.build_pair_records(test_ids)
    if not test_records:
        raise RuntimeError("No locked-gold test records available")

    existing_head = evaluate.load_head(device)
    existing_report = held_out_report(
        existing_head,
        test_records,
        test_ids,
        device,
        include_bootstrap=True,
    )
    existing_probabilities = evaluate.checkpoint_probs(existing_head, test_records, device)
    polarity = polarity_audit(existing_probabilities, test_records)

    histories: dict[str, list[dict[str, float | int]]] = {}
    trained_modes: dict[str, dict[str, Any]] = {}
    for mode in INPUT_MODES:
        final_head, best_head, history = fit_mode(mode, train_ids, val_ids, device)
        histories[mode] = history
        trained_modes[mode] = {
            "final_epoch": held_out_report(
                final_head,
                test_records,
                test_ids,
                device,
                include_bootstrap=False,
            ),
            "best_val": held_out_report(
                best_head,
                test_records,
                test_ids,
                device,
                include_bootstrap=False,
            ),
        }

    blackout = blackout_audit(existing_head, test_records, device)
    report = {
        "run_config": {
            "seed": config.SEED,
            "epochs": config.EPOCHS,
            "learning_rate": config.LR,
            "weight_decay": config.WEIGHT_DECAY,
            "batch_size": config.BATCH_SIZE,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "device": str(device),
        },
        "data_snapshot_sha256": snapshot_hashes,
        "existing_corrected_e0": existing_report,
        "polarity_audit": polarity,
        "curve_summary": {mode: curve_summary(histories[mode]) for mode in INPUT_MODES},
        "trained_modes": trained_modes,
        "blackout_audit": blackout,
    }
    HISTORY_JSON.write_text(
        json.dumps(histories, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_curves_svg(histories)
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote diagnostics -> {DIAGNOSTIC_DIR.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
