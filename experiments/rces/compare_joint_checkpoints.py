"""Train and compare independent vs joint-checkpoint R-CES heads.

This is an isolated experiment: it does not replace the default corrected E0
artifacts. Both arms use the same split, frozen 512px SigLIP 2 cache, optimizer,
epoch budget, seed, and validation checkpoint-AUROC selection rule.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
import exp_io  # noqa: E402
import metrics  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces import aggregate, evaluate, train  # noqa: E402
from rces.joint_dataset import (  # noqa: E402
    JointCheckpointDataset,
    build_joint_examples,
    collate_joint_batch,
)
from rces.joint_model import JointRCESHead  # noqa: E402
from rces.model import RCESHead  # noqa: E402


BOOTSTRAP_SAMPLES = 10_000
JOINT_HIDDEN = 176
JOINT_BATCH_SIZE = 32
OUTPUT_DIR = config.ARTIFACT_DIR / "joint_checkpoints"
CONTROL_CKPT = OUTPUT_DIR / "independent_best_val.pt"
JOINT_CKPT = OUTPUT_DIR / "joint_best_val.pt"
OUTPUT_JSON = OUTPUT_DIR / "comparison.json"
OUTPUT_MD = OUTPUT_DIR / "comparison.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_joint_loader(
    records: list[ds.PairRecord], *, shuffle: bool
) -> tuple[DataLoader, int]:
    examples = build_joint_examples(records)
    if not examples:
        raise RuntimeError("No joint-checkpoint examples for this split")
    generator = torch.Generator().manual_seed(config.SEED)
    return (
        DataLoader(
            JointCheckpointDataset(examples),
            batch_size=JOINT_BATCH_SIZE,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=collate_joint_batch,
            generator=generator,
        ),
        len(examples),
    )


def masked_bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, supervision_mask: torch.Tensor
) -> torch.Tensor:
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    weights = supervision_mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def run_joint_epoch(
    head: JointRCESHead,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    head.train(training)
    total_loss = 0.0
    total_targets = 0
    for batch in loader:
        patches = batch["patches"].to(device)
        checkpoint_text = batch["checkpoint_text"].to(device)
        checkpoint_mask = batch["checkpoint_mask"].to(device)
        critical_mask = batch["critical_mask"].to(device)
        targets = batch["target"].to(device)
        supervision_mask = batch["supervision_mask"].to(device)
        with torch.set_grad_enabled(training):
            logits = head(patches, checkpoint_text, checkpoint_mask, critical_mask)
            loss = masked_bce_loss(logits, targets, supervision_mask)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), config.GRAD_CLIP)
                optimizer.step()
        target_count = int(supervision_mask.sum())
        total_loss += float(loss.detach()) * target_count
        total_targets += target_count
    return total_loss / max(1, total_targets)


@torch.no_grad()
def joint_checkpoint_probs(
    head: JointRCESHead,
    records: list[ds.PairRecord],
    device: torch.device,
) -> dict[tuple[str, str], float]:
    loader, _ = make_joint_loader(records, shuffle=False)
    probabilities: dict[tuple[str, str], float] = {}
    for batch in loader:
        probs = head.resolvable_prob(
            batch["patches"].to(device),
            batch["checkpoint_text"].to(device),
            batch["checkpoint_mask"].to(device),
            batch["critical_mask"].to(device),
        ).cpu()
        masks = batch["supervision_mask"]
        for row, pair_id in enumerate(batch["pair_id"]):
            for column, key in enumerate(batch["checkpoint_keys"][row]):
                if bool(masks[row, column]):
                    probabilities[(pair_id, key)] = float(probs[row, column])
    return probabilities


def pair_scores(
    probabilities: dict[tuple[str, str], float], records: list[ds.PairRecord]
) -> dict[str, float]:
    return {
        record.pair_id: aggregate.aggregate_pair_score(
            record,
            {
                key: probabilities.get((record.pair_id, key), 1.0)
                for key in record.critical_keys
            },
        )
        for record in records
    }


def checkpoint_arrays(
    probabilities: dict[tuple[str, str], float], records: list[ds.PairRecord]
) -> tuple[list[float], list[int], list[str]]:
    scores: list[float] = []
    targets: list[int] = []
    images: list[str] = []
    for record in records:
        for key, target in record.resolvable_targets.items():
            scores.append(probabilities[(record.pair_id, key)])
            targets.append(target)
            images.append(record.image_id)
    return scores, targets, images


def auroc(
    probabilities: dict[tuple[str, str], float], records: list[ds.PairRecord]
) -> float:
    scores, targets, _ = checkpoint_arrays(probabilities, records)
    return float(metrics.binary_classification_metrics(scores, targets)["auroc"])


def train_with_best_val(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_epoch: Callable[[torch.optim.Optimizer | None], float],
    val_epoch: Callable[[torch.optim.Optimizer | None], float],
    val_probabilities: Callable[[], dict[tuple[str, str], float]],
    val_records: list[ds.PairRecord],
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]], int]:
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_auroc = float("-inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(optimizer)
        with torch.no_grad():
            val_loss = val_epoch(None)
            val_score = auroc(val_probabilities(), val_records)
        history.append(
            {
                "epoch": epoch,
                "train_bce": train_loss,
                "val_bce": val_loss,
                "val_checkpoint_auroc": val_score,
            }
        )
        selection_score = val_score if np.isfinite(val_score) else float("-inf")
        if selection_score > best_auroc:
            best_auroc = selection_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0 or epoch == config.EPOCHS:
            print(
                f"epoch {epoch:02d}/{config.EPOCHS} train={train_loss:.4f} "
                f"val={val_loss:.4f} val_auroc={val_score:.4f}"
            )
    return best_state, history, best_epoch


def paired_cluster_delta(
    control: list[float],
    candidate: list[float],
    targets: list[int],
    images: list[str],
) -> dict[str, float | int | bool]:
    first = np.asarray(control, dtype=float)
    second = np.asarray(candidate, dtype=float)
    target = np.asarray(targets, dtype=int)
    families = exp_io.original_image_families(images)
    group = np.asarray(families, dtype=object)
    unique_groups = list(dict.fromkeys(families))
    indices_by_group = {
        image_id: np.flatnonzero(group == image_id) for image_id in unique_groups
    }
    rng = np.random.default_rng(config.SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[image_id] for image_id in sampled])
        delta = metrics._auroc(second[indices], target[indices]) - metrics._auroc(
            first[indices], target[indices]
        )
        if np.isfinite(delta):
            values.append(float(delta))
    low, high = np.quantile(np.asarray(values), [0.025, 0.975]).tolist()
    delta = metrics._auroc(second, target) - metrics._auroc(first, target)
    return {
        "delta_joint_minus_independent": float(delta),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "ci_contains_zero": bool(low <= 0 <= high),
        "n_groups": len(unique_groups),
        "valid_bootstrap_samples": len(values),
    }


def arm_report(
    checkpoint_scores: list[float],
    checkpoint_targets: list[int],
    checkpoint_images: list[str],
    pair_score_values: list[float],
    pair_targets: list[int],
    pair_images: list[str],
) -> dict[str, Any]:
    return {
        "checkpoint": metrics.bootstrap_auroc(
            checkpoint_scores,
            checkpoint_targets,
            groups=exp_io.original_image_families(checkpoint_images),
            n_bootstrap=BOOTSTRAP_SAMPLES,
            seed=config.SEED,
        ),
        "visual_pair": metrics.bootstrap_auroc(
            pair_score_values,
            pair_targets,
            groups=exp_io.original_image_families(pair_images),
            n_bootstrap=BOOTSTRAP_SAMPLES,
            seed=config.SEED,
        ),
    }


def main() -> None:
    if config.VISION_ENCODER != "siglip2_512":
        raise ValueError("Joint experiment requires the canonical 512px SigLIP 2 profile")
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    device = evaluate._device()
    split = splits.load_split()
    train_records = ds.build_pair_records(set(split["train"]))
    val_records = ds.build_pair_records(set(split["val"]))
    test_records = ds.build_pair_records(set(split["test"]))
    if not train_records or not val_records or not test_records:
        raise RuntimeError("Joint experiment requires non-empty train, val, and test records")

    input_paths = {
        "gold": ds.exp_io.rio.GOLD_PATH,
        "rules": ds.exp_io.rio.RULES_PATH,
        "split": config.SPLIT_PATH,
        "feature_manifest": config.FEATURE_MANIFEST,
    }
    hashes_before = {name: sha256(path) for name, path in input_paths.items()}

    control_train_loader, _, control_train_examples = train.make_loader(
        set(split["train"]), shuffle=True
    )
    control_val_loader, _, _ = train.make_loader(set(split["val"]), shuffle=False)
    img_dim, txt_dim = train.infer_dims(control_train_loader)
    control = RCESHead(img_dim=img_dim, txt_dim=txt_dim, use_is_visual=False).to(device)
    control_optimizer = torch.optim.AdamW(
        control.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    print(f"control device={device} parameters={parameter_count(control):,}")
    control_state, control_history, control_best_epoch = train_with_best_val(
        control,
        control_optimizer,
        lambda optimizer: train.run_epoch(control, control_train_loader, device, optimizer),
        lambda optimizer: train.run_epoch(control, control_val_loader, device, optimizer),
        lambda: evaluate.checkpoint_probs(control, val_records, device),
        val_records,
    )
    control.load_state_dict(control_state)
    control.eval()

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    joint_train_loader, joint_train_pairs = make_joint_loader(train_records, shuffle=True)
    joint_val_loader, _ = make_joint_loader(val_records, shuffle=False)
    sample = next(iter(joint_train_loader))
    joint = JointRCESHead(
        img_dim=int(sample["patches"].shape[-1]),
        txt_dim=int(sample["checkpoint_text"].shape[-1]),
        hidden=JOINT_HIDDEN,
        dropout=config.DROPOUT,
        relation_layers=1,
        relation_heads=4,
    ).to(device)
    joint_optimizer = torch.optim.AdamW(
        joint.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    print(f"joint device={device} parameters={parameter_count(joint):,}")
    joint_state, joint_history, joint_best_epoch = train_with_best_val(
        joint,
        joint_optimizer,
        lambda optimizer: run_joint_epoch(joint, joint_train_loader, device, optimizer),
        lambda optimizer: run_joint_epoch(joint, joint_val_loader, device, optimizer),
        lambda: joint_checkpoint_probs(joint, val_records, device),
        val_records,
    )
    joint.load_state_dict(joint_state)
    joint.eval()

    control_probs = evaluate.checkpoint_probs(control, test_records, device)
    joint_probs = joint_checkpoint_probs(joint, test_records, device)
    control_checkpoint, checkpoint_targets, checkpoint_images = checkpoint_arrays(
        control_probs, test_records
    )
    joint_checkpoint, joint_targets, joint_images = checkpoint_arrays(joint_probs, test_records)
    if (checkpoint_targets, checkpoint_images) != (joint_targets, joint_images):
        raise RuntimeError("Control and joint checkpoint evaluation rows are not aligned")

    visual_records = [record for record in test_records if not record.external_evidence_required]
    control_pair_map = evaluate.visual_pair_scores(control, visual_records, device)
    joint_pair_map = pair_scores(joint_probs, visual_records)
    pair_targets = [record.sufficiency_gold for record in visual_records]
    pair_images = [record.image_id for record in visual_records]
    control_pairs = [control_pair_map[record.pair_id] for record in visual_records]
    joint_pairs = [joint_pair_map[record.pair_id] for record in visual_records]

    hashes_after = {name: sha256(path) for name, path in input_paths.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("Experiment inputs changed while training")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": control_state,
            "img_dim": img_dim,
            "txt_dim": txt_dim,
            "use_is_visual": False,
            "input_mode": "full",
            "best_epoch": control_best_epoch,
        },
        CONTROL_CKPT,
    )
    torch.save(
        {
            "state_dict": joint_state,
            "img_dim": int(sample["patches"].shape[-1]),
            "txt_dim": int(sample["checkpoint_text"].shape[-1]),
            "hidden": JOINT_HIDDEN,
            "relation_layers": 1,
            "relation_heads": 4,
            "best_epoch": joint_best_epoch,
        },
        JOINT_CKPT,
    )
    report: dict[str, Any] = {
        "experiment": "joint_required_checkpoint_context_with_role_embedding",
        "device": str(device),
        "vision_profile": config.VISION_ENCODER,
        "image_size": config.VISION_IMAGE_SIZE,
        "n_patch": config.N_PATCH,
        "input_sha256": hashes_after,
        "selection_metric": "validation checkpoint AUROC",
        "epochs": config.EPOCHS,
        "seed": config.SEED,
        "counts": {
            "train_pairs": len(train_records),
            "train_control_checkpoint_examples": len(control_train_examples),
            "train_joint_pair_examples": joint_train_pairs,
            "test_pairs": len(test_records),
            "test_checkpoint_examples": len(checkpoint_targets),
            "test_visual_pairs": len(visual_records),
            "test_checkpoint_positives": int(sum(checkpoint_targets)),
            "test_checkpoint_negatives": int(len(checkpoint_targets) - sum(checkpoint_targets)),
            "test_visual_pair_positives": int(sum(pair_targets)),
            "test_visual_pair_negatives": int(len(pair_targets) - sum(pair_targets)),
        },
        "independent": {
            "parameters": parameter_count(control),
            "best_epoch": control_best_epoch,
            "history": control_history,
            **arm_report(
                control_checkpoint,
                checkpoint_targets,
                checkpoint_images,
                control_pairs,
                pair_targets,
                pair_images,
            ),
        },
        "joint": {
            "parameters": parameter_count(joint),
            "best_epoch": joint_best_epoch,
            "history": joint_history,
            **arm_report(
                joint_checkpoint,
                checkpoint_targets,
                checkpoint_images,
                joint_pairs,
                pair_targets,
                pair_images,
            ),
        },
        "paired_delta": {
            "checkpoint": paired_cluster_delta(
                control_checkpoint, joint_checkpoint, checkpoint_targets, checkpoint_images
            ),
            "visual_pair": paired_cluster_delta(
                control_pairs, joint_pairs, pair_targets, pair_images
            ),
        },
        "interpretation_limit": (
            "Pilot comparison only; the test set contains few negative examples and the "
            "joint arm changes architecture while approximately matching parameter count. "
            "The joint arm also receives the rule's known critical/context role metadata."
        ),
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint_delta = report["paired_delta"]["checkpoint"]
    pair_delta = report["paired_delta"]["visual_pair"]
    lines = [
        "# Joint Checkpoint Experiment",
        "",
        f"Input gold SHA-256: `{hashes_after['gold']}`.",
        f"Vision profile: `{config.VISION_ENCODER}` ({config.VISION_IMAGE_SIZE}px, {config.N_PATCH} patches).",
        "",
        "| Metric | Independent | Joint | Joint - independent paired 95% CI |",
        "|---|---:|---:|---:|",
        f"| Parameters | {report['independent']['parameters']:,} | {report['joint']['parameters']:,} | |",
        f"| Best validation epoch | {control_best_epoch} | {joint_best_epoch} | |",
        f"| Checkpoint AUROC | {report['independent']['checkpoint']['auroc']:.4f} | {report['joint']['checkpoint']['auroc']:.4f} | {checkpoint_delta['delta_joint_minus_independent']:+.4f} [{checkpoint_delta['ci_95_low']:+.4f}, {checkpoint_delta['ci_95_high']:+.4f}] |",
        f"| Visual-pair AUROC | {report['independent']['visual_pair']['auroc']:.4f} | {report['joint']['visual_pair']['auroc']:.4f} | {pair_delta['delta_joint_minus_independent']:+.4f} [{pair_delta['ci_95_low']:+.4f}, {pair_delta['ci_95_high']:+.4f}] |",
        "",
        report["interpretation_limit"],
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT_MD.relative_to(config.RCES_ROOT)}")


if __name__ == "__main__":
    main()
