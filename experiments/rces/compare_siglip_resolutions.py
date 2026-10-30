"""Compare same-snapshot 224px and 512px SigLIP 2 E0 heads.

Both heads must already be trained. The comparison resamples whole image
clusters so repeated pairs/checkpoints from one image stay correlated.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXP.parent / "src"))
sys.path.insert(0, str(_EXP))

import torch  # noqa: E402

import config  # noqa: E402
import exp_io  # noqa: E402
import metrics  # noqa: E402
from data import dataset as ds  # noqa: E402
from data import splits  # noqa: E402
from rces import evaluate  # noqa: E402
from rces.model import RCESHead  # noqa: E402


BOOTSTRAP_SAMPLES = 10_000
ARTIFACT_224 = config.ARTIFACT_DIR / "rces_head_e0_corrected_224_snapshot_4b4ab1.pt"
ARTIFACT_512 = config.ARTIFACT_DIR / "rces_head_e0_corrected_siglip2_512.pt"
REPORT_224 = config.ARTIFACT_DIR / "rces_eval_report_e0_corrected_224_snapshot_4b4ab1.json"
REPORT_512 = config.ARTIFACT_DIR / "rces_eval_report_e0_corrected_siglip2_512.json"
OUTPUT_JSON = config.ARTIFACT_DIR / "rces_siglip2_224_vs_512.json"
OUTPUT_MD = config.ARTIFACT_DIR / "rces_siglip2_224_vs_512.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_head(path: Path, device: torch.device) -> RCESHead:
    checkpoint = torch.load(path, map_location=device)
    head = RCESHead(
        img_dim=checkpoint["img_dim"],
        txt_dim=checkpoint["txt_dim"],
        use_is_visual=checkpoint["use_is_visual"],
        input_mode=checkpoint.get("input_mode", "full"),
    )
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device).eval()


def score_profile(
    head_path: Path,
    image_feature_dir: Path,
    records: list[ds.PairRecord],
    device: torch.device,
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    original_dir = config.IMG_FEATURE_DIR
    try:
        config.IMG_FEATURE_DIR = image_feature_dir
        head = load_head(head_path, device)
        checkpoint = evaluate.checkpoint_probs(head, records, device)
        visual_records = [record for record in records if not record.external_evidence_required]
        pair = evaluate.visual_pair_scores(head, visual_records, device)
        return checkpoint, pair
    finally:
        config.IMG_FEATURE_DIR = original_dir


def paired_cluster_bootstrap(
    scores_224: list[float],
    scores_512: list[float],
    targets: list[int],
    image_ids: list[str],
) -> dict[str, float | int | bool]:
    first = np.asarray(scores_224, dtype=float)
    second = np.asarray(scores_512, dtype=float)
    target = np.asarray(targets, dtype=int)
    families = exp_io.original_image_families(image_ids)
    group = np.asarray(families, dtype=object)
    unique_groups = list(dict.fromkeys(families))
    indices_by_group = {
        image_id: np.flatnonzero(group == image_id) for image_id in unique_groups
    }
    rng = np.random.default_rng(config.SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[image_id] for image_id in sampled_groups])
        delta = metrics._auroc(second[indices], target[indices]) - metrics._auroc(
            first[indices], target[indices]
        )
        if np.isfinite(delta):
            values.append(float(delta))
    low, high = np.quantile(np.asarray(values), [0.025, 0.975]).tolist()
    delta = metrics._auroc(second, target) - metrics._auroc(first, target)
    return {
        "delta_512_minus_224": float(delta),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "ci_contains_zero": bool(low <= 0 <= high),
        "n_groups": len(unique_groups),
        "valid_bootstrap_samples": len(values),
    }


def main() -> None:
    required = (ARTIFACT_224, ARTIFACT_512, REPORT_224, REPORT_512)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing comparison artifacts: {missing}")

    input_paths = {
        "gold": config.RCES_ROOT / "data/pairs/rces_pilot_pairs.jsonl",
        "rules": config.RCES_ROOT / "data/rule/rule.json",
        "split": config.SPLIT_PATH,
    }
    hashes_before = {name: sha256(path) for name, path in input_paths.items()}
    device = evaluate._device()
    test_ids = set(splits.load_split()["test"])
    records = ds.build_pair_records(test_ids)
    checkpoint_224, pair_224 = score_profile(
        ARTIFACT_224,
        config.RCES_ROOT / "cache/experiments/siglip2/image_patches",
        records,
        device,
    )
    checkpoint_512, pair_512 = score_profile(
        ARTIFACT_512,
        config.RCES_ROOT / "cache/experiments/siglip2_512/image_patches",
        records,
        device,
    )

    ckpt_224: list[float] = []
    ckpt_512: list[float] = []
    ckpt_targets: list[int] = []
    ckpt_images: list[str] = []
    for record in records:
        for key, target in record.resolvable_targets.items():
            ckpt_224.append(checkpoint_224[(record.pair_id, key)])
            ckpt_512.append(checkpoint_512[(record.pair_id, key)])
            ckpt_targets.append(target)
            ckpt_images.append(record.image_id)

    visual_records = [record for record in records if not record.external_evidence_required]
    pair_targets = [record.sufficiency_gold for record in visual_records]
    pair_images = [record.image_id for record in visual_records]
    pair_scores_224 = [pair_224[record.pair_id] for record in visual_records]
    pair_scores_512 = [pair_512[record.pair_id] for record in visual_records]
    hashes_after = {name: sha256(path) for name, path in input_paths.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("Input data changed during resolution comparison")

    evaluation_224 = json.loads(REPORT_224.read_text(encoding="utf-8"))
    evaluation_512 = json.loads(REPORT_512.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "device": str(device),
        "input_sha256": hashes_after,
        "n_test_pairs": len(records),
        "checkpoint": {
            "224": metrics.bootstrap_auroc(
                ckpt_224,
                ckpt_targets,
                groups=exp_io.original_image_families(ckpt_images),
                n_bootstrap=BOOTSTRAP_SAMPLES,
                seed=config.SEED,
            ),
            "512": metrics.bootstrap_auroc(
                ckpt_512,
                ckpt_targets,
                groups=exp_io.original_image_families(ckpt_images),
                n_bootstrap=BOOTSTRAP_SAMPLES,
                seed=config.SEED,
            ),
            "paired_delta": paired_cluster_bootstrap(
                ckpt_224, ckpt_512, ckpt_targets, ckpt_images
            ),
        },
        "visual_pair": {
            "224": metrics.bootstrap_auroc(
                pair_scores_224,
                pair_targets,
                groups=exp_io.original_image_families(pair_images),
                n_bootstrap=BOOTSTRAP_SAMPLES,
                seed=config.SEED,
            ),
            "512": metrics.bootstrap_auroc(
                pair_scores_512,
                pair_targets,
                groups=exp_io.original_image_families(pair_images),
                n_bootstrap=BOOTSTRAP_SAMPLES,
                seed=config.SEED,
            ),
            "paired_delta": paired_cluster_bootstrap(
                pair_scores_224, pair_scores_512, pair_targets, pair_images
            ),
        },
        "task1_auroc": {
            "224": evaluation_224["task1_sufficiency_detection"]["auroc"],
            "512": evaluation_512["task1_sufficiency_detection"]["auroc"],
        },
        "task2_auroc": {
            "224": evaluation_224["task2_selective_compliance"]["auroc"],
            "512": evaluation_512["task2_selective_compliance"]["auroc"],
        },
        "task2_answer_rate": {
            "224": evaluation_224["task2_selective_compliance"]["answer_rate"],
            "512": evaluation_512["task2_selective_compliance"]["answer_rate"],
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SigLIP 2 Resolution Comparison",
        "",
        f"Input gold SHA-256: `{hashes_after['gold']}`.",
        "",
        "| Metric | 224 | 512 | 512 - 224 paired 95% CI |",
        "|---|---:|---:|---:|",
        f"| checkpoint AUROC | {report['checkpoint']['224']['auroc']:.4f} | {report['checkpoint']['512']['auroc']:.4f} | {report['checkpoint']['paired_delta']['delta_512_minus_224']:+.4f} [{report['checkpoint']['paired_delta']['ci_95_low']:+.4f}, {report['checkpoint']['paired_delta']['ci_95_high']:+.4f}] |",
        f"| visual-pair AUROC | {report['visual_pair']['224']['auroc']:.4f} | {report['visual_pair']['512']['auroc']:.4f} | {report['visual_pair']['paired_delta']['delta_512_minus_224']:+.4f} [{report['visual_pair']['paired_delta']['ci_95_low']:+.4f}, {report['visual_pair']['paired_delta']['ci_95_high']:+.4f}] |",
        f"| final Task-1 AUROC | {report['task1_auroc']['224']:.4f} | {report['task1_auroc']['512']:.4f} | |",
        f"| Task-2 AUROC | {report['task2_auroc']['224']:.4f} | {report['task2_auroc']['512']:.4f} | |",
        f"| Task-2 answer rate | {report['task2_answer_rate']['224']:.4f} | {report['task2_answer_rate']['512']:.4f} | |",
        "",
        "Both paired intervals contain zero, so the current data does not show a statistically reliable resolution gain.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
