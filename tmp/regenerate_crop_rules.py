"""Re-bind each crop-augmentation image to a rule by FRESH retrieval (SigLIP-224
first, 9B VLM for the rest), then regenerate rces_crop_augmentation.jsonl.

Why: the first crop file inherited each crop's rules from its mother's pilot
pairs. A random crop changes what is visible, so the inherited rule's subject is
often no longer in the crop -- retrieval precision was poor (84/200 piled onto
R-OPN-002). This re-runs retrieval on the CROP image itself, over the 8
`visual_screening_only` benchmark rules (the deterministic crop-line pool, 2 per
category), with the user's hybrid routing:

    SigLIP-224 ranks the 8 rules for the crop.
      - top-1 is an Opening rule  -> keep SigLIP (openings have a distinctive,
        texture-rich signature SigLIP locks onto reliably and cheaply);
      - top-1 is NOT an opening    -> hand the crop to the 9B VLM (Qwen3.5-9B)
        hierarchical retriever (Stage-A scene + Stage-B rule), whose semantic
        recognition is stronger on people / hoarding / edge scenes.
    This reflects the rule-retrieval comparison described in experiments/README.md.

Each crop binds its TOP-2 retrieved rules (rank0=target, rank1=related), capped
at TARGET_PAIR_RECORDS total. Judgment fields stay placeholders -- a human still
screens them with annotate_web 3, so "no model writes a judgment field" holds.
The crops themselves are NOT regenerated; only the rule bindings change. Any
already-`judged` (crop, rule) whose binding survives keeps its human labels.

SigLIP is forced to 224 here regardless of RCES_VISION_ENCODER. The VLM model is
selected by RCES_LOCAL_QWEN_MODEL (must be Qwen3.5-9B for this run).

    UV_CACHE_DIR=/tmp/uv-cache RCES_LOCAL_QWEN_MODEL=Qwen3.5-9B \
        uv run python tmp/regenerate_crop_rules.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

_RCES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RCES / "src"))
sys.path.insert(0, str(_RCES / "experiments"))
sys.path.insert(0, str(_RCES / "experiments" / "retrieval"))

import config  # noqa: E402
import rces_io as rio  # noqa: E402
import hierarchy as H  # noqa: E402
import vlm_retrieve as VR  # noqa: E402  (reuse retrieve_image: Stage-A + Stage-B)
from local_vlm.runtime import DryRunOptionRuntime, Qwen35OptionRuntime  # noqa: E402

# ---- Run controls (edit these) ----------------------------------------------
DRY_RUN = False                  # True: report routing only, write nothing.
SKIP_VLM = False                 # True: SigLIP-only (debug); non-openings fall back to SigLIP.
SIGLIP_IMAGE_SIZE = 224          # user requirement: SigLIP-224 (native), not 512.
TOP_K_PER_CROP = 2               # rules bound per crop (rank0=target, rank1=related).
TARGET_PAIR_RECORDS = 200        # global cap on emitted (crop, rule) records.
RANDOM_SEED = 20260615           # crop visiting order (diverse coverage under the cap).
OPENING_CATEGORY = "Opening Hazard"
SIGLIP_IMAGE_BATCH = 16
# Judgment fields that travel with a surviving already-`judged` (crop, rule).
CARRIED_JUDGMENT = (
    "decidability", "compliance_label", "image_quality", "evidence_sufficiency_label",
    "observed_checkpoints", "missing_checkpoints", "critical_missing_checkpoints",
    "visual_evidence", "missing_information", "annotation_status",
    "annotator_id", "annotation_seconds", "review_status", "annotation_notes",
)
# -----------------------------------------------------------------------------

PLACEHOLDER_JUDGMENT: dict[str, Any] = {
    "decidability": rio.PLACEHOLDER,
    "compliance_label": rio.PLACEHOLDER,
    "image_quality": rio.PLACEHOLDER,
    "evidence_sufficiency_label": rio.PLACEHOLDER,
    "observed_checkpoints": [],
    "missing_checkpoints": [],
    "critical_missing_checkpoints": [],
    "visual_evidence": [],
    "missing_information": [],
    "annotation_status": "pending",
}


def visual_pool(rules: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The 8 visual_screening_only benchmark-subset rules (2 per category)."""
    subset = set(
        json.loads(config.RETRIEVAL_SUBSET_PATH.read_text(encoding="utf-8"))["selected_rule_ids"]
    )
    pool = [
        rules[rid] for rid in subset
        if rules[rid]["decision_scope"] == "visual_screening_only"
    ]
    return sorted(pool, key=lambda r: r["rule_id"])


def crop_provenance(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """image_id -> the crop's stable per-image provenance (first record wins)."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        iid = r["image_id"]
        if iid not in out:
            out[iid] = {
                "base_image_id": r["base_image_id"],
                "crop_bbox": r["crop_bbox"],
                "augmentation": r.get("augmentation", "random_crop"),
                "image_weak_label": r["image_weak_label"],
            }
    return out


def judged_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """(image_id, rule_id) -> record, for rows a human already judged."""
    return {
        (r["image_id"], r["rule_id"]): r
        for r in rows
        if r.get("annotation_status") in ("judged", "described", "reviewed")
    }


# ---- SigLIP-224 -------------------------------------------------------------
def _pooled(out: Any) -> Any:
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    return out


def siglip_scores(
    crop_ids: list[str], pool: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """{image_id: {rule_id: cosine}} from SigLIP-224 over the crop images."""
    import torch
    from transformers import AutoModel, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(config.SIGLIP2_DIR, local_files_only=True)
    processor.image_processor.size = {"height": SIGLIP_IMAGE_SIZE, "width": SIGLIP_IMAGE_SIZE}
    model = AutoModel.from_pretrained(config.SIGLIP2_DIR, local_files_only=True).to(device).eval()
    native = int(model.vision_model.config.image_size)
    img_kwargs = {"interpolate_pos_encoding": SIGLIP_IMAGE_SIZE != native}

    def norm(t):
        return t / t.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

    texts = [H.rule_subject_text(r) for r in pool]
    t_inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        tfeat = norm(_pooled(model.get_text_features(**t_inputs))).cpu()

    feats = []
    for start in range(0, len(crop_ids), SIGLIP_IMAGE_BATCH):
        chunk = crop_ids[start : start + SIGLIP_IMAGE_BATCH]
        images = []
        for iid in chunk:
            path = rio.CROP_IMAGE_DIR / iid
            if not path.is_file():
                raise FileNotFoundError(f"Missing crop image: {path}")
            with Image.open(path) as src:
                images.append(ImageOps.exif_transpose(src).convert("RGB"))
        i_inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            feats.append(norm(_pooled(model.get_image_features(**i_inputs, **img_kwargs))).cpu())
    ifeat = torch.cat(feats, dim=0)

    matrix = ifeat @ tfeat.T
    rule_ids = [r["rule_id"] for r in pool]
    return {
        iid: {rule_ids[j]: float(matrix[i, j]) for j in range(len(rule_ids))}
        for i, iid in enumerate(crop_ids)
    }


# ---- Routing ----------------------------------------------------------------
def route_crop(
    image_id: str,
    sig: dict[str, float],
    rules: dict[str, dict[str, Any]],
    pool: list[dict[str, Any]],
    runtime,
    prompts,
    options,
) -> tuple[str, list[tuple[str, float]]]:
    """Return (method, [(rule_id, score) ...]) ranked, length<=TOP_K_PER_CROP.

    SigLIP top-1 opening -> keep SigLIP top-K; else 9B VLM hierarchical top-K
    (falling back to SigLIP if the VLM returns nothing)."""
    ranked = sorted(sig, key=lambda rid: (-sig[rid], rid))
    sig_topk = [(rid, sig[rid]) for rid in ranked[:TOP_K_PER_CROP]]
    top1_is_opening = rules[ranked[0]]["major_category"] == OPENING_CATEGORY

    if top1_is_opening or SKIP_VLM or runtime is None:
        return "siglip2", sig_topk

    image_path = rio.CROP_IMAGE_DIR / image_id
    body = VR.retrieve_image(runtime, image_path, pool, options, prompts)
    relevance = body["rule_relevance"]
    chosen = [(rid, float(relevance[rid])) for rid in body["applicable_rules"][:TOP_K_PER_CROP]]
    if not chosen:
        return "siglip2", sig_topk
    return "vlm_hierarchical", chosen


def make_record(
    pair_idx: int,
    image_id: str,
    rule: dict[str, Any],
    role: str,
    method: str,
    score: float,
    prov: dict[str, Any],
    target_category: str,
    judged: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "pair_id": f"rces_crop_{pair_idx:06d}",
        "image_id": image_id,
        "image_split": "train",  # crop augmentation is train-only, never gold/test.
        "rule_id": rule["rule_id"],
        "rule_name": rule["rule_name"],
        "major_category": rule["major_category"],
        "subcategory": rule["subcategory"],
        "decision_scope": rule["decision_scope"],
        "rule_role": role,
        "image_weak_label": prov["image_weak_label"],
        "weak_candidate_category": target_category,
        **PLACEHOLDER_JUDGMENT,
        "source": rio.SOURCE_HUMAN,
        "double_annotation": False,
        "base_image_id": prov["base_image_id"],
        "crop_bbox": prov["crop_bbox"],
        "augmentation": prov["augmentation"],
        "retrieval_method": method,
        "retrieval_score": round(score, 6),
    }
    prior = judged.get((image_id, rule["rule_id"]))
    if prior is not None:
        for field in CARRIED_JUDGMENT:
            if field in prior:
                rec[field] = prior[field]
    return rec


def main() -> None:
    rules = rio.load_rules()
    pool = visual_pool(rules)
    options = H.category_options(pool)

    existing = rio.load_jsonl(rio.CROP_AUG_PATH)
    # This script only re-binds TRAIN crops. val/test crops (image_split != "train",
    # written by generate_crop_augmentation_eval.py into their own id blocks) are
    # carried through verbatim so a train re-run never clobbers the eval splits.
    preserved = [r for r in existing if r.get("image_split") != "train"]
    train_rows = [r for r in existing if r.get("image_split") == "train"]
    prov = crop_provenance(train_rows)
    judged = judged_index(train_rows)
    crop_ids = sorted(prov)  # distinct TRAIN crop images

    print("=" * 64)
    print("R-CES crop rule RE-RETRIEVAL (SigLIP-224 -> 9B VLM)" + ("  [DRY RUN]" if DRY_RUN else ""))
    print("=" * 64)
    print(f"crop images          : {len(crop_ids)}")
    print(f"rule pool (vso)      : {len(pool)}  {[r['rule_id'] for r in pool]}")
    print(f"already judged pairs : {len(judged)}")
    print(f"top-k per crop       : {TOP_K_PER_CROP}   cap : {TARGET_PAIR_RECORDS}")
    print("-" * 64)

    sig = siglip_scores(crop_ids, pool)

    runtime = None
    if not SKIP_VLM and not DRY_RUN:
        if config.LOCAL_QWEN_MODEL_NAME != "Qwen3.5-9B":
            print(f"WARN: RCES_LOCAL_QWEN_MODEL={config.LOCAL_QWEN_MODEL_NAME!r}, expected Qwen3.5-9B")
        runtime = Qwen35OptionRuntime(
            config.LOCAL_QWEN35_DIR, dtype=config.LOCAL_QWEN_DTYPE, device_map=config.LOCAL_QWEN_DEVICE_MAP
        )
    elif not SKIP_VLM and DRY_RUN:
        runtime = DryRunOptionRuntime()
    prompts = (
        H.load_prompt(config.RETRIEVAL_CATEGORY_PROMPT_PATH),
        H.load_prompt(config.RETRIEVAL_RULES_PROMPT_PATH),
    )

    # Route every crop; record its ordered top-k bindings.
    routed: dict[str, tuple[str, list[tuple[str, float]]]] = {}
    n_vlm = n_sig = 0
    for i, iid in enumerate(crop_ids, 1):
        method, chosen = route_crop(iid, sig[iid], rules, pool, runtime, prompts, options)
        routed[iid] = (method, chosen)
        if method == "vlm_hierarchical":
            n_vlm += 1
        else:
            n_sig += 1
        print(f"[{i}/{len(crop_ids)}] {iid} -> {method:16s} {[rid for rid, _ in chosen]}")

    # Emit records: rank0 for every crop first (target), then rank1 (related)
    # in crop order, until the global cap. Keeps coverage broad and balanced.
    order = list(crop_ids)
    random.Random(RANDOM_SEED).shuffle(order)
    records: list[dict[str, Any]] = []
    for rank in range(TOP_K_PER_CROP):
        role = "target" if rank == 0 else "related"
        for iid in order:
            if len(records) >= TARGET_PAIR_RECORDS:
                break
            method, chosen = routed[iid]
            if rank >= len(chosen):
                continue
            rule_id, score = chosen[rank]
            target_category = rules[chosen[0][0]]["major_category"]
            records.append(
                make_record(
                    len(records) + 1, iid, rules[rule_id], role, method, score,
                    prov[iid], target_category, judged,
                )
            )
        if len(records) >= TARGET_PAIR_RECORDS:
            break

    survived = sum(
        1 for r in records
        if (r["image_id"], r["rule_id"]) in judged and r["annotation_status"] != "pending"
    )
    import collections
    rule_dist = collections.Counter(r["rule_id"] for r in records)
    method_dist = collections.Counter(r["retrieval_method"] for r in records)

    print("-" * 64)
    print(f"routing              : siglip {n_sig} | vlm {n_vlm}")
    print(f"records              : {len(records)}   (target {TARGET_PAIR_RECORDS})")
    print(f"per-record method    : {dict(method_dist)}")
    print(f"judged carried over  : {survived} / {len(judged)}")
    print(f"rule distribution    :")
    for rid, n in rule_dist.most_common():
        print(f"    {n:4d}  {rid}")

    if DRY_RUN:
        print("\n[dry-run] nothing written. sample record:")
        print(json.dumps(records[0], ensure_ascii=False))
        return

    rio.write_jsonl(rio.CROP_AUG_PATH, records + preserved)
    print(f"\nwrote -> {rio.CROP_AUG_PATH.relative_to(rio.PROJECT_ROOT)} "
          f"({len(records)} train + {len(preserved)} preserved val/test = {len(records) + len(preserved)} records)")
    print("Validate: uv run python tmp/validate_rces_pairs.py data/pairs/rces_crop_augmentation.jsonl")
    print("Annotate: uv run python Script/annotate_web.py 3")


if __name__ == "__main__":
    main()
