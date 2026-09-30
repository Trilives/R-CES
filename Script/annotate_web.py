"""Stage 1 - LAN web annotation UI (the only human-annotation frontend).

A zero-dependency web front end for human annotation: serves the image + rule +
checkpoints with big tap-to-select button groups (no dropdowns), so any device
on the LAN (phone / tablet / laptop) can annotate. It shows NO model text, so
the human's decidability / compliance / checkpoint calls cannot be anchored by
GPT. Judgment is locked per pair the moment it is saved (resumable).

This file owns all Stage-1 logic now: first/second pass routing, double-
annotation seeding/isolation, the ~20% flag, pending ordering, critical-missing
derivation, and the cross-field invariants. (There used to be a terminal twin,
annotate_human.py, but a CLI annotator can't actually see the image, so it was
removed and its logic folded in here.)

A "Back" control lets the annotator step back to pairs they already saved THIS
run, prefilled with the saved choices, to fix a mistake; saving overwrites that
pair and steps forward again.

Edit the constants below, or pass the annotator number on the command line:

    uv run python Script/annotate_web.py
    uv run python Script/annotate_web.py 2

Open http://<this-machine-LAN-ip>:<PORT>/ on any device on the same network
(the startup banner prints the URLs). One annotator at a time per server.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import mimetypes
import socket
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
import rces_io as rio  # noqa: E402

# Annotator-facing quick guide (the confusing-boundary cheat sheet), served raw
# to the "Guide" panel and rendered client-side. Read fresh per request so
# edits show up without restarting the server.
GUIDE_PATH = _HERE / "annotation_quick_guide.md"

# ---- Run controls (edit these; CLI annotator number can override pass) --------
ANNOTATION_PASS = "first"   # "first"  -> primary annotator -> gold file, all pending.
                            # "second" -> second annotator -> isolated double file,
                            #             ONLY the ~20% the first pass flagged; never
                            #             sees the first annotator's labels.
HOST = "0.0.0.0"            # bind all interfaces so the LAN can reach it
PORT = 8000
PORT_FALLBACK_COUNT = 20     # if PORT is busy, try PORT+1 ... PORT+N
ADVERTISED_LAN_IP = "127.0.0.1"
LIMIT = 0                   # max pairs to annotate this run (0 = all pending)
SAMPLE_MODE = "order"       # "order" | "stratified" (touchstone preview; pair w/ LIMIT)
# ------------------------------------------------------------------------------

# annotator_id is derived from the pass, not typed by hand.
PASS_SETTINGS = {
    "first":  {"annotator_id": "annotator_1", "out_path": rio.GOLD_PATH},
    "second": {"annotator_id": "annotator_2", "out_path": rio.DOUBLE_PATH},
    # Crop-augmentation screening: same full UI over the pending crop candidates
    # in rces_crop_augmentation.jsonl. Isolated, train-only, never double-annotated.
    "augment": {"annotator_id": "augment_annotator", "out_path": rio.CROP_AUG_PATH},
    # Test-expansion screening: same full UI over the pending enhance pairs in
    # rces_enhance_pairs.jsonl (image_split=test_expansion). Isolated, never
    # double-annotated.
    "enhance": {"annotator_id": "enhance_annotator", "out_path": rio.ENHANCE_PATH},
}

DOUBLE_ANNOTATION_RATIO = 0.20
ANNOTATOR_PASS_BY_NUMBER = {
    1: "first",
    2: "second",
    3: "augment",
    4: "enhance",
}

# Immutable candidate metadata carried unchanged into a second-pass working row.
METADATA_FIELDS = (
    "pair_id", "image_id", "image_split", "rule_id", "rule_name",
    "major_category", "subcategory", "decision_scope", "rule_role",
    "image_weak_label", "weak_candidate_category",
)

# evidence_sufficiency_label is NOT here: it is DERIVED from decidability
# (codebook §7), shown read-only in the UI, never clicked or POSTed as a choice.
FIELD_OPTIONS = {
    "decidability": list(rio.DECIDABILITY),
    "compliance_label": list(rio.COMPLIANCE),
    "image_quality": list(rio.IMAGE_QUALITY),
}
CHECKPOINT_OPTIONS = ("observed", "missing", "not_applicable")


# ---- Stage-1 logic (pass routing / seeding / ordering / double annotation) ---

def assign_double_annotation(rows: list[dict[str, Any]], ratio: float = DOUBLE_ANNOTATION_RATIO) -> int:
    """Deterministically flag ~ratio of JUDGED rows for double annotation,
    prioritising partially_decidable cases and requires_external_evidence rules
    (the hard cases the second annotator should cover). Idempotent."""
    judged = [r for r in rows if r.get("annotation_status") in ("judged", "described", "reviewed")]
    target = math.ceil(ratio * len(judged)) if judged else 0

    def priority(row: dict[str, Any]) -> tuple[int, str]:
        hard = (
            row.get("decidability") == "partially_decidable"
            or row.get("decision_scope") == "requires_external_evidence"
        )
        return (0 if hard else 1, row["pair_id"])

    chosen = {r["pair_id"] for r in sorted(judged, key=priority)[:target]}
    for row in rows:
        if row.get("annotation_status") in ("judged", "described", "reviewed"):
            row["double_annotation"] = row["pair_id"] in chosen
    return len(chosen)


def order_pending(rows: list[dict[str, Any]], pending_idx: list[int], mode: str) -> list[int]:
    """Order in which pending pairs are presented. Both modes keep an image's
    pairs CONSECUTIVE (so you judge all of an image's rules — and the "rule N/M
    for this image" counter advances 1..M — before the image changes). "order"
    keeps file order; "stratified" round-robins whole IMAGES across major_category
    buckets so a short LIMIT preview spreads across categories instead of
    clustering on the first few images."""
    by_image: dict[str, list[int]] = defaultdict(list)
    for i in pending_idx:
        by_image[rows[i]["image_id"]].append(i)
    if mode != "stratified":
        # File order already groups an image's pairs; just flatten image-by-image
        # to stay robust if some of an image's pairs are already judged.
        return [i for idxs in by_image.values() for i in idxs]
    # Bucket each image by its first pending pair's category, then round-robin
    # images across buckets, emitting each image's pairs together.
    buckets: dict[str, deque[str]] = defaultdict(deque)
    for image_id, idxs in by_image.items():
        buckets[rows[idxs[0]].get("major_category", "")].append(image_id)
    queues = [buckets[k] for k in sorted(buckets)]
    ordered: list[int] = []
    while queues:
        for q in queues:
            if q:
                ordered.extend(by_image[q.popleft()])
        queues = [q for q in queues if q]
    return ordered


def seed_second_row(gold_row: dict[str, Any]) -> dict[str, Any]:
    """A clean placeholder row for the second pass: candidate metadata only, with
    NO first-pass judgment copied across (isolation -> no anchoring)."""
    row = {field: gold_row[field] for field in METADATA_FIELDS}
    row.update(
        {
            "decidability": rio.PLACEHOLDER,
            "compliance_label": rio.PLACEHOLDER,
            "image_quality": rio.PLACEHOLDER,
            "evidence_sufficiency_label": rio.PLACEHOLDER,
            "observed_checkpoints": [],
            "missing_checkpoints": [],
            "critical_missing_checkpoints": [],
            "visual_evidence": [],
            "missing_information": [],
            "source": rio.SOURCE_HUMAN,
            "annotation_status": "pending",
            "double_annotation": True,
        }
    )
    return row


def prepare_second_pass() -> list[dict[str, Any]]:
    """Build the second annotator's working rows: the ~20% pairs the first pass
    flagged (double_annotation=true and already judged), seeded fresh unless the
    second file already holds them (resumable)."""
    gold = rio.load_jsonl(rio.GOLD_PATH)
    flagged = [
        r for r in gold
        if r.get("double_annotation")
        and r.get("annotation_status") in ("judged", "described", "reviewed")
    ]
    existing = {r["pair_id"]: r for r in rio.load_jsonl(rio.DOUBLE_PATH)}
    return [existing.get(g["pair_id"]) or seed_second_row(g) for g in flagged]


class AnnotatorState:
    """In-memory annotation session; one active annotator per server."""

    def __init__(self) -> None:
        if ANNOTATION_PASS not in PASS_SETTINGS:
            raise SystemExit(f"ANNOTATION_PASS must be one of {sorted(PASS_SETTINGS)}")
        settings = PASS_SETTINGS[ANNOTATION_PASS]
        self.annotator = settings["annotator_id"]
        self.out_path = settings["out_path"]
        self.is_second = ANNOTATION_PASS == "second"
        self.rules = rio.load_rules()
        self.rows = prepare_second_pass() if self.is_second else rio.load_jsonl(self.out_path)
        self.lock = threading.Lock()
        self.annotated_this_run = 0
        self.skipped: set[str] = set()
        self.served_at: float | None = None
        self.served_pid: str | None = None
        # Back/forward navigation over pairs saved THIS run.
        self.history: list[str] = []       # pair_ids saved this run, in save order
        self.cursor: int | None = None     # index into history while revisiting; None = live frontier
        self._rebuild_order()

    def _rebuild_order(self) -> None:
        pending = [i for i, r in enumerate(self.rows) if r.get("annotation_status", "pending") == "pending"]
        self.order = order_pending(self.rows, pending, SAMPLE_MODE)

    def progress(self) -> dict:
        total = len(self.rows)
        done = sum(1 for r in self.rows if r.get("annotation_status", "pending") != "pending")
        return {
            "total": total, "done": done, "pending": total - done,
            "annotated_this_run": self.annotated_this_run, "limit": LIMIT,
            "pass": ANNOTATION_PASS, "annotator": self.annotator,
        }

    def _index_of(self, pid: str) -> int | None:
        return next((i for i, r in enumerate(self.rows) if r["pair_id"] == pid), None)

    def _current_index(self) -> int | None:
        if LIMIT and self.annotated_this_run >= LIMIT:
            return None
        for i in self.order:
            row = self.rows[i]
            if row.get("annotation_status", "pending") == "pending" and row["pair_id"] not in self.skipped:
                return i
        return None

    def image_quality_for(self, image_id: str, exclude_pid: str | None = None) -> str | None:
        """image_quality is a property of the IMAGE, not the pair: it is annotated
        once (on the first pair of an image) and inherited by the image's remaining
        pairs. Return the quality already recorded for any judged pair of this image
        (other than exclude_pid), or None if no pair of the image has fixed it yet."""
        for row in self.rows:
            if (
                row["image_id"] == image_id
                and row["pair_id"] != exclude_pid
                and row.get("annotation_status", "pending") != "pending"
                and row.get("image_quality") in rio.IMAGE_QUALITY
            ):
                return row["image_quality"]
        return None

    def _propagate_image_quality(self, image_id: str, quality: str, exclude_pid: str) -> None:
        """Keep every already-judged pair of the same image consistent when the
        image's quality is (re)set, so editing it on one pair never leaves the
        image's other pairs carrying a stale value."""
        for row in self.rows:
            if (
                row["image_id"] == image_id
                and row["pair_id"] != exclude_pid
                and row.get("annotation_status", "pending") != "pending"
            ):
                row["image_quality"] = quality

    def _prefill(self, row: dict, required: list[str]) -> dict:
        """Reconstruct the button selections from a saved row, so a revisited pair
        comes back with the annotator's earlier choices already highlighted."""
        observed = set(row.get("observed_checkpoints", []))
        missing = set(row.get("missing_checkpoints", []))
        marks = {k: ("observed" if k in observed else "missing" if k in missing else "not_applicable")
                 for k in required}
        return {
            "decidability": row.get("decidability"),
            "compliance_label": row.get("compliance_label"),
            "image_quality": row.get("image_quality"),
            "checkpoints": marks,
        }

    def payload(self) -> dict:
        with self.lock:
            cursor = self.cursor
            revisiting = cursor is not None
            if cursor is not None:
                i = self._index_of(self.history[cursor])
            else:
                i = self._current_index()
            if i is None:
                return {"done": True, "progress": self.progress()}
            row = self.rows[i]
            rule = self.rules.get(row["rule_id"], {})
            required, critical, questions = rio.checkpoints_for(rule)
            self.served_at = time.monotonic()
            self.served_pid = row["pair_id"]
            # Which rule (of how many) this image is being judged against — the same
            # image recurs across several rules/roles, so this orients the annotator.
            same_image = [j for j, r in enumerate(self.rows) if r["image_id"] == row["image_id"]]
            # image_quality is rated once per image; lock it on later pairs of an
            # image that another pair already rated. Revisits stay editable so a
            # mistake can be fixed (and propagated) from the pair that set it.
            locked_quality = None if revisiting else self.image_quality_for(row["image_id"], row["pair_id"])
            pair = {
                "pair_id": row["pair_id"],
                "image_url": f"/img/{row['image_id']}",
                "image_id": row["image_id"],
                "image_split": row.get("image_split", ""),
                # Weak labels from professional construction annotators: display-only
                # hints, never written to gold. Shown de-emphasized so they assist
                # without anchoring the human's judgment (see docstring).
                "image_weak_label": row.get("image_weak_label", ""),
                "weak_candidate_category": row.get("weak_candidate_category", ""),
                "image_rule_index": same_image.index(i) + 1,
                "image_rule_total": len(same_image),
                "rule_name": row["rule_name"],
                "major_category": row["major_category"],
                "subcategory": row["subcategory"],
                "decision_scope": row["decision_scope"],
                "checkpoints": [
                    {"key": k, "question": questions.get(k, ""), "critical": k in critical}
                    for k in required
                ],
                # Subject-presence gate: for a decidable pair this checkpoint is
                # always observed (subject seen present, or seen absent), so the UI
                # pre-fills it on picking 'decidable' (still editable). None when
                # the gate is ambiguous (no auto-fill then).
                "subject_gate": rio.subject_gate(rule),
                "revisiting": revisiting,
                "can_back": bool(self.history) and self.cursor != 0,
                "can_forward": revisiting,
                "image_quality_locked": locked_quality,
            }
            if revisiting:
                pair["prefill"] = self._prefill(row, required)
            return {
                "done": False,
                "progress": self.progress(),
                "field_options": FIELD_OPTIONS,
                "checkpoint_options": list(CHECKPOINT_OPTIONS),
                "pair": pair,
            }

    def _validate(self, posted: dict, rule: dict) -> tuple[bool, str]:
        for field, options in FIELD_OPTIONS.items():
            if posted.get(field) not in options:
                return False, f"{field} must be one of {options}"
        required, _, _ = rio.checkpoints_for(rule)
        marks = posted.get("checkpoints", {})
        for key in required:
            if marks.get(key) not in CHECKPOINT_OPTIONS:
                return False, f"checkpoint {key} must be one of {list(CHECKPOINT_OPTIONS)}"
        # Codebook §8 cross-field invariants (the UI auto-locks these, but a
        # crafted POST must not be able to write inconsistent gold).
        missing = [k for k in required if marks.get(k) == "missing"]
        errs = rio.consistency_errors(
            posted["decidability"], posted["compliance_label"], rio.critical_missing(missing, rule)
        )
        if errs:
            return False, "; ".join(errs)
        return True, ""

    def _build_row(self, row: dict, rule: dict, posted: dict, seconds: float) -> dict:
        required, _, _ = rio.checkpoints_for(rule)
        marks = posted["checkpoints"]
        observed = [k for k in required if marks.get(k) == "observed"]
        missing = [k for k in required if marks.get(k) == "missing"]
        updated = dict(row)
        updated.update({
            "decidability": posted["decidability"],
            "compliance_label": posted["compliance_label"],
            "image_quality": posted["image_quality"],
            "evidence_sufficiency_label": rio.derive_evidence_sufficiency(posted["decidability"]),
            "observed_checkpoints": observed,
            "missing_checkpoints": missing,
            "critical_missing_checkpoints": rio.critical_missing(missing, rule),
            "visual_evidence": [],
            "missing_information": [],
            "source": rio.SOURCE_HUMAN,
            "annotation_status": "judged",
            "annotator_id": self.annotator,
            "annotation_seconds": seconds,
        })
        return updated

    def save(self, posted: dict) -> tuple[bool, str]:
        with self.lock:
            pid = posted.get("pair_id")
            idx = self._index_of(pid)
            if idx is None:
                return False, "unknown pair_id"
            rule = self.rules.get(self.rows[idx]["rule_id"], {})
            # image_quality is per-image: on a fresh pair whose image another pair
            # already rated, that value is authoritative (the field is locked in the
            # UI) -- don't let a stale/crafted POST diverge it. Revisits may change it.
            revisiting = pid in self.history
            known = self.image_quality_for(self.rows[idx]["image_id"], pid)
            if known is not None and not revisiting:
                posted = {**posted, "image_quality": known}
            ok, msg = self._validate(posted, rule)
            if not ok:
                return False, msg
            seconds = round(time.monotonic() - self.served_at, 1) if self.served_at and self.served_pid == pid else 0.0
            self.rows[idx] = self._build_row(self.rows[idx], rule, posted, seconds)
            self._propagate_image_quality(self.rows[idx]["image_id"], self.rows[idx]["image_quality"], pid)
            # Double-annotation flagging is a first-pass-only concern; the second
            # pass and the crop-augment pass never (re)flag rows.
            if ANNOTATION_PASS == "first":
                assign_double_annotation(self.rows)
            rio.write_jsonl(self.out_path, self.rows)  # lock after every pair (resumable)
            self.skipped.discard(pid)
            if pid in self.history:
                # Revisited an earlier pair: overwrite it, don't recount, step forward.
                self._step_forward()
            else:
                self.history.append(pid)
                self.annotated_this_run += 1
            return True, "saved"

    def _step_forward(self) -> None:
        if self.cursor is None:
            return
        if self.cursor < len(self.history) - 1:
            self.cursor += 1
        else:
            self.cursor = None  # back at the live frontier (next pending)

    def back(self) -> None:
        with self.lock:
            if not self.history:
                return
            if self.cursor is None:
                self.cursor = len(self.history) - 1
            elif self.cursor > 0:
                self.cursor -= 1

    def forward(self) -> None:
        with self.lock:
            self._step_forward()

    def skip(self, pid: str) -> None:
        with self.lock:
            if pid:
                self.skipped.add(pid)


STATE: AnnotatorState  # set in main()

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>R-CES annotation</title>
<style>
  :root { --bg:#0f1216; --card:#171c22; --line:#2a313a; --fg:#e6e9ee; --muted:#8a94a2;
          --sel:#2d6cdf; --crit:#e0a34a; --ok:#2f9e57; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.45 system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:20; background:var(--card); border-bottom:1px solid var(--line);
           padding:10px 14px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  header b { color:var(--sel); } .muted { color:var(--muted); }
  .pbar { flex:0 0 100%; height:4px; background:var(--line); border-radius:2px; overflow:hidden; }
  .pfill { height:100%; width:0; background:var(--ok); transition:width .3s ease; }
  main { max-width:1500px; margin:0 auto; padding:14px; }
  /* Two balanced columns: left = image + image-side judgments, right = rule + compliance + checkpoints. */
  .wrap { display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start; }
  @media (max-width:980px){ .wrap{ grid-template-columns:1fr; } }
  img#photo { display:block; width:100%; max-height:62vh; object-fit:contain;
              border:1px solid var(--line); border-radius:8px; background:#000; }
  .imgcol .card { margin-top:12px; margin-bottom:0; }
  .imginfo { font-size:.82rem; padding:8px 12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .imgrule { color:var(--crit); font-weight:600; }
  .imginfo .muted { margin-left:auto; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:12px; }
  h2 { margin:.1em 0 .4em; font-size:1.05rem; }
  .lab { color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; margin:10px 0 6px; }
  .grp { display:flex; gap:8px; flex-wrap:wrap; }
  .btn { flex:1 1 auto; min-width:96px; padding:12px 10px; border:1px solid var(--line); border-radius:8px;
         background:#1d242c; color:var(--fg); font-size:.95rem; cursor:pointer; text-align:center; }
  .btn:hover { border-color:var(--sel); }
  .btn.sel { background:var(--sel); border-color:var(--sel); color:#fff; font-weight:600; }
  .cp { display:flex; flex-direction:column; gap:8px; align-items:stretch; padding:10px 0; border-top:1px solid var(--line); }
  .cp:first-child{ border-top:none; }
  .cp .q .k { font-weight:600; } .cp .grp { flex:0 0 auto; }
  .crit { color:var(--crit); } .star { color:var(--crit); font-weight:700; }
  .cp .btn { flex:1 1 0; min-width:84px; padding:9px 8px; }
  .bar { display:flex; gap:10px; align-items:center; margin-top:6px; position:sticky; bottom:0;
         z-index:10; padding:10px 0; background:linear-gradient(transparent, var(--bg) 22%); }
  .save { background:var(--ok); border-color:var(--ok); color:#fff; font-weight:700; padding:14px; font-size:1.05rem; }
  .save:disabled { opacity:.45; cursor:not-allowed; }
  .btn.dis { opacity:.28; cursor:not-allowed; }
  .btn.dis:hover { border-color:var(--line); }
  .nav { background:#222a33; border-color:var(--line); flex:0 0 auto; min-width:0; padding:14px 16px; font-weight:600; }
  .nav:disabled { opacity:.3; cursor:not-allowed; }
  .revisit { background:#3a2f16; border:1px solid var(--crit); color:#f0d9a8; border-radius:8px;
             padding:8px 12px; margin-bottom:12px; font-size:.9rem; }
  /* Weak-label hint: deliberately low-key so it assists without anchoring. */
  .weakhint { font-size:.78rem; color:var(--muted); padding:7px 12px; display:flex;
              gap:8px; align-items:center; flex-wrap:wrap; opacity:.85; }
  .weakhint .wlk { color:var(--fg); background:#1d242c; border:1px solid var(--line);
                   border-radius:6px; padding:2px 8px; font-weight:600; }
  .note { color:var(--crit); font-size:.78rem; margin:4px 0 2px; }
  .nabtn { flex:0 0 auto; min-width:0; padding:7px 10px; font-size:.8rem; margin-left:auto; }
  .skip { background:#3a2530; border-color:#5a3340; color:#f0c9d3; }
  .done { text-align:center; padding:60px 20px; font-size:1.2rem; }
  code { color:var(--muted); }
  /* Guide panel */
  .ghbtn { background:#222a33; border:1px solid var(--line); color:var(--fg); border-radius:8px;
           padding:6px 12px; cursor:pointer; font-size:.85rem; margin-left:8px; }
  .ghbtn:hover { border-color:var(--sel); }
  .overlay { position:fixed; inset:0; background:rgba(0,0,0,.6); z-index:50; display:none; }
  .overlay.show { display:block; }
  .guide { position:absolute; top:0; right:0; height:100%; width:min(680px,94vw); background:var(--card);
           border-left:1px solid var(--line); overflow-y:auto; padding:16px 22px 60px; }
  .guide h1 { font-size:1.25rem; } .guide h2 { font-size:1.05rem; margin-top:1.5em; }
  .guide h3 { font-size:.95rem; color:var(--crit); }
  .guide table { border-collapse:collapse; width:100%; margin:8px 0; font-size:.85rem; }
  .guide th, .guide td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
  .guide th { background:#1d242c; }
  .guide blockquote { border-left:3px solid var(--crit); margin:8px 0; padding:4px 12px; color:var(--muted); background:#1b2129; }
  .guide code { background:#0f1216; padding:1px 4px; border-radius:4px; color:#d8b46a; }
  .guide ul, .guide ol { padding-left:1.3em; } .guide li { margin:3px 0; }
  .guide hr { border:none; border-top:1px solid var(--line); margin:16px 0; }
  .ghclose { position:sticky; top:0; float:right; background:#3a2530; border:1px solid #5a3340;
             color:#f0c9d3; border-radius:8px; padding:6px 12px; cursor:pointer; }
</style></head>
<body>
<header>
  <span><b>R-CES</b> annotation</span>
  <span class="muted" id="passinfo"></span>
  <span class="muted" id="prog" style="margin-left:auto"></span>
  <button class="ghbtn" id="guidebtn" onclick="toggleGuide()" title="Boundary cheat sheet">📖 Guide</button>
  <div class="pbar"><div class="pfill" id="pfill"></div></div>
</header>
<main id="app"><div class="done">Loading…</div></main>
<div class="overlay" id="overlay" onclick="if(event.target===this)toggleGuide()">
  <div class="guide"><button class="ghclose" onclick="toggleGuide()">✕ Close</button><div id="guidebody">Loading...</div></div>
</div>
<script>
let DATA=null, SEL=null;
const REF = {
  decidability:"single image settles it (incl. not applicable) / a critical checkpoint unresolved here / rule needs non-visual evidence",
  compliance_label:"present target meets rule / visible violation / subject absent (decidable) / abstain (pairs w/ partially|undecidable)",
  image_quality:"clarity of the image itself, independent of the rule",
};
// evidence_sufficiency is DERIVED from decidability (codebook §7), not clicked.
const TYPICAL_SUFF = {decidable:"sufficient", partially_decidable:"partial", undecidable:"insufficient"};
function allowedCompliance(dec){
  if(dec==='decidable') return ['compliant','non_compliant','not_applicable'];
  if(dec==='partially_decidable'||dec==='undecidable') return ['unknown'];
  return DATA.field_options.compliance_label;  // not picked yet -> no constraint
}
function criticalMissing(){
  return DATA.pair.checkpoints.filter(c=>c.critical && SEL.checkpoints[c.key]==='missing').map(c=>c.key);
}
function decidableBlocked(){ return criticalMissing().length>0; }  // §8.5
// Re-derive locked/dependent fields from the current selection before each render.
function applyGates(){
  if(SEL.decidability==='decidable' && decidableBlocked()) SEL.decidability=null;  // §8.5
  const allowed = allowedCompliance(SEL.decidability);
  if(SEL.compliance_label && !allowed.includes(SEL.compliance_label)) SEL.compliance_label=null;  // §8.1
  if(allowed.length===1) SEL.compliance_label=allowed[0];  // §8.2 force unknown
  SEL.evidence_sufficiency_label = SEL.decidability ? TYPICAL_SUFF[SEL.decidability] : null;  // §7 derived
}
async function load(){
  const r = await fetch('/api/state'); DATA = await r.json();
  document.getElementById('passinfo').textContent = `${DATA.progress.pass} pass · ${DATA.progress.annotator}`;
  const p = DATA.progress;
  document.getElementById('prog').textContent =
    `${p.done}/${p.total} done · ${p.pending} pending` + (p.limit?` · run ${p.annotated_this_run}/${p.limit}`:``);
  document.getElementById('pfill').style.width = (p.total ? p.done/p.total*100 : 0) + '%';
  if(DATA.done){ document.getElementById('app').innerHTML = '<div class="done">✅ Nothing pending. All done for this run.</div>'; return; }
  SEL = {decidability:null,compliance_label:null,image_quality:null,evidence_sufficiency_label:null,checkpoints:{}};
  DATA.pair.checkpoints.forEach(c=> SEL.checkpoints[c.key]=null);
  // image_quality is rated once per image; later pairs inherit a locked value.
  if(DATA.pair.image_quality_locked) SEL.image_quality = DATA.pair.image_quality_locked;
  const pf = DATA.pair.prefill;  // a revisited pair comes back with its saved choices
  if(pf){
    SEL.decidability=pf.decidability; SEL.compliance_label=pf.compliance_label; SEL.image_quality=pf.image_quality;
    Object.keys(pf.checkpoints||{}).forEach(k=> SEL.checkpoints[k]=pf.checkpoints[k]);
  }
  render();
}
function grp(field, options, current, disabled){
  return `<div class="grp" data-field="${field}">` + options.map(o=>{
    const dis = disabled && disabled.includes(o);
    return `<button class="btn ${current===o?'sel':''} ${dis?'dis':''}"`
      + (dis?'':` onclick="pick('${field}','${o}')"`) + `>${o}</button>`;
  }).join('') + `</div>`;
}
const CP_LABEL = {observed:'observed (can judge)', missing:'missing (can\'t make out)', not_applicable:'n/a (no subject)'};
function cpgrp(key, current){
  return `<div class="grp" data-cp="${key}">` + DATA.checkpoint_options.map(o=>
    `<button class="btn ${current===o?'sel':''}" onclick="pickcp('${key}','${o}')">${CP_LABEL[o]||o}</button>`).join('') + `</div>`;
}
function fieldBlock(f){
  const fo = DATA.field_options;
  let disabled=[], note='';
  if(f==='image_quality' && DATA.pair.image_quality_locked){
    const q = DATA.pair.image_quality_locked;  // rated on an earlier pair of this image
    return `<div class="lab">${f.replace(/_/g,' ')} <span class="muted" style="font-weight:400">(set once per image)</span></div>`
      + `<div class="muted" style="font-size:.78rem;margin-bottom:4px">${REF[f]||''}</div>`
      + `<div class="grp"><button class="btn sel" disabled style="cursor:default">${q}</button></div>`;
  }
  if(f==='decidability' && decidableBlocked()){
    disabled=['decidable'];
    note=`<div class="note">decidable unavailable — critical checkpoint ${criticalMissing().join(', ')} marked missing (§8.5)</div>`;
  }
  if(f==='compliance_label'){
    const allowed = allowedCompliance(SEL.decidability);
    disabled = fo.compliance_label.filter(o=>!allowed.includes(o));
    if(SEL.decidability && allowed.length===1)
      note=`<div class="note">locked to <b>unknown</b> by decidability=${SEL.decidability} (§8.2)</div>`;
  }
  return `<div class="lab">${f.replace(/_/g,' ')}</div>`
    + `<div class="muted" style="font-size:.78rem;margin-bottom:4px">${REF[f]||''}</div>`
    + note + grp(f, fo[f], SEL[f], disabled);
}
function render(){
  applyGates();
  const pr = DATA.pair, fo = DATA.field_options;
  // Image-side judgments (made by looking at the photo) live in the left column.
  let leftFields = ['decidability','image_quality'].map(fieldBlock).join('');
  // evidence_sufficiency is derived (read-only), shown so the annotator sees it.
  const suff = SEL.evidence_sufficiency_label;
  leftFields += `<div class="lab">evidence sufficiency <span class="muted" style="font-weight:400">(derived from decidability, §7)</span></div>`
    + `<div class="grp"><button class="btn sel" disabled style="cursor:default">${suff||'—'}</button></div>`;
  // Rule-side judgment lives in the right column with the rule and checkpoints.
  const compliance = fieldBlock('compliance_label');
  const naHelper = pr.checkpoints.length
    ? `<button class="btn nabtn" onclick="allNa()">set remaining → n/a</button>` : '';
  const cpHelp = pr.checkpoints.length
    ? `<div class="muted" style="font-size:.78rem;margin:2px 0 8px">
        observed = state resolved, not necessarily passing · missing = in scope but this image can't resolve it · n/a = nothing to assess: subject absent, or an alternative OR-branch already settles it (e.g. a fixed cover makes the guardrail branch n/a — use n/a, not missing) · composite checkpoints are judged by you, not auto-computed
       </div>` : '';
  let cps = pr.checkpoints.length ? pr.checkpoints.map(c=>
    `<div class="cp"><div class="q"><span class="k ${c.critical?'crit':''}">${c.critical?'<span class="star">★</span> ':''}${c.key}</span><br><span class="muted">${c.question||''}</span></div>${cpgrp(c.key,SEL.checkpoints[c.key])}</div>`).join('')
    : '<div class="muted">(rule has no visual checkpoints)</div>';
  const banner = pr.revisiting
    ? `<div class="revisit">↩ Revisiting an already-saved pair — your earlier choices are pre-selected. Saving will <b>overwrite</b> it.</div>` : '';
  // Weak labels (professional-annotator hints): display-only, never saved.
  const weakParts = [];
  if(pr.image_weak_label) weakParts.push(`<span class="wlk">${pr.image_weak_label}</span>`);
  if(pr.weak_candidate_category) weakParts.push(`<span class="wlk">${pr.weak_candidate_category}</span>`);
  const weak = weakParts.length
    ? `<div class="card weakhint">weak hint <span class="muted">(professional annotator · not gold, do not just copy)</span> ${weakParts.join(' ')}</div>` : '';
  document.getElementById('app').innerHTML = `
    <div class="wrap">
      <div class="imgcol">
        <img id="photo" src="${pr.image_url}" alt="image">
        <div class="card imginfo"><span class=" imgrule">rule ${pr.image_rule_index}/${pr.image_rule_total} for this image</span><span class="muted"><code>${pr.image_id}</code>${pr.image_split?` · ${pr.image_split}`:''} · <code>${pr.pair_id}</code></span></div>
        ${weak}
        <div class="card">${leftFields}</div>
      </div>
      <div>
        ${banner}
        <div class="card"><h2>${pr.rule_name}</h2>
          <div class="muted">${pr.major_category} / ${pr.subcategory} · <code>${pr.decision_scope}</code></div>
        </div>
        <div class="card">${compliance}</div>
        <div class="card"><div class="lab" style="display:flex;align-items:center">checkpoints (★ = critical)${naHelper}</div>${cpHelp}${cps}</div>
        <div class="bar">
          <button class="btn nav" onclick="back()" ${pr.can_back?'':'disabled'} title="Re-edit a pair you saved this run">↩ Back</button>
          <button class="btn save" id="savebtn" onclick="save()" disabled>${pr.revisiting?'Overwrite &amp; continue':'Save &amp; next'}</button>
          ${pr.can_forward?`<button class="btn nav" onclick="forward()" title="Skip back toward the latest pair">Forward ↪</button>`:``}
          <button class="btn skip" onclick="skip()">Skip</button>
        </div>
      </div>
    </div>`;
  refresh();
}
function complete(){
  return ['decidability','compliance_label','image_quality'].every(f=>SEL[f])
    && Object.values(SEL.checkpoints).every(v=>v);
}
function refresh(){ const b=document.getElementById('savebtn'); if(b) b.disabled=!complete(); }
function pick(f,o){
  SEL[f]=o;
  // Decidable ⇒ the subject is resolvable (seen present, or confirmed absent),
  // so its presence gate is `observed`. Pre-fill it (only if unset, so a manual
  // choice or a revisit prefill is never overwritten) — annotator can still change it.
  if(f==='decidability' && o==='decidable'){
    const g = DATA.pair.subject_gate;
    if(g && SEL.checkpoints[g]==null) SEL.checkpoints[g]='observed';
  }
  render();
}
function pickcp(k,o){ SEL.checkpoints[k]=o; render(); }
function allNa(){ DATA.pair.checkpoints.forEach(c=>{ if(!SEL.checkpoints[c.key]) SEL.checkpoints[c.key]='not_applicable'; }); render(); }
async function save(){
  if(!complete()) return;
  const body = Object.assign({pair_id:DATA.pair.pair_id}, SEL);
  const r = await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j = await r.json(); if(!j.ok){ alert('Save failed: '+j.error); return; }
  load();
}
async function back(){ await fetch('/api/back',{method:'POST'}); load(); }
async function forward(){ await fetch('/api/forward',{method:'POST'}); load(); }
async function skip(){
  await fetch('/api/skip',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pair_id:DATA.pair.pair_id})});
  load();
}
// ---- Guide panel: tiny Markdown -> HTML (headings, lists, tables, quote, code) ----
function inl(s){
  s = s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  s = s.replace(/`([^`]+)`/g,'<code>$1</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>');
  return s;
}
function tableHtml(rows){
  const cells = r => r.trim().replace(/^\|/,'').replace(/\|\s*$/,'').split('|').map(c=>c.trim());
  const head = cells(rows[0]);
  let body = rows.slice(1);
  if(body.length && /^[\s:|-]+$/.test(body[0])) body = body.slice(1);  // drop |---| separator
  return '<table><thead><tr>' + head.map(c=>`<th>${inl(c)}</th>`).join('') + '</tr></thead><tbody>'
    + body.map(r=>'<tr>'+cells(r).map(c=>`<td>${inl(c)}</td>`).join('')+'</tr>').join('') + '</tbody></table>';
}
function blockStart(l){ return /^\s*$/.test(l)||/^\s*[-*#>|]/.test(l)||/^---+\s*$/.test(l)||/^\s*\d+\.\s+/.test(l); }
function mdToHtml(src){
  const L = src.replace(/\r/g,'').split('\n'); let out='', i=0;
  while(i<L.length){
    const line = L[i];
    if(/^\s*$/.test(line)){ i++; continue; }
    if(/^---+\s*$/.test(line)){ out+='<hr>'; i++; continue; }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if(h){ const n=h[1].length; out+=`<h${n}>${inl(h[2])}</h${n}>`; i++; continue; }
    if(/^\s*\|/.test(line)){ const t=[]; while(i<L.length && /^\s*\|/.test(L[i])){ t.push(L[i]); i++; } out+=tableHtml(t); continue; }
    if(/^\s*>\s?/.test(line)){ const b=[]; while(i<L.length && /^\s*>\s?/.test(L[i])){ b.push(L[i].replace(/^\s*>\s?/,'')); i++; } out+=`<blockquote>${b.map(x=>inl(x.replace(/^\s*-\s+/,'• '))).join('<br>')}</blockquote>`; continue; }
    if(/^\s*\d+\.\s+/.test(line)){ const it=[]; while(i<L.length && /^\s*\d+\.\s+/.test(L[i])){ it.push(L[i].replace(/^\s*\d+\.\s+/,'')); i++; } out+='<ol>'+it.map(t=>`<li>${inl(t)}</li>`).join('')+'</ol>'; continue; }
    if(/^\s*[-*]\s+/.test(line)){ const it=[]; while(i<L.length && /^\s*[-*]\s+/.test(L[i])){ it.push(L[i].replace(/^\s*[-*]\s+/,'')); i++; } out+='<ul>'+it.map(t=>`<li>${inl(t)}</li>`).join('')+'</ul>'; continue; }
    const p=[line]; i++;
    while(i<L.length && !blockStart(L[i])){ p.push(L[i]); i++; }
    out+=`<p>${inl(p.join(' '))}</p>`;
  }
  return out;
}
let GUIDE_HTML=null;
async function toggleGuide(){
  const ov=document.getElementById('overlay');
  if(ov.classList.contains('show')){ ov.classList.remove('show'); return; }
  if(GUIDE_HTML===null){
    try{ GUIDE_HTML = mdToHtml(await (await fetch('/api/guide')).text()); }
    catch(e){ GUIDE_HTML='<p>Failed to load guide.</p>'; }
    document.getElementById('guidebody').innerHTML=GUIDE_HTML;
  }
  ov.classList.add('show');
}
document.addEventListener('keydown', e=>{
  if(e.key==='Escape'){ const ov=document.getElementById('overlay'); if(ov.classList.contains('show')) ov.classList.remove('show'); }
});
load();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(STATE.payload())
        elif self.path == "/api/guide":
            try:
                text = GUIDE_PATH.read_text(encoding="utf-8")
            except OSError:
                text = "# Guide file missing\n\nCould not find `annotation_quick_guide.md`."
            self._send(200, text.encode("utf-8"), "text/markdown; charset=utf-8")
        elif self.path.startswith("/img/"):
            name = self.path[len("/img/"):].split("?")[0]
            # Originals live in IMAGE_DIR, crop-augmentation images in the separate
            # CROP_IMAGE_DIR; serve from whichever holds the requested image_id.
            img = None
            for base in (rio.IMAGE_DIR, rio.CROP_IMAGE_DIR):
                cand = (base / name).resolve()
                if cand.parent == base.resolve() and cand.is_file():
                    img = cand
                    break
            if img is None:
                self._send(404, b"not found", "text/plain")
                return
            ctype = mimetypes.guess_type(str(img))[0] or "application/octet-stream"
            self._send(200, img.read_bytes(), ctype)
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except json.JSONDecodeError:
            self._json({"ok": False, "error": "bad json"}, 400)
            return
        if self.path == "/api/save":
            ok, msg = STATE.save(body)
            self._json({"ok": ok, "error": "" if ok else msg})
        elif self.path == "/api/skip":
            STATE.skip(body.get("pair_id", ""))
            self._json({"ok": True})
        elif self.path == "/api/back":
            STATE.back()
            self._json({"ok": True})
        elif self.path == "/api/forward":
            STATE.forward()
            self._json({"ok": True})
        else:
            self._json({"ok": False, "error": "unknown route"}, 404)


def lan_ips() -> list[str]:
    return [ADVERTISED_LAN_IP] if ADVERTISED_LAN_IP else []


def candidate_ports() -> list[int]:
    return [PORT, *range(PORT + 1, PORT + PORT_FALLBACK_COUNT + 1)]


def bind_annotation_server(handler_cls: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    occupied: list[int] = []
    for port in candidate_ports():
        try:
            return ThreadingHTTPServer((HOST, port), handler_cls)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            occupied.append(port)
    ports = ", ".join(str(port) for port in occupied)
    raise SystemExit(f"No available annotation port. Tried: {ports}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the R-CES LAN annotation UI.",
    )
    parser.add_argument(
        "annotator_number",
        nargs="?",
        type=int,
        choices=sorted(ANNOTATOR_PASS_BY_NUMBER),
        default=1,
        help="Annotator number: 1=first pass (gold), 2=second pass (double), "
             "3=augment pass (crop screening), 4=enhance pass (test expansion). "
             "Default: 1.",
    )
    return parser.parse_args(argv)


def configure_run(argv: list[str] | None = None) -> None:
    global ANNOTATION_PASS
    args = parse_args(argv)
    ANNOTATION_PASS = ANNOTATOR_PASS_BY_NUMBER[args.annotator_number]


def main(argv: list[str] | None = None) -> None:
    global STATE
    configure_run(argv)
    STATE = AnnotatorState()
    p = STATE.progress()
    server = bind_annotation_server(Handler)
    bound_port = server.server_address[1]
    print(f"[{p['pass']} pass / {p['annotator']}] {p['done']}/{p['total']} done, {p['pending']} pending "
          f"-> writes {STATE.out_path.relative_to(rio.PROJECT_ROOT)}")
    print(f"limit={LIMIT or 'all'} | sample={SAMPLE_MODE}")
    if bound_port != PORT:
        print(f"Port {PORT} is busy; using fallback port {bound_port}.")
    print("Open on this machine:  http://localhost:%d/" % bound_port)
    for ip in lan_ips():
        print(f"Open on the LAN:        http://{ip}:{bound_port}/")
    print("Ctrl-C to stop (progress is saved after every pair).", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
