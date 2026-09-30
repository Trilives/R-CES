"""Stage 1b - LAN web ADJUDICATION UI (resolve the double-annotation disagreements).

After the first pass (gold, annotator_1 -> rces_pilot_pairs.jsonl) and the
isolated second pass (annotator_2 -> rces_double_annotation.jsonl), the two
annotators disagree on some of the ~20% double-annotated subset. This tool walks
a human adjudicator through ONLY those disagreeing pairs and lets them resolve
each one. The disagreement set follows the same categorical-field comparison: any
of the four categorical judgment fields differ, or critical_missing_checkpoints
differ.

For each disagreement the page shows the image + rule + checkpoints and, side by
side, what annotator_1 and annotator_2 each chose (disagreeing fields
highlighted). Three actions per pair:

  - Keep ①  -> adopt annotator_1's judgment (what gold already holds)
  - Keep ②  -> adopt annotator_2's judgment
  - Re-label -> open the full annotation form and judge it fresh (same gates and
               cross-field invariants as annotate_web.py)

The adjudicated record is written to an ISOLATED file
(rces_adjudication.jsonl), one record per resolved pair, never touching gold
directly. A separate merge step (tmp/merge_adjudication.py) applies these
decisions back into rces_pilot_pairs.jsonl to form the final gold. Resumable:
already-adjudicated pairs are skipped on restart, and "Back" re-opens a pair
resolved this run.

    uv run python Script/adjudicate_web.py

Open http://<this-machine-LAN-ip>:<PORT>/ on any device on the same network
(the startup banner prints the URLs). One adjudicator at a time per server.
"""

from __future__ import annotations

import errno
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
import rces_io as rio  # noqa: E402

GUIDE_PATH = _HERE / "annotation_quick_guide.md"

# ---- Run controls (edit these; the repo deliberately avoids CLI flags) -------
HOST = "0.0.0.0"            # bind all interfaces so the LAN can reach it
PORT = 8000
PORT_FALLBACK_COUNT = 20    # if PORT is busy, try PORT+1 ... PORT+N
ADVERTISED_LAN_IP = "127.0.0.1"
ADJUDICATION_PATH = rio.PAIRS_DIR / "rces_adjudication.jsonl"  # isolated output
ADJUDICATOR_ID = "adjudicator"
# ------------------------------------------------------------------------------

# The four categorical judgment fields whose disagreement defines the worklist
# Categorical fields used for inter-annotator disagreement detection.
CATEGORICAL_FIELDS = ("decidability", "compliance_label", "image_quality", "evidence_sufficiency_label")
# Judgment fields copied wholesale when an annotator's labelling is adopted.
JUDGMENT_FIELDS = (
    "decidability", "compliance_label", "image_quality", "evidence_sufficiency_label",
    "observed_checkpoints", "missing_checkpoints", "critical_missing_checkpoints",
)

FIELD_OPTIONS = {
    "decidability": list(rio.DECIDABILITY),
    "compliance_label": list(rio.COMPLIANCE),
    "image_quality": list(rio.IMAGE_QUALITY),
}
CHECKPOINT_OPTIONS = ("observed", "missing", "not_applicable")


def _annotated(row: dict[str, Any]) -> bool:
    return row.get("annotation_status") in ("judged", "described", "reviewed")


def compute_diffs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, list[Any]]:
    """The fields on which the two annotators disagree, each mapped to
    [annotator_1_value, annotator_2_value]. Same rule as compute_iaa.py."""
    diffs: dict[str, list[Any]] = {}
    for field in CATEGORICAL_FIELDS:
        if a.get(field) != b.get(field):
            diffs[field] = [a.get(field), b.get(field)]
    a_cm = sorted(a.get("critical_missing_checkpoints", []))
    b_cm = sorted(b.get("critical_missing_checkpoints", []))
    if a_cm != b_cm:
        diffs["critical_missing_checkpoints"] = [a_cm, b_cm]
    return diffs


def build_worklist() -> list[dict[str, Any]]:
    """The disagreeing pairs both annotators judged, in gold (pair_id) order,
    each carrying both annotators' full rows and the diff."""
    gold = {r["pair_id"]: r for r in rio.load_jsonl(rio.GOLD_PATH) if _annotated(r)}
    double = {r["pair_id"]: r for r in rio.load_jsonl(rio.DOUBLE_PATH) if _annotated(r)}
    work: list[dict[str, Any]] = []
    for pid in sorted(set(gold) & set(double)):
        diffs = compute_diffs(gold[pid], double[pid])
        if diffs:
            work.append({"pair_id": pid, "a1": gold[pid], "a2": double[pid], "diffs": diffs})
    return work


class AdjudicationState:
    """In-memory adjudication session; one active adjudicator per server."""

    def __init__(self) -> None:
        self.rules = rio.load_rules()
        self.work = build_worklist()
        self.work_by_id = {w["pair_id"]: w for w in self.work}
        # Resume: keep prior decisions, keyed by pair_id.
        self.decisions = {r["pair_id"]: r for r in rio.load_jsonl(ADJUDICATION_PATH)}
        self.lock = threading.Lock()
        self.resolved_this_run = 0
        self.served_at: float | None = None
        self.served_pid: str | None = None
        # Back/forward over pairs resolved THIS run.
        self.history: list[str] = []
        self.cursor: int | None = None

    # ---- progress / navigation -------------------------------------------
    def progress(self) -> dict[str, Any]:
        total = len(self.work)
        done = sum(1 for w in self.work if w["pair_id"] in self.decisions)
        return {"total": total, "done": done, "pending": total - done,
                "resolved_this_run": self.resolved_this_run, "adjudicator": ADJUDICATOR_ID}

    def _current_pid(self) -> str | None:
        for w in self.work:
            if w["pair_id"] not in self.decisions:
                return w["pair_id"]
        return None

    def _step_forward(self) -> None:
        if self.cursor is None:
            return
        if self.cursor < len(self.history) - 1:
            self.cursor += 1
        else:
            self.cursor = None

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

    # ---- payload ---------------------------------------------------------
    def _annotator_view(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "annotator_id": row.get("annotator_id", ""),
            "decidability": row.get("decidability"),
            "compliance_label": row.get("compliance_label"),
            "image_quality": row.get("image_quality"),
            "evidence_sufficiency_label": row.get("evidence_sufficiency_label"),
            "observed_checkpoints": row.get("observed_checkpoints", []),
            "missing_checkpoints": row.get("missing_checkpoints", []),
            "critical_missing_checkpoints": row.get("critical_missing_checkpoints", []),
        }

    def payload(self) -> dict[str, Any]:
        with self.lock:
            revisiting = self.cursor is not None
            pid = self.history[self.cursor] if revisiting else self._current_pid()
            if pid is None:
                return {"done": True, "progress": self.progress()}
            w = self.work_by_id[pid]
            a1, a2 = w["a1"], w["a2"]
            rule = self.rules.get(a1["rule_id"], {})
            required, critical, questions = rio.checkpoints_for(rule)
            self.served_at = time.monotonic()
            self.served_pid = pid
            prior = self.decisions.get(pid)
            pair = {
                "pair_id": pid,
                "image_url": f"/img/{a1['image_id']}",
                "image_id": a1["image_id"],
                "image_split": a1.get("image_split", ""),
                "rule_name": a1["rule_name"],
                "major_category": a1["major_category"],
                "subcategory": a1["subcategory"],
                "decision_scope": a1["decision_scope"],
                "checkpoints": [
                    {"key": k, "question": questions.get(k, ""), "critical": k in critical}
                    for k in required
                ],
                "subject_gate": rio.subject_gate(rule),
                "diff_fields": sorted(w["diffs"].keys()),
                "a1": self._annotator_view(a1),
                "a2": self._annotator_view(a2),
                "revisiting": revisiting,
                "can_back": bool(self.history) and self.cursor != 0,
                "can_forward": revisiting,
                "prior_choice": (prior or {}).get("adjudication_choice"),
            }
            return {
                "done": False,
                "progress": self.progress(),
                "field_options": FIELD_OPTIONS,
                "checkpoint_options": list(CHECKPOINT_OPTIONS),
                "pair": pair,
            }

    # ---- adjudicate ------------------------------------------------------
    def _validate_relabel(self, posted: dict[str, Any], rule: dict[str, Any]) -> tuple[bool, str]:
        for field, options in FIELD_OPTIONS.items():
            if posted.get(field) not in options:
                return False, f"{field} must be one of {options}"
        required, _, _ = rio.checkpoints_for(rule)
        marks = posted.get("checkpoints", {})
        for key in required:
            if marks.get(key) not in CHECKPOINT_OPTIONS:
                return False, f"checkpoint {key} must be one of {list(CHECKPOINT_OPTIONS)}"
        missing = [k for k in required if marks.get(k) == "missing"]
        errs = rio.consistency_errors(
            posted["decidability"], posted["compliance_label"], rio.critical_missing(missing, rule)
        )
        return (False, "; ".join(errs)) if errs else (True, "")

    def _record(self, w: dict[str, Any], choice: str, posted: dict[str, Any],
                rule: dict[str, Any], seconds: float) -> dict[str, Any]:
        """Build the adjudicated record: gold's metadata, judgment from the chosen
        source (or the re-label form), and adjudication provenance."""
        a1 = w["a1"]
        base = dict(a1)  # metadata + double_annotation + weak labels + visual_evidence/missing_information
        diff_str = ",".join(sorted(w["diffs"].keys()))
        if choice in ("annotator_1", "annotator_2"):
            src = a1 if choice == "annotator_1" else w["a2"]
            for field in JUDGMENT_FIELDS:
                base[field] = src.get(field)
            annotator = choice
            note = f"adjudicated: kept {choice} over [{diff_str}]"
        elif choice == "relabel":
            required, _, _ = rio.checkpoints_for(rule)
            marks = posted["checkpoints"]
            base["decidability"] = posted["decidability"]
            base["compliance_label"] = posted["compliance_label"]
            base["image_quality"] = posted["image_quality"]
            base["evidence_sufficiency_label"] = rio.derive_evidence_sufficiency(posted["decidability"])
            base["observed_checkpoints"] = [k for k in required if marks.get(k) == "observed"]
            base["missing_checkpoints"] = [k for k in required if marks.get(k) == "missing"]
            base["critical_missing_checkpoints"] = rio.critical_missing(base["missing_checkpoints"], rule)
            annotator = ADJUDICATOR_ID
            note = f"adjudicated: relabel over [{diff_str}]"
        else:
            raise ValueError(f"unknown choice {choice!r}")
        base.update({
            "source": rio.SOURCE_HUMAN,
            "annotation_status": "judged",
            "review_status": "adjudicated",
            "annotator_id": annotator,
            "annotation_seconds": seconds,
            "annotation_notes": note,
            "adjudication_choice": choice,  # transient; merge consumes & drops it
        })
        return base

    def adjudicate(self, posted: dict[str, Any]) -> tuple[bool, str]:
        with self.lock:
            pid = posted.get("pair_id")
            choice = posted.get("choice")
            w = self.work_by_id.get(pid)
            if w is None:
                return False, "unknown pair_id"
            if choice not in ("annotator_1", "annotator_2", "relabel"):
                return False, "choice must be annotator_1 | annotator_2 | relabel"
            rule = self.rules.get(w["a1"]["rule_id"], {})
            if choice == "relabel":
                ok, msg = self._validate_relabel(posted, rule)
                if not ok:
                    return False, msg
            seconds = round(time.monotonic() - self.served_at, 1) if self.served_at and self.served_pid == pid else 0.0
            first_time = pid not in self.decisions
            self.decisions[pid] = self._record(w, choice, posted, rule, seconds)
            # Persist in worklist order so the file reads predictably.
            ordered = [self.decisions[x["pair_id"]] for x in self.work if x["pair_id"] in self.decisions]
            rio.write_jsonl(ADJUDICATION_PATH, ordered)
            if pid in self.history:
                self._step_forward()
            else:
                self.history.append(pid)
            if first_time:
                self.resolved_this_run += 1
            return True, "saved"


STATE: AdjudicationState  # set in main()

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>R-CES adjudication</title>
<style>
  :root { --bg:#0f1216; --card:#171c22; --line:#2a313a; --fg:#e6e9ee; --muted:#8a94a2;
          --sel:#2d6cdf; --crit:#e0a34a; --ok:#2f9e57; --a1:#2d6cdf; --a2:#9b59b6; --warn:#e0573a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.45 system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:20; background:var(--card); border-bottom:1px solid var(--line);
           padding:10px 14px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  header b { color:var(--sel); } .muted { color:var(--muted); }
  .pbar { flex:0 0 100%; height:4px; background:var(--line); border-radius:2px; overflow:hidden; }
  .pfill { height:100%; width:0; background:var(--ok); transition:width .3s ease; }
  main { max-width:1500px; margin:0 auto; padding:14px; }
  .wrap { display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start; }
  @media (max-width:980px){ .wrap{ grid-template-columns:1fr; } }
  img#photo { display:block; width:100%; max-height:62vh; object-fit:contain;
              border:1px solid var(--line); border-radius:8px; background:#000; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:12px; }
  .imginfo { font-size:.82rem; padding:8px 12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  h2 { margin:.1em 0 .4em; font-size:1.05rem; }
  .lab { color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; margin:10px 0 6px; }
  /* side-by-side annotator comparison */
  .cmp { width:100%; border-collapse:collapse; font-size:.9rem; }
  .cmp th, .cmp td { border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }
  .cmp th { background:#1d242c; font-weight:600; }
  .cmp td.field { color:var(--muted); white-space:nowrap; }
  .cmp .a1h { color:#9cc0ff; } .cmp .a2h { color:#d8a8ee; }
  .cmp tr.diff td { background:#2a211a; }
  .cmp tr.diff td.field { color:var(--crit); font-weight:600; }
  .pickrow { display:flex; gap:10px; margin-top:12px; flex-wrap:wrap; }
  .pick { flex:1 1 0; min-width:150px; padding:14px 12px; border:1px solid var(--line); border-radius:8px;
          background:#1d242c; color:var(--fg); font-size:.98rem; cursor:pointer; text-align:center; font-weight:600; }
  .pick:hover { border-color:var(--sel); }
  .pick.k1 { border-color:var(--a1); } .pick.k1:hover { background:#1b2740; }
  .pick.k2 { border-color:var(--a2); } .pick.k2:hover { background:#2a1d33; }
  .pick.rl { border-color:var(--warn); } .pick.rl:hover { background:#33231d; }
  .pick.sel { background:var(--sel); border-color:var(--sel); color:#fff; }
  .pick.k1.sel { background:var(--a1); } .pick.k2.sel { background:var(--a2); } .pick.rl.sel { background:var(--warn); }
  /* re-label form (reused from annotate_web) */
  .grp { display:flex; gap:8px; flex-wrap:wrap; }
  .btn { flex:1 1 auto; min-width:96px; padding:11px 10px; border:1px solid var(--line); border-radius:8px;
         background:#1d242c; color:var(--fg); font-size:.92rem; cursor:pointer; text-align:center; }
  .btn:hover { border-color:var(--sel); }
  .btn.sel { background:var(--sel); border-color:var(--sel); color:#fff; font-weight:600; }
  .btn.dis { opacity:.28; cursor:not-allowed; } .btn.dis:hover { border-color:var(--line); }
  .cp { display:flex; flex-direction:column; gap:8px; padding:10px 0; border-top:1px solid var(--line); }
  .cp:first-child{ border-top:none; }
  .cp .k { font-weight:600; } .crit { color:var(--crit); } .star { color:var(--crit); font-weight:700; }
  .note { color:var(--crit); font-size:.78rem; margin:4px 0 2px; }
  .relabelbox { margin-top:12px; padding-top:12px; border-top:1px dashed var(--line); display:none; }
  .relabelbox.show { display:block; }
  .bar { display:flex; gap:10px; align-items:center; margin-top:14px; position:sticky; bottom:0;
         z-index:10; padding:10px 0; background:linear-gradient(transparent, var(--bg) 22%); }
  .save { background:var(--ok); border-color:var(--ok); color:#fff; font-weight:700; padding:14px; font-size:1.05rem; flex:1 1 auto; }
  .save:disabled { opacity:.45; cursor:not-allowed; }
  .nav { background:#222a33; border-color:var(--line); flex:0 0 auto; padding:14px 16px; font-weight:600; }
  .nav:disabled { opacity:.3; cursor:not-allowed; }
  .revisit { background:#3a2f16; border:1px solid var(--crit); color:#f0d9a8; border-radius:8px;
             padding:8px 12px; margin-bottom:12px; font-size:.9rem; }
  .done { text-align:center; padding:60px 20px; font-size:1.2rem; }
  code { color:var(--muted); }
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
  <span><b>R-CES</b> adjudication</span>
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
let DATA=null, CHOICE=null, SEL=null;
const TYPICAL_SUFF = {decidable:"sufficient", partially_decidable:"partial", undecidable:"insufficient"};
function allowedCompliance(dec){
  if(dec==='decidable') return ['compliant','non_compliant','not_applicable'];
  if(dec==='partially_decidable'||dec==='undecidable') return ['unknown'];
  return DATA.field_options.compliance_label;
}
function criticalMissing(){
  return DATA.pair.checkpoints.filter(c=>c.critical && SEL.checkpoints[c.key]==='missing').map(c=>c.key);
}
function decidableBlocked(){ return criticalMissing().length>0; }
function applyGates(){
  if(SEL.decidability==='decidable' && decidableBlocked()) SEL.decidability=null;
  const allowed = allowedCompliance(SEL.decidability);
  if(SEL.compliance_label && !allowed.includes(SEL.compliance_label)) SEL.compliance_label=null;
  if(allowed.length===1) SEL.compliance_label=allowed[0];
  SEL.evidence_sufficiency_label = SEL.decidability ? TYPICAL_SUFF[SEL.decidability] : null;
}
async function load(){
  const r = await fetch('/api/state'); DATA = await r.json();
  document.getElementById('passinfo').textContent = `adjudicator · ${DATA.progress.adjudicator}`;
  const p = DATA.progress;
  document.getElementById('prog').textContent = `${p.done}/${p.total} resolved · ${p.pending} pending`;
  document.getElementById('pfill').style.width = (p.total ? p.done/p.total*100 : 0) + '%';
  if(DATA.done){ document.getElementById('app').innerHTML = '<div class="done">✅ All disagreements resolved. Run the merge step to fold them into gold.</div>'; return; }
  CHOICE = DATA.pair.prior_choice || null;
  SEL = {decidability:null,compliance_label:null,image_quality:null,evidence_sufficiency_label:null,checkpoints:{}};
  DATA.pair.checkpoints.forEach(c=> SEL.checkpoints[c.key]=null);
  render();
}
const FIELDS = ['decidability','compliance_label','image_quality','evidence_sufficiency_label','critical_missing_checkpoints'];
const FLABEL = {decidability:'decidability', compliance_label:'compliance', image_quality:'image quality',
                evidence_sufficiency_label:'evidence suff.', critical_missing_checkpoints:'critical missing'};
function fmt(v){ if(Array.isArray(v)) return v.length?v.join(', '):'—'; return v==null?'—':v; }
function cmpTable(){
  const pr=DATA.pair, diff=new Set(pr.diff_fields);
  const rows = FIELDS.map(f=>{
    const d = diff.has(f);
    return `<tr class="${d?'diff':''}"><td class="field">${FLABEL[f]}</td>`
      + `<td>${fmt(pr.a1[f])}</td><td>${fmt(pr.a2[f])}</td></tr>`;
  }).join('');
  return `<table class="cmp"><thead><tr><th>field</th>`
    + `<th class="a1h">① ${pr.a1.annotator_id||'annotator_1'}</th>`
    + `<th class="a2h">② ${pr.a2.annotator_id||'annotator_2'}</th></tr></thead><tbody>${rows}</tbody></table>`;
}
// ---- re-label form (mirrors annotate_web) ----
const REF = {
  decidability:"single image settles it (incl. not applicable) / a critical checkpoint unresolved here / rule needs non-visual evidence",
  compliance_label:"present target meets rule / visible violation / subject absent (decidable) / abstain (partially|undecidable)",
  image_quality:"clarity of the image itself, independent of the rule",
};
function grp(field, options, current, disabled){
  return `<div class="grp">` + options.map(o=>{
    const dis = disabled && disabled.includes(o);
    return `<button class="btn ${current===o?'sel':''} ${dis?'dis':''}"`
      + (dis?'':` onclick="pick('${field}','${o}')"`) + `>${o}</button>`;
  }).join('') + `</div>`;
}
const CP_LABEL = {observed:'observed', missing:'missing', not_applicable:'n/a'};
function cpgrp(key, current){
  return `<div class="grp">` + DATA.checkpoint_options.map(o=>
    `<button class="btn ${current===o?'sel':''}" onclick="pickcp('${key}','${o}')">${CP_LABEL[o]||o}</button>`).join('') + `</div>`;
}
function fieldBlock(f){
  const fo = DATA.field_options; let disabled=[], note='';
  if(f==='decidability' && decidableBlocked()){
    disabled=['decidable'];
    note=`<div class="note">decidable unavailable — critical checkpoint ${criticalMissing().join(', ')} missing (§8.5)</div>`;
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
function relabelForm(){
  const pr=DATA.pair;
  applyGates();
  let fields = ['decidability','compliance_label','image_quality'].map(fieldBlock).join('');
  const suff = SEL.evidence_sufficiency_label;
  fields += `<div class="lab">evidence sufficiency <span class="muted" style="font-weight:400">(derived, §7)</span></div>`
    + `<div class="grp"><button class="btn sel" disabled style="cursor:default">${suff||'—'}</button></div>`;
  const cps = pr.checkpoints.length ? pr.checkpoints.map(c=>
    `<div class="cp"><div><span class="k ${c.critical?'crit':''}">${c.critical?'<span class="star">★</span> ':''}${c.key}</span>`
    + `<br><span class="muted">${c.question||''}</span></div>${cpgrp(c.key,SEL.checkpoints[c.key])}</div>`).join('')
    : '<div class="muted">(rule has no visual checkpoints)</div>';
  return `<div class="lab">re-label — judge fresh</div>${fields}`
    + `<div class="lab" style="margin-top:12px">checkpoints (★ = critical)</div>${cps}`;
}
function render(){
  const pr=DATA.pair;
  const banner = pr.revisiting
    ? `<div class="revisit">↩ Revisiting a pair you already resolved this run${pr.prior_choice?` (was: <b>${pr.prior_choice}</b>)`:''}. Saving overwrites it.</div>` : '';
  const relabelShown = CHOICE==='relabel';
  document.getElementById('app').innerHTML = `
    <div class="wrap">
      <div>
        <img id="photo" src="${pr.image_url}" alt="image">
        <div class="card imginfo"><code>${pr.image_id}</code>${pr.image_split?` · ${pr.image_split}`:''} · <code>${pr.pair_id}</code></div>
      </div>
      <div>
        ${banner}
        <div class="card"><h2>${pr.rule_name}</h2>
          <div class="muted">${pr.major_category} / ${pr.subcategory} · <code>${pr.decision_scope}</code></div>
        </div>
        <div class="card">
          <div class="lab">two annotators — disagreeing fields highlighted</div>
          ${cmpTable()}
          <div class="pickrow">
            <button class="pick k1 ${CHOICE==='annotator_1'?'sel':''}" onclick="choose('annotator_1')">Keep ① ${pr.a1.annotator_id||'annotator_1'}</button>
            <button class="pick k2 ${CHOICE==='annotator_2'?'sel':''}" onclick="choose('annotator_2')">Keep ② ${pr.a2.annotator_id||'annotator_2'}</button>
            <button class="pick rl ${CHOICE==='relabel'?'sel':''}" onclick="choose('relabel')">Re-label ✎</button>
          </div>
          <div class="relabelbox ${relabelShown?'show':''}" id="relabelbox">${relabelShown?relabelForm():''}</div>
        </div>
        <div class="bar">
          <button class="btn nav" onclick="back()" ${pr.can_back?'':'disabled'} title="Re-open a pair you resolved this run">↩ Back</button>
          <button class="save" id="savebtn" onclick="save()" disabled>Save &amp; next</button>
          ${pr.can_forward?`<button class="btn nav" onclick="forward()">Forward ↪</button>`:``}
        </div>
      </div>
    </div>`;
  refresh();
}
function relabelComplete(){
  return ['decidability','compliance_label','image_quality'].every(f=>SEL[f])
    && Object.values(SEL.checkpoints).every(v=>v);
}
function complete(){
  if(CHOICE==='annotator_1' || CHOICE==='annotator_2') return true;
  if(CHOICE==='relabel') return relabelComplete();
  return false;
}
function refresh(){ const b=document.getElementById('savebtn'); if(b) b.disabled=!complete(); }
function choose(c){ CHOICE=c; render(); }
function pick(f,o){
  SEL[f]=o;
  if(f==='decidability' && o==='decidable'){
    const g = DATA.pair.subject_gate;
    if(g && SEL.checkpoints[g]==null) SEL.checkpoints[g]='observed';
  }
  render();
}
function pickcp(k,o){ SEL.checkpoints[k]=o; render(); }
async function save(){
  if(!complete()) return;
  const body = {pair_id:DATA.pair.pair_id, choice:CHOICE};
  if(CHOICE==='relabel') Object.assign(body, {
    decidability:SEL.decidability, compliance_label:SEL.compliance_label,
    image_quality:SEL.image_quality, checkpoints:SEL.checkpoints,
  });
  const r = await fetch('/api/adjudicate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j = await r.json(); if(!j.ok){ alert('Save failed: '+j.error); return; }
  load();
}
async function back(){ await fetch('/api/back',{method:'POST'}); load(); }
async function forward(){ await fetch('/api/forward',{method:'POST'}); load(); }
// ---- Guide panel (same tiny Markdown renderer as annotate_web) ----
function inl(s){ s=s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  s=s.replace(/`([^`]+)`/g,'<code>$1</code>'); s=s.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>'); return s; }
function tableHtml(rows){
  const cells = r => r.trim().replace(/^\|/,'').replace(/\|\s*$/,'').split('|').map(c=>c.trim());
  const head = cells(rows[0]); let body = rows.slice(1);
  if(body.length && /^[\s:|-]+$/.test(body[0])) body = body.slice(1);
  return '<table><thead><tr>' + head.map(c=>`<th>${inl(c)}</th>`).join('') + '</tr></thead><tbody>'
    + body.map(r=>'<tr>'+cells(r).map(c=>`<td>${inl(c)}</td>`).join('')+'</tr>').join('') + '</tbody></table>';
}
function blockStart(l){ return /^\s*$/.test(l)||/^\s*[-*#>|]/.test(l)||/^---+\s*$/.test(l)||/^\s*\d+\.\s+/.test(l); }
function mdToHtml(src){
  const L = src.replace(/\r/g,'').split('\n'); let out='', i=0;
  while(i<L.length){
    const line=L[i];
    if(/^\s*$/.test(line)){ i++; continue; }
    if(/^---+\s*$/.test(line)){ out+='<hr>'; i++; continue; }
    const h=line.match(/^(#{1,6})\s+(.*)$/);
    if(h){ const n=h[1].length; out+=`<h${n}>${inl(h[2])}</h${n}>`; i++; continue; }
    if(/^\s*\|/.test(line)){ const t=[]; while(i<L.length && /^\s*\|/.test(L[i])){ t.push(L[i]); i++; } out+=tableHtml(t); continue; }
    if(/^\s*>\s?/.test(line)){ const b=[]; while(i<L.length && /^\s*>\s?/.test(L[i])){ b.push(L[i].replace(/^\s*>\s?/,'')); i++; } out+=`<blockquote>${b.map(x=>inl(x.replace(/^\s*-\s+/,'• '))).join('<br>')}</blockquote>`; continue; }
    if(/^\s*\d+\.\s+/.test(line)){ const it=[]; while(i<L.length && /^\s*\d+\.\s+/.test(L[i])){ it.push(L[i].replace(/^\s*\d+\.\s+/,'')); i++; } out+='<ol>'+it.map(t=>`<li>${inl(t)}</li>`).join('')+'</ol>'; continue; }
    if(/^\s*[-*]\s+/.test(line)){ const it=[]; while(i<L.length && /^\s*[-*]\s+/.test(L[i])){ it.push(L[i].replace(/^\s*[-*]\s+/,'')); i++; } out+='<ul>'+it.map(t=>`<li>${inl(t)}</li>`).join('')+'</ul>'; continue; }
    const p=[line]; i++; while(i<L.length && !blockStart(L[i])){ p.push(L[i]); i++; } out+=`<p>${inl(p.join(' '))}</p>`;
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
            img = (rio.IMAGE_DIR / name).resolve()
            if img.parent != rio.IMAGE_DIR.resolve() or not img.is_file():
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
        if self.path == "/api/adjudicate":
            ok, msg = STATE.adjudicate(body)
            self._json({"ok": ok, "error": "" if ok else msg})
        elif self.path == "/api/back":
            STATE.back()
            self._json({"ok": True})
        elif self.path == "/api/forward":
            STATE.forward()
            self._json({"ok": True})
        else:
            self._json({"ok": False, "error": "unknown route"}, 404)


def candidate_ports() -> list[int]:
    return [PORT, *range(PORT + 1, PORT + PORT_FALLBACK_COUNT + 1)]


def bind_server(handler_cls: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    occupied: list[int] = []
    for port in candidate_ports():
        try:
            return ThreadingHTTPServer((HOST, port), handler_cls)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            occupied.append(port)
    raise SystemExit(f"No available port. Tried: {', '.join(map(str, occupied))}")


def main() -> None:
    global STATE
    STATE = AdjudicationState()
    p = STATE.progress()
    server = bind_server(Handler)
    bound_port = server.server_address[1]
    print(f"[adjudication] {p['done']}/{p['total']} resolved, {p['pending']} disagreements pending "
          f"-> writes {ADJUDICATION_PATH.relative_to(rio.PROJECT_ROOT)}")
    if bound_port != PORT:
        print(f"Port {PORT} is busy; using fallback port {bound_port}.")
    print("Open on this machine:  http://localhost:%d/" % bound_port)
    if ADVERTISED_LAN_IP:
        print(f"Open on the LAN:        http://{ADVERTISED_LAN_IP}:{bound_port}/")
    print("Ctrl-C to stop (each decision is saved immediately).", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
