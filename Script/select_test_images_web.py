"""Select unused R-CES images for test-set expansion in a local web UI.

The active pilot uses only the image IDs referenced by
``data/pairs/rces_pilot_pairs.jsonl``. This tool presents every other image one
at a time and immediately saves selected names to
``data/splits/test_expansion_selected_images.txt``. The output is a selection
pool, not a locked final test split; annotation and stratified re-splitting are
separate follow-up steps outside this review package.

Run:

    uv run python Script/select_test_images_web.py

Then open the printed localhost URL. The server deliberately binds only to the
local machine because it can modify the selection file.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hmac
import json
import mimetypes
import os
import secrets
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse


RCES_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = RCES_ROOT / "data" / "images"
ACTIVE_PAIRS_PATH = RCES_ROOT / "data" / "pairs" / "rces_pilot_pairs.jsonl"
OUTPUT_PATH = RCES_ROOT / "data" / "splits" / "test_expansion_selected_images.txt"
PROGRESS_PATH = RCES_ROOT / "cache" / "test_expansion_selector_state.json"
LOCK_PATH = RCES_ROOT / "cache" / "test_expansion_selector.lock"

HOST = "127.0.0.1"
PORT = 8010
PORT_FALLBACK_COUNT = 20
MAX_REQUEST_BYTES = 4096
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"})
VALID_ACTIONS = frozenset({"select_next", "skip_next", "deselect"})


def _read_active_image_ids(pairs_path: Path) -> frozenset[str]:
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Active pair file not found: {pairs_path}")

    image_ids: set[str] = set()
    with pairs_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {pairs_path}:{line_number}: {exc}") from exc
            image_id = row.get("image_id") if isinstance(row, dict) else None
            if not isinstance(image_id, str) or not image_id:
                raise ValueError(f"Missing image_id at {pairs_path}:{line_number}")
            if Path(image_id).name != image_id:
                raise ValueError(f"Unsafe image_id at {pairs_path}:{line_number}: {image_id!r}")
            image_ids.add(image_id)
    return frozenset(image_ids)


def discover_candidates(image_dir: Path, pairs_path: Path) -> tuple[tuple[str, ...], frozenset[str]]:
    """Return sorted unused image names and active-pair image IDs."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    all_images = tuple(
        sorted(
            path.name
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    )
    used = _read_active_image_ids(pairs_path)
    missing = used.difference(all_images)
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Active pairs reference {len(missing)} missing image(s): {examples}")
    return tuple(name for name in all_images if name not in used), used


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
        Path(temp_name).replace(path)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def _load_selected(path: Path, known_images: frozenset[str]) -> tuple[str, ...]:
    if not path.exists():
        return ()

    selected: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        image_id = raw_line.strip()
        if not image_id:
            continue
        if Path(image_id).name != image_id or image_id not in known_images:
            raise ValueError(f"Invalid selected image at {path}:{line_number}: {image_id!r}")
        if image_id not in seen:
            selected.append(image_id)
            seen.add(image_id)
    return tuple(selected)


def _load_cursor(path: Path, candidates: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid selector progress file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid selector progress file: {path}")

    if "current_image_id" in payload:
        current_image_id = payload["current_image_id"]
        if current_image_id is None:
            previous_candidates = payload.get("candidate_ids", [])
            if not isinstance(previous_candidates, list) or not all(
                isinstance(image_id, str) for image_id in previous_candidates
            ):
                raise ValueError(f"Invalid candidate_ids in {path}")
            previous_set = frozenset(previous_candidates)
            return next(
                (index for index, image_id in enumerate(candidates) if image_id not in previous_set),
                len(candidates),
            )
        if not isinstance(current_image_id, str):
            raise ValueError(f"Invalid current_image_id in {path}: {current_image_id!r}")
        try:
            return candidates.index(current_image_id)
        except ValueError:
            previous_candidates = payload.get("candidate_ids", [])
            previous_cursor = payload.get("cursor")
            if not isinstance(previous_candidates, list) or not all(
                isinstance(image_id, str) for image_id in previous_candidates
            ):
                raise ValueError(f"Invalid candidate_ids in {path}")
            if (
                not isinstance(previous_cursor, int)
                or isinstance(previous_cursor, bool)
                or not 0 <= previous_cursor <= len(previous_candidates)
            ):
                raise ValueError(f"Invalid cursor in {path}: {previous_cursor!r}")
            reviewed = frozenset(previous_candidates[:previous_cursor])
            return next(
                (index for index, image_id in enumerate(candidates) if image_id not in reviewed),
                len(candidates),
            )

    cursor = payload.get("cursor")
    if not isinstance(cursor, int) or isinstance(cursor, bool) or not 0 <= cursor <= len(candidates):
        raise ValueError(f"Selector cursor out of range in {path}: {cursor!r}")
    return cursor


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(RCES_ROOT))
    except ValueError:
        return str(path)


class SingleInstanceLock:
    """Prevent multiple selector processes from overwriting each other's state."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError("Another test-expansion selector process is already running") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        self.handle = handle

    def release(self) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


class SelectionStore:
    """Thread-safe immutable-state updates for one selector session."""

    def __init__(
        self,
        candidates: tuple[str, ...],
        known_images: frozenset[str],
        output_path: Path,
        progress_path: Path,
    ) -> None:
        if len(candidates) != len(set(candidates)):
            raise ValueError("Candidate image names must be unique")
        if not set(candidates).issubset(known_images):
            raise ValueError("Every candidate must exist in known_images")

        self.candidates = candidates
        self.candidate_set = frozenset(candidates)
        self.known_images = known_images
        self.output_path = output_path
        self.progress_path = progress_path
        loaded_selected = _load_selected(output_path, known_images)
        self.selected = tuple(image_id for image_id in loaded_selected if image_id in self.candidate_set)
        if self.selected != loaded_selected:
            self._persist_selected(self.selected)
        self.cursor = _load_cursor(progress_path, candidates)
        self.csrf_token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()

    def _persist_selected(self, selected: tuple[str, ...]) -> None:
        text = "".join(f"{image_id}\n" for image_id in selected)
        _atomic_write_text(self.output_path, text)

    def _persist_cursor(self, cursor: int) -> None:
        current_image_id = self.candidates[cursor] if cursor < len(self.candidates) else None
        payload = {
            "cursor": cursor,
            "current_image_id": current_image_id,
            "candidate_ids": list(self.candidates),
        }
        _atomic_write_text(self.progress_path, json.dumps(payload, indent=2) + "\n")

    def payload(self) -> dict[str, Any]:
        with self.lock:
            total = len(self.candidates)
            selected_set = frozenset(self.selected)
            if self.cursor >= total:
                return {
                    "done": True,
                    "index": total,
                    "total": total,
                    "selected_count": len(self.selected),
                    "output_path": _display_path(self.output_path),
                    "csrf_token": self.csrf_token,
                }

            image_id = self.candidates[self.cursor]
            return {
                "done": False,
                "image_id": image_id,
                "image_url": f"/image/{quote(image_id)}",
                "selected": image_id in selected_set,
                "selected_count": len(self.selected),
                "index": self.cursor + 1,
                "total": total,
                "can_back": self.cursor > 0,
                "output_path": _display_path(self.output_path),
                "csrf_token": self.csrf_token,
            }

    def apply(self, action: str, image_id: str) -> dict[str, Any]:
        with self.lock:
            if action not in VALID_ACTIONS:
                raise ValueError(f"Unknown action: {action!r}")
            if self.cursor >= len(self.candidates):
                raise ValueError("Selection is already complete")

            current = self.candidates[self.cursor]
            if image_id != current:
                raise ValueError("Stale or invalid image_id")

            updated = self.selected
            if action == "select_next" and image_id not in self.selected:
                updated = (*self.selected, image_id)
            elif action == "deselect":
                updated = tuple(name for name in self.selected if name != image_id)

            if updated != self.selected:
                self._persist_selected(updated)
                self.selected = updated

            if action in {"select_next", "skip_next"}:
                next_cursor = self.cursor + 1
                self._persist_cursor(next_cursor)
                self.cursor = next_cursor
            return self.payload()

    def back(self) -> dict[str, Any]:
        with self.lock:
            previous = max(0, self.cursor - 1)
            if previous != self.cursor:
                self._persist_cursor(previous)
                self.cursor = previous
            return self.payload()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>R-CES test-expansion image selector</title>
<style>
  :root { --bg:#0f1216; --card:#171c22; --line:#303844; --fg:#edf0f4;
          --muted:#9ba6b5; --blue:#2d6cdf; --green:#238636; --red:#9e3f4f; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.45 system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:10; display:flex; gap:14px; align-items:center;
           flex-wrap:wrap; padding:11px 16px; background:var(--card); border-bottom:1px solid var(--line); }
  header strong { color:#7ba6ff; } .muted { color:var(--muted); }
  .progress { margin-left:auto; }
  main { max-width:1500px; margin:auto; padding:16px; }
  .layout { display:grid; grid-template-columns:minmax(0, 3fr) minmax(300px, 1fr); gap:18px; }
  @media (max-width:900px) { .layout { grid-template-columns:1fr; } }
  .photo { min-height:72vh; display:flex; align-items:center; justify-content:center;
           background:#050607; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .photo img { display:block; max-width:100%; max-height:78vh; object-fit:contain; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin-bottom:12px; }
  h1 { margin:0; font-size:1.05rem; } h2 { margin:0 0 8px; font-size:1rem; }
  ul { margin:8px 0; padding-left:1.3em; } code { color:#b8c8e8; overflow-wrap:anywhere; }
  .status { padding:10px 12px; border-radius:8px; background:#222a33; margin-top:10px; }
  .status.selected { background:#173d25; color:#a7e8bb; }
  .actions { display:grid; gap:10px; position:sticky; bottom:12px; }
  button { width:100%; border:1px solid var(--line); border-radius:9px; padding:14px 12px;
           background:#222a33; color:var(--fg); font:inherit; font-weight:650; cursor:pointer; }
  button:hover { border-color:#7ba6ff; } button:disabled { opacity:.35; cursor:not-allowed; }
  .select { background:var(--green); border-color:var(--green); }
  .skip { background:var(--blue); border-color:var(--blue); }
  .deselect { background:#3a2530; border-color:var(--red); }
  .done { text-align:center; padding:18vh 20px; font-size:1.15rem; }
</style></head><body>
<header>
  <h1><strong>R-CES</strong> test-expansion image selector</h1>
  <span class="muted" id="selected"></span>
  <span class="progress" id="progress"></span>
</header>
<main id="app"><div class="done">Loading...</div></main>
<script>
let STATE = null;

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

async function load() {
  const response = await fetch('/api/state', {cache:'no-store'});
  if (!response.ok) throw new Error('state request failed');
  STATE = await response.json();
  render();
}

function render() {
  document.getElementById('selected').textContent = `Selected ${STATE.selected_count}`;
  document.getElementById('progress').textContent = `${STATE.index} / ${STATE.total}`;
  const app = document.getElementById('app');
  app.replaceChildren();
  if (STATE.done) {
    const done = node('div', 'done');
    done.append(node('div', '', 'All candidate images have been reviewed.'));
    done.append(node('div', 'muted', `Selection file: ${STATE.output_path}`));
    app.append(done);
    return;
  }

  const layout = node('div', 'layout');
  const photo = node('div', 'photo');
  const image = node('img');
  image.src = STATE.image_url;
  image.alt = STATE.image_id;
  photo.append(image);

  const side = node('aside');
  const info = node('section', 'card');
  info.append(node('h2', '', STATE.image_id));
  const hint = node('div', 'muted', 'Selection rule: the inspector\\'s primary photographed subject should be clear and judgeable. Select an image only when an edge or background region also contains another inspection subject from one of the four rule categories (Opening Hazard, Edge Hazard, Unsafe Behavior, or Civilized Construction Hazard), and that secondary subject is cropped, occluded, or touching the frame boundary so its rule-relevant state cannot be judged. Do not select images where only the primary subject is present and judgeable, or where no such secondary edge case appears.');
  info.append(hint);
  const status = node('div', STATE.selected ? 'status selected' : 'status', STATE.selected ? 'Current image selected' : 'Current image not selected');
  info.append(status);

  const output = node('section', 'card');
  output.append(node('h2', '', 'Output path'));
  output.append(node('code', '', STATE.output_path));
  output.append(node('div', 'muted', 'Every select or deselect action is written immediately. The browser cursor is stored under cache/ so the script can resume after restart.'));

  const actions = node('section', 'actions');
  actions.append(button('Select and next [S]', 'select', () => act('select_next')));
  actions.append(button('Skip and next [→]', 'skip', () => act('skip_next')));
  const back = button('Previous [←]', '', goBack);
  back.disabled = !STATE.can_back;
  actions.append(back);
  const deselect = button('Deselect current [D]', 'deselect', () => act('deselect'));
  deselect.disabled = !STATE.selected;
  actions.append(deselect);

  side.append(info, output, actions);
  layout.append(photo, side);
  app.append(layout);
}

function button(text, className, onclick) {
  const el = node('button', className, text);
  el.type = 'button';
  el.onclick = onclick;
  return el;
}

async function post(path, body) {
  const response = await fetch(path, {
    method:'POST',
    headers:{'Content-Type':'application/json', 'X-CSRF-Token':STATE.csrf_token},
    body:JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) throw new Error(payload.error || 'request failed');
  STATE = payload.state;
  render();
}

async function act(action) {
  try { await post('/api/action', {action, image_id:STATE.image_id}); }
  catch (error) { alert(`Action failed: ${error.message}`); await load(); }
}

async function goBack() {
  try { await post('/api/back', {}); }
  catch (error) { alert(`Action failed: ${error.message}`); await load(); }
}

document.addEventListener('keydown', event => {
  if (event.repeat || STATE === null || STATE.done) return;
  if (event.key.toLowerCase() === 's') act('select_next');
  else if (event.key.toLowerCase() === 'd' && STATE.selected) act('deselect');
  else if (event.key === 'ArrowRight') act('skip_next');
  else if (event.key === 'ArrowLeft' && STATE.can_back) goBack();
});

load().catch(error => {
  document.getElementById('app').textContent = `Load failed: ${error.message}`;
});
</script></body></html>
"""


def make_handler(store: SelectionStore, image_dir: Path) -> type[BaseHTTPRequestHandler]:
    image_root = image_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; "
                             "style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; "
                             "base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict[str, Any], code: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self._json(store.payload())
                return
            if path.startswith("/image/"):
                image_id = unquote(path.removeprefix("/image/"))
                if image_id not in store.candidate_set or Path(image_id).name != image_id:
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                image_path = (image_root / image_id).resolve()
                if image_path.parent != image_root or not image_path.is_file():
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                self._send(200, image_path.read_bytes(), content_type)
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def _read_json(self) -> dict[str, Any]:
            content_type = self.headers.get("Content-Type", "").partition(";")[0].strip().lower()
            if content_type != "application/json":
                raise ValueError("Content-Type must be application/json")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length") from exc
            if not 0 <= length <= MAX_REQUEST_BYTES:
                raise ValueError("Request body is too large")
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid JSON body") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def do_POST(self) -> None:
            supplied_token = self.headers.get("X-CSRF-Token", "")
            if not hmac.compare_digest(supplied_token, store.csrf_token):
                self._json({"ok": False, "error": "Invalid CSRF token"}, 403)
                return
            try:
                payload = self._read_json()
                path = urlparse(self.path).path
                if path == "/api/action":
                    action = payload.get("action")
                    image_id = payload.get("image_id")
                    if not isinstance(action, str) or not isinstance(image_id, str):
                        raise ValueError("action and image_id must be strings")
                    state = store.apply(action, image_id)
                elif path == "/api/back":
                    state = store.back()
                else:
                    self._json({"ok": False, "error": "Unknown route"}, 404)
                    return
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
                return
            self._json({"ok": True, "state": state})

    return Handler


def bind_server(handler: type[BaseHTTPRequestHandler], preferred_port: int) -> ThreadingHTTPServer:
    ports = [preferred_port, *range(preferred_port + 1, preferred_port + PORT_FALLBACK_COUNT + 1)]
    for port in ports:
        try:
            return ThreadingHTTPServer((HOST, port), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    raise SystemExit(f"No available port in range {ports[0]}-{ports[-1]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the local R-CES test-expansion image selector.")
    parser.add_argument("--port", type=int, default=PORT, help=f"Preferred localhost port (default: {PORT}).")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    instance_lock = SingleInstanceLock(LOCK_PATH)
    try:
        instance_lock.acquire()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    server: ThreadingHTTPServer | None = None
    try:
        candidates, used = discover_candidates(IMAGE_DIR, ACTIVE_PAIRS_PATH)
        known_images = frozenset(
            path.name
            for path in IMAGE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        store = SelectionStore(candidates, known_images, OUTPUT_PATH, PROGRESS_PATH)
        handler = make_handler(store, IMAGE_DIR)
        server = bind_server(handler, args.port)
        bound_port = server.server_address[1]

        print(f"Images: {len(known_images)} total | {len(used)} active-pair used | {len(candidates)} candidates")
        print(f"Already selected: {len(store.selected)}")
        print(f"Writes: {OUTPUT_PATH.relative_to(RCES_ROOT)}")
        print(f"Open: http://localhost:{bound_port}/")
        print("Ctrl-C to stop; selection and cursor are persisted after every action.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    finally:
        if server is not None:
            server.server_close()
        instance_lock.release()


if __name__ == "__main__":
    main()
