"""Tests for the R-CES test-expansion image selector."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


RCES_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = RCES_ROOT / "Script" / "select_test_images_web.py"


def load_module():
    spec = importlib.util.spec_from_file_location("select_test_images_web", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CandidateDiscoveryTest(unittest.TestCase):
    def test_active_pair_images_and_non_images_are_excluded(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            for name in ("img_0001.jpg", "img_0002.PNG", "img_0003.webp"):
                (image_dir / name).write_bytes(b"image")
            (image_dir / "notes.txt").write_text("not an image", encoding="utf-8")
            pairs_path = root / "pairs.jsonl"
            pairs_path.write_text(
                json.dumps({"image_id": "img_0002.PNG"}) + "\n",
                encoding="utf-8",
            )

            candidates, used = module.discover_candidates(image_dir, pairs_path)

            self.assertEqual(("img_0001.jpg", "img_0003.webp"), candidates)
            self.assertEqual(frozenset({"img_0002.PNG"}), used)


class SelectionStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.output_path = root / "splits" / "test_expansion_selected_images.txt"
        self.progress_path = root / "cache" / "selector_state.json"
        self.store = self.module.SelectionStore(
            candidates=("img_0001.jpg", "img_0002.jpg", "img_0003.jpg"),
            known_images=frozenset({"img_0001.jpg", "img_0002.jpg", "img_0003.jpg"}),
            output_path=self.output_path,
            progress_path=self.progress_path,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_select_is_idempotent_and_advances_with_immediate_persistence(self) -> None:
        first = self.store.apply("select_next", "img_0001.jpg")
        self.store.back()
        second = self.store.apply("select_next", "img_0001.jpg")

        self.assertEqual(["img_0001.jpg"], self.output_path.read_text(encoding="utf-8").splitlines())
        self.assertEqual(1, first["selected_count"])
        self.assertEqual(1, second["selected_count"])
        self.assertEqual("img_0002.jpg", second["image_id"])
        progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
        self.assertEqual(1, progress["cursor"])
        self.assertEqual("img_0002.jpg", progress["current_image_id"])

    def test_deselect_removes_the_name_without_mutating_other_selections(self) -> None:
        self.store.apply("select_next", "img_0001.jpg")
        self.store.apply("select_next", "img_0002.jpg")
        self.store.back()

        state = self.store.apply("deselect", "img_0002.jpg")

        self.assertEqual(["img_0001.jpg"], self.output_path.read_text(encoding="utf-8").splitlines())
        self.assertFalse(state["selected"])
        self.assertEqual("img_0002.jpg", state["image_id"])

    def test_invalid_or_stale_requests_do_not_change_output(self) -> None:
        self.store.apply("select_next", "img_0001.jpg")
        before = self.output_path.read_bytes()

        with self.assertRaises(ValueError):
            self.store.apply("select_next", "../img_0002.jpg")
        with self.assertRaises(ValueError):
            self.store.apply("delete_everything", "img_0002.jpg")
        with self.assertRaises(ValueError):
            self.store.apply("select_next", "img_0003.jpg")

        self.assertEqual(before, self.output_path.read_bytes())

    def test_restart_resumes_from_persisted_cursor(self) -> None:
        self.store.apply("skip_next", "img_0001.jpg")

        resumed = self.module.SelectionStore(
            candidates=("img_0001.jpg", "img_0002.jpg", "img_0003.jpg"),
            known_images=frozenset({"img_0001.jpg", "img_0002.jpg", "img_0003.jpg"}),
            output_path=self.output_path,
            progress_path=self.progress_path,
        )

        self.assertEqual("img_0002.jpg", resumed.payload()["image_id"])

    def test_restart_tracks_current_image_when_candidates_are_reordered(self) -> None:
        self.store.apply("skip_next", "img_0001.jpg")

        resumed = self.module.SelectionStore(
            candidates=("img_0000.jpg", "img_0001.jpg", "img_0002.jpg", "img_0003.jpg"),
            known_images=frozenset({"img_0000.jpg", "img_0001.jpg", "img_0002.jpg", "img_0003.jpg"}),
            output_path=self.output_path,
            progress_path=self.progress_path,
        )

        self.assertEqual("img_0002.jpg", resumed.payload()["image_id"])

    def test_restart_moves_to_next_unseen_image_when_current_was_removed(self) -> None:
        self.store.apply("skip_next", "img_0001.jpg")

        resumed = self.module.SelectionStore(
            candidates=("img_0001.jpg", "img_0003.jpg"),
            known_images=frozenset({"img_0001.jpg", "img_0003.jpg"}),
            output_path=self.output_path,
            progress_path=self.progress_path,
        )

        self.assertEqual("img_0003.jpg", resumed.payload()["image_id"])

    def test_startup_removes_selected_images_that_are_no_longer_candidates(self) -> None:
        self.output_path.parent.mkdir(parents=True)
        self.output_path.write_text("img_0001.jpg\nimg_0003.jpg\n", encoding="utf-8")

        store = self.module.SelectionStore(
            candidates=("img_0001.jpg", "img_0002.jpg"),
            known_images=frozenset({"img_0001.jpg", "img_0002.jpg", "img_0003.jpg"}),
            output_path=self.output_path,
            progress_path=self.progress_path,
        )

        self.assertEqual(("img_0001.jpg",), store.selected)
        self.assertEqual(["img_0001.jpg"], self.output_path.read_text(encoding="utf-8").splitlines())


class SingleInstanceLockTest(unittest.TestCase):
    def test_second_process_lock_is_rejected(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "selector.lock"
            first = module.SingleInstanceLock(lock_path)
            first.acquire()
            try:
                with self.assertRaises(RuntimeError):
                    module.SingleInstanceLock(lock_path).acquire()
            finally:
                first.release()


class SelectorHttpIntegrationTest(unittest.TestCase):
    def test_api_selects_image_and_rejects_unsafe_requests(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "img_0001.jpg").write_bytes(b"fake-jpeg")
            output_path = root / "splits" / "selected.txt"
            store = module.SelectionStore(
                candidates=("img_0001.jpg",),
                known_images=frozenset({"img_0001.jpg"}),
                output_path=output_path,
                progress_path=root / "cache" / "progress.json",
            )
            try:
                server = module.ThreadingHTTPServer(
                    ("127.0.0.1", 0),
                    module.make_handler(store, image_dir),
                )
            except PermissionError as exc:
                self.skipTest(f"local sockets are unavailable in this sandbox: {exc}")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urlopen(f"{base_url}/api/state") as response:
                    state = json.load(response)

                missing_token = Request(
                    f"{base_url}/api/action",
                    data=json.dumps({"action": "select_next", "image_id": "img_0001.jpg"}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as csrf_error:
                    urlopen(missing_token)
                self.assertEqual(403, csrf_error.exception.code)

                select = Request(
                    f"{base_url}/api/action",
                    data=json.dumps({"action": "select_next", "image_id": "img_0001.jpg"}).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "X-CSRF-Token": state["csrf_token"],
                    },
                    method="POST",
                )
                with urlopen(select) as response:
                    result = json.load(response)
                self.assertTrue(result["ok"])
                self.assertEqual(["img_0001.jpg"], output_path.read_text(encoding="utf-8").splitlines())

                with self.assertRaises(HTTPError) as traversal_error:
                    urlopen(f"{base_url}/image/%2E%2E%2Fselected.txt")
                self.assertEqual(404, traversal_error.exception.code)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
