import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracker_web import Workspace, NAMES, make_server, png
from virtual_render_io import sha256


class FakeModels:
    def segment(self, frame, points):
        mask = np.zeros(frame.shape[:2], np.uint8)
        for x, y, label in points:
            mask[y, x] = label
        return mask

    def propagate(self, frames, mask):
        for frame in frames:
            yield mask.copy()


def fixture(root, models=None):
    archive = root / "images.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name in NAMES:
            stream.writestr(name, png(np.full((24, 32, 3), 100, np.uint8)))
    camera = root / "virtual_cameras.json"
    camera.write_text(json.dumps(dict(cameras=[dict(image_name=n[:-4], image_height=24, image_width=32) for n in NAMES])))
    def record(path):
        return dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    session = dict(kind="paintmesh-tracking-session", status="in_progress", complete=False,
                   expected_masks=NAMES, input_archive=record(archive), input_cameras=record(camera))
    (root / "tracking_session.json").write_text(json.dumps(session))
    return Workspace(archive, root / "results", models or FakeModels())


def wait_done(workspace):
    deadline = time.monotonic() + 15
    while workspace.state()["phase"] == "tracking" and time.monotonic() < deadline:
        time.sleep(.01)
    return workspace.state()


class TrackerWebTests(unittest.TestCase):
    def test_click_undo_clear_and_exact_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = fixture(Path(tmp))
            self.assertFalse(w.state()["can_track"])
            with self.assertRaises(ValueError): w.annotate("point", [1, 2, 0])
            w.annotate("point", [1, 2, 1]); w.annotate("point", [3, 4, 1])
            w.annotate("undo"); self.assertEqual(w.mask.sum(), 1)
            w.annotate("clear"); self.assertIsNone(w.mask)
            w.annotate("point", [1, 2, 1]); w.start()
            self.assertEqual(wait_done(w)["phase"], "done")
            self.assertEqual(sorted(p.name for p in w.destination.iterdir()), NAMES)
            for path in w.destination.iterdir():
                with Image.open(path) as mask:
                    self.assertEqual(mask.mode, "L"); self.assertEqual(mask.size, (32, 24))
            with self.assertRaises(ValueError): w.annotate("clear")
            with self.assertRaises(ValueError): w.start()

    def test_existing_masks_and_changed_session_are_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = fixture(Path(tmp)); w.annotate("point", [1, 1, 1])
            w.destination.mkdir(parents=True)
            sentinel = w.destination / "keep.txt"; sentinel.write_text("keep")
            with self.assertRaisesRegex(ValueError, "已有 masks"): w.start()
            self.assertEqual(sentinel.read_text(), "keep")
            w.session_path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "session changed"): w.verify()

    def test_partial_tracking_never_publishes(self):
        class Broken(FakeModels):
            def propagate(self, frames, mask):
                yield mask
                raise RuntimeError("test model failure")
        with tempfile.TemporaryDirectory() as tmp:
            w = fixture(Path(tmp), Broken()); w.annotate("point", [1, 1, 1]); w.start()
            self.assertEqual(wait_done(w)["phase"], "error")
            self.assertFalse(w.destination.exists())
            self.assertIn("test model failure", w.message)
            w.models = FakeModels(); w.start()
            self.assertEqual(wait_done(w)["phase"], "done")

    def test_bad_coordinates_and_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = fixture(Path(tmp))
            for point in ([32, 1, 1], [-1, 2, 1], [1, 1, 3], [1.2, 1, 1]):
                with self.assertRaises(ValueError): w.annotate("point", point)
            with zipfile.ZipFile(w.archive, "a") as stream: stream.writestr("../escape", "x")
            with self.assertRaises(ValueError): w.verify()

    def test_http_page_security_actions_and_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = fixture(Path(tmp)); server = make_server(w, port=0)
            worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
            opener = build_opener(ProxyHandler({}))
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                page = opener.open(url).read().decode()
                self.assertIn("Start Tracking", page)
                token = re.search("const token='([^']+)'", page)[1]
                with self.assertRaises(HTTPError): opener.open(url + "/state")
                def action(name, point=None):
                    req = Request(url + "/action", data=json.dumps(dict(action=name, point=point)).encode(), headers={"X-PaintMesh-Token":token})
                    return json.loads(opener.open(req).read())
                with self.assertRaises(HTTPError): action("finish")
                action("point", [2, 3, 1]); action("track")
                self.assertEqual(wait_done(w)["phase"], "done")
                preview = opener.open(Request(url + "/preview?frame=29", headers={"X-PaintMesh-Token":token})).read()
                self.assertEqual(Image.open(io.BytesIO(preview)).size, (32,24))
                action("finish"); worker.join(3); self.assertFalse(worker.is_alive())
            finally:
                server.shutdown(); server.server_close()


if __name__ == "__main__": unittest.main()
