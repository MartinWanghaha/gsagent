import io
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracker_web import Workspace, make_server, png, print_access, public_host, request_host
from virtual_render_io import sha256
from virtual_render_io import camera_contract
from types import SimpleNamespace

NAMES = [f"{i:05d}.png" for i in range(30)]  # Legacy fixture, not a UI contract.


class FakeModels:
    def segment(self, frame, points):
        mask = np.zeros(frame.shape[:2], np.uint8)
        for x, y, label in points:
            mask[y, x] = label
        return mask

    def propagate(self, frames, mask):
        for frame in frames:
            yield mask.copy()


def fixture(root, models=None, count=30):
    names = [f"{i:05d}.png" for i in range(count)]
    archive = root / "images.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name in names:
            stream.writestr(name, png(np.full((24, 32, 3), 100, np.uint8)))
    camera = root / "virtual_cameras.json"
    cameras = [SimpleNamespace(image_name=n[:-4], image_height=24, image_width=32,
        R=np.eye(3), T=np.zeros(3), FoVx=1., FoVy=1., znear=.01, zfar=100., trans=np.zeros(3), scale=1.) for n in names]
    camera_contract().write_virtual_camera_manifest(camera, cameras, iteration=2000, circle_radius=1.)
    def record(path):
        return dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    session = dict(kind="paintmesh-tracking-session", status="in_progress", complete=False,
                   expected_masks=names, input_archive=record(archive), input_cameras=record(camera))
    (root / "tracking_session.json").write_text(json.dumps(session))
    return Workspace(archive, root / "results", models or FakeModels())


def wait_done(workspace):
    deadline = time.monotonic() + 15
    while workspace.state()["phase"] == "tracking" and time.monotonic() < deadline:
        time.sleep(.01)
    return workspace.state()


class TrackerWebTests(unittest.TestCase):
    def test_browser_addresses_and_ssh_hint(self):
        with patch.dict(os.environ, {"SSH_CONNECTION":"10.192.54.248 55307 10.193.25.175 1021"}, clear=True):
            self.assertEqual(public_host("0.0.0.0"), "10.193.25.175")
            self.assertEqual(public_host("127.0.0.1"), "127.0.0.1")
            server = SimpleNamespace(public_host="10.193.25.175", server_port=7860,
                                     require_page_token=True, access_token="example-session-key")
            output = io.StringIO()
            with patch("tracker_web.getpass.getuser", return_value="member"), redirect_stdout(output):
                print_access(server, "0.0.0.0")
            self.assertIn("http://10.193.25.175:7860/?token=example-session-key", output.getvalue())
            self.assertIn("-L 18760:127.0.0.1:7860 member@10.193.25.175", output.getvalue())
            self.assertIn("http://127.0.0.1:18760/?token=example-session-key", output.getvalue())
        for value in ("localhost:18760", "127.0.0.1:18761", "[::1]:18762"):
            self.assertIsNotNone(request_host(value))
        for value in ("", "user@localhost:18760", "localhost/path", "localhost:bad", "localhost:65536"):
            self.assertIsNone(request_host(value))

    def test_non30_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = fixture(Path(tmp), count=7)
            self.assertEqual(w.state()["total"], 7)
            w.annotate("point", [1, 1, 1]); w.start(); wait_done(w)
            self.assertEqual(w.state()["phase"], "done")
            self.assertEqual(sorted(p.name for p in w.destination.glob("*.png")), w.names)
            with self.assertRaises(ValueError):
                w.preview(7)

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
                # SSH may forward a different local port from the server's bind port.
                forwarded = {"Host":"localhost:18761", "X-PaintMesh-Token":token}
                self.assertEqual(opener.open(Request(url, headers=forwarded)).status, 200)
                self.assertEqual(json.loads(opener.open(Request(url + "/state", headers=forwarded)).read())["total"], 30)
                self.assertEqual(opener.open(Request(url, headers=forwarded, method="HEAD")).read(), b"")
                with self.assertRaises(HTTPError) as rejected:
                    opener.open(Request(url, headers={"Host":"untrusted.example:18761"}))
                self.assertEqual(rejected.exception.code, 403)
                with self.assertRaises(HTTPError): opener.open(url + "/state")
                def action(name, point=None):
                    req = Request(url + "/action", data=json.dumps(dict(action=name, point=point)).encode(), headers=forwarded)
                    return json.loads(opener.open(req).read())
                with self.assertRaises(HTTPError): action("finish")
                action("point", [2, 3, 1]); action("track")
                self.assertEqual(wait_done(w)["phase"], "done")
                preview = opener.open(Request(url + "/preview?frame=29", headers={"X-PaintMesh-Token":token})).read()
                self.assertEqual(Image.open(io.BytesIO(preview)).size, (32,24))
                action("finish"); worker.join(3); self.assertFalse(worker.is_alive())
            finally:
                server.shutdown(); server.server_close()

    def test_direct_server_access_requires_session_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = fixture(Path(tmp), count=7)
            server = make_server(w, host="0.0.0.0", port=0, advertised_host="10.193.25.175")
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            opener = build_opener(ProxyHandler({}))
            url = f"http://127.0.0.1:{server.server_port}"
            browser = {"Host":f"10.193.25.175:{server.server_port}"}
            try:
                for path in ("/", "/?token=wrong", "/?token=%E9%94%99", "/state", "/preview?frame=0"):
                    with self.subTest(path=path), self.assertRaises(HTTPError) as rejected:
                        opener.open(Request(url + path, headers=browser))
                    self.assertEqual(rejected.exception.code, 403)
                with opener.open(Request(url + "/?token=" + server.access_token, headers=browser)) as response:
                    self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")
                    self.assertIn("Start Tracking", response.read().decode())
                authenticated = {**browser, "X-PaintMesh-Token":server.access_token}
                for host in ("10.193.25.175:18760", "127.0.0.1:18761"):
                    headers = {**authenticated, "Host":host}
                    state = json.loads(opener.open(Request(url + "/state", headers=headers)).read())
                    self.assertEqual(state["total"], 7)
                with self.assertRaises(HTTPError) as rejected:
                    opener.open(Request(url + "/action", data=b'{"action":"clear"}', headers=browser))
                self.assertEqual(rejected.exception.code, 403)
                request = Request(url + "/action", data=b'{"action":"point","point":[2,3,1]}', headers=authenticated)
                self.assertTrue(json.loads(opener.open(request).read())["can_track"])
                preview = opener.open(Request(url + "/preview?frame=0", headers=authenticated)).read()
                self.assertEqual(Image.open(io.BytesIO(preview)).size, (32,24))
                for path in ("/", "/action"):
                    request = Request(url + path, data=b'{}' if path == "/action" else None,
                                      headers={**authenticated, "Host":"untrusted.example"})
                    with self.assertRaises(HTTPError) as rejected:
                        opener.open(request)
                    self.assertEqual(rejected.exception.code, 403)
            finally:
                server.shutdown(); server.server_close(); worker.join(3)


if __name__ == "__main__": unittest.main()
