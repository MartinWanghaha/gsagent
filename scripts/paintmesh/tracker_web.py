#!/usr/bin/env python3
"""Run-local PaintMesh point-prompt SAM + DeAOT UI (no Gradio dependency)."""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from virtual_render_io import sha256, verify_tracking_render

NAMES = [f"{i:05d}.png" for i in range(30)]


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def overlay(frame, mask, points=()):
    result = frame.copy()
    if mask is not None:
        selected = mask != 0
        result[selected] = (result[selected] * .55 + np.array([40, 210, 155]) * .45).astype(np.uint8)
    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)
    radius = max(2, min(frame.shape[:2]) // 90)
    for x, y, label in points:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="#26dbac" if label else "#ff536c", outline="white")
    return np.asarray(image)


class Models:
    """Load only point-SAM and DeAOT; no text detector or segment-everything."""
    def __init__(self):
        self.segmentor = None
        self.tracker = None

    def segment(self, frame, points):
        if self.segmentor is None:
            from model_args import sam_args
            from tool.segmentor import Segmentor
            self.segmentor = Segmentor(copy.deepcopy(sam_args))
        return self.segmentor.segment_with_click(
            frame, np.asarray([p[:2] for p in points]), np.asarray([p[2] for p in points]), True
        )

    def propagate(self, frames, mask):
        import torch
        from model_args import aot_args
        from aot_tracker import get_aot
        if self.tracker is None:
            self.tracker = get_aot(copy.deepcopy(aot_args))
        self.tracker.restart()
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.tracker.add_reference_frame(frames[0], mask, 1, 0)
            yield mask
            for frame in frames[1:]:
                predicted = self.tracker.track(frame)
                self.tracker.update_memory(predicted)
                yield predicted.squeeze().detach().cpu().numpy().astype(np.uint8)


class Workspace:
    def __init__(self, archive, results, models=None):
        self.archive = Path(archive).resolve(strict=True)
        self.run = self.archive.parent
        self.session_path = self.run / "tracking_session.json"
        self.session_bytes = self.session_path.read_bytes()
        self.session = json.loads(self.session_bytes)
        self.digest = sha256(self.archive)
        self.verify()
        self.results = Path(results).absolute()
        if self.results.is_symlink() or self.results.resolve() != self.run / "results":
            raise ValueError("results must be the current tracker's run-local results directory")
        self.destination = self.results / "images" / "images_masks"
        self.frames = []
        with zipfile.ZipFile(self.archive) as archive_file:
            if sorted(archive_file.namelist()) != NAMES:
                raise ValueError("images.zip must contain exactly 00000.png..00029.png at its root")
            for name in NAMES:
                with Image.open(io.BytesIO(archive_file.read(name))) as image:
                    self.frames.append(np.asarray(image.convert("RGB")))
        if len({frame.shape for frame in self.frames}) != 1:
            raise ValueError("virtual sequence dimensions must match")
        camera_path = Path(self.session["input_cameras"]["path"])
        cameras = json.loads(camera_path.read_text())["cameras"]
        if [c["image_name"] + ".png" for c in cameras] != NAMES:
            raise ValueError("camera frame set differs from archive")
        for frame, camera in zip(self.frames, cameras):
            if frame.shape[:2] != (camera["image_height"], camera["image_width"]):
                raise ValueError("camera dimensions differ from archive")
        self.models = models or Models()
        self.points = []
        self.mask = None
        self.masks = []
        self.phase = "ready"
        self.message = "已加载 30 帧。请在首帧点击需要补全的区域。"
        self.lock = threading.RLock()
        self.busy = threading.Lock()
        self.version = 0

    def verify(self):
        if self.session_path.read_bytes() != self.session_bytes:
            raise ValueError("tracking session changed; restart Stage 5")
        if (self.session.get("kind") != "paintmesh-tracking-session"
                or self.session.get("status") != "in_progress"
                or self.session.get("complete") is not False
                or self.session.get("expected_masks") != NAMES):
            raise ValueError("Stage 5 requires an active tracking session; completed masks are not overwritten")
        for key, path in (("input_archive", self.archive), ("input_cameras", Path(self.session["input_cameras"]["path"]))):
            record = self.session[key]
            stat = path.stat()
            if (Path(record["path"]).resolve() != path.resolve() or record["sha256"] != sha256(path)
                    or record["size_bytes"] != stat.st_size or record["mtime_ns"] != stat.st_mtime_ns):
                raise ValueError(f"{key} changed after tracking session started")
        verify_tracking_render(self.session)

    def state(self):
        with self.lock:
            return dict(phase=self.phase, message=self.message, count=len(self.masks), total=30,
                        points=len(self.points), version=self.version, run=self.run.parent.name,
                        width=self.frames[0].shape[1], height=self.frames[0].shape[0],
                        can_track=self.mask is not None and bool(self.mask.any()) and not bool(self.mask.all()))

    def preview(self, index=0):
        if not 0 <= index < 30:
            raise ValueError("frame index out of range")
        with self.lock:
            mask = self.masks[index] if index < len(self.masks) else self.mask if index == 0 else None
            return png(overlay(self.frames[index], mask, self.points if index == 0 else ()))

    def annotate(self, action, point=None):
        if not self.busy.acquire(blocking=False):
            raise ValueError("正在处理，请稍候")
        try:
            with self.lock:
                if self.phase in {"tracking", "done"}:
                    raise ValueError("跟踪开始后不能修改首帧；已有结果不会被覆盖")
                points = list(self.points)
            if action == "clear":
                points = []
            elif action == "undo":
                points = points[:-1]
            elif action == "point":
                if not isinstance(point, list) or len(point) != 3 or any(type(v) is not int for v in point):
                    raise ValueError("point must be integer [x,y,label]")
                x, y, label = point
                h, w = self.frames[0].shape[:2]
                if not (0 <= x < w and 0 <= y < h and label in (0, 1)) or len(points) >= 100:
                    raise ValueError("invalid point or too many prompts")
                if label == 0 and not any(p[2] for p in points):
                    raise ValueError("请先添加一个补全区域的正点")
                points.append(point)
            else:
                raise ValueError("unknown annotation action")
            mask = None
            if any(p[2] for p in points):
                mask = np.asarray(self.models.segment(self.frames[0], points)) != 0
                if mask.shape != self.frames[0].shape[:2]:
                    raise ValueError("SAM returned incorrect dimensions")
            with self.lock:
                self.points, self.mask = points, mask
                self.phase = "ready"
                self.version += 1
                self.message = "绿色为待补全区域。可继续加点/排除，确认后点击 Start Tracking。" if mask is not None else "请点击需要补全的区域。"
            return self.state()
        finally:
            self.busy.release()

    def start(self):
        if not self.busy.acquire(blocking=False):
            raise ValueError("正在处理，请稍候")
        try:
            with self.lock:
                if self.phase == "done" or not self.state()["can_track"]:
                    raise ValueError("请先得到非空、非全图的首帧分割")
            self.verify()
            self.results.mkdir(parents=True, exist_ok=True)
            parent = self.destination.parent
            if parent.is_symlink() or self.destination.exists() or self.destination.is_symlink():
                raise ValueError("已有 masks，拒绝覆盖；请使用新的 removal run")
            parent.mkdir(exist_ok=True)
            with self.lock:
                self.phase, self.message, self.masks = "tracking", "正在跟踪，请勿关闭服务…", []
            threading.Thread(target=self._track, daemon=True).start()
        except Exception:
            self.busy.release()
            raise
        return self.state()

    def _track(self):
        try:
            # Publish only a complete sequence; an interrupted attempt cannot
            # be mistaken for complete masks by the shell's session validator.
            with tempfile.TemporaryDirectory(prefix=".paintmesh-masks-", dir=self.destination.parent) as temporary:
                staged = Path(temporary) / "images_masks"
                staged.mkdir()
                for index, mask in enumerate(self.models.propagate(self.frames, self.mask.astype(np.uint8))):
                    if index >= 30:
                        raise ValueError("tracker returned too many frames")
                    mask = np.asarray(mask)
                    if mask.shape != self.frames[index].shape[:2] or not np.isfinite(mask).all():
                        raise ValueError(f"invalid tracker mask at frame {index}")
                    mask = (mask != 0).astype(np.uint8)
                    if not mask.any() or mask.all():
                        raise ValueError(f"frame {index:05d} mask is empty/full; adjust first-frame prompts and retry")
                    Image.fromarray(mask, mode="L").save(staged / NAMES[index])
                    with self.lock:
                        self.masks.append(mask)
                        self.version += 1
                        self.message = f"已跟踪 {index+1}/30 帧"
                if len(self.masks) != 30:
                    raise ValueError("tracker returned fewer than 30 frames")
                self.verify()
                if self.destination.exists():
                    raise ValueError("masks appeared during tracking; refusing overwrite")
                staged.rename(self.destination)
            with self.lock:
                self.phase = "done"
                self.message = "30 帧 masks 已保存。检查预览后，点击「完成并返回流水线」提交校验。"
        except Exception as exc:
            with self.lock:
                self.phase = "error"
                self.message = str(exc)
                self.masks = []
                self.version += 1
        finally:
            self.busy.release()


def make_server(workspace, host="127.0.0.1", port=7860):
    token = secrets.token_urlsafe(32)
    page = Path(__file__).with_name("tracker_web.html").read_text().replace("__TOKEN__", token).encode()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, content_type="application/json"):
            if isinstance(data, dict):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            return secrets.compare_digest(self.headers.get("X-PaintMesh-Token", ""), token)

        def do_GET(self):
            # Reject DNS rebinding; loopback-only binding is the default.
            if self.headers.get("Host") not in {f"{host}:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                return self.reply(403, {"error": "invalid host"})
            url = urlsplit(self.path)
            if url.path == "/":
                return self.reply(200, page, "text/html; charset=utf-8")
            if not self.authorized():
                return self.reply(403, {"error": "invalid session token"})
            try:
                if url.path == "/state":
                    return self.reply(200, workspace.state())
                if url.path == "/preview":
                    index = int(parse_qs(url.query).get("frame", ["0"])[0])
                    return self.reply(200, workspace.preview(index), "image/png")
                self.reply(404, {"error": "not found"})
            except (ValueError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

        def do_POST(self):
            if not self.authorized():
                return self.reply(403, {"error": "invalid session token"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("invalid request size")
                value = json.loads(self.rfile.read(length))
                action = value["action"]
                if self.path != "/action":
                    return self.reply(404, {"error": "not found"})
                if action == "track":
                    result = workspace.start()
                elif action == "finish":
                    if workspace.state()["phase"] != "done":
                        raise ValueError("tracking is not complete")
                    workspace.verify()
                    result = {"message": "正在返回流水线，终端将验证并提交 tracking session。"}
                    self.reply(200, result)
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                else:
                    result = workspace.annotate(action, value.get("point"))
                self.reply(200, result)
            except Exception as exc:
                self.reply(400, {"error": str(exc)})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=os.environ.get("TRACKER_IMAGE_SEQUENCE"))
    parser.add_argument("--results", type=Path, default=os.environ.get("TRACKING_RESULTS_ROOT"))
    args = parser.parse_args()
    if args.archive is None or args.results is None:
        parser.error("launch from run_remove.sh Stage 5, or provide --archive and --results")
    tracker_root = Path(__file__).resolve().parents[2] / "submodules/Inpaint360GS/Segment-and-Track-Anything"
    sys.path.insert(0, str(tracker_root))
    os.chdir(tracker_root)  # Upstream checkpoint and AOT config paths are relative.
    workspace = Workspace(args.archive, args.results)
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("PaintMesh UI is loopback-only; use SSH port forwarding for remote access")
    server = make_server(workspace, host, int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    print(f"PaintMesh masks: http://{host}:{server.server_port} — {workspace.run.parent.name}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
