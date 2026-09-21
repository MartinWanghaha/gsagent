"""Shared, CPU-only artifact contract for the two virtual-render backends."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

BACKENDS = ("inpaint360gs", "edgs-pgsr")
KIND = "paintmesh-virtual-render"


@lru_cache(maxsize=1)
def camera_contract():
    # Load by filename: EDGS and Inpaint360GS both have a package called utils.
    path = Path(__file__).resolve().parents[2] / "submodules/Inpaint360GS/utils/virtual_camera_manifest.py"
    spec = importlib.util.spec_from_file_location("paintmesh_camera_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_camera_manifest(path):
    camera_contract().load_virtual_camera_manifest(path)
    return read_json(path)


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(
        path,
        lambda stream: stream.write(
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        ),
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_backend(root, backend):
    if backend not in BACKENDS:
        raise ValueError(f"unknown virtual renderer: {backend}")
    root = Path(root)
    marker = root / "render_backend.json"
    if marker.exists():
        if read_json(marker).get("backend") != backend:
            raise ValueError(
                "virtual renderer changed; use a new RUN_NAME/removal workspace"
            )
    elif backend != "inpaint360gs" and root.exists() and any(root.glob("ours*")):
        raise ValueError(
            "legacy native virtual renders exist; use a new RUN_NAME for edgs-pgsr"
        )


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def decode(package, backend, alpha_min=0.01):
    """Keep native depth semantics; invalidate PGSR depth/normal independently."""
    if backend not in BACKENDS:
        raise ValueError(backend)
    if not math.isfinite(alpha_min) or not 0 <= alpha_min <= 1:
        raise ValueError("alpha_min must be finite in [0,1]")
    rgb = np.moveaxis(array(package["render"]), 0, -1)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("RGB must be finite [3,H,W]")
    shape = rgb.shape[:2]
    depth = array(package["plane_depth" if backend == "edgs-pgsr" else "depth_3dgs"])
    alpha = array(package["rendered_alpha" if backend == "edgs-pgsr" else "alpha"])
    if depth.shape == (1, *shape):
        depth = depth[0]
    if alpha.shape == (1, *shape):
        alpha = alpha[0]
    if depth.shape != shape or alpha.shape != shape:
        raise ValueError("depth/alpha dimensions do not match RGB")
    if (
        not np.isfinite(alpha).all()
        or (alpha < -1e-6).any()
        or (alpha > 1.000001).any()
    ):
        raise ValueError("alpha must be finite in [0,1]")
    if backend == "inpaint360gs" and (
        not np.isfinite(depth).all() or (depth < 0).any()
    ):
        raise ValueError("native depth must be finite and nonnegative")
    depth_valid = np.isfinite(depth) & (depth > 0)
    result = {
        "rgb_raw": np.clip(rgb, 0, 1),
        "depth": np.where(depth_valid, depth, 0),
        "alpha": alpha,
    }
    if backend == "edgs-pgsr":
        normal = np.moveaxis(array(package["rendered_normal"]), 0, -1)
        if normal.shape != rgb.shape:
            raise ValueError("normal dimensions do not match RGB")
        normal = normal / np.maximum(alpha[..., None], 1e-6)
        length = np.linalg.norm(normal, axis=-1)
        valid = (
            depth_valid
            & (alpha >= alpha_min)
            & np.isfinite(normal).all(-1)
            & np.isfinite(length)
            & (length > 1e-6)
        )
        unit = np.zeros_like(normal)
        unit[valid] = normal[valid] / length[valid, None]
        result.update(normal=unit, normal_valid=valid)
    return result


class FrameWriter:
    def __init__(self, root, backend, cameras, inputs, alpha_min=0.01):
        self.root = Path(root)
        self.backend = backend
        self.alpha_min = alpha_min
        self.shapes = {
            c["image_name"]: (c["image_height"], c["image_width"])
            for c in cameras["cameras"]
        }
        self.payload = {
            "schema_version": 1,
            "kind": KIND,
            "complete": False,
            "status": "in_progress",
            "backend": backend,
            "inputs": inputs,
            "camera_artifact_id": cameras["artifact_id"],
            "capabilities": ["rgb", "depth", "alpha"]
            + (["normal"] if backend == "edgs-pgsr" else ["objects"]),
            "depth_kind": "plane_z" if backend == "edgs-pgsr" else "native_depth_3dgs",
            "depth_unit": "scene",
            "normal_source": "rendered_normal" if backend == "edgs-pgsr" else None,
            "normal_space": "camera" if backend == "edgs-pgsr" else None,
            "normal_axes": (
                "+x right,+y down,+z forward" if backend == "edgs-pgsr" else None
            ),
            "normal_orientation": "toward_camera" if backend == "edgs-pgsr" else None,
            "alpha_min": alpha_min,
            "frames": {},
        }
        path = self.root / "render_manifest.json"
        if path.exists() and read_json(path).get("backend") != backend:
            raise ValueError("output directory belongs to another renderer")
        write_json(path, self.payload)

    def write(self, view, package):
        name = view.image_name
        if name not in self.shapes or name in self.payload["frames"]:
            raise ValueError(f"unexpected or duplicate virtual frame {name}")
        data = decode(package, self.backend, self.alpha_min)
        if data["depth"].shape != self.shapes[name]:
            raise ValueError(f"{name}: rendered size differs from camera manifest")
        outputs = {}
        for modality, value in data.items():
            suffix = ".png" if modality == "normal_valid" else ".npy"
            relative = f"{modality}/{name}{suffix}"
            if modality == "normal_valid":
                atomic_write(
                    self.root / relative,
                    lambda f: Image.fromarray(value.astype(np.uint8) * 255).save(
                        f, format="PNG"
                    ),
                )
            else:
                atomic_write(
                    self.root / relative,
                    lambda f: np.save(f, value.astype(np.float32), allow_pickle=False),
                )
            outputs[relative] = sha256(self.root / relative)
        previews = {"renders": np.floor(data["rgb_raw"] * 255 + 0.5).astype(np.uint8)}
        if "normal" in data:
            previews["normal_vis"] = np.where(
                data["normal_valid"][..., None],
                np.clip((data["normal"] + 1) * 127.5, 0, 255),
                0,
            ).astype(np.uint8)
        for directory, value in previews.items():
            relative = f"{directory}/{name}.png"
            atomic_write(
                self.root / relative,
                lambda f: Image.fromarray(value).save(f, format="PNG"),
            )
            outputs[relative] = sha256(self.root / relative)
        self.payload["frames"][name] = {
            "shape": list(self.shapes[name]),
            "outputs": outputs,
        }

    def finish(self):
        if set(self.payload["frames"]) != set(self.shapes):
            raise ValueError("virtual render is missing frames")
        self.payload.update(complete=True, status="complete")
        self.payload["artifact_id"] = identity(self.payload)
        write_json(self.root / "render_manifest.json", self.payload)
        try:
            validate_render(self.root, self.backend)
        except (OSError, ValueError):
            self.payload.update(complete=False, status="in_progress")
            self.payload.pop("artifact_id", None)
            write_json(self.root / "render_manifest.json", self.payload)
            raise


def validate_render(root, backend=None):
    root = Path(root)
    value = read_json(root / "render_manifest.json")
    if (
        value.get("kind") != KIND
        or not value.get("complete")
        or value.get("status") != "complete"
    ):
        raise ValueError(f"incomplete virtual render: {root}")
    if backend is not None and value.get("backend") != backend:
        raise ValueError("virtual render backend mismatch")
    unsigned = {k: v for k, v in value.items() if k != "artifact_id"}
    if identity(unsigned) != value.get("artifact_id"):
        raise ValueError("virtual render manifest identity mismatch")
    expected_by_directory = {}
    for name, frame in value["frames"].items():
        for relative, digest in frame["outputs"].items():
            path = Path(relative)
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) != 2
                or path.stem != name
            ):
                raise ValueError("unsafe virtual artifact path")
            if sha256(root / path) != digest:
                raise ValueError(f"virtual artifact changed: {root / path}")
            expected_by_directory.setdefault(path.parent, set()).add(path.name)
    for directory, expected in expected_by_directory.items():
        actual = {
            p.name
            for p in (root / directory).iterdir()
            if p.suffix
            == (
                ".npy"
                if directory.name in {"rgb_raw", "depth", "alpha", "normal"}
                else ".png"
            )
        }
        if actual != expected:
            raise ValueError(f"extra or missing frames in {root / directory}")
    return value


def verify_tracking_render(session):
    """Old native sessions have no render record; new sessions bind exact geometry."""
    record = session.get("input_virtual_render")
    if record is None:
        return
    path = Path(record["path"])
    value = validate_render(path.parent, record["backend"])
    if (
        value["artifact_id"] != record["artifact_id"]
        or sha256(path) != record["sha256"]
    ):
        raise ValueError("virtual render changed after tracking started")
