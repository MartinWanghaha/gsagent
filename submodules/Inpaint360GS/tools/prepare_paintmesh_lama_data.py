#!/usr/bin/env python3
"""Prepare and validate isolated LaMa inputs for the PaintMesh pipeline.

The historical Inpaint360GS adapter writes into the source dataset and into
``LaMa/data``/``LaMa/output``.  This adapter deliberately accepts every path
on the command line, validates the complete virtual-view sequence, and writes
only to run-local input directories. Raw normal completion is enabled by the
upstream render contract, never by a separate feature switch.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import importlib.util
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
from PIL import Image

try:
    from tools import paintmesh_normal as normals
except ModuleNotFoundError:  # Direct script invocation without the repo PYTHONPATH.
    import paintmesh_normal as normals

SCHEMA_VERSION = 1
INPUT_KIND = "paintmesh-lama-inputs"
COMPLETION_KIND = "paintmesh-lama-completion"
FRAME_PATTERN = re.compile(r"^[0-9]{5}$")


class LamaDataError(RuntimeError):
    """Raised when a LaMa artifact violates the PaintMesh data contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise LamaDataError(f"artifact is not a regular file: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path),
    }


def _identity(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    _atomic_bytes(path, payload.encode("utf-8"))


def _atomic_png(path: Path, array: np.ndarray, mode: str) -> None:
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    _atomic_bytes(path, buffer.getvalue())


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _atomic_bytes(path, buffer.getvalue())


def _expected_stems(frames: int) -> list[str]:
    if isinstance(frames, bool) or frames <= 0 or frames > 100_000:
        raise LamaDataError("frames must be a positive integer no larger than 100000")
    return [f"{index:05d}" for index in range(frames)]


def _collect_frames(
    root: Path,
    suffix: str,
    expected: Iterable[str],
    label: str,
    *,
    allow_tracker_diagnostics: bool = False,
) -> dict[str, Path]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise LamaDataError(f"{label} is not a directory: {root}")
    expected_set = set(expected)
    result: dict[str, Path] = {}
    unexpected: list[str] = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        stem = path.stem
        if allow_tracker_diagnostics and re.fullmatch(r"[0-9]{5}_new", stem):
            continue
        if not FRAME_PATTERN.fullmatch(stem) or stem not in expected_set:
            unexpected.append(path.name)
            continue
        result[stem] = path.resolve(strict=True)
    missing = sorted(expected_set - result.keys())
    if missing or unexpected or len(result) != len(expected_set):
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(sorted(unexpected)))
        raise LamaDataError(
            f"{label} must contain exactly {len(expected_set)} canonical frames: "
            + "; ".join(details)
        )
    return result


def _read_rgb(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise LamaDataError(f"cannot read {label} {path}: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0:
        raise LamaDataError(f"{label} must be a non-empty HxWx3 image: {path}")
    return rgb


def _read_index_mask(path: Path) -> np.ndarray:
    """Read the original label indices without applying a PNG palette."""

    try:
        with Image.open(path) as image:
            if image.mode not in {"P", "L", "1"}:
                raise LamaDataError(
                    f"mask must use Pillow mode P, L, or 1; got {image.mode!r}: {path}"
                )
            labels = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise LamaDataError(f"cannot read mask {path}: {exc}") from exc
    if labels.ndim != 2 or min(labels.shape) <= 0:
        raise LamaDataError(f"mask must be a non-empty 2D label image: {path}")
    # Palette RGB values must never participate in this conversion.  Every
    # non-zero tracker identity is part of the requested inpainting region.
    return labels != 0


def _clean_mask(mask: np.ndarray, min_area: int, dilation: int) -> np.ndarray:
    if min_area <= 0:
        raise LamaDataError("min_area must be a positive integer")
    if dilation < 0:
        raise LamaDataError("dilation must be a non-negative integer")
    binary = mask.astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)
    eligible: list[tuple[int, int]] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 1
        eligible.append((area, label))
    # Preserve the legacy behavior for a real but very small target: keep its
    # largest component rather than silently turning it into an empty mask.
    if not cleaned.any() and eligible:
        _, largest = max(eligible)
        cleaned[labels == largest] = 1
    if dilation:
        size = 2 * dilation + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    foreground = int(cleaned.sum())
    if foreground == 0:
        raise LamaDataError("mask has no foreground pixels")
    if foreground == cleaned.size:
        raise LamaDataError("mask covers the entire frame")
    return cleaned.astype(bool)


def _read_depth(path: Path, label: str) -> np.ndarray:
    try:
        depth = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise LamaDataError(f"cannot read {label} {path}: {exc}") from exc
    if depth.ndim != 2 or min(depth.shape) <= 0:
        raise LamaDataError(f"{label} must be a non-empty 2D array: {path}")
    if not np.issubdtype(depth.dtype, np.floating):
        depth = depth.astype(np.float32)
    if not np.isfinite(depth).all():
        raise LamaDataError(f"{label} contains NaN or infinity: {path}")
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_min < 0.0 or depth_max <= depth_min:
        raise LamaDataError(
            f"{label} must have a finite non-negative, non-zero range; "
            f"got [{depth_min}, {depth_max}]: {path}"
        )
    return depth


def _load_complete_manifest(path: Path, kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LamaDataError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LamaDataError(f"manifest root must be an object: {path}")
    if payload.get("kind") != kind or payload.get("complete") is not True:
        raise LamaDataError(f"manifest is not a complete {kind} artifact: {path}")
    return payload


def _verify_record(record: Mapping[str, Any], path: Path, label: str) -> None:
    current = _artifact(path)
    for field in ("path", "size_bytes", "sha256"):
        if record.get(field) != current[field]:
            raise LamaDataError(f"{label} changed: {path} ({field})")


def _managed_input_names(stems: Iterable[str]) -> tuple[set[str], set[str]]:
    stems = list(stems)
    color = {f"{stem}.png" for stem in stems} | {f"{stem}_mask.png" for stem in stems}
    depth = (
        {f"{stem}.npy" for stem in stems}
        | {f"{stem}_mask.png" for stem in stems}
        | {"depth_original"}
    )
    return color, depth


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _validate_run_local_outputs(
    outputs: Iterable[Path], source_roots: Iterable[Path]
) -> None:
    outputs = [path.resolve() for path in outputs]
    source_roots = [path.resolve(strict=True) for path in source_roots]
    shared_lama = Path(__file__).resolve().parents[1] / "LaMa"
    forbidden = (shared_lama / "data", shared_lama / "output")
    for output in outputs:
        for source in source_roots:
            if _is_within(output, source):
                raise LamaDataError(
                    f"run-local output cannot be inside an input directory: {output}"
                )
        for root in forbidden:
            if _is_within(output, root):
                raise LamaDataError(
                    f"explicit PaintMesh output cannot use shared LaMa storage: {output}"
                )


def _validate_partial_destination(root: Path, allowed: set[str], label: str) -> None:
    if root.is_symlink():
        raise LamaDataError(f"{label} cannot be a symlink: {root}")
    if root.exists() and not root.is_dir():
        raise LamaDataError(f"{label} must be a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name for path in root.iterdir() if path.name not in allowed
    )
    if unexpected:
        raise LamaDataError(
            f"{label} contains unmanaged files: " + ", ".join(unexpected)
        )


def _normal_sources(removed_rgb, removed_depth, camera_manifest, stems):
    root = Path(removed_depth).absolute().parent
    path = root / "render_manifest.json"
    has_files = any((root / name).exists() for name in ("normal", "normal_valid", "normal_vis"))
    if not path.exists():
        if has_files:
            raise LamaDataError("removed normal requires a verified render manifest; rerender virtual views")
        return None
    render = json.loads(path.read_text())
    if (
        render.get("kind") != "paintmesh-virtual-render"
        or render.get("complete") is not True
        or render.get("status") != "complete"
        or render.get("artifact_id") != normals.identity({k: v for k, v in render.items() if k != "artifact_id"})
    ):
        raise LamaDataError("invalid or incomplete removed render manifest")
    if "normal" not in render.get("capabilities", []):
        if has_files:
            raise LamaDataError("normal files contradict render capabilities")
        return None
    if camera_manifest is None:
        raise LamaDataError("removed normal requires --camera-manifest")
    if Path(removed_rgb).resolve() != (root / "renders").resolve():
        raise LamaDataError("normal/RGB must come from the same removed render")
    # Shared render validator is neutral (numpy/Pillow only), not a GS renderer.
    module_path = Path(__file__).resolve().parents[3] / "scripts/paintmesh/virtual_render_io.py"
    spec = importlib.util.spec_from_file_location("paintmesh_normal_render_io", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    render = module.validate_render(root)
    if (render.get("normal_space"), render.get("normal_orientation")) != ("camera", "toward_camera"):
        raise LamaDataError("unsupported removed normal coordinate convention")
    if render.get("normal_axes") != "+x right,+y down,+z forward":
        raise LamaDataError("unsupported removed normal axis convention")
    cameras, records = normals.read_cameras(camera_manifest, stems)
    if render.get("camera_artifact_id") != cameras["artifact_id"] or set(render["frames"]) != set(stems):
        raise LamaDataError("removed normal render/camera frame identity mismatch")
    for directory, suffix in (("normal", ".npy"), ("normal_valid", ".png"), ("alpha", ".npy")):
        _collect_frames(root / directory, suffix, stems, f"removed {directory}")
        for stem in stems:
            if f"{directory}/{stem}{suffix}" not in render["frames"][stem]["outputs"]:
                raise LamaDataError(f"unmanifested removed {directory} frame: {stem}")
    return {
        "root": root, "render": render, "cameras": records,
        "metadata": {
            "method": normals.METHOD, "encoding": normals.ENCODING,
            "space": "camera", "orientation": "toward_camera",
            "inference_mask_rule": "hole_or_invalid", "unit_tolerance": 1e-4,
            "render_manifest": _artifact(path),
            "camera_manifest": _artifact(Path(camera_manifest)),
            "render_artifact_id": render["artifact_id"],
            "camera_artifact_id": cameras["artifact_id"],
        },
    }


def _normal_paths(root, stem):
    return {
        "normal": root / f"{stem}.npy",
        "normal_mask": root / f"{stem}_mask.png",
        "normal_valid": root / "valid" / f"{stem}.png",
        "normal_inference_mask": root / "inference_mask" / f"{stem}.png",
    }


def _normal_destination(root, stems):
    allowed = {f"{s}.npy" for s in stems} | {f"{s}_mask.png" for s in stems}
    _validate_partial_destination(root, allowed | {"valid", "inference_mask", "cameras.json"}, "normal-input")
    for name in ("valid", "inference_mask"):
        _validate_partial_destination(root / name, {f"{s}.png" for s in stems}, f"normal {name}")


def prepare_lama_inputs(
    tracking_masks: Path,
    removed_rgb: Path,
    removed_depth: Path,
    reference_depth: Path,
    color_input: Path,
    depth_input: Path,
    manifest_path: Path,
    *,
    frames: int = 30,
    min_area: int = 50,
    dilation: int = 10,
    camera_manifest: Path | None = None,
) -> dict[str, Any]:
    stems = _expected_stems(frames)
    normal_sources = _normal_sources(removed_rgb, removed_depth, camera_manifest, stems)
    masks = _collect_frames(
        tracking_masks,
        ".png",
        stems,
        "tracking masks",
        allow_tracker_diagnostics=True,
    )
    rgbs = _collect_frames(removed_rgb, ".png", stems, "removed RGB renders")
    removed_depths = _collect_frames(removed_depth, ".npy", stems, "removed depth maps")
    reference_depths = _collect_frames(
        reference_depth, ".npy", stems, "reference depth maps"
    )

    color_input = color_input.expanduser().resolve()
    depth_input = depth_input.expanduser().resolve()
    normal_input = depth_input.parent / "normal"
    roots = {"color_input": str(color_input), "depth_input": str(depth_input)}
    if normal_sources:
        roots["normal_input"] = str(normal_input)
    elif normal_input.exists():
        raise LamaDataError("stale normal input exists for a render without normal; use a new inpaint run")
    manifest_path = manifest_path.expanduser().resolve()
    if color_input == depth_input:
        raise LamaDataError("color-input and depth-input must be different directories")
    source_roots = {
        Path(path).expanduser().resolve(strict=True)
        for path in (tracking_masks, removed_rgb, removed_depth, reference_depth)
    }
    _validate_run_local_outputs((color_input, depth_input), source_roots)
    if normal_sources:
        _validate_run_local_outputs((normal_input,), source_roots | {normal_sources["root"].resolve()})
        if any(_is_within(normal_input, p) or _is_within(p, normal_input) for p in (color_input, depth_input)):
            raise LamaDataError("normal input must be independent of RGB/depth inputs")

    frame_data: dict[str, dict[str, Any]] = {}
    identity_frames: dict[str, Any] = {}
    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    prepared_normals = {}
    for stem in stems:
        rgb = _read_rgb(rgbs[stem], "removed RGB")
        raw_mask = _read_index_mask(masks[stem])
        mask = _clean_mask(raw_mask, min_area, dilation)
        hole_depth = _read_depth(removed_depths[stem], "removed depth")
        full_depth = _read_depth(reference_depths[stem], "reference depth")
        shape = tuple(rgb.shape[:2])
        for value, label in (
            (raw_mask, "tracking mask"),
            (hole_depth, "removed depth"),
            (full_depth, "reference depth"),
        ):
            if tuple(value.shape) != shape:
                raise LamaDataError(
                    f"{stem} {label} shape {tuple(value.shape)} does not match RGB {shape}"
                )
        inputs = {
            "mask": _artifact(masks[stem]),
            "removed_rgb": _artifact(rgbs[stem]),
            "removed_depth": _artifact(removed_depths[stem]),
            "reference_depth": _artifact(reference_depths[stem]),
        }
        if normal_sources:
            root = normal_sources["root"]
            normal_path = root / "normal" / f"{stem}.npy"
            valid_path = root / "normal_valid" / f"{stem}.png"
            alpha_path = root / "alpha" / f"{stem}.npy"
            normal, valid = normals.read_normal(normal_path, valid_path)
            if normal.shape[:2] != shape:
                raise LamaDataError(f"{stem} normal shape does not match RGB")
            normals.camera_rays(normal_sources["cameras"][stem], shape)
            alpha = np.load(alpha_path, allow_pickle=False)
            if alpha.shape != shape or not np.isfinite(alpha).all() or np.any((alpha < -1e-6) | (alpha > 1.000001)):
                raise LamaDataError(f"{stem} invalid removed alpha")
            if np.any(valid & ((alpha < normal_sources["render"]["alpha_min"]) | (hole_depth <= 0))):
                raise LamaDataError(f"{stem} normal validity contradicts alpha/depth")
            infer_mask = normals.inference_mask(mask, valid)
            prepared_normals[stem] = (normal, valid, infer_mask)
            inputs.update(removed_normal=_artifact(normal_path), removed_normal_valid=_artifact(valid_path), removed_alpha=_artifact(alpha_path))
        frame_data[stem] = {
            "shape": [shape[0], shape[1]],
            "mask_foreground_before": int(raw_mask.sum()),
            "mask_foreground_after": int(mask.sum()),
            "removed_depth_range": [float(hole_depth.min()), float(hole_depth.max())],
            "reference_depth_range": [float(full_depth.min()), float(full_depth.max())],
            "inputs": inputs,
        }
        identity_frames[stem] = {
            name: record["sha256"] for name, record in inputs.items()
        }
        prepared[stem] = (rgb, mask, hole_depth, full_depth)

    parameters = {
        "frames": frames,
        "frame_names": stems,
        "min_area": min_area,
        "dilation": dilation,
        "mask_rule": "original_index_nonzero",
    }
    if normal_sources:
        parameters.update(required_modalities=["rgb", "depth", "normal"], normal=normal_sources["metadata"])
    artifact_id = _identity(
        {
            "kind": INPUT_KIND,
            "schema_version": SCHEMA_VERSION,
            "parameters": parameters,
            "inputs": identity_frames,
        }
    )
    if manifest_path.exists():
        existing = _load_complete_manifest(manifest_path, INPUT_KIND)
        if existing.get("artifact_id") != artifact_id:
            raise LamaDataError(
                "existing LaMa input manifest belongs to different inputs or parameters; "
                "use a new INPAINT_RUN_NAME and rebuild Stage 2/3 (including normal if available)"
            )
        if existing.get("roots") != roots:
            raise LamaDataError(
                "existing LaMa input manifest uses different output roots"
            )
        color_names, depth_names = _managed_input_names(stems)
        _validate_partial_destination(color_input, color_names, "color-input")
        _validate_partial_destination(depth_input, depth_names, "depth-input")
        if normal_sources:
            _normal_destination(normal_input, stems)
            _verify_record(existing["normal_camera"], normal_input / "cameras.json", "normal camera snapshot")
        for stem in stems:
            outputs = existing["frames"][stem]["outputs"]
            for name, path in (
                ("color", color_input / f"{stem}.png"),
                ("color_mask", color_input / f"{stem}_mask.png"),
                ("depth", depth_input / f"{stem}.npy"),
                ("depth_mask", depth_input / f"{stem}_mask.png"),
                ("reference_depth", depth_input / "depth_original" / f"{stem}.npy"),
            ):
                _verify_record(outputs[name], path, f"prepared {name}")
            if normal_sources:
                for name, path in _normal_paths(normal_input, stem).items():
                    _verify_record(outputs[name], path, f"prepared {name}")
        return existing

    color_names, depth_names = _managed_input_names(stems)
    _validate_partial_destination(color_input, color_names, "color-input")
    _validate_partial_destination(depth_input, depth_names, "depth-input")
    reference_output = depth_input / "depth_original"
    _validate_partial_destination(
        reference_output,
        {f"{stem}.npy" for stem in stems},
        "depth reference directory",
    )
    if normal_sources:
        _normal_destination(normal_input, stems)
        _atomic_bytes(normal_input / "cameras.json", Path(camera_manifest).read_bytes())

    for stem, (rgb, mask, hole_depth, full_depth) in prepared.items():
        mask_u8 = mask.astype(np.uint8) * 255
        _atomic_png(color_input / f"{stem}.png", rgb, "RGB")
        _atomic_png(color_input / f"{stem}_mask.png", mask_u8, "L")
        _atomic_npy(depth_input / f"{stem}.npy", hole_depth)
        _atomic_png(depth_input / f"{stem}_mask.png", mask_u8, "L")
        _atomic_npy(reference_output / f"{stem}.npy", full_depth)
        frame_data[stem]["outputs"] = {
            "color": _artifact(color_input / f"{stem}.png"),
            "color_mask": _artifact(color_input / f"{stem}_mask.png"),
            "depth": _artifact(depth_input / f"{stem}.npy"),
            "depth_mask": _artifact(depth_input / f"{stem}_mask.png"),
            "reference_depth": _artifact(reference_output / f"{stem}.npy"),
        }
        if normal_sources:
            normal, valid, infer_mask = prepared_normals[stem]
            paths = _normal_paths(normal_input, stem)
            _atomic_npy(paths["normal"], normal)
            for name, value in (("normal_mask", mask), ("normal_valid", valid), ("normal_inference_mask", infer_mask)):
                _atomic_png(paths[name], value.astype(np.uint8) * 255, "L")
            frame_data[stem]["outputs"].update({name: _artifact(path) for name, path in paths.items()})

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": INPUT_KIND,
        "complete": True,
        "status": "complete",
        "artifact_id": artifact_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "roots": roots,
        "frames": frame_data,
    }
    if normal_sources:
        payload["normal_camera"] = _artifact(normal_input / "cameras.json")
    _atomic_json(manifest_path, payload)
    return payload


def verify_normal_inputs(input_manifest):
    """Validate the prepared normal artifact before inference or cache reuse."""
    stems = input_manifest["parameters"]["frame_names"]
    metadata = input_manifest["parameters"].get("normal", {})
    if metadata.get("method") != normals.METHOD or metadata.get("encoding") != normals.ENCODING:
        raise LamaDataError("unsupported normal completion contract")
    root = Path(input_manifest["roots"]["normal_input"])
    _normal_destination(root, stems)
    for key in ("render_manifest", "camera_manifest"):
        record = metadata[key]
        _verify_record(record, Path(record["path"]), f"normal source {key}")
    _verify_record(input_manifest["normal_camera"], root / "cameras.json", "normal camera snapshot")
    camera_payload, cameras = normals.read_cameras(root / "cameras.json", stems)
    if camera_payload["artifact_id"] != metadata["camera_artifact_id"]:
        raise LamaDataError("normal camera identity mismatch")
    for stem in stems:
        frame = input_manifest["frames"][stem]
        for key in ("removed_normal", "removed_normal_valid", "removed_alpha"):
            record = frame["inputs"][key]
            _verify_record(record, Path(record["path"]), key)
        for key, path in _normal_paths(root, stem).items():
            _verify_record(frame["outputs"][key], path, f"normal input {key}")
        normal, valid = normals.read_normal(root / f"{stem}.npy", root / "valid" / f"{stem}.png")
        mask = normals.read_mask(root / f"{stem}_mask.png")
        normals.camera_rays(cameras[stem], valid.shape)
        for modality in ("color", "depth"):
            other = Path(input_manifest["roots"][f"{modality}_input"]) / f"{stem}_mask.png"
            _verify_record(frame["outputs"][f"{modality}_mask"], other, f"{modality} mask")
            if not np.array_equal(mask, normals.read_mask(other)):
                raise LamaDataError("normal/color/depth hole masks differ")
        if not np.array_equal(normals.inference_mask(mask, valid), normals.read_mask(root / "inference_mask" / f"{stem}.png")):
            raise LamaDataError("normal inference mask differs from hole/valid")
    return root, cameras


def validate_lama_outputs(
    color_input: Path,
    depth_input: Path,
    color_output: Path,
    depth_output: Path,
    model_path: Path,
    input_manifest_path: Path,
    manifest_path: Path,
    *,
    frames: int = 30,
    recursive_guide: bool = False,
) -> dict[str, Any]:
    stems = _expected_stems(frames)
    input_manifest_path = input_manifest_path.expanduser().resolve(strict=True)
    input_manifest = _load_complete_manifest(input_manifest_path, INPUT_KIND)
    if input_manifest.get("parameters", {}).get("frame_names") != stems:
        raise LamaDataError("input manifest frame set differs from --frames")

    color_input = color_input.expanduser().resolve(strict=True)
    depth_input = depth_input.expanduser().resolve(strict=True)
    normal_enabled = "normal" in input_manifest.get("parameters", {}).get("required_modalities", [])
    normal_output = Path(depth_output).expanduser().resolve().parent / "normal"
    roots = {"color_input": str(color_input), "depth_input": str(depth_input)}
    if normal_enabled:
        roots["normal_input"] = str(depth_input.parent / "normal")
    elif (depth_input.parent / "normal").exists() or normal_output.exists():
        raise LamaDataError("normal artifacts exist but are absent from LaMa input modalities")
    if input_manifest.get("roots") != roots:
        raise LamaDataError("input directories do not match the input manifest")
    if normal_enabled:
        normal_input, cameras = verify_normal_inputs(input_manifest)
        for directory, suffix in ((normal_output, ".npy"), (normal_output / "valid", ".png"), (normal_output / "vis", ".png")):
            _collect_frames(directory, suffix, stems, "completed normal")
    for stem in stems:
        records = input_manifest["frames"][stem]["outputs"]
        for name, path in (
            ("color", color_input / f"{stem}.png"),
            ("color_mask", color_input / f"{stem}_mask.png"),
            ("depth", depth_input / f"{stem}.npy"),
            ("depth_mask", depth_input / f"{stem}_mask.png"),
            ("reference_depth", depth_input / "depth_original" / f"{stem}.npy"),
        ):
            _verify_record(records[name], path, f"manifested LaMa input {name}")
    color_outputs = _collect_frames(color_output, ".png", stems, "completed RGB images")
    depth_outputs = _collect_frames(depth_output, ".npy", stems, "completed depth maps")

    model_path = model_path.expanduser().resolve(strict=True)
    model_config = model_path / "config.yaml"
    model_checkpoint = model_path / "models" / "best.ckpt"
    model_records = {
        "config": _artifact(model_config),
        "checkpoint": _artifact(model_checkpoint),
    }
    if normal_enabled:
        receipt_path = normal_output / "prediction.json"
        receipt = _load_complete_manifest(receipt_path, "paintmesh-normal-prediction")
        if receipt.get("artifact_id") != normals.identity({k: v for k, v in receipt.items() if k != "artifact_id"}):
            raise LamaDataError("normal prediction receipt identity mismatch")
        if (
            receipt.get("input_artifact_id") != input_manifest["artifact_id"]
            or receipt.get("input_sha256") != _sha256(input_manifest_path)
            or receipt.get("model") != {k: v["sha256"] for k, v in model_records.items()}
            or receipt.get("encoding") != normals.ENCODING
            or receipt.get("method") != normals.METHOD
            or set(receipt.get("frames", {})) != set(stems)
        ):
            raise LamaDataError("normal prediction provenance mismatch; rerun Stage 3 in a new inpaint run")
        prediction_config = Path(__file__).resolve().parents[1] / "LaMa/configs/prediction/default.yaml"
        if receipt.get("prediction_config_sha256") != _sha256(prediction_config):
            raise LamaDataError("normal prediction configuration changed")
    frame_records: dict[str, Any] = {}
    output_identity: dict[str, Any] = {}
    for stem in stems:
        source_rgb = _read_rgb(color_input / f"{stem}.png", "LaMa RGB input")
        mask = _read_index_mask(color_input / f"{stem}_mask.png")
        completed_rgb = _read_rgb(color_outputs[stem], "completed RGB")
        source_depth = _read_depth(depth_input / f"{stem}.npy", "LaMa depth input")
        reference = _read_depth(
            depth_input / "depth_original" / f"{stem}.npy",
            "LaMa reference depth",
        )
        completed_depth = _read_depth(depth_outputs[stem], "completed depth")
        shape = tuple(source_rgb.shape[:2])
        for value, label in (
            (mask, "mask"),
            (completed_rgb, "completed RGB"),
            (source_depth, "source depth"),
            (reference, "reference depth"),
            (completed_depth, "completed depth"),
        ):
            if tuple(value.shape[:2]) != shape:
                raise LamaDataError(
                    f"{stem} {label} shape {tuple(value.shape[:2])} does not match {shape}"
                )
        outside = ~mask
        if not np.array_equal(completed_rgb[outside], source_rgb[outside]):
            raise LamaDataError(
                f"completed RGB changed pixels outside the mask for frame {stem}"
            )
        if not np.array_equal(completed_depth[outside], source_depth[outside]):
            raise LamaDataError(
                f"completed depth changed values outside the mask for frame {stem}"
            )
        outputs = {
            "color": _artifact(color_outputs[stem]),
            "depth": _artifact(depth_outputs[stem]),
        }
        if normal_enabled:
            normal, valid = normals.read_normal(normal_output / f"{stem}.npy", normal_output / "valid" / f"{stem}.png")
            source, source_valid = normals.read_normal(normal_input / f"{stem}.npy", normal_input / "valid" / f"{stem}.png")
            if normal.shape[:2] != shape:
                raise LamaDataError("completed normal shape does not match RGB")
            if not np.array_equal(normal[outside], source[outside]) or not np.array_equal(valid[outside], source_valid[outside]):
                raise LamaDataError(f"completed normal changed values outside the mask for frame {stem}")
            if not (valid & mask).any():
                raise LamaDataError(f"no valid completed normals inside the hole: {stem}")
            rays = normals.camera_rays(cameras[stem], shape)
            if np.any(np.sum(normal * rays, axis=-1)[valid & mask] > 1e-5):
                raise LamaDataError(f"completed normal points away from camera: {stem}")
            vis_path = normal_output / "vis" / f"{stem}.png"
            if not np.array_equal(_read_rgb(vis_path, "normal preview"), normals.normal_preview(normal, valid)):
                raise LamaDataError(f"normal preview differs from raw normal: {stem}")
            for key, path in (("normal", normal_output / f"{stem}.npy"), ("normal_valid", normal_output / "valid" / f"{stem}.png"), ("normal_vis", vis_path)):
                _verify_record(receipt["frames"][stem][key], path, f"predicted {key}")
                outputs[key] = _artifact(path)
        frame_records[stem] = {
            "shape": [shape[0], shape[1]],
            "completed_depth_range": [
                float(completed_depth.min()),
                float(completed_depth.max()),
            ],
            "outputs": outputs,
        }
        if normal_enabled:
            frame_records[stem]["normal_hole_valid_fraction"] = float(valid[mask].mean())
        output_identity[stem] = {
            name: record["sha256"] for name, record in outputs.items()
        }

    parameters = {
        "frames": frames,
        "frame_names": stems,
        "recursive_guide": bool(recursive_guide),
        "outside_mask_policy": "preserve_input_exactly",
    }
    if normal_enabled:
        parameters.update(required_modalities=["rgb", "depth", "normal"], normal_method=normals.METHOD, normal_prediction_artifact_id=receipt["artifact_id"])
    artifact_id = _identity(
        {
            "kind": COMPLETION_KIND,
            "schema_version": SCHEMA_VERSION,
            "input_artifact_id": input_manifest["artifact_id"],
            "model": {name: record["sha256"] for name, record in model_records.items()},
            "parameters": parameters,
            "outputs": output_identity,
        }
    )
    manifest_path = manifest_path.expanduser().resolve()
    if manifest_path.exists():
        existing = _load_complete_manifest(manifest_path, COMPLETION_KIND)
        if existing.get("artifact_id") != artifact_id:
            raise LamaDataError(
                "existing completion manifest belongs to different outputs or parameters"
            )
        return existing

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": COMPLETION_KIND,
        "complete": True,
        "status": "complete",
        "artifact_id": artifact_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": _artifact(input_manifest_path),
        "input_artifact_id": input_manifest["artifact_id"],
        "model": model_records,
        "parameters": parameters,
        "roots": {
            "color_output": str(Path(color_output).expanduser().resolve()),
            "depth_output": str(Path(depth_output).expanduser().resolve()),
        },
        "frames": frame_records,
    }
    if normal_enabled:
        payload["roots"]["normal_output"] = str(normal_output)
        payload["normal_prediction"] = _artifact(receipt_path)
    _atomic_json(manifest_path, payload)
    return payload


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or validate isolated PaintMesh LaMa artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare run-local LaMa inputs")
    prepare.add_argument("--tracking-masks", type=Path, required=True)
    prepare.add_argument("--removed-rgb", type=Path, required=True)
    prepare.add_argument("--removed-depth", type=Path, required=True)
    prepare.add_argument("--reference-depth", type=Path, required=True)
    prepare.add_argument("--color-input", type=Path, required=True)
    prepare.add_argument("--depth-input", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--frames", type=_positive_int, default=30)
    prepare.add_argument("--min-area", type=_positive_int, default=50)
    prepare.add_argument("--dilation", type=_non_negative_int, default=10)
    prepare.add_argument("--camera-manifest", type=Path)

    validate = subparsers.add_parser(
        "validate-output", help="Validate LaMa outputs and commit their manifest"
    )
    validate.add_argument("--color-input", type=Path, required=True)
    validate.add_argument("--depth-input", type=Path, required=True)
    validate.add_argument("--color-output", type=Path, required=True)
    validate.add_argument("--depth-output", type=Path, required=True)
    validate.add_argument("--model-path", type=Path, required=True)
    validate.add_argument("--input-manifest", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--frames", type=_positive_int, default=30)
    validate.add_argument("--recursive-guide", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            payload = prepare_lama_inputs(
                args.tracking_masks,
                args.removed_rgb,
                args.removed_depth,
                args.reference_depth,
                args.color_input,
                args.depth_input,
                args.manifest,
                frames=args.frames,
                min_area=args.min_area,
                dilation=args.dilation,
                camera_manifest=args.camera_manifest,
            )
        else:
            payload = validate_lama_outputs(
                args.color_input,
                args.depth_input,
                args.color_output,
                args.depth_output,
                args.model_path,
                args.input_manifest,
                args.manifest,
                frames=args.frames,
                recursive_guide=args.recursive_guide,
            )
    except (LamaDataError, ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"PaintMesh LaMa data error: {exc}") from exc
    print(f"{payload['kind']}: {payload['artifact_id']}")


if __name__ == "__main__":
    main()
