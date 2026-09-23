"""Stable interchange format for Inpaint360GS virtual cameras.

The original pipeline regenerated its circular camera path in every stage from
``circle_radius``.  Persisting the actual per-frame camera parameters avoids
small pose changes caused by rounded configuration values and makes staged
runs independently verifiable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

MANIFEST_KIND = "inpaint360gs-virtual-cameras"
SCHEMA_VERSION = 1
ROTATION_TOLERANCE = 1e-5
MIN_LINEAR_SCALE = 1e-12
HEMISPHERE_ALGORITHM_REVISION = 3


def hemisphere_spiral_layout(count, max_elevation_deg=85.0):
    """Equal-height samples with automatic pitch for spherical area coverage.

    Returns unit-sphere offsets at zero starting azimuth and the angular pitch.
    Both endpoints belong to the open path; only its first frame is at the
    equator. This CPU-only policy is shared by generation and validation.
    """
    if type(count) is not int or count < 2:
        raise ValueError("hemisphere requires total count >= 2")
    maximum = _finite_float(max_elevation_deg, "hemisphere elevation")
    if not 0 < maximum < 90:
        raise ValueError("hemisphere elevation must be finite in (0,90)")
    maximum = math.radians(maximum)
    heights = np.linspace(0., math.sin(maximum), count)
    pitch = math.sqrt(2 * math.pi * math.sin(maximum) / (count - 1))
    # cos(arcsin(h)) loses the final latitude extremely close to 90 degrees
    # when sin(maximum) rounds to 1; retain the configured endpoint exactly.
    latitudes = np.arcsin(heights)
    latitudes[-1] = maximum
    azimuths = 2 * math.pi * latitudes / pitch
    radial = np.cos(latitudes)
    return np.column_stack([radial*np.cos(azimuths), radial*np.sin(azimuths), heights]), pitch


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{label} must be a finite {qualifier}number")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric array with shape {shape}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(
            f"{label} must be finite with shape {shape}, got {result.shape}"
        )
    return result


def _validate_scaled_rotation(rotation: np.ndarray, label: str) -> None:
    """Validate a right-handed rotation with an optional uniform PCA scale.

    ``generate_ellipse_path`` maps cameras through a PCA similarity transform.
    Its returned 3x3 block is therefore ``scale * rotation`` rather than always
    a member of SO(3).  The scale must be preserved because it is also present
    in the RGB/depth views consumed by later fusion stages.
    """

    gram = rotation.T @ rotation
    squared_scale = float(np.trace(gram) / 3.0)
    if not math.isfinite(squared_scale) or squared_scale <= MIN_LINEAR_SCALE**2:
        raise ValueError(f"{label} has a singular or near-zero linear scale")

    scale = math.sqrt(squared_scale)
    normalized = rotation / scale
    if not np.allclose(
        normalized.T @ normalized,
        np.eye(3),
        rtol=ROTATION_TOLERANCE,
        atol=ROTATION_TOLERANCE,
    ):
        raise ValueError(
            f"{label} is not orthonormal up to a single uniform positive scale"
        )

    determinant = float(np.linalg.det(normalized))
    if not math.isclose(
        determinant,
        1.0,
        rel_tol=ROTATION_TOLERANCE,
        abs_tol=ROTATION_TOLERANCE,
    ):
        raise ValueError(f"{label} normalized determinant is {determinant}, expected 1")


def _camera_record(view: Any) -> dict[str, Any]:
    return {
        "image_name": str(view.image_name),
        "R": np.asarray(view.R, dtype=np.float64).tolist(),
        "T": np.asarray(view.T, dtype=np.float64).tolist(),
        "FoVx": float(view.FoVx),
        "FoVy": float(view.FoVy),
        "image_width": int(view.image_width),
        "image_height": int(view.image_height),
        "znear": float(view.znear),
        "zfar": float(view.zfar),
        "trans": np.asarray(view.trans, dtype=np.float64).tolist(),
        "scale": float(view.scale),
    }


def build_virtual_camera_manifest(
    views: Sequence[Any], *, iteration: int, circle_radius: float, trajectory=None
) -> dict[str, Any]:
    """Create a JSON-serializable manifest without rounding camera values."""

    iteration = _positive_int(iteration, "iteration")
    circle_radius = _finite_float(circle_radius, "circle_radius", positive=True)
    records = [_camera_record(view) for view in views]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "frame_count": len(records),
        "iteration": iteration,
        "circle_radius": circle_radius,
        "cameras": records,
    }
    if trajectory is not None:
        identity.update(schema_version=2, trajectory=trajectory)
    # Validate the exact representation that will be consumed later.
    identity = validate_virtual_camera_manifest(identity)
    artifact_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **identity,
        "complete": True,
        "status": "complete",
        "artifact_id": artifact_id,
    }


def write_virtual_camera_manifest(
    path: str | os.PathLike[str],
    views: Sequence[Any],
    *,
    iteration: int,
    circle_radius: float,
    trajectory=None,
) -> Path:
    """Atomically persist virtual cameras and return the absolute path."""

    output = Path(path).expanduser().resolve()
    payload = build_virtual_camera_manifest(
        views, iteration=iteration, circle_radius=circle_radius, trajectory=trajectory
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.is_file():
        try:
            if output.read_text(encoding="utf-8") == serialized:
                return output
        except (OSError, UnicodeDecodeError):
            # Fall through to the atomic replacement, which preserves the
            # established error behaviour for an unreadable destination.
            pass
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def validate_virtual_camera_manifest(
    payload: Mapping[str, Any],
    *,
    expected_iteration: int | None = None,
    expected_frame_count: int | None = None,
) -> dict[str, Any]:
    """Validate and normalize one virtual-camera manifest."""

    if not isinstance(payload, Mapping):
        raise ValueError("virtual camera manifest root must be an object")
    if payload.get("kind") != MANIFEST_KIND:
        raise ValueError(
            f"unexpected virtual camera manifest kind: {payload.get('kind')!r}"
        )
    if payload.get("schema_version") not in (1, 2):
        raise ValueError(
            f"unsupported virtual camera schema: {payload.get('schema_version')!r}"
        )

    iteration = _positive_int(payload.get("iteration"), "iteration")
    if expected_iteration is not None and iteration != expected_iteration:
        raise ValueError(
            f"virtual camera iteration is {iteration}, expected {expected_iteration}"
        )
    circle_radius = _finite_float(
        payload.get("circle_radius"), "circle_radius", positive=True
    )
    frame_count = _positive_int(payload.get("frame_count"), "frame_count")
    if expected_frame_count is not None and frame_count != expected_frame_count:
        raise ValueError(
            f"virtual camera frame_count is {frame_count}, expected {expected_frame_count}"
        )
    cameras = payload.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != frame_count:
        raise ValueError("cameras must be a list matching frame_count")

    expected_names = [f"{index:05d}" for index in range(frame_count)]
    normalized_records: list[dict[str, Any]] = []
    for index, (record, expected_name) in enumerate(zip(cameras, expected_names)):
        if not isinstance(record, Mapping):
            raise ValueError(f"cameras[{index}] must be an object")
        image_name = record.get("image_name")
        if image_name != expected_name:
            raise ValueError(
                f"cameras[{index}].image_name is {image_name!r}, expected {expected_name!r}"
            )
        rotation = _array(record.get("R"), (3, 3), f"cameras[{index}].R")
        translation = _array(record.get("T"), (3,), f"cameras[{index}].T")
        _validate_scaled_rotation(rotation, f"cameras[{index}].R")

        fov_x = _finite_float(
            record.get("FoVx"), f"cameras[{index}].FoVx", positive=True
        )
        fov_y = _finite_float(
            record.get("FoVy"), f"cameras[{index}].FoVy", positive=True
        )
        if fov_x >= math.pi or fov_y >= math.pi:
            raise ValueError(f"cameras[{index}] field of view must be smaller than pi")
        width = _positive_int(
            record.get("image_width"), f"cameras[{index}].image_width"
        )
        height = _positive_int(
            record.get("image_height"), f"cameras[{index}].image_height"
        )
        znear = _finite_float(
            record.get("znear"), f"cameras[{index}].znear", positive=True
        )
        zfar = _finite_float(
            record.get("zfar"), f"cameras[{index}].zfar", positive=True
        )
        if zfar <= znear:
            raise ValueError(f"cameras[{index}].zfar must be larger than znear")
        trans = _array(record.get("trans"), (3,), f"cameras[{index}].trans")
        scale = _finite_float(
            record.get("scale"), f"cameras[{index}].scale", positive=True
        )

        normalized_records.append(
            {
                "image_name": image_name,
                "R": rotation.tolist(),
                "T": translation.tolist(),
                "FoVx": fov_x,
                "FoVy": fov_y,
                "image_width": width,
                "image_height": height,
                "znear": znear,
                "zfar": zfar,
                "trans": trans.tolist(),
                "scale": scale,
            }
        )

    result = {
        "schema_version": payload["schema_version"],
        "kind": MANIFEST_KIND,
        "frame_count": frame_count,
        "iteration": iteration,
        "circle_radius": circle_radius,
        "cameras": normalized_records,
    }
    if payload["schema_version"] == 2:
        trajectory = payload.get("trajectory")
        validate_trajectory(trajectory, frame_count)
        result["trajectory"] = copy.deepcopy(trajectory)
    elif "trajectory" in payload:
        raise ValueError("legacy camera schema cannot declare a trajectory")
    return result


def load_virtual_camera_manifest(
    path: str | os.PathLike[str],
    *,
    expected_iteration: int | None = None,
    expected_frame_count: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read virtual camera manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"virtual camera manifest root must be an object: {manifest_path}"
        )
    if payload.get("complete") is not True or payload.get("status") != "complete":
        raise ValueError(f"virtual camera manifest is incomplete: {manifest_path}")
    artifact_id = payload.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError(f"virtual camera manifest has no artifact_id: {manifest_path}")
    normalized = validate_virtual_camera_manifest(
        payload,
        expected_iteration=expected_iteration,
        expected_frame_count=expected_frame_count,
    )
    keys = list(normalized)
    expected_id = hashlib.sha256(json.dumps(
        {k: payload[k] for k in keys}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if artifact_id != expected_id:
        raise ValueError("virtual camera manifest artifact_id mismatch")
    return normalized


def validate_trajectory(value, count):
    if not isinstance(value, dict) or value.get("type") not in ("circle", "hemisphere"):
        raise ValueError("unknown/missing camera trajectory")
    revision = HEMISPHERE_ALGORITHM_REVISION if value["type"] == "hemisphere" else 1
    if value.get("algorithm_revision") != revision or value.get("frame_count") != count:
        raise ValueError("camera trajectory revision/count mismatch; regenerate with a new RUN_NAME")
    for key in ("center_pca", "focus_pca"):
        _array(value.get(key), (3,), key)
    transform = _array(value.get("world_to_pca"), (4, 4), "world_to_pca")
    _validate_scaled_rotation(transform[:3, :3], "world_to_pca")
    if not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("invalid PCA homogeneous transform")
    _finite_float(value.get("radius_pca"), "radius_pca", positive=True)
    _finite_float(value.get("start_azimuth_rad"), "start_azimuth_rad")
    if value.get("up_axis") != "+z_pca":
        raise ValueError("unsupported trajectory up axis")
    source = value.get("source_cameras_sha256", "")
    if not isinstance(source, str) or len(source) != 64 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("missing source camera identity")
    if value["type"] == "hemisphere":
        maximum = _finite_float(value.get("max_elevation_deg"), "max_elevation_deg")
        _, pitch = hemisphere_spiral_layout(count, maximum)
        if (value.get("sampling") != "equal_height_auto_pitch" or
                value.get("orientation") != "scene_up" or
                not math.isclose(_finite_float(value.get("pitch_rad"), "pitch_rad", positive=True),
                                 pitch, rel_tol=1e-12) or
                not math.isclose(_finite_float(value.get("turns"), "turns", positive=True),
                                 math.radians(maximum)/pitch, rel_tol=1e-12)):
            raise ValueError("invalid hemisphere sampling policy")
        up = _array(value.get("scene_up_pca"), (3,), "scene_up_pca")
        if not (np.count_nonzero(up) == 1 and np.max(np.abs(up)) == 1.):
            raise ValueError("invalid hemisphere scene up")
    elif value.get("sampling") != "legacy_circle" or value.get("orientation") != "legacy_up":
        raise ValueError("invalid circle sampling policy")


def check_camera_request(payload, path_type, count, max_elevation=85.0):
    """Check a requested generation policy before reusing any downstream data."""
    if path_type not in ("circle", "hemisphere") or type(count) is not int or count < 2:
        raise ValueError("camera path must be circle/hemisphere and count an integer >= 2")
    if path_type == "hemisphere":
        hemisphere_spiral_layout(count, max_elevation)
    if payload is None:
        return
    trajectory = payload.get("trajectory", {"type": "circle"})
    if path_type == "hemisphere" and trajectory.get("algorithm_revision") != HEMISPHERE_ALGORITHM_REVISION:
        raise ValueError("virtual camera algorithm changed; choose a new RUN_NAME/removal workspace")
    if (payload["frame_count"] != count or trajectory["type"] != path_type or
            (path_type == "hemisphere" and trajectory["max_elevation_deg"] != float(max_elevation))):
        raise ValueError("virtual camera trajectory/count changed; choose a new RUN_NAME/removal workspace")


def require_declared_camera_manifest(config, path):
    """Never silently regenerate a legacy circle for a nondefault run."""
    mode = config.get("virtual_camera_path", "circle")
    count = config.get("virtual_camera_count", 30)
    if (mode != "circle" or count != 30) and not path:
        raise ValueError("nondefault virtual cameras require --camera_manifest; no circle fallback")
    if path:
        payload = load_virtual_camera_manifest(path)
        if "virtual_camera_path" in config or "virtual_camera_count" in config:
            check_camera_request(payload, mode, count, config.get("virtual_hemisphere_max_elevation_deg", 85.))


def trajectory_diagnostics(views, trajectory):
    """CPU diagnostics in the generation PCA frame (not a quality guarantee)."""
    transform = np.asarray(trajectory["world_to_pca"])
    matrices = []
    for view in views:
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = np.asarray(view.R).T, view.T
        matrices.append(transform @ np.linalg.inv(w2c))
    matrices = np.stack(matrices)
    positions = matrices[:, :3, 3]
    rotations = matrices[:, :3, :3]
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    unit = (positions - trajectory["center_pca"]) / trajectory["radius_pca"]
    elevation = np.degrees(np.arcsin(np.clip(unit[:, 2], -1, 1)))
    nearest = []
    for start in range(0, len(unit), 256):
        dots = unit[start:start+256] @ unit.T
        dots[np.arange(len(dots)), np.arange(start, start+len(dots))] = -np.inf
        nearest.extend(np.degrees(np.arccos(np.clip(dots.max(1), -1, 1))).tolist())
    up = np.asarray(trajectory.get("scene_up_pca", [0., 0., 1.]))
    azimuth = np.degrees(np.unwrap(np.arctan2(unit[:, 1], unit[:, 0])))
    if trajectory["type"] == "hemisphere":
        # np.unwrap cannot recover >180 degree steps near the pole or for small
        # N. Record the generation parameter, with modulo agreement tested.
        h = np.linspace(0., math.sin(math.radians(trajectory["max_elevation_deg"])), len(views))
        phi = np.arcsin(h)
        phi[-1] = math.radians(trajectory["max_elevation_deg"])
        azimuth = np.degrees(trajectory["start_azimuth_rad"] + 2*math.pi*phi/trajectory["pitch_rad"])
    frames = []
    for i, view in enumerate(views):
        step = np.linalg.norm(positions[i]-positions[i-1]) if i else 0.
        axis = np.dot(rotations[i, :, 2], rotations[i-1, :, 2]) if i else 1.
        angle = (np.trace(rotations[i-1].T @ rotations[i])-1)/2 if i else 1.
        backward = -rotations[i, :, 2]  # renderer axes -> look-at backward axis
        reference_right = np.cross(up, backward)
        roll = None
        if np.linalg.norm(reference_right) > 1e-12:
            reference_right /= np.linalg.norm(reference_right)
            reference_up = np.cross(backward, reference_right)
            roll = float(np.degrees(np.arctan2(np.dot(rotations[i, :, 0], reference_up),
                                               np.dot(rotations[i, :, 0], reference_right))))
        frames.append(dict(image_name=view.image_name, center_pca=positions[i].tolist(),
            azimuth_unwrapped_deg=float(azimuth[i]), scene_up_roll_deg=roll,
            elevation_deg=float(elevation[i]), previous_distance_pca=float(step),
            previous_view_axis_deg=float(np.degrees(np.arccos(np.clip(axis,-1,1)))),
            previous_rotation_deg=float(np.degrees(np.arccos(np.clip(angle,-1,1)))),
            nearest_camera_angle_deg=nearest[i]))
    nearest = np.asarray(nearest)
    result = dict(trajectory=trajectory, frames=frames, nearest_camera_angle_deg=dict(
        minimum=float(nearest.min()), median=float(np.median(nearest)), maximum=float(nearest.max()),
        coefficient_of_variation=float(nearest.std()/max(nearest.mean(), 1e-12))))
    rolls = [abs(f["scene_up_roll_deg"]) for f in frames if f["scene_up_roll_deg"] is not None]
    result["scene_up_roll"] = dict(max_abs_deg=max(rolls) if rolls else None,
                                    undefined_frames=len(frames)-len(rolls))
    if trajectory["type"] == "hemisphere":
        # Equal-solid-angle probes over the sampled belt, not just the curve.
        h, theta = np.meshgrid((np.arange(64)+.5)/64 * math.sin(math.radians(
            trajectory["max_elevation_deg"])), (np.arange(256)+.5)/256*2*np.pi, indexing="ij")
        h, theta = h.ravel(), theta.ravel()
        radial = np.sqrt(1-h*h)
        probes = np.column_stack([radial*np.cos(theta), radial*np.sin(theta), h])
        gaps, owners = [], []
        for start in range(0, len(probes), 256):
            dots = probes[start:start+256] @ unit.T
            owners.extend(dots.argmax(1).tolist())
            gaps.extend(np.degrees(np.arccos(np.clip(dots.max(1), -1, 1))).tolist())
        areas = np.bincount(owners, minlength=len(unit)) / len(probes)
        result["surface_coverage"] = dict(probe_grid=[64, 256],
            max_gap_deg=float(np.max(gaps)), p95_gap_deg=float(np.percentile(gaps, 95)),
            cell_area_cv=float(areas.std()/areas.mean()))
    return result


def trajectory_svg(diagnostics):
    """Numbered top/front projections; no plotting dependency in LaMa environment."""
    positions = np.array([f["center_pca"] for f in diagnostics["frames"]])
    info = diagnostics["trajectory"]
    positions = (positions - info["center_pca"]) / info["radius_pca"]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="540" viewBox="0 0 1000 540">',
             '<rect width="1000" height="540" fill="white"/>']
    for panel, (a, b, title) in enumerate(((0, 1, "PCA XY (top)"), (0, 2, "PCA XZ (front)"))):
        x = positions[:, a]*200 + 250 + panel*500
        y = 285 - positions[:, b]*200
        parts.append(f'<text x="{panel*500+25}" y="30" font-size="18">{title}</text>')
        parts.append('<polyline fill="none" stroke="#7495a8" points="' +
                     ' '.join(f'{u:.2f},{v:.2f}' for u,v in zip(x,y)) + '"/>')
        for i, (u, v) in enumerate(zip(x, y)):
            parts.append(f'<circle cx="{u:.2f}" cy="{v:.2f}" r="3" fill="#0d7055"/>')
            parts.append(f'<text x="{u+4:.2f}" y="{v-4:.2f}" font-size="9">{i:05d}</text>')
    return '\n'.join(parts + ['</svg>']) + '\n'


def virtual_views_from_manifest(
    base_view: Any,
    manifest: Mapping[str, Any],
) -> list[Any]:
    """Recreate render-camera objects from validated, exact per-frame values."""

    import torch
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    payload = validate_virtual_camera_manifest(manifest)
    views = []
    device = base_view.world_view_transform.device
    dtype = base_view.world_view_transform.dtype
    for index, record in enumerate(payload["cameras"]):
        if record["image_width"] != int(base_view.image_width) or record[
            "image_height"
        ] != int(base_view.image_height):
            raise ValueError(
                f"camera {record['image_name']} is {record['image_width']}x"
                f"{record['image_height']}, but the loaded scene is "
                f"{base_view.image_width}x{base_view.image_height}"
            )
        view = copy.deepcopy(base_view)
        rotation = np.asarray(record["R"], dtype=np.float64)
        translation = np.asarray(record["T"], dtype=np.float64)
        trans = np.asarray(record["trans"], dtype=np.float64)
        scale = float(record["scale"])
        view.R = rotation
        view.T = translation
        view.FoVx = float(record["FoVx"])
        view.FoVy = float(record["FoVy"])
        view.znear = float(record["znear"])
        view.zfar = float(record["zfar"])
        view.trans = trans
        view.scale = scale
        view.image_name = record["image_name"]
        world_view = getWorld2View2(rotation, translation, trans, scale)
        view.world_view_transform = torch.as_tensor(
            world_view, dtype=dtype, device=device
        ).transpose(0, 1)
        view.projection_matrix = (
            getProjectionMatrix(
                znear=view.znear,
                zfar=view.zfar,
                fovX=view.FoVx,
                fovY=view.FoVy,
            )
            .to(device=device, dtype=dtype)
            .transpose(0, 1)
        )
        view.full_proj_transform = (
            view.world_view_transform.unsqueeze(0)
            .bmm(view.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        views.append(view)
    return views
