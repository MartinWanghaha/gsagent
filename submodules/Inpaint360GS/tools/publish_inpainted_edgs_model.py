#!/usr/bin/env python3
"""Publish an Inpaint360GS result in the standard EDGS model layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.fusion_manifest_identity import (
    FUSION_IDENTITY_VERSION,
    fusion_artifact_id,
)

try:
    from tools.prepare_inpaint_workspace import _strict_manifest
    from tools.prepare_removal_workspace import (
        ArtifactError,
        _absolute_output,
        _artifact,
        _atomic_relative_symlink,
        _atomic_write_json,
        _ensure_independent_output,
        _identity,
        _mapping,
        _parse_namespace,
        _preflight_output,
        _regular_file,
        _relative_link_target,
        _sha256,
        _verify_manifest_artifact,
    )
    from tools.publish_removed_edgs_model import (
        _ensure_managed_directory,
        _load_edgs_config,
        _validate_removed_ply,
        _validate_selection,
    )
except ModuleNotFoundError:  # Direct repository-local execution.
    from prepare_inpaint_workspace import _strict_manifest
    from prepare_removal_workspace import (
        ArtifactError,
        _absolute_output,
        _artifact,
        _atomic_relative_symlink,
        _atomic_write_json,
        _ensure_independent_output,
        _identity,
        _mapping,
        _parse_namespace,
        _preflight_output,
        _regular_file,
        _relative_link_target,
        _sha256,
        _verify_manifest_artifact,
    )
    from publish_removed_edgs_model import (
        _ensure_managed_directory,
        _load_edgs_config,
        _validate_removed_ply,
        _validate_selection,
    )


MODEL_KIND = "paintmesh-inpainted-edgs-model"
MODEL_SCHEMA_VERSION = 1
MODEL_IDENTITY_VERSION = 2
MANIFEST_NAME = "model_manifest.json"
REMOVED_MODEL_KIND = "paintmesh-removed-edgs-model"
REMOVAL_KIND = "paintmesh-object-removal"
WORKSPACE_KIND = "paintmesh-inpaint-workspace"
TRACKING_KIND = "paintmesh-tracking-session"
LAMA_KIND = "paintmesh-lama-completion"
LAMA_INPUT_KIND = "paintmesh-lama-inputs"
FUSION_KIND = "paintmesh-rgbd-fusion"
CAMERA_KIND = "inpaint360gs-virtual-cameras"

CONTENT_INPUT_NAMES = (
    "inpainted_gaussian_ply",
    "classifier",
    "edgs_config",
    "cfg_args",
    "inpaint_config",
)

COPY_CHUNK_BYTES = 8 * 1024 * 1024


def _validate_edgs_joint(path, ply, lama_path, camera_path):
    import sys
    neutral = Path(__file__).resolve().parents[3] / "scripts/paintmesh"
    if str(neutral) not in sys.path:
        sys.path.insert(0, str(neutral))
    from edgs_inpaint_io import validate_joint
    try:
        return validate_joint(path, selected_ply=ply, lama_path=lama_path, camera_path=camera_path)
    except (ValueError, KeyError, OSError) as exc:
        raise ArtifactError(f"invalid EDGS joint output: {exc}") from exc


def _validate_local_geometry(path, ply, lama_path, camera_path):
    """Lazy neutral import: RGB-only publishing keeps its original contract."""
    import sys
    neutral_root = Path(__file__).resolve().parents[3] / "scripts/paintmesh"
    if str(neutral_root) not in sys.path:
        sys.path.insert(0, str(neutral_root))
    from local_geometry_io import validate_local
    try:
        return validate_local(path, selected_ply=ply, lama_path=lama_path, camera_path=camera_path)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise ArtifactError(f"invalid local geometry result: {exc}") from exc


def _validate_density(path, rgb, ply, local, lama, camera, fusion, config):
    import sys
    neutral_root = Path(__file__).resolve().parents[3] / "scripts/paintmesh"
    if str(neutral_root) not in sys.path:
        sys.path.insert(0, str(neutral_root))
    from support_density_io import validate_density_rgb
    try:
        return validate_density_rgb(path, rgb, selected_ply=ply, local_manifest=local,
            lama_path=lama, camera_path=camera, fusion_path=fusion, config_path=config)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise ArtifactError(f"invalid density producer chain: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    """Persist an atomic directory-entry replacement on POSIX filesystems."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_materialize_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> str:
    """Materialize ``source`` as an independent, atomically published file.

    An intact regular-file copy is reused without changing its inode or mtime.
    A legacy symlink or a damaged managed copy is replaced, while the symlink
    target/source is never modified.
    """

    source = _regular_file(source, "materialized file source")
    destination = destination.absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not destination.is_symlink():
        if not destination.is_file():
            raise ArtifactError(
                f"materialized file destination is not a regular file: {destination}"
            )
        same_file = os.path.samefile(source, destination)
        if (
            not same_file
            and destination.stat().st_size == source.stat().st_size
            and _sha256(destination) == expected_sha256
        ):
            return "reuse"

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".copy.tmp",
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    copied_bytes = 0
    source_stat = source.stat()
    try:
        with source.open("rb") as input_stream, os.fdopen(
            descriptor, "wb"
        ) as output_stream:
            os.fchmod(output_stream.fileno(), stat.S_IMODE(source_stat.st_mode))
            for chunk in iter(lambda: input_stream.read(COPY_CHUNK_BYTES), b""):
                digest.update(chunk)
                output_stream.write(chunk)
                copied_bytes += len(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())

        if copied_bytes != source_stat.st_size:
            raise ArtifactError(
                "inpainted Gaussian PLY changed size while it was being copied"
            )
        if digest.hexdigest() != expected_sha256:
            raise ArtifactError(
                "inpainted Gaussian PLY changed content while it was being copied"
            )

        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)

    if destination.is_symlink() or not destination.is_file():
        raise ArtifactError(f"failed to materialize a regular file at {destination}")
    if os.path.samefile(source, destination):
        raise ArtifactError(
            f"materialized file is not independent of its source: {destination}"
        )
    return "materialize"


def _recoverable_inpainted_model_output(output: Path, iteration: int) -> bool:
    """Recognize a publish interrupted before its model manifest commit."""

    entries = {path.name: path for path in output.iterdir()}
    if not entries or not set(entries).issubset(
        {"config.yaml", "cfg_args", "point_cloud"}
    ):
        return False
    for name in ("config.yaml", "cfg_args"):
        path = entries.get(name)
        if path is not None and not path.is_symlink():
            return False

    point_cloud = entries.get("point_cloud")
    if point_cloud is None:
        return True
    if point_cloud.is_symlink() or not point_cloud.is_dir():
        return False
    iteration_name = f"iteration_{iteration}"
    point_entries = {path.name: path for path in point_cloud.iterdir()}
    if not set(point_entries).issubset({iteration_name}):
        return False
    iteration_root = point_entries.get(iteration_name)
    if iteration_root is None:
        return True
    if iteration_root.is_symlink() or not iteration_root.is_dir():
        return False

    artifacts = {path.name: path for path in iteration_root.iterdir()}
    if not set(artifacts).issubset({"point_cloud.ply", "classifier.pth"}):
        return False
    classifier = artifacts.get("classifier.pth")
    if classifier is not None and not classifier.is_symlink():
        return False
    point_cloud_ply = artifacts.get("point_cloud.ply")
    return (
        point_cloud_ply is None
        or point_cloud_ply.is_symlink()
        or (point_cloud_ply.is_file() and point_cloud_ply.stat().st_size > 0)
    )


def _verified_artifact(value: Any, location: str, *, verify_mtime: bool = True) -> Path:
    """Resolve and verify one producer-style artifact record.

    Size and SHA-256 define file content.  ``mtime_ns`` remains useful for
    strict legacy snapshots, but content-addressed v2 manifests deliberately
    tolerate an idempotent producer rewriting identical bytes.
    """

    record = _mapping(value, location)
    path_text = record.get("path")
    if not isinstance(path_text, str) or not path_text.strip():
        raise ArtifactError(f"{location}.path must be a non-empty path string")
    path = _regular_file(Path(path_text), location)
    stat = path.stat()
    expected_values = {
        "size_bytes": int(stat.st_size),
        "sha256": _sha256(path),
    }
    if verify_mtime:
        expected_values["mtime_ns"] = int(stat.st_mtime_ns)
    for key, expected in expected_values.items():
        observed = record.get(key)
        if isinstance(observed, bool) or observed != expected:
            raise ArtifactError(
                f"{location}.{key} is {observed!r}, expected {expected!r}"
            )
    return path


def _recorded_path(value: Any, location: str) -> Path:
    """Resolve a provenance record without binding identity to file metadata."""

    record = _mapping(value, location)
    path_text = record.get("path")
    if not isinstance(path_text, str) or not path_text.strip():
        raise ArtifactError(f"{location}.path must be a non-empty path string")
    return _regular_file(Path(path_text), location)


def _validate_camera_artifact_id(camera: Mapping[str, Any]) -> str:
    """Verify the camera content-addressed ID while ignoring envelope metadata."""

    identity = {
        key: camera.get(key)
        for key in (
            "schema_version",
            "kind",
            "frame_count",
            "iteration",
            "circle_radius",
            "cameras",
        )
    }
    if camera.get("schema_version") == 2:
        identity["trajectory"] = camera.get("trajectory")
    expected = _identity(identity)
    observed = camera.get("artifact_id")
    if not isinstance(observed, str) or not observed:
        raise ArtifactError("virtual-camera manifest has no artifact_id")
    if observed != expected:
        raise ArtifactError(
            "virtual-camera manifest artifact_id does not match its camera content"
        )
    return expected


def _expected_frames(parameters: Any, location: str) -> list[str]:
    values = _mapping(parameters, location)
    count = values.get("frames")
    if type(count) is not int or count < 1:
        raise ArtifactError(f"{location} has an invalid frame count")
    expected = [f"{index:05d}" for index in range(count)]
    if values.get("frame_names") != expected:
        raise ArtifactError(
            f"{location} must declare exactly the ordered camera frame set"
        )
    return expected


def _frame_mapping(
    value: Any, expected: Sequence[str], location: str
) -> Mapping[str, Any]:
    frames = _mapping(value, location)
    if list(frames.keys()) != list(expected):
        raise ArtifactError(
            f"{location} must contain exactly the ordered camera frame set"
        )
    return frames


def _frame_shape(value: Any, location: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise ArtifactError(f"{location} must be [positive_height, positive_width]")
    return int(value[0]), int(value[1])


def _validate_lama_chain(
    lama_path: Path,
    lama: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any]]:
    """Validate the real LaMa input -> completion provenance chain."""

    del lama_path  # Its own strict manifest check is performed by the caller.
    input_path = _verified_artifact(
        lama.get("input_manifest"), "LaMa completion.input_manifest"
    )
    input_path, input_manifest = _strict_manifest(
        input_path, "LaMa input manifest", LAMA_INPUT_KIND
    )
    if lama.get("input_artifact_id") != input_manifest.get("artifact_id"):
        raise ArtifactError(
            "LaMa completion input_artifact_id does not match its input manifest"
        )

    expected = _expected_frames(
        input_manifest.get("parameters"), "LaMa input parameters"
    )
    if (
        _expected_frames(lama.get("parameters"), "LaMa completion parameters")
        != expected
    ):
        raise ArtifactError("LaMa input and completion frame sets differ")
    input_frames = _frame_mapping(
        input_manifest.get("frames"), expected, "LaMa input frames"
    )
    completion_frames = _frame_mapping(
        lama.get("frames"), expected, "LaMa completion frames"
    )

    model = _mapping(lama.get("model"), "LaMa completion.model")
    for name in ("config", "checkpoint"):
        _verified_artifact(model.get(name), f"LaMa completion.model.{name}")

    normal_required = "normal" in input_manifest["parameters"].get("required_modalities", [])
    if normal_required != ("normal" in lama["parameters"].get("required_modalities", [])):
        raise ArtifactError("LaMa input/completion normal modalities differ")
    if normal_required:
        _verified_artifact(input_manifest.get("normal_camera"), "LaMa normal camera snapshot")
        normal_metadata = _mapping(input_manifest["parameters"].get("normal"), "LaMa normal metadata")
        for name in ("render_manifest", "camera_manifest"):
            _verified_artifact(normal_metadata.get(name), f"LaMa normal {name}")
        _verified_artifact(lama.get("normal_prediction"), "LaMa normal prediction receipt")

    for stem in expected:
        source = _mapping(input_frames.get(stem), f"LaMa input frames.{stem}")
        source_shape = _frame_shape(
            source.get("shape"), f"LaMa input frames.{stem}.shape"
        )
        source_inputs = _mapping(
            source.get("inputs"), f"LaMa input frames.{stem}.inputs"
        )
        input_names = ("mask", "removed_rgb", "removed_depth", "reference_depth")
        if normal_required:
            input_names += ("removed_normal", "removed_normal_valid", "removed_alpha")
        for name in input_names:
            _verified_artifact(
                source_inputs.get(name), f"LaMa input frames.{stem}.inputs.{name}"
            )
        source_outputs = _mapping(
            source.get("outputs"), f"LaMa input frames.{stem}.outputs"
        )
        prepared_names = (
            "color",
            "color_mask",
            "depth",
            "depth_mask",
            "reference_depth",
        )
        if normal_required:
            prepared_names += ("normal", "normal_mask", "normal_valid", "normal_inference_mask")
        for name in prepared_names:
            _verified_artifact(
                source_outputs.get(name), f"LaMa input frames.{stem}.outputs.{name}"
            )

        completed = _mapping(
            completion_frames.get(stem), f"LaMa completion frames.{stem}"
        )
        if (
            _frame_shape(completed.get("shape"), f"LaMa completion frames.{stem}.shape")
            != source_shape
        ):
            raise ArtifactError(
                f"LaMa completion frame {stem} shape differs from its input"
            )
        completed_outputs = _mapping(
            completed.get("outputs"), f"LaMa completion frames.{stem}.outputs"
        )
        output_names = ("color", "depth")
        if normal_required:
            output_names += ("normal", "normal_valid", "normal_vis")
        for name in output_names:
            _verified_artifact(
                completed_outputs.get(name),
                f"LaMa completion frames.{stem}.outputs.{name}",
            )
    return input_path, input_manifest


def _validate_workspace_chain(
    workspace_path: Path,
    workspace: Mapping[str, Any],
    removal_path: Path,
    removal: Mapping[str, Any],
    tracking_path: Path,
    tracking: Mapping[str, Any],
    tracking_id: str,
    source_iteration: int,
) -> tuple[Path, Mapping[str, Any]]:
    """Validate independently supplied removal/tracking/camera workspace inputs."""

    del workspace_path  # Its own strict manifest check is performed by the caller.
    parameters = _mapping(workspace.get("parameters"), "inpaint workspace parameters")
    if parameters.get("source_iteration") != source_iteration:
        raise ArtifactError("inpaint workspace source iteration mismatch")

    workspace_inputs = _mapping(workspace.get("inputs"), "inpaint workspace inputs")
    stable_workspace_identity = workspace.get("identity_version") == 2

    removal_record = workspace_inputs.get("removal_manifest")
    removal_input = (
        _recorded_path(removal_record, "inpaint workspace inputs.removal_manifest")
        if stable_workspace_identity
        else _verified_artifact(
            removal_record, "inpaint workspace inputs.removal_manifest"
        )
    )
    if removal_input != removal_path:
        raise ArtifactError(
            "inpaint workspace removal_manifest does not match the supplied input"
        )

    # Tracking schema 2 has no producer artifact_id. Its fallback identity is
    # the complete manifest SHA, so keep the strict snapshot verification.
    tracking_input = _verified_artifact(
        workspace_inputs.get("tracking_session"),
        "inpaint workspace inputs.tracking_session",
    )
    if tracking_input != tracking_path:
        raise ArtifactError(
            "inpaint workspace tracking_session does not match the supplied input"
        )

    _reference(
        workspace,
        "removal_manifest",
        removal_path,
        str(removal["artifact_id"]),
        "inpaint workspace manifest",
    )
    _reference(
        workspace,
        "tracking_session",
        tracking_path,
        tracking_id,
        "inpaint workspace manifest",
    )

    camera_record = workspace_inputs.get("camera_manifest")
    camera_path = (
        _recorded_path(camera_record, "inpaint workspace inputs.camera_manifest")
        if stable_workspace_identity
        else _verified_artifact(
            camera_record,
            "inpaint workspace inputs.camera_manifest",
        )
    )
    camera_path, camera = _strict_manifest(
        camera_path, "virtual-camera manifest", CAMERA_KIND
    )
    camera_id = _validate_camera_artifact_id(camera)
    _reference(
        workspace,
        "camera_manifest",
        camera_path,
        camera_id,
        "inpaint workspace manifest",
    )
    expected = [f"{index:05d}" for index in range(camera["frame_count"])]
    cameras = camera.get("cameras")
    names = (
        [
            value.get("image_name") if isinstance(value, Mapping) else None
            for value in cameras
        ]
        if isinstance(cameras, list)
        else []
    )
    if (
        camera.get("frame_count") != len(expected)
        or camera.get("iteration") != source_iteration
        or names != expected
    ):
        raise ArtifactError(
            "virtual-camera manifest must contain ordered camera frame set "
            "for the source iteration"
        )

    tracking_camera = _mapping(
        tracking.get("input_cameras"), "tracking session input_cameras"
    )
    if stable_workspace_identity:
        tracked_camera_path = _recorded_path(
            tracking_camera, "tracking session input_cameras"
        )
        if (
            tracked_camera_path != camera_path
            or tracking_camera.get("artifact_id") != camera_id
        ):
            raise ArtifactError(
                "tracking session input_cameras does not match the logical camera artifact"
            )
    else:
        legacy_camera_record = dict(
            _mapping(workspace_inputs.get("camera_manifest"), "camera record")
        )
        legacy_camera_record["artifact_id"] = camera_id
        if tracking_camera != legacy_camera_record:
            raise ArtifactError(
                "tracking session input_cameras does not match the workspace camera manifest"
            )
    return camera_path, camera


def _fusion_upstream(
    fusion: Mapping[str, Any],
    key: str,
    expected_path: Path,
    expected_id: str,
) -> None:
    upstream = _mapping(fusion.get("upstream"), "RGB-D fusion.upstream")
    reference = _mapping(upstream.get(key), f"RGB-D fusion.upstream.{key}")
    artifact = reference.get("artifact")
    actual = _verified_artifact(artifact, f"RGB-D fusion.upstream.{key}.artifact")
    if actual != expected_path:
        raise ArtifactError(f"RGB-D fusion upstream {key} path mismatch")
    if reference.get("artifact_id") != expected_id:
        raise ArtifactError(f"RGB-D fusion upstream {key} artifact_id mismatch")


def _workspace_frame_record(
    workspace: Mapping[str, Any], set_name: str, filename: str
) -> Mapping[str, Any]:
    sets = _mapping(workspace.get("frame_sets"), "inpaint workspace.frame_sets")
    frame_set = _mapping(sets.get(set_name), f"inpaint workspace.frame_sets.{set_name}")
    files = frame_set.get("files")
    if not isinstance(files, list):
        raise ArtifactError(f"inpaint workspace frame set {set_name} has no files")
    matches = [
        value
        for value in files
        if isinstance(value, Mapping) and value.get("file") == filename
    ]
    if len(matches) != 1:
        raise ArtifactError(
            f"inpaint workspace frame set {set_name} must contain {filename} exactly once"
        )
    return matches[0]


def _validate_fusion_chain(
    fusion_path: Path,
    fusion: Mapping[str, Any],
    camera_path: Path,
    camera: Mapping[str, Any],
    lama_path: Path,
    lama: Mapping[str, Any],
    lama_input: Mapping[str, Any],
    workspace: Mapping[str, Any],
    source_iteration: int,
) -> None:
    """Validate fusion's true camera/LaMa edges and every frame artifact."""

    del fusion_path  # Its own strict manifest check is performed by the caller.
    parameters = _mapping(fusion.get("parameters"), "RGB-D fusion.parameters")
    if (
        parameters.get("iteration") != source_iteration
        or parameters.get("frame_count") != camera["frame_count"]
        or not isinstance(parameters.get("write_hole_ply"), bool)
    ):
        raise ArtifactError(
            "RGB-D fusion must declare the camera frame count for the source iteration and a "
            "boolean write_hole_ply"
        )
    write_hole = bool(parameters["write_hole_ply"])
    identity_version = fusion.get("identity_version")
    if identity_version not in (None, FUSION_IDENTITY_VERSION):
        raise ArtifactError(
            f"RGB-D fusion identity_version {identity_version!r} is unsupported"
        )
    content_addressed = identity_version == FUSION_IDENTITY_VERSION
    _fusion_upstream(
        fusion,
        "camera_manifest",
        camera_path,
        str(camera["artifact_id"]),
    )
    _fusion_upstream(
        fusion,
        "lama_completion_manifest",
        lama_path,
        str(lama["artifact_id"]),
    )

    expected = [f"{index:05d}" for index in range(camera["frame_count"])]
    inputs = fusion.get("inputs")
    outputs = fusion.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ArtifactError("RGB-D fusion inputs and outputs must be arrays")
    if [
        value.get("frame") if isinstance(value, Mapping) else None for value in inputs
    ] != expected:
        raise ArtifactError("RGB-D fusion inputs must be ordered camera frame set")
    if [
        value.get("frame") if isinstance(value, Mapping) else None for value in outputs
    ] != expected:
        raise ArtifactError("RGB-D fusion outputs must be ordered camera frame set")

    if content_addressed:
        expected_upstream_ids = {
            "camera_manifest": str(camera["artifact_id"]),
            "lama_completion_manifest": str(lama["artifact_id"]),
        }
        if fusion.get("upstream_artifact_ids") != expected_upstream_ids:
            raise ArtifactError(
                "RGB-D fusion upstream_artifact_ids do not match its inputs"
            )
        expected_artifact_id = fusion_artifact_id(fusion)
        if fusion.get("artifact_id") != expected_artifact_id:
            raise ArtifactError(
                "RGB-D fusion artifact_id does not match its content-addressed payload"
            )

    completion_frames = _mapping(lama.get("frames"), "LaMa completion.frames")
    input_frames = _mapping(lama_input.get("frames"), "LaMa input.frames")
    for stem, input_value, output_value in zip(expected, inputs, outputs):
        frame_input = _mapping(input_value, f"RGB-D fusion.inputs[{stem}]")
        frame_output = _mapping(output_value, f"RGB-D fusion.outputs[{stem}]")
        completed_outputs = _mapping(
            _mapping(completion_frames.get(stem), f"LaMa completion.frames.{stem}").get(
                "outputs"
            ),
            f"LaMa completion.frames.{stem}.outputs",
        )
        prepared_outputs = _mapping(
            _mapping(input_frames.get(stem), f"LaMa input.frames.{stem}").get(
                "outputs"
            ),
            f"LaMa input.frames.{stem}.outputs",
        )
        for fusion_key, expected_record in (
            ("completed_rgb", completed_outputs.get("color")),
            ("completed_depth", completed_outputs.get("depth")),
            ("inpaint_mask", prepared_outputs.get("color_mask")),
        ):
            actual = _verified_artifact(
                frame_input.get(fusion_key),
                f"RGB-D fusion.inputs[{stem}].{fusion_key}",
            )
            expected_path = _verified_artifact(
                expected_record,
                f"upstream artifact for RGB-D fusion frame {stem} {fusion_key}",
            )
            if actual != expected_path:
                raise ArtifactError(
                    f"RGB-D fusion frame {stem} {fusion_key} does not match LaMa"
                )

        if write_hole:
            for key, set_name, suffix in (
                ("removed_rgb", "removed_render", ".png"),
                ("removed_depth", "removed_depth", ".npy"),
            ):
                actual = _verified_artifact(
                    frame_input.get(key),
                    f"RGB-D fusion.inputs[{stem}].{key}",
                )
                expected_path = _verified_artifact(
                    _workspace_frame_record(workspace, set_name, f"{stem}{suffix}"),
                    f"inpaint workspace {set_name} frame {stem}",
                )
                if actual != expected_path:
                    raise ArtifactError(
                        f"RGB-D fusion frame {stem} {key} does not match the workspace"
                    )
        elif "removed_rgb" in frame_input or "removed_depth" in frame_input:
            raise ArtifactError(
                f"RGB-D fusion frame {stem} has removal inputs while write_hole_ply is false"
            )

        _verified_artifact(
            frame_output.get("fused_mask_ply"),
            f"RGB-D fusion.outputs[{stem}].fused_mask_ply",
            verify_mtime=not content_addressed,
        )
        if write_hole:
            _verified_artifact(
                frame_output.get("fused_hole_ply"),
                f"RGB-D fusion.outputs[{stem}].fused_hole_ply",
                verify_mtime=not content_addressed,
            )
        elif "fused_hole_ply" in frame_output:
            raise ArtifactError(
                f"RGB-D fusion frame {stem} has fused_hole_ply while disabled"
            )


def _tracking_manifest(path: Path) -> tuple[Path, Mapping[str, Any], str]:
    resolved, payload = _strict_manifest(
        path,
        "tracking session",
        TRACKING_KIND,
        expected_schema=2,
        require_artifact_id=False,
    )
    identity = payload.get("artifact_id")
    if not isinstance(identity, str) or not identity:
        identity = _sha256(resolved)
    return resolved, payload, identity


def _ids_from_manifest(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise ArtifactError(f"{label} must be a list of positive integer IDs")
    if len(value) != len(set(value)):
        raise ArtifactError(f"{label} contains duplicate IDs")
    return sorted(value)


def _reference(
    manifest: Mapping[str, Any],
    key: str,
    expected_path: Path,
    expected_id: str,
    label: str,
) -> None:
    record: Mapping[str, Any] | None = None
    # Prefer the explicit identity edge.  ``inputs`` may also contain a hashed
    # file record with the same key but intentionally no artifact_id.
    for section_name in ("upstream", "inputs"):
        section = manifest.get(section_name)
        if isinstance(section, Mapping) and isinstance(section.get(key), Mapping):
            record = section[key]
            break
    if record is None:
        raise ArtifactError(f"{label} has no {key!r} manifest reference")
    if record.get("artifact_id") != expected_id:
        raise ArtifactError(
            f"{label} {key} artifact_id does not match its upstream manifest"
        )
    path_text = record.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise ArtifactError(f"{label} {key} reference has no path")
    try:
        actual = Path(path_text).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactError(f"{label} {key} path does not exist: {path_text}") from exc
    if actual != expected_path:
        raise ArtifactError(
            f"{label} {key} path resolves to {actual}, expected {expected_path}"
        )


def _load_json(path: Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    resolved = _regular_file(path, label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"{label} must contain a JSON object")
    return resolved, payload


def _validate_config(
    path: Path,
    output_iteration: int | None,
    target_ids: list[int],
    surrounding_ids: list[int],
    removal_threshold: float,
) -> tuple[Path, Mapping[str, Any]]:
    resolved, config = _load_json(path, "inpaint config")
    expected = {
        "finetune_iteration": output_iteration,
        "target_id": target_ids,
        "surrounding_ids": surrounding_ids,
        "select_obj_id": target_ids + surrounding_ids,
    }
    if output_iteration is None:
        # The peer pipeline owns its iteration budget; old RGB settings are
        # still recorded but must not constrain joint optimization.
        expected.pop("finetune_iteration")
    for name, value in expected.items():
        if config.get(name) != value:
            raise ArtifactError(
                f"inpaint config {name}={config.get(name)!r}, expected {value!r}"
            )
    actual_threshold = config.get("removal_thresh")
    if (
        isinstance(actual_threshold, bool)
        or not isinstance(actual_threshold, (int, float))
        or not math.isclose(
            float(actual_threshold), removal_threshold, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ArtifactError(
            "inpaint config removal_thresh does not match the removal manifest"
        )
    return resolved, config


def _legacy_model_matches(
    existing: Mapping[str, Any], current_identity: Mapping[str, Any]
) -> bool:
    """Accept only a self-consistent v1 model with identical logical inputs."""

    try:
        if (
            existing.get("schema_version") != MODEL_SCHEMA_VERSION
            or existing.get("identity_version") is not None
            or existing.get("parameters") != current_identity["parameters"]
            or existing.get("upstream_artifact_ids")
            != current_identity["upstream_artifact_ids"]
        ):
            return False

        inputs = _mapping(existing.get("inputs"), "legacy model inputs")
        legacy_hashes: dict[str, str] = {}
        for name, value in inputs.items():
            record = _mapping(value, f"legacy model inputs.{name}")
            digest = record.get("sha256")
            if not isinstance(digest, str) or not digest:
                return False
            legacy_hashes[str(name)] = digest
        if set(CONTENT_INPUT_NAMES) - legacy_hashes.keys():
            return False
        for name, digest in current_identity["content_hashes"].items():
            if legacy_hashes.get(name) != digest:
                return False

        legacy_identity = {
            "kind": MODEL_KIND,
            "schema_version": MODEL_SCHEMA_VERSION,
            "parameters": current_identity["parameters"],
            "upstream_artifact_ids": current_identity["upstream_artifact_ids"],
            "input_hashes": legacy_hashes,
        }
        return existing.get("artifact_id") == _identity(legacy_identity)
    except (ArtifactError, KeyError, TypeError):
        return False


def _legacy_model_can_migrate(
    existing: Mapping[str, Any],
    current_identity: Mapping[str, Any],
    output: Path,
) -> bool:
    """Migrate v1 only before render, mesh, semantic, or final consumers exist."""

    if not _legacy_model_matches(existing, current_identity):
        return False
    downstream = (
        output / "train",
        output / "test",
        output / "mesh",
        output.parent / "inpainted_mesh",
        output.parent / "inpaint_manifest.json",
    )
    return not any(path.exists() or path.is_symlink() for path in downstream)


def publish_inpainted_model(
    inpainted_ply: Path,
    classifier: Path,
    edgs_config: Path,
    cfg_args: Path,
    source_iteration: int,
    output_iteration: int,
    target_ids_text: str,
    surrounding_ids_text: str,
    removed_model_manifest_path: Path,
    removal_manifest_path: Path,
    workspace_manifest_path: Path,
    tracking_session_path: Path,
    lama_manifest_path: Path,
    fusion_manifest_path: Path,
    inpaint_config_path: Path,
    output: Path,
    *,
    fusion_seed_frame: int = 4,
    local_geometry_manifest_path: Path | None = None,
    density_manifest_path: Path | None = None,
    rgb_manifest_path: Path | None = None,
    edgs_joint_manifest_path: Path | None = None,
) -> dict[str, Any]:
    for value, name in (
        (source_iteration, "--source-iteration"),
        (output_iteration, "--output-iteration"),
    ):
        if isinstance(value, bool) or value <= 0:
            raise ArtifactError(f"{name} must be a positive integer")
    if (
        isinstance(fusion_seed_frame, bool)
        or not isinstance(fusion_seed_frame, int)
        or fusion_seed_frame < 0
    ):
        raise ArtifactError("--fusion-seed-frame must be a nonnegative integer")

    inpainted_ply = _regular_file(inpainted_ply, "inpainted Gaussian PLY")
    classifier = _regular_file(classifier, "classifier")
    cfg_args = _regular_file(cfg_args, "cfg_args")
    config_path, _, sh_degree = _load_edgs_config(edgs_config)
    cfg_values = _parse_namespace(cfg_args)
    if cfg_values.get("sh_degree") != sh_degree:
        raise ArtifactError("cfg_args sh_degree does not match EDGS config")
    num_classes = cfg_values.get("num_classes")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 0
    ):
        raise ArtifactError("cfg_args num_classes must be a positive integer")
    target_ids, surrounding_ids = _validate_selection(
        target_ids_text, surrounding_ids_text, num_classes=num_classes
    )

    removed_model_path, removed_model = _strict_manifest(
        removed_model_manifest_path, "removed model manifest", REMOVED_MODEL_KIND
    )
    removal_path, removal = _strict_manifest(
        removal_manifest_path, "removal manifest", REMOVAL_KIND
    )
    workspace_path, workspace = _strict_manifest(
        workspace_manifest_path, "inpaint workspace manifest", WORKSPACE_KIND
    )
    tracking_path, tracking, tracking_id = _tracking_manifest(tracking_session_path)
    lama_path, lama = _strict_manifest(
        lama_manifest_path, "LaMa completion manifest", LAMA_KIND
    )
    direct = edgs_joint_manifest_path is not None
    if not direct and fusion_manifest_path is None:
        raise ArtifactError("Inpaint360GS publication requires --fusion-manifest")
    if direct and any(p is not None for p in (fusion_manifest_path, local_geometry_manifest_path, density_manifest_path, rgb_manifest_path)):
        raise ArtifactError("EDGS joint publication cannot include old-path fusion/5a/5b inputs")
    fusion_path, fusion = (None, None) if direct else _strict_manifest(
        fusion_manifest_path, "RGB-D fusion manifest", FUSION_KIND)

    removed_parameters = _mapping(
        removed_model.get("parameters"), "removed model manifest.parameters"
    )
    if removed_parameters.get("iteration") != source_iteration:
        raise ArtifactError("removed model iteration does not match source iteration")
    if (
        _ids_from_manifest(removed_parameters.get("target_ids"), "removed target IDs")
        != target_ids
    ):
        raise ArtifactError("target IDs do not match the removed model")
    if (
        _ids_from_manifest(
            removed_parameters.get("surrounding_ids"), "removed surrounding IDs"
        )
        != surrounding_ids
    ):
        raise ArtifactError("surrounding IDs do not match the removed model")
    threshold = removed_parameters.get("removal_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ArtifactError("removed model has no numeric removal_threshold")
    removal_threshold = float(threshold)

    if (
        _ids_from_manifest(removal.get("target_ids"), "removal target IDs")
        != target_ids
    ):
        raise ArtifactError("target IDs do not match the removal result")
    if (
        _ids_from_manifest(removal.get("surrounding_ids"), "removal surrounding IDs")
        != surrounding_ids
    ):
        raise ArtifactError("surrounding IDs do not match the removal result")
    removal_parameters = _mapping(removal.get("parameters"), "removal parameters")
    if removal_parameters.get("distill_iteration") != source_iteration:
        raise ArtifactError("removal source iteration mismatch")
    actual_removal_threshold = removal_parameters.get("removal_threshold")
    if (
        isinstance(actual_removal_threshold, bool)
        or not isinstance(actual_removal_threshold, (int, float))
        or not math.isclose(
            float(actual_removal_threshold),
            removal_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ArtifactError("removal threshold mismatch")
    _reference(
        removal,
        "model_manifest",
        removed_model_path,
        str(removed_model["artifact_id"]),
        "removal manifest",
    )

    camera_path, camera = _validate_workspace_chain(
        workspace_path,
        workspace,
        removal_path,
        removal,
        tracking_path,
        tracking,
        tracking_id,
        source_iteration,
    )
    if not direct and fusion_seed_frame >= camera["frame_count"]:
        raise ArtifactError("fusion-seed-frame is outside the camera frame set")
    lama_input_path, lama_input = _validate_lama_chain(lama_path, lama)
    if lama["parameters"]["frames"] != camera["frame_count"]:
        raise ArtifactError("LaMa frame count differs from virtual cameras")
    if not direct:
        _validate_fusion_chain(
            fusion_path, fusion, camera_path, camera, lama_path, lama,
            lama_input, workspace, source_iteration,
        )

    inpaint_config_path, inpaint_config = _validate_config(
        inpaint_config_path,
        None if direct else output_iteration,
        target_ids,
        surrounding_ids,
        removal_threshold,
    )
    removed_inputs = _mapping(removed_model.get("inputs"), "removed model inputs")
    for key, actual, label in (
        ("classifier", classifier, "classifier"),
        ("edgs_config", config_path, "EDGS config"),
        ("cfg_args", cfg_args, "cfg_args"),
    ):
        record = _mapping(removed_inputs.get(key), f"removed model inputs.{key}")
        _verify_manifest_artifact(record, actual, f"removed model inputs.{label}")

    point_count, embedding_properties = _validate_removed_ply(inpainted_ply, sh_degree)
    output = _absolute_output(output)
    sources = (
        inpainted_ply,
        classifier,
        config_path,
        cfg_args,
        removed_model_path,
        removal_path,
        workspace_path,
        tracking_path,
        camera_path,
        lama_input_path,
        lama_path,
        fusion_path,
        inpaint_config_path,
    )
    _ensure_independent_output(output, tuple(p for p in sources if p is not None))
    joint = None
    if direct:
        joint = _validate_edgs_joint(edgs_joint_manifest_path, inpainted_ply, lama_path, camera_path)
        from edgs_inpaint_io import validate_init, verify_record
        initial = validate_init(joint["inputs"]["initialization"]["path"])
        if joint["parameters"]["iterations"] != output_iteration:
            raise ArtifactError("EDGS joint iteration differs from publish iteration")
        for key, path in (("classifier", classifier), ("removed_model", removed_model_path),
                          ("inpaint_config", inpaint_config_path)):
            if verify_record(initial["inputs"][key]) != path:
                raise ArtifactError(f"EDGS initialization {key} differs from publish chain")
        if verify_record(joint["inputs"]["edgs_config"]) != config_path:
            raise ArtifactError("EDGS joint config differs from publish chain")
    local_geometry = None
    if local_geometry_manifest_path is not None:
        local_geometry = _validate_local_geometry(
            local_geometry_manifest_path, inpainted_ply, lama_path, camera_path)
        local_parameters = local_geometry["parameters"]
        if local_parameters["rgb_iterations"] != output_iteration:
            raise ArtifactError("local geometry RGB iteration differs from publish iteration")
        from local_geometry_io import read_json, verify_record
        rgb_receipt = read_json(verify_record(local_geometry["inputs"]["rgb_manifest"]))
        if bool(rgb_receipt["inputs"].get("density")) != bool(density_manifest_path):
            raise ArtifactError("local geometry density provenance must be published explicitly")
        for key, expected in (("classifier", classifier), ("fusion", fusion_path),
                              ("inpaint_config", inpaint_config_path)):
            if verify_record(rgb_receipt["inputs"][key]) != expected:
                raise ArtifactError(f"local geometry {key} differs from published chain")
        if verify_record(local_geometry["inputs"]["edgs_config"]) != config_path:
            raise ArtifactError("local geometry EDGS config differs from published chain")
    density = None
    if density_manifest_path is not None:
        if rgb_manifest_path is None:
            raise ArtifactError("density publishing requires RGB receipt")
        density = _validate_density(density_manifest_path, rgb_manifest_path, inpainted_ply,
            local_geometry_manifest_path, lama_path, camera_path, fusion_path, inpaint_config_path)
    inputs = {
        "inpainted_gaussian_ply": _artifact(inpainted_ply, "inpainted Gaussian PLY"),
        "classifier": _artifact(classifier, "classifier"),
        "edgs_config": _artifact(config_path, "EDGS config"),
        "cfg_args": _artifact(cfg_args, "cfg_args"),
        "removed_model_manifest": _artifact(
            removed_model_path, "removed model manifest"
        ),
        "removal_manifest": _artifact(removal_path, "removal manifest"),
        "workspace_manifest": _artifact(workspace_path, "inpaint workspace manifest"),
        "tracking_session": _artifact(tracking_path, "tracking session"),
        "camera_manifest": _artifact(camera_path, "virtual-camera manifest"),
        "lama_input_manifest": _artifact(lama_input_path, "LaMa input manifest"),
        "lama_manifest": _artifact(lama_path, "LaMa completion manifest"),
        "inpaint_config": _artifact(inpaint_config_path, "inpaint config"),
    }
    upstream_ids = {
        "removed_model": removed_model["artifact_id"],
        "removal": removal["artifact_id"],
        "workspace": workspace["artifact_id"],
        "tracking": tracking_id,
        "camera": camera["artifact_id"],
        "lama_inputs": lama_input["artifact_id"],
        "lama": lama["artifact_id"],
    }
    if not direct:
        inputs["fusion_manifest"] = _artifact(fusion_path, "RGB-D fusion manifest")
        upstream_ids["fusion"] = fusion["artifact_id"]
    else:
        inputs["edgs_joint_manifest"] = _artifact(Path(edgs_joint_manifest_path).resolve(), "EDGS joint manifest")
        upstream_ids["edgs_joint"] = joint["artifact_id"]
    content_hashes = {name: inputs[name]["sha256"] for name in CONTENT_INPUT_NAMES}
    if local_geometry is not None:
        inputs["local_geometry_manifest"] = _artifact(
            Path(local_geometry_manifest_path).resolve(), "local geometry manifest")
        upstream_ids["local_geometry"] = local_geometry["artifact_id"]
    if density is not None:
        inputs["density_manifest"] = _artifact(Path(density_manifest_path).resolve(), "support density manifest")
        inputs["rgb_manifest"] = _artifact(Path(rgb_manifest_path).resolve(), "RGB finetune manifest")
        upstream_ids["support_density"] = density["artifact_id"]
    identity_payload = {
        "kind": MODEL_KIND,
        "schema_version": MODEL_SCHEMA_VERSION,
        "identity_version": MODEL_IDENTITY_VERSION,
        "parameters": {
            "source_iteration": source_iteration,
            "output_iteration": output_iteration,
            "target_ids": target_ids,
            "surrounding_ids": surrounding_ids,
            "removal_threshold": removal_threshold,
            "fusion_seed_frame": fusion_seed_frame,
        },
        "upstream_artifact_ids": upstream_ids,
        "content_hashes": content_hashes,
    }
    if local_geometry is not None:
        identity_payload["parameters"].update(
            local_geometry_refine=True,
            rgb_iterations=output_iteration,
            local_geometry_iterations=local_geometry["parameters"]["local_geometry_iterations"],
        )
    if direct:
        identity_payload["parameters"].pop("fusion_seed_frame")
        identity_payload["parameters"].update(pipeline="edgs-pgsr", joint_iterations=output_iteration)
    if density is not None:
        identity_payload["parameters"]["support_density_mode"] = density['parameters']['mode']
    artifact_id = _identity(identity_payload)
    action, existing = _preflight_output(
        output,
        artifact_id,
        manifest_name=MANIFEST_NAME,
        expected_kind=MODEL_KIND,
        recoverable_partial=lambda path: _recoverable_inpainted_model_output(
            path, output_iteration
        ),
        compatible_existing=lambda value: _legacy_model_can_migrate(
            value, identity_payload, output
        ),
    )
    if existing is not None and (
        existing.get("schema_version") != MODEL_SCHEMA_VERSION
        or existing.get("status") != "complete"
    ):
        raise ArtifactError("existing inpainted model manifest is invalid")
    if (
        existing is not None
        and action != "migrate"
        and (existing.get("identity_version") != MODEL_IDENTITY_VERSION)
    ):
        raise ArtifactError("existing inpainted model identity is unsupported")

    created_at = datetime.now(timezone.utc).isoformat()
    if existing is not None and isinstance(existing.get("created_at"), str):
        created_at = str(existing["created_at"])
    iteration_root = output / "point_cloud" / f"iteration_{output_iteration}"
    config_link = output / "config.yaml"
    cfg_link = output / "cfg_args"
    ply_link = iteration_root / "point_cloud.ply"
    classifier_link = iteration_root / "classifier.pth"
    manifest: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "identity_version": MODEL_IDENTITY_VERSION,
        "kind": MODEL_KIND,
        "complete": True,
        "status": "complete",
        "artifact_id": artifact_id,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": identity_payload["parameters"],
        "upstream_artifact_ids": upstream_ids,
        "inputs": inputs,
        "inpaint_config": dict(inpaint_config),
        "gaussian": {
            "point_count": point_count,
            "sh_degree": sh_degree,
            "object_embedding_dimensions": 16,
            "object_embedding_properties": embedding_properties,
        },
        "model": {
            "root": str(output),
            "config": "config.yaml",
            "config_link_target": _relative_link_target(config_path, config_link),
            "cfg_args": "cfg_args",
            "cfg_args_link_target": _relative_link_target(cfg_args, cfg_link),
            "point_cloud": (
                f"point_cloud/iteration_{output_iteration}/point_cloud.ply"
            ),
            "point_cloud_storage": {
                "type": "regular_copy",
                "source_input": "inpainted_gaussian_ply",
                "sha256": inputs["inpainted_gaussian_ply"]["sha256"],
            },
            "classifier": (f"point_cloud/iteration_{output_iteration}/classifier.pth"),
            "classifier_link_target": _relative_link_target(
                classifier, classifier_link
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    _ensure_managed_directory(output / "point_cloud", output)
    _ensure_managed_directory(iteration_root, output)
    _atomic_relative_symlink(config_path, config_link)
    _atomic_relative_symlink(cfg_args, cfg_link)
    _atomic_relative_symlink(classifier, classifier_link)
    point_cloud_action = _atomic_materialize_file(
        inpainted_ply,
        ply_link,
        expected_sha256=str(inputs["inpainted_gaussian_ply"]["sha256"]),
    )
    # A logical refresh repairs managed links but leaves the manifest byte-stable
    # so downstream identities cannot drift on ``updated_at`` alone.
    if action != "refresh":
        _atomic_write_json(output / MANIFEST_NAME, manifest)
    return {
        "action": action,
        "artifact_id": artifact_id,
        "source_iteration": source_iteration,
        "output_iteration": output_iteration,
        "point_count": point_count,
        "output": str(output),
        "point_cloud": str(ply_link),
        "point_cloud_action": point_cloud_action,
        "point_cloud_storage": "regular_copy",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish an Inpaint360GS checkpoint as an EDGS model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inpainted-ply", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--edgs-config", type=Path, required=True)
    parser.add_argument("--cfg-args", type=Path, required=True)
    parser.add_argument("--source-iteration", type=int, required=True)
    parser.add_argument("--output-iteration", type=int, required=True)
    parser.add_argument("--target-ids", required=True)
    parser.add_argument("--surrounding-ids", required=True)
    parser.add_argument("--removed-model-manifest", type=Path, required=True)
    parser.add_argument("--removal-manifest", type=Path, required=True)
    parser.add_argument("--workspace-manifest", type=Path, required=True)
    parser.add_argument("--tracking-session", type=Path, required=True)
    parser.add_argument("--lama-manifest", type=Path, required=True)
    parser.add_argument("--fusion-manifest", type=Path)
    parser.add_argument("--edgs-joint-manifest", type=Path)
    parser.add_argument("--fusion-seed-frame", type=int, default=4)
    parser.add_argument("--density-manifest", type=Path)
    parser.add_argument("--rgb-manifest", type=Path)
    parser.add_argument("--local-geometry-manifest", type=Path)
    parser.add_argument("--inpaint-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = publish_inpainted_model(
            args.inpainted_ply,
            args.classifier,
            args.edgs_config,
            args.cfg_args,
            args.source_iteration,
            args.output_iteration,
            args.target_ids,
            args.surrounding_ids,
            args.removed_model_manifest,
            args.removal_manifest,
            args.workspace_manifest,
            args.tracking_session,
            args.lama_manifest,
            args.fusion_manifest,
            args.inpaint_config,
            args.output,
            fusion_seed_frame=args.fusion_seed_frame,
            local_geometry_manifest_path=args.local_geometry_manifest,
            edgs_joint_manifest_path=args.edgs_joint_manifest,
            density_manifest_path=args.density_manifest,
            rgb_manifest_path=args.rgb_manifest,
        )
    except (ArtifactError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
