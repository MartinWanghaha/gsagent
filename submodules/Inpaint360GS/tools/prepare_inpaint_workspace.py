#!/usr/bin/env python3
"""Create an isolated Inpaint360GS workspace from a completed removal run.

The removal workspace is treated as immutable input.  Only the semantic model,
the object-removal checkpoints, the three virtual-view input directories and
the verified tracker masks are exposed through relative symlinks.  Directories
written by LaMa fusion and 3DGS inpainting remain local to the new workspace.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tools.prepare_removal_workspace import (
        ArtifactError,
        _absolute_output,
        _artifact,
        _atomic_relative_symlink,
        _atomic_write_json,
        _ensure_independent_output,
        _identity,
        _mapping,
        _preflight_output,
        _regular_file,
        _relative_link_target,
        _sha256,
    )
except ModuleNotFoundError:  # Direct repository-local execution.
    from prepare_removal_workspace import (
        ArtifactError,
        _absolute_output,
        _artifact,
        _atomic_relative_symlink,
        _atomic_write_json,
        _ensure_independent_output,
        _identity,
        _mapping,
        _preflight_output,
        _regular_file,
        _relative_link_target,
        _sha256,
    )


WORKSPACE_KIND = "paintmesh-inpaint-workspace"
WORKSPACE_SCHEMA_VERSION = 1
WORKSPACE_IDENTITY_VERSION = 2
MANIFEST_NAME = "workspace_manifest.json"
REMOVAL_WORKSPACE_KIND = "paintmesh-removal-workspace"
REMOVAL_RESULT_KIND = "paintmesh-object-removal"
TRACKING_KIND = "paintmesh-tracking-session"
CAMERA_KIND = "inpaint360gs-virtual-cameras"


def _strict_manifest(
    path: Path,
    label: str,
    expected_kind: str,
    *,
    expected_schema: int = 1,
    require_artifact_id: bool = True,
) -> tuple[Path, Mapping[str, Any]]:
    resolved = _regular_file(path, label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError(f"{label} must contain a JSON object: {resolved}")
    if expected_kind == CAMERA_KIND:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "paintmesh_camera_contract", Path(__file__).resolve().parents[1] / "utils/virtual_camera_manifest.py")
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        try:
            contract.load_virtual_camera_manifest(resolved)
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc
        expected_schema = payload["schema_version"]
    if payload.get("schema_version") != expected_schema:
        raise ArtifactError(
            f"{label} must use schema_version {expected_schema}: {resolved}"
        )
    if payload.get("kind") != expected_kind:
        raise ArtifactError(
            f"{label} kind is {payload.get('kind')!r}, expected "
            f"{expected_kind!r}: {resolved}"
        )
    if payload.get("complete") is not True or payload.get("status") != "complete":
        raise ArtifactError(f"{label} is not complete: {resolved}")
    artifact_id = payload.get("artifact_id")
    if require_artifact_id and (
        not isinstance(artifact_id, str) or not artifact_id.strip()
    ):
        raise ArtifactError(f"{label} has no artifact_id: {resolved}")
    return resolved, payload


def _resolved_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ArtifactError(f"{label} must be a directory: {resolved}")
    return resolved


def _manifest_reference(
    manifest: Mapping[str, Any],
    section: str,
    key: str,
    expected_path: Path,
    expected_artifact_id: str,
    label: str,
) -> None:
    values = _mapping(manifest.get(section), f"{label}.{section}")
    record = _mapping(values.get(key), f"{label}.{section}.{key}")
    recorded_id = record.get("artifact_id")
    if recorded_id != expected_artifact_id:
        raise ArtifactError(
            f"{label}.{section}.{key}.artifact_id is {recorded_id!r}, "
            f"expected {expected_artifact_id!r}"
        )
    recorded_path = record.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ArtifactError(f"{label}.{section}.{key}.path is missing")
    try:
        resolved = Path(recorded_path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactError(
            f"path recorded by {label}.{section}.{key} does not exist: "
            f"{recorded_path}"
        ) from exc
    if resolved != expected_path:
        raise ArtifactError(
            f"{label}.{section}.{key}.path resolves to {resolved}, "
            f"expected {expected_path}"
        )


def _frame_names(tracking: Mapping[str, Any]) -> list[str]:
    names = tracking.get("expected_masks")
    if not isinstance(names, list) or not names:
        raise ArtifactError("tracking session has no frames")
    expected = [f"{index:05d}.png" for index in range(len(names))]
    if names != expected:
        raise ArtifactError(
            "tracking session must declare the exact ordered camera frame names"
        )
    return expected


def _snapshot_files(
    root: Path,
    names: Sequence[str],
    label: str,
) -> dict[str, Any]:
    root = _resolved_directory(root, label)
    records: list[dict[str, Any]] = []
    for name in names:
        if Path(name).name != name:
            raise ArtifactError(f"unsafe file name in {label}: {name!r}")
        record = _artifact(root / name, f"{label} frame {name}")
        record["file"] = name
        records.append(record)
    digest = _identity(
        {
            "kind": "paintmesh-frame-set",
            "schema_version": 1,
            "files": [
                {
                    "file": record["file"],
                    "sha256": record["sha256"],
                }
                for record in records
            ],
        }
    )
    return {"root": str(root), "artifact_id": digest, "files": records}


def _validate_tracking_outputs(
    tracking: Mapping[str, Any],
    masks: Mapping[str, Any],
) -> None:
    outputs = _mapping(tracking.get("outputs"), "tracking session.outputs")
    expected_records = outputs.get("masks")
    if not isinstance(expected_records, list) or len(expected_records) != len(masks["files"]):
        raise ArtifactError("tracking session must record all expected output masks")
    actual_by_name = {record["file"]: record for record in masks["files"]}
    for expected in expected_records:
        if not isinstance(expected, Mapping):
            raise ArtifactError("tracking session mask record must be an object")
        name = expected.get("file")
        if name not in actual_by_name:
            raise ArtifactError(
                f"tracking session records an unexpected mask: {name!r}"
            )
        actual = actual_by_name[name]
        for key in ("size_bytes", "mtime_ns", "sha256"):
            if expected.get(key) != actual[key]:
                raise ArtifactError(
                    f"tracked mask {name} changed: {key}={actual[key]!r}, "
                    f"manifest records {expected.get(key)!r}"
                )


def _recoverable_workspace(output: Path) -> bool:
    """Accept only the exact symlink skeleton this tool can leave behind."""

    allowed_top = {
        "cfg_args",
        "point_cloud",
        "point_cloud_object_removal",
        "tracking_masks",
        "virtual",
    }
    entries = {entry.name: entry for entry in output.iterdir()}
    if not entries or not set(entries).issubset(allowed_top):
        return False
    for name in allowed_top - {"virtual"}:
        path = entries.get(name)
        if path is not None and not path.is_symlink():
            return False
    virtual = entries.get("virtual")
    if virtual is None:
        return True
    if virtual.is_symlink() or not virtual.is_dir():
        return False
    for path in virtual.rglob("*"):
        if path.is_symlink():
            continue
        if not path.is_dir():
            return False
    return True


def _has_material_inpaint_outputs(output: Path, source_iteration: int) -> bool:
    """Report whether a legacy workspace has already produced mutable results.

    Changing the workspace artifact ID after fusion or fine-tuning would break
    the provenance chain of those results.  A skeleton-only Stage-1 workspace
    is safe to migrate; a workspace containing later-stage outputs is not.
    """

    candidates = (
        output / "point_cloud_object_inpaint_virtual",
        output
        / "virtual"
        / "ours_object_removal"
        / f"iteration_{source_iteration}"
        / "depth_completed",
        output
        / "virtual"
        / "ours_object_removal"
        / f"iteration_{source_iteration}"
        / "fused_mask_col_dep_ply",
        output
        / "virtual"
        / "ours_object_removal"
        / f"iteration_{source_iteration}"
        / "fused_hole_col_dep_ply",
    )
    return any(path.exists() or path.is_symlink() for path in candidates)


def _legacy_workspace_matches(
    existing: Mapping[str, Any],
    current_identity: Mapping[str, Any],
) -> bool:
    """Allow a one-time migration from the volatile version-1 identity.

    The old identity included raw hashes of upstream manifest files.  An
    idempotent upstream refresh changes ``updated_at`` and therefore that hash,
    even when the upstream artifact ID and every frame are unchanged.  A
    legacy manifest is accepted only if its own old artifact ID recomputes and
    all stable upstream/frame identities match the current request.
    """

    try:
        if (
            existing.get("schema_version") != WORKSPACE_SCHEMA_VERSION
            or existing.get("identity_version") is not None
        ):
            return False
        parameters = _mapping(existing.get("parameters"), "legacy parameters")
        if (
            parameters.get("source_iteration") != current_identity["source_iteration"]
            or parameters.get("expected_frames") != 30
        ):
            return False

        upstream = _mapping(existing.get("upstream"), "legacy upstream")
        stable_ids = {
            "removal_workspace_manifest": current_identity[
                "removal_workspace_artifact_id"
            ],
            "removal_manifest": current_identity["removal_artifact_id"],
            "tracking_session": current_identity["tracking_artifact_id"],
            "camera_manifest": current_identity["camera_artifact_id"],
        }
        for name, expected_id in stable_ids.items():
            reference = _mapping(upstream.get(name), f"legacy upstream.{name}")
            if reference.get("artifact_id") != expected_id:
                return False

        inputs = _mapping(existing.get("inputs"), "legacy inputs")
        input_hashes: dict[str, str] = {}
        for name in stable_ids:
            record = _mapping(inputs.get(name), f"legacy inputs.{name}")
            digest = record.get("sha256")
            if not isinstance(digest, str) or not digest:
                return False
            input_hashes[name] = digest

        frame_sets = _mapping(existing.get("frame_sets"), "legacy frame_sets")
        recorded_frame_ids: dict[str, str] = {}
        for name, expected_id in current_identity["frame_set_artifact_ids"].items():
            frame_set = _mapping(frame_sets.get(name), f"legacy frame_sets.{name}")
            recorded_id = frame_set.get("artifact_id")
            if recorded_id != expected_id:
                return False
            recorded_frame_ids[name] = str(recorded_id)

        legacy_identity = {
            "kind": WORKSPACE_KIND,
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "source_iteration": current_identity["source_iteration"],
            "removal_workspace_artifact_id": current_identity[
                "removal_workspace_artifact_id"
            ],
            "removal_artifact_id": current_identity["removal_artifact_id"],
            "tracking_artifact_id": current_identity["tracking_artifact_id"],
            "camera_artifact_id": current_identity["camera_artifact_id"],
            "input_hashes": input_hashes,
            "frame_set_artifact_ids": recorded_frame_ids,
        }
        return existing.get("artifact_id") == _identity(legacy_identity)
    except (ArtifactError, KeyError, TypeError):
        return False


def prepare_inpaint_workspace(
    removal_workspace: Path,
    removal_manifest_path: Path,
    tracking_session_path: Path,
    camera_manifest_path: Path,
    tracking_masks: Path,
    source_iteration: int,
    output: Path,
) -> dict[str, Any]:
    if isinstance(source_iteration, bool) or source_iteration <= 0:
        raise ArtifactError("--source-iteration must be a positive integer")

    removal_workspace = _resolved_directory(removal_workspace, "removal workspace")
    removal_workspace_manifest_path, removal_workspace_manifest = _strict_manifest(
        removal_workspace / "workspace_manifest.json",
        "removal workspace manifest",
        REMOVAL_WORKSPACE_KIND,
    )
    removal_manifest_path, removal_manifest = _strict_manifest(
        removal_manifest_path, "removal manifest", REMOVAL_RESULT_KIND
    )
    tracking_session_path, tracking_session = _strict_manifest(
        tracking_session_path,
        "tracking session",
        TRACKING_KIND,
        expected_schema=2,
        require_artifact_id=False,
    )
    camera_manifest_path, camera_manifest = _strict_manifest(
        camera_manifest_path, "virtual-camera manifest", CAMERA_KIND
    )

    parameters = _mapping(
        removal_workspace_manifest.get("parameters"),
        "removal workspace manifest.parameters",
    )
    if parameters.get("iteration") != source_iteration:
        raise ArtifactError(
            "removal workspace iteration does not match --source-iteration"
        )
    removal_parameters = _mapping(
        removal_manifest.get("parameters"), "removal manifest.parameters"
    )
    if removal_parameters.get("distill_iteration") != source_iteration:
        raise ArtifactError(
            "removal manifest distill_iteration does not match --source-iteration"
        )
    _manifest_reference(
        removal_manifest,
        "inputs",
        "workspace_manifest",
        removal_workspace_manifest_path,
        str(removal_workspace_manifest["artifact_id"]),
        "removal manifest",
    )

    expected_png = _frame_names(tracking_session)
    expected_npy = [f"{Path(name).stem}.npy" for name in expected_png]
    if (
        camera_manifest.get("frame_count") != len(expected_png)
        or camera_manifest.get("iteration") != source_iteration
    ):
        raise ArtifactError(
            "virtual-camera manifest must match the frame count and source iteration"
        )
    cameras = camera_manifest.get("cameras")
    if not isinstance(cameras, list) or [
        camera.get("image_name") if isinstance(camera, Mapping) else None
        for camera in cameras
    ] != [Path(name).stem for name in expected_png]:
        raise ArtifactError(
            "virtual-camera manifest camera names must match the ordered tracker frames"
        )
    camera_record = _artifact(camera_manifest_path, "virtual-camera manifest")
    expected_camera_record = {
        "path": camera_record["path"],
        "size_bytes": camera_record["size_bytes"],
        "mtime_ns": camera_record["mtime_ns"],
        "sha256": camera_record["sha256"],
        "artifact_id": camera_manifest["artifact_id"],
    }
    if tracking_session.get("input_cameras") != expected_camera_record:
        raise ArtifactError(
            "tracking session input_cameras does not match the virtual-camera manifest"
        )
    tracking_masks = _resolved_directory(tracking_masks, "tracking mask directory")
    masks = _snapshot_files(tracking_masks, expected_png, "tracking masks")
    _validate_tracking_outputs(tracking_session, masks)

    cfg_args = _regular_file(removal_workspace / "cfg_args", "semantic cfg_args")
    point_cloud = _resolved_directory(
        removal_workspace / "point_cloud", "semantic point_cloud"
    )
    removal_points = _resolved_directory(
        removal_workspace / "point_cloud_object_removal",
        "object-removal point_cloud",
    )
    source_removal_iteration = _resolved_directory(
        removal_points / f"iteration_{source_iteration}",
        "object-removal iteration",
    )
    _regular_file(
        source_removal_iteration / "point_cloud.ply",
        "target-removed Gaussian PLY",
    )

    baseline_depth_root = (
        removal_workspace / "virtual" / f"ours_{source_iteration}" / "depth"
    )
    removed_iteration_root = (
        removal_workspace
        / "virtual"
        / "ours_object_removal"
        / f"iteration_{source_iteration}"
    )
    removed_render_root = removed_iteration_root / "renders"
    removed_depth_root = removed_iteration_root / "depth"
    frame_sets = {
        "tracking_masks": masks,
        "baseline_depth": _snapshot_files(
            baseline_depth_root, expected_npy, "baseline virtual depths"
        ),
        "removed_render": _snapshot_files(
            removed_render_root, expected_png, "removed virtual renders"
        ),
        "removed_depth": _snapshot_files(
            removed_depth_root, expected_npy, "removed virtual depths"
        ),
    }

    # New renderer artifacts are optional only for legacy native runs. Capture
    # geometry and its provenance together so a normal/alpha change invalidates
    # downstream workspace reuse, even if the RGB bytes happen to be identical.
    output = _absolute_output(output)
    extra_links = {}
    render_inputs = {}
    render_pair_path = removal_workspace / "virtual" / "virtual_render_manifest.json"
    if render_pair_path.exists():
        import sys
        from types import SimpleNamespace
        scripts = Path(__file__).resolve().parents[3] / "scripts" / "paintmesh"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from render_virtual_views import validate_pair
        from virtual_render_io import verify_tracking_render

        pair = json.loads(render_pair_path.read_text())
        expected_backend = os.environ.get("VIRTUAL_RENDERER")
        if expected_backend and expected_backend != pair.get("backend"):
            raise ArtifactError("VIRTUAL_RENDERER differs from removal artifacts")
        validate_pair(SimpleNamespace(
            model_path=removal_workspace, iteration=source_iteration,
            backend=pair["backend"], camera_manifest=camera_manifest_path,
            tracker_archive=Path(tracking_session["input_archive"]["path"]),
        ))
        if "input_virtual_render" not in tracking_session:
            raise ArtifactError("tracking session is not bound to the new virtual renders")
        verify_tracking_render(tracking_session)
        if Path(tracking_session["input_virtual_render"]["path"]).resolve() != (removed_iteration_root / "render_manifest.json").resolve():
            raise ArtifactError("tracking session belongs to another virtual render")
        render_inputs["virtual_render_pair"] = _artifact(render_pair_path, "virtual render pair")
        for label, directory in (("baseline", baseline_depth_root.parent), ("removed", removed_iteration_root)):
            manifest_path = directory / "render_manifest.json"
            render_inputs[f"{label}_virtual_render"] = _artifact(manifest_path, f"{label} virtual render")
            extra_links[f"{label}_virtual_render"] = (manifest_path, output / manifest_path.relative_to(removal_workspace))
            modalities = ["alpha", "rgb_raw"]
            if label == "baseline":
                modalities.append("renders")
            if pair["backend"] == "edgs-pgsr":
                modalities += ["normal", "normal_valid", "normal_vis"]
            for modality in modalities:
                source = directory / modality
                names = expected_png if modality in {"renders", "normal_valid", "normal_vis"} else expected_npy
                frame_sets[f"{label}_{modality}"] = _snapshot_files(source, names, f"{label} {modality}")
                extra_links[f"{label}_{modality}"] = (source, output / source.relative_to(removal_workspace))
    elif (removal_workspace / "virtual" / "render_backend.json").exists() or "input_virtual_render" in tracking_session:
        raise ArtifactError("virtual render pair is missing; rerun removal Stage 4")
    elif os.environ.get("VIRTUAL_RENDERER", "inpaint360gs") != "inpaint360gs":
        raise ArtifactError("legacy virtual artifacts only support inpaint360gs")

    _ensure_independent_output(
        output,
        (
            removal_workspace,
            tracking_masks,
            removal_manifest_path,
            tracking_session_path,
            camera_manifest_path,
        ),
    )
    manifest_inputs = {
        "removal_workspace_manifest": _artifact(
            removal_workspace_manifest_path, "removal workspace manifest"
        ),
        "removal_manifest": _artifact(removal_manifest_path, "removal manifest"),
        "tracking_session": _artifact(tracking_session_path, "tracking session"),
        "camera_manifest": camera_record,
        **render_inputs,
    }
    tracking_identity = tracking_session.get("artifact_id")
    if not isinstance(tracking_identity, str) or not tracking_identity:
        tracking_identity = manifest_inputs["tracking_session"]["sha256"]
    identity_payload = {
        "kind": WORKSPACE_KIND,
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "identity_version": WORKSPACE_IDENTITY_VERSION,
        "source_iteration": source_iteration,
        "removal_workspace_artifact_id": removal_workspace_manifest["artifact_id"],
        "removal_artifact_id": removal_manifest["artifact_id"],
        "tracking_artifact_id": tracking_identity,
        "camera_artifact_id": camera_manifest["artifact_id"],
        "frame_set_artifact_ids": {
            name: value["artifact_id"] for name, value in frame_sets.items()
        },
    }
    if render_inputs:
        identity_payload["virtual_render_inputs"] = {name: record["sha256"] for name, record in render_inputs.items()}
    artifact_id = _identity(identity_payload)
    action, existing = _preflight_output(
        output,
        artifact_id,
        manifest_name=MANIFEST_NAME,
        expected_kind=WORKSPACE_KIND,
        recoverable_partial=_recoverable_workspace,
        compatible_existing=lambda value: (
            not _has_material_inpaint_outputs(output, source_iteration)
            and _legacy_workspace_matches(value, identity_payload)
        ),
    )
    if existing is not None:
        if existing.get("schema_version") != WORKSPACE_SCHEMA_VERSION:
            raise ArtifactError("existing inpaint workspace has an unsupported schema")
        if existing.get("status") != "complete":
            raise ArtifactError("existing inpaint workspace is not complete")
        if action != "migrate" and (
            existing.get("identity_version") != WORKSPACE_IDENTITY_VERSION
        ):
            raise ArtifactError("existing inpaint workspace identity is unsupported")

    created_at = datetime.now(timezone.utc).isoformat()
    if existing is not None and isinstance(existing.get("created_at"), str):
        created_at = str(existing["created_at"])

    links = {
        **extra_links,
        "cfg_args": (cfg_args, output / "cfg_args"),
        "point_cloud": (point_cloud, output / "point_cloud"),
        "point_cloud_object_removal": (
            removal_points,
            output / "point_cloud_object_removal",
        ),
        "tracking_masks": (tracking_masks, output / "tracking_masks"),
        "baseline_depth": (
            baseline_depth_root,
            output / "virtual" / f"ours_{source_iteration}" / "depth",
        ),
        "removed_renders": (
            removed_render_root,
            output
            / "virtual"
            / "ours_object_removal"
            / f"iteration_{source_iteration}"
            / "renders",
        ),
        "removed_depth": (
            removed_depth_root,
            output
            / "virtual"
            / "ours_object_removal"
            / f"iteration_{source_iteration}"
            / "depth",
        ),
        "virtual_cameras": (
            camera_manifest_path,
            output / "virtual" / "cameras.json",
        ),
    }
    workspace_links = {
        name: {
            "path": str(destination.relative_to(output)),
            "link_target": _relative_link_target(source, destination),
        }
        for name, (source, destination) in links.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "identity_version": WORKSPACE_IDENTITY_VERSION,
        "kind": WORKSPACE_KIND,
        "complete": True,
        "status": "complete",
        "artifact_id": artifact_id,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "source_iteration": source_iteration,
            "expected_frames": len(expected_png),
        },
        "upstream": {
            "removal_workspace_manifest": {
                "path": str(removal_workspace_manifest_path),
                "artifact_id": removal_workspace_manifest["artifact_id"],
            },
            "removal_manifest": {
                "path": str(removal_manifest_path),
                "artifact_id": removal_manifest["artifact_id"],
            },
            "tracking_session": {
                "path": str(tracking_session_path),
                "artifact_id": tracking_identity,
            },
            "camera_manifest": {
                "path": str(camera_manifest_path),
                "artifact_id": camera_manifest["artifact_id"],
            },
        },
        "inputs": manifest_inputs,
        "frame_sets": frame_sets,
        "workspace": {
            "root": str(output),
            "links": workspace_links,
            "writable_outputs": [
                "virtual/ours_object_removal/"
                f"iteration_{source_iteration}/depth_completed",
                "virtual/ours_object_removal/"
                f"iteration_{source_iteration}/fused_mask_col_dep_ply",
                "virtual/ours_object_removal/"
                f"iteration_{source_iteration}/fused_hole_col_dep_ply",
                "point_cloud_object_inpaint_virtual",
            ],
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    for source, destination in links.values():
        _atomic_relative_symlink(source, destination)
    # Keep the manifest byte-stable on logical refresh so downstream artifacts
    # do not drift merely because ``updated_at`` changed.
    if action != "refresh":
        _atomic_write_json(output / MANIFEST_NAME, manifest)
    return {
        "action": action,
        "artifact_id": artifact_id,
        "source_iteration": source_iteration,
        "output": str(output),
        "manifest": str(output / MANIFEST_NAME),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an isolated workspace for PaintMesh 3DGS inpainting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--removal-workspace", type=Path, required=True)
    parser.add_argument("--removal-manifest", type=Path, required=True)
    parser.add_argument("--tracking-session", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument("--tracking-masks", type=Path, required=True)
    parser.add_argument("--source-iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_inpaint_workspace(
            args.removal_workspace,
            args.removal_manifest,
            args.tracking_session,
            args.camera_manifest,
            args.tracking_masks,
            args.source_iteration,
            args.output,
        )
    except (ArtifactError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
