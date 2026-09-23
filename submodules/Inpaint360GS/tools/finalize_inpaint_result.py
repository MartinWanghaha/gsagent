#!/usr/bin/env python3
"""Validate and atomically commit a complete PaintMesh inpainting result."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from plyfile import PlyData

try:
    from tools.prepare_inpaint_workspace import _strict_manifest
    from tools.prepare_removal_workspace import (
        ArtifactError,
        _artifact,
        _atomic_write_json,
        _identity,
        _mapping,
        _regular_file,
        _sha256,
        _verify_manifest_artifact,
    )
    from tools.publish_inpainted_edgs_model import (
        _validate_edgs_joint,
        _validate_local_geometry,
        _tracking_manifest,
        _validate_fusion_chain,
        _validate_lama_chain,
        _validate_workspace_chain,
        _verified_artifact,
    )
except ModuleNotFoundError:  # Direct repository-local execution.
    from prepare_inpaint_workspace import _strict_manifest
    from prepare_removal_workspace import (
        ArtifactError,
        _artifact,
        _atomic_write_json,
        _identity,
        _mapping,
        _regular_file,
        _sha256,
        _verify_manifest_artifact,
    )
    from publish_inpainted_edgs_model import (
        _validate_edgs_joint,
        _validate_local_geometry,
        _tracking_manifest,
        _validate_fusion_chain,
        _validate_lama_chain,
        _validate_workspace_chain,
        _verified_artifact,
    )


RESULT_KIND = "paintmesh-object-inpaint"
RESULT_SCHEMA_VERSION = 1
RESULT_IDENTITY_VERSION = 2
MODEL_KIND = "paintmesh-inpainted-edgs-model"
MESH_KIND = "paintmesh-pgsr-inpaint-mesh"
REMOVAL_KIND = "paintmesh-object-removal"
WORKSPACE_KIND = "paintmesh-inpaint-workspace"
LAMA_KIND = "paintmesh-lama-completion"
FUSION_KIND = "paintmesh-rgbd-fusion"

ARRAY_OUTPUTS = {
    "gaussian_label": ("gaussians", "label"),
    "gaussian_confidence": ("gaussians", "confidence"),
    "vertex_label": ("mesh_vertices", "label"),
    "vertex_confidence": ("mesh_vertices", "confidence"),
    "face_label": ("mesh_triangles", "label"),
    "face_confidence": ("mesh_triangles", "confidence"),
}


def _legacy_semantic_manifest(
    path: Path,
) -> tuple[Path, Mapping[str, Any], str]:
    resolved = _regular_file(path, "semantic manifest")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read semantic manifest {resolved}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError("semantic manifest must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ArtifactError("semantic manifest must use schema_version 1")
    if payload.get("complete") is not True or payload.get("status") != "complete":
        raise ArtifactError("semantic manifest is not complete")
    kind = payload.get("kind")
    if kind is not None and kind != "paintmesh-instance-semantic-mesh":
        raise ArtifactError(f"unexpected semantic manifest kind: {kind!r}")
    artifact_id = payload.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        artifact_id = _sha256(resolved)
    return resolved, payload, artifact_id


def _reference(
    manifest: Mapping[str, Any],
    section_name: str,
    key: str,
    expected_path: Path,
    expected_id: str,
    label: str,
) -> None:
    section = _mapping(manifest.get(section_name), f"{label}.{section_name}")
    record = _mapping(section.get(key), f"{label}.{section_name}.{key}")
    if record.get("artifact_id") != expected_id:
        raise ArtifactError(f"{label} {key} artifact_id mismatch")
    path_text = record.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise ArtifactError(f"{label} {key} has no path")
    try:
        actual = Path(path_text).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactError(f"{label} {key} path does not exist: {path_text}") from exc
    if actual != expected_path:
        raise ArtifactError(
            f"{label} {key} path resolves to {actual}, expected {expected_path}"
        )


def _model_artifact_path(
    manifest: Mapping[str, Any], key: str, manifest_path: Path
) -> Path:
    model = _mapping(manifest.get("model"), "model manifest.model")
    root_text = model.get("root")
    relative = model.get(key)
    if not isinstance(root_text, str) or not isinstance(relative, str):
        raise ArtifactError(f"model manifest has no model.{key}")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ArtifactError(f"model.{key} is not a safe relative path")
    root = Path(root_text).expanduser().resolve(strict=True)
    if root != manifest_path.parent.resolve(strict=True):
        raise ArtifactError("model manifest root does not match its directory")
    artifact = root / relative_path
    if artifact.is_symlink():
        if os.path.isabs(os.readlink(artifact)):
            raise ArtifactError(
                f"published model artifact must use a relative symlink: {artifact}"
            )
        return artifact.resolve(strict=True)

    # Current inpaint publishers materialize the large Gaussian PLY as an
    # independent regular file for browser/editor compatibility.  Accept a
    # legacy manifest after its old symlink has been materialized, but bind the
    # copy to the manifest's content-addressed source record.
    if key == "point_cloud" and artifact.is_file():
        inputs = _mapping(manifest.get("inputs"), "model manifest.inputs")
        source = _mapping(
            inputs.get("inpainted_gaussian_ply"),
            "model manifest.inputs.inpainted_gaussian_ply",
        )
        expected_size = source.get("size_bytes")
        expected_sha256 = source.get("sha256")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or not isinstance(expected_sha256, str)
            or not expected_sha256
            or artifact.stat().st_size != expected_size
            or _sha256(artifact) != expected_sha256
        ):
            raise ArtifactError(
                f"published model point-cloud copy does not match its source: {artifact}"
            )
        return artifact.resolve(strict=True)

    raise ArtifactError(
        f"published model artifact is neither a relative symlink nor a valid "
        f"point-cloud copy: {artifact}"
    )


def _ply_counts(gaussian_path: Path, mesh_path: Path) -> tuple[int, int, int]:
    gaussian = PlyData.read(str(gaussian_path), mmap=True)
    try:
        gaussian_count = int(gaussian["vertex"].count)
    except KeyError as exc:
        raise ArtifactError("inpainted Gaussian PLY has no vertex element") from exc
    if gaussian_count <= 0:
        raise ArtifactError("inpainted Gaussian PLY contains no points")

    mesh = PlyData.read(
        str(mesh_path),
        mmap=True,
        known_list_len={"face": {"vertex_indices": 3, "vertex_index": 3}},
    )
    try:
        vertices = int(mesh["vertex"].count)
        face_element = mesh["face"]
    except KeyError as exc:
        raise ArtifactError("PGSR mesh must contain vertex and face elements") from exc
    faces = int(face_element.count)
    if vertices <= 0 or faces <= 0:
        raise ArtifactError("PGSR mesh must contain vertices and triangles")
    names = set(face_element.data.dtype.names or ())
    face_name = "vertex_indices" if "vertex_indices" in names else "vertex_index"
    if face_name not in names:
        raise ArtifactError("PGSR mesh face element has no vertex-index property")
    indices = face_element.data[face_name]
    is_dense_triangles = indices.ndim == 2 and indices.shape == (faces, 3)
    is_object_triangles = (
        indices.ndim == 1
        and len(indices) == faces
        and all(np.asarray(face).shape == (3,) for face in indices)
    )
    if not (is_dense_triangles or is_object_triangles):
        raise ArtifactError("PGSR mesh must be triangulated")
    return gaussian_count, vertices, faces


def _semantic_output_path(
    semantic_root: Path,
    outputs: Mapping[str, Any],
    name: str,
) -> tuple[Path, Mapping[str, Any]]:
    record = _mapping(outputs.get(name), f"semantic outputs.{name}")
    filename = record.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ArtifactError(f"semantic output {name} has an unsafe file name")
    path = _regular_file(semantic_root / filename, f"semantic output {name}")
    if record.get("size_bytes") != path.stat().st_size:
        raise ArtifactError(f"semantic output size mismatch: {path}")
    return path, record


def _validate_semantic_outputs(
    semantic_path: Path,
    semantic: Mapping[str, Any],
    counts: Mapping[str, int],
    target_ids: Sequence[int],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    semantic_root = semantic_path.parent
    semantic_counts = _mapping(semantic.get("counts"), "semantic manifest.counts")
    for key, expected in counts.items():
        if semantic_counts.get(key) != expected:
            raise ArtifactError(
                f"semantic count {key}={semantic_counts.get(key)!r}, expected {expected}"
            )
    outputs = _mapping(semantic.get("outputs"), "semantic manifest.outputs")
    artifacts: dict[str, dict[str, Any]] = {}
    residuals = {
        "gaussians": {str(value): 0 for value in target_ids},
        "mesh_vertices": {str(value): 0 for value in target_ids},
        "mesh_triangles": {str(value): 0 for value in target_ids},
    }
    for name, (count_key, role) in ARRAY_OUTPUTS.items():
        path, record = _semantic_output_path(semantic_root, outputs, name)
        array = np.load(path, mmap_mode="r")
        expected_count = counts[count_key]
        if array.ndim != 1 or tuple(array.shape) != (expected_count,):
            raise ArtifactError(
                f"semantic output {path} has shape {array.shape}, "
                f"expected {(expected_count,)}"
            )
        if record.get("shape") != [expected_count] or record.get("dtype") != str(
            array.dtype
        ):
            raise ArtifactError(f"semantic output metadata mismatch: {path}")
        if role == "label":
            if array.dtype.kind not in "iu":
                raise ArtifactError(f"semantic labels must be integer arrays: {path}")
            for target_id in target_ids:
                residuals[count_key][str(target_id)] = int(
                    np.count_nonzero(array == target_id)
                )
        elif array.dtype.kind != "f":
            raise ArtifactError(f"semantic confidences must be floating arrays: {path}")
        del array
        artifacts[name] = _artifact(path, f"semantic output {name}")

    palette, _ = _semantic_output_path(semantic_root, outputs, "palette")
    artifacts["palette"] = _artifact(palette, "semantic palette")
    writes_colored = bool(
        _mapping(semantic.get("parameters"), "semantic parameters").get(
            "write_colored_ply", False
        )
    )
    if writes_colored:
        colored, _ = _semantic_output_path(semantic_root, outputs, "semantic_mesh")
        artifacts["semantic_mesh"] = _artifact(colored, "colored semantic mesh")
    elif "semantic_mesh" in outputs or (semantic_root / "semantic_mesh.ply").exists():
        raise ArtifactError(
            "colored semantic mesh exists although write_colored_ply is false"
        )
    return artifacts, residuals


def finalize_inpaint_result(
    model_manifest_path: Path,
    mesh_manifest_path: Path,
    semantic_manifest_path: Path,
    removal_manifest_path: Path,
    workspace_manifest_path: Path,
    lama_manifest_path: Path,
    fusion_manifest_path: Path,
    gaussian_ply: Path,
    mesh_ply: Path,
    geometry_path: Path,
    output: Path,
) -> dict[str, Any]:
    model_path, model = _strict_manifest(
        model_manifest_path, "inpainted model manifest", MODEL_KIND
    )
    mesh_manifest_path, mesh_manifest = _strict_manifest(
        mesh_manifest_path, "PGSR inpaint mesh manifest", MESH_KIND
    )
    semantic_path, semantic, semantic_id = _legacy_semantic_manifest(
        semantic_manifest_path
    )
    removal_path, removal = _strict_manifest(
        removal_manifest_path, "removal manifest", REMOVAL_KIND
    )
    workspace_path, workspace = _strict_manifest(
        workspace_manifest_path, "inpaint workspace manifest", WORKSPACE_KIND
    )
    lama_path, lama = _strict_manifest(
        lama_manifest_path, "LaMa completion manifest", LAMA_KIND
    )
    pipeline = model.get("parameters", {}).get("pipeline", "inpaint360gs")
    if pipeline not in ("edgs-pgsr", "inpaint360gs"):
        raise ArtifactError("unknown inpaint pipeline")
    direct = pipeline == "edgs-pgsr"
    if direct and fusion_manifest_path is not None:
        raise ArtifactError("EDGS joint finalization must not supply fusion manifest")
    if not direct and fusion_manifest_path is None:
        raise ArtifactError("Inpaint360GS finalization requires fusion manifest")
    fusion_path, fusion = (None, None) if direct else _strict_manifest(
        fusion_manifest_path, "RGB-D fusion manifest", FUSION_KIND)
    gaussian_ply = _regular_file(gaussian_ply, "published inpainted Gaussian PLY")
    mesh_ply = _regular_file(mesh_ply, "PGSR inpaint mesh")
    geometry_path = geometry_path.expanduser()
    if not geometry_path.is_symlink() or os.path.isabs(os.readlink(geometry_path)):
        raise ArtifactError("geometry.ply must be a relative symlink")
    if geometry_path.resolve(strict=True) != mesh_ply:
        raise ArtifactError("geometry.ply does not resolve to the PGSR inpaint mesh")

    if _model_artifact_path(model, "point_cloud", model_path) != gaussian_ply:
        raise ArtifactError("published model point cloud does not match --gaussian-ply")
    model_parameters = _mapping(model.get("parameters"), "model parameters")
    target_ids = [int(value) for value in model_parameters.get("target_ids", [])]
    surrounding_ids = [
        int(value) for value in model_parameters.get("surrounding_ids", [])
    ]
    if not target_ids:
        raise ArtifactError("inpainted model has no target IDs")
    if (
        removal.get("target_ids") != target_ids
        or removal.get("surrounding_ids") != surrounding_ids
    ):
        raise ArtifactError("inpainted model selection does not match removal result")

    source_iteration = model_parameters.get("source_iteration")
    if (
        isinstance(source_iteration, bool)
        or not isinstance(source_iteration, int)
        or source_iteration <= 0
    ):
        raise ArtifactError("inpainted model has no valid source_iteration")
    model_inputs = _mapping(model.get("inputs"), "model manifest.inputs")
    for key, path in (
        ("removal_manifest", removal_path),
        ("workspace_manifest", workspace_path),
        ("lama_manifest", lama_path),
        ("fusion_manifest", fusion_path),
    ):
        if path is None:
            if key in model_inputs:
                raise ArtifactError("EDGS joint model has unexpected fusion dependency")
            continue
        actual = _verified_artifact(model_inputs.get(key), f"model inputs.{key}")
        if actual != path:
            raise ArtifactError(
                f"model input {key} does not match the supplied manifest"
            )

    tracking_path = _verified_artifact(
        model_inputs.get("tracking_session"), "model inputs.tracking_session"
    )
    tracking_path, tracking, tracking_id = _tracking_manifest(tracking_path)
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
    lama_input_path, lama_input = _validate_lama_chain(lama_path, lama)
    if not direct:
        _validate_fusion_chain(fusion_path, fusion, camera_path, camera, lama_path,
                               lama, lama_input, workspace, source_iteration)
    for key, expected_path in (
        ("camera_manifest", camera_path),
        ("lama_input_manifest", lama_input_path),
    ):
        actual = _verified_artifact(model_inputs.get(key), f"model inputs.{key}")
        if actual != expected_path:
            raise ArtifactError(f"model input {key} does not match its producer chain")

    upstream_ids = _mapping(
        model.get("upstream_artifact_ids"), "model upstream_artifact_ids"
    )
    expected_upstream_ids = {
        "removal": removal["artifact_id"],
        "workspace": workspace["artifact_id"],
        "tracking": tracking_id,
        "camera": camera["artifact_id"],
        "lama_inputs": lama_input["artifact_id"],
        "lama": lama["artifact_id"],
    }
    if not direct:
        expected_upstream_ids["fusion"] = fusion["artifact_id"]
        if "edgs_joint_manifest" in model_inputs or "edgs_joint" in upstream_ids:
            raise ArtifactError("joint provenance contradicts selected pipeline")
    else:
        if model_parameters.get("local_geometry_refine") or model_parameters.get("support_density_mode"):
            raise ArtifactError("EDGS joint cannot include old-path local/density stages")
        joint_path = _verified_artifact(model_inputs.get("edgs_joint_manifest"), "model inputs.edgs_joint_manifest")
        joint = _validate_edgs_joint(joint_path, gaussian_ply, lama_path, camera_path)
        if joint["parameters"]["iterations"] != model_parameters.get("output_iteration") or model_parameters.get("joint_iterations") != model_parameters.get("output_iteration"):
            raise ArtifactError("EDGS joint iteration metadata mismatch")
        expected_upstream_ids["edgs_joint"] = joint["artifact_id"]
    if model_parameters.get("local_geometry_refine", False):
        local_path = _verified_artifact(
            model_inputs.get("local_geometry_manifest"), "model inputs.local_geometry_manifest")
        local = _validate_local_geometry(local_path, gaussian_ply, lama_path, camera_path)
        if (local["parameters"]["rgb_iterations"] != model_parameters.get("rgb_iterations") or
                local["parameters"]["local_geometry_iterations"] != model_parameters.get("local_geometry_iterations")):
            raise ArtifactError("local geometry iteration metadata differs from published model")
        expected_upstream_ids["local_geometry"] = local["artifact_id"]
    elif "local_geometry_manifest" in model_inputs or "local_geometry" in upstream_ids:
        raise ArtifactError("local geometry provenance contradicts disabled model settings")
    if model_parameters.get("support_density_mode") == "mass_adaptive":
        try:
            from tools.publish_inpainted_edgs_model import _validate_density
        except ImportError:
            from publish_inpainted_edgs_model import _validate_density
        density_path = _verified_artifact(model_inputs.get("density_manifest"), "density manifest")
        rgb_path = _verified_artifact(model_inputs.get("rgb_manifest"), "RGB manifest")
        density = _validate_density(density_path, rgb_path, gaussian_ply,
            local_path if model_parameters.get("local_geometry_refine", False) else None,
            lama_path, camera_path, fusion_path,
            _verified_artifact(model_inputs.get("inpaint_config"), "inpaint config"))
        expected_upstream_ids["support_density"] = density["artifact_id"]
        if density.get('parameters',{}).get('mode')!=model_parameters['support_density_mode']:
            raise ArtifactError('density mode differs from published model')
    elif "density_manifest" in model_inputs or "support_density" in upstream_ids or model_parameters.get("support_density_mode"):
        raise ArtifactError("density provenance contradicts disabled model settings")
    for key, expected_id in expected_upstream_ids.items():
        if upstream_ids.get(key) != expected_id:
            raise ArtifactError(f"model upstream {key} artifact_id mismatch")

    _reference(
        mesh_manifest,
        "inputs",
        "model_manifest",
        model_path,
        str(model["artifact_id"]),
        "PGSR mesh manifest",
    )
    mesh_inputs = _mapping(mesh_manifest.get("inputs"), "mesh manifest.inputs")
    _verify_manifest_artifact(
        _mapping(mesh_inputs.get("gaussian_ply"), "mesh inputs.gaussian_ply"),
        gaussian_ply,
        "mesh inputs.gaussian_ply",
    )
    mesh_outputs = _mapping(mesh_manifest.get("outputs"), "mesh manifest.outputs")
    _verify_manifest_artifact(
        _mapping(mesh_outputs.get("mesh"), "mesh outputs.mesh"),
        mesh_ply,
        "mesh outputs.mesh",
    )
    if _mapping(mesh_manifest.get("parameters"), "mesh parameters").get(
        "iteration"
    ) != model_parameters.get("output_iteration"):
        raise ArtifactError("PGSR mesh iteration does not match the inpainted model")

    semantic_inputs = _mapping(semantic.get("inputs"), "semantic manifest.inputs")
    _verify_manifest_artifact(
        _mapping(semantic_inputs.get("gaussian_ply"), "semantic inputs.gaussian_ply"),
        gaussian_ply,
        "semantic inputs.gaussian_ply",
    )
    _verify_manifest_artifact(
        _mapping(semantic_inputs.get("mesh"), "semantic inputs.mesh"),
        mesh_ply,
        "semantic inputs.mesh",
    )

    gaussian_count, vertex_count, face_count = _ply_counts(gaussian_ply, mesh_ply)
    counts = {
        "gaussians": gaussian_count,
        "mesh_vertices": vertex_count,
        "mesh_triangles": face_count,
    }
    if (
        _mapping(model.get("gaussian"), "model manifest.gaussian").get("point_count")
        != gaussian_count
    ):
        raise ArtifactError("model manifest Gaussian count does not match its PLY")
    semantic_artifacts, target_residuals = _validate_semantic_outputs(
        semantic_path, semantic, counts, target_ids
    )

    manifest_inputs = {
        "model_manifest": _artifact(model_path, "inpainted model manifest"),
        "mesh_manifest": _artifact(mesh_manifest_path, "PGSR mesh manifest"),
        "semantic_manifest": _artifact(semantic_path, "semantic manifest"),
        "removal_manifest": _artifact(removal_path, "removal manifest"),
        "workspace_manifest": _artifact(workspace_path, "workspace manifest"),
        "tracking_session": _artifact(tracking_path, "tracking session"),
        "camera_manifest": _artifact(camera_path, "virtual-camera manifest"),
        "lama_input_manifest": _artifact(lama_input_path, "LaMa input manifest"),
        "lama_manifest": _artifact(lama_path, "LaMa manifest"),
    }
    if direct:
        manifest_inputs["edgs_joint_manifest"] = _artifact(joint_path, "EDGS joint manifest")
    else:
        manifest_inputs["fusion_manifest"] = _artifact(fusion_path, "fusion manifest")
    output_artifacts = {
        "inpainted_gaussian_ply": _artifact(gaussian_ply, "inpainted Gaussian PLY"),
        "pgsr_mesh": _artifact(mesh_ply, "PGSR inpaint mesh"),
        **semantic_artifacts,
    }
    upstream_ids = {
        "model": model["artifact_id"],
        "mesh": mesh_manifest["artifact_id"],
        "semantic": semantic_id,
        "removal": removal["artifact_id"],
        "workspace": workspace["artifact_id"],
        "tracking": tracking_id,
        "camera": camera["artifact_id"],
        "lama_inputs": lama_input["artifact_id"],
        "lama": lama["artifact_id"],
    }
    if direct:
        upstream_ids["edgs_joint"] = joint["artifact_id"]
    else:
        upstream_ids["fusion"] = fusion["artifact_id"]
    identity_payload = {
        "kind": RESULT_KIND,
        "schema_version": RESULT_SCHEMA_VERSION,
        "identity_version": RESULT_IDENTITY_VERSION,
        "target_ids": target_ids,
        "surrounding_ids": surrounding_ids,
        "parameters": {
            "model": dict(model_parameters),
            "mesh": dict(_mapping(mesh_manifest.get("parameters"), "mesh parameters")),
            "semantic": dict(
                _mapping(semantic.get("parameters"), "semantic parameters")
            ),
        },
        "counts": counts,
        "target_residuals": target_residuals,
        "upstream_artifact_ids": upstream_ids,
        "output_hashes": {
            name: record["sha256"] for name, record in output_artifacts.items()
        },
    }
    artifact_id = _identity(identity_payload)

    output = Path(os.path.abspath(output.expanduser()))
    if output.is_symlink() or output == Path(output.anchor):
        raise ArtifactError("output manifest path is unsafe")
    if output.exists():
        existing_path, existing = _strict_manifest(
            output, "existing inpaint result manifest", RESULT_KIND
        )
        if existing_path != output.resolve(strict=True):
            raise ArtifactError("existing output manifest resolves unexpectedly")
        if existing.get("artifact_id") != artifact_id:
            raise ArtifactError(
                "existing inpaint result belongs to different inputs or parameters"
            )
        if existing.get("identity_version") != RESULT_IDENTITY_VERSION:
            raise ArtifactError("existing inpaint result identity is unsupported")
        return {
            "action": "refresh",
            "artifact_id": artifact_id,
            "counts": counts,
            "target_residuals": target_residuals,
            "output": str(output),
        }

    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "identity_version": RESULT_IDENTITY_VERSION,
        "kind": RESULT_KIND,
        "complete": True,
        "status": "complete",
        "artifact_id": artifact_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "target_ids": target_ids,
        "surrounding_ids": surrounding_ids,
        "parameters": identity_payload["parameters"],
        "counts": counts,
        "target_residuals": target_residuals,
        "upstream_artifact_ids": upstream_ids,
        "inputs": manifest_inputs,
        "outputs": {
            **output_artifacts,
            "geometry": {
                "path": str(geometry_path.absolute()),
                "link_target": os.readlink(geometry_path),
            },
            "semantic_root": str(semantic_path.parent.resolve(strict=True)),
        },
    }
    _atomic_write_json(output, payload)
    return {
        "action": "create",
        "artifact_id": artifact_id,
        "counts": counts,
        "target_residuals": target_residuals,
        "output": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and commit a complete PaintMesh inpainting result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--mesh-manifest", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--removal-manifest", type=Path, required=True)
    parser.add_argument("--workspace-manifest", type=Path, required=True)
    parser.add_argument("--lama-manifest", type=Path, required=True)
    parser.add_argument("--fusion-manifest", type=Path)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = finalize_inpaint_result(
            args.model_manifest,
            args.mesh_manifest,
            args.semantic_manifest,
            args.removal_manifest,
            args.workspace_manifest,
            args.lama_manifest,
            args.fusion_manifest,
            args.gaussian_ply,
            args.mesh,
            args.geometry,
            args.output,
        )
    except (ArtifactError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
