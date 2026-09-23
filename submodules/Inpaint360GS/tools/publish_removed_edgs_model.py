#!/usr/bin/env python3
"""Publish an Inpaint360GS removal result as a renderable EDGS model.

Object removal writes a standalone Gaussian PLY outside Graphdeco's standard
model layout.  This tool validates that PLY, binds it to the original EDGS
configuration and semantic classifier, and creates the conventional
``point_cloud/iteration_<N>`` layout using relative symlinks.  No large model
artifact is copied or modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

try:  # Package import in tests and repository-local execution.
    from tools.build_edgs_bridge import BridgeError, validate_gaussian_ply
    from tools.prepare_removal_workspace import (
        ArtifactError,
        _absolute_output,
        _artifact,
        _atomic_relative_symlink,
        _atomic_write_json,
        _cfg_path,
        _ensure_independent_output,
        _identity,
        _load_complete_manifest,
        _mapping,
        _manifest_artifact,
        _parse_namespace,
        _path_from_manifest,
        _preflight_output,
        _regular_file,
        _relative_link_target,
        _sha256,
        _verify_manifest_artifact,
    )
except ModuleNotFoundError:  # ``python tools/publish_removed_edgs_model.py``.
    from build_edgs_bridge import BridgeError, validate_gaussian_ply
    from prepare_removal_workspace import (
        ArtifactError,
        _absolute_output,
        _artifact,
        _atomic_relative_symlink,
        _atomic_write_json,
        _cfg_path,
        _ensure_independent_output,
        _identity,
        _load_complete_manifest,
        _mapping,
        _manifest_artifact,
        _parse_namespace,
        _path_from_manifest,
        _preflight_output,
        _regular_file,
        _relative_link_target,
        _sha256,
        _verify_manifest_artifact,
    )


MODEL_KIND = "paintmesh-removed-edgs-model"
MODEL_SCHEMA_VERSION = 1
MANIFEST_NAME = "model_manifest.json"
OBJECT_EMBEDDING_DIMENSIONS = 16
_FLOAT_TYPES = {"float", "float32", "double", "float64"}


def _load_edgs_config(path: Path) -> tuple[Path, Mapping[str, Any], int]:
    resolved = _regular_file(path, "EDGS config")
    try:
        with resolved.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ArtifactError(f"cannot read EDGS config {resolved}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ArtifactError(f"EDGS config root must be a mapping: {resolved}")
    gs = _mapping(config.get("gs"), "EDGS config gs")
    sh_degree = gs.get("sh_degree")
    if isinstance(sh_degree, bool) or not isinstance(sh_degree, int) or sh_degree < 0:
        raise ArtifactError("EDGS config gs.sh_degree must be a non-negative integer")
    return resolved, config, sh_degree


def _parse_ids(text: str, label: str, *, allow_none: bool) -> list[int]:
    stripped = text.strip()
    if allow_none and stripped.lower() == "none":
        return []
    if not stripped:
        suffix = " or 'none'" if allow_none else ""
        raise ArtifactError(f"{label} must be a comma-separated ID list{suffix}")
    values: list[int] = []
    for token in stripped.split(","):
        token = token.strip()
        if not token or not token.isdecimal():
            raise ArtifactError(
                f"{label} must contain canonical non-negative integers: {text!r}"
            )
        value = int(token)
        if value <= 0 or value >= 65535:
            raise ArtifactError(f"{label} ID must be in [1, 65534], got {value}")
        values.append(value)
    if len(values) != len(set(values)):
        raise ArtifactError(f"{label} contains duplicate IDs: {text!r}")
    return sorted(values)


def _validate_selection(
    target_text: str,
    surrounding_text: str,
    *,
    num_classes: int,
) -> tuple[list[int], list[int]]:
    target_ids = _parse_ids(target_text, "target-ids", allow_none=False)
    surrounding_ids = _parse_ids(surrounding_text, "surrounding-ids", allow_none=True)
    overlap = sorted(set(target_ids) & set(surrounding_ids))
    if overlap:
        raise ArtifactError(
            "target-ids and surrounding-ids overlap: " + ", ".join(map(str, overlap))
        )
    invalid = [
        value for value in (*target_ids, *surrounding_ids) if value >= num_classes
    ]
    if invalid:
        raise ArtifactError(
            f"object IDs must be smaller than num_classes={num_classes}; got "
            + ", ".join(map(str, invalid))
        )
    return target_ids, surrounding_ids


def _validate_removed_ply(path: Path, sh_degree: int) -> tuple[int, list[str]]:
    try:
        header = validate_gaussian_ply(path, sh_degree)
    except BridgeError as exc:
        raise ArtifactError(str(exc)) from exc
    vertex = header.element("vertex")
    if vertex is None or vertex.count <= 0:  # Also enforced by the base validator.
        raise ArtifactError(f"removed Gaussian PLY has no points: {path}")
    properties = {prop.name: prop for prop in vertex.properties}
    expected = {f"obj_dc_{index}" for index in range(OBJECT_EMBEDDING_DIMENSIONS)}
    actual = {name for name in properties if name.startswith("obj_dc_")}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ArtifactError(
            "removed Gaussian PLY must contain exactly obj_dc_0..obj_dc_15: "
            + "; ".join(details)
        )
    invalid = [
        name
        for name in sorted(expected)
        if properties[name].is_list or properties[name].value_type not in _FLOAT_TYPES
    ]
    if invalid:
        raise ArtifactError(
            "removed Gaussian object embeddings must be floating-point scalars: "
            + ", ".join(invalid)
        )
    return int(vertex.count), sorted(expected, key=lambda name: int(name[7:]))


def _bridge_config_consistency(
    bridge_manifest: Mapping[str, Any], config_path: Path
) -> None:
    edgs = _mapping(bridge_manifest.get("edgs"), "bridge manifest edgs")
    recorded = _path_from_manifest(
        edgs.get("config_path"), "bridge manifest edgs.config_path"
    )
    if recorded != config_path:
        raise ArtifactError(
            f"EDGS config is {config_path}, but the bridge records {recorded}"
        )
    expected_hash = edgs.get("config_sha256")
    if not isinstance(expected_hash, str) or expected_hash != _sha256(config_path):
        raise ArtifactError("EDGS config hash does not match the bridge manifest")


def _semantic_consistency(
    semantic_manifest: Mapping[str, Any],
    bridge_manifest: Mapping[str, Any],
    classifier: Path,
    cfg_path: Path,
    cfg_values: Mapping[str, Any],
    iteration: int,
    sh_degree: int,
) -> int:
    cfg_sh_degree = cfg_values.get("sh_degree")
    if cfg_sh_degree != sh_degree:
        raise ArtifactError(
            f"cfg_args sh_degree={cfg_sh_degree!r} does not match "
            f"EDGS config sh_degree={sh_degree}"
        )
    num_classes = cfg_values.get("num_classes")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 0
    ):
        raise ArtifactError("cfg_args num_classes must be a positive integer")

    classifier_record = _manifest_artifact(
        semantic_manifest, "classifier", "semantic manifest"
    )
    _verify_manifest_artifact(
        classifier_record, classifier, "semantic manifest.inputs.classifier"
    )
    gaussian_record = _manifest_artifact(
        semantic_manifest, "gaussian_ply", "semantic manifest"
    )
    semantic_ply = _path_from_manifest(
        gaussian_record.get("path"), "semantic manifest.inputs.gaussian_ply.path"
    )
    expected_iteration_root = semantic_ply.parent
    if expected_iteration_root.name != f"iteration_{iteration}":
        raise ArtifactError(
            f"semantic manifest Gaussian comes from {expected_iteration_root.name}, "
            f"not iteration_{iteration}"
        )
    semantic_model = expected_iteration_root.parent.parent
    cfg_model = _cfg_path(cfg_values.get("model_path"), "model_path", cfg_path)
    if cfg_model != semantic_model:
        raise ArtifactError(
            f"cfg_args model_path resolves to {cfg_model}, but semantic artifacts "
            f"belong to {semantic_model}"
        )

    dataset = _mapping(bridge_manifest.get("dataset"), "bridge manifest dataset")
    bridge_source = _path_from_manifest(
        dataset.get("source_path"), "bridge manifest dataset.source_path"
    )
    cfg_source = _cfg_path(cfg_values.get("source_path"), "source_path", cfg_path)
    if cfg_source != bridge_source:
        raise ArtifactError(
            f"cfg_args source_path resolves to {cfg_source}, but the bridge records "
            f"{bridge_source}"
        )

    counts = _mapping(semantic_manifest.get("counts"), "semantic manifest counts")
    if counts.get("embedding_dimensions") != OBJECT_EMBEDDING_DIMENSIONS:
        raise ArtifactError(
            "semantic manifest must report exactly 16 object embedding dimensions"
        )
    if counts.get("classes") != num_classes:
        raise ArtifactError("semantic manifest class count does not match cfg_args")
    return num_classes


def _ensure_managed_directory(path: Path, root: Path) -> None:
    if path.is_symlink():
        raise ArtifactError(f"managed model directory cannot be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise ArtifactError(f"managed model path must be a directory: {path}")
    path.mkdir(exist_ok=True)
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ArtifactError(
            f"managed model directory escapes output root: {path}"
        ) from exc


def _recoverable_model_output(output: Path, iteration: int) -> bool:
    """Recognize an interrupted publish before its manifest was committed."""

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
    return all(path.is_symlink() for path in artifacts.values())


def publish_model(
    removed_ply: Path,
    classifier: Path,
    edgs_config: Path,
    cfg_args: Path,
    iteration: int,
    target_ids_text: str,
    surrounding_ids_text: str,
    bridge_manifest_path: Path,
    semantic_manifest_path: Path,
    output: Path,
    *,
    removal_threshold: float | None = None,
) -> dict[str, Any]:
    if isinstance(iteration, bool) or iteration <= 0:
        raise ArtifactError("--iteration must be a positive integer")
    if removal_threshold is not None and (
        not math.isfinite(removal_threshold) or not 0.0 <= removal_threshold <= 1.0
    ):
        raise ArtifactError("--removal-threshold must be finite and in [0, 1]")
    removed_ply = _regular_file(removed_ply, "removed Gaussian PLY")
    classifier = _regular_file(classifier, "classifier")
    cfg_path = _regular_file(cfg_args, "cfg_args")
    config_path, _, sh_degree = _load_edgs_config(edgs_config)
    bridge_path, bridge_manifest = _load_complete_manifest(
        bridge_manifest_path,
        "bridge manifest",
        expected_kind="edgs-inpaint360gs-bridge",
    )
    semantic_path, semantic_manifest = _load_complete_manifest(
        semantic_manifest_path, "semantic manifest"
    )
    _bridge_config_consistency(bridge_manifest, config_path)
    cfg_values = _parse_namespace(cfg_path)
    num_classes = _semantic_consistency(
        semantic_manifest,
        bridge_manifest,
        classifier,
        cfg_path,
        cfg_values,
        iteration,
        sh_degree,
    )
    target_ids, surrounding_ids = _validate_selection(
        target_ids_text, surrounding_ids_text, num_classes=num_classes
    )
    point_count, embedding_properties = _validate_removed_ply(removed_ply, sh_degree)

    output = _absolute_output(output)
    _ensure_independent_output(
        output,
        (
            removed_ply,
            classifier,
            config_path,
            cfg_path,
            bridge_path,
            semantic_path,
        ),
    )
    inputs = {
        "removed_gaussian_ply": _artifact(removed_ply, "removed Gaussian PLY"),
        "classifier": _artifact(classifier, "classifier"),
        "edgs_config": _artifact(config_path, "EDGS config"),
        "cfg_args": _artifact(cfg_path, "cfg_args"),
        "bridge_manifest": _artifact(bridge_path, "bridge manifest"),
        "semantic_manifest": _artifact(semantic_path, "semantic manifest"),
    }
    identity_payload = {
        "kind": MODEL_KIND,
        "schema_version": MODEL_SCHEMA_VERSION,
        "iteration": iteration,
        "target_ids": target_ids,
        "surrounding_ids": surrounding_ids,
        "removal_threshold": removal_threshold,
        "bridge_artifact_id": bridge_manifest.get("artifact_id"),
        "inputs": {name: value["sha256"] for name, value in inputs.items()},
    }
    artifact_id = _identity(identity_payload)
    action, existing = _preflight_output(
        output,
        artifact_id,
        manifest_name=MANIFEST_NAME,
        expected_kind=MODEL_KIND,
        recoverable_partial=lambda path: _recoverable_model_output(path, iteration),
    )
    created_at = datetime.now(timezone.utc).isoformat()
    if existing is not None and isinstance(existing.get("created_at"), str):
        created_at = str(existing["created_at"])

    config_link = output / "config.yaml"
    cfg_link = output / "cfg_args"
    iteration_root = output / "point_cloud" / f"iteration_{iteration}"
    ply_link = iteration_root / "point_cloud.ply"
    classifier_link = iteration_root / "classifier.pth"
    manifest: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "kind": MODEL_KIND,
        "complete": True,
        "status": "complete",
        "artifact_id": artifact_id,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "iteration": iteration,
            "target_ids": target_ids,
            "surrounding_ids": surrounding_ids,
            "removal_threshold": removal_threshold,
        },
        "bridge_artifact_id": bridge_manifest.get("artifact_id"),
        "inputs": inputs,
        "gaussian": {
            "point_count": point_count,
            "sh_degree": sh_degree,
            "object_embedding_dimensions": OBJECT_EMBEDDING_DIMENSIONS,
            "object_embedding_properties": embedding_properties,
        },
        "model": {
            "root": str(output),
            "config": "config.yaml",
            "config_link_target": _relative_link_target(config_path, config_link),
            "cfg_args": "cfg_args",
            "cfg_args_link_target": _relative_link_target(cfg_path, cfg_link),
            "point_cloud": f"point_cloud/iteration_{iteration}/point_cloud.ply",
            "point_cloud_link_target": _relative_link_target(removed_ply, ply_link),
            "classifier": f"point_cloud/iteration_{iteration}/classifier.pth",
            "classifier_link_target": _relative_link_target(
                classifier, classifier_link
            ),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    _ensure_managed_directory(output / "point_cloud", output)
    _ensure_managed_directory(iteration_root, output)
    _atomic_relative_symlink(config_path, config_link)
    _atomic_relative_symlink(cfg_path, cfg_link)
    _atomic_relative_symlink(removed_ply, ply_link)
    _atomic_relative_symlink(classifier, classifier_link)
    # Preserve the commit marker byte-for-byte on logical refresh.  Tracker
    # sessions bind this producer by artifact ID and keep the file record only
    # as an audit snapshot; changing ``updated_at`` would invalidate that
    # snapshot without changing the removed model.
    if action != "refresh":
        _atomic_write_json(output / MANIFEST_NAME, manifest)
    return {
        "action": action,
        "artifact_id": artifact_id,
        "iteration": iteration,
        "target_ids": target_ids,
        "surrounding_ids": surrounding_ids,
        "point_count": point_count,
        "output": str(output),
        "point_cloud": str(ply_link),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and publish an Inpaint360GS object-removal PLY as an "
            "EDGS/PGSR-renderable model directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--removed-ply", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--edgs-config", type=Path, required=True)
    parser.add_argument("--cfg-args", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--target-ids",
        required=True,
        help="comma-separated semantic instance IDs to remove",
    )
    parser.add_argument(
        "--surrounding-ids",
        required=True,
        help="comma-separated temporary surrounding IDs, or 'none'",
    )
    parser.add_argument("--bridge-manifest", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument(
        "--removal-threshold",
        type=float,
        required=True,
        help="classifier probability threshold used to produce removed-ply",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = publish_model(
            args.removed_ply,
            args.classifier,
            args.edgs_config,
            args.cfg_args,
            args.iteration,
            args.target_ids,
            args.surrounding_ids,
            args.bridge_manifest,
            args.semantic_manifest,
            args.output,
            removal_threshold=args.removal_threshold,
        )
    except (ArtifactError, BridgeError, OSError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
