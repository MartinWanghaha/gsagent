#!/usr/bin/env python3
"""Prepare an isolated, read-only input view for object removal.

The semantic 3DGS produced by ``scripts/paintmesh/run_seg`` remains the source
of truth.  This tool validates that model and the two manifests which bind it
to the EDGS/PGSR reconstruction, then exposes only ``cfg_args`` and
``point_cloud`` through relative symlinks in a separate work directory.

The workspace manifest is the commit marker.  Re-running with identical inputs
repairs the managed links and preserves downstream removal outputs; reusing the
same directory for different inputs is deliberately rejected.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from tools.build_edgs_bridge import BridgeError, validate_gaussian_ply
except ModuleNotFoundError:  # ``python tools/prepare_removal_workspace.py``.
    from build_edgs_bridge import BridgeError, validate_gaussian_ply


WORKSPACE_KIND = "paintmesh-removal-workspace"
WORKSPACE_SCHEMA_VERSION = 1
MANIFEST_NAME = "workspace_manifest.json"
OBJECT_EMBEDDING_DIMENSIONS = 16
_PLY_FLOAT_TYPES = {"float", "float32", "double", "float64"}


class ArtifactError(RuntimeError):
    """Raised when inputs cannot safely form a removal artifact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str, *, nonempty: bool = True) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise ArtifactError(f"{label} is not a regular file: {resolved}")
    if nonempty and resolved.stat().st_size <= 0:
        raise ArtifactError(f"{label} is empty: {resolved}")
    return resolved


def _artifact(path: Path, label: str) -> dict[str, Any]:
    resolved = _regular_file(path, label)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(resolved),
    }


def _load_json_mapping(path: Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    resolved = _regular_file(path, label)
    try:
        with resolved.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label} must contain a JSON object: {resolved}")
    return resolved, value


def _load_complete_manifest(
    path: Path,
    label: str,
    *,
    expected_kind: str | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    resolved, manifest = _load_json_mapping(path, label)
    if manifest.get("complete") is not True:
        raise ArtifactError(f"{label} is not marked complete: {resolved}")
    status = manifest.get("status")
    if status is not None and status != "complete":
        raise ArtifactError(
            f"{label} has status {status!r}, expected 'complete': {resolved}"
        )
    if expected_kind is not None and manifest.get("kind") != expected_kind:
        raise ArtifactError(
            f"{label} kind is {manifest.get('kind')!r}, expected "
            f"{expected_kind!r}: {resolved}"
        )
    return resolved, manifest


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{location} must be an object")
    return value


def _manifest_artifact(
    manifest: Mapping[str, Any], key: str, manifest_label: str
) -> Mapping[str, Any]:
    inputs = _mapping(manifest.get("inputs"), f"{manifest_label}.inputs")
    return _mapping(inputs.get(key), f"{manifest_label}.inputs.{key}")


def _path_from_manifest(value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"{location} must be a non-empty path string")
    try:
        return Path(value).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactError(
            f"path recorded by {location} does not exist: {value}"
        ) from exc


def _verify_manifest_artifact(
    record: Mapping[str, Any], actual: Path, location: str
) -> None:
    recorded = _path_from_manifest(record.get("path"), f"{location}.path")
    if recorded != actual:
        raise ArtifactError(
            f"{location}.path resolves to {recorded}, expected {actual}"
        )
    stat = actual.stat()
    for key, observed in (
        ("size_bytes", int(stat.st_size)),
        ("mtime_ns", int(stat.st_mtime_ns)),
    ):
        expected = record.get(key)
        if expected is not None and (
            isinstance(expected, bool) or expected != observed
        ):
            raise ArtifactError(
                f"{location}.{key} is {expected!r}, but the file reports {observed}"
            )
    expected_hash = record.get("sha256")
    if expected_hash is not None:
        if not isinstance(expected_hash, str) or expected_hash != _sha256(actual):
            raise ArtifactError(f"{location}.sha256 does not match {actual}")


def _parse_namespace(path: Path) -> dict[str, Any]:
    """Parse argparse's ``Namespace(...)`` representation without ``eval``."""

    resolved = _regular_file(path, "semantic cfg_args")
    try:
        expression = ast.parse(
            resolved.read_text(encoding="utf-8").strip(), mode="eval"
        )
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ArtifactError(f"invalid cfg_args {resolved}: {exc}") from exc
    call = expression.body
    if not isinstance(call, ast.Call) or call.args:
        raise ArtifactError(f"cfg_args must contain exactly Namespace(...): {resolved}")
    is_namespace = isinstance(call.func, ast.Name) and call.func.id == "Namespace"
    if not is_namespace:
        raise ArtifactError(f"cfg_args must contain exactly Namespace(...): {resolved}")

    values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in values:
            raise ArtifactError(
                f"cfg_args contains invalid keyword arguments: {resolved}"
            )
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise ArtifactError(
                f"cfg_args value for {keyword.arg!r} is not a literal: {resolved}"
            ) from exc
    return values


def _required_cfg(values: Mapping[str, Any], key: str) -> Any:
    if key not in values:
        raise ArtifactError(f"semantic cfg_args is missing {key!r}")
    return values[key]


def _cfg_path(value: Any, key: str, cfg_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactError(f"semantic cfg_args field {key!r} must be a path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = cfg_path.parent / candidate
    return candidate.resolve()


def _validate_cfg_args(
    cfg_path: Path,
    semantic_model: Path,
    bridge_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    values = _parse_namespace(cfg_path)
    sh_degree = _required_cfg(values, "sh_degree")
    if isinstance(sh_degree, bool) or not isinstance(sh_degree, int) or sh_degree < 0:
        raise ArtifactError(
            "semantic cfg_args sh_degree must be a non-negative integer"
        )
    num_classes = _required_cfg(values, "num_classes")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 0
    ):
        raise ArtifactError("semantic cfg_args num_classes must be a positive integer")

    recorded_model = _cfg_path(
        _required_cfg(values, "model_path"), "model_path", cfg_path
    )
    if recorded_model != semantic_model:
        raise ArtifactError(
            f"semantic cfg_args model_path resolves to {recorded_model}, "
            f"expected {semantic_model}"
        )

    dataset = _mapping(bridge_manifest.get("dataset"), "bridge manifest dataset")
    bridge_source = _path_from_manifest(
        dataset.get("source_path"), "bridge manifest dataset.source_path"
    )
    cfg_source = _cfg_path(
        _required_cfg(values, "source_path"), "source_path", cfg_path
    )
    if cfg_source != bridge_source:
        raise ArtifactError(
            f"semantic cfg_args source_path resolves to {cfg_source}, "
            f"but the bridge records {bridge_source}"
        )

    bridge = _mapping(bridge_manifest.get("bridge"), "bridge manifest bridge")
    bridge_root = _path_from_manifest(bridge.get("root"), "bridge manifest bridge.root")
    vanilla_path = _cfg_path(
        _required_cfg(values, "vanilla_3dgs_path"),
        "vanilla_3dgs_path",
        cfg_path,
    )
    if vanilla_path != bridge_root:
        raise ArtifactError(
            f"semantic cfg_args vanilla_3dgs_path resolves to {vanilla_path}, "
            f"but the bridge root is {bridge_root}"
        )
    return values


def _absolute_output(path: Path) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    # Resolve parent symlinks so containment checks use canonical locations,
    # but never follow a caller-supplied symlink at the output leaf itself.
    if absolute.is_symlink():
        raise ArtifactError(f"output cannot be a symlink: {absolute}")
    return absolute.parent.resolve(strict=False) / absolute.name


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _ensure_independent_output(output: Path, sources: Sequence[Path]) -> None:
    if output == Path(output.anchor):
        raise ArtifactError("output cannot be a filesystem root")
    if output.is_symlink():
        raise ArtifactError(f"output cannot be a symlink: {output}")
    if output.exists() and not output.is_dir():
        raise ArtifactError(f"output must be a directory: {output}")
    for source in sources:
        if _is_within(output, source) or _is_within(source, output):
            raise ArtifactError(
                f"output must be independent of input path {source}: {output}"
            )


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_link_target(source: Path, destination: Path) -> str:
    return os.path.relpath(source, start=destination.parent)


def _atomic_relative_symlink(source: Path, destination: Path) -> str:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_symlink():
        raise ArtifactError(f"refusing to replace non-symlink: {destination}")
    relative_target = _relative_link_target(source, destination)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        os.symlink(relative_target, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if destination.resolve(strict=True) != source:
        raise ArtifactError(
            f"created symlink does not resolve to {source}: {destination}"
        )
    return relative_target


def _preflight_output(
    output: Path,
    artifact_id: str,
    *,
    manifest_name: str,
    expected_kind: str,
    recoverable_partial: Callable[[Path], bool] | None = None,
    compatible_existing: Callable[[Mapping[str, Any]], bool] | None = None,
) -> tuple[str, Mapping[str, Any] | None]:
    manifest_path = output / manifest_name
    if manifest_path.exists() or manifest_path.is_symlink():
        _, existing = _load_complete_manifest(
            manifest_path, "existing output manifest", expected_kind=expected_kind
        )
        if existing.get("artifact_id") != artifact_id:
            if compatible_existing is not None and compatible_existing(existing):
                return "migrate", existing
            raise ArtifactError(
                f"{output} belongs to different parameters or inputs; "
                "choose a new output directory"
            )
        return "refresh", existing
    if output.exists() and any(output.iterdir()):
        if recoverable_partial is not None and recoverable_partial(output):
            return "recover", None
        raise ArtifactError(
            f"output is non-empty and has no managed {manifest_name}: {output}"
        )
    return "create", None


def _recoverable_workspace(output: Path) -> bool:
    """Recognize only links that this tool can leave before manifest commit."""

    entries = {path.name: path for path in output.iterdir()}
    if not entries or not set(entries).issubset({"cfg_args", "point_cloud"}):
        return False
    return all(path.is_symlink() for path in entries.values())


def _semantic_input_consistency(
    semantic_manifest: Mapping[str, Any],
    semantic_ply: Path,
    classifier: Path,
    bridge_manifest: Mapping[str, Any],
) -> None:
    gaussian_record = _manifest_artifact(
        semantic_manifest, "gaussian_ply", "semantic manifest"
    )
    classifier_record = _manifest_artifact(
        semantic_manifest, "classifier", "semantic manifest"
    )
    _verify_manifest_artifact(
        gaussian_record, semantic_ply, "semantic manifest.inputs.gaussian_ply"
    )
    _verify_manifest_artifact(
        classifier_record, classifier, "semantic manifest.inputs.classifier"
    )

    semantic_mesh = _manifest_artifact(semantic_manifest, "mesh", "semantic manifest")
    semantic_mesh_path = _path_from_manifest(
        semantic_mesh.get("path"), "semantic manifest.inputs.mesh.path"
    )
    edgs = _mapping(bridge_manifest.get("edgs"), "bridge manifest edgs")
    bridge_mesh_path = _path_from_manifest(
        edgs.get("mesh_path"), "bridge manifest edgs.mesh_path"
    )
    if semantic_mesh_path != bridge_mesh_path:
        raise ArtifactError(
            "semantic manifest mesh does not match the EDGS bridge mesh: "
            f"{semantic_mesh_path} != {bridge_mesh_path}"
        )


def _validate_semantic_ply(path: Path, sh_degree: int) -> int:
    try:
        header = validate_gaussian_ply(path, sh_degree)
    except BridgeError as exc:
        raise ArtifactError(str(exc)) from exc
    vertex = header.element("vertex")
    if vertex is None or vertex.count <= 0:
        raise ArtifactError(f"semantic Gaussian PLY has no points: {path}")
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
            "semantic Gaussian PLY must contain exactly obj_dc_0..obj_dc_15: "
            + "; ".join(details)
        )
    invalid = [
        name
        for name in sorted(expected)
        if properties[name].is_list
        or properties[name].value_type not in _PLY_FLOAT_TYPES
    ]
    if invalid:
        raise ArtifactError(
            "semantic Gaussian object embeddings must be floating-point scalars: "
            + ", ".join(invalid)
        )
    return int(vertex.count)


def prepare_workspace(
    semantic_model: Path,
    iteration: int,
    bridge_manifest_path: Path,
    semantic_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    if isinstance(iteration, bool) or iteration <= 0:
        raise ArtifactError("--iteration must be a positive integer")
    try:
        semantic_model = semantic_model.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactError(f"semantic model does not exist: {semantic_model}") from exc
    if not semantic_model.is_dir():
        raise ArtifactError(f"semantic model must be a directory: {semantic_model}")

    cfg_path = _regular_file(semantic_model / "cfg_args", "semantic cfg_args")
    iteration_root = semantic_model / "point_cloud" / f"iteration_{iteration}"
    semantic_ply = _regular_file(
        iteration_root / "point_cloud.ply", "semantic Gaussian PLY"
    )
    classifier = _regular_file(iteration_root / "classifier.pth", "classifier")
    bridge_path, bridge_manifest = _load_complete_manifest(
        bridge_manifest_path,
        "bridge manifest",
        expected_kind="edgs-inpaint360gs-bridge",
    )
    semantic_manifest_resolved, semantic_manifest = _load_complete_manifest(
        semantic_manifest_path, "semantic manifest"
    )
    cfg_values = _validate_cfg_args(cfg_path, semantic_model, bridge_manifest)
    _semantic_input_consistency(
        semantic_manifest, semantic_ply, classifier, bridge_manifest
    )
    point_count = _validate_semantic_ply(semantic_ply, cfg_values["sh_degree"])

    counts = _mapping(semantic_manifest.get("counts"), "semantic manifest counts")
    if counts.get("embedding_dimensions") != OBJECT_EMBEDDING_DIMENSIONS:
        raise ArtifactError(
            "semantic manifest must report exactly 16 object embedding dimensions"
        )
    if counts.get("classes") != cfg_values["num_classes"]:
        raise ArtifactError(
            "semantic manifest class count does not match semantic cfg_args"
        )
    if counts.get("gaussians") != point_count:
        raise ArtifactError(
            "semantic manifest Gaussian count does not match the semantic PLY"
        )

    output = _absolute_output(output)
    _ensure_independent_output(
        output,
        (semantic_model, bridge_path.parent, semantic_manifest_resolved.parent),
    )

    inputs = {
        "semantic_cfg_args": _artifact(cfg_path, "semantic cfg_args"),
        "semantic_gaussian_ply": _artifact(semantic_ply, "semantic Gaussian PLY"),
        "classifier": _artifact(classifier, "classifier"),
        "bridge_manifest": _artifact(bridge_path, "bridge manifest"),
        "semantic_manifest": _artifact(semantic_manifest_resolved, "semantic manifest"),
    }
    identity_payload = {
        "kind": WORKSPACE_KIND,
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "iteration": iteration,
        "bridge_artifact_id": bridge_manifest.get("artifact_id"),
        "inputs": {name: value["sha256"] for name, value in inputs.items()},
    }
    artifact_id = _identity(identity_payload)
    action, existing = _preflight_output(
        output,
        artifact_id,
        manifest_name=MANIFEST_NAME,
        expected_kind=WORKSPACE_KIND,
        recoverable_partial=_recoverable_workspace,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    if existing is not None and isinstance(existing.get("created_at"), str):
        created_at = str(existing["created_at"])

    cfg_link = output / "cfg_args"
    point_cloud_link = output / "point_cloud"
    manifest: dict[str, Any] = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "kind": WORKSPACE_KIND,
        "complete": True,
        "status": "complete",
        "artifact_id": artifact_id,
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {"iteration": iteration},
        "semantic_model": str(semantic_model),
        "gaussian_point_count": point_count,
        "bridge_artifact_id": bridge_manifest.get("artifact_id"),
        "inputs": inputs,
        "workspace": {
            "root": str(output),
            "cfg_args": "cfg_args",
            "cfg_args_link_target": _relative_link_target(cfg_path, cfg_link),
            "point_cloud": "point_cloud",
            "point_cloud_link_target": _relative_link_target(
                semantic_model / "point_cloud", point_cloud_link
            ),
            "iteration_root": f"point_cloud/iteration_{iteration}",
            "point_cloud_ply": f"point_cloud/iteration_{iteration}/point_cloud.ply",
            "classifier": f"point_cloud/iteration_{iteration}/classifier.pth",
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    _atomic_relative_symlink(cfg_path, cfg_link)
    _atomic_relative_symlink(semantic_model / "point_cloud", point_cloud_link)
    # A refresh may repair managed links, but it must not make the immutable
    # logical artifact look different to downstream stages merely by changing
    # ``updated_at`` (and therefore the manifest SHA-256).
    if action != "refresh":
        _atomic_write_json(output / MANIFEST_NAME, manifest)
    return {
        "action": action,
        "artifact_id": artifact_id,
        "iteration": iteration,
        "output": str(output),
        "point_cloud": str(
            output / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        ),
        "classifier": str(
            output / "point_cloud" / f"iteration_{iteration}" / "classifier.pth"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a completed paintmesh semantic reconstruction and create "
            "an isolated Inpaint360GS object-removal workspace."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--semantic-model",
        type=Path,
        required=True,
        help="semantic_3dgs model containing cfg_args and point_cloud/",
    )
    parser.add_argument(
        "--iteration", type=int, required=True, help="semantic 3DGS iteration"
    )
    parser.add_argument(
        "--bridge-manifest",
        type=Path,
        required=True,
        help="completed EDGS bridge_manifest.json",
    )
    parser.add_argument(
        "--semantic-manifest",
        type=Path,
        required=True,
        help="completed mesh semantic_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="independent work_model directory to create",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = prepare_workspace(
            args.semantic_model,
            args.iteration,
            args.bridge_manifest,
            args.semantic_manifest,
            args.output,
        )
    except (ArtifactError, OSError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
