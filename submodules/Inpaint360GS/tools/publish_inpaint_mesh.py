#!/usr/bin/env python3
"""Bind a PGSR/TSDF mesh to an immutable PaintMesh inpainted model."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tools.prepare_removal_workspace import (
        ArtifactError,
        _artifact,
        _atomic_write_json,
        _identity,
        _load_complete_manifest,
        _mapping,
        _regular_file,
    )
except ModuleNotFoundError:  # Direct repository-local execution.
    from prepare_removal_workspace import (
        ArtifactError,
        _artifact,
        _atomic_write_json,
        _identity,
        _load_complete_manifest,
        _mapping,
        _regular_file,
    )


KIND = "paintmesh-pgsr-inpaint-mesh"
SCHEMA_VERSION = 1
IDENTITY_VERSION = 2


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _load_render_manifest(
    path: Path, iteration: int, split: str
) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, f"{split} render manifest")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read render manifest {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"render manifest must be a JSON object: {resolved}")
    expected = {
        "backend": "pgsr",
        "complete": True,
        "iteration": iteration,
        "split": split,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ArtifactError(
                f"{split} render manifest {key}={payload.get(key)!r}, expected {value!r}"
            )
    views = payload.get("views")
    if not isinstance(views, Mapping) or len(views) != payload.get("num_views"):
        raise ArtifactError(f"{split} render manifest has an invalid view mapping")
    return resolved, payload


def _mesh_counts(path: Path) -> tuple[int, int]:
    """Read only the PLY header so multi-GB meshes remain memory-safe."""

    vertices = faces = None
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ArtifactError(f"PGSR output is not a PLY file: {path}")
        for _ in range(100_000):
            line = stream.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped.startswith(b"element vertex "):
                vertices = int(stripped.split()[2])
            elif stripped.startswith(b"element face "):
                faces = int(stripped.split()[2])
            elif stripped == b"end_header":
                break
        else:  # pragma: no cover - defensive malformed-file guard.
            raise ArtifactError(f"PLY header is unreasonably long: {path}")
    if vertices is None or faces is None or vertices <= 0 or faces <= 0:
        raise ArtifactError(
            f"PGSR mesh must contain vertices and faces, got {vertices}/{faces}: {path}"
        )
    return vertices, faces


def publish_mesh(
    *,
    model_manifest_path: Path,
    gaussian_ply: Path,
    mesh: Path,
    train_render_manifest: Path,
    test_render_manifest: Path | None,
    output: Path,
    iteration: int,
    source_path: Path,
    images: str,
    resolution: int,
    max_depth: float,
    voxel_size: float,
    num_clusters: int,
    use_depth_filter: bool,
) -> dict[str, Any]:
    model_path, model_manifest = _load_complete_manifest(
        model_manifest_path,
        "inpainted model manifest",
        expected_kind="paintmesh-inpainted-edgs-model",
    )
    model_artifact_id = model_manifest.get("artifact_id")
    if not isinstance(model_artifact_id, str) or not model_artifact_id:
        raise ArtifactError("inpainted model manifest has no artifact_id")
    model_parameters = _mapping(model_manifest.get("parameters"), "model parameters")
    if model_parameters.get("output_iteration") != iteration:
        raise ArtifactError(
            "mesh iteration does not match the inpainted model output iteration"
        )

    gaussian = _regular_file(gaussian_ply, "inpainted Gaussian PLY")
    model_root = model_path.parent.resolve(strict=True)
    model_info = _mapping(model_manifest.get("model"), "model manifest model")
    relative_gaussian = model_info.get("point_cloud")
    if not isinstance(relative_gaussian, str) or not relative_gaussian:
        raise ArtifactError("model manifest does not record model.point_cloud")
    published_gaussian = (model_root / relative_gaussian).resolve(strict=True)
    if gaussian.resolve(strict=True) != published_gaussian:
        raise ArtifactError(
            f"Gaussian PLY is not the model-manifest checkpoint: {gaussian} != {published_gaussian}"
        )

    mesh_path = _regular_file(mesh, "PGSR TSDF mesh")
    vertices, triangles = _mesh_counts(mesh_path)
    train_path, train_payload = _load_render_manifest(
        train_render_manifest, iteration, "train"
    )
    test_path = None
    test_payload = None
    if test_render_manifest is not None:
        test_path, test_payload = _load_render_manifest(
            test_render_manifest, iteration, "test"
        )

    source = source_path.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ArtifactError(f"source path is not a directory: {source}")
    if (
        not images
        or Path(images).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(images).parts)
    ):
        raise ArtifactError("--images must be a safe relative directory name")

    inputs = {
        "model_manifest": {
            **_artifact(model_path, "inpainted model manifest"),
            "artifact_id": model_artifact_id,
        },
        "gaussian_ply": _artifact(gaussian, "inpainted Gaussian PLY"),
    }
    outputs: dict[str, Any] = {
        "mesh": _artifact(mesh_path, "PGSR TSDF mesh"),
        "train_render_manifest": _artifact(train_path, "train render manifest"),
        "test_render_manifest": (
            _artifact(test_path, "test render manifest") if test_path else None
        ),
    }
    parameters = {
        "iteration": iteration,
        "renderer": "pgsr",
        "source_path": str(source),
        "images": images,
        "resolution": resolution,
        "max_depth": max_depth,
        "voxel_size": voxel_size,
        "num_clusters": num_clusters,
        "use_depth_filter": bool(use_depth_filter),
        "render_test": test_path is not None,
    }
    identity_payload = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "identity_version": IDENTITY_VERSION,
        # The producer's artifact ID is the semantic identity of the model
        # manifest.  Its JSON bytes may change during an idempotent refresh.
        "model_artifact_id": model_artifact_id,
        "parameters": parameters,
        "inputs": {
            "gaussian_ply": inputs["gaussian_ply"]["sha256"],
        },
        # Bind every PGSR output consumed by later stages.  The explicit None
        # keeps train-only renders distinct from train+test renders.
        "outputs": {
            "mesh": outputs["mesh"]["sha256"],
            "train_render_manifest": outputs["train_render_manifest"]["sha256"],
            "test_render_manifest": (
                outputs["test_render_manifest"]["sha256"]
                if outputs["test_render_manifest"] is not None
                else None
            ),
        },
    }
    artifact_id = _identity(identity_payload)
    output_path = output.expanduser().resolve()
    if output_path.exists():
        _, existing = _load_complete_manifest(
            output_path, "existing mesh-stage manifest", expected_kind=KIND
        )
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactError(
                f"{output_path} uses an unsupported schema_version; "
                "choose a new inpaint run"
            )
        if existing.get("identity_version") != IDENTITY_VERSION:
            raise ArtifactError(
                f"{output_path} uses a legacy mesh identity; "
                "choose a new inpaint run instead of migrating it implicitly"
            )
        if existing.get("artifact_id") != artifact_id:
            raise ArtifactError(
                f"{output_path} belongs to different inputs/settings; choose a new inpaint run"
            )
        # A matching v2 identity already binds the current model, Gaussian,
        # mesh and render manifests.  Preserve the original commit marker
        # byte-for-byte so downstream artifact records remain valid.
        return dict(existing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "identity_version": IDENTITY_VERSION,
        "kind": KIND,
        "complete": True,
        "status": "complete",
        "artifact_id": artifact_id,
        "created_at": now,
        "model_artifact_id": model_artifact_id,
        "parameters": parameters,
        "inputs": inputs,
        "outputs": outputs,
        "counts": {
            "mesh_vertices": vertices,
            "mesh_triangles": triangles,
            "train_views": int(train_payload["num_views"]),
            "test_views": int(test_payload["num_views"]) if test_payload else 0,
        },
    }
    _atomic_write_json(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--gaussian-ply", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--train-render-manifest", required=True, type=Path)
    parser.add_argument("--test-render-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=_positive_int)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", required=True, type=_positive_int)
    parser.add_argument("--max-depth", default=5.0, type=_positive_float)
    parser.add_argument("--voxel-size", default=0.002, type=_positive_float)
    parser.add_argument("--num-clusters", default=1, type=_positive_int)
    parser.add_argument("--use-depth-filter", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = publish_mesh(
            model_manifest_path=args.model_manifest,
            gaussian_ply=args.gaussian_ply,
            mesh=args.mesh,
            train_render_manifest=args.train_render_manifest,
            test_render_manifest=args.test_render_manifest,
            output=args.output,
            iteration=args.iteration,
            source_path=args.source_path,
            images=args.images,
            resolution=args.resolution,
            max_depth=args.max_depth,
            voxel_size=args.voxel_size,
            num_clusters=args.num_clusters,
            use_depth_filter=args.use_depth_filter,
        )
    except ArtifactError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"PGSR inpaint mesh: {args.output.resolve()} ({manifest['artifact_id']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
