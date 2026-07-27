#!/usr/bin/env python3
"""Extract a high-precision mesh from the exact surface field used in training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Optional
import uuid

from mesh.io import export_mesh
from mesh.training_field_extraction import (
    ALGORITHM,
    SCHEMA_VERSION,
    TrainingFieldMeshConfig,
    TrainingFieldMeshExtractor,
)
from model_io import load_trained_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        help="trained SemanticGaussianWrapping experiment directory",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="complete checkpoint iteration; -1 selects the latest",
    )
    parser.add_argument("--output", help="output binary .ply mesh")
    parser.add_argument("--device", default="cuda", help="inference device")
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="load the final training field and report the sparse grid without meshing",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="optional explicit world-space extraction bounds",
    )

    quality = parser.add_argument_group("training-field extraction")
    quality.add_argument("--resolution", type=int, default=512)
    quality.add_argument("--block-cells", type=int, default=32)
    quality.add_argument(
        "--support-halo",
        choices=("none", "face", "full"),
        default="face",
    )
    quality.add_argument("--support-sigma", type=float, default=3.0)
    quality.add_argument("--relative-padding", type=float, default=0.02)
    quality.add_argument("--trim-quantile", type=float, default=0.001)
    quality.add_argument("--min-opacity", type=float, default=0.05)
    quality.add_argument("--min-semantic-confidence", type=float, default=0.35)
    quality.add_argument(
        "--allow-unobserved",
        action="store_true",
        help="do not require positive training observation_count",
    )
    quality.add_argument("--query-chunk-size", type=int, default=2_048)
    quality.add_argument(
        "--scout-resolution",
        type=int,
        default=0,
        help="coarse exact-field resolution used to select a high-resolution narrow band",
    )
    quality.add_argument(
        "--scout-near-surface-voxels",
        type=float,
        default=0.75,
    )
    quality.add_argument(
        "--no-boundary-completion",
        action="store_true",
        help="disable dynamic expansion across zero-crossing block faces",
    )
    quality.add_argument("--projection-steps", type=int, default=4)
    quality.add_argument("--projection-step-voxels", type=float, default=0.5)
    quality.add_argument(
        "--projection-tolerance-voxels",
        type=float,
        default=0.01,
    )
    quality.add_argument("--semantic-decode-chunk-size", type=int, default=8_192)
    quality.add_argument("--min-component-faces", type=int, default=64)
    quality.add_argument("--weld-tolerance-voxels", type=float, default=1e-4)
    return parser


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.iteration == 0 or args.iteration < -1:
        parser.error("--iteration must be positive or -1")
    if not args.layout_only and not args.output:
        parser.error("--output is required unless --layout-only is used")
    if args.output and Path(args.output).suffix.lower() != ".ply":
        parser.error("--output must end in .ply")
    if args.bounds is not None:
        lower, upper = args.bounds[:3], args.bounds[3:]
        if not all(math.isfinite(value) for value in args.bounds):
            parser.error("--bounds values must be finite")
        if any(high <= low for low, high in zip(lower, upper)):
            parser.error("--bounds maximum must exceed minimum on every axis")


def _resolved_config(args: argparse.Namespace) -> TrainingFieldMeshConfig:
    return TrainingFieldMeshConfig(
        resolution=args.resolution,
        block_cells=args.block_cells,
        support_halo=args.support_halo,
        support_sigma=args.support_sigma,
        relative_padding=args.relative_padding,
        trim_quantile=args.trim_quantile,
        min_opacity=args.min_opacity,
        min_semantic_confidence=args.min_semantic_confidence,
        require_observation=not args.allow_unobserved,
        scout_resolution=args.scout_resolution,
        scout_near_surface_voxels=args.scout_near_surface_voxels,
        complete_boundary_neighbors=not args.no_boundary_completion,
        query_chunk_size=args.query_chunk_size,
        projection_steps=args.projection_steps,
        projection_step_voxels=args.projection_step_voxels,
        projection_tolerance_voxels=args.projection_tolerance_voxels,
        semantic_decode_chunk_size=args.semantic_decode_chunk_size,
        min_component_faces=args.min_component_faces,
        weld_tolerance_voxels=args.weld_tolerance_voxels,
    )


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _atomic_export(mesh: Any, output: Path) -> None:
    temporary = _temporary_path(output)
    try:
        export_mesh(mesh, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], output: Path) -> None:
    temporary = _temporary_path(output)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".json")


def _config_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _field_manifest(bundle: dict[str, Any]) -> dict[str, Any]:
    surface = bundle["config"]["surface"]
    field = bundle["surface_field"]
    return {
        "implementation": "SemanticSurfaceField.query_geometry",
        "scalar": "sdf",
        "level": 0.0,
        "occupancy_isovalue": float(surface["occupancy_isovalue"]),
        "density_scale": float(field.density_scale),
        "field_neighbors": int(surface["field_neighbors"]),
        "region_candidate_neighbors": int(
            surface["region_candidate_neighbors"]
        ),
        "neighbor_backend": str(surface["neighbor_backend"]),
        "support_log_cutoff": float(surface["support_log_cutoff"]),
        "support_candidate_budget": int(
            surface["support_candidate_budget"]
        ),
        "support_routing_query_chunk": int(
            surface["support_routing_query_chunk"]
        ),
        "max_distance_bytes": int(surface["max_distance_bytes"]),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    extraction_config = _resolved_config(args)
    started = time.monotonic()

    model_path = Path(args.model_path).expanduser().resolve()
    bundle = load_trained_scene(
        model_path,
        iteration=args.iteration,
        device=args.device,
        with_surface_field=True,
        inference_scope="surface",
    )
    checkpoint = bundle.get("checkpoint_path")
    if checkpoint is None or not Path(checkpoint).is_file():
        raise RuntimeError(
            "training-field mesh extraction requires a complete chkpnt*.pth; "
            "a PLY-only model is insufficient"
        )
    surface_field = bundle["surface_field"]
    semantic_decoder = bundle["semantic_decoder"]
    if surface_field is None or semantic_decoder is None:
        raise RuntimeError(
            "checkpoint did not restore the training surface field and decoder"
        )

    extractor = TrainingFieldMeshExtractor(
        surface_field,
        bundle["gaussians"],
        semantic_decoder,
        config=extraction_config,
        bounds=args.bounds,
        progress_callback=lambda message: print(message, flush=True),
    )
    mesh, layout = extractor.extract(layout_only=args.layout_only)
    decoder_classes = int(semantic_decoder.linear.out_features)
    train_cameras = len(bundle["scene"].getTrainCameras())
    topology_generation = int(
        getattr(bundle["gaussians"], "topology_generation", 0)
    )
    topology_churn = int(
        getattr(bundle["gaussians"], "cumulative_topology_churn", 0)
    )
    config_payload = extraction_config.as_dict()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "field_source": "training_semantic_surface_field",
        "model_path": str(model_path),
        "checkpoint": str(Path(checkpoint).resolve()),
        "iteration": int(bundle["iteration"]),
        "device": str(bundle["device"]),
        "train_cameras": int(train_cameras),
        "gaussians": int(len(bundle["gaussians"])),
        "topology_generation": topology_generation,
        "cumulative_topology_churn": topology_churn,
        "semantic_dim": int(bundle["gaussians"].semantic_dim),
        "semantic_classes": decoder_classes,
        "geometry_experts": int(bundle["gaussians"].geometry_experts),
        "coordinate_frame": "training_world",
        "surface_field": _field_manifest(bundle),
        "extraction_config": config_payload,
        "extraction_config_sha256": _config_fingerprint(config_payload),
        "layout": layout.as_dict(),
        "elapsed_seconds": float(time.monotonic() - started),
    }
    if args.layout_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    assert mesh is not None
    output = Path(args.output).expanduser().resolve()
    publication_id = uuid.uuid4().hex
    manifest.update(
        {
            "publication_id": publication_id,
            "output": str(output),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "metadata": mesh.metadata,
        }
    )
    sidecar = _manifest_path(output)
    # The manifest is the commit marker for the mesh/metadata pair.
    sidecar.unlink(missing_ok=True)
    _atomic_export(mesh, output)
    manifest["output_bytes"] = int(output.stat().st_size)
    manifest["elapsed_seconds"] = float(time.monotonic() - started)
    _atomic_json(manifest, sidecar)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
