#!/usr/bin/env python3
"""Extract a visually validated mesh from multiview-visible semantic Gaussians."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Optional
import uuid

from extract_training_field_mesh import _atomic_export, _atomic_json
from mesh.multiview_gaussian_extraction import (
    ALGORITHM,
    SCHEMA_VERSION,
    MultiviewGaussianMeshConfig,
    MultiviewGaussianMeshExtractor,
)
from model_io import load_trained_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view-count", type=int, default=60)
    parser.add_argument("--visibility-width", type=int, default=480)
    parser.add_argument("--minimum-visible-views", type=int, default=3)
    parser.add_argument(
        "--depth-tolerance-extent-fraction", type=float, default=0.0016
    )
    parser.add_argument("--voxel-extent-fraction", type=float, default=0.005)
    parser.add_argument("--normal-neighbors", type=int, default=32)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--poisson-scale", type=float, default=1.05)
    parser.add_argument("--poisson-threads", type=int, default=8)
    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--min-semantic-confidence", type=float, default=0.35)
    parser.add_argument("--allow-unobserved", action="store_true")
    parser.add_argument("--trim-quantile", type=float, default=0.001)
    parser.add_argument("--query-chunk-size", type=int, default=2_048)
    parser.add_argument("--semantic-decode-chunk-size", type=int, default=8_192)
    return parser


def _config(args: argparse.Namespace) -> MultiviewGaussianMeshConfig:
    return MultiviewGaussianMeshConfig(
        view_count=args.view_count,
        visibility_width=args.visibility_width,
        minimum_visible_views=args.minimum_visible_views,
        depth_tolerance_extent_fraction=args.depth_tolerance_extent_fraction,
        voxel_extent_fraction=args.voxel_extent_fraction,
        normal_neighbors=args.normal_neighbors,
        poisson_depth=args.poisson_depth,
        poisson_scale=args.poisson_scale,
        poisson_threads=args.poisson_threads,
        min_opacity=args.min_opacity,
        min_semantic_confidence=args.min_semantic_confidence,
        require_observation=not args.allow_unobserved,
        trim_quantile=args.trim_quantile,
        query_chunk_size=args.query_chunk_size,
        semantic_decode_chunk_size=args.semantic_decode_chunk_size,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".ply":
        parser.error("--output must end in .ply")
    config = _config(args)
    model_path = Path(args.model_path).expanduser().resolve()
    started = time.monotonic()
    bundle = load_trained_scene(
        model_path,
        iteration=args.iteration,
        device=args.device,
        with_surface_field=True,
        inference_scope="surface",
    )
    checkpoint = bundle.get("checkpoint_path")
    if checkpoint is None or not Path(checkpoint).is_file():
        raise RuntimeError("a complete chkpnt*.pth is required")
    scene = bundle["scene"]
    cameras = scene.getTrainCameras()
    scene_extent = float(scene.cameras_extent)
    mesh = MultiviewGaussianMeshExtractor(
        bundle["surface_field"],
        bundle["gaussians"],
        bundle["semantic_decoder"],
        cameras,
        scene_extent,
        config=config,
        progress_callback=lambda message: print(message, flush=True),
    ).extract()
    payload = config.as_dict()
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "model_path": str(model_path),
        "checkpoint": str(Path(checkpoint).resolve()),
        "iteration": int(bundle["iteration"]),
        "device": str(bundle["device"]),
        "gaussians": int(len(bundle["gaussians"])),
        "train_cameras": int(len(cameras)),
        "semantic_dim": int(bundle["gaussians"].semantic_dim),
        "semantic_classes": int(bundle["semantic_decoder"].linear.out_features),
        "geometry_experts": int(bundle["gaussians"].geometry_experts),
        "coordinate_frame": "training_world",
        "extraction_config": payload,
        "extraction_config_sha256": fingerprint,
        "publication_id": uuid.uuid4().hex,
        "output": str(output),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "metadata": mesh.metadata,
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.unlink(missing_ok=True)
    _atomic_export(mesh, output)
    manifest["output_bytes"] = int(output.stat().st_size)
    manifest["elapsed_seconds"] = float(time.monotonic() - started)
    _atomic_json(manifest, sidecar)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
