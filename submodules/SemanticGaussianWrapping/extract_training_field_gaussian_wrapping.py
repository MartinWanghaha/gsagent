#!/usr/bin/env python3
"""Extract a mesh with GaussianWrapping topology and the trained surface field."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Optional
import uuid

from extract_training_field_mesh import _atomic_export, _atomic_json
from mesh.training_field_gaussian_wrapping import (
    ALGORITHM,
    SCHEMA_VERSION,
    TrainingFieldGaussianWrappingConfig,
    TrainingFieldGaussianWrappingExtractor,
)
from model_io import load_trained_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model-path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-gaussians", type=int, default=300_000)
    parser.add_argument(
        "--local-charts",
        action="store_true",
        help="use bounded local Delaunay charts instead of original global GW topology",
    )
    parser.add_argument("--max-core-gaussians", type=int, default=2_500)
    parser.add_argument("--max-halo-gaussians", type=int, default=5_000)
    parser.add_argument("--halo-spacing-factor", type=float, default=4.0)
    parser.add_argument("--pivot-sigma-factor", type=float, default=3.0)
    parser.add_argument("--min-pivot-spacing-factor", type=float, default=0.08)
    parser.add_argument("--max-pivot-spacing-factor", type=float, default=1.0)
    parser.add_argument("--support-sigma-factor", type=float, default=3.0)
    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--min-semantic-confidence", type=float, default=0.35)
    parser.add_argument("--trim-quantile", type=float, default=0.001)
    parser.add_argument("--allow-unobserved", action="store_true")
    parser.add_argument("--query-chunk-size", type=int, default=2_048)
    parser.add_argument("--root-steps", type=int, default=14)
    parser.add_argument("--root-tolerance", type=float, default=1e-5)
    parser.add_argument("--semantic-decode-chunk-size", type=int, default=8_192)
    parser.add_argument("--min-component-faces", type=int, default=64)
    parser.add_argument("--max-edge-support-factor", type=float, default=1.25)
    parser.add_argument("--max-edge-spacing-factor", type=float, default=4.0)
    parser.add_argument("--max-tetra-edge-ratio", type=float, default=30.0)
    parser.add_argument("--min-tetra-volume-ratio", type=float, default=1e-5)
    parser.add_argument("--max-circumradius-to-edge", type=float, default=4.0)
    parser.add_argument("--max-face-aspect-ratio", type=float, default=30.0)
    return parser


def _config(args: argparse.Namespace) -> TrainingFieldGaussianWrappingConfig:
    return TrainingFieldGaussianWrappingConfig(
        max_gaussians=args.max_gaussians,
        global_delaunay=not args.local_charts,
        max_core_gaussians=args.max_core_gaussians,
        max_halo_gaussians=args.max_halo_gaussians,
        halo_spacing_factor=args.halo_spacing_factor,
        pivot_sigma_factor=args.pivot_sigma_factor,
        min_pivot_spacing_factor=args.min_pivot_spacing_factor,
        max_pivot_spacing_factor=args.max_pivot_spacing_factor,
        support_sigma_factor=args.support_sigma_factor,
        min_opacity=args.min_opacity,
        min_semantic_confidence=args.min_semantic_confidence,
        trim_quantile=args.trim_quantile,
        require_observation=not args.allow_unobserved,
        query_chunk_size=args.query_chunk_size,
        root_steps=args.root_steps,
        root_tolerance=args.root_tolerance,
        semantic_decode_chunk_size=args.semantic_decode_chunk_size,
        min_component_faces=args.min_component_faces,
        max_edge_support_factor=args.max_edge_support_factor,
        max_edge_spacing_factor=args.max_edge_spacing_factor,
        max_tetra_edge_ratio=args.max_tetra_edge_ratio,
        min_tetra_volume_ratio=args.min_tetra_volume_ratio,
        max_circumradius_to_edge=args.max_circumradius_to_edge,
        max_face_aspect_ratio=args.max_face_aspect_ratio,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.iteration == 0 or args.iteration < -1:
        parser.error("--iteration must be positive or -1")
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".ply":
        parser.error("--output must end in .ply")
    config = _config(args)
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
        raise RuntimeError("a complete chkpnt*.pth is required")
    mesh = TrainingFieldGaussianWrappingExtractor(
        bundle["surface_field"],
        bundle["gaussians"],
        bundle["semantic_decoder"],
        config=config,
        progress_callback=lambda message: print(message, flush=True),
    ).extract()
    config_payload = config.as_dict()
    fingerprint = hashlib.sha256(
        json.dumps(
            config_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "field_source": "training_semantic_surface_field",
        "model_path": str(model_path),
        "checkpoint": str(Path(checkpoint).resolve()),
        "iteration": int(bundle["iteration"]),
        "device": str(bundle["device"]),
        "gaussians": int(len(bundle["gaussians"])),
        "semantic_dim": int(bundle["gaussians"].semantic_dim),
        "semantic_classes": int(bundle["semantic_decoder"].linear.out_features),
        "geometry_experts": int(bundle["gaussians"].geometry_experts),
        "coordinate_frame": "training_world",
        "extraction_config": config_payload,
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
