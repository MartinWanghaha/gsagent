#!/usr/bin/env python3
"""Extract one Region-Conditioned Gaussian Wrapping mesh."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Optional

from mesh.extraction_context import MeshExtractionContext
from mesh.io import export_mesh, load_mesh, load_points
from mesh.metrics import compute_mesh_metrics
from mesh.region_wrapping import (
    ALGORITHM,
    SCHEMA_VERSION,
    RegionConditionedGaussianWrappingExtractor,
    RegionGaussianWrappingConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model-path", required=True, help="trained scene directory")
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="checkpoint iteration; -1 selects the latest complete checkpoint",
    )
    parser.add_argument("--output", required=True, help="output binary .ply mesh")
    parser.add_argument("--device", default="cuda", help="inference device")

    quality = parser.add_argument_group("region-conditioned Gaussian Wrapping")
    quality.add_argument(
        "--max-gaussians",
        type=int,
        help="semantic-balanced Gaussian anchor budget",
    )
    quality.add_argument(
        "--max-chart-gaussians",
        type=int,
        help="maximum owned Gaussians in one local Delaunay chart",
    )
    quality.add_argument(
        "--view-stride",
        type=int,
        help="use every Nth ordered training view; default uses every view",
    )
    quality.add_argument(
        "--camera-scale",
        type=float,
        help="surface-evidence render scale; default preserves training resolution",
    )
    quality.add_argument(
        "--target-faces",
        type=int,
        help="optional seam-aware final face budget",
    )

    metrics = parser.add_argument_group("optional evaluation")
    metrics.add_argument("--reference", help="reference mesh or point cloud")
    metrics.add_argument("--metric-threshold", type=float, default=0.01)
    metrics.add_argument("--metric-samples", type=int, default=100_000)
    metrics.add_argument("--metric-seed", type=int, default=0)
    metrics.add_argument("--metrics-json", help="optional standalone metrics JSON")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.iteration == 0 or args.iteration < -1:
        parser.error("--iteration must be positive or -1")
    if Path(args.output).suffix.lower() != ".ply":
        parser.error("--output must end in .ply")
    if args.max_gaussians is not None and args.max_gaussians < 4:
        parser.error("--max-gaussians must be at least 4")
    if args.max_chart_gaussians is not None and args.max_chart_gaussians < 4:
        parser.error("--max-chart-gaussians must be at least 4")
    if args.view_stride is not None and args.view_stride < 1:
        parser.error("--view-stride must be positive")
    if args.camera_scale is not None and not 0.0 < args.camera_scale <= 1.0:
        parser.error("--camera-scale must lie in (0,1]")
    if args.target_faces is not None and args.target_faces < 1:
        parser.error("--target-faces must be positive")
    if not math.isfinite(args.metric_threshold) or args.metric_threshold <= 0.0 or args.metric_samples < 1:
        parser.error("metric threshold and sample count must be positive")
    if args.metric_seed < 0:
        parser.error("--metric-seed must be non-negative")
    if args.metrics_json and not args.reference:
        parser.error("--metrics-json requires --reference")


def _resolved_config(
    context: MeshExtractionContext,
    args: argparse.Namespace,
) -> RegionGaussianWrappingConfig:
    configured = context.experiment_config.get("mesh_export", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise TypeError("mesh_export configuration must be a mapping")
    config = RegionGaussianWrappingConfig.from_mapping(configured)
    overrides: dict[str, Any] = {}
    for argument, field in (
        ("max_gaussians", "max_gaussians"),
        ("max_chart_gaussians", "max_chart_gaussians"),
        ("view_stride", "view_stride"),
        ("camera_scale", "camera_scale"),
        ("target_faces", "target_faces"),
    ):
        value = getattr(args, argument)
        if value is not None:
            overrides[field] = value
    return replace(config, **overrides).validated()


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".json")


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _atomic_export(mesh, output: Path) -> None:
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


def _reference_geometry(path: str):
    suffix = Path(path).suffix.lower()
    return load_mesh(path) if suffix in {".ply", ".obj"} else load_points(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    context = MeshExtractionContext.load(
        args.model_path,
        iteration=args.iteration,
        device=args.device,
    )
    config = _resolved_config(context, args)
    extractor = RegionConditionedGaussianWrappingExtractor(
        context,
        config=config,
        progress_callback=print,
    )
    mesh = extractor.extract()

    metrics = None
    if args.reference:
        metrics = compute_mesh_metrics(
            mesh,
            _reference_geometry(args.reference),
            threshold=args.metric_threshold,
            sample_count=args.metric_samples,
            seed=args.metric_seed,
        ).as_dict()

    output = Path(args.output).expanduser().resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "model_path": str(context.model_path),
        "checkpoint": str(context.checkpoint_path),
        "iteration": context.iteration,
        "train_cameras": len(context.cameras),
        "output": str(output),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extraction_config": config.as_dict(),
        "metadata": mesh.metadata,
    }
    if metrics is not None:
        manifest["metrics"] = metrics

    _atomic_export(mesh, output)
    _atomic_json(manifest, _manifest_path(output))
    if args.metrics_json and metrics is not None:
        _atomic_json(metrics, Path(args.metrics_json).expanduser().resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
