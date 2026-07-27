#!/usr/bin/env python3
"""Extract a semantic Gaussian mesh from a trained scene.

This public entry point resolves one complete SemanticGaussianWrapping
checkpoint and delegates either to the legacy region-conditioned extractor or
to the visually validated multiview-visible high-precision extractor.

Examples:
    python scripts/semantic_gaussian_wrapping/extract_semantic_gaussian_wrapping_mesh.py \
        outputs/semantic_gaussian_wrapping_mipnerf360/counter_rerun_003 --gpu 0

    python scripts/semantic_gaussian_wrapping/extract_semantic_gaussian_wrapping_mesh.py \
        outputs/semantic_gaussian_wrapping_mipnerf360/counter_rerun_003 \
        --iteration 30000 --max-gaussians 500000 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SGW_ROOT = GSAGENT_ROOT / "submodules" / "SemanticGaussianWrapping"
CHECKPOINT_PATTERN = re.compile(r"^chkpnt(\d+)\.pth$")
ALGORITHM = "region_conditioned_gaussian_wrapping"
SCHEMA_VERSION = 2
MULTIVIEW_ALGORITHM = "multiview-visible-oriented-semantic-gaussians"
MULTIVIEW_SCHEMA_VERSION = 1
METHOD_REGION = "region-conditioned"
METHOD_MULTIVIEW = "multiview-visible"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene_path",
        type=Path,
        help="Trained SemanticGaussianWrapping experiment directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output mesh. Default: "
            "<scene>/mesh/semantic_gaussian_wrapping_iteration_<iter>.ply"
        ),
    )
    parser.add_argument(
        "--sgw-root",
        type=Path,
        default=DEFAULT_SGW_ROOT,
        help=f"SemanticGaussianWrapping checkout. Default: {DEFAULT_SGW_ROOT}",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Checkpoint iteration; -1 selects the latest complete checkpoint.",
    )
    parser.add_argument(
        "--gpu",
        help="Value for CUDA_VISIBLE_DEVICES, for example 0.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device passed to the native extractor. Default: cuda.",
    )
    parser.add_argument(
        "--method",
        choices=(METHOD_REGION, METHOD_MULTIVIEW),
        default=METHOD_REGION,
        help=(
            "Extraction method. Use multiview-visible for the visually "
            "validated high-precision path. Default: region-conditioned."
        ),
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        help="Optional semantic-balanced Gaussian anchor budget.",
    )
    parser.add_argument(
        "--max-chart-gaussians",
        type=int,
        help="Maximum owned Gaussians in one local Delaunay chart.",
    )
    parser.add_argument(
        "--view-stride",
        type=int,
        help="Use every Nth training camera in deterministic UID order.",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        help="Camera-resolution scale in (0,1] used by visibility rendering.",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        help="Optional topology-aware simplification target.",
    )
    parser.add_argument("--reference", type=Path, help="Reference mesh or point cloud.")
    parser.add_argument("--metric-threshold", type=float, default=0.01)
    parser.add_argument("--metric-samples", type=int, default=100_000)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Optional metric JSON output; requires --reference.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing primary output mesh.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the native command without executing it.",
    )

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.iteration == 0 or args.iteration < -1:
        parser.error("--iteration must be positive or -1 for latest")
    if args.output is not None and args.output.suffix.lower() != ".ply":
        parser.error("--output must end in .ply")
    if args.max_gaussians is not None and args.max_gaussians < 4:
        parser.error("--max-gaussians must be at least 4")
    if args.max_chart_gaussians is not None and args.max_chart_gaussians < 4:
        parser.error("--max-chart-gaussians must be at least 4")
    if args.view_stride is not None and args.view_stride < 1:
        parser.error("--view-stride must be positive")
    if args.camera_scale is not None and (
        not math.isfinite(args.camera_scale)
        or args.camera_scale <= 0
        or args.camera_scale > 1
    ):
        parser.error("--camera-scale must lie in (0,1]")
    if args.target_faces is not None and args.target_faces < 1:
        parser.error("--target-faces must be positive")
    if (
        not math.isfinite(args.metric_threshold)
        or args.metric_threshold <= 0
        or args.metric_samples < 1
    ):
        parser.error("metric threshold and sample count must be positive")
    if args.metrics_json is not None and args.reference is None:
        parser.error("--metrics-json requires --reference")
    if args.method == METHOD_MULTIVIEW:
        incompatible = {
            "--max-gaussians": args.max_gaussians,
            "--max-chart-gaussians": args.max_chart_gaussians,
            "--view-stride": args.view_stride,
            "--camera-scale": args.camera_scale,
            "--target-faces": args.target_faces,
            "--reference": args.reference,
            "--metrics-json": args.metrics_json,
        }
        supplied = [name for name, value in incompatible.items() if value is not None]
        if supplied:
            parser.error(
                "multiview-visible does not accept legacy RC-GW options: "
                + ", ".join(supplied)
            )
    return args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_scene_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    candidates = [
        expanded.resolve(),
        (GSAGENT_ROOT / expanded).resolve(),
        (GSAGENT_ROOT.parent / expanded).resolve(),
    ]
    if expanded.parts and expanded.parts[0] == GSAGENT_ROOT.name:
        candidates.insert(0, (GSAGENT_ROOT.parent / expanded).resolve())
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("chkpnt*.pth")):
            return candidate
    return candidates[0]


def checkpoint_iterations(scene_path: Path) -> list[int]:
    return sorted(
        int(match.group(1))
        for checkpoint in scene_path.glob("chkpnt*.pth")
        if (match := CHECKPOINT_PATTERN.fullmatch(checkpoint.name))
    )


def resolve_iteration(scene_path: Path, requested: int) -> int:
    available = checkpoint_iterations(scene_path)
    if not available:
        raise FileNotFoundError(
            "Semantic mesh extraction requires a complete checkpoint with its "
            f"trained decoder: no chkpnt*.pth under {scene_path}"
        )
    if requested == -1:
        return available[-1]
    if requested not in available:
        raise FileNotFoundError(
            f"checkpoint iteration {requested} does not exist under {scene_path}"
        )
    return requested


def validate_inputs(
    scene_path: Path,
    sgw_root: Path,
    iteration: int,
    reference: Path | None = None,
    native_script: str = "extract_mesh.py",
) -> None:
    config_paths = (scene_path / "config.yaml", scene_path / "resolved_config.yaml")
    required = [
        sgw_root / native_script,
        scene_path / f"chkpnt{iteration}.pth",
    ]
    if reference is not None:
        required.append(reference)
    missing = [path for path in required if not path.is_file()]
    if not any(path.is_file() for path in config_paths):
        missing.append(scene_path / "config.yaml")
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(str(path) for path in missing)
        )


def default_output_path(
    scene_path: Path,
    iteration: int,
    method: str = METHOD_REGION,
) -> Path:
    if method == METHOD_MULTIVIEW:
        return (
            scene_path
            / "mesh"
            / f"semantic_multiview_gaussian_wrapping_iteration_{iteration}_high_precision.ply"
        )
    return scene_path / "mesh" / f"semantic_gaussian_wrapping_iteration_{iteration}.ply"


def manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


def output_is_complete(
    output_path: Path,
    iteration: int,
    *,
    algorithm: str = ALGORITHM,
    schema_version: int = SCHEMA_VERSION,
) -> bool:
    sidecar = manifest_path(output_path)
    if not output_path.is_file() or not sidecar.is_file():
        return False
    try:
        with output_path.open("rb") as handle:
            if handle.read(4) != b"ply\n":
                return False
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        payload.get("algorithm") == algorithm
        and int(payload.get("schema_version", -1)) == schema_version
        and int(payload.get("iteration", -1)) == int(iteration)
        and int(payload.get("vertices", 0)) > 0
        and int(payload.get("faces", 0)) > 0
    )


def build_command(
    args: argparse.Namespace,
    scene_path: Path,
    iteration: int,
    output_path: Path,
) -> list[str]:
    if args.method == METHOD_MULTIVIEW:
        return [
            sys.executable,
            "extract_multiview_gaussian_mesh.py",
            "-m",
            str(scene_path),
            "--iteration",
            str(iteration),
            "--output",
            str(output_path),
            "--device",
            str(args.device),
        ]
    command = [
        sys.executable,
        "extract_mesh.py",
        "-m",
        str(scene_path),
        "--iteration",
        str(iteration),
        "--output",
        str(output_path),
        "--device",
        str(args.device),
    ]
    extraction_options = (
        ("max_gaussians", "--max-gaussians"),
        ("max_chart_gaussians", "--max-chart-gaussians"),
        ("view_stride", "--view-stride"),
        ("camera_scale", "--camera-scale"),
        ("target_faces", "--target-faces"),
    )
    for attribute, option in extraction_options:
        value = getattr(args, attribute)
        if value is not None:
            command.extend((option, str(value)))

    if args.reference is not None:
        command.extend(
            (
                "--reference",
                str(resolve_path(args.reference)),
                "--metric-threshold",
                str(args.metric_threshold),
                "--metric-samples",
                str(args.metric_samples),
            )
        )
        if args.metrics_json is not None:
            command.extend(("--metrics-json", str(resolve_path(args.metrics_json))))
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scene_path = resolve_scene_path(args.scene_path)
    sgw_root = resolve_path(args.sgw_root)
    iteration = resolve_iteration(scene_path, args.iteration)
    output_path = (
        resolve_path(args.output)
        if args.output is not None
        else default_output_path(scene_path, iteration, args.method)
    )
    reference = None if args.reference is None else resolve_path(args.reference)
    multiview = args.method == METHOD_MULTIVIEW
    native_script = (
        "extract_multiview_gaussian_mesh.py"
        if multiview
        else "extract_mesh.py"
    )
    expected_algorithm = MULTIVIEW_ALGORITHM if multiview else ALGORITHM
    expected_schema = (
        MULTIVIEW_SCHEMA_VERSION if multiview else SCHEMA_VERSION
    )
    validate_inputs(
        scene_path,
        sgw_root,
        iteration,
        reference,
        native_script,
    )

    if output_is_complete(
        output_path,
        iteration,
        algorithm=expected_algorithm,
        schema_version=expected_schema,
    ) and not args.force:
        print(f"[skip] Complete mesh output already exists: {output_path}")
        return 0

    command = build_command(args, scene_path, iteration, output_path)
    print(f"[extract] {shlex.join(command)}")
    if args.dry_run:
        print(f"[dry-run] Final output would be: {output_path}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    completed = subprocess.run(
        command,
        cwd=sgw_root,
        env=environment,
        check=False,
    )
    if completed.returncode == 0:
        if not output_is_complete(
            output_path,
            iteration,
            algorithm=expected_algorithm,
            schema_version=expected_schema,
        ):
            print(
                "[failed] Native extractor returned success without publishing "
                f"the mesh/manifest pair: {output_path}",
                file=sys.stderr,
            )
            return 2
        print(f"[mesh] {output_path}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
