#!/usr/bin/env python3
"""Run reproducible SemanticGaussianWrapping image/mesh ablations.

Every variant owns an isolated ``<output-root>/<variant>/<scene>`` directory.
Fresh runs select an explicit config; resumed runs never pass a config or
fresh-data option, so the checkpoint remains authoritative.
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
from typing import Any, Iterable


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
SGW_ROOT = GSAGENT_ROOT / "submodules" / "SemanticGaussianWrapping"
TRAIN_LAUNCHER = (
    GSAGENT_ROOT
    / "scripts"
    / "semantic_gaussian_wrapping"
    / "train_semantic_gaussian_wrapping_mipnerf360.py"
)
MESH_LAUNCHER = (
    GSAGENT_ROOT
    / "scripts"
    / "semantic_gaussian_wrapping"
    / "extract_semantic_gaussian_wrapping_mesh.py"
)
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "mip-nerf" / "360_v2"
DEFAULT_OUTPUT_ROOT = (
    GSAGENT_ROOT / "outputs" / "semantic_gaussian_wrapping_mipnerf360_ablations"
)
VARIANT_CONFIGS = {
    "rgb_only": "rgb_only.yaml",
    "semantic_render_only": "semantic_render_only.yaml",
    "full": "full.yaml",
    "full_no_mesh_feedback": "full_no_mesh_feedback.yaml",
    "full_no_surface_topology": "full_no_surface_topology.yaml",
    "full_no_confidence_propagation": "full_no_confidence_propagation.yaml",
    "full_no_expert_certainty": "full_no_expert_certainty.yaml",
    "full_no_prune_replace": "full_no_prune_replace.yaml",
}
VALID_STEPS = ("train", "render", "image", "mesh")
CHECKPOINT_PATTERN = re.compile(r"^chkpnt(\d+)\.pth$")
FRESH_ONLY_OPTIONS = {
    "--resolution",
    "-r",
    "--semantic-path",
    "--semantic-confidence",
    "--semantic-boundary",
    "--holdout",
}


def _comma_values(values: Iterable[str] | None) -> list[str]:
    return [
        item
        for value in values or ()
        for item in (part.strip() for part in value.split(","))
        if item
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        help="Repeat or comma-separate variant names. Default: the complete matrix.",
    )
    parser.add_argument(
        "--steps",
        action="append",
        default=None,
        help="Comma-separated train,render,image,mesh stages. Default: all four.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sgw-root", type=Path, default=SGW_ROOT)
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Evaluation iteration; default: latest.",
    )
    parser.add_argument("-r", "--resolution", type=int, default=-1)
    parser.add_argument("--semantic-path", default="sam_mask")
    parser.add_argument("--semantic-confidence", default="")
    parser.add_argument("--semantic-boundary", default="")
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--set", action="append", dest="overrides", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend", choices=("auto", "cuda", "reference"), default="auto"
    )
    parser.add_argument("--mesh-max-gaussians", type=int, default=500_000)
    parser.add_argument("--mesh-max-chart-gaussians", type=int, default=12_000)
    parser.add_argument("--mesh-view-stride", type=int, default=1)
    parser.add_argument("--mesh-camera-scale", type=float, default=1.0)
    parser.add_argument("--mesh-target-faces", type=int)
    parser.add_argument(
        "--mesh-reference",
        help="Reference geometry path; may contain {scene} and {variant} placeholders.",
    )
    parser.add_argument(
        "--skip-mesh-metrics",
        action="store_true",
        help="Extract meshes without reference metrics (not a dual-metric benchmark).",
    )
    parser.add_argument("--metric-threshold", type=float, default=0.01)
    parser.add_argument("--metric-samples", type=int, default=100_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    raw = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw)

    args.scenes = list(dict.fromkeys(_comma_values(args.scene)))
    if not args.scenes:
        parser.error("at least one concrete --scene is required")
    requested_variants = _comma_values(args.variant) or list(VARIANT_CONFIGS)
    unknown_variants = sorted(set(requested_variants) - set(VARIANT_CONFIGS))
    if unknown_variants:
        parser.error(
            "unknown variant(s): "
            + ", ".join(unknown_variants)
            + "; expected "
            + ", ".join(VARIANT_CONFIGS)
        )
    args.variants = list(dict.fromkeys(requested_variants))
    requested_steps = _comma_values(args.steps) or list(VALID_STEPS)
    unknown_steps = sorted(set(requested_steps) - set(VALID_STEPS))
    if unknown_steps:
        parser.error("unknown step(s): " + ", ".join(unknown_steps))
    args.steps = list(dict.fromkeys(requested_steps))

    supplied_options = {
        value.split("=", 1)[0] for value in raw if value.startswith("-")
    }
    resume_conflicts = sorted(supplied_options.intersection(FRESH_ONLY_OPTIONS))
    if args.resume and resume_conflicts:
        parser.error(
            "fresh data options cannot be combined with --resume: "
            + ", ".join(resume_conflicts)
        )
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.iteration == 0 or args.iteration < -1:
        parser.error("--iteration must be positive or -1 for latest")
    if args.mesh_max_gaussians < 4:
        parser.error("--mesh-max-gaussians must be at least 4")
    if args.mesh_max_chart_gaussians < 4:
        parser.error("--mesh-max-chart-gaussians must be at least 4")
    if args.mesh_view_stride < 1:
        parser.error("--mesh-view-stride must be positive")
    if (
        not math.isfinite(args.mesh_camera_scale)
        or args.mesh_camera_scale <= 0
        or args.mesh_camera_scale > 1
    ):
        parser.error("--mesh-camera-scale must lie in (0,1]")
    if args.mesh_target_faces is not None and args.mesh_target_faces < 1:
        parser.error("--mesh-target-faces must be positive")
    if (
        not math.isfinite(args.metric_threshold)
        or args.metric_threshold <= 0
        or args.metric_samples < 1
    ):
        parser.error("mesh metric threshold and sample count must be positive")
    if "mesh" in args.steps and not args.mesh_reference and not args.skip_mesh_metrics:
        parser.error(
            "dual-metric mesh evaluation requires --mesh-reference; pass "
            "--skip-mesh-metrics only for extraction-only diagnostics"
        )
    return args


def _config_path(args: argparse.Namespace, variant: str) -> Path:
    return args.sgw_root.expanduser().resolve() / "configs" / VARIANT_CONFIGS[variant]


def model_path(args: argparse.Namespace, variant: str, scene: str) -> Path:
    return args.output_root.expanduser().resolve() / variant / scene


def build_train_command(
    args: argparse.Namespace,
    variant: str,
    scene: str,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_LAUNCHER),
        "--scene",
        scene,
        "--data-root",
        str(args.data_root.expanduser().resolve()),
        "--output-root",
        str(args.output_root.expanduser().resolve() / variant),
        "--sgw-root",
        str(args.sgw_root.expanduser().resolve()),
        "--device",
        args.device,
    ]
    if args.gpu is not None:
        command.extend(["--gpu", args.gpu])
    if args.resume:
        command.append("--resume")
        if args.iterations is not None:
            command.extend(["--iterations", str(args.iterations)])
    else:
        command.extend(["--config", str(_config_path(args, variant))])
        command.extend(["--iterations", str(args.iterations or 30_000)])
        command.extend(["--resolution", str(args.resolution)])
        command.extend(["--semantic-path", args.semantic_path])
        if args.semantic_confidence:
            command.extend(["--semantic-confidence", args.semantic_confidence])
        if args.semantic_boundary:
            command.extend(["--semantic-boundary", args.semantic_boundary])
        command.extend(["--eval", "--holdout", str(args.holdout)])
    for expression in args.overrides:
        command.extend(["--set", expression])
    return command


def latest_checkpoint_iteration(root: Path) -> int | None:
    iterations = []
    for path in root.glob("chkpnt*.pth"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            iterations.append(int(match.group(1)))
    return max(iterations) if iterations else None


def evaluation_iteration(args: argparse.Namespace, root: Path) -> int:
    if args.iteration >= 0:
        return int(args.iteration)
    latest = latest_checkpoint_iteration(root)
    if latest is not None:
        return latest
    return int(args.iterations or 30_000) if not args.resume else -1


def build_render_command(
    args: argparse.Namespace, root: Path, iteration: int
) -> list[str]:
    command = [
        sys.executable,
        "render.py",
        "-m",
        str(root),
        "--skip_train",
        "--device",
        args.device,
        "--backend",
        args.backend,
    ]
    if iteration >= 0:
        command.extend(["--iteration", str(iteration)])
    return command


def build_image_command(
    args: argparse.Namespace, root: Path, iteration: int
) -> list[str]:
    command = [
        sys.executable,
        "metrics.py",
        "-m",
        str(root),
        "--split",
        "test",
        "--device",
        args.device,
    ]
    if iteration >= 0:
        command.extend(["--iteration", str(iteration)])
    return command


def _reference_path(args: argparse.Namespace, scene: str, variant: str) -> Path | None:
    if not args.mesh_reference:
        return None
    value = args.mesh_reference.format(scene=scene, variant=variant)
    return Path(value).expanduser().resolve()


def build_mesh_command(
    args: argparse.Namespace,
    root: Path,
    iteration: int,
    scene: str,
    variant: str,
) -> tuple[list[str], Path, Path | None]:
    token = str(iteration) if iteration >= 0 else "latest"
    mesh_root = root / "mesh" / f"iteration_{token}"
    mesh_output = mesh_root / "semantic_surface.ply"
    metrics_output = None
    command = [
        sys.executable,
        str(MESH_LAUNCHER),
        str(root),
        "--sgw-root",
        str(args.sgw_root.expanduser().resolve()),
        "--iteration",
        str(iteration),
        "--output",
        str(mesh_output),
        "--device",
        args.device,
        "--max-gaussians",
        str(args.mesh_max_gaussians),
        "--max-chart-gaussians",
        str(args.mesh_max_chart_gaussians),
        "--view-stride",
        str(args.mesh_view_stride),
        "--camera-scale",
        str(args.mesh_camera_scale),
        "--force",
    ]
    if args.mesh_target_faces is not None:
        command.extend(["--target-faces", str(args.mesh_target_faces)])
    if args.gpu is not None:
        command.extend(["--gpu", args.gpu])
    reference = _reference_path(args, scene, variant)
    if reference is not None and not args.skip_mesh_metrics:
        metrics_output = mesh_root / "metrics.json"
        command.extend(
            [
                "--reference",
                str(reference),
                "--metric-threshold",
                str(args.metric_threshold),
                "--metric-samples",
                str(args.metric_samples),
                "--metrics-json",
                str(metrics_output),
            ]
        )
    return command, mesh_output, metrics_output


def _run(
    command: list[str], cwd: Path, environment: dict[str, str], dry_run: bool
) -> int:
    print(shlex.join(command))
    if dry_run:
        return 0
    return subprocess.run(command, cwd=cwd, env=environment, check=False).returncode


def _read_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf8"))


def _write_matrix_preserving(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for index in range(1, 10_000):
            candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
            if not candidate.exists():
                path = candidate
                break
        else:
            raise RuntimeError(f"cannot allocate matrix result beside {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workdir = args.sgw_root.expanduser().resolve()
    for variant in args.variants:
        config_path = _config_path(args, variant)
        if not args.dry_run and not config_path.is_file():
            raise FileNotFoundError(f"missing ablation config: {config_path}")
    if "mesh" in args.steps and not args.skip_mesh_metrics and not args.dry_run:
        missing_references = []
        for variant in args.variants:
            for scene in args.scenes:
                reference = _reference_path(args, scene, variant)
                if reference is None or not reference.is_file():
                    missing_references.append(f"{variant}/{scene}: {reference}")
        if missing_references:
            print(
                "[failed] mesh reference preflight failed before training:\n"
                + "\n".join(missing_references),
                file=sys.stderr,
            )
            return 2
    environment = os.environ.copy()
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    records: dict[str, Any] = {}
    failures: list[str] = []
    for variant in args.variants:
        config_path = _config_path(args, variant)
        for scene in args.scenes:
            key = f"{variant}/{scene}"
            root = model_path(args, variant, scene)
            record: dict[str, Any] = {
                "variant": variant,
                "scene": scene,
                "config": str(config_path),
                "model_path": str(root),
                "status": "ok",
            }
            records[key] = record
            if "train" in args.steps:
                return_code = _run(
                    build_train_command(args, variant, scene),
                    GSAGENT_ROOT,
                    environment,
                    args.dry_run,
                )
                if return_code:
                    record["status"] = "failed"
                    record["failed_stage"] = "train"
                    record["return_code"] = return_code
                    failures.append(key)
            # Training is a separate process and may have advanced a resumed
            # run. Resolve its newly published checkpoint only after it exits.
            iteration = (
                -1
                if args.dry_run and args.resume and "train" in args.steps
                else evaluation_iteration(args, root)
            )
            commands: list[tuple[str, list[str], Path]] = []
            if "render" in args.steps:
                commands.append(
                    ("render", build_render_command(args, root, iteration), workdir)
                )
            if "image" in args.steps:
                commands.append(
                    ("image", build_image_command(args, root, iteration), workdir)
                )
            mesh_metrics_path = None
            mesh_output = None
            if "mesh" in args.steps:
                mesh_command, mesh_output, mesh_metrics_path = build_mesh_command(
                    args,
                    root,
                    iteration,
                    scene,
                    variant,
                )
                commands.append(("mesh", mesh_command, GSAGENT_ROOT))
            for stage, command, cwd in commands:
                if record["status"] != "ok":
                    break
                return_code = _run(command, cwd, environment, args.dry_run)
                if return_code:
                    record["status"] = "failed"
                    record["failed_stage"] = stage
                    record["return_code"] = return_code
                    failures.append(key)
                    break
            if not args.dry_run and record["status"] == "ok":
                resolved_iteration = evaluation_iteration(args, root)
                record["iteration"] = resolved_iteration
                if resolved_iteration >= 0:
                    record["image_metrics"] = _read_json(
                        root / "results" / "test" / f"ours_{resolved_iteration}.json"
                    )
                record["mesh"] = None if mesh_output is None else str(mesh_output)
                record["mesh_metrics"] = _read_json(mesh_metrics_path)
            if record["status"] != "ok" and not args.keep_going:
                break
        if failures and not args.keep_going:
            break
    if args.dry_run:
        return 0
    matrix = {
        "variants": args.variants,
        "scenes": args.scenes,
        "steps": args.steps,
        "records": records,
    }
    matrix_path = _write_matrix_preserving(
        args.output_root.expanduser().resolve() / "ablation_matrix.json",
        matrix,
    )
    print(f"[matrix] {matrix_path}")
    if failures:
        print("[failed] " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
