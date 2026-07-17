#!/usr/bin/env python3
"""Extract a mesh from a trained Gaussian Wrapping scene.

Examples:
    python scripts/gaussian_wrapping/extract_gaussian_wrapping_mesh.py outputs/gaussian_wrapping_mushroom/classroom --gpu 0
    python scripts/gaussian_wrapping/extract_gaussian_wrapping_mesh.py outputs/gaussian_wrapping_mushroom/classroom --rasterizer radegs
    python scripts/gaussian_wrapping/extract_gaussian_wrapping_mesh.py outputs/gaussian_wrapping_mushroom/classroom --mtet-on-cpu
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GW_ROOT = GSAGENT_ROOT / "submodules" / "GaussianWrapping"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Extract a mesh from a trained Gaussian Wrapping scene."
    )
    parser.add_argument(
        "scene_path",
        type=Path,
        help="Trained scene directory, e.g. outputs/gaussian_wrapping_mushroom/classroom.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path. Default: <scene>/mesh/<rasterizer>_iteration_<iter>.ply",
    )
    parser.add_argument(
        "--gaussian-wrapping-root",
        "--gw-root",
        dest="gw_root",
        type=Path,
        default=DEFAULT_GW_ROOT,
        help=f"GaussianWrapping checkout root. Default: {DEFAULT_GW_ROOT}",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help="Override the COLMAP source path stored in <scene>/cfg_args.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to extract. Default: latest point_cloud/iteration_*.",
    )
    parser.add_argument(
        "--rasterizer",
        choices=("ours", "radegs"),
        default="ours",
        help="Use the matching official extraction preset. Default: ours.",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=None,
        help="Override the image resolution/downsample value from cfg_args.",
    )
    parser.add_argument(
        "--data-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used to store source images. Default: cpu.",
    )
    parser.add_argument(
        "--isosurface-value",
        type=float,
        default=0.0,
        help="Isosurface offset. Try 0.2 when fine details are missing. Default: 0.0.",
    )
    parser.add_argument(
        "--n-pivots",
        type=int,
        default=2,
        help="Pivots sampled per Gaussian. Default: 2.",
    )
    parser.add_argument(
        "--n-binary-steps",
        type=int,
        default=10,
        help="Binary-search refinement steps. Default: 10.",
    )
    parser.add_argument(
        "--dtype",
        choices=("int32", "int64"),
        default="int32",
        help="Delaunay index dtype. Default: int32.",
    )
    parser.add_argument(
        "--sdf-batch-size",
        type=int,
        default=None,
        help="Maximum points per SDF evaluation; lower this to reduce peak VRAM.",
    )
    parser.add_argument(
        "--mtet-on-cpu",
        action="store_true",
        help="Run marching tetrahedra on CPU to reduce peak VRAM.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Keep disconnected components instead of running official postprocessing.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Value for CUDA_VISIBLE_DEVICES, for example 0.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the requested output mesh if it exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the extraction command without running it.",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=None,
        help="Simplify the extracted mesh to approximately this many triangles.",
    )
    parser.add_argument(
        "--simplify-ratio",
        type=float,
        default=None,
        help="Simplify to this fraction of the original triangles, e.g. 0.25.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean duplicated, degenerate, or unreferenced geometry.",
    )

    argv = sys.argv[1:]
    args, extra_args = parser.parse_known_args(argv)
    if extra_args:
        if "--" not in argv:
            parser.error(
                "unexpected extra arguments: "
                + " ".join(extra_args)
                + ". Put native Gaussian Wrapping arguments after a standalone '--'."
            )
        if extra_args[0] == "--":
            extra_args = extra_args[1:]

    if args.iteration == 0 or args.iteration < -1:
        parser.error("--iteration must be positive or -1 for latest.")
    if args.n_pivots <= 0:
        parser.error("--n-pivots must be positive.")
    if args.n_binary_steps < 0:
        parser.error("--n-binary-steps cannot be negative.")
    if args.sdf_batch_size is not None and args.sdf_batch_size <= 0:
        parser.error("--sdf-batch-size must be positive.")
    if args.target_faces is not None and args.target_faces <= 0:
        parser.error("--target-faces must be positive.")
    if args.simplify_ratio is not None and not (0.0 < args.simplify_ratio <= 1.0):
        parser.error("--simplify-ratio must be in the range (0, 1].")
    if args.target_faces is not None and args.simplify_ratio is not None:
        parser.error("Use only one of --target-faces or --simplify-ratio.")

    output_affecting_args = {
        "--sdf_mode",
        "--rasterizer",
        "--n_pivots",
        "--isosurface_value",
        "--use_scores",
        "--random_pivots",
        "--use_searched_pivots",
        "--use_tetra_points",
        "--postprocess",
    }
    native_option_names = {arg.split("=", 1)[0] for arg in extra_args}
    unsupported = sorted(output_affecting_args.intersection(native_option_names))
    if unsupported:
        parser.error(
            "output-affecting native arguments must use wrapper options: "
            + ", ".join(unsupported)
        )
    return args, extra_args


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
        if candidate.is_dir() and (candidate / "point_cloud").is_dir():
            return candidate
    return candidates[0]


def latest_iteration(scene_path: Path) -> int:
    point_cloud_root = scene_path / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"Missing point_cloud directory: {point_cloud_root}")

    iterations = []
    for child in point_cloud_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            iterations.append(int(child.name.removeprefix("iteration_")))
        except ValueError:
            continue
    if not iterations:
        raise FileNotFoundError(f"No point_cloud/iteration_* directories in {scene_path}")
    return max(iterations)


def validate_inputs(scene_path: Path, workdir: Path, iteration: int) -> None:
    required = [
        workdir / "pivot_based_mesh_extraction.py",
        scene_path / "cfg_args",
        scene_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(str(path) for path in missing)
        )


def generated_mesh_path(
    scene_path: Path,
    rasterizer: str,
    n_pivots: int,
    isosurface_value: float,
    postprocess: bool,
) -> Path:
    if rasterizer == "ours":
        sdf_mode = "ours"
        iso_suffix = (
            f"_iso_{isosurface_value}" if isosurface_value != 0.0 else ""
        )
        pivot_suffix = ""
    else:
        sdf_mode = "exact_computation"
        threshold = 0.5 + isosurface_value
        iso_suffix = (
            f"_transmittance_threshold_{threshold}" if threshold != 0.5 else ""
        )
        pivot_suffix = "_searched"

    suffix = "_post" if postprocess else ""
    filename = (
        f"mesh_{sdf_mode}_{n_pivots}pivots{iso_suffix}{pivot_suffix}{suffix}.ply"
    )
    return scene_path / filename


def default_output_path(scene_path: Path, rasterizer: str, iteration: int) -> Path:
    return scene_path / "mesh" / f"{rasterizer}_iteration_{iteration}.ply"


def build_command(
    args: argparse.Namespace,
    scene_path: Path,
    iteration: int,
    extra_args: list[str],
) -> list[str]:
    sdf_mode = "ours" if args.rasterizer == "ours" else "exact_computation"
    cmd = [
        sys.executable,
        "pivot_based_mesh_extraction.py",
        "-m",
        str(scene_path),
        "--iteration",
        str(iteration),
        "--rasterizer",
        args.rasterizer,
        "--sdf_mode",
        sdf_mode,
        "--dtype",
        args.dtype,
        "--n_pivots",
        str(args.n_pivots),
        "--isosurface_value",
        str(args.isosurface_value),
        "--n_binary_steps",
        str(args.n_binary_steps),
        "--data_device",
        args.data_device,
        "--use_valid_mask",
    ]
    if args.source_path is not None:
        cmd.extend(["-s", str(resolve_path(args.source_path))])
    if args.resolution is not None:
        cmd.extend(["-r", str(args.resolution)])
    if args.sdf_batch_size is not None:
        cmd.extend(
            ["--n_points_per_sdf_evaluation", str(args.sdf_batch_size)]
        )
    if args.mtet_on_cpu:
        cmd.append("--mtet_on_cpu")
    if not args.no_postprocess:
        cmd.append("--postprocess")

    if args.rasterizer == "ours":
        cmd.append("--filter_large_edges")
    else:
        cmd.extend(
            [
                "--std_factor",
                "3.33",
                "--use_searched_pivots",
                "--search_iter",
                "5",
                "--search_step_size",
                "0.33",
            ]
        )
    cmd.extend(extra_args)
    return cmd


def simplify_mesh(
    input_path: Path,
    output_path: Path,
    *,
    target_faces: int | None,
    simplify_ratio: float | None,
    clean: bool,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Mesh simplification requires open3d.") from exc

    mesh = o3d.io.read_triangle_mesh(str(input_path))
    original_faces = len(mesh.triangles)
    if original_faces <= 0:
        raise ValueError(f"No triangles found in mesh: {input_path}")
    if simplify_ratio is not None:
        target_faces = max(1, round(original_faces * simplify_ratio))
    assert target_faces is not None
    target_faces = min(target_faces, original_faces)

    print(f"[simplify] {original_faces:,} -> {target_faces:,} triangles")
    if target_faces < original_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)
    if clean:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.remove_non_manifold_edges()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_path = output_path
    replace_after_write = output_path.resolve() == input_path.resolve()
    if replace_after_write:
        write_path = output_path.with_name(
            output_path.stem + ".tmp_simplified" + output_path.suffix
        )
    if not o3d.io.write_triangle_mesh(str(write_path), mesh):
        raise RuntimeError(f"Failed to write simplified mesh: {write_path}")
    if replace_after_write:
        shutil.move(str(write_path), str(output_path))
    print(f"[simplify] wrote {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")


def write_output_mesh(
    generated_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    if args.target_faces is not None or args.simplify_ratio is not None:
        simplify_mesh(
            generated_path,
            output_path,
            target_faces=args.target_faces,
            simplify_ratio=args.simplify_ratio,
            clean=not args.no_clean,
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() != generated_path.resolve():
        shutil.copy2(generated_path, output_path)


def main() -> int:
    args, extra_args = parse_args()
    scene_path = resolve_scene_path(args.scene_path)
    workdir = resolve_path(args.gw_root) / "gaussian_wrapping"
    iteration = args.iteration if args.iteration > 0 else latest_iteration(scene_path)
    output_path = (
        resolve_path(args.output)
        if args.output is not None
        else default_output_path(scene_path, args.rasterizer, iteration)
    )
    generated_path = generated_mesh_path(
        scene_path,
        args.rasterizer,
        args.n_pivots,
        args.isosurface_value,
        not args.no_postprocess,
    )
    validate_inputs(scene_path, workdir, iteration)

    if output_path.exists() and not args.force:
        print(f"[skip] Output already exists: {output_path}")
        return 0

    cmd = build_command(args, scene_path, iteration, extra_args)
    print(f"[extract] {shlex.join(cmd)}")
    print(f"[expect] {generated_path}")
    if args.dry_run:
        print(f"[dry-run] Final output would be: {output_path}")
        return 0

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    completed = subprocess.run(cmd, cwd=workdir, env=env, check=False)
    if completed.returncode != 0:
        return completed.returncode
    if not generated_path.exists():
        raise FileNotFoundError(
            f"Gaussian Wrapping finished but did not create: {generated_path}"
        )

    write_output_mesh(generated_path, output_path, args)
    print(f"[mesh] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
