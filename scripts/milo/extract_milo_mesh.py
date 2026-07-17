#!/usr/bin/env python3
"""Extract a mesh from a trained MILo scene.

Example:
    python gsagent/scripts/milo/extract_milo_mesh.py gsagent/outputs/milo_mushroom/classroom
    python gsagent/scripts/milo/extract_milo_mesh.py outputs/milo_mushroom/classroom --method integration
    python gsagent/scripts/milo/extract_milo_mesh.py outputs/milo_mushroom/classroom -o classroom.ply
    python gsagent/scripts/milo/extract_milo_mesh.py outputs/milo_mushroom/classroom --target-faces 200000
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MILO_ROOT = GSAGENT_ROOT / "submodules" / "MILo"

# Map method name -> (script, output filename template)
_METHOD_SCRIPT = {
    "sdf": "mesh_extract_sdf.py",
    "integration": "mesh_extract_integration.py",
    "tsdf": "mesh_extract_regular_tsdf.py",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run MILo mesh extraction for a trained scene directory."
    )
    parser.add_argument(
        "scene_path",
        type=Path,
        help="Trained MILo scene directory, e.g. gsagent/outputs/milo_mushroom/classroom.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output mesh path. Default: <scene_path>/mesh/<method>_iteration_<iter>.ply",
    )
    parser.add_argument(
        "--milo-root",
        type=Path,
        default=DEFAULT_MILO_ROOT,
        help=f"MILo checkout root. Default: {DEFAULT_MILO_ROOT}",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to extract. Default: latest point_cloud/iteration_*.",
    )
    parser.add_argument(
        "--method",
        choices=("sdf", "integration", "tsdf"),
        default="sdf",
        help="Mesh extraction method. Default: sdf (mesh_extract_sdf.py).",
    )
    parser.add_argument(
        "--rasterizer",
        choices=("radegs", "gof"),
        default="radegs",
        help="Rasterizer for sdf/integration methods. Default: radegs.",
    )
    # sdf-specific
    parser.add_argument(
        "--config",
        default="default",
        help="[sdf] Mesh config name (default/highres/veryhighres/lowres/verylowres).",
    )
    parser.add_argument(
        "--refine-iter",
        type=int,
        default=1000,
        help="[sdf] Number of SDF refinement iterations. Default: 1000.",
    )
    # integration-specific
    parser.add_argument(
        "--sdf-mode",
        choices=("integration", "depth_fusion"),
        default="integration",
        help="[integration] SDF computation mode. Default: integration.",
    )
    parser.add_argument(
        "--isosurface-value",
        type=float,
        default=-1.0,
        help="[integration] Isosurface value (default -1 = auto).",
    )
    parser.add_argument(
        "--trunc-margin",
        type=float,
        default=-1.0,
        help="[integration/depth_fusion] Truncation margin (default -1 = auto).",
    )
    # tsdf-specific
    parser.add_argument(
        "--mesh-res",
        type=int,
        default=1024,
        help="[tsdf] Voxel grid resolution. Default: 1024.",
    )
    # shared
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES value, for example 0.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output mesh if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running it.",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=None,
        help="Post-process output mesh to approximately this many triangles.",
    )
    parser.add_argument(
        "--simplify-ratio",
        type=float,
        default=None,
        help="Post-process output mesh to this fraction of its original triangles, e.g. 0.25.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove duplicated/degenerate/unreferenced geometry after simplification.",
    )

    argv = sys.argv[1:]
    args, extra_args = parser.parse_known_args(argv)
    if extra_args:
        if "--" not in argv:
            parser.error(
                "unexpected extra arguments: "
                + " ".join(extra_args)
                + ". If these are MILo arguments, put them after a standalone '--'."
            )
        if extra_args[0] == "--":
            extra_args = extra_args[1:]
    if args.target_faces is not None and args.target_faces <= 0:
        parser.error("--target-faces must be positive.")
    if args.simplify_ratio is not None and not (0.0 < args.simplify_ratio <= 1.0):
        parser.error("--simplify-ratio must be in the range (0, 1].")
    if args.target_faces is not None and args.simplify_ratio is not None:
        parser.error("Use only one of --target-faces or --simplify-ratio.")
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

    parts = expanded.parts
    if parts and parts[0] == GSAGENT_ROOT.name:
        candidates.insert(0, (GSAGENT_ROOT.parent / expanded).resolve())

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "point_cloud").is_dir():
            return candidate

    return candidates[0]


def latest_iteration(scene_path: Path) -> int:
    point_cloud_root = scene_path / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"Missing point_cloud directory: {point_cloud_root}")

    iterations: list[int] = []
    for child in point_cloud_root.iterdir():
        if not child.is_dir() or not child.name.startswith("iteration_"):
            continue
        try:
            iterations.append(int(child.name.split("_")[-1]))
        except ValueError:
            continue
    if not iterations:
        raise FileNotFoundError(f"No point_cloud/iteration_* directories in {scene_path}")
    return max(iterations)


def milo_mesh_path(scene_path: Path, method: str, sdf_mode: str, mesh_res: int) -> Path:
    if method == "sdf":
        return scene_path / "mesh_learnable_sdf.ply"
    if method == "integration":
        return scene_path / f"mesh_{sdf_mode}_sdf.ply"
    # tsdf
    return scene_path / f"mesh_regular_tsdf_res{mesh_res}_post.ply"


def default_output_path(scene_path: Path, method: str, iteration: int) -> Path:
    return scene_path / "mesh" / f"{method}_iteration_{iteration}.ply"


def build_command(
    args: argparse.Namespace,
    scene_path: Path,
    iteration: int,
    extra_args: list[str],
) -> list[str]:
    script = _METHOD_SCRIPT[args.method]
    cmd = [sys.executable, script, "-m", str(scene_path), "--iteration", str(iteration)]

    if args.method == "sdf":
        cmd += ["--rasterizer", args.rasterizer, "--config", args.config, "--refine_iter", str(args.refine_iter)]
    elif args.method == "integration":
        cmd += [
            "--rasterizer", args.rasterizer,
            "--sdf_mode", args.sdf_mode,
            "--isosurface_value", str(args.isosurface_value),
            "--trunc_margin", str(args.trunc_margin),
        ]
    else:  # tsdf
        cmd += ["--mesh_res", str(args.mesh_res)]

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
        raise RuntimeError(
            "Mesh simplification requires open3d. Install it or run without --target-faces/--simplify-ratio."
        ) from exc

    mesh = o3d.io.read_triangle_mesh(str(input_path))
    original_faces = len(mesh.triangles)
    if original_faces <= 0:
        raise ValueError(f"No triangles found in mesh: {input_path}")

    if simplify_ratio is not None:
        target_faces = max(1, int(round(original_faces * simplify_ratio)))
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
    replace_after_write = False
    if output_path.resolve() == input_path.resolve():
        write_path = output_path.with_name(output_path.stem + ".tmp_simplified" + output_path.suffix)
        replace_after_write = True

    ok = o3d.io.write_triangle_mesh(str(write_path), mesh)
    if not ok:
        raise RuntimeError(f"Failed to write simplified mesh: {write_path}")
    if replace_after_write:
        shutil.move(str(write_path), str(output_path))

    print(f"[simplify] wrote {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")


def write_final_mesh(
    generated_path: Path,
    output_path: Path,
    *,
    target_faces: int | None,
    simplify_ratio: float | None,
    clean: bool,
) -> None:
    if target_faces is not None or simplify_ratio is not None:
        simplify_mesh(
            generated_path,
            output_path,
            target_faces=target_faces,
            simplify_ratio=simplify_ratio,
            clean=clean,
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() != generated_path.resolve():
        shutil.copy2(generated_path, output_path)


def main() -> int:
    args, extra_args = parse_args()
    scene_path = resolve_scene_path(args.scene_path)
    milo_root = resolve_path(args.milo_root)
    milo_cwd = milo_root / "milo"
    iteration = args.iteration if args.iteration >= 0 else latest_iteration(scene_path)
    output_path = resolve_path(args.output) if args.output else default_output_path(scene_path, args.method, iteration)
    generated_path = milo_mesh_path(scene_path, args.method, args.sdf_mode, args.mesh_res)

    if not milo_cwd.is_dir():
        raise FileNotFoundError(f"MILo milo/ directory not found: {milo_cwd}")

    if output_path.exists() and not args.force:
        print(f"[skip] Output already exists: {output_path}")
        return 0

    cmd = build_command(args, scene_path, iteration, extra_args)
    print("[extract]", " ".join(cmd))
    print(f"[expect] {generated_path}")

    if args.dry_run:
        print(f"[dry-run] Final output would be: {output_path}")
        return 0

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    completed = subprocess.run(cmd, cwd=milo_cwd, env=env, check=False)
    if completed.returncode != 0:
        return completed.returncode

    if not generated_path.exists():
        raise FileNotFoundError(f"MILo finished but did not create mesh: {generated_path}")

    write_final_mesh(
        generated_path,
        output_path,
        target_faces=args.target_faces,
        simplify_ratio=args.simplify_ratio,
        clean=not args.no_clean,
    )
    print(f"[mesh] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
