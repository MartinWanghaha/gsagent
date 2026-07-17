#!/usr/bin/env python3
"""Extract a mesh from a trained CoMe scene.

Example:
    python gsagent/scripts/come/extract_come_mesh.py gsagent/outputs/come_mushroom/classroom
    cd gsagent && python scripts/come/extract_come_mesh.py outputs/come_mushroom/classroom
    python gsagent/scripts/come/extract_come_mesh.py gsagent/outputs/come_mushroom/classroom -o classroom_mesh.ply
    python scripts/come/extract_come_mesh.py outputs/come_mushroom/classroom --target-faces 200000
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COME_ROOT = GSAGENT_ROOT / "submodules" / "CoMe"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run CoMe mesh extraction for a trained scene directory."
    )
    parser.add_argument(
        "scene_path",
        type=Path,
        help="Trained CoMe scene directory, e.g. gsagent/outputs/come_mushroom/classroom.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output mesh path. Default: <scene_path>/mesh/<method>_iteration_<iter>.ply",
    )
    parser.add_argument(
        "--come-root",
        type=Path,
        default=DEFAULT_COME_ROOT,
        help=f"CoMe checkout root. Default: {DEFAULT_COME_ROOT}",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to extract. Default: latest point_cloud/iteration_*.",
    )
    parser.add_argument(
        "--method",
        choices=("tets", "tsdf"),
        default="tets",
        help="Mesh extraction backend. Default: tets.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES value, for example 0. Omit to keep current environment.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the requested output mesh if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the CoMe command without running it.",
    )
    parser.add_argument(
        "--texture-mesh",
        action="store_true",
        help="For --method tets, request vertex colors from CoMe.",
    )
    parser.add_argument(
        "--mesh-name",
        default="mesh_faster_binary_search",
        help="For --method tets, base mesh name used by CoMe.",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.02,
        help="For --method tets, near plane for culling.",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=1e6,
        help="For --method tets, far plane for culling.",
    )
    parser.add_argument(
        "--bounding-mode",
        choices=("SIGMA_3", "SIGMA_333", "STP"),
        default="STP",
        help="For --method tets, CoMe bounding mode.",
    )
    parser.add_argument(
        "--disable-near-far-culling",
        action="store_true",
        help="For --method tets, pass --disable_near_far_culling.",
    )
    parser.add_argument(
        "--opacity-cutoff-tetra",
        type=float,
        default=0.0039,
        help="For --method tets, opacity cutoff.",
    )
    parser.add_argument(
        "--load-cells",
        action="store_true",
        help="For --method tets, reuse CoMe tetra cells if present.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.002,
        help="For --method tsdf, Open3D TSDF voxel size.",
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
                + ". If these are CoMe arguments, put them after a standalone '--'. "
                + "If you meant to run two commands, separate them with a newline or '&&'."
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
        if candidate.is_dir() and (candidate / "point_cloud.ply").is_file():
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


def validate_scene(scene_path: Path, iteration: int) -> None:
    required = [
        scene_path / "cfg_args",
        scene_path / "config.json",
        scene_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Scene is missing required file(s): "
            + ", ".join(str(path) for path in missing)
        )


def validate_come_root(come_root: Path, method: str) -> None:
    script_name = "extract_mesh_tets.py" if method == "tets" else "extract_mesh_tsdf.py"
    required = [come_root / script_name]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "CoMe root is missing required file(s): "
            + ", ".join(str(path) for path in missing)
        )


def default_output_path(scene_path: Path, method: str, iteration: int) -> Path:
    return scene_path / "mesh" / f"{method}_iteration_{iteration}.ply"


def come_mesh_path(scene_path: Path, method: str, iteration: int, mesh_name: str) -> Path:
    render_path = scene_path / "test" / f"ours_{iteration}"
    if method == "tets":
        return render_path / f"{mesh_name}_7.ply"
    return render_path / "tsdf.ply"


def build_command(
    args: argparse.Namespace,
    scene_path: Path,
    iteration: int,
    extra_args: list[str],
) -> list[str]:
    if args.method == "tets":
        cmd = [
            sys.executable,
            "extract_mesh_tets.py",
            "-m",
            str(scene_path),
            "--iteration",
            str(iteration),
            "--near",
            str(args.near),
            "--far",
            str(args.far),
            "--bounding_mode",
            args.bounding_mode,
            "--mesh_name",
            args.mesh_name,
            "--opacity_cutoff_tetra",
            str(args.opacity_cutoff_tetra),
        ]
        if args.texture_mesh:
            cmd.append("--texture_mesh")
        if args.disable_near_far_culling:
            cmd.append("--disable_near_far_culling")
        if args.load_cells:
            cmd.append("--load_cells")
    else:
        cmd = [
            sys.executable,
            "extract_mesh_tsdf.py",
            "-m",
            str(scene_path),
            "--iteration",
            str(iteration),
            "--voxel_size",
            str(args.voxel_size),
        ]

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
            "Mesh simplification requires open3d. Install it in the active "
            "environment, or run without --target-faces/--simplify-ratio."
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
    should_simplify = target_faces is not None or simplify_ratio is not None
    if should_simplify:
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
    come_root = resolve_path(args.come_root)
    iteration = args.iteration if args.iteration >= 0 else latest_iteration(scene_path)
    output_path = resolve_path(args.output) if args.output else default_output_path(scene_path, args.method, iteration)
    generated_path = come_mesh_path(scene_path, args.method, iteration, args.mesh_name)

    validate_scene(scene_path, iteration)
    validate_come_root(come_root, args.method)

    if output_path.exists() and not args.force:
        print(f"[skip] Output already exists: {output_path}")
        return 0

    cmd = build_command(args, scene_path, iteration, extra_args)
    print("[extract]", " ".join(cmd))
    print(f"[expect] {generated_path}")

    if args.dry_run:
        print(f"[dry-run] Final output would be: {output_path}")
        if args.target_faces is not None:
            print(f"[dry-run] Would simplify to approximately {args.target_faces:,} triangles")
        if args.simplify_ratio is not None:
            print(f"[dry-run] Would simplify to {args.simplify_ratio:.4g} of original triangles")
        return 0

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    completed = subprocess.run(cmd, cwd=come_root, env=env, check=False)
    if completed.returncode != 0:
        return completed.returncode

    if not generated_path.exists():
        raise FileNotFoundError(f"CoMe finished but did not create mesh: {generated_path}")

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
