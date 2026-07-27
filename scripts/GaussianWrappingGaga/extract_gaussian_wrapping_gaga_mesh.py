#!/usr/bin/env python3
"""Extract Gaussian Wrapping geometry and Gaga semantic meshes.

Examples:
    python scripts/GaussianWrappingGaga/extract_gaussian_wrapping_gaga_mesh.py \
        --scene counter --mode two-stage --mask-method entityseg --gpu 0

    python scripts/GaussianWrappingGaga/extract_gaussian_wrapping_gaga_mesh.py \
        --scene counter --mode joint --texture --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

try:
    from ._common import (
        DEFAULT_DATA_ROOT,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_PROJECT_ROOT,
        discover_scenes,
        execute,
        model_directories,
        point_cloud_path,
        resolve_iteration,
        resolve_path,
        resolve_semantic_iteration,
        run_directory,
        select_scenes,
        semantic_checkpoint_path,
        validate_project,
        write_manifest,
    )
except ImportError:
    from _common import (  # type: ignore[no-redef]
        DEFAULT_DATA_ROOT,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_PROJECT_ROOT,
        discover_scenes,
        execute,
        model_directories,
        point_cloud_path,
        resolve_iteration,
        resolve_path,
        resolve_semantic_iteration,
        run_directory,
        select_scenes,
        semantic_checkpoint_path,
        validate_project,
        write_manifest,
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Gaussian Wrapping geometry, optionally refine texture, "
            "and transfer Gaga labels to the resulting mesh."
        )
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help=(
            "Scene name; repeat or use comma-separated values. With no value, "
            "process all matching trained runs. Use 'all' for every data scene."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("two-stage", "joint"),
        default="two-stage",
    )
    parser.add_argument(
        "--mask-method",
        "--seg-method",
        dest="mask_method",
        choices=("sam", "entityseg"),
        default="entityseg",
    )
    parser.add_argument("--images", default="images")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--project-root",
        "--gaussian-wrapping-gaga-root",
        dest="project_root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--geometry-iteration", type=int, default=-1)
    parser.add_argument("--semantic-iteration", type=int, default=-1)
    parser.add_argument("--rasterizer", choices=("ours", "radegs"))
    parser.add_argument("-r", "--resolution", type=int)
    parser.add_argument(
        "--data-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--isosurface-value", type=float, default=0.0)
    parser.add_argument("--n-pivots", type=int, default=2)
    parser.add_argument("--n-binary-steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument("--sdf-batch-size", type=int)
    parser.add_argument("--mtet-on-cpu", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--texture", action="store_true")
    parser.add_argument("--texture-iterations", type=int, default=1_000)
    parser.add_argument("--no-semantic-mesh", action="store_true")
    parser.add_argument("--semantic-workers", type=int, default=-1)
    parser.add_argument("--target-faces", type=int)
    parser.add_argument("--simplify-ratio", type=float)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value, e.g. 0.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-scenes", action="store_true")

    argv = sys.argv[1:]
    args, native_args = parser.parse_known_args(argv)
    if native_args:
        if "--" not in argv:
            parser.error(
                "Unexpected native extraction arguments: "
                + " ".join(native_args)
                + ". Put them after a standalone '--'."
            )
        if native_args[0] == "--":
            native_args = native_args[1:]
    for name in ("geometry_iteration", "semantic_iteration"):
        value = getattr(args, name)
        if value == 0 or value < -1:
            parser.error(f"--{name.replace('_', '-')} must be -1 or positive.")
    if args.resolution is not None and (
        args.resolution == 0 or args.resolution < -1
    ):
        parser.error("--resolution must be -1 or positive.")
    if args.n_pivots <= 0 or args.n_binary_steps < 0:
        parser.error("Pivot count must be positive and binary steps non-negative.")
    if args.sdf_batch_size is not None and args.sdf_batch_size <= 0:
        parser.error("--sdf-batch-size must be positive.")
    if args.texture_iterations <= 0:
        parser.error("--texture-iterations must be positive.")
    if args.target_faces is not None and args.target_faces <= 0:
        parser.error("--target-faces must be positive.")
    if args.simplify_ratio is not None and not 0 < args.simplify_ratio <= 1:
        parser.error("--simplify-ratio must be in (0, 1].")
    if args.target_faces is not None and args.simplify_ratio is not None:
        parser.error("Use only one of --target-faces and --simplify-ratio.")

    controlled = {
        "--sdf_mode",
        "--rasterizer",
        "--n_pivots",
        "--isosurface_value",
        "--n_binary_steps",
        "--postprocess",
        "--use_searched_pivots",
        "--filter_large_edges",
    }
    native_names = {value.split("=", 1)[0] for value in native_args}
    conflicts = sorted(controlled & native_names)
    if conflicts:
        parser.error(
            "Use wrapper options instead of these native options: "
            + ", ".join(conflicts)
        )
    return args, native_args


def load_run_manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid run manifest: {path}") from error
    return payload if isinstance(payload, dict) else {}


def generated_mesh_path(
    model_dir: Path,
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
    post_suffix = "_post" if postprocess else ""
    return model_dir / (
        f"mesh_{sdf_mode}_{n_pivots}pivots"
        f"{iso_suffix}{pivot_suffix}{post_suffix}.ply"
    )


def extraction_command(
    model_dir: Path,
    iteration: int,
    source: Path,
    rasterizer: str,
    args: argparse.Namespace,
    native_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "pivot_based_mesh_extraction.py",
        "-s",
        str(source),
        "-m",
        str(model_dir),
        "--iteration",
        str(iteration),
        "--rasterizer",
        rasterizer,
        "--sdf_mode",
        "ours" if rasterizer == "ours" else "exact_computation",
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
    if args.resolution is not None:
        command.extend(["-r", str(args.resolution)])
    if args.sdf_batch_size is not None:
        command.extend(
            ["--n_points_per_sdf_evaluation", str(args.sdf_batch_size)]
        )
    if args.mtet_on_cpu:
        command.append("--mtet_on_cpu")
    if not args.no_postprocess:
        command.append("--postprocess")
    if rasterizer == "ours":
        command.append("--filter_large_edges")
    else:
        command.extend(
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
    command.extend(native_args)
    return command


def simplify_or_copy(
    source: Path,
    output: Path,
    args: argparse.Namespace,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.target_faces is None and args.simplify_ratio is None:
        shutil.copy2(source, output)
        return
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Mesh simplification requires open3d.") from error
    mesh = o3d.io.read_triangle_mesh(str(source))
    original_faces = len(mesh.triangles)
    if original_faces < 1:
        raise ValueError(f"Extracted mesh has no triangles: {source}")
    target = args.target_faces
    if args.simplify_ratio is not None:
        target = max(1, round(original_faces * args.simplify_ratio))
    assert target is not None
    target = min(target, original_faces)
    if target < original_faces:
        mesh = mesh.simplify_quadric_decimation(target)
    if not args.no_clean:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.remove_non_manifold_edges()
    if not o3d.io.write_triangle_mesh(str(output), mesh):
        raise RuntimeError(f"Failed to write mesh: {output}")


def texture_command(
    source: Path,
    model_dir: Path,
    mesh: Path,
    iteration: int,
    rasterizer: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "texture_mesh.py",
        "-s",
        str(source),
        "-m",
        str(model_dir),
        "--iteration",
        str(iteration),
        "--rasterizer",
        rasterizer,
        "--mesh",
        str(mesh),
        "--n_iter",
        str(args.texture_iterations),
        "--data_device",
        args.data_device,
    ]
    if args.resolution is not None:
        command.extend(["-r", str(args.resolution)])
    return command


def semantic_mesh_command(
    mesh: Path,
    semantic_model: Path,
    semantic_iteration: int,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        "semantic_mesh.py",
        "--mesh",
        str(mesh),
        "--semantic_ply",
        str(point_cloud_path(semantic_model, semantic_iteration)),
        "--semantic_checkpoint",
        str(semantic_checkpoint_path(semantic_model, semantic_iteration)),
        "--output",
        str(output),
        "--workers",
        str(args.semantic_workers),
    ]


def extract_scene(
    scene: str,
    args: argparse.Namespace,
    native_args: list[str],
    *,
    data_root: Path,
    output_root: Path,
    workdir: Path,
    env: dict[str, str],
) -> int:
    source = data_root / scene
    run_dir = run_directory(output_root, scene, args.mode, args.mask_method)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training run does not exist: {run_dir}")
    geometry_model, semantic_model = model_directories(run_dir, args.mode)
    geometry_iteration = resolve_iteration(
        geometry_model,
        args.geometry_iteration,
    )
    manifest = load_run_manifest(run_dir)
    rasterizer = args.rasterizer or manifest.get("rasterizer", "radegs")
    mesh_dir = run_dir / "mesh"
    mesh = mesh_dir / f"{rasterizer}_iteration_{geometry_iteration}.ply"
    generated = generated_mesh_path(
        geometry_model,
        rasterizer,
        args.n_pivots,
        args.isosurface_value,
        not args.no_postprocess,
    )

    extract_cmd = extraction_command(
        geometry_model,
        geometry_iteration,
        source,
        rasterizer,
        args,
        native_args,
    )
    if mesh.is_file() and not args.force:
        print(f"[skip:{scene}:mesh] {mesh}")
    else:
        returncode = execute(
            extract_cmd,
            stage=f"mesh:{scene}:extract",
            cwd=workdir,
            env=env,
            dry_run=args.dry_run,
        )
        if returncode:
            return returncode
        if not args.dry_run:
            if not generated.is_file():
                raise FileNotFoundError(
                    f"Extraction completed without expected mesh: {generated}"
                )
            simplify_or_copy(generated, mesh, args)
            print(f"[mesh:{scene}] {mesh}")

    textured_mesh = mesh_dir / f"{mesh.stem}_textured.ply"
    if args.texture:
        texture_cmd = texture_command(
            source,
            geometry_model,
            mesh,
            geometry_iteration,
            rasterizer,
            args,
        )
        if textured_mesh.is_file() and not args.force:
            print(f"[skip:{scene}:texture] {textured_mesh}")
        else:
            returncode = execute(
                texture_cmd,
                stage=f"mesh:{scene}:texture",
                cwd=workdir,
                env=env,
                dry_run=args.dry_run,
            )
            if returncode:
                return returncode
            if not args.dry_run:
                generated_texture = geometry_model / (
                    f"{mesh.stem}_texture_refined_"
                    f"{args.texture_iterations - 1}.ply"
                )
                if not generated_texture.is_file():
                    raise FileNotFoundError(
                        "Texture refinement completed without expected mesh: "
                        f"{generated_texture}"
                    )
                mesh_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(generated_texture, textured_mesh)
                print(f"[texture:{scene}] {textured_mesh}")

    semantic_iteration = None
    semantic_mesh = mesh_dir / f"{mesh.stem}_semantic.ply"
    if not args.no_semantic_mesh:
        semantic_iteration = resolve_semantic_iteration(
            semantic_model,
            args.semantic_iteration,
        )
        semantic_cmd = semantic_mesh_command(
            mesh,
            semantic_model,
            semantic_iteration,
            semantic_mesh,
            args,
        )
        if semantic_mesh.is_file() and not args.force:
            print(f"[skip:{scene}:semantic] {semantic_mesh}")
        else:
            returncode = execute(
                semantic_cmd,
                stage=f"mesh:{scene}:semantic",
                cwd=workdir,
                env=env,
                dry_run=args.dry_run,
            )
            if returncode:
                return returncode

    if not args.dry_run:
        write_manifest(
            mesh_dir / f"{mesh.stem}_manifest.json",
            {
                "format_version": 1,
                "scene": scene,
                "mode": args.mode,
                "mask_method": args.mask_method,
                "rasterizer": rasterizer,
                "geometry_model": str(geometry_model),
                "geometry_iteration": geometry_iteration,
                "mesh": str(mesh),
                "textured_mesh": str(textured_mesh) if args.texture else None,
                "semantic_model": (
                    str(semantic_model) if not args.no_semantic_mesh else None
                ),
                "semantic_iteration": semantic_iteration,
                "semantic_mesh": (
                    str(semantic_mesh) if not args.no_semantic_mesh else None
                ),
                "extraction_command": extract_cmd,
            },
        )
    print(f"[complete:{scene}] {mesh_dir}")
    return 0


def main() -> int:
    args, native_args = parse_args()
    data_root = resolve_path(args.data_root)
    project_root = resolve_path(args.project_root)
    output_root = resolve_path(args.output_root)
    workdir = validate_project(project_root)
    available = discover_scenes(data_root, args.images)
    if args.list_scenes:
        print("\n".join(available))
        return 0
    if args.scenes:
        scenes = select_scenes(args.scenes, available)
    else:
        scenes = [
            scene
            for scene in available
            if run_directory(
                output_root,
                scene,
                args.mode,
                args.mask_method,
            ).is_dir()
        ]
        if not scenes:
            raise FileNotFoundError(
                "No matching trained runs found. Pass --scene or run the "
                "GaussianWrappingGaga training launcher first."
            )

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    failures: list[tuple[str, int]] = []
    for scene in scenes:
        try:
            returncode = extract_scene(
                scene,
                args,
                native_args,
                data_root=data_root,
                output_root=output_root,
                workdir=workdir,
                env=env,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"[failed:{scene}] {error}", file=sys.stderr)
            returncode = 2
        if returncode:
            failures.append((scene, returncode))
            if not args.keep_going:
                break

    if failures:
        for scene, returncode in failures:
            print(
                f"[summary:failed] {scene}: return code {returncode}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
