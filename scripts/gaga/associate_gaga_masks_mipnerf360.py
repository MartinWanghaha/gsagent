#!/usr/bin/env python3
"""Associate Gaga 2D masks across Mip-NeRF 360 camera views.

Examples:
    python scripts/gaga/associate_gaga_masks_mipnerf360.py \
        --scene counter \
        --point-cloud outputs/gaussian_wrapping_mipnerf360/counter/point_cloud/iteration_30000/point_cloud.ply \
        --gpu 0 --visualize

    python scripts/gaga/associate_gaga_masks_mipnerf360.py \
        --scene bicycle,counter \
        --point-cloud outputs/gaussian_wrapping_mipnerf360 \
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


GSAGENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = GSAGENT_ROOT / "data" / "mip-nerf" / "360_v2"
DEFAULT_GAGA_ROOT = GSAGENT_ROOT / "submodules" / "Gaga"
ITERATION_PATTERN = re.compile(r"^iteration_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Associate per-view Gaga masks using a scene-aligned 3DGS point cloud. "
            "Results are written back into each dataset scene."
        )
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scenes",
        help=(
            "Scene name. Repeat the option or use comma-separated names. "
            "Default: all discovered scenes."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Mip-NeRF 360 root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--gaga-root",
        type=Path,
        default=DEFAULT_GAGA_ROOT,
        help=f"Gaga checkout root. Default: {DEFAULT_GAGA_ROOT}",
    )
    parser.add_argument(
        "--point-cloud",
        "--point-cloud-root",
        dest="point_cloud",
        type=Path,
        help=(
            "A point_cloud.ply for one selected scene, a 3DGS model directory, "
            "or a root containing one model directory per scene."
        ),
    )
    parser.add_argument(
        "--images",
        default="images",
        help="Image subdirectory matching the COLMAP cameras. Default: images.",
    )
    parser.add_argument(
        "--seg-method",
        choices=("sam", "entityseg"),
        default="sam",
        help="Raw-mask source and associated-mask output name. Default: sam.",
    )
    parser.add_argument(
        "--front-percentage",
        type=float,
        default=0.2,
        help="Closest projected Gaussians retained in each mask patch. Default: 0.2.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.1,
        help="Minimum Gaussian overlap for matching an existing group. Default: 0.1.",
    )
    parser.add_argument(
        "--num-patch",
        type=int,
        default=32,
        help="Patch grid size per image axis. Default: 32.",
    )
    parser.add_argument(
        "--camera-resolution",
        choices=(1, 2, 4, 8),
        type=int,
        default=1,
        help=(
            "Downsample factor for RGB tensors loaded by Gaga. RGB pixels are not "
            "used during association; masks remain at their saved resolution. Default: 8."
        ),
    )
    parser.add_argument(
        "--gpu",
        help="Value assigned to CUDA_VISIBLE_DEVICES, for example 0.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also write false-color associated masks to <method>_mask_vis.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run association even when a complete associated-mask directory exists.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later scenes after an association failure.",
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List discovered scenes and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print resolved point clouds without using CUDA.",
    )

    args = parser.parse_args()
    if not 0.0 < args.front_percentage <= 1.0:
        parser.error("--front-percentage must be in (0, 1].")
    if not 0.0 <= args.overlap_threshold <= 1.0:
        parser.error("--overlap-threshold must be in [0, 1].")
    if args.num_patch <= 0:
        parser.error("--num-patch must be positive.")
    return args


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def is_colmap_scene(scene_dir: Path, images: str) -> bool:
    sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.is_dir():
        sparse_dir = scene_dir / "sparse"
    has_cameras = (sparse_dir / "cameras.bin").is_file() or (sparse_dir / "cameras.txt").is_file()
    has_images = (sparse_dir / "images.bin").is_file() or (sparse_dir / "images.txt").is_file()
    return (scene_dir / images).is_dir() and has_cameras and has_images


def discover_scenes(data_root: Path, images: str) -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    return [
        item.name
        for item in sorted(data_root.iterdir())
        if item.is_dir() and is_colmap_scene(item, images)
    ]


def select_scenes(requested: list[str] | None, available: list[str]) -> list[str]:
    if not requested:
        return available
    selected = [
        name.strip()
        for item in requested
        for name in item.split(",")
        if name.strip()
    ]
    if any(name.lower() == "all" for name in selected):
        return available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(
            f"Unknown scene(s): {', '.join(missing)}. Available: {', '.join(available)}"
        )
    selected_set = set(selected)
    return [scene for scene in available if scene in selected_set]


def iteration_number(point_cloud: Path) -> int:
    match = ITERATION_PATTERN.match(point_cloud.parent.name)
    return int(match.group(1)) if match else -1


def point_cloud_candidates(model_dir: Path) -> list[Path]:
    candidates: set[Path] = set()
    direct = model_dir / "point_cloud.ply"
    if direct.is_file():
        candidates.add(direct.resolve())
    point_cloud_dir = model_dir / "point_cloud"
    if point_cloud_dir.is_dir():
        candidates.update(
            path.resolve()
            for path in point_cloud_dir.glob("iteration_*/point_cloud.ply")
            if path.is_file()
        )
    candidates.update(
        path.resolve()
        for path in model_dir.glob("*/point_cloud/iteration_*/point_cloud.ply")
        if path.is_file()
    )
    return sorted(
        candidates,
        key=lambda path: (iteration_number(path), path.stat().st_mtime_ns, str(path)),
    )


def resolve_scene_point_cloud(
    source: Path,
    scene: str,
    selected_scene_count: int,
) -> Path:
    if source.is_file():
        if source.name != "point_cloud.ply":
            raise ValueError(f"Expected a file named point_cloud.ply: {source}")
        if selected_scene_count != 1:
            raise ValueError(
                "A direct point_cloud.ply can only be used with exactly one selected --scene"
            )
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"Point-cloud path does not exist: {source}")

    search_dirs = [source / scene]
    if selected_scene_count == 1:
        search_dirs.append(source)
    candidates: list[Path] = []
    for search_dir in search_dirs:
        if search_dir.is_dir():
            candidates.extend(point_cloud_candidates(search_dir))
    candidates = sorted(
        set(candidates),
        key=lambda path: (iteration_number(path), path.stat().st_mtime_ns, str(path)),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No point_cloud.ply found for scene '{scene}' under: {source}"
        )
    return candidates[-1]


def inspect_point_cloud(path: Path) -> tuple[int, int]:
    from plyfile import PlyData  # pylint: disable=import-outside-toplevel

    ply = PlyData.read(str(path), mmap="c")
    try:
        vertex = ply["vertex"]
    except KeyError as exc:
        raise ValueError(f"PLY has no vertex element: {path}") from exc
    properties = {prop.name for prop in vertex.properties}
    required = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required - properties)
    if missing:
        raise ValueError(
            f"PLY is not a standard trained 3DGS point cloud; missing: {', '.join(missing)}"
        )
    if not any(name.startswith("scale_") for name in properties):
        raise ValueError("PLY has no scale_* Gaussian properties")
    if not any(name.startswith("rot_") for name in properties):
        raise ValueError("PLY has no rot_* Gaussian properties")

    rest_count = sum(name.startswith("f_rest_") for name in properties)
    coefficient_count = rest_count / 3 + 1
    sh_degree = round(math.sqrt(coefficient_count) - 1)
    expected_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    if expected_rest != rest_count:
        raise ValueError(
            f"Cannot infer a valid SH degree from {rest_count} f_rest_* properties"
        )
    return len(vertex.data), sh_degree


def validate_raw_masks(scene_dir: Path, images: str, seg_method: str) -> tuple[Path, int]:
    raw_dir = scene_dir / f"raw_{seg_method}_mask"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw mask directory does not exist: {raw_dir}")
    raw_masks = sorted(raw_dir.glob("*.png"))
    if not raw_masks:
        raise FileNotFoundError(f"No PNG masks found in: {raw_dir}")

    image_stems = {
        path.stem
        for path in (scene_dir / images).iterdir()
        if path.is_file()
    }
    mask_stems = {path.stem for path in raw_masks}
    missing = sorted(image_stems - mask_stems)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Raw masks are missing for {len(missing)} image(s): {preview}{suffix}"
        )
    return raw_dir, len(raw_masks)


def association_is_complete(scene_dir: Path, seg_method: str, expected: int) -> bool:
    associated_dir = scene_dir / f"{seg_method}_mask"
    info_path = associated_dir / "info.json"
    if not info_path.is_file():
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    output_count = sum(path.is_file() for path in associated_dir.glob("*.png"))
    return "num_mask" in info and output_count >= expected


def load_projector_class(gaga_root: Path) -> type[Any]:
    required = [gaga_root / "mask" / "projector.py", gaga_root / "scene" / "gaussian_model.py"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Gaga checkout is missing required file(s): " + ", ".join(map(str, missing))
        )
    sys.path.insert(0, str(gaga_root))

    import torch  # pylint: disable=import-outside-toplevel
    from mask.projector import GaussianProjector  # pylint: disable=import-outside-toplevel

    class MemoryEfficientGaussianProjector(GaussianProjector):
        """Equivalent patch selection without a num_patch^2 x W x H tensor."""

        def __init__(self, dataset, pipeline, iteration, params, device):
            requested_patches = params["num_patch"]
            init_params = dict(params)
            init_params["num_patch"] = 1
            super().__init__(dataset, pipeline, iteration, init_params, device)

            del self.patch_mask
            del self.flatten_patch_mask
            self.num_patches = requested_patches
            self.patch_width = math.ceil(self.image_width / self.num_patches)
            self.patch_height = math.ceil(self.image_height / self.num_patches)

        def initialize(self, viewpoint):
            front_gaussian, _ = self.get_patch_front_gaussian_of_mask(viewpoint)
            if not front_gaussian:
                self.assigned_gaussians = torch.empty(
                    0, dtype=torch.long, device=self.device
                )
                return torch.empty(0, dtype=torch.long, device=self.device)
            self.gaussian_idx_bank.extend(front_gaussian)
            self.assigned_gaussians = torch.unique(torch.cat(front_gaussian))
            self.get_num_mask
            return torch.arange(self.num_mask, dtype=torch.long, device=self.device)

        def get_patch_front_gaussian_of_mask(self, viewpoint):
            projected = self.project_gaussian(viewpoint)
            projected_flat = projected["p_proj_flatten"]
            inside_indices = projected["p_proj_inside_indices"]
            projected_depth = projected["p_hom_z"]
            mask = self.load_mask(viewpoint)

            pixel_x = torch.div(
                projected_flat, self.image_height, rounding_mode="floor"
            )
            pixel_y = projected_flat.remainder(self.image_height)
            patch_x = torch.div(pixel_x, self.patch_width, rounding_mode="floor")
            patch_y = torch.div(pixel_y, self.patch_height, rounding_mode="floor")
            patch_x = patch_x.clamp(max=self.num_patches - 1)
            patch_y = patch_y.clamp(max=self.num_patches - 1)
            patch_ids = patch_x * self.num_patches + patch_y

            front_gaussian = []
            for object_mask in mask:
                projected_in_object = object_mask.flatten()[projected_flat]
                object_patch_ids = torch.unique(patch_ids[projected_in_object])
                patch_front = []
                for patch_id in object_patch_ids:
                    projected_positions = torch.nonzero(
                        projected_in_object & (patch_ids == patch_id),
                        as_tuple=False,
                    ).squeeze(-1)
                    gaussian_indices = inside_indices[projected_positions]
                    if gaussian_indices.numel() == 0:
                        continue
                    count = max(
                        int(self.front_percentage * gaussian_indices.numel()), 1
                    )
                    depth = projected_depth[gaussian_indices]
                    patch_front.append(gaussian_indices[torch.argsort(depth)[:count]])
                if patch_front:
                    front_gaussian.append(torch.cat(patch_front))
                else:
                    front_gaussian.append(
                        torch.empty(0, dtype=torch.long, device=self.device)
                    )
            return front_gaussian, mask

    return MemoryEfficientGaussianProjector


def build_gaga_args(
    scene_dir: Path,
    staged_model: Path,
    args: argparse.Namespace,
    sh_degree: int,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    dataset = SimpleNamespace(
        sh_degree=sh_degree,
        source_path=str(scene_dir),
        model_path=str(staged_model),
        trained_model_path="",
        images=args.images,
        resolution=args.camera_resolution,
        white_background=False,
        data_device="cpu",
        eval=False,
        n_views=100,
        random_init=False,
        train_split=False,
        object_path=f"{args.seg_method}_mask",
        num_classes=256,
        lift=False,
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    return dataset, pipeline


def run_association(
    projector_class: type[Any],
    scene_dir: Path,
    point_cloud: Path,
    args: argparse.Namespace,
    sh_degree: int,
) -> None:
    import torch  # pylint: disable=import-outside-toplevel

    with tempfile.TemporaryDirectory(prefix="gaga-association-") as temp_dir:
        staged_model = Path(temp_dir)
        staged_iteration = staged_model / "point_cloud" / "iteration_1"
        staged_iteration.mkdir(parents=True)
        (staged_iteration / "point_cloud.ply").symlink_to(point_cloud)

        dataset, pipeline = build_gaga_args(scene_dir, staged_model, args, sh_degree)
        params = {
            "front_percentage": args.front_percentage,
            "overlap_threshold": args.overlap_threshold,
            "num_patch": args.num_patch,
            "seg_method": args.seg_method,
            "visualize": args.visualize,
        }
        with torch.no_grad():
            projector = projector_class(
                dataset,
                pipeline,
                iteration=1,
                params=params,
                device=torch.device("cuda"),
            )
            projector.build_mask_association()
        del projector


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    gaga_root = resolve_path(args.gaga_root)
    available = discover_scenes(data_root, args.images)
    if not available:
        raise RuntimeError(f"No Mip-NeRF 360 COLMAP scenes found in: {data_root}")
    if args.list_scenes:
        print("\n".join(available))
        return 0
    if args.point_cloud is None:
        raise ValueError("--point-cloud is required unless --list-scenes is used")

    scenes = select_scenes(args.scenes, available)
    point_cloud_source = resolve_path(args.point_cloud)
    jobs: list[tuple[str, Path, Path, int, int]] = []
    for scene in scenes:
        scene_dir = data_root / scene
        point_cloud = resolve_scene_point_cloud(
            point_cloud_source, scene, len(scenes)
        )
        _, raw_count = validate_raw_masks(scene_dir, args.images, args.seg_method)
        point_count, sh_degree = inspect_point_cloud(point_cloud)
        jobs.append((scene, scene_dir, point_cloud, point_count, sh_degree))
        output_dir = scene_dir / f"{args.seg_method}_mask"
        print(
            f"[resolve] {scene}: {point_cloud} "
            f"({point_count} Gaussians, SH degree {sh_degree}) -> {output_dir}"
        )

    if args.dry_run:
        return 0
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch  # pylint: disable=import-outside-toplevel

    if not torch.cuda.is_available():
        raise RuntimeError("Gaga mask association requires a CUDA GPU")
    projector_class = load_projector_class(gaga_root)

    failures: list[tuple[str, str]] = []
    for scene, scene_dir, point_cloud, _, sh_degree in jobs:
        _, raw_count = validate_raw_masks(scene_dir, args.images, args.seg_method)
        if association_is_complete(scene_dir, args.seg_method, raw_count) and not args.force:
            print(f"[skip] {scene}: associated masks are already complete")
            continue
        print(f"[associate] {scene}")
        try:
            run_association(
                projector_class,
                scene_dir,
                point_cloud,
                args,
                sh_degree,
            )
            print(f"[done] {scene}: {scene_dir / f'{args.seg_method}_mask'}")
        except Exception as exc:  # Batch mode should report the exact failed scene.
            failures.append((scene, str(exc)))
            print(f"[failed] {scene}: {exc}", file=sys.stderr)
            if not args.keep_going:
                break
        finally:
            torch.cuda.empty_cache()

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
