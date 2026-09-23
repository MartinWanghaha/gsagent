"""Render images and optional PGSR geometry from an EDGS training output.

The saved ``config.yaml`` is the source of truth for the dataset, resolution,
pipeline and renderer.  This is important for EDGS because ``cfg_args`` exists
only for compatibility with upstream Gaussian Splatting tools and older runs
may contain placeholder values.
"""

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the applicable repository license.
#
# For inquiries contact george.drettakis@inria.fr
#
# The output contract and TSDF workflow are adapted from Gaussian
# Splatting/PGSR while the model and camera path remains EDGS-native.

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from source.data_utils import scene_cameras_train_test_split
from source.pgsr_geometry import camera_intrinsics, camera_rays
from source.renderers import build_renderer
from source.vendor import bootstrap_gaussian_splatting

_ITERATION_PATTERN = re.compile(r"^iteration_(\d+)$")


def available_iterations(model_path: Path) -> list[int]:
    """Return sorted iterations that contain a loadable Gaussian PLY."""

    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.is_dir():
        return []

    iterations: list[int] = []
    for candidate in point_cloud_root.iterdir():
        match = _ITERATION_PATTERN.fullmatch(candidate.name)
        if (
            match
            and int(match.group(1)) > 0
            and (candidate / "point_cloud.ply").is_file()
        ):
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def resolve_iteration(model_path: Path, requested: int) -> int:
    """Resolve ``-1`` to the latest valid PLY and validate explicit values."""

    iterations = available_iterations(model_path)
    if not iterations:
        raise FileNotFoundError(
            f"no point_cloud/iteration_<N>/point_cloud.ply found in {model_path}"
        )
    if requested == -1:
        return iterations[-1]
    if requested <= 0:
        raise ValueError("iteration must be -1 (latest) or a positive integer")
    if requested not in iterations:
        available = ", ".join(map(str, iterations))
        raise FileNotFoundError(
            f"iteration {requested} is unavailable in {model_path}; "
            f"available iterations: {available}"
        )
    return requested


def load_training_config(
    model_path: Path,
    *,
    source_path: Path | None = None,
    images: str | None = None,
    resolution: int | None = None,
) -> DictConfig:
    """Load the resolved Hydra snapshot written by :mod:`train`.

    ``model_path`` is always overridden so a completed run can be moved.  The
    dataset location can also be overridden explicitly when it has moved.
    """

    config_path = model_path / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"missing {config_path}; render.py requires train.py's config.yaml"
        )

    config = OmegaConf.load(config_path)
    if not OmegaConf.select(config, "gs.dataset"):
        raise ValueError(f"{config_path} has no gs.dataset configuration")
    if not OmegaConf.select(config, "gs.pipe"):
        raise ValueError(f"{config_path} has no gs.pipe configuration")

    OmegaConf.resolve(config)
    config.gs.dataset.model_path = str(model_path)

    if source_path is not None:
        config.gs.dataset.source_path = str(source_path)
    else:
        saved_source = Path(str(config.gs.dataset.source_path)).expanduser()
        if not saved_source.is_absolute():
            saved_source = (Path(__file__).resolve().parent / saved_source).resolve()
        config.gs.dataset.source_path = str(saved_source)
    if images is not None:
        config.gs.dataset.images = images
    if resolution is not None:
        config.gs.dataset.resolution = int(resolution)

    resolved_source = Path(str(config.gs.dataset.source_path))
    if not resolved_source.is_dir():
        raise FileNotFoundError(
            f"dataset source_path does not exist: {resolved_source}; "
            "provide --source-path if the dataset was moved"
        )
    return config


def _configure_cuda(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError(
            "EDGS rendering currently requires CUDA because its pinned Camera "
            "and GaussianModel allocate CUDA tensors directly"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the active PyTorch environment")
    if device.index is not None:
        torch.cuda.set_device(device.index)
    return torch.device("cuda", torch.cuda.current_device())


def _output_filename(image_name: str) -> str:
    basename = Path(str(image_name)).name
    if not basename:
        raise ValueError(f"invalid empty camera image name: {image_name!r}")
    return str(Path(basename).with_suffix(".png"))


def _view_filenames(views: Sequence[Any]) -> list[str]:
    names = [_output_filename(view.image_name) for view in views]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(
            "camera names collide after conversion to PNG: " + ", ".join(duplicates)
        )
    return names


def _rgb_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach()
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        raise ValueError(f"expected image shaped [C,H,W], got {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if image.shape[0] < 3:
        raise ValueError(f"expected at least three channels, got {image.shape[0]}")
    array = (
        image[:3]
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return np.ascontiguousarray(array)


def _save_png(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    Image.fromarray(_rgb_uint8(image), mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


def _depth_visualization(depth: torch.Tensor) -> torch.Tensor:
    """Map positive finite depth to a compact JET-style RGB visualization."""

    depth = depth.detach().squeeze()
    if depth.ndim != 2:
        raise ValueError(f"expected a 2D depth map, got {tuple(depth.shape)}")
    valid = torch.isfinite(depth) & (depth > 0)
    normalized = torch.zeros_like(depth)
    if valid.any():
        minimum = depth[valid].min()
        maximum = depth[valid].max()
        normalized[valid] = (depth[valid] - minimum) / (maximum - minimum + 1e-12)
    red = (1.5 - (4.0 * normalized - 3.0).abs()).clamp(0.0, 1.0)
    green = (1.5 - (4.0 * normalized - 2.0).abs()).clamp(0.0, 1.0)
    blue = (1.5 - (4.0 * normalized - 1.0).abs()).clamp(0.0, 1.0)
    color = torch.stack((red, green, blue))
    return color * valid.unsqueeze(0)


def _normal_visualization(normal: torch.Tensor) -> torch.Tensor:
    normal = normal.detach()
    if normal.ndim != 3 or normal.shape[0] != 3:
        raise ValueError(f"expected normals shaped [3,H,W], got {tuple(normal.shape)}")
    normal = F.normalize(normal, p=2, dim=0, eps=1e-8)
    valid = torch.isfinite(normal).all(dim=0, keepdim=True)
    return torch.where(valid, (normal + 1.0) * 0.5, 0.0)


def _filtered_depth(
    camera: Any,
    package: dict[str, torch.Tensor],
    *,
    max_depth: float,
    use_depth_filter: bool,
) -> torch.Tensor:
    depth = package["plane_depth"].detach().squeeze().clone()
    valid = torch.isfinite(depth) & (depth > 0) & (depth <= max_depth)

    alpha_mask = getattr(camera, "alpha_mask", None)
    if isinstance(alpha_mask, torch.Tensor):
        alpha_mask = alpha_mask.detach().squeeze()
        if alpha_mask.shape != depth.shape:
            alpha_mask = F.interpolate(
                alpha_mask[None, None].float(),
                size=depth.shape,
                mode="nearest",
            )[0, 0]
        valid &= alpha_mask.to(device=depth.device) >= 0.5

    if use_depth_filter:
        depth_normal = package.get("depth_normal")
        if depth_normal is None:
            raise KeyError("depth filtering requires renderer output 'depth_normal'")
        rays = camera_rays(
            camera,
            device=depth.device,
            dtype=depth.dtype,
            normalize=True,
        )
        normals = F.normalize(depth_normal.permute(1, 2, 0), p=2, dim=-1, eps=1e-8)
        cosine = (rays * normals).sum(dim=-1).abs().clamp(0.0, 1.0)
        valid &= torch.acos(cosine) <= math.radians(80.0)

    return torch.where(valid, depth, 0.0)


class TSDFIntegrator:
    """Small lazy Open3D adapter used only when mesh extraction is requested."""

    def __init__(self, voxel_size: float, max_depth: float) -> None:
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError(
                "TSDF mesh extraction requires Open3D; install it in paintmesh "
                "before using --extract-mesh"
            ) from error

        self.o3d = o3d
        self.max_depth = float(max_depth)
        self.integrated_frames = 0
        self.valid_depth_pixels = 0
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size),
            sdf_trunc=4.0 * float(voxel_size),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def integrate(
        self,
        camera: Any,
        color: torch.Tensor,
        depth: torch.Tensor,
    ) -> None:
        height, width = map(int, depth.shape)
        color_image = self.o3d.geometry.Image(_rgb_uint8(color))
        depth_meters = depth.detach().clamp(0.0, self.max_depth).float().cpu().numpy()
        self.integrated_frames += 1
        self.valid_depth_pixels += int(np.count_nonzero(depth_meters > 0))
        depth_image = self.o3d.geometry.Image(np.ascontiguousarray(depth_meters))
        rgbd = self.o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            depth_scale=1.0,
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False,
        )

        intrinsics = camera_intrinsics(camera, device="cpu").numpy()
        intrinsic = self.o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
        )
        # EDGS stores the OpenGL-facing matrix transposed for CUDA/GLM.  Open3D
        # expects the conventional world-to-camera matrix.
        extrinsic = (
            torch.as_tensor(camera.world_view_transform)
            .detach()
            .transpose(0, 1)
            .cpu()
            .numpy()
        )
        self.volume.integrate(rgbd, intrinsic, np.ascontiguousarray(extrinsic))

    def extract(self):
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh


def _post_process_mesh(mesh: Any, clusters_to_keep: int):
    """Keep the largest connected components and remove degenerate geometry."""

    processed = copy.deepcopy(mesh)
    if len(processed.triangles) == 0:
        return processed

    triangle_clusters, cluster_counts, _ = processed.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_counts = np.asarray(cluster_counts)
    keep_count = min(int(clusters_to_keep), len(cluster_counts))
    minimum_size = max(int(np.sort(cluster_counts)[-keep_count]), 50)
    processed.remove_triangles_by_mask(cluster_counts[triangle_clusters] < minimum_size)
    processed.remove_unreferenced_vertices()
    processed.remove_degenerate_triangles()
    processed.compute_vertex_normals()
    return processed


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _remove_unexpected_images(directory: Path, expected: set[str]) -> None:
    """Remove stale image outputs after a complete replacement render."""

    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if (
            path.is_file()
            and path.suffix.lower()
            in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
            and path.name not in expected
        ):
            path.unlink()


def render_set(
    *,
    model_path: Path,
    split: str,
    iteration: int,
    views: Sequence[Any],
    renderer: Any,
    gaussians: Any,
    pipeline: Any,
    background: torch.Tensor,
    max_depth: float,
    use_depth_filter: bool,
    tsdf: TSDFIntegrator | None = None,
    quiet: bool = False,
) -> Path:
    """Render one camera split and return its PGSR-compatible method path."""

    method_path = model_path / split / f"ours_{iteration}"
    renders_path = method_path / "renders"
    ground_truth_path = method_path / "gt"
    renders_path.mkdir(parents=True, exist_ok=True)
    ground_truth_path.mkdir(parents=True, exist_ok=True)

    has_geometry = renderer.backend == "pgsr"
    depth_path = method_path / "renders_depth"
    normal_path = method_path / "renders_normal"
    if has_geometry:
        depth_path.mkdir(parents=True, exist_ok=True)
        normal_path.mkdir(parents=True, exist_ok=True)

    filenames = _view_filenames(views)
    mapping = {
        str(view.image_name): filename for view, filename in zip(views, filenames)
    }
    manifest_path = method_path / "render_manifest.json"
    manifest = {
        "backend": renderer.backend,
        "complete": False,
        "iteration": iteration,
        "num_views": len(views),
        "split": split,
        "views": mapping,
    }
    # Mark the method incomplete before touching image files.  Metrics can then
    # reject interrupted reruns instead of silently mixing old and new frames.
    _write_json(manifest_path, manifest)
    for view, filename in tqdm(
        zip(views, filenames),
        total=len(views),
        desc=f"Rendering {split}",
        disable=quiet,
    ):
        package = renderer.render(
            view,
            gaussians,
            pipeline,
            background,
            return_plane=has_geometry,
            return_depth_normal=has_geometry and use_depth_filter,
        )
        rendering = package["render"].clamp(0.0, 1.0)
        ground_truth = view.original_image[:3].clamp(0.0, 1.0)
        _save_png(rendering, renders_path / filename)
        _save_png(ground_truth, ground_truth_path / filename)

        if has_geometry:
            _save_png(
                _depth_visualization(package["plane_depth"]),
                depth_path / filename,
            )
            _save_png(
                _normal_visualization(package["rendered_normal"]),
                normal_path / filename,
            )
            if tsdf is not None:
                fusion_depth = _filtered_depth(
                    view,
                    package,
                    max_depth=max_depth,
                    use_depth_filter=use_depth_filter,
                )
                tsdf.integrate(view, rendering, fusion_depth)

    expected = set(filenames)
    _remove_unexpected_images(renders_path, expected)
    _remove_unexpected_images(ground_truth_path, expected)
    if has_geometry:
        _remove_unexpected_images(depth_path, expected)
        _remove_unexpected_images(normal_path, expected)
    else:
        # A backend override may replace an older PGSR render at the same
        # iteration.  Do not leave geometry that no longer describes RGB.
        _remove_unexpected_images(depth_path, set())
        _remove_unexpected_images(normal_path, set())

    manifest["complete"] = True
    _write_json(manifest_path, manifest)
    return method_path


def _save_meshes(
    tsdf: TSDFIntegrator,
    model_path: Path,
    iteration: int,
    clusters_to_keep: int,
) -> Path:
    mesh_path = model_path / "mesh" / f"ours_{iteration}"
    mesh_path.mkdir(parents=True, exist_ok=True)
    raw_mesh = tsdf.extract()
    if len(raw_mesh.vertices) == 0 or len(raw_mesh.triangles) == 0:
        raise RuntimeError(
            "TSDF fusion produced an empty mesh after "
            f"{tsdf.integrated_frames} frames and {tsdf.valid_depth_pixels} "
            "valid depth pixels; adjust --max-depth/--voxel-size or disable "
            "--use-depth-filter"
        )
    raw_path = mesh_path / "tsdf_fusion.ply"
    if not tsdf.o3d.io.write_triangle_mesh(str(raw_path), raw_mesh):
        raise RuntimeError(f"Open3D failed to write {raw_path}")

    processed = _post_process_mesh(raw_mesh, clusters_to_keep)
    if len(processed.vertices) == 0 or len(processed.triangles) == 0:
        raise RuntimeError(
            "mesh post-processing removed every component; increase scene "
            "coverage or inspect tsdf_fusion.ply"
        )
    processed_path = mesh_path / "tsdf_fusion_post.ply"
    if not tsdf.o3d.io.write_triangle_mesh(str(processed_path), processed):
        raise RuntimeError(f"Open3D failed to write {processed_path}")
    return mesh_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a model produced by EDGS train.py"
    )
    parser.add_argument(
        "-m",
        "--model-path",
        "--model_path",
        required=True,
        type=Path,
        help="EDGS output directory containing config.yaml and point_cloud/",
    )
    parser.add_argument(
        "--iteration",
        default=-1,
        type=int,
        help="saved iteration to render; -1 selects the latest",
    )
    parser.add_argument("--skip-train", "--skip_train", action="store_true")
    parser.add_argument("--skip-test", "--skip_test", action="store_true")
    parser.add_argument(
        "--extract-mesh",
        "--extract_mesh",
        action="store_true",
        help="fuse PGSR training-view depths into a TSDF mesh (requires Open3D)",
    )
    parser.add_argument("--max-depth", "--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel-size", "--voxel_size", default=0.002, type=float)
    parser.add_argument(
        "--num-clusters",
        "--num_cluster",
        default=1,
        type=int,
        help="number of largest connected mesh components to retain",
    )
    parser.add_argument("--use-depth-filter", "--use_depth_filter", action="store_true")
    parser.add_argument(
        "--renderer",
        choices=("auto", "native", "pgsr"),
        default="auto",
        help="default: renderer saved in config.yaml",
    )
    parser.add_argument(
        "--source-path",
        "--source_path",
        type=Path,
        help="override the saved dataset path",
    )
    parser.add_argument("--images", help="override the saved image directory name")
    parser.add_argument(
        "--resolution", type=int, help="override the saved dataset resolution"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device, e.g. cuda or cuda:1 (default: cuda)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model_path = args.model_path.expanduser().resolve()
    if args.skip_train and args.skip_test:
        raise ValueError("--skip-train and --skip-test cannot be used together")
    if args.max_depth <= 0 or not math.isfinite(args.max_depth):
        raise ValueError("--max-depth must be finite and positive")
    if args.voxel_size <= 0 or not math.isfinite(args.voxel_size):
        raise ValueError("--voxel-size must be finite and positive")
    if args.num_clusters <= 0:
        raise ValueError("--num-clusters must be positive")
    if args.resolution is not None and args.resolution != -1 and args.resolution <= 0:
        raise ValueError("--resolution must be -1 or a positive integer")
    if args.extract_mesh and args.skip_train:
        raise ValueError("--extract-mesh requires the training split")
    if args.use_depth_filter and not args.extract_mesh:
        raise ValueError("--use-depth-filter is only meaningful with --extract-mesh")

    iteration = resolve_iteration(model_path, args.iteration)
    config = load_training_config(
        model_path,
        source_path=(
            args.source_path.expanduser().resolve()
            if args.source_path is not None
            else None
        ),
        images=args.images,
        resolution=args.resolution,
    )
    device = _configure_cuda(args.device)
    saved_data_device = torch.device(str(config.gs.dataset.data_device))
    if saved_data_device.type == "cuda":
        config.gs.dataset.data_device = str(device)

    renderer_config = config.gs.get("renderer", {"backend": "native"})
    if args.renderer != "auto":
        clamp_rgb = (
            bool(renderer_config.get("clamp_rgb", True))
            if hasattr(renderer_config, "get")
            else True
        )
        renderer_config = OmegaConf.create(
            {
                "backend": args.renderer,
                "clamp_rgb": clamp_rgb,
            }
        )
    renderer = build_renderer(renderer_config)
    if args.extract_mesh and renderer.backend != "pgsr":
        raise ValueError("TSDF extraction needs PGSR plane depth; use --renderer pgsr")

    bootstrap_gaussian_splatting()
    from scene import GaussianModel, Scene

    if not args.quiet:
        print(f"Model: {model_path}")
        print(f"Iteration: {iteration}")
        print(f"Renderer: {renderer.backend}")
        print(f"Resolution: {config.gs.dataset.resolution}")

    # PGSR's rasterizer creates differentiable screen-space placeholders even
    # for evaluation, so use no_grad instead of the stricter inference_mode.
    with torch.no_grad():
        gaussians = GaussianModel(int(config.gs.sh_degree))
        scene = Scene(
            config.gs.dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
        )
        scene_cameras_train_test_split(scene, verbose=not args.quiet)
        background = torch.tensor(
            (
                [1.0, 1.0, 1.0]
                if bool(config.gs.dataset.white_background)
                else [0.0, 0.0, 0.0]
            ),
            dtype=gaussians.get_xyz.dtype,
            device=gaussians.get_xyz.device,
        )

        tsdf = (
            TSDFIntegrator(args.voxel_size, args.max_depth)
            if args.extract_mesh
            else None
        )
        if not args.skip_train:
            render_set(
                model_path=model_path,
                split="train",
                iteration=iteration,
                views=scene.getTrainCameras(),
                renderer=renderer,
                gaussians=gaussians,
                pipeline=config.gs.pipe,
                background=background,
                max_depth=args.max_depth,
                use_depth_filter=args.use_depth_filter,
                tsdf=tsdf,
                quiet=args.quiet,
            )
            if tsdf is not None:
                mesh_path = _save_meshes(
                    tsdf,
                    model_path,
                    iteration,
                    args.num_clusters,
                )
                print(f"Meshes: {mesh_path}")

        if not args.skip_test:
            render_set(
                model_path=model_path,
                split="test",
                iteration=iteration,
                views=scene.getTestCameras(),
                renderer=renderer,
                gaussians=gaussians,
                pipeline=config.gs.pipe,
                background=background,
                max_depth=args.max_depth,
                use_depth_filter=args.use_depth_filter,
                quiet=args.quiet,
            )

    print(f"Rendering complete: ours_{iteration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
