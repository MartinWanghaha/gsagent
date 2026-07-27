"""Render Gaga-compatible images and Gaussian Wrapping diagnostics."""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
SUBMODULES_DIR = os.path.join(ROOT_DIR, "submodules")
sys.path.extend((ROOT_DIR, SUBMODULES_DIR))

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel, Scene
from semantic import (
    GagaObservationStore,
    SemanticHead,
    gaga_palette,
    load_semantic_checkpoint,
)
from utils.general_utils import safe_state


GAGA_COMMON_DIRECTORIES = (
    "renders",
    "gt",
    "objects_feature16",
    "objects_pred",
    "objects_test",
)
GAGA_TRAIN_DIRECTORIES = ("gt_objects", "gt_objects_color")
GW_IMAGE_DIRECTORIES = (
    "ground_truth",
    "depth",
    "expected_depth",
    "median_depth",
    "normal",
    "alpha",
    "semantic_labels",
    "semantic_color",
)
GW_NUMERIC_DIRECTORIES = (
    "depth_npy",
    "expected_depth_npy",
    "median_depth_npy",
    "expected_coord",
    "median_coord",
    "normal_npy",
    "alpha_npy",
    "radii",
    "renderer_aux",
    "semantic_features",
    "semantic_logits",
)


def _rgb_image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def _scalar_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().squeeze().cpu().numpy()
    finite = np.isfinite(array)
    if finite.any():
        low, high = np.percentile(array[finite], (1, 99))
        array = np.clip((array - low) / max(high - low, 1e-8), 0, 1)
    else:
        array = np.zeros_like(array)
    return Image.fromarray((array * 255).astype(np.uint8))


def _unit_scalar_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().squeeze().clamp(0, 1).mul(255).byte()
    return Image.fromarray(array.cpu().numpy())


def _save_label_image(labels: np.ndarray, path: Path) -> None:
    maximum = int(labels.max()) if labels.size else 0
    if maximum > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Semantic label {maximum} exceeds the lossless uint16 PNG range"
        )
    Image.fromarray(labels.astype(np.uint16, copy=False)).save(path)


def feature_pca_image(
    features: torch.Tensor,
    *,
    chunk_size: int = 262_144,
) -> Image.Image:
    """Visualize a CxHxW feature map using a memory-bounded 3D PCA."""
    channels, height, width = features.shape
    if channels < 3:
        raise ValueError("At least three semantic channels are required for PCA")
    flat = features.detach().float().cpu().reshape(channels, -1).transpose(0, 1)
    mean = flat.mean(dim=0)
    covariance = torch.zeros(
        (channels, channels),
        dtype=torch.float64,
    )
    for start in range(0, flat.shape[0], chunk_size):
        centered = flat[start : start + chunk_size].double() - mean.double()
        covariance.add_(centered.transpose(0, 1) @ centered)
    covariance.div_(max(flat.shape[0] - 1, 1))
    _, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -3:].float()

    projected = np.empty((flat.shape[0], 3), dtype=np.float32)
    for start in range(0, flat.shape[0], chunk_size):
        values = (flat[start : start + chunk_size] - mean) @ components
        projected[start : start + values.shape[0]] = values.numpy()
    low = projected.min(axis=0, keepdims=True)
    high = projected.max(axis=0, keepdims=True)
    projected = np.clip((projected - low) / np.maximum(high - low, 1e-8), 0, 1)
    rgb = (projected.reshape(height, width, 3) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


@torch.inference_mode()
def classify_features_chunked(
    head: SemanticHead,
    features: torch.Tensor,
    *,
    chunk_size: int,
    logits_path: Path | None = None,
) -> np.ndarray:
    """Classify pixels without materializing a CxHxW CUDA logits tensor."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    channels, height, width = features.shape
    if channels != head.semantic_dim:
        raise ValueError(
            f"Expected {head.semantic_dim} feature channels, got {channels}"
        )
    if head.num_classes > np.iinfo(np.uint16).max + 1:
        raise ValueError("The Gaga-compatible PNG exporter supports <=65536 classes")

    pixels = features.detach().reshape(channels, -1).transpose(0, 1)
    weight = head.classifier.weight[:, :, 0, 0]
    bias = head.classifier.bias
    labels = np.empty(pixels.shape[0], dtype=np.uint16)
    logits_file = None
    logits_flat = None
    if logits_path is not None:
        logits_file = np.lib.format.open_memmap(
            logits_path,
            mode="w+",
            dtype=np.float16,
            shape=(head.num_classes, height, width),
        )
        logits_flat = logits_file.reshape(head.num_classes, -1)

    start = 0
    active_chunk_size = int(chunk_size)
    while start < pixels.shape[0]:
        end = min(start + active_chunk_size, pixels.shape[0])
        try:
            logits = F.linear(pixels[start:end], weight, bias)
        except torch.cuda.OutOfMemoryError:
            if active_chunk_size <= 1_024:
                raise
            active_chunk_size = max(active_chunk_size // 2, 1_024)
            torch.cuda.empty_cache()
            continue
        labels[start:end] = (
            logits.argmax(dim=1).cpu().numpy().astype(np.uint16, copy=False)
        )
        if logits_flat is not None:
            logits_flat[:, start:end] = (
                logits.transpose(0, 1).half().cpu().numpy()
            )
        del logits
        start = end

    if logits_file is not None:
        logits_file.flush()
        del logits_flat
        del logits_file
    return labels.reshape(height, width)


def select_renderer(name: str):
    if name == "radegs":
        from gaussian_renderer.radegs import render_radegs

        return render_radegs
    from gaussian_renderer.ours import render_ours

    return render_ours


def _make_directories(
    root: Path,
    *,
    split: str,
    output_profile: str,
) -> dict[str, Path]:
    names = [
        *GAGA_COMMON_DIRECTORIES,
        *GW_IMAGE_DIRECTORIES,
    ]
    if split == "train":
        names.extend(GAGA_TRAIN_DIRECTORIES)
    if output_profile == "full":
        names.extend(GW_NUMERIC_DIRECTORIES)
    directories = {name: root / name for name in dict.fromkeys(names)}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _save_numeric_outputs(
    package: dict,
    features: torch.Tensor,
    directories: dict[str, Path],
    stem: str,
) -> None:
    expected_depth = package["expected_depth"]
    median_depth = package["median_depth"]
    np.save(directories["depth_npy"] / f"{stem}.npy", median_depth.cpu().numpy())
    np.save(
        directories["expected_depth_npy"] / f"{stem}.npy",
        expected_depth.cpu().numpy(),
    )
    np.save(
        directories["median_depth_npy"] / f"{stem}.npy",
        median_depth.cpu().numpy(),
    )
    for key in ("expected_coord", "median_coord"):
        if package[key] is not None:
            torch.save(package[key].float().cpu(), directories[key] / f"{stem}.pt")
    np.save(
        directories["normal_npy"] / f"{stem}.npy",
        package["normal"].float().cpu().numpy(),
    )
    alpha = package.get("mask")
    if alpha is None:
        alpha = torch.ones_like(median_depth)
    np.save(directories["alpha_npy"] / f"{stem}.npy", alpha.float().cpu().numpy())
    torch.save(package["radii"].cpu(), directories["radii"] / f"{stem}.pt")
    auxiliary = {
        key: value.detach().cpu()
        for key, value in package.items()
        if torch.is_tensor(value)
        and key not in {"render", "semantic_features", "radii"}
    }
    torch.save(auxiliary, directories["renderer_aux"] / f"{stem}.pt")
    torch.save(features.half().cpu(), directories["semantic_features"] / f"{stem}.pt")


@torch.inference_mode()
def render_camera_set(
    *,
    split: str,
    cameras,
    output_root: Path,
    iteration: int,
    gaussians: GaussianModel,
    head: SemanticHead,
    render,
    pipe,
    background: torch.Tensor,
    observations: GagaObservationStore | None,
    output_profile: str,
    class_chunk_size: int,
) -> list[dict]:
    root = output_root / split / f"ours_{iteration}"
    directories = _make_directories(
        root,
        split=split,
        output_profile=output_profile,
    )
    colors = gaga_palette(head.num_classes)
    manifest = []
    for index, camera in enumerate(tqdm(cameras, desc=f"Render {split}")):
        package = render(
            camera,
            gaussians,
            pipe,
            background,
            render_semantics=True,
            require_coord=False,
            require_depth=True,
        )
        features = package["semantic_features"]
        stem = Path(camera.image_name).stem
        logits_path = (
            directories["semantic_logits"] / f"{stem}.npy"
            if output_profile == "full"
            else None
        )
        labels = classify_features_chunked(
            head,
            features,
            chunk_size=class_chunk_size,
            logits_path=logits_path,
        )
        predicted_color = colors[labels]

        rendered = _rgb_image(package["render"])
        ground_truth = _rgb_image(camera.original_image[:3])
        rendered.save(directories["renders"] / f"{stem}.png")
        ground_truth.save(directories["gt"] / f"{stem}.png")
        ground_truth.save(directories["ground_truth"] / f"{stem}.png")
        feature_pca_image(features).save(
            directories["objects_feature16"] / f"{stem}.png"
        )
        Image.fromarray(predicted_color).save(
            directories["objects_pred"] / f"{stem}.png"
        )
        _save_label_image(labels, directories["objects_test"] / f"{stem}.png")
        _save_label_image(labels, directories["semantic_labels"] / f"{stem}.png")
        Image.fromarray(predicted_color).save(
            directories["semantic_color"] / f"{stem}.png"
        )

        expected_depth = package["expected_depth"]
        median_depth = package["median_depth"]
        _scalar_image(median_depth).save(directories["depth"] / f"{stem}.png")
        _scalar_image(expected_depth).save(
            directories["expected_depth"] / f"{stem}.png"
        )
        _scalar_image(median_depth).save(
            directories["median_depth"] / f"{stem}.png"
        )
        _rgb_image(package["normal"] * 0.5 + 0.5).save(
            directories["normal"] / f"{stem}.png"
        )
        alpha = package.get("mask")
        if alpha is None:
            alpha = torch.ones_like(median_depth)
        _unit_scalar_image(alpha).save(directories["alpha"] / f"{stem}.png")

        if split == "train":
            if observations is None:
                raise ValueError(
                    "Train rendering requires --semantic_masks for Gaga GT outputs"
                )
            observation = observations.load(
                camera.image_name,
                camera.image_height,
                camera.image_width,
            )
            gt_labels = observation.labels.numpy()
            _save_label_image(
                gt_labels,
                directories["gt_objects"] / f"{stem}.png",
            )
            valid_labels = np.where(observation.valid.numpy(), gt_labels, 0)
            Image.fromarray(colors[valid_labels]).save(
                directories["gt_objects_color"] / f"{stem}.png"
            )

        if output_profile == "full":
            _save_numeric_outputs(
                package,
                features,
                directories,
                stem,
            )
        manifest.append(
            {
                "index": index,
                "image_name": camera.image_name,
                "stem": stem,
            }
        )

    with (root / "manifest.json").open("w", encoding="utf8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


@torch.inference_mode()
def render_sets(dataset, pipe, args) -> None:
    gaussians = GaussianModel(
        dataset.sh_degree,
        semantic_dim=16,
        use_unbounded_opacity=dataset.use_unbounded_opacity,
    )
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.load_iteration,
        shuffle=False,
    )
    payload = torch.load(args.semantic_checkpoint, map_location="cpu")
    checkpoint_iteration = int(payload["iteration"])
    if checkpoint_iteration != scene.loaded_iter:
        raise ValueError(
            "Semantic checkpoint and PLY iterations differ: "
            f"{checkpoint_iteration} vs {scene.loaded_iter}"
        )
    head = SemanticHead(16, int(payload["num_classes"])).cuda().eval()
    load_semantic_checkpoint(
        args.semantic_checkpoint,
        head=head,
        gaussian_model=gaussians,
    )
    render = select_renderer(args.rasterizer)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    mask_dir = args.semantic_masks
    if mask_dir is None:
        mask_dir = payload.get("metadata", {}).get("mask_dir")
    observations = (
        GagaObservationStore(mask_dir, require_all=True)
        if mask_dir is not None
        else None
    )
    requested_splits = (
        ("train", "test") if args.split == "all" else (args.split,)
    )
    camera_sets = {
        "train": scene.getTrainCameras(),
        "test": scene.getTestCameras(),
    }
    output_root = Path(args.output)
    split_manifests = {}
    for split in requested_splits:
        cameras = camera_sets[split]
        if not cameras:
            print(f"[INFO] No {split} cameras; skipping this split.")
            continue
        split_manifests[split] = render_camera_set(
            split=split,
            cameras=cameras,
            output_root=output_root,
            iteration=scene.loaded_iter,
            gaussians=gaussians,
            head=head,
            render=render,
            pipe=pipe,
            background=background,
            observations=observations,
            output_profile=args.output_profile,
            class_chunk_size=args.class_chunk_size,
        )

    render_manifest = {
        "format_version": 1,
        "model_path": str(Path(dataset.model_path).resolve()),
        "semantic_checkpoint": str(Path(args.semantic_checkpoint).resolve()),
        "iteration": scene.loaded_iter,
        "resolution": dataset.resolution,
        "rasterizer": args.rasterizer,
        "output_profile": args.output_profile,
        "class_chunk_size": args.class_chunk_size,
        "num_classes": head.num_classes,
        "predicted_mask_dtype": "uint16",
        "splits": {
            split: {"count": len(entries)}
            for split, entries in split_manifests.items()
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "render_manifest.json").open(
        "w",
        encoding="utf8",
    ) as stream:
        json.dump(render_manifest, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render Gaussian Wrapping Gaga outputs")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--semantic_checkpoint", required=True)
    parser.add_argument("--semantic_masks")
    parser.add_argument("--output", required=True)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--rasterizer", choices=("radegs", "ours"), default="radegs")
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
    )
    parser.add_argument(
        "--output_profile",
        choices=("images", "full"),
        default="images",
        help="'images' writes Gaga-compatible and GW diagnostic PNGs; "
        "'full' additionally writes numerical tensors and chunked logits.",
    )
    parser.add_argument(
        "--class_chunk_size",
        type=int,
        default=32_768,
        help="Maximum pixels classified per CUDA chunk.",
    )
    parser.add_argument("--quiet", action="store_true")
    arguments = get_combined_args(parser)
    if arguments.class_chunk_size <= 0:
        parser.error("--class_chunk_size must be positive")
    safe_state(arguments.quiet)
    render_sets(model.extract(arguments), pipeline.extract(arguments), arguments)
