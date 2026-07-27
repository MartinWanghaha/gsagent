"""Render RGB, semantic, depth, alpha, and normals from a trained model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch

from gaussian_renderer import render
from model_io import available_iterations, load_trained_scene
from utils.image_utils import composite_background
from utils.io_utils import save_array, save_gray16, save_labels, save_rgb


def _ground_truth_image(camera, background: torch.Tensor) -> torch.Tensor:
    """Match the alpha/background composition used by the training engine."""

    return composite_background(
        camera.original_image, getattr(camera, "gt_mask", None), background
    )


def _has_semantic_ground_truth(camera) -> bool:
    """Return whether a camera contains at least one supervised semantic pixel."""

    semantic_ids = getattr(camera, "semantic_ids", None)
    if semantic_ids is None:
        return False
    semantic_confidence = getattr(camera, "semantic_confidence", None)
    if semantic_confidence is not None:
        return bool((semantic_confidence > 0).any().item())
    ignore_label = int(getattr(camera, "ignore_label", -1))
    return bool((semantic_ids != ignore_label).any().item())


def _view_name_aliases(camera) -> set[str]:
    value = str(getattr(camera, "image_name", ""))
    if not value:
        return set()
    path = Path(value)
    return {value, path.name, path.stem}


def select_views(
    views: Iterable,
    requested_indices: Iterable[int] = (),
    requested_names: Iterable[str] = (),
) -> tuple[list, list[int], set[int], set[str]]:
    """Select a stable union while retaining original split-local indices."""

    cameras = list(views)
    indices = set(int(value) for value in requested_indices)
    names = {str(value) for value in requested_names}
    if not indices and not names:
        return cameras, list(range(len(cameras))), set(), set()
    selected: list = []
    selected_indices: list[int] = []
    matched_indices: set[int] = set()
    matched_names: set[str] = set()
    for index, camera in enumerate(cameras):
        aliases = _view_name_aliases(camera)
        index_match = index in indices
        name_matches = names.intersection(aliases)
        if index_match or name_matches:
            selected.append(camera)
            selected_indices.append(index)
            if index_match:
                matched_indices.add(index)
            matched_names.update(name_matches)
    return selected, selected_indices, matched_indices, matched_names


def _decode_semantic_labels(
    gaussians,
    semantic: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Materialize final discrete output labels from rendered embeddings.

    This is an intentional lossy output boundary. Internal semantic routing
    consumes soft memberships and must not call this helper.
    """

    if gaussians.semantic_decoder is None:
        raise RuntimeError(
            "semantic label output requires a trained scene semantic decoder"
        )
    if semantic.ndim != 3:
        raise ValueError("rendered semantic embedding must have shape [D,H,W]")
    if isinstance(chunk_size, bool) or int(chunk_size) < 1:
        raise ValueError("semantic_chunk_size must be positive")
    embedding = semantic.permute(1, 2, 0).reshape(-1, semantic.shape[0])
    decoder = gaussians.semantic_decoder
    labels = torch.empty(embedding.shape[0], dtype=torch.long, device=embedding.device)
    for start in range(0, embedding.shape[0], chunk_size):
        stop = min(start + chunk_size, embedding.shape[0])
        selected = embedding[start:stop].float()
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            logits = decoder(selected)
        if logits.ndim != 2 or logits.shape[0] != selected.shape[0]:
            raise ValueError("semantic decoder must return logits with shape [P,C]")
        labels[start:stop] = logits.argmax(-1)
    return labels.reshape(semantic.shape[-2:])


@torch.no_grad()
def render_set(
    output_root: Path,
    name: str,
    iteration: int,
    views,
    gaussians,
    pipeline,
    background: torch.Tensor,
    backend: str,
    num_classes: int,
    *,
    view_indices: Sequence[int] | None = None,
    total_views: int | None = None,
    semantic_chunk_size: int = 32768,
) -> None:
    views = list(views)
    if view_indices is None:
        view_indices = list(range(len(views)))
    if len(view_indices) != len(views):
        raise ValueError("view_indices must contain one original index per camera")
    if semantic_chunk_size < 1:
        raise ValueError("semantic_chunk_size must be positive")
    semantic_output = int(num_classes) > 1
    if semantic_output and getattr(gaussians, "semantic_decoder", None) is None:
        raise RuntimeError(
            "semantic label output requires a trained scene semantic decoder"
        )
    method = output_root / name / f"ours_{iteration}"
    semantic_ground_truth = [_has_semantic_ground_truth(camera) for camera in views]
    directories = [
        "renders",
        "gt",
        "semantic_feature",
        "depth",
        "alpha",
        "normal",
    ]
    if semantic_output:
        directories.append("semantic_id")
    for directory in directories:
        (method / directory).mkdir(parents=True, exist_ok=True)
    if any(semantic_ground_truth):
        (method / "gt_semantic_id").mkdir(parents=True, exist_ok=True)
    rendered_names: list[str] = []
    for selected_index, camera in enumerate(views):
        index = int(view_indices[selected_index])
        stem = f"{index:05d}"
        rendered_names.append(str(getattr(camera, "image_name", stem)))
        package = render(camera, gaussians, pipeline, background, backend=backend)
        save_rgb(method / "renders" / f"{stem}.png", package["render"])
        save_rgb(method / "gt" / f"{stem}.png", _ground_truth_image(camera, background))
        save_array(method / "semantic_feature" / f"{stem}.npy", package["semantic"])
        save_array(method / "depth" / f"{stem}.npy", package["expected_depth"])
        save_gray16(method / "depth" / f"{stem}.png", package["expected_depth"])
        save_gray16(method / "alpha" / f"{stem}.png", package["alpha"], maximum=1.0)
        save_rgb(method / "normal" / f"{stem}.png", package["normal"] * 0.5 + 0.5)
        if semantic_ground_truth[selected_index]:
            save_array(method / "gt_semantic_id" / f"{stem}.npy", camera.semantic_ids)
        if semantic_output:
            labels = _decode_semantic_labels(
                gaussians,
                package["semantic"],
                semantic_chunk_size,
            )
            save_array(method / "semantic_id" / f"{stem}.npy", labels)
            save_labels(method / "semantic_id" / f"{stem}.png", labels)
    metadata = {
        "iteration": iteration,
        "views": len(views),
        "total_views": len(views) if total_views is None else int(total_views),
        "view_indices": [int(value) for value in view_indices],
        "view_names": rendered_names,
        "num_semantic_classes": num_classes,
        "semantic_label_output": semantic_output,
        "has_semantic_ground_truth": any(semantic_ground_truth),
    }
    (method / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")


def select_iterations(
    model_path: str | Path,
    requested: Iterable[int] = (),
    *,
    all_iterations: bool = False,
) -> list[int]:
    """Resolve repeated/all iteration selection without loading any model."""

    available = available_iterations(model_path)
    if not available:
        raise FileNotFoundError(f"no checkpoint or point-cloud iteration below {model_path}")
    values = list(dict.fromkeys(int(value) for value in requested))
    if all_iterations:
        if values:
            raise ValueError("explicit iterations cannot be combined with all_iterations")
        return available
    if not values or values == [-1]:
        return [available[-1]]
    if -1 in values or any(value < 0 for value in values):
        raise ValueError("iterations must be non-negative, or use -1 alone for latest")
    missing = sorted(set(values) - set(available))
    if missing:
        raise FileNotFoundError(
            "missing model iteration(s): " + ", ".join(str(value) for value in missing)
        )
    return values


def _render_iteration(args: argparse.Namespace, iteration: int) -> None:
    bundle = load_trained_scene(
        args.model_path,
        iteration,
        args.device,
        with_surface_field=False,
    )
    scene, gaussians = bundle["scene"], bundle["gaussians"]
    config, device = bundle["config"], bundle["device"]
    value = 1.0 if config["model"].get("white_background", False) else 0.0
    background = torch.full((3,), value, device=device)
    root = Path(args.model_path)
    requested_indices = set(args.view_index)
    requested_names = set(args.view_name)
    matched_indices: set[int] = set()
    matched_names: set[str] = set()
    split_views = []
    if not args.skip_train:
        all_views = scene.getTrainCameras()
        views, indices, index_matches, name_matches = select_views(
            all_views,
            requested_indices,
            requested_names,
        )
        split_views.append(("train", all_views, views, indices))
        matched_indices.update(index_matches)
        matched_names.update(name_matches)
    if not args.skip_test:
        all_views = scene.getTestCameras()
        views, indices, index_matches, name_matches = select_views(
            all_views,
            requested_indices,
            requested_names,
        )
        split_views.append(("test", all_views, views, indices))
        matched_indices.update(index_matches)
        matched_names.update(name_matches)
    missing_indices = sorted(requested_indices - matched_indices)
    missing_names = sorted(requested_names - matched_names)
    if missing_indices or missing_names:
        details = []
        if missing_indices:
            details.append("indices=" + ",".join(str(value) for value in missing_indices))
        if missing_names:
            details.append("names=" + ",".join(missing_names))
        raise ValueError(
            "requested views were not found in enabled splits: " + "; ".join(details)
        )

    semantic_chunk_size = int(
        config.get("semantic", {}).get("region_decode_chunk_size", 32768)
    )
    for split, all_views, views, indices in split_views:
        if not views:
            continue
        render_set(
            root,
            split,
            bundle["iteration"],
            views,
            gaussians,
            bundle["pipeline"],
            background,
            args.backend,
            scene.num_semantic_classes,
            view_indices=indices,
            total_views=len(all_views),
            semantic_chunk_size=semantic_chunk_size,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", "-m", required=True)
    parser.add_argument(
        "--iteration",
        action="append",
        type=int,
        default=[],
        help="Render one iteration; repeat for several. Default: latest.",
    )
    parser.add_argument(
        "--all-iterations",
        action="store_true",
        help="Render every loadable checkpoint/PLY iteration in ascending order.",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("auto", "cuda", "reference"), default="auto")
    parser.add_argument(
        "--view-index",
        action="append",
        type=int,
        default=[],
        help="Render one split-local camera index; repeat for multiple views.",
    )
    parser.add_argument(
        "--view-name",
        action="append",
        default=[],
        help="Render an image name, basename, or stem; repeat for multiple views.",
    )
    args = parser.parse_args(argv)
    if args.skip_train and args.skip_test:
        parser.error("--skip_train and --skip_test would render no split")
    if any(index < 0 for index in args.view_index):
        parser.error("--view-index values must be non-negative")
    if args.all_iterations and args.iteration:
        parser.error("--all-iterations cannot be combined with --iteration")
    try:
        iterations = select_iterations(
            args.model_path,
            args.iteration,
            all_iterations=args.all_iterations,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    for iteration in iterations:
        try:
            _render_iteration(args, iteration)
        except ValueError as error:
            parser.error(str(error))
        # Iterations are deliberately loaded one at a time. Release the prior
        # model before mmap/GPU storage for the next checkpoint is materialized.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
