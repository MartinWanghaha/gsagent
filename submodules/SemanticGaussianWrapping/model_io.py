"""Shared experiment construction and trained-model loading."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import re
from typing import Any, Literal, Mapping
import warnings

import torch

from scene import GaussianModel, Scene
from scene.gaussian_model import SURFACE_INFERENCE_ATTRIBUTES
from semantic import GaussianNeighborIndex, SemanticSurfaceField
from utils.config_utils import load_config, to_namespace, validate_config


def resolve_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA is unavailable; using the reference CPU backend", RuntimeWarning, stacklevel=2)
        return torch.device("cpu")
    return device


def dataset_namespace(
    config: dict[str, Any],
    source_path: str | Path,
    model_path: str | Path,
    device: str | torch.device,
) -> Namespace:
    model = config["model"]
    data = config["data"]
    semantic = config.get("semantic", {})
    return Namespace(
        sh_degree=int(model["sh_degree"]),
        semantic_dim=int(model["semantic_dim"]),
        geometry_experts=int(model["geometry_experts"]),
        source_path=str(Path(source_path).resolve()),
        model_path=str(Path(model_path).resolve()),
        images=str(data.get("images", "images")),
        resolution=int(data.get("resolution", -1)),
        white_background=bool(model.get("white_background", False)),
        data_device=str(resolve_device(data.get("data_device", "cpu"))),
        eval=bool(data.get("eval", False)),
        llffhold=int(data.get("holdout", data.get("llffhold", 8))),
        semantic_path=data.get("semantic_path", data.get("semantic_root", "sam_mask")),
        semantic_confidence_path=data.get("semantic_confidence", data.get("semantic_confidence_path", "")),
        semantic_boundary_path=data.get("semantic_boundary", data.get("semantic_boundary_path", "")),
        semantic_ignore_label=int(data.get("ignore_label", -1)),
        semantic_background_label=int(data.get("background_label", 0)),
        semantic_temperature=float(semantic.get("temperature", 0.1)),
        boundary_width=int(semantic.get("boundary_width", 2)),
        random_points=int(data.get("random_points", 100_000)),
    )


def optimization_namespace(config: dict[str, Any]) -> Namespace:
    values = dict(config["optimization"])
    values.setdefault("feature_lr", values.get("feature_dc_lr", 2.5e-3))
    values.setdefault("percent_dense", config["density"].get("clone_scale_ratio", 0.01))
    return to_namespace(values)


def pipeline_namespace(config: dict[str, Any]) -> Namespace:
    renderer = config.get("renderer", {})
    return Namespace(
        convert_SHs_python=bool(renderer.get("convert_SHs_python", False)),
        compute_cov3D_python=bool(renderer.get("compute_cov3D_python", False)),
        debug=bool(renderer.get("debug", False)),
        reference_chunk_size=int(renderer.get("reference_chunk_size", 32)),
        antialias_sigma=float(renderer.get("antialias_sigma", 0.3)),
    )


def build_neighbor_index(
    gaussians: GaussianModel,
    config: dict[str, Any],
) -> GaussianNeighborIndex:
    """Build the one spatial index shared by all geometry consumers."""

    surface = config["surface"]
    return GaussianNeighborIndex(
        gaussians,
        backend=str(surface["neighbor_backend"]),
        gaussian_chunk_size=int(surface["gaussian_chunk"]),
        query_chunk_size=int(surface["neighbor_query_chunk"]),
        max_distance_bytes=int(surface["max_distance_bytes"]),
        support_candidate_budget=int(surface["support_candidate_budget"]),
        support_routing_query_chunk=int(surface["support_routing_query_chunk"]),
        scipy_workers=int(surface["scipy_workers"]),
    )


def build_surface_field(
    gaussians: GaussianModel,
    config: dict[str, Any],
    neighbor_index: GaussianNeighborIndex | None = None,
) -> SemanticSurfaceField:
    surface = config["surface"]
    semantic = config["semantic"]
    neighbor_index = neighbor_index or build_neighbor_index(gaussians, config)
    return SemanticSurfaceField(
        gaussians,
        policy_bank=gaussians.policy_bank,
        k_neighbors=int(surface["field_neighbors"]),
        query_chunk_size=int(surface["query_chunk"]),
        gaussian_chunk_size=int(surface["gaussian_chunk"]),
        occupancy_iso=float(surface["occupancy_isovalue"]),
        semantic_decoder=gaussians.semantic_decoder,
        neighbor_index=neighbor_index,
        max_distance_bytes=int(surface["max_distance_bytes"]),
        neighbor_backend=str(surface["neighbor_backend"]),
        support_log_cutoff=float(surface["support_log_cutoff"]),
        support_candidate_budget=int(surface["support_candidate_budget"]),
        support_routing_query_chunk=int(surface["support_routing_query_chunk"]),
        scipy_workers=int(surface["scipy_workers"]),
        region_top_k=int(semantic["region_top_k"]),
        region_decode_chunk_size=int(semantic["region_decode_chunk_size"]),
        region_candidate_neighbors=int(surface["region_candidate_neighbors"]),
        region_min_membership=float(surface["region_min_membership"]),
    )


def _resolved_config_path(model_path: Path) -> Path:
    for name in ("config.yaml", "resolved_config.yaml"):
        candidate = model_path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no resolved config.yaml found in {model_path}")


def _checkpoint_iteration_numbers(model_path: Path) -> set[int]:
    values: set[int] = set()
    for checkpoint in model_path.glob("chkpnt*.pth"):
        match = re.fullmatch(r"chkpnt(\d+)\.pth", checkpoint.name)
        if match:
            values.add(int(match.group(1)))
    return values


def _point_cloud_iteration_numbers(model_path: Path) -> set[int]:
    values: set[int] = set()
    point_cloud = model_path / "point_cloud"
    if point_cloud.is_dir():
        for path in point_cloud.glob("iteration_*"):
            match = re.fullmatch(r"iteration_(\d+)", path.name)
            if match and (path / "point_cloud.ply").is_file():
                values.add(int(match.group(1)))
    return values


def _iteration_numbers(model_path: Path) -> set[int]:
    return _checkpoint_iteration_numbers(model_path) | _point_cloud_iteration_numbers(
        model_path
    )


def available_iterations(model_path: str | Path) -> list[int]:
    """Return full-model iterations, falling back to PLY-only experiments."""

    root = Path(model_path)
    checkpoints = _checkpoint_iteration_numbers(root)
    return sorted(checkpoints or _point_cloud_iteration_numbers(root))


def resolve_iteration(model_path: str | Path, iteration: int = -1) -> int:
    root = Path(model_path)
    values = _iteration_numbers(root)
    if iteration == -1:
        preferred = _checkpoint_iteration_numbers(root)
        if not preferred:
            preferred = _point_cloud_iteration_numbers(root)
        if not preferred:
            raise FileNotFoundError(f"no checkpoint or point-cloud iteration below {model_path}")
        return max(preferred)
    if iteration not in values:
        raise FileNotFoundError(f"iteration {iteration} does not exist below {model_path}")
    return int(iteration)


def load_inference_gaussian_state(
    checkpoint_path: str | Path,
    *,
    scope: Literal["full", "surface"] = "full",
) -> dict[str, Any]:
    """Load only the Gaussian inference snapshot, always through CPU memory.

    Native training checkpoints also contain Adam moments, density windows,
    camera stacks, RNG state, and feedback-mesh caches.  None of those belong
    in rendering or offline extraction.  A shallow snapshot copy deliberately
    drops the Gaussian optimizer too; tensor storage is still shared until the
    full training checkpoint goes out of scope, so no second CPU copy is made.
    ``GaussianModel.restore(..., training_args=None)`` then transfers only the
    registry and semantic-decoder tensors to the requested runtime device.
    """

    load_options = {"map_location": "cpu", "weights_only": True}
    try:
        # mmap keeps unselected Adam/density/cache tensor pages out of host RAM
        # while the shallow Gaussian-only view is assembled.
        checkpoint = torch.load(checkpoint_path, mmap=True, **load_options)
    except (TypeError, RuntimeError) as error:
        # PyTorch before mmap support raises TypeError; legacy non-zip archives
        # raise a targeted RuntimeError. Corrupt/unrelated failures still surface.
        if "mmap" not in str(error).lower():
            raise
        checkpoint = torch.load(checkpoint_path, **load_options)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {checkpoint_path}")
    gaussian_state = checkpoint.get("gaussians", checkpoint)
    if not isinstance(gaussian_state, Mapping) or "registry" not in gaussian_state:
        raise ValueError(f"checkpoint has no Gaussian inference state: {checkpoint_path}")
    if scope not in {"full", "surface"}:
        raise ValueError("inference scope must be 'full' or 'surface'")
    inference_state = dict(gaussian_state)
    if scope == "surface":
        registry = gaussian_state["registry"]
        if not isinstance(registry, Mapping):
            raise ValueError("Gaussian registry snapshot must be a mapping")
        tensors = registry.get("tensors")
        specs = registry.get("specs")
        if not isinstance(tensors, Mapping) or not isinstance(specs, list):
            raise ValueError("Gaussian registry snapshot is incomplete")
        missing = [
            name for name in SURFACE_INFERENCE_ATTRIBUTES if name not in tensors
        ]
        if missing:
            raise ValueError(
                "surface inference checkpoint is missing attributes: "
                + ", ".join(missing)
            )
        selected = set(SURFACE_INFERENCE_ATTRIBUTES)
        inference_state["registry"] = {
            **dict(registry),
            "specs": [
                dict(spec)
                for spec in specs
                if isinstance(spec, Mapping) and spec.get("name") in selected
            ],
            "tensors": {
                name: tensors[name] for name in SURFACE_INFERENCE_ATTRIBUTES
            },
        }
    inference_state["optimizer"] = None
    del gaussian_state
    del checkpoint
    return inference_state


def load_trained_scene(
    model_path: str | Path,
    iteration: int = -1,
    device: str | torch.device = "cuda",
    *,
    with_surface_field: bool = True,
    inference_scope: Literal["full", "surface"] = "full",
) -> dict[str, Any]:
    """Reconstruct cameras and an inference-only Gaussian model.

    ``with_surface_field=False`` is the lean rendering path used by the legacy
    renderer-opacity wrapping extractor. Training-consistent mesh extraction
    must request ``with_surface_field=True`` so its zero set is identical to the
    field optimized by surface consistency and mesh feedback.
    """

    root = Path(model_path).resolve()
    config = load_config(_resolved_config_path(root))
    validate_config(config)
    runtime = config.get("runtime", config.get("_runtime", {}))
    source_path = runtime.get("source_path")
    if not source_path:
        raise ValueError("resolved config does not contain runtime.source_path")
    target_device = resolve_device(device)
    selected_iteration = resolve_iteration(root, iteration)
    dataset = dataset_namespace(config, source_path, root, target_device)
    gaussians = GaussianModel(
        int(config["model"]["sh_degree"]),
        int(config["model"]["semantic_dim"]),
        int(config["model"]["geometry_experts"]),
        target_device,
        confidence_floor=float(
            config.get("semantic", {}).get("confidence_floor", 0.05)
        ),
    )
    checkpoint_path = root / f"chkpnt{selected_iteration}.pth"
    ply_path = root / "point_cloud" / f"iteration_{selected_iteration}" / "point_cloud.ply"
    # A self-contained checkpoint is sufficient even if a user retained no PLY
    # snapshot. Scene still reads cameras/normalization before the registry is
    # restored from the checkpoint below.
    scene = Scene(
        dataset,
        gaussians,
        # Loading the matching multi-million-point PLY before restoring a
        # self-contained checkpoint creates a needless duplicate device model.
        load_iteration=(
            selected_iteration
            if not checkpoint_path.is_file() and ply_path.is_file()
            else None
        ),
        shuffle=False,
    )
    if checkpoint_path.is_file():
        gaussian_state = load_inference_gaussian_state(
            checkpoint_path,
            scope=inference_scope,
        )
        gaussians.restore(gaussian_state, training_args=None)
        del gaussian_state
    else:
        # Scene configures a decoder from dataset cardinality before loading a
        # PLY, but PLY has no global decoder weights. Keeping that freshly
        # initialized module would silently emit random semantic IDs.
        gaussians.semantic_decoder = None
        warnings.warn(
            "loading PLY without a checkpoint: discrete semantic output is unavailable",
            RuntimeWarning,
            stacklevel=2,
        )
    gaussians.eval()
    neighbor_index = None
    surface_field = None
    if with_surface_field:
        neighbor_index = build_neighbor_index(gaussians, config)
        surface_field = build_surface_field(gaussians, config, neighbor_index).to(target_device)
        surface_field.eval()
    return {
        "scene": scene,
        "gaussians": gaussians,
        "surface_field": surface_field,
        "neighbor_index": neighbor_index,
        "semantic_decoder": gaussians.semantic_decoder,
        "pipeline": pipeline_namespace(config),
        "config": config,
        "checkpoint_path": checkpoint_path if checkpoint_path.is_file() else None,
        "iteration": selected_iteration,
        "device": target_device,
    }


__all__ = [
    "available_iterations",
    "build_neighbor_index",
    "build_surface_field",
    "dataset_namespace",
    "load_inference_gaussian_state",
    "load_trained_scene",
    "optimization_namespace",
    "pipeline_namespace",
    "resolve_device",
    "resolve_iteration",
]
