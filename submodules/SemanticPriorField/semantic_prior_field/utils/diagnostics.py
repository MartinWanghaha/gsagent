"""Training diagnostics: file-based intermediate-state recording.

Everything lands under ``<model_path>/diagnostics/`` so a training run can be
audited and mined offline for architecture decisions:

  diagnostics/
    scalars.jsonl        one record per logged iteration: every active loss
                         term, Gaussian count, SPF statistics
    events.jsonl         every discrete decision: densify / split / prune
                         events with before/after counts and kinds
    prior_field/         per-refresh instance tables (proxy type, fit
                         residual, weight, member count) + global stats
    images/iter_XXXXXX/  visualizations of the current training view:
                         render vs GT, depth, normals, alpha, semantic
                         prediction / GT / PCA features, per-pixel semantic
                         error, depth-normal error, boundary weights
    snapshots/           compressed per-Gaussian state (xyz, opacity, scale,
                         labels, confidence, prior type/weight, densify
                         multiplier, SH energy) for offline analysis

Every public method is failure-isolated: a diagnostics bug prints a warning
and never interrupts training.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image


def _never_raise(method):
    """Diagnostics must never kill a training run."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exception:  # noqa: BLE001 - deliberate isolation
            print(f"[WARNING] Diagnostics '{method.__name__}' failed: {exception!r}")
            return None

    return wrapper


# ---------------------------------------------------------------------------
# Visualization helpers (all return HxWx3 or HxW uint8 arrays)
# ---------------------------------------------------------------------------

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) or (1, H, W) or (H, W) float tensor in [0,1] -> uint8."""
    array = tensor.detach().float().cpu()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.permute(1, 2, 0).squeeze(-1)
    array = array.clamp(0.0, 1.0).numpy()
    return (array * 255.0).astype(np.uint8)


def _apply_colormap(normalized: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    """(H, W) in [0,1] -> uint8 RGB, matplotlib if available else grayscale."""
    try:
        import matplotlib

        try:
            cmap = matplotlib.colormaps[cmap_name]
        except AttributeError:  # matplotlib < 3.5
            cmap = matplotlib.cm.get_cmap(cmap_name)
        return (cmap(normalized)[..., :3] * 255.0).astype(np.uint8)
    except Exception:  # noqa: BLE001 - matplotlib optional
        gray = (normalized * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def colorize_depth(depth: torch.Tensor) -> np.ndarray:
    """(1, H, W) or (H, W) depth -> turbo colormap; invalid (<=0) is black."""
    array = depth.detach().float().cpu().squeeze().numpy()
    valid = array > 0
    normalized = np.zeros_like(array)
    if valid.any():
        low, high = np.percentile(array[valid], [2.0, 98.0])
        normalized[valid] = np.clip((array[valid] - low) / max(high - low, 1e-8), 0, 1)
    colored = _apply_colormap(normalized)
    colored[~valid] = 0
    return colored


def colorize_normal(normal: torch.Tensor) -> np.ndarray:
    """(3, H, W) unit normals -> RGB via (n+1)/2."""
    return tensor_to_image((normal + 1.0) * 0.5)


def colorize_scalar_map(scalar_map: torch.Tensor, vmax: Optional[float] = None) -> np.ndarray:
    """(H, W) non-negative scalar map -> inferno colormap."""
    array = scalar_map.detach().float().cpu().squeeze().numpy()
    if vmax is None:
        vmax = float(np.percentile(array, 99.0)) if array.size else 1.0
    normalized = np.clip(array / max(vmax, 1e-8), 0.0, 1.0)
    return _apply_colormap(normalized, "inferno")


def colorize_labels(labels: torch.Tensor, num_classes: Optional[int] = None) -> np.ndarray:
    """(H, W) long labels -> deterministic instance colors; <0 is gray."""
    from semantic.palette import gaga_palette

    array = labels.detach().cpu().long().numpy()
    if num_classes is None:
        num_classes = int(max(array.max() + 1, 1))
    palette = gaga_palette(max(num_classes, 1))
    clipped = np.clip(array, 0, palette.shape[0] - 1)
    colored = palette[clipped]
    colored[array < 0] = 128
    return colored


def feature_pca_rgb(features: torch.Tensor, max_samples: int = 65_536) -> np.ndarray:
    """(C, H, W) feature map -> PCA-to-RGB visualization."""
    channels, height, width = features.shape
    flat = features.detach().float().reshape(channels, -1).t()  # (P, C)
    sample = flat
    if flat.shape[0] > max_samples:
        sample = flat[torch.randperm(flat.shape[0], device=flat.device)[:max_samples]]
    mean = sample.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(sample - mean, q=3)
    projected = (flat - mean) @ v[:, :3]  # (P, 3)
    low = projected.quantile(0.01, dim=0)
    high = projected.quantile(0.99, dim=0)
    normalized = ((projected - low) / (high - low).clamp_min(1e-8)).clamp(0, 1)
    return (
        (normalized.reshape(height, width, 3).cpu().numpy() * 255.0).astype(np.uint8)
    )


@torch.no_grad()
def classify_feature_map(
    features: torch.Tensor,
    semantic_head,
    chunk_size: int = 65_536,
) -> torch.Tensor:
    """(16, H, W) -> (H, W) argmax labels, chunked over pixels."""
    height, width = features.shape[-2:]
    flat = features.detach().reshape(features.shape[0], -1).t()  # (P, 16)
    weight = semantic_head.classifier.weight.view(
        semantic_head.num_classes, semantic_head.semantic_dim
    )
    bias = semantic_head.classifier.bias
    labels = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        labels[start:end] = (flat[start:end] @ weight.t() + bias).argmax(dim=-1)
    return labels.view(height, width)


@torch.no_grad()
def semantic_error_map(
    features: torch.Tensor,
    semantic_head,
    labels: torch.Tensor,
    ignore_label: int,
    chunk_size: int = 65_536,
) -> torch.Tensor:
    """(16, H, W) + (H, W) GT labels -> (H, W) per-pixel CE, chunked."""
    import math

    import torch.nn.functional as F

    height, width = features.shape[-2:]
    flat = features.detach().reshape(features.shape[0], -1).t()
    labels_flat = labels.reshape(-1).to(flat.device)
    weight = semantic_head.classifier.weight.view(
        semantic_head.num_classes, semantic_head.semantic_dim
    )
    bias = semantic_head.classifier.bias
    error = torch.zeros(flat.shape[0], device=flat.device)
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        error[start:end] = F.cross_entropy(
            flat[start:end] @ weight.t() + bias,
            labels_flat[start:end],
            ignore_index=ignore_label,
            reduction="none",
        )
    return (error / max(math.log(semantic_head.num_classes), 1.0)).view(height, width)


# ---------------------------------------------------------------------------
# Diagnostics recorder
# ---------------------------------------------------------------------------

class TrainingDiagnostics:
    def __init__(
        self,
        model_path: str,
        scalar_interval: int = 10,
        image_interval: int = 1000,
        snapshot_interval: int = 5000,
    ):
        self.root = Path(model_path) / "diagnostics"
        self.images_dir = self.root / "images"
        self.prior_field_dir = self.root / "prior_field"
        self.snapshots_dir = self.root / "snapshots"
        for directory in (self.root, self.images_dir, self.prior_field_dir, self.snapshots_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.scalars_path = self.root / "scalars.jsonl"
        self.events_path = self.root / "events.jsonl"
        self.scalar_interval = max(int(scalar_interval), 1)
        self.image_interval = max(int(image_interval), 1)
        self.snapshot_interval = int(snapshot_interval)
        self._last_prior_field_dump = -1
        print(f"[INFO] Diagnostics enabled: {self.root}")
        print(f"        > scalars every {self.scalar_interval}, images every "
              f"{self.image_interval}, snapshots every {self.snapshot_interval or 'never'}.")

    # --- scheduling -------------------------------------------------------

    def wants_scalars(self, iteration: int) -> bool:
        return iteration % self.scalar_interval == 0

    def wants_images(self, iteration: int) -> bool:
        return iteration % self.image_interval == 0

    def wants_snapshot(self, iteration: int) -> bool:
        return self.snapshot_interval > 0 and iteration % self.snapshot_interval == 0

    # --- scalar / event streams ------------------------------------------

    @staticmethod
    def _to_scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().item())
        return value

    @_never_raise
    def log_scalars(self, iteration: int, scalars: Dict) -> None:
        record = {"iteration": int(iteration)}
        record.update({key: self._to_scalar(value) for key, value in scalars.items()})
        with self.scalars_path.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(record) + "\n")

    @_never_raise
    def log_event(self, iteration: int, kind: str, **payload) -> None:
        record = {"iteration": int(iteration), "kind": kind}
        record.update({key: self._to_scalar(value) for key, value in payload.items()})
        with self.events_path.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(record) + "\n")

    # --- image dumps ------------------------------------------------------

    @_never_raise
    def dump_training_view(
        self,
        iteration: int,
        viewpoint_cam,
        render_pkg: Dict,
        gt_image: torch.Tensor,
        semantic_head=None,
        observation=None,
        ignore_label: int = -1,
        boundary_weight_map: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        """Visualize every intermediate buffer of the current training view."""
        images: Dict[str, np.ndarray] = {}
        images["render"] = tensor_to_image(render_pkg["render"])
        images["gt"] = tensor_to_image(gt_image)

        median_depth = render_pkg.get("median_depth")
        if median_depth is not None:
            images["depth_median"] = colorize_depth(median_depth)
        expected_depth = render_pkg.get("expected_depth")
        if expected_depth is not None and expected_depth is not median_depth:
            images["depth_expected"] = colorize_depth(expected_depth)
        if render_pkg.get("normal") is not None:
            images["normal"] = colorize_normal(render_pkg["normal"])
        if render_pkg.get("alpha") is not None:
            images["alpha"] = tensor_to_image(render_pkg["alpha"])

        # Depth-normal consistency error map (the quantity the 7000+ loss sees)
        if median_depth is not None and render_pkg.get("normal") is not None:
            from utils.geometry_utils import depth_to_normal_with_mask

            depth_normal, valid = depth_to_normal_with_mask(viewpoint_cam, median_depth)
            error = 1.0 - (render_pkg["normal"] * depth_normal).sum(dim=0)
            error = torch.where(valid.squeeze(), error, torch.zeros_like(error))
            images["depth_normal_error"] = colorize_scalar_map(error, vmax=2.0)

        # Semantic buffers
        semantic_features = render_pkg.get("semantic_features")
        if semantic_features is not None:
            images["semantic_pca"] = feature_pca_rgb(semantic_features)
            if semantic_head is not None:
                predicted = classify_feature_map(semantic_features, semantic_head)
                images["semantic_pred"] = colorize_labels(predicted, num_classes)
                if observation is not None:
                    error = semantic_error_map(
                        semantic_features, semantic_head,
                        observation.labels, ignore_label,
                    )
                    error = error * observation.confidence.to(error.device)
                    images["semantic_error"] = colorize_scalar_map(error, vmax=1.0)
        if observation is not None:
            images["semantic_gt"] = colorize_labels(observation.labels, num_classes)
            images["semantic_confidence"] = tensor_to_image(observation.confidence)
        if boundary_weight_map is not None:
            images["boundary_weight"] = tensor_to_image(boundary_weight_map)

        output_dir = self.images_dir / f"iter_{iteration:06d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, array in images.items():
            Image.fromarray(array).save(output_dir / f"{name}.png")
        (output_dir / "meta.json").write_text(
            json.dumps({"iteration": iteration, "view": viewpoint_cam.image_name}),
            encoding="utf8",
        )

    # --- Semantic Prior Field state --------------------------------------

    @_never_raise
    def maybe_dump_prior_field(self, iteration: int, prior_field) -> None:
        """Dump the instance table right after a refresh (once per refresh)."""
        if not prior_field.valid:
            return
        if prior_field.last_refresh_iteration != iteration:
            return
        if self._last_prior_field_dump == iteration:
            return
        self._last_prior_field_dump = iteration

        from semantic.prior_field import PRIOR_TYPE_NAMES

        instances = []
        for label, instance in sorted(prior_field.instances.items()):
            instances.append(
                {
                    "label": int(label),
                    "type": instance.type_name(),
                    "n_gaussians": int(instance.n_gaussians),
                    "fit_residual": float(instance.fit_residual),
                    "weight": float(instance.weight),
                    "extent": float(instance.extent),
                }
            )
        labels = prior_field.labels
        confidence = prior_field.label_confidence
        record = {
            "iteration": int(iteration),
            "n_gaussians": int(labels.shape[0]),
            "labelled_fraction": float((labels >= 0).float().mean().item()),
            "mean_label_confidence": float(confidence.mean().item()),
            "mean_prior_weight": float(prior_field.prior_weight.mean().item()),
            "prior_type_counts": {
                name: int((prior_field.prior_type == type_id).sum().item())
                for type_id, name in PRIOR_TYPE_NAMES.items()
            },
            "instances": instances,
        }
        path = self.prior_field_dir / f"iter_{iteration:06d}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf8")
        self.log_event(
            iteration, "prior_field_refresh",
            n_instances=len(instances),
            labelled_fraction=record["labelled_fraction"],
        )

    # --- per-Gaussian snapshots ------------------------------------------

    @_never_raise
    def dump_snapshot(self, iteration: int, gaussians, prior_field=None) -> None:
        """Compressed per-Gaussian state for offline architecture analysis."""
        data: Dict[str, np.ndarray] = {
            "xyz": gaussians._xyz.detach().cpu().numpy().astype(np.float32),
            "opacity": gaussians.get_opacity.detach().cpu().numpy().astype(np.float32),
            "scaling": gaussians.get_scaling.detach().cpu().numpy().astype(np.float32),
        }
        features_rest = getattr(gaussians, "_features_rest", None)
        if features_rest is not None and features_rest.numel() > 0:
            data["sh_rest_energy"] = (
                features_rest.detach().flatten(1).norm(dim=-1).cpu().numpy().astype(np.float32)
            )
        if getattr(gaussians, "use_gaussian_features", False):
            data["normals"] = (
                gaussians.convert_features_to_normals(normalize=True)
                .detach().cpu().numpy().astype(np.float32)
            )
        if (
            prior_field is not None
            and prior_field.valid
            and prior_field.labels is not None
            and prior_field.labels.shape[0] == data["xyz"].shape[0]
        ):
            data["labels"] = prior_field.labels.cpu().numpy().astype(np.int32)
            data["label_confidence"] = prior_field.label_confidence.cpu().numpy().astype(np.float32)
            data["prior_type"] = prior_field.prior_type.cpu().numpy().astype(np.uint8)
            data["prior_weight"] = prior_field.prior_weight.cpu().numpy().astype(np.float32)
            data["densify_multiplier"] = prior_field.densify_multiplier.cpu().numpy().astype(np.float32)
        path = self.snapshots_dir / f"iter_{iteration:06d}.npz"
        np.savez_compressed(path, **data)
        print(f"[INFO] Diagnostics snapshot saved: {path} ({data['xyz'].shape[0]} Gaussians)")
