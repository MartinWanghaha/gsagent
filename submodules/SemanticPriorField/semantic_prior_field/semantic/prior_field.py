"""Semantic Prior Field (SPF).

The SPF is the core abstraction of Semantic Prior Field Splatting: a live,
periodically refreshed map from the current Gaussian model to

  * per-Gaussian instance labels and label confidences, derived from the
    16D semantic embedding and the scene-level 1x1 classifier head;
  * per-instance geometric proxies (plane / quadric / thin / freeform),
    fitted online from the labelled Gaussian centers;
  * per-Gaussian prior quantities consumed by the regularizers and the
    adaptive density control: proxy normals, prior weights, densification
    threshold multipliers.

All priors are soft. An instance whose proxy fits poorly degrades to
"freeform" (no orientation prior), so the worst case is the unregularized
baseline. Nothing in this module touches CUDA: every consumer lives in
PyTorch on top of the unmodified rasterizers.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

PRIOR_NONE = 0
PRIOR_PLANAR = 1
PRIOR_QUADRIC = 2
PRIOR_THIN = 3

PRIOR_TYPE_NAMES = {
    PRIOR_NONE: "freeform",
    PRIOR_PLANAR: "planar",
    PRIOR_QUADRIC: "quadric",
    PRIOR_THIN: "thin",
}


def head_logits_per_gaussian(head, semantic_features: torch.Tensor) -> torch.Tensor:
    """Apply the scene-level 1x1 classifier to raw per-Gaussian embeddings.

    Mirrors the pixel path exactly: the Conv2d(16, C, 1) is a linear map,
    so per-Gaussian logits are ``features @ W^T + b``.
    """
    weight = head.classifier.weight.view(head.num_classes, head.semantic_dim)
    bias = head.classifier.bias
    return semantic_features @ weight.t() + bias


@dataclass
class PriorInstance:
    """Geometric proxy fitted to one semantic instance."""

    label: int
    prior_type: int
    weight: float
    n_gaussians: int
    fit_residual: float
    extent: float
    normal: Optional[torch.Tensor] = None  # (3,) for planar proxies
    quadric: Optional[torch.Tensor] = None  # (10,) coefficients, normalized frame
    quadric_center: Optional[torch.Tensor] = None  # (3,)
    quadric_scale: float = 1.0

    def type_name(self) -> str:
        return PRIOR_TYPE_NAMES[self.prior_type]


def fit_plane_ransac(
    points: torch.Tensor,
    n_iterations: int = 32,
    inlier_threshold: float = 0.01,
) -> tuple[torch.Tensor, float, float]:
    """Vectorized RANSAC plane fit.

    Args:
        points: (P, 3) instance Gaussian centers.
        n_iterations: number of random 3-point hypotheses.
        inlier_threshold: absolute point-to-plane distance for inliers.

    Returns:
        (normal (3,), inlier_ratio, mean_inlier_residual)
    """
    n_points = points.shape[0]
    if n_points < 3:
        return points.new_zeros(3), 0.0, float("inf")

    generator_idx = torch.randint(0, n_points, (n_iterations, 3), device=points.device)
    p0 = points[generator_idx[:, 0]]
    p1 = points[generator_idx[:, 1]]
    p2 = points[generator_idx[:, 2]]
    normals = torch.linalg.cross(p1 - p0, p2 - p0)  # (I, 3)
    norms = normals.norm(dim=-1, keepdim=True)
    valid = norms.squeeze(-1) > 1e-12
    if not valid.any():
        return points.new_zeros(3), 0.0, float("inf")
    normals = torch.where(valid.unsqueeze(-1), normals / norms.clamp_min(1e-12), normals)
    offsets = -(normals * p0).sum(dim=-1)  # (I,)

    distances = (points @ normals.t() + offsets.unsqueeze(0)).abs()  # (P, I)
    inliers = (distances < inlier_threshold) & valid.unsqueeze(0)
    counts = inliers.sum(dim=0)  # (I,)
    best = counts.argmax()
    best_inliers = inliers[:, best]
    inlier_ratio = counts[best].item() / n_points
    if counts[best] < 3:
        return points.new_zeros(3), 0.0, float("inf")

    # Least-squares refit on the consensus set: smallest principal axis.
    inlier_points = points[best_inliers]
    centered = inlier_points - inlier_points.mean(dim=0, keepdim=True)
    covariance = centered.t() @ centered / centered.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    residual = (centered @ normal).abs().mean().item()
    return normal, inlier_ratio, residual


def fit_quadric(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Algebraic quadric fit q(x) = 0 in a normalized frame.

    Coordinates are centered and scaled to unit RMS radius before fitting so
    the design matrix stays well conditioned.

    Returns:
        (coefficients (10,), center (3,), scale, mean_geometric_residual)
    """
    center = points.mean(dim=0)
    scale = (points - center).norm(dim=-1).mean().clamp_min(1e-8).item()
    x = (points - center) / scale
    design = torch.stack(
        [
            x[:, 0] * x[:, 0], x[:, 1] * x[:, 1], x[:, 2] * x[:, 2],
            x[:, 0] * x[:, 1], x[:, 0] * x[:, 2], x[:, 1] * x[:, 2],
            x[:, 0], x[:, 1], x[:, 2],
            torch.ones_like(x[:, 0]),
        ],
        dim=-1,
    )  # (P, 10)
    _, _, vh = torch.linalg.svd(design, full_matrices=False)
    coefficients = vh[-1]
    values = design @ coefficients
    gradients = quadric_gradient(coefficients, x)
    residual = (values.abs() / gradients.norm(dim=-1).clamp_min(1e-6)).mean().item()
    return coefficients, center, scale, residual


def quadric_gradient(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Analytic gradient of the quadric at normalized points x: (P, 3)."""
    a, b, c, d, e, f, g, h, i, _ = coefficients.unbind()
    gx = 2.0 * a * x[:, 0] + d * x[:, 1] + e * x[:, 2] + g
    gy = 2.0 * b * x[:, 1] + d * x[:, 0] + f * x[:, 2] + h
    gz = 2.0 * c * x[:, 2] + e * x[:, 0] + f * x[:, 1] + i
    return torch.stack([gx, gy, gz], dim=-1)


class SemanticPriorField:
    """Live per-Gaussian labels and per-instance geometric proxies.

    The field is refreshed every ``refresh_interval`` iterations and
    invalidated whenever the Gaussian set changes (densify / prune / reset),
    following the same cache discipline as the normal-field and MILo state.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.labels: Optional[torch.Tensor] = None  # (N,) long, -1 = unlabeled
        self.label_confidence: Optional[torch.Tensor] = None  # (N,) in [0, 1]
        self.prior_normals: Optional[torch.Tensor] = None  # (N, 3), zero rows = no prior
        self.prior_weight: Optional[torch.Tensor] = None  # (N,) soft gate
        self.prior_type: Optional[torch.Tensor] = None  # (N,) uint8 PRIOR_*
        self.densify_multiplier: Optional[torch.Tensor] = None  # (N,) in [m_thin, m_planar]
        self.instances: Dict[int, PriorInstance] = {}
        self.last_refresh_iteration: int = -1
        self._valid = False

    @property
    def valid(self) -> bool:
        return self._valid

    def invalidate(self) -> None:
        """Drop all cached per-Gaussian state (topology changed)."""
        self._valid = False
        self.labels = None
        self.label_confidence = None
        self.prior_normals = None
        self.prior_weight = None
        self.prior_type = None
        self.densify_multiplier = None
        self.instances = {}

    def needs_refresh(self, iteration: int) -> bool:
        if not self._valid:
            return True
        return (iteration - self.last_refresh_iteration) >= int(
            self.config.get("refresh_interval", 500)
        )

    @torch.no_grad()
    def refresh(self, gaussians, semantic_head, iteration: int) -> None:
        """Recompute labels, proxies and per-Gaussian prior quantities."""
        semantic_features = gaussians.get_semantic_features.detach()
        xyz = gaussians.get_xyz.detach()
        scaling = gaussians.get_scaling.detach()
        n_gaussians = xyz.shape[0]
        device = xyz.device

        # Chunked label derivation: with C classes the full (N, C) logits can
        # reach gigabytes (scene class counts run into the hundreds), so only
        # per-chunk logits are ever materialized.
        chunk_size = int(self.config.get("label_chunk_size", 262_144))
        labels_chunks = []
        confidence_chunks = []
        for start in range(0, n_gaussians, chunk_size):
            end = min(start + chunk_size, n_gaussians)
            logits = head_logits_per_gaussian(
                semantic_head, semantic_features[start:end]
            )
            probabilities = torch.softmax(logits, dim=-1)
            top2 = probabilities.topk(k=min(2, probabilities.shape[-1]), dim=-1)
            labels_chunks.append(top2.indices[:, 0].long())
            if top2.values.shape[-1] > 1:
                confidence_chunks.append(top2.values[:, 0] - top2.values[:, 1])
            else:
                confidence_chunks.append(top2.values[:, 0])
        labels = torch.cat(labels_chunks)
        confidence = torch.cat(confidence_chunks)

        min_confidence = float(self.config.get("min_label_confidence", 0.3))
        labels = torch.where(
            confidence >= min_confidence,
            labels,
            torch.full_like(labels, -1),
        )

        self.labels = labels
        self.label_confidence = confidence
        self.prior_normals = torch.zeros(n_gaussians, 3, device=device)
        self.prior_weight = torch.zeros(n_gaussians, device=device)
        self.prior_type = torch.zeros(n_gaussians, dtype=torch.uint8, device=device)
        self.densify_multiplier = torch.ones(n_gaussians, device=device)
        self.instances = {}

        self._fit_instances(xyz, scaling, labels, confidence)

        self.last_refresh_iteration = iteration
        self._valid = True

    @torch.no_grad()
    def _fit_instances(
        self,
        xyz: torch.Tensor,
        scaling: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
    ) -> None:
        config = self.config
        min_instance_gaussians = int(config.get("min_instance_gaussians", 512))
        max_instances = int(config.get("max_instances", 256))
        max_points = int(config.get("max_points_per_instance", 20_000))
        background_label = int(config.get("background_label", 0))
        skip_background = bool(config.get("skip_background_instance", True))

        unique_labels, counts = torch.unique(labels[labels >= 0], return_counts=True)
        if unique_labels.numel() == 0:
            return
        order = counts.argsort(descending=True)
        unique_labels = unique_labels[order][:max_instances]
        counts = counts[order][:max_instances]

        multiplier_planar = float(config.get("densify_multiplier_planar", 1.5))
        multiplier_thin = float(config.get("densify_multiplier_thin", 0.7))
        weight_sigma = float(config.get("prior_weight_sigma", 0.5))

        for label_value, count in zip(unique_labels.tolist(), counts.tolist()):
            if count < min_instance_gaussians:
                continue
            if skip_background and label_value == background_label:
                continue
            member_mask = labels == label_value
            member_idx = member_mask.nonzero(as_tuple=True)[0]
            if member_idx.shape[0] > max_points:
                keep = torch.randperm(member_idx.shape[0], device=xyz.device)[:max_points]
                sample_idx = member_idx[keep]
            else:
                sample_idx = member_idx
            points = xyz[sample_idx]
            extent = (points.max(dim=0).values - points.min(dim=0).values).norm().clamp_min(1e-8).item()
            instance_confidence = confidence[member_idx].mean().item()

            instance = self._fit_single_instance(
                label_value, points, scaling[sample_idx], extent, instance_confidence,
                n_members=member_idx.shape[0],
            )
            self.instances[label_value] = instance
            self._scatter_instance(instance, member_idx, xyz, confidence, weight_sigma)

            if instance.prior_type == PRIOR_PLANAR:
                self.densify_multiplier[member_idx] = multiplier_planar
            elif instance.prior_type == PRIOR_THIN:
                self.densify_multiplier[member_idx] = multiplier_thin

    @torch.no_grad()
    def _fit_single_instance(
        self,
        label_value: int,
        points: torch.Tensor,
        scaling: torch.Tensor,
        extent: float,
        instance_confidence: float,
        n_members: int,
    ) -> PriorInstance:
        config = self.config

        # Thin structures first: needle-like Gaussians (spokes, railings,
        # branches) must be protected from flattening and size pruning, and
        # neither a plane nor a quadric is a meaningful proxy for them.
        sorted_scales = scaling.sort(dim=-1, descending=True).values
        needle_ratio = sorted_scales[:, 0] / sorted_scales[:, 1].clamp_min(1e-8)
        median_needle_ratio = needle_ratio.median().item()
        centered = points - points.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered / max(extent, 1e-8))
        filament_score = (
            singular_values[0] / singular_values[1].clamp_min(1e-8)
        ).item() if singular_values.numel() >= 2 else 1.0
        thin_threshold = float(config.get("thin_anisotropy_threshold", 4.0))
        if median_needle_ratio > thin_threshold or filament_score > 2.0 * thin_threshold:
            return PriorInstance(
                label=label_value,
                prior_type=PRIOR_THIN,
                weight=instance_confidence,
                n_gaussians=n_members,
                fit_residual=0.0,
                extent=extent,
            )

        # Planar proxy: RANSAC with a relative inlier threshold.
        inlier_threshold = float(config.get("planar_inlier_threshold_rel", 0.01)) * extent
        normal, inlier_ratio, residual = fit_plane_ransac(
            points,
            n_iterations=int(config.get("ransac_iterations", 32)),
            inlier_threshold=inlier_threshold,
        )
        if inlier_ratio >= float(config.get("planar_inlier_ratio", 0.8)):
            relative_residual = residual / max(extent, 1e-8)
            return PriorInstance(
                label=label_value,
                prior_type=PRIOR_PLANAR,
                weight=instance_confidence,
                n_gaussians=n_members,
                fit_residual=relative_residual,
                extent=extent,
                normal=normal,
            )

        # Quadric proxy for smooth curved instances.
        if points.shape[0] >= int(config.get("quadric_min_points", 64)):
            coefficients, center, scale, quadric_residual = fit_quadric(points)
            relative_residual = quadric_residual * scale / max(extent, 1e-8)
            if relative_residual <= float(config.get("quadric_max_residual_rel", 0.02)):
                return PriorInstance(
                    label=label_value,
                    prior_type=PRIOR_QUADRIC,
                    weight=instance_confidence,
                    n_gaussians=n_members,
                    fit_residual=relative_residual,
                    extent=extent,
                    quadric=coefficients,
                    quadric_center=center,
                    quadric_scale=scale,
                )

        # Freeform: keep the label (boundaries, SH, density still apply),
        # but expose no orientation prior.
        return PriorInstance(
            label=label_value,
            prior_type=PRIOR_NONE,
            weight=0.0,
            n_gaussians=n_members,
            fit_residual=float("inf"),
            extent=extent,
        )

    @torch.no_grad()
    def _scatter_instance(
        self,
        instance: PriorInstance,
        member_idx: torch.Tensor,
        xyz: torch.Tensor,
        confidence: torch.Tensor,
        weight_sigma: float,
    ) -> None:
        self.prior_type[member_idx] = instance.prior_type
        if instance.prior_type == PRIOR_PLANAR:
            fit_gate = math.exp(-instance.fit_residual / weight_sigma)
            self.prior_normals[member_idx] = instance.normal.to(xyz.device)
            self.prior_weight[member_idx] = confidence[member_idx] * instance.weight * fit_gate
        elif instance.prior_type == PRIOR_QUADRIC:
            fit_gate = math.exp(-instance.fit_residual / weight_sigma)
            normalized = (xyz[member_idx] - instance.quadric_center) / instance.quadric_scale
            gradients = quadric_gradient(instance.quadric, normalized)
            normals = F.normalize(gradients, dim=-1, eps=1e-8)
            self.prior_normals[member_idx] = normals
            self.prior_weight[member_idx] = confidence[member_idx] * instance.weight * fit_gate
        # PRIOR_THIN and PRIOR_NONE expose no orientation prior; thin keeps
        # its type so density control and flattening exemption can see it.

    def summary(self) -> str:
        if not self._valid or not self.instances:
            return "SemanticPriorField: empty"
        type_counts: Dict[str, int] = {}
        for instance in self.instances.values():
            name = instance.type_name()
            type_counts[name] = type_counts.get(name, 0) + 1
        labelled = int((self.labels >= 0).sum().item()) if self.labels is not None else 0
        total = int(self.labels.shape[0]) if self.labels is not None else 0
        parts = ", ".join(f"{name}: {count}" for name, count in sorted(type_counts.items()))
        return (
            f"SemanticPriorField: {len(self.instances)} instances ({parts}), "
            f"{labelled}/{total} Gaussians labelled"
        )


@torch.no_grad()
def compute_boundary_weight_map(
    labels: torch.Tensor,
    ignore_label: int = -1,
    boundary_radius: int = 4,
    boundary_weight: float = 0.25,
) -> torch.Tensor:
    """Per-pixel down-weighting map around instance-label discontinuities.

    Depth discontinuities across instance boundaries are legitimate; the
    depth-normal consistency and multiview NCC losses should not smooth
    across them. Pixels within ``boundary_radius`` of a label change get
    weight ``boundary_weight``; interior pixels keep weight 1. Ignored
    pixels are treated as boundary-neutral (weight 1) so unlabeled regions
    keep the baseline behaviour.

    Args:
        labels: (H, W) long tensor of instance labels.
        ignore_label: pixels with this value never generate boundaries.
        boundary_radius: dilation radius in pixels.
        boundary_weight: weight assigned inside the boundary band.

    Returns:
        (H, W) float tensor in [boundary_weight, 1].
    """
    labels = labels.long()
    valid = labels != ignore_label
    edge = torch.zeros_like(labels, dtype=torch.bool)
    # 4-neighbourhood label discontinuities between mutually valid pixels;
    # both pixels of a discontinuous pair belong to the boundary band.
    horizontal = (labels[:, :-1] != labels[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    vertical = (labels[:-1, :] != labels[1:, :]) & valid[:-1, :] & valid[1:, :]
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical

    if boundary_radius > 1:
        kernel_size = 2 * int(boundary_radius) + 1
        edge = (
            F.max_pool2d(
                edge[None, None].float(),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )[0, 0]
            > 0
        )

    weight = torch.ones_like(labels, dtype=torch.float32)
    weight[edge] = float(boundary_weight)
    return weight


class BoundaryWeightCache:
    """Cache of per-view boundary weight maps keyed by image name and size.

    Masks are static during training, so each map is computed once.
    """

    def __init__(self, observations, config: Dict):
        self.observations = observations
        self.config = config
        self._cache: Dict[tuple, torch.Tensor] = {}

    def get(self, image_name: str, height: int, width: int, device) -> torch.Tensor:
        key = (image_name, height, width)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.to(device)
        observation = self.observations.load(image_name, height, width)
        weight = compute_boundary_weight_map(
            observation.labels,
            ignore_label=self.observations.ignore_label,
            boundary_radius=int(self.config.get("boundary_radius", 4)),
            boundary_weight=float(self.config.get("boundary_weight", 0.25)),
        )
        self._cache[key] = weight.cpu()
        return weight.to(device)
