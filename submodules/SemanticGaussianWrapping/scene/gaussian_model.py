"""Semantic Gaussian model with registry-driven topology mutation."""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from semantic.geometry_policy import SoftGeometryPolicyBank
    from semantic.region_membership import (
        SparseRegionMembership,
        decode_sparse_region_memberships,
    )
except ImportError:
    from ..semantic.geometry_policy import SoftGeometryPolicyBank
    from ..semantic.region_membership import (
        SparseRegionMembership,
        decode_sparse_region_memberships,
    )

from .gaussian_attributes import GaussianAttributeRegistry


SH_C0 = 0.28209479177387814

SURFACE_INFERENCE_ATTRIBUTES = (
    "xyz",
    "opacity",
    "scaling",
    "rotation",
    "semantic_embedding",
    "geometry_logits",
    "semantic_confidence",
    "propagated_semantic_confidence",
    "boundary_score",
    "geometry_error",
    "observation_count",
)


@dataclass(frozen=True)
class TopologyStamp:
    """Monotonic identity of one Gaussian topology state.

    ``gaussian_count`` alone cannot distinguish a prune-and-replace update
    whose net growth is zero.  The generation and cumulative churn counters
    therefore travel with optimizer and inference snapshots so asynchronous
    consumers can reason about exactly which topology they captured.
    """

    generation: int
    cumulative_topology_churn: int
    gaussian_count: int


def RGB2SH(rgb: Tensor) -> Tensor:
    return (rgb - 0.5) / SH_C0


def SH2RGB(sh: Tensor) -> Tensor:
    return sh * SH_C0 + 0.5


def inverse_sigmoid(value: Tensor) -> Tensor:
    value = value.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(value / (1.0 - value))


def quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    quaternion = F.normalize(quaternion, dim=-1, eps=1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def strip_symmetric(matrix: Tensor) -> Tensor:
    result = torch.empty((matrix.shape[0], 6), dtype=matrix.dtype, device=matrix.device)
    result[:, 0] = matrix[:, 0, 0]
    result[:, 1] = matrix[:, 0, 1]
    result[:, 2] = matrix[:, 0, 2]
    result[:, 3] = matrix[:, 1, 1]
    result[:, 4] = matrix[:, 1, 2]
    result[:, 5] = matrix[:, 2, 2]
    return result


def _selection_mask(selection: Tensor | Sequence[int] | None, size: int, device: torch.device) -> Tensor:
    if selection is None:
        return torch.zeros(size, dtype=torch.bool, device=device)
    value = torch.as_tensor(selection, device=device)
    if value.dtype == torch.bool:
        if value.ndim != 1 or value.numel() != size:
            raise ValueError(f"topology mask must have shape [{size}]")
        return value
    result = torch.zeros(size, dtype=torch.bool, device=device)
    indices = value.long().flatten()
    if indices.numel() and (indices.min() < 0 or indices.max() >= size):
        raise IndexError("Gaussian topology index out of range")
    result[indices] = True
    return result


def _exponential_lr(
    step: int,
    lr_init: float,
    lr_final: float,
    max_steps: int,
    delay_mult: float = 1.0,
    delay_steps: int = 0,
) -> float:
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    delay = 1.0
    if delay_steps > 0:
        delay = delay_mult + (1.0 - delay_mult) * math.sin(
            0.5 * math.pi * min(max(step / delay_steps, 0.0), 1.0)
        )
    t = min(max(step / max(max_steps, 1), 0.0), 1.0)
    return delay * math.exp(math.log(max(lr_init, 1e-30)) * (1 - t) + math.log(max(lr_final, 1e-30)) * t)


class SemanticDecoder(nn.Module):
    """Lightweight Gaga-compatible decoder over continuous embeddings."""

    def __init__(self, embedding_dim: int, num_classes: int, temperature: float = 1.0) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.temperature = float(temperature)

    def forward(self, embedding: Tensor) -> Tensor:
        return self.linear(F.normalize(embedding, dim=-1, eps=1e-8)) / max(self.temperature, 1e-4)


class GaussianModel(nn.Module):
    """Complete trainable 3DGS state, including semantics and evidence."""

    def __init__(
        self,
        sh_degree: int = 3,
        semantic_dim: int = 16,
        geometry_experts: int = 5,
        device: str | torch.device | None = None,
        confidence_floor: float = 0.05,
    ) -> None:
        super().__init__()
        if sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        if semantic_dim < 1:
            raise ValueError("semantic_dim must be positive")
        target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_sh_degree = int(sh_degree)
        self.active_sh_degree = 0
        self.semantic_dim = int(semantic_dim)
        self.geometry_experts = int(geometry_experts)
        self.spatial_lr_scale = 1.0
        self.percent_dense = 0.01
        self.topology_generation = 0
        self.cumulative_topology_churn = 0
        self.optimizer: torch.optim.Optimizer | None = None
        self._training_args: Any | None = None
        self.registry = GaussianAttributeRegistry()
        sh_rest = (self.max_sh_degree + 1) ** 2 - 1
        self.registry.register("xyz", torch.empty(0, 3, device=target), optimizer_group="xyz", role="render")
        self.registry.register("features_dc", torch.empty(0, 1, 3, device=target), optimizer_group="f_dc", role="render", storage_order="channel_sh")
        self.registry.register("features_rest", torch.empty(0, sh_rest, 3, device=target), optimizer_group="f_rest", role="render", storage_order="channel_sh")
        self.registry.register("opacity", torch.empty(0, 1, device=target), optimizer_group="opacity", role="render")
        self.registry.register("scaling", torch.empty(0, 3, device=target), optimizer_group="scaling", role="render")
        self.registry.register("rotation", torch.empty(0, 4, device=target), optimizer_group="rotation", role="render")
        self.registry.register("semantic_embedding", torch.empty(0, semantic_dim, device=target), optimizer_group="semantic", role="semantic", ply_prefix="semantic")
        self.registry.register("geometry_logits", torch.empty(0, geometry_experts, device=target), optimizer_group="geometry", role="geometry", ply_prefix="geometry")
        self.registry.register("semantic_confidence", torch.empty(0, 1, device=target), trainable=False, role="evidence")
        # Keep inferred neighborhood evidence separate from direct image
        # observations.  Consumers use the calibrated maximum, while density
        # observation updates continue to own ``semantic_confidence``.  This
        # prevents a propagated guess from being counted as a camera hit.
        self.registry.register(
            "propagated_semantic_confidence",
            torch.empty(0, 1, device=target),
            trainable=False,
            role="evidence",
        )
        self.registry.register("boundary_score", torch.empty(0, 1, device=target), trainable=False, role="evidence")
        self.registry.register("geometry_error", torch.empty(0, 1, device=target), trainable=False, role="evidence")
        self.registry.register("observation_count", torch.empty(0, 1, device=target), trainable=False, role="evidence")
        self.registry.register("max_radii2D", torch.empty(0, device=target), trainable=False, role="statistics")
        self.registry.register("xyz_gradient_accum", torch.empty(0, 1, device=target), trainable=False, role="statistics")
        self.registry.register("denom", torch.empty(0, 1, device=target), trainable=False, role="statistics")
        self.policy_bank = SoftGeometryPolicyBank(
            geometry_experts,
            confidence_floor=confidence_floor,
        )
        self.semantic_decoder: SemanticDecoder | None = None

    def __len__(self) -> int:
        return len(self.registry)

    @property
    def device(self) -> torch.device:
        return self.registry.device

    @property
    def topology_stamp(self) -> TopologyStamp:
        return TopologyStamp(
            generation=int(self.topology_generation),
            cumulative_topology_churn=int(self.cumulative_topology_churn),
            gaussian_count=len(self),
        )

    @property
    def xyz(self) -> Tensor:
        return self.registry["xyz"]

    @property
    def features_dc(self) -> Tensor:
        return self.registry["features_dc"]

    @property
    def features_rest(self) -> Tensor:
        return self.registry["features_rest"]

    @property
    def opacity(self) -> Tensor:
        return self.registry["opacity"]

    @property
    def scaling(self) -> Tensor:
        return self.registry["scaling"]

    @property
    def rotation(self) -> Tensor:
        return self.registry["rotation"]

    @property
    def semantic_embedding(self) -> Tensor:
        return self.registry["semantic_embedding"]

    @property
    def geometry_logits(self) -> Tensor:
        return self.registry["geometry_logits"]

    @property
    def semantic_confidence(self) -> Tensor:
        return self.registry["semantic_confidence"]

    @property
    def propagated_semantic_confidence(self) -> Tensor:
        if "propagated_semantic_confidence" not in self.registry:
            return torch.zeros_like(self.semantic_confidence)
        return self.registry["propagated_semantic_confidence"]

    @property
    def boundary_score(self) -> Tensor:
        return self.registry["boundary_score"]

    @property
    def geometry_error(self) -> Tensor:
        return self.registry["geometry_error"]

    @property
    def observation_count(self) -> Tensor:
        return self.registry["observation_count"]

    @property
    def max_radii2D(self) -> Tensor:
        return self.registry["max_radii2D"]

    @property
    def xyz_gradient_accum(self) -> Tensor:
        return self.registry["xyz_gradient_accum"]

    @property
    def denom(self) -> Tensor:
        return self.registry["denom"]

    # Original 3DGS private aliases are read-only on purpose: topology must go
    # through the registry rather than assigning individual tensors.
    _xyz = property(lambda self: self.xyz)
    _features_dc = property(lambda self: self.features_dc)
    _features_rest = property(lambda self: self.features_rest)
    _opacity = property(lambda self: self.opacity)
    _scaling = property(lambda self: self.scaling)
    _rotation = property(lambda self: self.rotation)
    _objects_dc = property(lambda self: self.semantic_embedding[:, None, :])

    @property
    def get_xyz(self) -> Tensor:
        return self.xyz

    @property
    def get_features(self) -> Tensor:
        return torch.cat((self.features_dc, self.features_rest), dim=1)

    @property
    def get_opacity(self) -> Tensor:
        return torch.sigmoid(self.opacity)

    @property
    def get_scaling(self) -> Tensor:
        return torch.exp(self.scaling)

    @property
    def get_rotation(self) -> Tensor:
        return F.normalize(self.rotation, dim=-1, eps=1e-8)

    @property
    def get_semantic(self) -> Tensor:
        return self.semantic_embedding

    @property
    def get_objects(self) -> Tensor:
        return self.semantic_embedding[:, None, :]

    @property
    def get_geometry_posterior(self) -> Tensor:
        return F.softmax(self.geometry_logits, dim=-1)

    @property
    def get_geometry_logits(self) -> Tensor:
        return self.geometry_logits

    @property
    def get_semantic_confidence(self) -> Tensor:
        return torch.maximum(
            self.semantic_confidence,
            self.propagated_semantic_confidence,
        )

    @property
    def get_boundary_score(self) -> Tensor:
        return self.boundary_score

    @property
    def get_geometry_error(self) -> Tensor:
        return self.geometry_error

    @property
    def get_normal(self) -> Tensor:
        if len(self) == 0:
            return self.xyz.new_empty((0, 3))
        rotation = quaternion_to_matrix(self.get_rotation)
        axis = self.get_scaling.argmin(dim=-1)
        gather = axis[:, None, None].expand(-1, 3, 1)
        return F.normalize(rotation.gather(2, gather).squeeze(-1), dim=-1, eps=1e-8)

    def get_covariance(self, scaling_modifier: float = 1.0) -> Tensor:
        rotation = quaternion_to_matrix(self.get_rotation)
        transform = rotation @ torch.diag_embed(self.get_scaling * scaling_modifier)
        return strip_symmetric(transform @ transform.transpose(-1, -2))

    def configure_semantic_decoder(
        self,
        num_classes: int,
        temperature: float = 1.0,
        preserve: bool = True,
    ) -> SemanticDecoder:
        if self.semantic_decoder is not None and self.semantic_decoder.linear.out_features == num_classes:
            return self.semantic_decoder
        old = self.semantic_decoder
        decoder = SemanticDecoder(self.semantic_dim, num_classes, temperature).to(self.device)
        if preserve and old is not None:
            rows = min(old.linear.out_features, decoder.linear.out_features)
            with torch.no_grad():
                decoder.linear.weight[:rows].copy_(old.linear.weight[:rows])
                decoder.linear.bias[:rows].copy_(old.linear.bias[:rows])
        self.semantic_decoder = decoder
        return decoder

    def decode_semantic(self, embedding: Tensor | None = None) -> Tensor:
        if self.semantic_decoder is None:
            raise RuntimeError("call configure_semantic_decoder(num_classes) before decoding")
        return self.semantic_decoder(self.semantic_embedding if embedding is None else embedding)

    @torch.no_grad()
    def point_region_memberships(
        self,
        indices: Tensor,
        *,
        top_k: int,
        chunk_size: int,
    ) -> SparseRegionMembership:
        """Decode selected Gaussians into sparse soft region memberships."""

        if self.semantic_decoder is None:
            raise RuntimeError("call configure_semantic_decoder(num_classes) before decoding")
        return decode_sparse_region_memberships(
            self.get_semantic,
            indices,
            decoder=self.semantic_decoder,
            num_classes=self.semantic_decoder.linear.out_features,
            top_k=top_k,
            chunk_size=chunk_size,
            confidence=self.get_semantic_confidence,
        )

    def oneupSHdegree(self) -> None:
        self.active_sh_degree = min(self.active_sh_degree + 1, self.max_sh_degree)

    def _replace_all(self, values: Mapping[str, Tensor]) -> None:
        snapshot = self.registry.capture()
        snapshot["tensors"] = {
            name: values.get(name, self.registry[name].detach()) for name in self.registry.names
        }
        self.registry.restore(snapshot, next(iter(snapshot["tensors"].values())).device)
        self.optimizer = None

    @staticmethod
    def _nearest_distance_squared(points: Tensor) -> Tensor:
        count = points.shape[0]
        if count <= 1:
            return points.new_full((count,), 1e-4)
        try:
            from scipy.spatial import cKDTree

            distances, _ = cKDTree(points.detach().cpu().numpy()).query(
                points.detach().cpu().numpy(), k=2, workers=-1
            )
            return torch.as_tensor(distances[:, 1] ** 2, device=points.device, dtype=points.dtype)
        except (ImportError, TypeError):
            if count > 20_000:
                extent = (points.max(0).values - points.min(0).values).norm()
                estimate = extent / max(count ** (1.0 / 3.0), 1.0)
                return points.new_full((count,), float(estimate.square().clamp_min(1e-7)))
            best = points.new_full((count,), float("inf"))
            chunk = 1024
            for start in range(0, count, chunk):
                distance = torch.cdist(points[start : start + chunk], points).square()
                row = torch.arange(distance.shape[0], device=points.device)
                distance[row, row + start] = float("inf")
                best[start : start + chunk] = distance.min(dim=1).values
            return best

    def create_from_pcd(self, pcd: Any, spatial_lr_scale: float) -> None:
        self.spatial_lr_scale = float(spatial_lr_scale)
        points = torch.as_tensor(np.asarray(pcd.points), dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(np.asarray(pcd.colors), dtype=torch.float32, device=self.device).clamp(0.0, 1.0)
        if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
            raise ValueError("point cloud points/colors must both have shape [N,3]")
        count = points.shape[0]
        distance2 = self._nearest_distance_squared(points).clamp_min(1e-7)
        scale = torch.log(torch.sqrt(distance2))[:, None].repeat(1, 3)
        rotation = points.new_zeros((count, 4))
        rotation[:, 0] = 1.0
        dc = RGB2SH(colors)[:, None, :]
        rest = points.new_zeros((count, (self.max_sh_degree + 1) ** 2 - 1, 3))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(0)
        semantic = torch.randn((count, self.semantic_dim), generator=generator, device=self.device) * 0.01
        values = {
            "xyz": points,
            "features_dc": dc,
            "features_rest": rest,
            "opacity": inverse_sigmoid(points.new_full((count, 1), 0.1)),
            "scaling": scale,
            "rotation": rotation,
            "semantic_embedding": semantic,
            "geometry_logits": points.new_zeros((count, self.geometry_experts)),
            "semantic_confidence": points.new_zeros((count, 1)),
            "propagated_semantic_confidence": points.new_zeros((count, 1)),
            "boundary_score": points.new_zeros((count, 1)),
            "geometry_error": points.new_zeros((count, 1)),
            "observation_count": points.new_zeros((count, 1)),
            "max_radii2D": points.new_zeros((count,)),
            "xyz_gradient_accum": points.new_zeros((count, 1)),
            "denom": points.new_zeros((count, 1)),
        }
        self._replace_all(values)
        # An imported point cloud defines a new baseline topology. Its initial
        # population is not optimizer churn and must not make a fresh mesh look
        # stale before training starts.
        self.topology_generation = 0
        self.cumulative_topology_churn = 0

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "render": list(self.render_parameters()),
            "semantic": list(self.semantic_parameters()),
            "geometry": list(self.geometry_parameters()),
        }

    def render_parameters(self) -> Iterator[nn.Parameter]:
        for name in ("xyz", "features_dc", "features_rest", "opacity", "scaling", "rotation"):
            yield self.registry.trainable[name]

    def semantic_parameters(self) -> Iterator[nn.Parameter]:
        yield self.registry.trainable["semantic_embedding"]
        if self.semantic_decoder is not None:
            yield from self.semantic_decoder.parameters()

    def geometry_parameters(self) -> Iterator[nn.Parameter]:
        yield self.registry.trainable["geometry_logits"]
        yield self.registry.trainable["xyz"]
        yield self.registry.trainable["scaling"]
        yield self.registry.trainable["rotation"]

    def set_training_stage(self, stage: str) -> None:
        stages = {"bootstrap", "semantic_lift", "joint_geometry", "surface_refine"}
        if stage not in stages:
            raise ValueError(f"unknown training stage {stage!r}; expected one of {sorted(stages)}")
        render_names = {"xyz", "features_dc", "features_rest", "opacity", "scaling", "rotation"}
        for name, spec in self.registry.specs.items():
            if not spec.trainable:
                continue
            enabled = (
                (stage == "bootstrap" and name in render_names)
                or (
                    stage == "semantic_lift"
                    and (name in render_names or name == "semantic_embedding")
                )
                or stage in {"joint_geometry", "surface_refine"}
            )
            self.registry.trainable[name].requires_grad_(enabled)
        if self.semantic_decoder is not None:
            decoder_enabled = stage in {"semantic_lift", "joint_geometry", "surface_refine"}
            for parameter in self.semantic_decoder.parameters():
                parameter.requires_grad_(decoder_enabled)

    def training_setup(self, training_args: Any) -> None:
        self._training_args = training_args
        self.percent_dense = float(getattr(training_args, "percent_dense", 0.01))
        feature_lr = float(getattr(training_args, "feature_lr", 2.5e-3))
        feature_dc_lr = float(getattr(training_args, "feature_dc_lr", feature_lr))
        feature_rest_lr = float(
            getattr(training_args, "feature_rest_lr", feature_dc_lr / 20.0)
        )
        named = {
            "xyz": (self.xyz, getattr(training_args, "position_lr_init", 1.6e-4) * self.spatial_lr_scale),
            "f_dc": (self.features_dc, feature_dc_lr),
            "f_rest": (self.features_rest, feature_rest_lr),
            "opacity": (self.opacity, getattr(training_args, "opacity_lr", 5e-2)),
            "scaling": (self.scaling, getattr(training_args, "scaling_lr", 5e-3)),
            "rotation": (self.rotation, getattr(training_args, "rotation_lr", 1e-3)),
            "semantic": (self.semantic_embedding, getattr(training_args, "semantic_lr", 2.5e-3)),
            "geometry": (self.geometry_logits, getattr(training_args, "geometry_lr", 1e-3)),
        }
        groups = [{"params": [parameter], "lr": float(lr), "name": name} for name, (parameter, lr) in named.items()]
        # The scene decoder has a stable, non-topological lifecycle and is
        # owned exclusively by the trainer's head optimizer. Keeping it out of
        # this registry-bound Adam guarantees identical Gaussian param groups
        # for RGB-only and semantic scenes and makes checkpoint resume robust.
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.registry.bind_optimizer(self.optimizer)

    def update_learning_rate(self, iteration: int) -> float:
        if self.optimizer is None or self._training_args is None:
            raise RuntimeError("training_setup must be called before update_learning_rate")
        args = self._training_args
        lr = _exponential_lr(
            iteration,
            float(getattr(args, "position_lr_init", 1.6e-4)) * self.spatial_lr_scale,
            float(getattr(args, "position_lr_final", 1.6e-6)) * self.spatial_lr_scale,
            int(getattr(args, "position_lr_max_steps", 30_000)),
            float(getattr(args, "position_lr_delay_mult", 0.01)),
            int(getattr(args, "position_lr_delay_steps", 0)),
        )
        for group in self.optimizer.param_groups:
            if group.get("name") == "xyz":
                group["lr"] = lr
                break
        return lr

    def capture(self) -> dict[str, Any]:
        return {
            "format_version": 3,
            "active_sh_degree": self.active_sh_degree,
            "max_sh_degree": self.max_sh_degree,
            "semantic_dim": self.semantic_dim,
            "geometry_experts": self.geometry_experts,
            "policy_confidence_floor": self.policy_bank.confidence_floor,
            "spatial_lr_scale": self.spatial_lr_scale,
            "percent_dense": self.percent_dense,
            "topology_generation": self.topology_generation,
            "cumulative_topology_churn": self.cumulative_topology_churn,
            "policy_bank": {
                name: value.detach().clone()
                for name, value in self.policy_bank.state_dict().items()
            },
            "registry": self.registry.capture(),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "semantic_classes": None if self.semantic_decoder is None else self.semantic_decoder.linear.out_features,
            "semantic_temperature": None if self.semantic_decoder is None else self.semantic_decoder.temperature,
            "semantic_decoder": None if self.semantic_decoder is None else self.semantic_decoder.state_dict(),
        }

    @torch.no_grad()
    def capture_inference(
        self,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Create an immutable optimizer-free snapshot for background readers.

        Unlike :meth:`capture`, this routine copies tensors directly to the
        requested device and never materializes Adam state. It is therefore the
        safe hand-off primitive for full inference workers while live training
        continues to mutate parameters. Surface and mesh workers use the
        smaller :meth:`capture_surface_inference` contract.
        """

        target = self.device if device is None else torch.device(device)
        tensors = {
            name: value.detach().to(device=target, copy=True)
            for name, value in self.registry.named_attributes()
        }
        decoder = None
        if self.semantic_decoder is not None:
            decoder = {
                name: value.detach().to(device=target, copy=True)
                for name, value in self.semantic_decoder.state_dict().items()
            }
        return {
            "format_version": 3,
            "active_sh_degree": self.active_sh_degree,
            "max_sh_degree": self.max_sh_degree,
            "semantic_dim": self.semantic_dim,
            "geometry_experts": self.geometry_experts,
            "policy_confidence_floor": self.policy_bank.confidence_floor,
            "spatial_lr_scale": self.spatial_lr_scale,
            "percent_dense": self.percent_dense,
            "topology_generation": self.topology_generation,
            "cumulative_topology_churn": self.cumulative_topology_churn,
            "policy_bank": {
                name: value.detach().to(device=target, copy=True)
                for name, value in self.policy_bank.state_dict().items()
            },
            "registry": {
                "format_version": self.registry.FORMAT_VERSION,
                "specs": [asdict(spec) for spec in self.registry.specs.values()],
                "tensors": tensors,
            },
            "optimizer": None,
            "semantic_classes": (
                None
                if self.semantic_decoder is None
                else self.semantic_decoder.linear.out_features
            ),
            "semantic_temperature": (
                None if self.semantic_decoder is None else self.semantic_decoder.temperature
            ),
            "semantic_decoder": decoder,
        }

    @torch.no_grad()
    def capture_surface_inference(
        self,
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Create a minimal immutable snapshot for surface and mesh workers.

        Rendering SH coefficients and transient densification statistics are
        intentionally absent.  The returned object remains a normal Gaussian
        model schema-v3 snapshot, so :meth:`restore` is the only deserializer
        needed by both full inference and background surface extraction.
        """

        target = self.device if device is None else torch.device(device)
        missing = [
            name for name in SURFACE_INFERENCE_ATTRIBUTES if name not in self.registry
        ]
        if missing:
            raise ValueError(
                "surface inference snapshot is missing attributes: "
                + ", ".join(missing)
            )
        tensors = {
            name: self.registry[name].detach().to(device=target, copy=True)
            for name in SURFACE_INFERENCE_ATTRIBUTES
        }
        decoder = None
        if self.semantic_decoder is not None:
            decoder = {
                name: value.detach().to(device=target, copy=True)
                for name, value in self.semantic_decoder.state_dict().items()
            }
        return {
            "format_version": 3,
            "active_sh_degree": self.active_sh_degree,
            "max_sh_degree": self.max_sh_degree,
            "semantic_dim": self.semantic_dim,
            "geometry_experts": self.geometry_experts,
            "policy_confidence_floor": self.policy_bank.confidence_floor,
            "spatial_lr_scale": self.spatial_lr_scale,
            "percent_dense": self.percent_dense,
            "topology_generation": self.topology_generation,
            "cumulative_topology_churn": self.cumulative_topology_churn,
            "policy_bank": {
                name: value.detach().to(device=target, copy=True)
                for name, value in self.policy_bank.state_dict().items()
            },
            "registry": {
                "format_version": self.registry.FORMAT_VERSION,
                "specs": [
                    asdict(self.registry.specs[name])
                    for name in SURFACE_INFERENCE_ATTRIBUTES
                ],
                "tensors": tensors,
            },
            "optimizer": None,
            "semantic_classes": (
                None
                if self.semantic_decoder is None
                else self.semantic_decoder.linear.out_features
            ),
            "semantic_temperature": (
                None
                if self.semantic_decoder is None
                else self.semantic_decoder.temperature
            ),
            "semantic_decoder": decoder,
        }

    def restore(self, snapshot: Mapping[str, Any], training_args: Any | None = None) -> None:
        if int(snapshot.get("format_version", 1)) > 3:
            raise ValueError("checkpoint was produced by a newer Gaussian model schema")
        self.active_sh_degree = int(snapshot["active_sh_degree"])
        self.max_sh_degree = int(snapshot.get("max_sh_degree", self.max_sh_degree))
        self.semantic_dim = int(snapshot.get("semantic_dim", self.semantic_dim))
        self.geometry_experts = int(snapshot.get("geometry_experts", self.geometry_experts))
        confidence_floor = float(
            snapshot.get(
                "policy_confidence_floor",
                self.policy_bank.confidence_floor,
            )
        )
        if not 0.0 <= confidence_floor < 1.0:
            raise ValueError("checkpoint policy confidence floor must lie in [0,1)")
        self.policy_bank.confidence_floor = confidence_floor
        policy_state = snapshot.get("policy_bank")
        if policy_state is not None:
            self.policy_bank.load_state_dict(policy_state)
        self.spatial_lr_scale = float(snapshot.get("spatial_lr_scale", 1.0))
        self.percent_dense = float(snapshot.get("percent_dense", 0.01))
        topology_generation = int(snapshot.get("topology_generation", 0))
        cumulative_topology_churn = int(
            snapshot.get("cumulative_topology_churn", 0)
        )
        if topology_generation < 0 or cumulative_topology_churn < 0:
            raise ValueError("checkpoint topology counters must be non-negative")
        self.topology_generation = topology_generation
        self.cumulative_topology_churn = cumulative_topology_churn
        self.registry.restore(snapshot["registry"], self.device)
        # Checkpoints produced before propagated evidence was introduced have
        # a complete legacy registry schema.  Migrate them in-place before an
        # optimizer is rebound so resume remains exact for all old attributes.
        if "propagated_semantic_confidence" not in self.registry:
            self.registry.register(
                "propagated_semantic_confidence",
                torch.zeros_like(self.semantic_confidence),
                trainable=False,
                role="evidence",
            )
        classes = snapshot.get("semantic_classes")
        if classes is not None:
            self.configure_semantic_decoder(int(classes), float(snapshot.get("semantic_temperature") or 1.0), preserve=False)
            self.semantic_decoder.load_state_dict(snapshot["semantic_decoder"])
        if training_args is not None:
            self.training_setup(training_args)
            if snapshot.get("optimizer") is not None:
                self.optimizer.load_state_dict(snapshot["optimizer"])

    def save_ply(self, path: str | Path) -> None:
        self.registry.save_ply(path)

    def load_ply(self, path: str | Path) -> None:
        self.registry.load_ply(path, self.device)
        if "propagated_semantic_confidence" not in self.registry:
            self.registry.register(
                "propagated_semantic_confidence",
                torch.zeros_like(self.semantic_confidence),
                trainable=False,
                role="evidence",
            )
        self.optimizer = None
        self.active_sh_degree = self.max_sh_degree
        # PLY stores per-Gaussian attributes but no global mutation history.
        # Treat the loaded topology as a new baseline instead of inventing a
        # generation that cannot be recovered from the file.
        self.topology_generation = 0
        self.cumulative_topology_churn = 0

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.capture(), path)

    def load_checkpoint(self, path: str | Path, training_args: Any | None = None) -> None:
        snapshot = torch.load(path, map_location=self.device, weights_only=True)
        self.restore(snapshot, training_args)

    def replace_tensor_to_optimizer(self, tensor: Tensor, name: str) -> dict[str, Tensor]:
        """Compatibility helper for topology-preserving single-group updates."""

        lookup = {spec.optimizer_group: attr for attr, spec in self.registry.specs.items()}
        if name not in lookup:
            raise KeyError(name)
        attribute = lookup[name]
        if tensor.shape[0] != len(self):
            raise ValueError("replacement cannot alter topology")
        self.registry.replace(attribute, tensor, reset_optimizer_moments=True)
        return {name: self.registry[attribute]}

    @torch.no_grad()
    def reset_opacity(self, max_opacity: float = 0.01, maximum: float | None = None) -> None:
        if maximum is not None:
            max_opacity = maximum
        target = inverse_sigmoid(torch.minimum(self.get_opacity, self.opacity.new_full((), max_opacity)))
        self.replace_tensor_to_optimizer(target, "opacity")

    @torch.no_grad()
    def update_evidence(
        self,
        indices: Tensor | Sequence[int] | None = None,
        semantic_confidence: Tensor | float | None = None,
        boundary_score: Tensor | float | None = None,
        geometry_error: Tensor | float | None = None,
        observation_count: Tensor | float | None = None,
        momentum: float = 0.9,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0,1)")
        if indices is None:
            selected = torch.arange(len(self), device=self.device)
        else:
            mask = _selection_mask(indices, len(self), self.device)
            selected = mask.nonzero(as_tuple=False).flatten()
        rows = selected.numel()
        if rows == 0:
            return
        old_count = self.observation_count[selected]
        # Start as an arithmetic mean, then smoothly become an EMA.
        effective_momentum = torch.minimum(
            old_count / (old_count + 1.0), old_count.new_full((), momentum)
        )

        def update(name: str, value: Tensor | float | None) -> None:
            if value is None:
                return
            target = self.registry[name]
            incoming = torch.as_tensor(value, device=target.device, dtype=target.dtype)
            if incoming.ndim == 0:
                incoming = incoming.expand(rows, 1)
            elif incoming.ndim == 1:
                incoming = incoming[:, None]
            if incoming.shape != (rows, 1):
                raise ValueError(f"{name} evidence must have shape [{rows}] or [{rows},1]")
            target[selected] = effective_momentum * target[selected] + (1.0 - effective_momentum) * incoming

        update("semantic_confidence", semantic_confidence)
        update("boundary_score", boundary_score)
        update("geometry_error", geometry_error)
        if observation_count is None:
            self.observation_count[selected] = old_count + 1.0
        else:
            count = torch.as_tensor(observation_count, device=self.device, dtype=self.observation_count.dtype)
            if count.ndim == 0:
                count = count.expand(rows, 1)
            elif count.ndim == 1:
                count = count[:, None]
            self.observation_count[selected] = count

    @torch.no_grad()
    def update_propagated_semantic_confidence(
        self,
        indices: Tensor | Sequence[int],
        confidence: Tensor | float,
        *,
        momentum: float = 0.5,
        decay: float = 0.995,
        maximum: float = 0.85,
    ) -> None:
        """Update inferred confidence without changing direct observations.

        Missing neighborhood support decays stale propagation, while supported
        estimates use an EMA.  The explicit ceiling ensures propagated
        evidence can guide geometry but can never masquerade as a near-certain
        multi-view observation.
        """

        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0,1)")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must lie in [0,1]")
        if not 0.0 < maximum <= 1.0:
            raise ValueError("maximum must lie in (0,1]")
        selection = torch.as_tensor(indices, device=self.device)
        if selection.dtype == torch.bool:
            if selection.shape != (len(self),):
                raise ValueError(f"propagated confidence mask must have shape [{len(self)}]")
            selected = selection.nonzero(as_tuple=False).flatten()
        else:
            selected = selection.long().flatten()
            if selected.numel() and (
                bool((selected < 0).any()) or bool((selected >= len(self)).any())
            ):
                raise IndexError("propagated confidence index is out of range")
        if selected.numel() == 0:
            return
        incoming = torch.as_tensor(
            confidence,
            device=self.device,
            dtype=self.propagated_semantic_confidence.dtype,
        )
        if incoming.ndim == 0:
            incoming = incoming.expand(selected.numel(), 1)
        elif incoming.ndim == 1:
            incoming = incoming[:, None]
        if incoming.shape != (selected.numel(), 1):
            raise ValueError(
                "propagated confidence must have shape "
                f"[{selected.numel()}] or [{selected.numel()},1]"
            )
        incoming = torch.nan_to_num(incoming, nan=0.0, posinf=maximum, neginf=0.0)
        incoming = incoming.clamp(0.0, maximum)
        target = self.registry["propagated_semantic_confidence"]
        previous = target[selected]
        supported = incoming > 0
        updated = momentum * previous + (1.0 - momentum) * incoming
        target[selected] = torch.where(supported, updated, decay * previous).clamp(0.0, maximum)

    def mutate_topology(
        self,
        clone_mask: Tensor | Sequence[int] | None,
        split_mask: Tensor | Sequence[int] | None,
        prune_mask: Tensor | Sequence[int] | None,
        *,
        children: int = 2,
        offsets: Tensor | None = None,
        scale_factor: float = 0.8,
        overrides: Mapping[str, Tensor] | None = None,
    ) -> dict[str, int]:
        if children < 1 or scale_factor <= 0:
            raise ValueError("children and scale_factor must be positive")
        size = len(self)
        clone = _selection_mask(clone_mask, size, self.device)
        split = _selection_mask(split_mask, size, self.device)
        prune = _selection_mask(prune_mask, size, self.device)
        clone &= ~prune
        split &= ~prune
        if (clone & split).any():
            raise ValueError("clone_mask and split_mask must be disjoint")
        clone_indices = clone.nonzero(as_tuple=False).flatten()
        split_indices = split.nonzero(as_tuple=False).flatten()
        survivor_mask = ~(prune | split)
        survivors = survivor_mask.nonzero(as_tuple=False).flatten()
        split_source = split_indices.repeat_interleave(children)
        source = torch.cat((survivors, clone_indices, split_source))
        fresh = torch.cat(
            (
                torch.zeros_like(survivors, dtype=torch.bool),
                torch.ones_like(clone_indices, dtype=torch.bool),
                torch.ones_like(split_source, dtype=torch.bool),
            )
        )
        values = {name: self.registry[name].detach().index_select(0, source) for name in self.registry.names}
        split_start = survivors.numel() + clone_indices.numel()
        split_count = split_source.numel()
        if split_count:
            selected_scale = self.get_scaling[split_indices]
            selected_rotation = quaternion_to_matrix(self.get_rotation[split_indices])
            if offsets is None:
                local = torch.randn(
                    (split_indices.numel(), children, 3),
                    device=self.device,
                    dtype=self.xyz.dtype,
                ) * selected_scale[:, None, :]
                world_offsets = torch.einsum("mij,mcj->mci", selected_rotation, local)
            else:
                world_offsets = torch.as_tensor(offsets, device=self.device, dtype=self.xyz.dtype)
                if world_offsets.shape == (split_count, 3):
                    world_offsets = world_offsets.reshape(split_indices.numel(), children, 3)
                if world_offsets.shape != (split_indices.numel(), children, 3):
                    raise ValueError(
                        f"offsets must have shape [{split_indices.numel()},{children},3] or [{split_count},3]"
                    )
            values["xyz"][split_start:] = (
                self.xyz[split_indices, None, :] + world_offsets
            ).reshape(split_count, 3)
            values["scaling"][split_start:] = torch.log(
                (
                    selected_scale[:, None, :].expand(-1, children, -1)
                    / (scale_factor * children)
                ).clamp_min(1e-8)
            ).reshape(split_count, 3)
        # Densification statistics describe old projected points and must never
        # be inherited by new children.
        for name in ("max_radii2D", "xyz_gradient_accum", "denom"):
            values[name][fresh] = 0
        for name, override in (overrides or {}).items():
            if name not in values:
                raise KeyError(name)
            override = torch.as_tensor(override, device=values[name].device, dtype=values[name].dtype)
            new_rows = clone_indices.numel() + split_count
            if override.shape == values[name].shape:
                values[name] = override
            elif override.shape == (new_rows,) + tuple(values[name].shape[1:]):
                values[name][survivors.numel() :] = override
            else:
                raise ValueError(
                    f"override {name!r} must have complete shape {tuple(values[name].shape)} "
                    f"or new-row shape {(new_rows,) + tuple(values[name].shape[1:])}"
                )
        self.registry.mutate(source, overrides=values, fresh_mask=fresh)
        created = int(clone_indices.numel()) + int(split_count)
        removed = int(prune.sum().item()) + int(split_indices.numel())
        if created or removed:
            # Update only after the atomic registry transaction succeeds.
            # max(created, removed) measures replaced slots without double
            # counting zero-net-growth prune-and-replace transactions.
            self.topology_generation += 1
            self.cumulative_topology_churn += max(created, removed)
        return {
            "old_size": size,
            "new_size": len(self),
            "survived": int(survivors.numel()),
            "cloned": int(clone_indices.numel()),
            "split_parents": int(split_indices.numel()),
            "split_children": int(split_count),
            "pruned": int(prune.sum().item()) + int(split.sum().item()),
        }

    def clone(self, mask: Tensor | Sequence[int], overrides: Mapping[str, Tensor] | None = None) -> int:
        result = self.mutate_topology(mask, None, None, overrides=overrides)
        return result["cloned"]

    def split(
        self,
        mask: Tensor | Sequence[int],
        children: int = 2,
        offsets: Tensor | None = None,
        scale_factor: float = 0.8,
    ) -> int:
        result = self.mutate_topology(None, mask, None, children=children, offsets=offsets, scale_factor=scale_factor)
        return result["split_children"]

    def prune(self, mask: Tensor | Sequence[int]) -> int:
        result = self.mutate_topology(None, None, mask)
        return result["pruned"]

    clone_points = clone
    split_points = split
    prune_points = prune

    @torch.no_grad()
    def add_densification_stats(
        self,
        viewspace_point_tensor: Tensor,
        update_filter: Tensor,
        normal_error: Tensor | None = None,
        semantic_boundary: Tensor | None = None,
    ) -> None:
        mask = _selection_mask(update_filter, len(self), self.device)
        if viewspace_point_tensor.grad is not None:
            gradient = torch.norm(viewspace_point_tensor.grad[mask, :2], dim=-1, keepdim=True)
            self.xyz_gradient_accum[mask] += gradient
            self.denom[mask] += 1
        indices = mask.nonzero(as_tuple=False).flatten()
        if indices.numel() and (normal_error is not None or semantic_boundary is not None):
            self.update_evidence(
                indices,
                geometry_error=None if normal_error is None else normal_error[mask],
                boundary_score=None if semantic_boundary is None else semantic_boundary[mask],
            )

    def densify_and_clone(self, grads: Tensor, grad_threshold: float, scene_extent: float) -> int:
        gradient = grads.squeeze(-1)
        mask = (gradient >= grad_threshold) & (self.get_scaling.max(dim=1).values <= self.percent_dense * scene_extent)
        return self.clone(mask)

    def densify_and_split(
        self,
        grads: Tensor,
        grad_threshold: float,
        scene_extent: float,
        children: int = 2,
    ) -> int:
        gradient = grads.squeeze(-1)
        mask = (gradient >= grad_threshold) & (self.get_scaling.max(dim=1).values > self.percent_dense * scene_extent)
        return self.split(mask, children)


__all__ = [
    "GaussianAttributeRegistry",
    "GaussianModel",
    "RGB2SH",
    "SH2RGB",
    "SemanticDecoder",
    "TopologyStamp",
    "inverse_sigmoid",
    "quaternion_to_matrix",
]
