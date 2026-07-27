"""Four-stage optimization engine for Semantic Gaussian Wrapping."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from densification import DensityController, DensityReport, TopologyBudget
from gaussian_renderer import render
from regularization import Phase, PhaseScheduler, PhotometricParetoGuard, SemanticLossSystem
from regularization.surface import (
    gaussian_surface_consistency,
    prepare_gaussian_surface_consistency,
)
from .checkpointing import (
    capture_rng_state,
    restore_rng_state,
    validate_training_checkpoint_header,
)
from semantic import GaussianNeighborIndex
from utils.graphics_utils import DEFAULT_NORMAL_ALPHA_THRESHOLD, depth_normal_residual
from utils.image_utils import composite_background


@dataclass(frozen=True)
class TrainingResult:
    iteration: int
    gaussian_count: int
    elapsed_seconds: float
    last_metrics: dict[str, float]


@dataclass(frozen=True)
class DensityLifecycle:
    """Per-iteration standard or surface topology actions."""

    observe: bool
    topology_step: bool
    reset_opacity: bool
    enable_size_pruning: bool
    window: str | None
    topology_budget: TopologyBudget | None


class _StepProfiler:
    """Synchronized wall-clock timings for one explicitly sampled step.

    CUDA work is asynchronous, so every phase boundary synchronizes only while
    this object exists. The default training path never constructs it and
    therefore retains its original synchronization behavior.
    """

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self._started_at: float | None = None
        self._last_boundary: float | None = None
        self._metrics: dict[str, float] = {}

    def _boundary(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def start(self) -> None:
        if self._started_at is not None:
            raise RuntimeError("step profiler has already started")
        boundary = self._boundary()
        self._started_at = boundary
        self._last_boundary = boundary

    def mark(self, phase: str) -> None:
        if self._last_boundary is None:
            raise RuntimeError("step profiler must be started before marking a phase")
        boundary = self._boundary()
        self._metrics[f"time_{phase}_ms"] = 1_000.0 * (
            boundary - self._last_boundary
        )
        self._last_boundary = boundary

    def finish(self, phase: str) -> dict[str, float]:
        self.mark(phase)
        assert self._started_at is not None and self._last_boundary is not None
        self._metrics["time_step_ms"] = 1_000.0 * (
            self._last_boundary - self._started_at
        )
        return dict(self._metrics)


def _should_profile_step(iteration: int, interval: int) -> bool:
    return interval > 0 and iteration % interval == 0


def _density_lifecycle(
    iteration: int,
    config: Mapping[str, Any],
    *,
    surface_config: Mapping[str, Any] | None = None,
) -> DensityLifecycle:
    """Resolve density actions without shortening the first statistics window.

    Standard 3DGS collects projected-gradient and radius statistics from the
    beginning of training, then starts mutating topology strictly *after* the
    configured warm-up iteration. Screen/world-size pruning is an optional,
    late-stage policy; opacity pruning remains part of every topology step.
    """

    until_iteration = int(config["until_iter"])
    interval = int(config["interval"])
    opacity_reset_interval = int(config["opacity_reset_interval"])
    standard_observe = 1 <= iteration < until_iteration
    standard_step = (
        standard_observe
        and iteration > int(config["from_iter"])
        and iteration % interval == 0
    )
    reset_opacity = standard_observe and iteration % opacity_reset_interval == 0
    standard_size_pruning = (
        standard_step
        and bool(config.get("enable_size_pruning", False))
        and iteration > opacity_reset_interval
    )

    surface_observe = False
    surface_step = False
    surface_size_pruning = False
    surface_budget = None
    if surface_config is not None:
        surface_enabled = bool(surface_config["enabled"]) and bool(
            surface_config["topology_enabled"]
        )
        topology_from = int(surface_config["topology_from"])
        topology_until = int(surface_config["topology_until"])
        surface_interval = int(surface_config["topology_interval"])
        surface_observe = surface_enabled and topology_from <= iteration <= topology_until
        # The opening iteration starts a clean evidence window. The first
        # mutation follows only after one complete surface interval.
        surface_step = (
            surface_observe
            and iteration > topology_from
            and (iteration - topology_from) % surface_interval == 0
        )
        surface_size_pruning = surface_step and bool(
            surface_config["topology_enable_size_pruning"]
        )
        if surface_observe:
            surface_budget = TopologyBudget(
                max_net_growth=int(surface_config["topology_max_net_growth"]),
                replacement_budget=int(surface_config["topology_replacement_budget"]),
                protect_min_confidence=float(
                    surface_config["topology_protect_min_confidence"]
                ),
                protect_boundary=float(
                    surface_config["topology_protect_boundary"]
                ),
                protect_thin_probability=float(
                    surface_config["topology_protect_thin_probability"]
                ),
            )

    # In a deliberately overlapping configuration, standard 3DGS owns the
    # original window; surface replacement begins only after it ends.
    observe = standard_observe or surface_observe
    use_surface = surface_observe and not standard_observe
    return DensityLifecycle(
        observe=observe,
        topology_step=surface_step if use_surface else standard_step,
        reset_opacity=reset_opacity if standard_observe else False,
        enable_size_pruning=(
            surface_size_pruning if use_surface else standard_size_pruning
        ),
        window="surface" if use_surface else ("standard" if standard_observe else None),
        topology_budget=surface_budget if use_surface else None,
    )


def _parameters_from_optimizer(optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def _named_optimizer_parameters(optimizer: torch.optim.Optimizer, token: str) -> list[nn.Parameter]:
    token = token.lower()
    result = []
    for group in optimizer.param_groups:
        if token in str(group.get("name", "")).lower():
            result.extend(group["params"])
    return result


def _unique_parameters(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    result = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            result.append(parameter)
            seen.add(id(parameter))
    return result


class JsonlLogger:
    def __init__(self, path: str | Path | None) -> None:
        self.path = None if path is None else Path(path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, values: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(values, sort_keys=True) + "\n")

    @staticmethod
    def _atomic_write(path: Path, lines: list[str]) -> None:
        """Replace ``path`` from a temporary file in the same directory."""

        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf8") as stream:
                stream.writelines(lines)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _archive_path(self, start_iteration: int) -> Path:
        assert self.path is not None
        suffix = self.path.suffix or ".jsonl"
        stem = self.path.name[: -len(self.path.suffix)] if self.path.suffix else self.path.name
        base_name = f"{stem}.abandoned_after_{start_iteration}"
        candidate = self.path.with_name(f"{base_name}{suffix}")
        sequence = 1
        while candidate.exists():
            candidate = self.path.with_name(f"{base_name}_{sequence:03d}{suffix}")
            sequence += 1
        return candidate

    def rewind(self, start_iteration: int) -> Path | None:
        """Archive records not belonging to a resumed checkpoint timeline.

        Training logs are append ordered, so the first record newer than the
        checkpoint marks the abandoned suffix. A malformed/truncated record is
        also treated as the start of that suffix: preserving it in the archive
        keeps the active JSONL valid without discarding crash evidence.
        """

        if self.path is None or not self.path.is_file():
            return None

        lines = self.path.read_text(encoding="utf8").splitlines(keepends=True)
        tail_start: int | None = None
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
                iteration = record["iteration"]
                if not isinstance(iteration, int) or isinstance(iteration, bool):
                    raise ValueError("iteration must be an integer")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                tail_start = index
                break
            if iteration > start_iteration:
                tail_start = index
                break

        if tail_start is None:
            return None

        archive_path = self._archive_path(start_iteration)
        self._atomic_write(archive_path, lines[tail_start:])
        self._atomic_write(self.path, lines[:tail_start])
        return archive_path


class SemanticGaussianTrainer:
    """Own the complete differentiable training lifecycle.

    Scene construction and serialization stay in the standard 3DGS entrypoint;
    this class owns optimization, curriculum, semantic gradient routing,
    topology changes, and optional mesh feedback.
    """

    def __init__(
        self,
        scene,
        gaussians,
        pipeline,
        config: dict[str, Any],
        *,
        surface_field=None,
        mesh_feedback=None,
        policy_bank=None,
        evidence_projector=None,
        neighbor_index: GaussianNeighborIndex | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        self.scene = scene
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.config = config
        self.surface_field = surface_field
        self.mesh_feedback = mesh_feedback
        self.policy_bank = policy_bank
        self.output_path = None if output_path is None else Path(output_path)

        surface_cfg = config["surface"]
        self.neighbor_index = (
            neighbor_index
            or getattr(surface_field, "neighbor_index", None)
            or getattr(evidence_projector, "neighbor_index", None)
            or GaussianNeighborIndex(
                gaussians,
                backend=str(surface_cfg.get("neighbor_backend", "auto")),
                gaussian_chunk_size=int(surface_cfg.get("gaussian_chunk", 8192)),
                query_chunk_size=int(surface_cfg.get("neighbor_query_chunk", 2048)),
                max_distance_bytes=int(surface_cfg.get("max_distance_bytes", 64 * 1024 * 1024)),
                support_candidate_budget=int(
                    surface_cfg.get("support_candidate_budget", 2_048)
                ),
                support_routing_query_chunk=int(
                    surface_cfg.get("support_routing_query_chunk", 8_192)
                ),
                scipy_workers=int(surface_cfg.get("scipy_workers", 4)),
            )
        )
        if self.neighbor_index.gaussians is not gaussians:
            self.neighbor_index.set_gaussians(gaussians)
        surface_index_setter = getattr(surface_field, "set_neighbor_index", None)
        if callable(surface_index_setter):
            surface_index_setter(self.neighbor_index)
        evidence_index_setter = getattr(evidence_projector, "set_neighbor_index", None)
        if callable(evidence_index_setter):
            evidence_index_setter(self.neighbor_index)

        if getattr(gaussians, "optimizer", None) is None:
            raise RuntimeError("gaussians.training_setup(opt) must be called before creating the trainer")
        self.gaussian_parameters = _parameters_from_optimizer(gaussians.optimizer)
        semantic_parameters = getattr(gaussians, "semantic_parameters", None)
        if callable(semantic_parameters):
            self.semantic_parameters = list(semantic_parameters())
        else:
            self.semantic_parameters = _named_optimizer_parameters(gaussians.optimizer, "semantic")

        self.device = gaussians.get_xyz.device
        number_classes = int(scene.num_semantic_classes)
        if number_classes < 2:
            raise ValueError(
                "SemanticGaussianTrainer requires Gaga observations with at least "
                "one foreground region"
            )
        model_cfg = config["model"]
        semantic_cfg = config.get("semantic", {})
        self.region_decode_chunk_size = int(
            semantic_cfg["region_decode_chunk_size"]
        )
        if self.region_decode_chunk_size < 1:
            raise ValueError("semantic.region_decode_chunk_size must be positive")
        self.region_top_k = int(semantic_cfg["region_top_k"])
        if self.region_top_k < 1:
            raise ValueError("semantic.region_top_k must be positive")
        semantic_decoder = gaussians.configure_semantic_decoder(
            number_classes,
            float(semantic_cfg.get("temperature", 0.1)),
        )
        self.loss_system = SemanticLossSystem(
            int(model_cfg["semantic_dim"]),
            number_classes,
            config["loss"],
            semantic_decoder,
            policy_bank=policy_bank,
            evidence_projector=evidence_projector,
            evidence_interval=int(semantic_cfg.get("evidence_interval", 100)),
            evidence_samples=int(semantic_cfg.get("evidence_samples", 2_048)),
            evidence_weight=float(semantic_cfg.get("evidence_weight", 1.0)),
            evidence_entropy_weight=float(
                semantic_cfg.get("evidence_entropy_weight", 0.05)
            ),
            evidence_balance_weight=float(
                semantic_cfg.get("evidence_balance_weight", 0.10)
            ),
            neighbor_index=self.neighbor_index,
            normal_alpha_threshold=float(
                config["loss"].get("normal_alpha_threshold", DEFAULT_NORMAL_ALPHA_THRESHOLD)
            ),
            region_top_k=self.region_top_k,
            region_decode_chunk_size=self.region_decode_chunk_size,
        ).to(self.device)
        head_lr = float(config["optimization"].get("semantic_head_lr", 5e-4))
        gaussian_ids = {id(parameter) for parameter in self.gaussian_parameters}
        extra_parameters: list[nn.Parameter] = [
            parameter for parameter in self.loss_system.parameters() if id(parameter) not in gaussian_ids
        ]
        if isinstance(surface_field, nn.Module):
            extra_parameters.extend(parameter for parameter in surface_field.parameters() if id(parameter) not in gaussian_ids)
        self.head_parameters = _unique_parameters(extra_parameters)
        self.head_optimizer = torch.optim.Adam(self.head_parameters, lr=head_lr, eps=1e-15)

        phase_cfg = config["phases"]
        self.scheduler = PhaseScheduler(
            semantic_from=int(phase_cfg["semantic_from"]),
            joint_from=int(phase_cfg["joint_from"]),
            surface_from=int(phase_cfg["surface_from"]),
            total_iterations=int(config["optimization"]["iterations"]),
            ramp_iterations=int(phase_cfg.get("ramp_iterations", 1000)),
        )
        self.pareto = PhotometricParetoGuard(bool(config["loss"].get("pareto_guard", True)))
        self.density = DensityController(config["density"], float(scene.cameras_extent), policy_bank=policy_bank)
        self.logger = JsonlLogger(None if self.output_path is None else self.output_path / "training.jsonl")
        background = 1.0 if bool(model_cfg.get("white_background", False)) else 0.0
        self.background = torch.full((3,), background, dtype=torch.float32, device=self.device)
        self.last_metrics: dict[str, float] = {}
        self._camera_stack_ids: list[int] = []

    @property
    def num_semantic_classes(self) -> int:
        return self.loss_system.num_classes

    def state_dict(self, iteration: int) -> dict[str, Any]:
        return {
            "version": 3,
            "iteration": int(iteration),
            "gaussians": self.gaussians.capture(),
            "semantic_heads": self.loss_system.state_dict(),
            "geometry_evidence": self.loss_system.evidence_state_dict(),
            "head_optimizer": self.head_optimizer.state_dict(),
            "density": self.density.state_dict(self.gaussians),
            "mesh_feedback": (
                None
                if self.mesh_feedback is None
                else self.mesh_feedback.state_dict()
            ),
            "rng": capture_rng_state(),
            "camera_stack_ids": list(self._camera_stack_ids),
            "config": self.config,
            "num_semantic_classes": self.num_semantic_classes,
            "semantic_label_mapping": getattr(self.scene, "semantic_label_mapping", None),
        }

    def load_state_dict(self, state: dict[str, Any], optimization_config: Any) -> int:
        saved_iteration = validate_training_checkpoint_header(state)
        saved_classes = int(state["num_semantic_classes"])
        if saved_classes != self.num_semantic_classes:
            raise ValueError(
                "semantic class count differs from checkpoint: "
                f"{self.num_semantic_classes} != {saved_classes}"
            )
        current_mapping = getattr(self.scene, "semantic_label_mapping", None)
        if state["semantic_label_mapping"] != current_mapping:
            raise ValueError("semantic label mapping differs from checkpoint")
        gaussian_state = state.get("gaussians")
        if not isinstance(gaussian_state, Mapping):
            raise ValueError("checkpoint is missing Gaussian model state")
        if type(gaussian_state.get("format_version")) is not int or int(
            gaussian_state["format_version"]
        ) != 3:
            raise ValueError(
                "checkpoint does not contain Gaussian model schema version 3; "
                "start a fresh run"
            )
        registry_state = gaussian_state.get("registry")
        registry_version = getattr(self.gaussians.registry, "FORMAT_VERSION", None)
        if (
            not isinstance(registry_state, Mapping)
            or type(registry_state.get("format_version")) is not int
            or registry_state["format_version"] != registry_version
        ):
            raise ValueError(
                "checkpoint does not contain the current Gaussian attribute registry; "
                "start a fresh run"
            )
        self.gaussians.restore(gaussian_state, optimization_config)
        self.loss_system.load_state_dict(state["semantic_heads"])
        self.loss_system.load_evidence_state_dict(state["geometry_evidence"])
        self.head_optimizer.load_state_dict(state["head_optimizer"])
        self.density.load_state_dict(state["density"], self.gaussians)
        if self.mesh_feedback is not None:
            mesh_feedback_state = state["mesh_feedback"]
            if mesh_feedback_state is None:
                raise ValueError("checkpoint is missing mesh-feedback state")
            self.mesh_feedback.load_state_dict(
                mesh_feedback_state,
                device=self.device,
                dtype=self.gaussians.get_xyz.dtype,
            )
        self.gaussian_parameters = _parameters_from_optimizer(self.gaussians.optimizer)
        self.semantic_parameters = list(self.gaussians.semantic_parameters())
        self.neighbor_index.invalidate()
        self._camera_stack_ids = [int(value) for value in state["camera_stack_ids"]]
        restore_rng_state(state["rng"])
        return saved_iteration

    def _backward(self, phase: Phase, photometric: torch.Tensor, auxiliary: torch.Tensor) -> None:
        if phase is Phase.BOOTSTRAP or not auxiliary.requires_grad or float(auxiliary.detach()) == 0.0:
            photometric.backward()
            return
        if phase is Phase.SEMANTIC_LIFT:
            # Semantic lift learns embedding/decoder while the RGB objective
            # keeps optimizing core Gaussians. Auxiliary gradients are routed
            # only to semantic parameters, so geometry is truly stop-gradient.
            targets = _unique_parameters([*self.semantic_parameters, *self.head_parameters])
            photometric.backward(retain_graph=True)
            gradients = torch.autograd.grad(auxiliary, targets, allow_unused=True)
            for parameter, gradient in zip(targets, gradients):
                if gradient is not None:
                    parameter.grad = gradient.detach() if parameter.grad is None else parameter.grad + gradient.detach()
            return
        self.pareto.backward(photometric, auxiliary, self.gaussian_parameters)

    def _semantic_residual(self, package: dict[str, torch.Tensor], camera) -> torch.Tensor | None:
        return self.loss_system.semantic_residual(package["semantic"], camera)

    @torch.no_grad()
    def _region_memberships(self, candidate_indices: torch.Tensor):
        """Resolve soft foreground memberships only for density candidates."""

        return self.gaussians.point_region_memberships(
            candidate_indices,
            top_k=self.region_top_k,
            chunk_size=self.region_decode_chunk_size,
        )

    def _geometry_residual(self, package: dict[str, torch.Tensor], camera) -> torch.Tensor | None:
        if "normal" not in package or "expected_depth" not in package:
            return None
        residual, _ = depth_normal_residual(
            camera,
            package["expected_depth"],
            package["normal"],
            package.get("alpha"),
            alpha_threshold=self.loss_system.normal_alpha_threshold,
        )
        return residual

    def _surface_losses(
        self,
        iteration: int,
        phase: Phase,
        render_package: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            phase is not Phase.SURFACE_REFINE
            or self.surface_field is None
            or not bool(self.config["surface"].get("enabled", True))
        ):
            return None, None
        cfg = self.config["surface"]
        interval = max(int(cfg.get("consistency_interval", 1)), 1)
        surface_batch = None
        if iteration % interval == 0:
            surface_batch = prepare_gaussian_surface_consistency(
                self.gaussians,
                sample_points=int(cfg.get("sample_points", 8192)),
                region_top_k=self.region_top_k,
                region_decode_chunk_size=self.region_decode_chunk_size,
                minimum_region_weight=float(cfg["region_surface_min_weight"]),
            )
        mesh_batch = (
            None
            if self.mesh_feedback is None
            else self.mesh_feedback.prepare(
                iteration,
                self.gaussians,
                render_package=render_package,
            )
        )

        if surface_batch is None and mesh_batch is None:
            return None, None

        reference = self.gaussians.get_xyz
        global_points = (
            reference.new_empty((0, 3))
            if mesh_batch is None
            else mesh_batch.query_points
        )
        regional_points = (
            reference.new_empty((0, 3))
            if surface_batch is None
            else surface_batch.query_points
        )
        regional_ids = (
            torch.empty(
                (0, self.region_top_k),
                dtype=torch.long,
                device=reference.device,
            )
            if surface_batch is None
            else surface_batch.region_ids.repeat((3, 1))
        )
        shared = self.surface_field.query_partitioned(
            global_points,
            regional_points,
            regional_ids,
        )

        surface_loss = None
        if surface_batch is not None:
            surface_loss, _ = gaussian_surface_consistency(
                self.gaussians,
                self.surface_field,
                prepared=surface_batch,
                query_result=shared.point_regions,
            )
        mesh_loss = None
        if mesh_batch is not None and self.mesh_feedback is not None:
            mesh_loss = self.mesh_feedback.loss(
                iteration,
                self.gaussians,
                self.surface_field,
                prepared=mesh_batch,
                query_result=shared.global_field,
            )
        return surface_loss, mesh_loss

    def _refresh_neighbor_index(self, iteration: int) -> None:
        """Refresh moved centers once for every shared neighbor consumer."""

        interval = max(
            int(self.config["surface"].get("spatial_index_refresh_interval", 100)),
            1,
        )
        if iteration % interval == 0:
            self.neighbor_index.refresh(force=True)

    @torch.no_grad()
    def _semantic_geometry_diagnostics(self) -> dict[str, float]:
        """Estimate coverage and expert certainty on a bounded uniform sample."""

        count = int(self.gaussians.get_xyz.shape[0])
        if count == 0:
            return {}
        sample_count = min(count, 65_536)
        indices = (
            torch.arange(sample_count, device=self.device, dtype=torch.long)
            * count
            // sample_count
        )
        threshold = float(
            self.config.get("semantic", {}).get(
                "geometry_confidence_threshold",
                0.35,
            )
        )
        values: dict[str, float] = {"diagnostic_sample_size": float(sample_count)}

        def sampled_confidence(source: torch.Tensor | None) -> torch.Tensor | None:
            if isinstance(source, torch.Tensor) and source.shape[0] == count:
                selected = source.index_select(0, indices).reshape(sample_count, -1)
                return selected.max(dim=-1).values
            return None

        direct = sampled_confidence(
            getattr(self.gaussians, "semantic_confidence", None)
        )
        propagated = sampled_confidence(
            getattr(self.gaussians, "propagated_semantic_confidence", None)
        )
        if direct is not None:
            values["semantic_direct_coverage"] = float(
                (direct >= threshold).float().mean()
            )
        if propagated is not None:
            values["semantic_propagated_coverage"] = float(
                (propagated >= threshold).float().mean()
            )
        if direct is not None or propagated is not None:
            effective = direct if propagated is None else propagated
            if direct is not None and propagated is not None:
                effective = torch.maximum(direct, propagated)
            assert effective is not None
            values["semantic_effective_coverage"] = float(
                (effective >= threshold).float().mean()
            )
        logits = getattr(self.gaussians, "get_geometry_logits", None)
        if isinstance(logits, torch.Tensor) and logits.shape[0] == count:
            posterior = logits.index_select(0, indices).float().softmax(dim=-1)
            normalizer = math.log(max(posterior.shape[-1], 2))
            entropy = -(posterior * posterior.clamp_min(1e-8).log()).sum(dim=-1)
            values["expert_entropy"] = float((entropy / normalizer).mean())
            values["expert_max_probability"] = float(posterior.max(dim=-1).values.mean())
        return values

    def _log(
        self,
        iteration: int,
        phase: Phase,
        bundle,
        density_report: DensityReport | None,
        elapsed: float,
        profile_metrics: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        values = {name: float(value.detach()) for name, value in bundle.terms.items()}
        values.update(
            iteration=iteration,
            phase=phase.value,
            loss=float(bundle.total.detach()),
            photometric=float(bundle.photometric.detach()),
            auxiliary=float(bundle.auxiliary.detach()),
            gaussians=int(self.gaussians.get_xyz.shape[0]),
            pareto_cosine=float(self.pareto.last_cosine),
            elapsed_seconds=elapsed,
        )
        values.update(self._semantic_geometry_diagnostics())
        if self.mesh_feedback is not None:
            values.update(self.mesh_feedback.diagnostics())
        if density_report is not None:
            values.update(
                density_before=density_report.before,
                density_cloned=density_report.cloned,
                density_split=density_report.split_parents,
                density_split_children=density_report.split_children,
                density_pruned=density_report.pruned,
                density_after=density_report.after,
                density_net_change=density_report.after - density_report.before,
                density_score_mean=density_report.score_mean,
                density_score_threshold=density_report.score_threshold,
            )
        if profile_metrics is not None:
            values.update(profile_metrics)
        self.logger.write(values)
        self.last_metrics = {key: float(value) for key, value in values.items() if isinstance(value, (int, float))}
        return self.last_metrics

    def train(
        self,
        *,
        start_iteration: int = 0,
        save_iterations: Iterable[int] = (),
        save_callback: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> TrainingResult:
        cameras = list(self.scene.getTrainCameras())
        if not cameras:
            raise RuntimeError("training scene contains no cameras")
        optimization = self.config["optimization"]
        density_cfg = self.config["density"]
        total = int(optimization["iterations"])
        save_set = {int(value) for value in save_iterations}
        save_set.add(total)
        logging_config = self.config.get("logging", {})
        log_interval = max(int(logging_config.get("log_interval", 10)), 1)
        profile_interval = int(logging_config.get("profile_interval", 0))
        if start_iteration > 0:
            archived_log = self.logger.rewind(start_iteration)
            if archived_log is not None:
                print(
                    f"[resume] archived abandoned log tail after iteration "
                    f"{start_iteration}: {archived_log}"
                )
        start_time = time.perf_counter()
        progress = tqdm(range(start_iteration + 1, total + 1), desc="Training progress")

        cameras_by_uid = {int(camera.uid): camera for camera in cameras}
        missing_uids = [uid for uid in self._camera_stack_ids if uid not in cameras_by_uid]
        if missing_uids:
            raise ValueError(f"checkpoint camera stack contains unknown IDs: {missing_uids}")
        camera_stack: list[Any] = [cameras_by_uid[uid] for uid in self._camera_stack_ids]
        active_phase: Phase | None = None
        for iteration in progress:
            phase = self.scheduler.phase(iteration)
            if phase is not active_phase:
                set_stage = getattr(self.gaussians, "set_training_stage", None)
                if callable(set_stage):
                    set_stage(phase.value)
                active_phase = phase
            self.gaussians.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()
            if not camera_stack:
                camera_stack = cameras.copy()
            camera = camera_stack.pop(random.randrange(len(camera_stack)))
            self._camera_stack_ids = [int(value.uid) for value in camera_stack]

            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.head_optimizer.zero_grad(set_to_none=True)
            if bool(optimization.get("random_background", False)):
                background = torch.rand(3, device=self.device)
            else:
                background = self.background
            backend = self.config.get("renderer", {}).get("backend", "auto")
            profiler = (
                _StepProfiler(self.device)
                if _should_profile_step(iteration, profile_interval)
                else None
            )
            if profiler is not None:
                profiler.start()
            package = render(camera, self.gaussians, self.pipeline, background, backend=backend)
            target = camera.original_image.to(self.device)
            target = composite_background(target, getattr(camera, "gt_mask", None), background)
            if profiler is not None:
                profiler.mark("render")

            self._refresh_neighbor_index(iteration)
            surface_loss, mesh_loss = self._surface_losses(
                iteration,
                phase,
                package,
            )
            mesh_coverage_signal = (
                None
                if self.mesh_feedback is None
                else self.mesh_feedback.pop_coverage_signal()
            )
            if profiler is not None:
                profiler.mark("surface")
            bundle = self.loss_system(
                package,
                camera,
                self.gaussians,
                self.scheduler.weights(iteration),
                surface_loss,
                mesh_loss,
                iteration,
                target,
            )
            self._backward(phase, bundle.photometric, bundle.auxiliary)
            if profiler is not None:
                profiler.mark("backward")

            density_lifecycle = _density_lifecycle(
                iteration,
                density_cfg,
                surface_config=self.config["surface"],
            )
            if density_lifecycle.observe:
                assert density_lifecycle.window is not None
                self.density.activate_window(
                    self.gaussians,
                    density_lifecycle.window,
                )
                semantic_topology = (
                    self.density.semantic_guidance_enabled
                    and phase in {Phase.JOINT_GEOMETRY, Phase.SURFACE_REFINE}
                )
                self.density.observe(
                    self.gaussians,
                    package,
                    camera if semantic_topology else None,
                    rgb_residual=(package["render"] - target).abs().mean(0),
                    semantic_residual=self._semantic_residual(package, camera) if semantic_topology else None,
                    geometry_error=self._geometry_residual(package, camera) if semantic_topology else None,
                )
                if mesh_coverage_signal is not None:
                    coverage_indices, coverage_residual, coverage_valid = (
                        mesh_coverage_signal
                    )
                    self.density.observe_mesh_coverage(
                        self.gaussians,
                        coverage_indices,
                        coverage_residual,
                        coverage_valid,
                    )

            self.gaussians.optimizer.step()
            self.head_optimizer.step()
            density_report = None
            if density_lifecycle.topology_step:
                density_report = self.density.step(
                    self.gaussians,
                    region_membership_resolver=(
                        self._region_memberships
                        if self.density.semantic_guidance_enabled
                        and phase in {Phase.JOINT_GEOMETRY, Phase.SURFACE_REFINE}
                        else None
                    ),
                    percent_dense=float(optimization.get("percent_dense", 0.01)),
                    enable_size_pruning=density_lifecycle.enable_size_pruning,
                    topology_budget=density_lifecycle.topology_budget,
                )
                # Registry replacement changes Parameter identities.
                self.gaussian_parameters = _parameters_from_optimizer(self.gaussians.optimizer)
                semantic_parameters = getattr(self.gaussians, "semantic_parameters", None)
                if callable(semantic_parameters):
                    self.semantic_parameters = list(semantic_parameters())
                self.loss_system.invalidate_geometry_evidence()
                self.neighbor_index.refresh(force=True)
            if density_lifecycle.reset_opacity:
                self.density.reset_opacity(self.gaussians)

            if self.mesh_feedback is not None:
                refresh_horizon = max(
                    int(getattr(self.mesh_feedback, "refresh_interval", 1)),
                    int(getattr(self.mesh_feedback, "blend_iterations", 0)),
                )
                self.mesh_feedback.after_optimizer_step(
                    iteration,
                    self.gaussians,
                    density_report,
                    allow_refresh=(
                        phase is Phase.SURFACE_REFINE
                        and iteration + refresh_horizon < total
                    ),
                )

            profile_metrics = (
                None if profiler is None else profiler.finish("topology")
            )

            elapsed = time.perf_counter() - start_time
            if (
                iteration % log_interval == 0
                or density_report is not None
                or profile_metrics is not None
            ):
                metrics = self._log(
                    iteration,
                    phase,
                    bundle,
                    density_report,
                    elapsed,
                    profile_metrics,
                )
                progress.set_postfix(loss=f"{metrics['loss']:.5f}", n=int(self.gaussians.get_xyz.shape[0]))
            if iteration in save_set and save_callback is not None:
                save_callback(iteration, self.state_dict(iteration))

        return TrainingResult(total, int(self.gaussians.get_xyz.shape[0]), time.perf_counter() - start_time, self.last_metrics)
