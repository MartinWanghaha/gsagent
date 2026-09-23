"""Typed configuration for the optional MV-RoMa correspondence backend.

The legacy RoMa initializer deliberately does not import this module.  Keeping
all parsing and validation here prevents Hydra, shell launchers and runtime
code from developing subtly different defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_MISSING = object()


def config_value(config: Any, name: str, default: Any = _MISSING) -> Any:
    """Read one value from a mapping, DictConfig or simple object."""

    if isinstance(config, Mapping):
        if name in config:
            return config[name]
    elif config is not None and hasattr(config, name):
        return getattr(config, name)
    if default is _MISSING:
        raise KeyError(f"missing MV-RoMa setting: {name}")
    return default


def nested_config(config: Any, name: str) -> Any:
    value = config_value(config, name, None)
    return {} if value is None else value


def as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be boolean, got {value!r}")


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "auto"}:
        return None
    return Path(text).expanduser()


@dataclass(frozen=True)
class PairStoreSettings:
    mode: str = "auto"
    dtype: str = "float32"
    max_ram_gb: float = 8.0
    temp_dir: Path | None = None

    @classmethod
    def from_config(cls, config: Any) -> "PairStoreSettings":
        settings = cls(
            mode=str(config_value(config, "mode", "auto")).lower(),
            dtype=str(config_value(config, "dtype", "float32")).lower(),
            max_ram_gb=float(config_value(config, "max_ram_gb", 8.0)),
            temp_dir=optional_path(config_value(config, "temp_dir", None)),
        )
        if settings.mode not in {"auto", "memory", "mmap"}:
            raise ValueError("pair_store.mode must be auto, memory or mmap")
        if settings.dtype != "float32":
            raise ValueError(
                "pair_store.dtype must be float32; lossy correspondence storage "
                "is intentionally unsupported"
            )
        if not math.isfinite(settings.max_ram_gb) or settings.max_ram_gb <= 0:
            raise ValueError("pair_store.max_ram_gb must be finite and positive")
        return settings


@dataclass(frozen=True)
class SeedBudgetSettings:
    per_source: int
    total: int | None = None
    proposal_batch_size: int = 32768

    @classmethod
    def from_config(
        cls, config: Any, *, legacy_matches_per_ref: int
    ) -> "SeedBudgetSettings":
        per_source_value = config_value(config, "per_source", None)
        per_source = (
            int(legacy_matches_per_ref)
            if per_source_value is None
            else int(per_source_value)
        )
        total_value = config_value(config, "total", None)
        total = None if total_value is None else int(total_value)
        settings = cls(
            per_source=per_source,
            total=total,
            proposal_batch_size=int(
                config_value(config, "proposal_batch_size", 32768)
            ),
        )
        if settings.per_source <= 0:
            raise ValueError("seed_budget.per_source must be positive")
        if settings.total is not None and settings.total <= 0:
            raise ValueError("seed_budget.total must be positive or null")
        if settings.proposal_batch_size <= 0:
            raise ValueError("seed_budget.proposal_batch_size must be positive")
        return settings


@dataclass(frozen=True)
class DedupSettings:
    enabled: bool = True
    voxel_size: float | None = None
    voxel_scale: float = 0.1

    @classmethod
    def from_config(cls, config: Any) -> "DedupSettings":
        voxel_value = config_value(config, "voxel_size", None)
        settings = cls(
            enabled=as_bool(
                config_value(config, "enabled", True), name="dedup.enabled"
            ),
            voxel_size=None if voxel_value is None else float(voxel_value),
            voxel_scale=float(config_value(config, "voxel_scale", 0.1)),
        )
        if settings.voxel_size is not None and settings.voxel_size <= 0:
            raise ValueError("dedup.voxel_size must be positive or null")
        if not math.isfinite(settings.voxel_scale) or settings.voxel_scale <= 0:
            raise ValueError("dedup.voxel_scale must be finite and positive")
        return settings


@dataclass(frozen=True)
class PaperPostprocessSettings:
    mode: str
    confidence_threshold: float
    cycle_threshold_px: float
    nms_radius_px: int
    min_target_views: int
    sampling_strategy: str
    sampling_grid_size_px: int
    visibility_score_weight: float
    triangulation_batch_size: int
    pair_store: PairStoreSettings
    seed_budget: SeedBudgetSettings
    dedup: DedupSettings

    @classmethod
    def from_config(
        cls, mvroma_config: Any, *, legacy_matches_per_ref: int
    ) -> "PaperPostprocessSettings":
        postprocess = nested_config(mvroma_config, "postprocess")
        mode = str(config_value(postprocess, "mode", "legacy")).lower()
        default_confidence = 0.3 if mode == "paper" else float(
            config_value(mvroma_config, "confidence_threshold", 0.5)
        )
        settings = cls(
            mode=mode,
            confidence_threshold=float(
                config_value(postprocess, "confidence_threshold", default_confidence)
            ),
            cycle_threshold_px=float(
                config_value(postprocess, "cycle_threshold_px", 3.0)
            ),
            nms_radius_px=int(config_value(postprocess, "nms_radius_px", 2)),
            min_target_views=int(
                config_value(
                    postprocess,
                    "min_target_views",
                    config_value(mvroma_config, "min_target_views", 2),
                )
            ),
            sampling_strategy=str(
                config_value(postprocess, "sampling_strategy", "score")
            ).lower(),
            sampling_grid_size_px=int(
                config_value(postprocess, "sampling_grid_size_px", 32)
            ),
            visibility_score_weight=float(
                config_value(postprocess, "visibility_score_weight", 1.0)
            ),
            triangulation_batch_size=int(
                config_value(postprocess, "triangulation_batch_size", 32768)
            ),
            pair_store=PairStoreSettings.from_config(
                nested_config(postprocess, "pair_store")
            ),
            seed_budget=SeedBudgetSettings.from_config(
                nested_config(mvroma_config, "seed_budget"),
                legacy_matches_per_ref=legacy_matches_per_ref,
            ),
            dedup=DedupSettings.from_config(nested_config(mvroma_config, "dedup")),
        )
        if settings.mode not in {"paper", "legacy"}:
            raise ValueError("postprocess.mode must be paper or legacy")
        if not 0 <= settings.confidence_threshold <= 1:
            raise ValueError("postprocess.confidence_threshold must be in [0,1]")
        if settings.cycle_threshold_px <= 0:
            raise ValueError("postprocess.cycle_threshold_px must be positive")
        if settings.nms_radius_px < 0:
            raise ValueError("postprocess.nms_radius_px must be non-negative")
        if settings.min_target_views <= 0:
            raise ValueError("postprocess.min_target_views must be positive")
        if settings.sampling_strategy not in {"score", "grid_balanced"}:
            raise ValueError(
                "postprocess.sampling_strategy must be score or grid_balanced"
            )
        if settings.sampling_grid_size_px <= 0:
            raise ValueError("postprocess.sampling_grid_size_px must be positive")
        if not math.isfinite(settings.visibility_score_weight) or (
            settings.visibility_score_weight < 0
        ):
            raise ValueError(
                "postprocess.visibility_score_weight must be finite and non-negative"
            )
        if settings.triangulation_batch_size <= 0:
            raise ValueError("postprocess.triangulation_batch_size must be positive")
        return settings


@dataclass(frozen=True)
class MVRoMaTrainingSettings:
    pgsr_neighbor_strategy: str = "pose"
    fallback_to_pose: bool = True
    min_overlap: float = 0.05
    min_pair_quality: float = 0.0
    max_neighbors: int = 8
    overlap_weight: float = 1.0
    pair_quality_weight: float = 1.0
    pose_rank_weight: float = 1.0

    @classmethod
    def from_config(cls, mvroma_config: Any) -> "MVRoMaTrainingSettings":
        config = nested_config(mvroma_config, "training")
        settings = cls(
            pgsr_neighbor_strategy=str(
                config_value(config, "pgsr_neighbor_strategy", "pose")
            ).lower(),
            fallback_to_pose=as_bool(
                config_value(config, "fallback_to_pose", True),
                name="training.fallback_to_pose",
            ),
            min_overlap=float(config_value(config, "min_overlap", 0.05)),
            min_pair_quality=float(
                config_value(config, "min_pair_quality", 0.0)
            ),
            max_neighbors=int(config_value(config, "max_neighbors", 8)),
            overlap_weight=float(config_value(config, "overlap_weight", 1.0)),
            pair_quality_weight=float(
                config_value(config, "pair_quality_weight", 1.0)
            ),
            pose_rank_weight=float(config_value(config, "pose_rank_weight", 1.0)),
        )
        if settings.pgsr_neighbor_strategy not in {"pose", "hybrid"}:
            raise ValueError(
                "training.pgsr_neighbor_strategy must be pose or hybrid"
            )
        if not 0 <= settings.min_overlap <= 1:
            raise ValueError("training.min_overlap must be in [0,1]")
        if not 0 <= settings.min_pair_quality <= 1:
            raise ValueError("training.min_pair_quality must be in [0,1]")
        if settings.max_neighbors < 0:
            raise ValueError("training.max_neighbors must be non-negative")
        if min(
            settings.overlap_weight,
            settings.pair_quality_weight,
            settings.pose_rank_weight,
        ) < 0:
            raise ValueError("hybrid neighbor weights must be non-negative")
        return settings


def resolve_group_budget(policy: str, camera_count: int, fixed: Any = None) -> int:
    """Resolve a primary-group budget without overloading ``null`` semantics."""

    normalized = str(policy).lower()
    if normalized in {"auto", "one_per_source"}:
        return camera_count
    if normalized == "paper_half":
        return max(camera_count, math.ceil(0.5 * camera_count * math.sqrt(camera_count)))
    if normalized == "paper_full":
        return max(camera_count, math.ceil(camera_count * math.sqrt(camera_count)))
    if normalized == "fixed":
        if fixed is None:
            raise ValueError("group_planner.fixed_budget is required for fixed policy")
        budget = int(fixed)
        if budget < camera_count:
            raise ValueError("fixed group budget must be at least camera_count")
        return budget
    raise ValueError(
        "group_planner.budget_policy must be one_per_source, paper_half, "
        "paper_full or fixed"
    )
