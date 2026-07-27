"""Curriculum and continuous loss-weight scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    BOOTSTRAP = "bootstrap"
    SEMANTIC_LIFT = "semantic_lift"
    JOINT_GEOMETRY = "joint_geometry"
    SURFACE_REFINE = "surface_refine"


@dataclass(frozen=True)
class PhaseScheduler:
    semantic_from: int = 7_000
    joint_from: int = 12_000
    surface_from: int = 24_000
    total_iterations: int = 30_000
    ramp_iterations: int = 1_000

    def __post_init__(self) -> None:
        boundaries = (0, self.semantic_from, self.joint_from, self.surface_from, self.total_iterations)
        if tuple(sorted(boundaries)) != boundaries:
            raise ValueError(f"Phase boundaries must be ordered, got {boundaries}")

    def phase(self, iteration: int) -> Phase:
        if iteration < self.semantic_from:
            return Phase.BOOTSTRAP
        if iteration < self.joint_from:
            return Phase.SEMANTIC_LIFT
        if iteration < self.surface_from:
            return Phase.JOINT_GEOMETRY
        return Phase.SURFACE_REFINE

    def ramp(self, iteration: int, start: int) -> float:
        return max(0.0, min(1.0, (iteration - start + 1) / max(self.ramp_iterations, 1)))

    def weights(self, iteration: int) -> dict[str, float]:
        semantic = self.ramp(iteration, self.semantic_from)
        geometry = self.ramp(iteration, self.joint_from)
        surface = self.ramp(iteration, self.surface_from)
        return {
            "rgb": 1.0,
            "region_rgb": geometry,
            "semantic": semantic,
            "boundary": semantic,
            "manifold": geometry,
            "geometry": geometry,
            "sh": geometry,
            "surface": surface,
            "mesh": surface,
        }
