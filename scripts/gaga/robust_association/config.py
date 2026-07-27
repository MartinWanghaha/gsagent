"""Configuration for robust Gaussian association."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RobustAssociationConfig:
    front_percentage: float = 0.2
    num_patches: int = 32
    core_radius: int = 2
    boundary_weight: float = 0.25
    depth_decay: float = 2.0
    view_neighbors: int = 12
    candidate_threshold: float = 0.12
    match_threshold: float = 0.28
    match_margin: float = 0.03
    gaussian_weight: float = 0.50
    coverage_weight: float = 0.25
    appearance_weight: float = 0.15
    spatial_weight: float = 0.10
    fragment_merge_threshold: float = 0.48
    fragment_appearance_threshold: float = 0.82
    min_track_views: int = 2
    min_track_quality: float = 0.20
    tentative_min_area_fraction: float = 0.002
    tentative_min_quality: float = 0.80
    tentative_min_gaussians: int = 64
    tentative_propagation_threshold: float = 0.22
    tentative_propagation_margin: float = 0.03
    tentative_min_neighbor_views: int = 2
    tentative_max_neighbor_edges: int = 8
    split_fraction: float = 0.20
    split_min_seed_points: int = 8
    split_seed_purity: float = 0.75
    split_min_area_pixels: int = 1_024
    split_min_area_fraction: float = 0.005
    superpixel_size: int = 32
    superpixel_compactness: float = 10.0
    superpixel_max_edge: int = 1_024
    gaussian_label_margin: float = 0.15
    ignore_label: int = 65_535
    qa_max_ignore_fraction: float = 0.05
    qa_min_region_purity: float = 0.85
    qa_max_label_jump_rate: float = 0.003

    def validate(self) -> None:
        probabilities = {
            "front_percentage": self.front_percentage,
            "boundary_weight": self.boundary_weight,
            "candidate_threshold": self.candidate_threshold,
            "match_threshold": self.match_threshold,
            "match_margin": self.match_margin,
            "fragment_merge_threshold": self.fragment_merge_threshold,
            "fragment_appearance_threshold": self.fragment_appearance_threshold,
            "min_track_quality": self.min_track_quality,
            "tentative_min_area_fraction": self.tentative_min_area_fraction,
            "tentative_min_quality": self.tentative_min_quality,
            "tentative_propagation_threshold": (self.tentative_propagation_threshold),
            "tentative_propagation_margin": self.tentative_propagation_margin,
            "split_fraction": self.split_fraction,
            "split_seed_purity": self.split_seed_purity,
            "split_min_area_fraction": self.split_min_area_fraction,
            "gaussian_label_margin": self.gaussian_label_margin,
            "qa_max_ignore_fraction": self.qa_max_ignore_fraction,
            "qa_min_region_purity": self.qa_min_region_purity,
            "qa_max_label_jump_rate": self.qa_max_label_jump_rate,
        }
        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.front_percentage <= 0:
            raise ValueError("front_percentage must be positive")
        if self.num_patches <= 0 or self.view_neighbors <= 0:
            raise ValueError("num_patches and view_neighbors must be positive")
        if self.core_radius < 0:
            raise ValueError("core_radius cannot be negative")
        if self.depth_decay < 0:
            raise ValueError("depth_decay cannot be negative")
        positive_counts = {
            "min_track_views": self.min_track_views,
            "tentative_min_gaussians": self.tentative_min_gaussians,
            "tentative_min_neighbor_views": self.tentative_min_neighbor_views,
            "tentative_max_neighbor_edges": self.tentative_max_neighbor_edges,
            "split_min_seed_points": self.split_min_seed_points,
            "split_min_area_pixels": self.split_min_area_pixels,
            "superpixel_size": self.superpixel_size,
            "superpixel_max_edge": self.superpixel_max_edge,
        }
        for name, value in positive_counts.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.superpixel_compactness <= 0:
            raise ValueError("superpixel_compactness must be positive")
        cue_sum = (
            self.gaussian_weight
            + self.coverage_weight
            + self.appearance_weight
            + self.spatial_weight
        )
        if cue_sum <= 0:
            raise ValueError("At least one association cue must have positive weight")
        if not 0 <= self.ignore_label <= 65_535:
            raise ValueError("ignore_label must fit uint16")

    def to_dict(self) -> dict:
        return asdict(self)
