"""GaussianWrapping topology evaluated on the field actually used in training.

This module deliberately separates two responsibilities:

* Gaussian pivots and spatial Delaunay tetrahedra provide the adaptive mesh
  lattice used by GaussianWrapping.
* ``SemanticSurfaceField.sdf`` supplies every sign test and every refined
  surface root.

Semantic classes are decoded only after the geometry has been extracted.  In
particular, class IDs never create Delaunay adjacency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .bounds import MeshSupportPolicy
from .spatial_tetrahedral import (
    SpatialTetrahedralConfig,
    delaunay_chart,
    filter_refined_faces,
    merge_chart_surfaces,
)
from .training_field_extraction import (
    TrainingFieldMeshConfig,
    TrainingFieldMeshExtractor,
    _clean_faces,
    _filter_small_components,
)
from .types import TriangleMesh


ALGORITHM = "training-field-gaussian-wrapping"
SCHEMA_VERSION = 1


def _quaternion_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = F.normalize(quaternion, dim=-1, eps=1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


@dataclass(frozen=True)
class TrainingFieldGaussianWrappingConfig:
    """Resource and quality policy for training-consistent wrapping."""

    max_gaussians: int = 300_000
    global_delaunay: bool = True
    max_core_gaussians: int = 2_500
    max_halo_gaussians: int = 5_000
    halo_spacing_factor: float = 4.0
    pivot_sigma_factor: float = 3.0
    min_pivot_spacing_factor: float = 0.08
    max_pivot_spacing_factor: float = 1.0
    support_sigma_factor: float = 3.0
    min_opacity: float = 0.05
    min_semantic_confidence: float = 0.35
    require_observation: bool = True
    trim_quantile: float = 0.001
    query_chunk_size: int = 2_048
    root_steps: int = 14
    root_tolerance: float = 1e-5
    semantic_decode_chunk_size: int = 8_192
    min_component_faces: int = 64
    max_edge_support_factor: float = 1.25
    max_edge_spacing_factor: float = 4.0
    max_tetra_edge_ratio: float = 30.0
    min_tetra_volume_ratio: float = 1e-5
    max_circumradius_to_edge: float = 4.0
    max_face_aspect_ratio: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "max_gaussians",
            "max_core_gaussians",
            "max_halo_gaussians",
            "query_chunk_size",
            "root_steps",
            "semantic_decode_chunk_size",
            "min_component_faces",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "halo_spacing_factor",
            "pivot_sigma_factor",
            "min_pivot_spacing_factor",
            "max_pivot_spacing_factor",
            "support_sigma_factor",
            "root_tolerance",
            "max_edge_support_factor",
            "max_edge_spacing_factor",
            "max_tetra_edge_ratio",
            "min_tetra_volume_ratio",
            "max_circumradius_to_edge",
            "max_face_aspect_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_pivot_spacing_factor > self.max_pivot_spacing_factor:
            raise ValueError("minimum pivot spacing factor exceeds maximum")
        for name in ("min_opacity", "min_semantic_confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not 0.0 <= float(self.trim_quantile) < 0.5:
            raise ValueError("trim_quantile must lie in [0,0.5)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def tetrahedral_config(self) -> SpatialTetrahedralConfig:
        return SpatialTetrahedralConfig(
            level=0.0,
            max_edge_support_factor=self.max_edge_support_factor,
            max_edge_spacing_factor=self.max_edge_spacing_factor,
            max_tetra_edge_ratio=self.max_tetra_edge_ratio,
            min_tetra_volume_ratio=self.min_tetra_volume_ratio,
            max_circumradius_to_edge=self.max_circumradius_to_edge,
            max_face_edge_support_factor=self.max_edge_support_factor,
            max_face_edge_spacing_factor=self.max_edge_spacing_factor,
            max_face_aspect_ratio=self.max_face_aspect_ratio,
        )


@dataclass(frozen=True)
class _Chart:
    chart_id: int
    core_rows: np.ndarray
    all_rows: np.ndarray


class TrainingFieldGaussianWrappingExtractor:
    """Extract a semantic mesh using GW topology and the trained exact field."""

    def __init__(
        self,
        surface_field: Any,
        gaussians: Any,
        semantic_decoder: Any,
        *,
        config: Optional[TrainingFieldGaussianWrappingConfig] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not callable(getattr(surface_field, "query_geometry", None)):
            raise TypeError("surface field must expose query_geometry")
        if not callable(getattr(surface_field, "query", None)):
            raise TypeError("surface field must expose query")
        if not callable(semantic_decoder):
            raise TypeError("a trained semantic decoder is required")
        self.surface_field = surface_field
        self.gaussians = gaussians
        self.semantic_decoder = semantic_decoder
        self.config = config or TrainingFieldGaussianWrappingConfig()
        self.progress_callback = progress_callback

    @property
    def device(self) -> torch.device:
        return self.gaussians.get_xyz.device

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def _trusted_indices(self) -> torch.Tensor:
        policy = MeshSupportPolicy(
            min_opacity=self.config.min_opacity,
            min_semantic_confidence=self.config.min_semantic_confidence,
            require_observation=self.config.require_observation,
            trim_quantile=0.0,
        )
        selected = policy.selected_indices(self.gaussians)
        if not len(selected):
            raise RuntimeError("no Gaussian passes the trusted support policy")
        if self.config.trim_quantile > 0.0 and len(selected) > 8:
            xyz = self.gaussians.get_xyz.index_select(0, selected)
            lower = torch.quantile(
                xyz, float(self.config.trim_quantile), dim=0
            )
            upper = torch.quantile(
                xyz, 1.0 - float(self.config.trim_quantile), dim=0
            )
            keep = ((xyz >= lower) & (xyz <= upper)).all(dim=1)
            selected = selected[keep]
            if not len(selected):
                raise RuntimeError("robust support trim removed every Gaussian")
        return selected

    def _quality(self, indices: torch.Tensor) -> np.ndarray:
        opacity = self.gaussians.get_opacity.index_select(0, indices).reshape(-1)
        confidence = self.gaussians.get_semantic_confidence.index_select(
            0, indices
        ).reshape(-1)
        geometry = self.gaussians.get_geometry_posterior.index_select(
            0, indices
        ).max(dim=1).values
        boundary = self.gaussians.get_boundary_score.index_select(
            0, indices
        ).reshape(-1)
        observation = self.gaussians.observation_count.index_select(
            0, indices
        ).reshape(-1)
        visibility = 1.0 - torch.exp(-0.5 * observation.clamp_min(0))
        base = (
            opacity.clamp(0, 1)
            * confidence.clamp(0, 1)
            * geometry.clamp(0, 1)
            * visibility.clamp(0, 1)
        ).clamp_min(1e-12).pow(0.25)
        # Boundary Gaussians receive a modest priority without allowing class
        # boundaries to alter spatial connectivity.
        quality = base * (1.0 + 0.20 * boundary.clamp(0, 1))
        return quality.detach().float().cpu().numpy()

    def _select_anchors(
        self,
        trusted: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        count = len(trusted)
        if count <= self.config.max_gaussians:
            return trusted.sort().values, {
                "trusted_gaussians": int(count),
                "anchor_gaussians": int(count),
                "anchor_selection": "all_trusted",
            }
        xyz = (
            self.gaussians.get_xyz.index_select(0, trusted)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        quality = self._quality(trusted)
        lower = xyz.min(axis=0)
        extent = np.maximum(xyz.max(axis=0) - lower, 1e-8)
        target = int(self.config.max_gaussians)
        side = max(1, int(round(target ** (1.0 / 3.0))))
        cells = np.maximum(
            1,
            np.rint(side * extent / np.cbrt(np.prod(extent))).astype(np.int64),
        )
        # Increase the grid until it can provide enough spatially distributed
        # representatives; deterministic score ordering resolves collisions.
        for _ in range(5):
            grid = np.minimum(
                cells - 1,
                np.floor((xyz - lower) / extent * cells).astype(np.int64),
            )
            order = np.lexsort(
                (
                    trusted.detach().cpu().numpy(),
                    -quality,
                    grid[:, 2],
                    grid[:, 1],
                    grid[:, 0],
                )
            )
            ordered_grid = grid[order]
            first = np.ones(len(order), dtype=bool)
            first[1:] = np.any(ordered_grid[1:] != ordered_grid[:-1], axis=1)
            representatives = order[first]
            if len(representatives) >= target or np.max(cells) > 4096:
                break
            cells = np.maximum(cells + 1, np.ceil(cells * 1.25).astype(np.int64))
        if len(representatives) > target:
            representative_grid = grid[representatives]
            spatial_order = np.lexsort(
                (
                    representative_grid[:, 2],
                    representative_grid[:, 1],
                    representative_grid[:, 0],
                )
            )
            evenly = np.linspace(
                0, len(spatial_order) - 1, target, dtype=np.int64
            )
            representatives = representatives[spatial_order[evenly]]
        elif len(representatives) < target:
            used = np.zeros(count, dtype=bool)
            used[representatives] = True
            remaining = np.flatnonzero(~used)
            fill = remaining[
                np.lexsort(
                    (
                        trusted[remaining].detach().cpu().numpy(),
                        -quality[remaining],
                    )
                )[: target - len(representatives)]
            ]
            representatives = np.concatenate((representatives, fill))
        selected = trusted[
            torch.as_tensor(
                representatives,
                device=trusted.device,
                dtype=torch.long,
            )
        ].sort().values
        return selected, {
            "trusted_gaussians": int(count),
            "anchor_gaussians": int(len(selected)),
            "anchor_selection": "spatial_voxel_quality",
            "anchor_grid_cells": cells.tolist(),
        }

    @staticmethod
    def _recursive_cores(
        xyz: np.ndarray,
        maximum: int,
    ) -> list[np.ndarray]:
        leaves: list[np.ndarray] = []
        pending = [np.arange(len(xyz), dtype=np.int64)]
        while pending:
            rows = pending.pop()
            if len(rows) <= maximum:
                leaves.append(np.sort(rows))
                continue
            points = xyz[rows]
            axis = int(np.argmax(points.max(axis=0) - points.min(axis=0)))
            order = rows[np.argsort(points[:, axis], kind="mergesort")]
            middle = len(order) // 2
            pending.append(order[middle:])
            pending.append(order[:middle])
        return leaves

    def _build_charts(
        self,
        xyz: np.ndarray,
        spacing: np.ndarray,
    ) -> tuple[list[_Chart], np.ndarray]:
        from scipy.spatial import cKDTree

        cores = self._recursive_cores(xyz, self.config.max_core_gaussians)
        tree = cKDTree(xyz)
        owner = np.full(len(xyz), -1, dtype=np.int64)
        charts: list[_Chart] = []
        for chart_id, core in enumerate(cores):
            owner[core] = chart_id
        for chart_id, core in enumerate(cores):
            points = xyz[core]
            halo = float(
                self.config.halo_spacing_factor
                * np.median(spacing[core])
            )
            lower = points.min(axis=0) - halo
            upper = points.max(axis=0) + halo
            center = 0.5 * (lower + upper)
            radius = float(np.max(0.5 * (upper - lower)))
            candidates = np.asarray(
                tree.query_ball_point(center, radius, p=np.inf, workers=-1),
                dtype=np.int64,
            )
            if len(candidates):
                inside = (
                    (xyz[candidates] >= lower)
                    & (xyz[candidates] <= upper)
                ).all(axis=1)
                candidates = candidates[inside]
            halo_rows = np.setdiff1d(candidates, core, assume_unique=False)
            if len(halo_rows) > self.config.max_halo_gaussians:
                below = np.maximum(lower[None] - xyz[halo_rows], 0.0)
                above = np.maximum(xyz[halo_rows] - upper[None], 0.0)
                distance = np.linalg.norm(below + above, axis=1)
                order = np.lexsort((halo_rows, distance))
                halo_rows = halo_rows[order[: self.config.max_halo_gaussians]]
            all_rows = np.unique(np.concatenate((core, halo_rows)))
            charts.append(_Chart(chart_id, core, all_rows))
        if np.any(owner < 0):
            raise RuntimeError("spatial partition left anchors without an owner")
        return charts, owner

    @staticmethod
    def _global_chart(anchor_count: int) -> tuple[list[_Chart], np.ndarray]:
        rows = np.arange(anchor_count, dtype=np.int64)
        return [_Chart(0, rows, rows)], np.zeros(anchor_count, dtype=np.int64)

    def _pivots(
        self,
        anchors: torch.Tensor,
        spacing: np.ndarray,
        owner: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xyz = self.gaussians.get_xyz.index_select(0, anchors)
        scale = self.gaussians.get_scaling.index_select(0, anchors)
        rotation = self.gaussians.get_rotation.index_select(0, anchors)
        matrix = _quaternion_matrix(rotation)
        minimum_axis = scale.argmin(dim=1)
        normal = matrix.gather(
            2, minimum_axis[:, None, None].expand(-1, 3, 1)
        ).squeeze(-1)
        normal = F.normalize(normal, dim=1, eps=1e-8)
        sigma = scale.gather(1, minimum_axis[:, None]).squeeze(1)
        spacing_tensor = torch.as_tensor(
            spacing, device=xyz.device, dtype=xyz.dtype
        )
        offset = (sigma * self.config.pivot_sigma_factor).clamp(
            min=spacing_tensor * self.config.min_pivot_spacing_factor,
            max=spacing_tensor * self.config.max_pivot_spacing_factor,
        )
        roles = torch.tensor(
            (-1.0, 0.0, 1.0), device=xyz.device, dtype=xyz.dtype
        )
        points = (
            xyz[:, None]
            + roles[None, :, None] * offset[:, None, None] * normal[:, None]
        ).reshape(-1, 3)
        support = (
            scale.max(dim=1).values * self.config.support_sigma_factor
        )[:, None].expand(-1, 3).reshape(-1)
        pivot_spacing = spacing_tensor[:, None].expand(-1, 3).reshape(-1)
        pivot_owner = np.repeat(owner, 3)
        return (
            points.detach().float().cpu().numpy(),
            support.detach().float().cpu().numpy(),
            pivot_spacing.detach().float().cpu().numpy(),
            pivot_owner,
        )

    @torch.no_grad()
    def _query_sdf(self, points: np.ndarray) -> np.ndarray:
        values: list[np.ndarray] = []
        for start in range(0, len(points), self.config.query_chunk_size):
            end = min(start + self.config.query_chunk_size, len(points))
            query = self.surface_field.query_geometry(
                torch.as_tensor(
                    points[start:end],
                    device=self.device,
                    dtype=torch.float32,
                ),
                chunk_size=self.config.query_chunk_size,
            )
            values.append(query.sdf.detach().float().cpu().numpy())
            if end == len(points) or end % (self.config.query_chunk_size * 50) == 0:
                self._progress(f"[gw] queried pivots {end:,}/{len(points):,}")
        return np.concatenate(values).reshape(-1)

    @torch.no_grad()
    def _refine_roots(
        self,
        points: np.ndarray,
        phi: np.ndarray,
        edge_keys: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        roots = np.empty((len(edge_keys), 3), dtype=np.float32)
        interpolation = np.empty(len(edge_keys), dtype=np.float32)
        valid = np.zeros(len(edge_keys), dtype=bool)
        residual_parts: list[np.ndarray] = []
        for start in range(0, len(edge_keys), self.config.query_chunk_size):
            end = min(start + self.config.query_chunk_size, len(edge_keys))
            edges = edge_keys[start:end]
            a = points[edges[:, 0]].astype(np.float32, copy=True)
            b = points[edges[:, 1]].astype(np.float32, copy=True)
            fa = phi[edges[:, 0]].astype(np.float32, copy=True)
            fb = phi[edges[:, 1]].astype(np.float32, copy=True)
            reverse = fa > 0.0
            low = np.where(reverse[:, None], b, a)
            high = np.where(reverse[:, None], a, b)
            flow = np.where(reverse, fb, fa)
            fhigh = np.where(reverse, fa, fb)
            bracket = (
                np.isfinite(flow)
                & np.isfinite(fhigh)
                & (flow <= 0.0)
                & (fhigh >= 0.0)
            )
            midpoint = 0.5 * (low + high)
            fmid = np.full(len(edges), np.nan, dtype=np.float32)
            for _ in range(self.config.root_steps):
                midpoint = 0.5 * (low + high)
                query = self.surface_field.query_geometry(
                    torch.as_tensor(
                        midpoint,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    chunk_size=self.config.query_chunk_size,
                )
                fmid = query.sdf.detach().float().cpu().numpy().reshape(-1)
                goes_low = fmid <= 0.0
                low = np.where(goes_low[:, None], midpoint, low)
                high = np.where(goes_low[:, None], high, midpoint)
                if np.nanmax(np.abs(fmid[bracket]), initial=0.0) <= self.config.root_tolerance:
                    break
            midpoint = 0.5 * (low + high)
            edge_vector = b - a
            denominator = np.sum(edge_vector * edge_vector, axis=1)
            t = np.sum((midpoint - a) * edge_vector, axis=1) / np.maximum(
                denominator, np.finfo(np.float32).eps
            )
            roots[start:end] = midpoint
            interpolation[start:end] = np.clip(t, 0.0, 1.0)
            valid[start:end] = bracket & np.isfinite(midpoint).all(axis=1)
            residual_parts.append(np.abs(fmid[bracket]))
            if end == len(edge_keys) or end % (self.config.query_chunk_size * 25) == 0:
                self._progress(f"[gw] refined roots {end:,}/{len(edge_keys):,}")
        residual = (
            np.concatenate(residual_parts)
            if residual_parts
            else np.asarray((np.nan,), dtype=np.float32)
        )
        return roots, valid, interpolation, {
            "root_sdf_abs_mean": float(np.nanmean(residual)),
            "root_sdf_abs_p90": float(np.nanquantile(residual, 0.90)),
            "root_sdf_abs_p99": float(np.nanquantile(residual, 0.99)),
        }

    def extract(self) -> TriangleMesh:
        from scipy.spatial import cKDTree

        trusted = self._trusted_indices()
        anchors, selection_stats = self._select_anchors(trusted)
        xyz = (
            self.gaussians.get_xyz.index_select(0, anchors)
            .detach().float().cpu().numpy()
        )
        if len(xyz) < 4:
            raise RuntimeError("fewer than four trusted Gaussian anchors")
        distance, _ = cKDTree(xyz).query(xyz, k=2, workers=-1)
        spacing = distance[:, 1].astype(np.float32)
        finite_positive = spacing[np.isfinite(spacing) & (spacing > 0)]
        fallback = (
            float(np.median(finite_positive))
            if len(finite_positive)
            else max(float(np.ptp(xyz, axis=0).max()) * 1e-4, 1e-6)
        )
        spacing[~np.isfinite(spacing) | (spacing <= 0)] = fallback
        if self.config.global_delaunay:
            charts, owner = self._global_chart(len(xyz))
        else:
            charts, owner = self._build_charts(xyz, spacing)
        points, support, pivot_spacing, pivot_owner = self._pivots(
            anchors, spacing, owner
        )
        self._progress(
            f"[gw] {len(anchors):,} anchors, {len(points):,} pivots, "
            f"{len(charts):,} spatial charts"
        )
        phi = self._query_sdf(points)
        valid = np.isfinite(phi) & np.isfinite(points).all(axis=1)
        scene_extent = max(float(np.ptp(xyz, axis=0).max()), 1e-6)
        tetra_config = self.config.tetrahedral_config()
        chart_surfaces = []
        tetrahedra = accepted = 0
        for number, chart in enumerate(charts, 1):
            role_rows = np.arange(3, dtype=np.int64)
            pivot_indices = (
                chart.all_rows[:, None] * 3 + role_rows[None]
            ).reshape(-1)
            surface = delaunay_chart(
                chart_id=chart.chart_id,
                pivot_indices=pivot_indices,
                pivot_owner_chart=pivot_owner,
                points=points,
                phi=phi,
                valid=valid,
                support=support,
                spacing=pivot_spacing,
                scene_extent=scene_extent,
                config=tetra_config,
            )
            chart_surfaces.append(surface)
            tetrahedra += surface.tetrahedra
            accepted += surface.accepted_tetrahedra
            if number == len(charts) or number % 10 == 0:
                self._progress(
                    f"[gw] tetrahedralized charts {number:,}/{len(charts):,}; "
                    f"accepted cells {accepted:,}"
                )
        topology = merge_chart_surfaces(chart_surfaces)
        if not len(topology.faces):
            raise RuntimeError("GaussianWrapping topology produced no zero-set faces")
        roots, root_valid, interpolation, root_stats = self._refine_roots(
            points, phi, topology.edge_keys
        )
        topology, quality_stats = filter_refined_faces(
            topology,
            root_vertices=roots,
            root_valid=root_valid,
            root_interpolation=interpolation,
            pivot_support=support,
            pivot_spacing=pivot_spacing,
            scene_extent=scene_extent,
            config=tetra_config,
        )
        faces = _clean_faces(roots, topology.faces)
        faces, component_stats = _filter_small_components(
            faces, minimum_faces=self.config.min_component_faces
        )
        if not len(faces):
            raise RuntimeError("GaussianWrapping quality cleanup removed every face")
        used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        vertices = np.ascontiguousarray(roots[used], dtype=np.float32)
        faces = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int64)

        attribute_helper = TrainingFieldMeshExtractor(
            self.surface_field,
            self.gaussians,
            self.semantic_decoder,
            config=TrainingFieldMeshConfig(
                query_chunk_size=self.config.query_chunk_size,
                semantic_decode_chunk_size=self.config.semantic_decode_chunk_size,
                min_component_faces=self.config.min_component_faces,
            ),
            progress_callback=self.progress_callback,
        )
        normals, semantic, semantic_id, uncertainty, field_stats = (
            attribute_helper._vertex_attributes(vertices)
        )
        faces, flipped = attribute_helper._orient_faces(
            vertices, faces, normals
        )
        labels = semantic_id[faces]
        unanimous = np.all(labels == labels[:, :1], axis=1)
        face_region_id = np.where(
            unanimous, labels[:, 0], -2
        ).astype(np.int32)
        metadata: dict[str, Any] = {
            "algorithm": ALGORITHM,
            "field": "SemanticSurfaceField.query_geometry().sdf",
            "field_level": 0.0,
            **selection_stats,
            "pivots": int(len(points)),
            "spatial_charts": int(len(charts)),
            "delaunay_tetrahedra": int(tetrahedra),
            "crossing_tetrahedra": int(accepted),
            "shared_roots_before_cleanup": int(len(roots)),
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "flipped_faces": int(flipped),
            "mixed_semantic_faces": int(np.count_nonzero(~unanimous)),
            "spacing_median": float(np.median(spacing)),
            "spacing_p90": float(np.quantile(spacing, 0.90)),
            **root_stats,
            **quality_stats,
            **component_stats,
            **field_stats,
        }
        self._progress(
            f"[gw] final mesh: {len(vertices):,} vertices, "
            f"{len(faces):,} faces"
        )
        return TriangleMesh(
            vertices=vertices,
            faces=faces,
            normals=normals,
            semantic=semantic,
            semantic_id=semantic_id,
            uncertainty=uncertainty,
            face_region_id=face_region_id,
            metadata=metadata,
        )


__all__ = [
    "ALGORITHM",
    "SCHEMA_VERSION",
    "TrainingFieldGaussianWrappingConfig",
    "TrainingFieldGaussianWrappingExtractor",
]
