"""Region-Conditioned Gaussian Wrapping (RC-GW) mesh extraction.

One renderer-consistent opacity field determines geometry everywhere.  Semantic
posteriors allocate local Delaunay capacity and overlapping charts, but never
replace or calibrate the global zero set.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .gaussian_pivots import (
    GaussianAdaptivePivotBuilder,
    GaussianPivotConfig,
)
from .opacity_field import (
    OpacityFieldConfig,
    RendererOpacityField,
)
from .postprocess import recompute_vertex_normals, simplify_to_face_budget
from .region_atlas import RegionAtlasBuilder, RegionAtlasConfig
from .region_tetrahedral import (
    RegionTetrahedralConfig,
    delaunay_chart,
    filter_invalid_root_faces,
    merge_chart_surfaces,
    refine_shared_roots,
)
from .topology import connected_face_components
from .types import TriangleMesh


ALGORITHM = "region_conditioned_gaussian_wrapping"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RegionGaussianWrappingConfig:
    """Complete quality policy for the single RC-GW extraction path."""

    max_gaussians: int = 500_000
    max_chart_gaussians: int = 12_000
    region_top_k: int = 3
    decoder_chunk_size: int = 32_768
    min_opacity: float = 0.02
    max_gaussian_extent_fraction: float = 0.05
    min_region_gaussians: int = 12
    confident_probability: float = 0.5
    confident_semantic: float = 0.5
    boundary_margin: float = 0.15
    boundary_score_threshold: float = 0.5
    boundary_fraction: float = 0.3
    contact_neighbors: int = 8
    contact_radius_factor: float = 2.5
    chart_halo_factor: float = 1.5
    background_id: Optional[int] = None
    view_stride: int = 1
    camera_scale: float = 1.0
    occupancy_threshold: float = 0.5
    minimum_views: int = 1
    candidate_views: int = 2
    pivot_sigma_factor: float = 1.0
    min_pivot_sigma_to_local: float = 0.05
    max_pivot_sigma_to_local: float = 0.75
    max_crossing_edge_factor: float = 2.0
    binary_steps: int = 10
    query_chunk_size: int = 65_536
    min_component_faces: int = 8
    target_faces: Optional[int] = None

    def __post_init__(self) -> None:
        for name in (
            "max_gaussians",
            "max_chart_gaussians",
            "region_top_k",
            "decoder_chunk_size",
            "min_region_gaussians",
            "contact_neighbors",
            "view_stride",
            "minimum_views",
            "candidate_views",
            "query_chunk_size",
            "min_component_faces",
        ):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_gaussians < 4 or self.max_chart_gaussians < 4:
            raise ValueError("Gaussian budgets must allow a tetrahedron")
        for name in (
            "min_opacity",
            "confident_probability",
            "confident_semantic",
            "boundary_margin",
            "boundary_score_threshold",
            "boundary_fraction",
            "occupancy_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not 0.0 < self.occupancy_threshold < 1.0:
            raise ValueError("occupancy_threshold must lie in (0,1)")
        for name in (
            "max_gaussian_extent_fraction",
            "contact_radius_factor",
            "chart_halo_factor",
            "pivot_sigma_factor",
            "min_pivot_sigma_to_local",
            "max_pivot_sigma_to_local",
            "max_crossing_edge_factor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_gaussian_extent_fraction >= 1.0:
            raise ValueError("max_gaussian_extent_fraction must be smaller than one")
        if self.min_pivot_sigma_to_local > self.max_pivot_sigma_to_local:
            raise ValueError("pivot sigma bounds must satisfy min <= max")
        if (
            not math.isfinite(self.camera_scale)
            or not 0.0 < self.camera_scale <= 1.0
        ):
            raise ValueError("camera_scale must lie in (0,1]")
        if self.binary_steps < 0:
            raise ValueError("binary_steps must be non-negative")
        if self.background_id is not None and self.background_id < 0:
            raise ValueError("background_id must be non-negative or None")
        if self.target_faces is not None and self.target_faces < 1:
            raise ValueError("target_faces must be positive")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "RegionGaussianWrappingConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                "unknown RC-GW mesh_export fields: " + ", ".join(unknown)
            )
        return cls(**dict(values))

    def validated(self) -> "RegionGaussianWrappingConfig":
        return self

    def as_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def atlas_config(self, scene_extent: float) -> RegionAtlasConfig:
        return RegionAtlasConfig(
            max_gaussians=self.max_gaussians,
            max_core_gaussians=self.max_chart_gaussians,
            top_k=self.region_top_k,
            decoder_chunk_size=self.decoder_chunk_size,
            min_opacity=self.min_opacity,
            max_gaussian_extent=(
                scene_extent * self.max_gaussian_extent_fraction
            ),
            min_region_gaussians=self.min_region_gaussians,
            confident_probability=self.confident_probability,
            confident_semantic=self.confident_semantic,
            boundary_margin=self.boundary_margin,
            boundary_score_threshold=self.boundary_score_threshold,
            boundary_fraction=self.boundary_fraction,
            contact_neighbors=self.contact_neighbors,
            contact_radius_factor=self.contact_radius_factor,
            halo_factor=self.chart_halo_factor,
            background_id=self.background_id,
            residual_region_id=-1,
        )

    def pivot_config(self) -> GaussianPivotConfig:
        return GaussianPivotConfig(
            sigma_factor=self.pivot_sigma_factor,
            min_sigma_to_local=self.min_pivot_sigma_to_local,
            max_sigma_to_local=self.max_pivot_sigma_to_local,
        )

    def field_config(self) -> OpacityFieldConfig:
        return OpacityFieldConfig(
            occupancy_threshold=self.occupancy_threshold,
            minimum_views=self.minimum_views,
            candidate_views=self.candidate_views,
            query_chunk_size=self.query_chunk_size,
        )

    def tetrahedral_config(self) -> RegionTetrahedralConfig:
        return RegionTetrahedralConfig(
            max_crossing_edge_factor=self.max_crossing_edge_factor,
            binary_steps=self.binary_steps,
            query_chunk_size=self.query_chunk_size,
        )


def _decode_semantic_ids(
    decoder: Callable[[Tensor], Tensor],
    embedding: Tensor,
    chunk_size: int,
) -> Tensor:
    labels: list[Tensor] = []
    with torch.no_grad():
        for chunk in embedding.split(chunk_size):
            logits = decoder(chunk)
            if logits.ndim != 2 or len(logits) != len(chunk):
                raise ValueError(
                    "semantic decoder must return logits with shape [N,C]"
                )
            labels.append(logits.argmax(dim=1))
    return torch.cat(labels).long()


def _oriented_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    reference_normals: np.ndarray,
) -> np.ndarray:
    if not len(faces):
        return faces
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    target = reference_normals[faces].sum(axis=1)
    flip = np.sum(face_normals * target, axis=1) < 0.0
    result = faces.copy()
    result[flip, 1], result[flip, 2] = (
        faces[flip, 2],
        faces[flip, 1],
    )
    return result


def _orient_reference_normals(
    vertices: np.ndarray,
    normals: np.ndarray,
    camera_centers: np.ndarray,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    if not len(vertices):
        return normals
    _, nearest = cKDTree(camera_centers).query(vertices, k=1, workers=-1)
    towards_camera = camera_centers[nearest] - vertices
    result = normals.copy()
    flip = np.sum(result * towards_camera, axis=1) < 0.0
    result[flip] *= -1.0
    return result


def _region_component_cleanup(
    mesh: TriangleMesh,
    minimum_faces: int,
) -> TriangleMesh:
    """Remove floaters per region while preserving every region's main body."""

    if (
        not len(mesh.faces)
        or mesh.face_region_id is None
        or minimum_faces <= 1
    ):
        return mesh.copy()
    keep: list[np.ndarray] = []
    removed = 0
    preserved_small = 0
    for region_id in np.unique(mesh.face_region_id):
        region_rows = np.flatnonzero(mesh.face_region_id == region_id)
        if int(region_id) == -2:
            keep.append(region_rows)
            continue
        components = connected_face_components(mesh.faces[region_rows])
        if not components:
            continue
        largest = int(np.argmax([len(component) for component in components]))
        for index, component in enumerate(components):
            if len(component) >= minimum_faces or index == largest:
                keep.append(region_rows[component])
                if len(component) < minimum_faces:
                    preserved_small += 1
            else:
                removed += 1
    result = mesh.copy()
    selected = (
        np.sort(np.concatenate(keep))
        if keep
        else np.empty((0,), dtype=np.int64)
    )
    result.faces = result.faces[selected]
    result.face_region_id = result.face_region_id[selected]
    result.metadata["components_removed"] = int(removed)
    result.metadata["small_regions_preserved"] = int(preserved_small)
    return result.compact()


def _region_counts(face_region_id: Optional[np.ndarray]) -> dict[str, int]:
    if face_region_id is None:
        return {}
    ids, counts = np.unique(face_region_id, return_counts=True)
    return {str(int(region)): int(count) for region, count in zip(ids, counts)}


class RegionConditionedGaussianWrappingExtractor:
    """Execute the sole offline RC-GW extraction pipeline."""

    def __init__(
        self,
        context: Any,
        *,
        config: RegionGaussianWrappingConfig | None = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.context = context
        self.config = (
            config or RegionGaussianWrappingConfig()
        ).validated()
        self.progress_callback = progress_callback

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    @torch.no_grad()
    def extract(self) -> TriangleMesh:
        config = self.config
        if self.context.device.type != "cuda":
            raise RuntimeError(
                "RC-GW requires CUDA renderer-consistent point integration"
            )
        cameras = tuple(
            camera.scaled(config.camera_scale)
            for camera in self.context.cameras[:: config.view_stride]
        )
        if len(cameras) < config.minimum_views:
            raise ValueError("selected cameras are fewer than minimum_views")

        self._progress("building coverage-conserving semantic region atlas")
        atlas = RegionAtlasBuilder(
            config.atlas_config(self.context.scene_extent)
        ).build(self.context.gaussians)
        if not atlas.charts:
            raise RuntimeError("semantic region atlas produced no charts")
        self._progress(
            f"atlas: {len(atlas):,} Gaussians, "
            f"{len(atlas.charts):,} bounded charts"
        )

        pivots = GaussianAdaptivePivotBuilder(
            config.pivot_config()
        ).build(self.context.gaussians, atlas)
        self._progress(f"generated {len(pivots):,} Gaussian-adaptive pivots")

        field = RendererOpacityField(
            cameras,
            self.context.gaussians,
            self.context.pipeline,
            config=config.field_config(),
            progress_callback=self.progress_callback,
        )
        pivot_field = field.query(
            pivots.points,
            chunk_size=config.query_chunk_size,
        )
        valid_pivots = int(pivot_field.valid.sum().item())
        self._progress(
            f"global opacity field supports {valid_pivots:,}/{len(pivots):,} pivots"
        )
        if valid_pivots < 4:
            raise RuntimeError(
                "renderer-consistent field supports fewer than four pivots"
            )

        points_cpu = pivots.points.detach().cpu().numpy()
        phi_cpu = pivot_field.phi.detach().cpu().numpy()
        valid_cpu = pivot_field.valid.detach().cpu().numpy()
        radius_cpu = pivots.local_scale.detach().cpu().numpy()
        owner_chart = np.full(len(pivots), -1, dtype=np.int64)
        for chart in atlas.charts:
            owned = pivots.indices_for_gaussians(
                chart.owned_indices
            ).detach().cpu().numpy()
            if np.any(owner_chart[owned] >= 0):
                raise RuntimeError("region atlas assigned a pivot to two charts")
            owner_chart[owned] = chart.chart_id
        if np.any(owner_chart < 0):
            raise RuntimeError("region atlas left pivots without chart ownership")

        tetra_config = config.tetrahedral_config()
        chart_surfaces = []
        tetrahedra_count = 0
        for index, chart in enumerate(atlas.charts, start=1):
            chart_pivots = pivots.indices_for_gaussians(
                chart.gaussian_indices
            ).detach().cpu().numpy()
            surface = delaunay_chart(
                chart_id=chart.chart_id,
                region_id=chart.region_id,
                pivot_indices=chart_pivots,
                pivot_owner_chart=owner_chart,
                points=points_cpu,
                phi=phi_cpu,
                valid=valid_cpu,
                radius=radius_cpu,
                scene_extent=self.context.scene_extent,
                config=tetra_config,
            )
            chart_surfaces.append(surface)
            tetrahedra_count += surface.tetrahedra
            if index == len(atlas.charts) or index % 10 == 0:
                self._progress(
                    f"local Delaunay charts {index}/{len(atlas.charts)}"
                )

        topology = merge_chart_surfaces(chart_surfaces)
        if not len(topology.faces):
            raise RuntimeError(
                "region-conditioned Delaunay produced no crossing faces"
            )
        self._progress(
            f"shared topology: {len(topology.edge_keys):,} roots, "
            f"{len(topology.faces):,} faces"
        )
        roots = refine_shared_roots(
            topology,
            pivot_points=pivots.points,
            pivot_view_ids=pivot_field.view_ids,
            field=field,
            config=tetra_config,
        )
        topology = filter_invalid_root_faces(topology, roots)
        if not len(topology.faces):
            raise RuntimeError(
                "candidate-first root refinement rejected every face"
            )

        edge_indices = torch.as_tensor(
            topology.edge_keys,
            device=pivots.points.device,
            dtype=torch.long,
        )
        interpolation = roots.interpolation
        gaussian_semantic = self.context.gaussians.get_semantic
        pivot_semantic = gaussian_semantic.index_select(
            0,
            pivots.gaussian_indices,
        )
        semantic = F.normalize(
            torch.lerp(
                pivot_semantic[edge_indices[:, 0]],
                pivot_semantic[edge_indices[:, 1]],
                interpolation[:, None],
            ),
            dim=1,
            eps=1e-8,
        )
        semantic_id = _decode_semantic_ids(
            self.context.semantic_decoder,
            semantic,
            config.decoder_chunk_size,
        )
        reference_normal = F.normalize(
            torch.lerp(
                pivots.normals[edge_indices[:, 0]],
                pivots.normals[edge_indices[:, 1]],
                interpolation[:, None],
            ),
            dim=1,
            eps=1e-8,
        )
        semantic_confidence = torch.lerp(
            pivots.membership_confidence[edge_indices[:, 0]],
            pivots.membership_confidence[edge_indices[:, 1]],
            interpolation,
        )
        pivot_quality = torch.lerp(
            pivots.quality[edge_indices[:, 0]],
            pivots.quality[edge_indices[:, 1]],
            interpolation,
        )
        uncertainty = 1.0 - (
            roots.confidence
            * torch.sqrt(
                semantic_confidence.clamp(0.0, 1.0)
                * pivot_quality.clamp(0.0, 1.0)
            )
        ).clamp(0.0, 1.0)

        vertices = roots.vertices.detach().cpu().numpy()
        normal_numpy = reference_normal.detach().cpu().numpy()
        camera_centers = np.stack(
            [camera.camera_center.detach().cpu().numpy() for camera in cameras]
        )
        normal_numpy = _orient_reference_normals(
            vertices,
            normal_numpy,
            camera_centers,
        )
        faces = _oriented_faces(
            vertices,
            topology.faces,
            normal_numpy,
        )
        raw_mesh = TriangleMesh(
            vertices=vertices,
            faces=faces,
            normals=normal_numpy,
            semantic=semantic.detach().cpu().numpy(),
            semantic_id=semantic_id.detach().cpu().numpy(),
            uncertainty=uncertainty.detach().cpu().numpy(),
            face_region_id=topology.face_region_id,
            metadata={
                "algorithm": ALGORITHM,
                "training_views_used": int(len(cameras)),
                "atlas_gaussians": int(len(atlas)),
                "pivots": int(len(pivots)),
                "valid_pivots": valid_pivots,
                "charts": int(len(atlas.charts)),
                "semantic_regions": int(
                    len(torch.unique(atlas.owner_region_ids))
                ),
                "contact_pairs": int(len(atlas.contact_pairs)),
                "tetrahedra": int(tetrahedra_count),
                "shared_roots": int(len(topology.edge_keys)),
                "raw_faces": int(len(topology.faces)),
                "chart_face_count": {
                    str(key): int(value)
                    for key, value in topology.chart_face_count.items()
                },
            },
        ).compact()
        raw_mesh.metadata["raw_vertices"] = int(len(raw_mesh.vertices))
        raw_mesh.metadata["region_face_count_before_cleanup"] = _region_counts(
            raw_mesh.face_region_id
        )

        mesh = _region_component_cleanup(
            raw_mesh,
            config.min_component_faces,
        )
        if config.target_faces is not None:
            mesh = simplify_to_face_budget(mesh, config.target_faces)
        mesh = recompute_vertex_normals(mesh)
        if not len(mesh.faces):
            raise RuntimeError("region-aware cleanup removed every face")
        mesh.metadata["region_face_count"] = _region_counts(
            mesh.face_region_id
        )
        mesh.metadata["vertices"] = int(len(mesh.vertices))
        mesh.metadata["faces"] = int(len(mesh.faces))
        self._progress(
            f"final RC-GW mesh: {len(mesh.vertices):,} vertices, "
            f"{len(mesh.faces):,} faces"
        )
        return mesh


__all__ = [
    "ALGORITHM",
    "SCHEMA_VERSION",
    "RegionConditionedGaussianWrappingExtractor",
    "RegionGaussianWrappingConfig",
]
