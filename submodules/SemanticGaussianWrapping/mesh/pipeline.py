"""End-to-end semantic mesh extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .extractors import delaunay_marching_tetra, marching_cubes_blocks
from .field import FieldContractError, SurfaceFieldAdapter, as_field_adapter
from .io import export_mesh
from .postprocess import postprocess_mesh
from .sampling import (
    AdaptiveOctreeSampler,
    AdaptiveSamplingConfig,
    BlockedGridSampler,
    Bounds,
)
from .topology import ContactGraph, filter_semantic_topology
from .types import RegionAwareMesh, TriangleMesh


@dataclass
class MeshExtractionConfig:
    method: str = "cubes"
    scalar: str = "sdf"
    level: float = 0.0
    blocks_per_axis: int | tuple[int, int, int] = 4
    block_cells: int = 8
    max_grid_refinement: int = 2
    marching_cubes_backend: str = "skimage"
    octree_min_depth: int = 1
    octree_max_depth: int = 6
    max_tetra_points: int = 150_000
    semantic_cosine_threshold: float = 0.85
    max_edge_length: Optional[float] = None
    min_component_faces: int = 20
    preserve_semantic_instances: bool = True
    min_instance_vertices: int = 3
    background_id: Optional[int] = None
    simplify_voxel_size: Optional[float] = None
    target_faces: Optional[int] = None
    protect_uncertainty_threshold: Optional[float] = 0.25
    sampling: AdaptiveSamplingConfig = field(default_factory=AdaptiveSamplingConfig)

    def __post_init__(self) -> None:
        method_aliases = {
            "cubes": "cubes",
            "marching_cubes": "cubes",
            "tetra": "tetra",
            "marching_tetrahedra": "tetra",
        }
        if self.method not in method_aliases:
            raise ValueError(
                "method must be cubes/marching_cubes or tetra/marching_tetrahedra"
            )
        self.method = method_aliases[self.method]
        if self.scalar not in {"sdf", "occupancy"}:
            raise ValueError("scalar must be sdf or occupancy")
        if self.marching_cubes_backend != "skimage":
            raise ValueError("marching_cubes_backend must be skimage")
        if self.max_tetra_points < 4:
            raise ValueError("max_tetra_points must be at least four")


class SemanticMeshExtractor:
    """Consumes only the unified surface field, never Gaussian internals."""

    def __init__(
        self,
        surface_field: SurfaceFieldAdapter | object,
        *,
        attribute_field: SurfaceFieldAdapter | object | None = None,
        config: Optional[MeshExtractionConfig] = None,
        contact_graph: Optional[ContactGraph] = None,
        semantic_decoder: Optional[Callable[[Any], Any]] = None,
        device: Optional[str] = None,
        query_chunk_size: int = 262_144,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.field = as_field_adapter(
            surface_field, device=device, chunk_size=query_chunk_size
        )
        self.attribute_field = (
            self.field
            if attribute_field is None
            else as_field_adapter(
                attribute_field,
                device=device,
                chunk_size=query_chunk_size,
            )
        )
        self.geometry_field = self.field.geometry_view()
        self.config = config or MeshExtractionConfig()
        self.contact_graph = contact_graph
        self.semantic_decoder = semantic_decoder
        self.progress_callback = progress_callback

    def _select_tetra_samples(self, samples):
        if len(samples) <= self.config.max_tetra_points:
            return samples
        detail = np.max(samples.geometry_posterior[:, 2:], axis=1)
        scalar = getattr(samples, self.config.scalar)
        score = -np.abs(scalar - self.config.level) + samples.uncertainty + detail
        indices = np.argpartition(score, -self.config.max_tetra_points)[
            -self.config.max_tetra_points :
        ]
        return samples.take(np.sort(indices))

    def extract(self, bounds: Bounds | Sequence[float]) -> TriangleMesh:
        bounds = bounds if isinstance(bounds, Bounds) else Bounds.from_array(bounds)
        if self.config.method == "cubes":
            sampler = BlockedGridSampler(
                self.geometry_field,
                bounds,
                decision_field=self.attribute_field,
                blocks_per_axis=self.config.blocks_per_axis,
                block_cells=self.config.block_cells,
                max_refinement=self.config.max_grid_refinement,
                config=self.config.sampling,
                progress_callback=self.progress_callback,
            )
            blocks = sampler.sample_blocks()
            mesh = marching_cubes_blocks(
                blocks,
                self.attribute_field,
                value=self.config.scalar,
                level=self.config.level,
                backend=self.config.marching_cubes_backend,
                semantic_decoder=self.semantic_decoder,
            )
            mesh.metadata["sampled_blocks"] = len(blocks)
            mesh.metadata["halo_blocks"] = int(sampler.last_halo_blocks)
            mesh.metadata["sampled_points"] = int(sum(len(block.samples) for block in blocks))
        else:
            sampler = AdaptiveOctreeSampler(
                self.field,
                bounds,
                min_depth=self.config.octree_min_depth,
                max_depth=self.config.octree_max_depth,
                config=self.config.sampling,
            )
            samples = self._select_tetra_samples(sampler.sample(active_only=True))
            edge_limit = self.config.max_edge_length
            if edge_limit is None:
                edge_limit = bounds.diagonal * 2.5 / (2**self.config.octree_max_depth)
            mesh = delaunay_marching_tetra(
                samples,
                self.field,
                value=self.config.scalar,
                level=self.config.level,
                cosine_threshold=self.config.semantic_cosine_threshold,
                contact_graph=self.contact_graph,
                max_edge_length=edge_limit,
                semantic_decoder=self.semantic_decoder,
            )
            mesh.metadata["sampled_points"] = len(samples)
            mesh.metadata["tetra_edge_limit"] = edge_limit

        mesh = filter_semantic_topology(
            mesh,
            cosine_threshold=self.config.semantic_cosine_threshold,
            contact_graph=self.contact_graph,
            max_edge_length=self.config.max_edge_length,
        )
        mesh = postprocess_mesh(
            mesh,
            min_component_faces=self.config.min_component_faces,
            preserve_semantic_instances=self.config.preserve_semantic_instances,
            min_instance_vertices=self.config.min_instance_vertices,
            background_id=self.config.background_id,
            simplify_voxel_size=self.config.simplify_voxel_size,
            target_faces=self.config.target_faces,
            protect_uncertainty_threshold=self.config.protect_uncertainty_threshold,
        )
        mesh.metadata.update(
            {
                "method": self.config.method,
                "scalar": self.config.scalar,
                "level": self.config.level,
                "semantic_id_source": (
                    "decoder" if self.field.last_semantic_decoded else "unknown"
                ),
            }
        )
        return mesh

    def extract_and_export(
        self, bounds: Bounds | Sequence[float], output: str | Path
    ) -> TriangleMesh:
        mesh = self.extract(bounds)
        export_mesh(mesh, output)
        return mesh


class RegionAwareSemanticMeshExtractor:
    """Assign region ownership to one mesh extracted from the global field."""

    def __init__(
        self,
        surface_field: SurfaceFieldAdapter | object,
        *,
        region_ids: Sequence[int],
        min_region_fraction: float = 0.05,
        config: Optional[MeshExtractionConfig] = None,
        device: Optional[str] = None,
        query_chunk_size: int = 262_144,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        raw_region_ids = np.asarray(region_ids)
        if raw_region_ids.ndim != 1 or raw_region_ids.dtype.kind not in "iu":
            raise ValueError("region_ids must be a one-dimensional integer sequence")
        self.region_ids = np.ascontiguousarray(raw_region_ids, dtype=np.int64)
        if not len(self.region_ids):
            raise ValueError("at least one foreground region ID is required")
        if np.any(self.region_ids <= 0):
            raise ValueError("region_ids must contain foreground IDs greater than zero")
        if np.any(np.diff(self.region_ids) <= 0):
            raise ValueError("region_ids must be sorted and unique")
        if isinstance(min_region_fraction, (bool, np.bool_)) or not np.isscalar(
            min_region_fraction
        ):
            raise TypeError("min_region_fraction must be a scalar")
        self.min_region_fraction = float(min_region_fraction)
        if not np.isfinite(self.min_region_fraction) or not (
            0.0 < self.min_region_fraction <= 1.0
        ):
            raise ValueError("min_region_fraction must lie in (0, 1]")

        self.field = as_field_adapter(
            surface_field, device=device, chunk_size=query_chunk_size
        )
        if not callable(getattr(self.field.field, "query_region_ownership", None)):
            raise FieldContractError(
                "region-aware extraction requires "
                "query_region_ownership(points, *, region_ids)"
            )
        decode_semantic = getattr(self.field.field, "decode_semantic", None)
        if callable(decode_semantic):
            field_decoder = getattr(self.field.field, "semantic_decoder", None)
            gaussians = getattr(self.field.field, "gaussians", None)
            gaussian_decoder = getattr(gaussians, "semantic_decoder", None)
            if not callable(field_decoder) and not callable(gaussian_decoder):
                raise FieldContractError(
                    "region-aware extraction requires the scene semantic decoder"
                )
        geometry_field = self.field.geometry_view()
        self.global_extractor = SemanticMeshExtractor(
            geometry_field,
            attribute_field=self.field,
            config=config,
            query_chunk_size=query_chunk_size,
            progress_callback=progress_callback,
        )

    @property
    def config(self) -> MeshExtractionConfig:
        return self.global_extractor.config

    def extract(self, bounds: Bounds | Sequence[float]) -> RegionAwareMesh:
        global_mesh = self.global_extractor.extract(bounds)
        vertex_count = len(global_mesh.vertices)
        if not vertex_count:
            result = RegionAwareMesh(
                global_mesh=global_mesh,
                region_ids=self.region_ids.copy(),
                vertex_region_id=np.empty((0,), dtype=np.int64),
                vertex_region_confidence=np.empty((0,), dtype=np.float32),
                face_region_id=np.full(len(global_mesh.faces), -1, dtype=np.int64),
            )
            global_mesh.metadata.update(
                {
                    "region_ids": self.region_ids.tolist(),
                    "min_region_fraction": self.min_region_fraction,
                    "mixed_region_faces": int(len(global_mesh.faces)),
                    "region_topology_source": "global_geometry",
                }
            )
            return result

        ownership = self.field.query_region_ownership(
            global_mesh.vertices,
            region_ids=self.region_ids,
        )
        confidence = ownership.confidence
        owned = ownership.valid & (confidence >= self.min_region_fraction)
        vertex_region_id = np.full(vertex_count, -1, dtype=np.int64)
        vertex_region_id[owned] = ownership.region_id[owned]

        if len(global_mesh.faces):
            face_vertex_ids = vertex_region_id[global_mesh.faces]
            unanimous = (face_vertex_ids[:, 0] > 0) & np.all(
                face_vertex_ids == face_vertex_ids[:, :1], axis=1
            )
            face_region_id = np.where(unanimous, face_vertex_ids[:, 0], -1).astype(
                np.int64
            )
        else:
            face_region_id = np.empty((0,), dtype=np.int64)

        global_mesh.metadata.update(
            {
                "region_ids": self.region_ids.tolist(),
                "min_region_fraction": self.min_region_fraction,
                "owned_vertices": int(np.count_nonzero(owned)),
                "mixed_region_faces": int(np.count_nonzero(face_region_id == -1)),
                "region_topology_source": "global_geometry",
            }
        )
        return RegionAwareMesh(
            global_mesh=global_mesh,
            region_ids=self.region_ids.copy(),
            vertex_region_id=vertex_region_id,
            vertex_region_confidence=confidence,
            face_region_id=face_region_id,
        )
