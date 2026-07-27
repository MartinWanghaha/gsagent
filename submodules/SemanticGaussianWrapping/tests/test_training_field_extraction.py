from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mesh.sampling import Bounds
from mesh.training_field_extraction import (
    SparseBlockLayout,
    TrainingFieldMeshConfig,
    TrainingFieldMeshExtractor,
)


class _SphereField:
    def __init__(self, radius: float = 0.6) -> None:
        self.radius = float(radius)
        self.maximum_query_rows = 0

    def _result(self, points: torch.Tensor, semantic_dim: int):
        self.maximum_query_rows = max(self.maximum_query_rows, len(points))
        length = points.norm(dim=1)
        sdf = length - self.radius
        normal = points / length.clamp_min(1e-8)[:, None]
        semantic = points.new_zeros((len(points), semantic_dim))
        semantic[:, 0] = 1.0
        if semantic_dim > 1:
            semantic[:, 1] = points[:, 2]
        return SimpleNamespace(
            occupancy=torch.sigmoid(-10.0 * sdf),
            sdf=sdf,
            normal=normal,
            semantic=semantic,
            geometry_posterior=points.new_full((len(points), 5), 0.2),
            uncertainty=points.new_full((len(points),), 0.1),
            local_scale=points.new_ones(len(points)),
        )

    def query_geometry(self, points: torch.Tensor, chunk_size: int | None = None):
        del chunk_size
        return self._result(points, 1)

    def query(self, points: torch.Tensor, chunk_size: int | None = None):
        del chunk_size
        return self._result(points, 2)


class _RecordingDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
                    dtype=torch.float32,
                )
            )
        self.seen_dimensions: list[int] = []

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        self.seen_dimensions.append(int(embedding.shape[1]))
        return self.linear(embedding)


def _gaussians(
    *,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    count = 3
    xyz = torch.zeros(count, 3)
    rotation = torch.zeros(count, 4)
    rotation[:, 0] = 1.0
    return SimpleNamespace(
        get_xyz=xyz,
        get_scaling=torch.tensor([scale], dtype=torch.float32).repeat(count, 1),
        get_rotation=rotation,
        get_opacity=torch.full((count, 1), 0.9),
        get_semantic_confidence=torch.full((count, 1), 0.9),
        observation_count=torch.ones(count, 1),
    )


def _edge_incidence(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]),
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0, return_counts=True)[1]


def test_exact_sphere_extracts_closed_low_residual_semantic_mesh() -> None:
    field = _SphereField(radius=0.6)
    decoder = _RecordingDecoder()
    config = TrainingFieldMeshConfig(
        resolution=32,
        block_cells=8,
        support_halo="none",
        query_chunk_size=64,
        semantic_decode_chunk_size=128,
        projection_steps=3,
        projection_tolerance_voxels=1e-5,
        min_component_faces=1,
    )
    extractor = TrainingFieldMeshExtractor(
        field,
        _gaussians(),
        decoder,
        config=config,
        bounds=Bounds(
            np.asarray([-1.0, -1.0, -1.0]),
            np.asarray([1.0, 1.0, 1.0]),
        ),
    )

    mesh, _ = extractor.extract()

    assert mesh is not None
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
    assert np.all(_edge_incidence(mesh.faces) == 2)
    radius = np.linalg.norm(mesh.vertices, axis=1)
    assert np.quantile(np.abs(radius - 0.6), 0.99) < 1e-4
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    triangles = mesh.vertices[mesh.faces]
    face_normal = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    assert np.median(
        np.sum(face_normal * mesh.normals[mesh.faces].sum(axis=1), axis=1)
    ) > 0.0
    assert decoder.seen_dimensions and set(decoder.seen_dimensions) == {2}
    assert mesh.semantic.shape == (len(mesh.vertices), 2)
    assert mesh.semantic_id.shape == (len(mesh.vertices),)
    assert field.maximum_query_rows <= config.query_chunk_size


def test_layout_rasterizes_rotated_support_aabb_across_blocks() -> None:
    extractor = TrainingFieldMeshExtractor(
        _SphereField(),
        _gaussians(scale=(0.1, 0.1, 0.9)),
        _RecordingDecoder(),
        config=TrainingFieldMeshConfig(
            resolution=32,
            block_cells=8,
            support_halo="none",
            support_sigma=1.5,
            min_component_faces=1,
        ),
        bounds=[-2.0, -2.0, -2.0, 2.0, 2.0, 2.0],
    )

    layout = extractor.build_layout()
    active = {tuple(value) for value in layout.active_blocks.tolist()}

    assert (2, 2, 0) in active
    assert (2, 2, 3) in active


def test_projection_respects_nonzero_field_level() -> None:
    level = 0.1
    field = _SphereField(radius=0.6)
    extractor = TrainingFieldMeshExtractor(
        field,
        _gaussians(),
        _RecordingDecoder(),
        config=TrainingFieldMeshConfig(
            resolution=24,
            block_cells=8,
            support_halo="none",
            level=level,
            query_chunk_size=128,
            projection_steps=3,
            projection_tolerance_voxels=1e-5,
            min_component_faces=1,
        ),
        bounds=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
    )

    mesh, _ = extractor.extract()

    assert mesh is not None
    residual = np.abs(np.linalg.norm(mesh.vertices, axis=1) - 0.6 - level)
    assert np.quantile(residual, 0.99) < 1e-4


def test_coarse_scout_keeps_closed_surface_narrow_band() -> None:
    field = _SphereField(radius=0.6)
    extractor = TrainingFieldMeshExtractor(
        field,
        _gaussians(),
        _RecordingDecoder(),
        config=TrainingFieldMeshConfig(
            resolution=64,
            block_cells=8,
            support_halo="full",
            scout_resolution=16,
            scout_near_surface_voxels=0.5,
            query_chunk_size=128,
            projection_steps=1,
            projection_tolerance_voxels=1e-5,
            min_component_faces=1,
        ),
        bounds=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
    )

    mesh, layout = extractor.extract()

    assert mesh is not None
    assert len(layout.active_blocks) < int(np.prod(layout.blocks_per_axis))
    assert layout.scout_stats is not None
    assert layout.scout_stats["crossing_cells"] > 0
    assert np.all(_edge_incidence(mesh.faces) == 2)


def test_boundary_crossings_dynamically_complete_missing_neighbor_blocks() -> None:
    field = _SphereField(radius=0.6)
    extractor = TrainingFieldMeshExtractor(
        field,
        _gaussians(),
        _RecordingDecoder(),
        config=TrainingFieldMeshConfig(
            resolution=32,
            block_cells=8,
            support_halo="none",
            query_chunk_size=128,
            projection_steps=0,
            min_component_faces=1,
        ),
        bounds=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
    )
    layout = SparseBlockLayout(
        bounds=Bounds(
            np.asarray([-1.0, -1.0, -1.0]),
            np.asarray([1.0, 1.0, 1.0]),
        ),
        blocks_per_axis=np.asarray([4, 4, 4]),
        block_cells=8,
        spacing=np.full(3, 0.0625),
        active_blocks=np.asarray([[2, 2, 2]]),
        trusted_gaussians=3,
    )

    vertices, faces, stats = extractor._sample_zero_set(layout)
    vertices, faces = extractor._weld(vertices, faces, layout.voxel_size)

    assert stats["boundary_added_blocks"] > 0
    assert stats["sampled_blocks"] > stats["initial_blocks"]
    assert np.all(_edge_incidence(faces) == 2)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_config_rejects_nonfinite_field_level(value: float) -> None:
    with pytest.raises(ValueError, match="level"):
        TrainingFieldMeshConfig(level=value)
