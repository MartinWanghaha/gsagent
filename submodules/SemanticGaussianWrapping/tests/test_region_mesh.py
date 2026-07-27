from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mesh import (
    FieldContractError,
    RegionAwareSemanticMeshExtractor,
    SurfaceFieldAdapter,
    TriangleMesh,
)


class _RegionField(torch.nn.Module):
    def __init__(self, support: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("anchor", torch.zeros(()))
        self.support = torch.as_tensor(support, dtype=torch.float32)
        self.query_chunk_size = 64
        self.ownership_calls = 0
        self.returned_region_ids = None

    def query(self, points, chunk_size=None):
        return self._global(points)

    @staticmethod
    def _global(points):
        count = len(points)
        return SimpleNamespace(
            occupancy=points.new_full((count,), 0.5),
            sdf=points.new_zeros(count),
            normal=points.new_tensor([0.0, 0.0, 1.0]).expand(count, -1),
            semantic=points.new_zeros((count, 2)),
            geometry_posterior=points.new_full((count, 5), 0.2),
            uncertainty=points.new_zeros(count),
        )

    def query_region_ownership(self, points, *, region_ids, chunk_size=None):
        self.ownership_calls += 1
        assert region_ids.dtype == torch.long
        assert chunk_size == 64
        support = self.support.to(points.device)
        assert support.shape == (len(points), len(region_ids))
        confidence, winner = support.max(dim=1)
        valid = confidence > 0
        owner = region_ids.index_select(0, winner)
        owner = torch.where(valid, owner, torch.full_like(owner, -1))
        returned_ids = (
            region_ids if self.returned_region_ids is None else self.returned_region_ids
        )
        return SimpleNamespace(
            requested_region_ids=returned_ids,
            region_id=owner,
            confidence=torch.where(valid, confidence, torch.zeros_like(confidence)),
            valid=valid,
        )


def _global_mesh() -> TriangleMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return TriangleMesh(
        vertices=vertices,
        faces=np.array([[0, 1, 2], [2, 3, 4], [3, 4, 5]], dtype=np.int64),
        normals=np.tile([0.0, 0.0, 1.0], (6, 1)),
        semantic=np.zeros((6, 2), dtype=np.float32),
        semantic_id=np.full(6, -1, dtype=np.int32),
        uncertainty=np.zeros(6, dtype=np.float32),
    )


def test_adapter_region_ownership_is_sparse_and_single_call():
    support = np.array([[0.8, 0.2], [0.0, 0.9]], dtype=np.float32)
    field = _RegionField(support)
    adapter = SurfaceFieldAdapter(field, chunk_size=64)

    ownership = adapter.query_region_ownership(
        np.zeros((2, 3), dtype=np.float32),
        region_ids=np.array([3, 7]),
    )

    assert field.ownership_calls == 1
    np.testing.assert_array_equal(ownership.region_id, [3, 7])
    np.testing.assert_allclose(ownership.confidence, [0.8, 0.9])

    field.returned_region_ids = torch.tensor([3, 8], dtype=torch.long)
    with pytest.raises(FieldContractError, match="do not match"):
        adapter.query_region_ownership(
            np.zeros((2, 3), dtype=np.float32),
            region_ids=np.array([3, 7]),
        )


def test_embeddings_never_implicitly_become_semantic_ids():
    mesh = TriangleMesh(
        vertices=np.eye(3, dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        semantic=np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]], dtype=np.float32),
    )
    assert mesh.semantic_id is None

    field = SimpleNamespace(query=lambda points, chunk_size=None: None)
    adapter = SurfaceFieldAdapter(field, input_type="numpy")
    with pytest.raises(FieldContractError, match="requires a scene semantic decoder"):
        adapter.semantic_ids(mesh.semantic)
    with pytest.raises(FieldContractError, match="returned no labels"):
        adapter.semantic_ids(mesh.semantic, decoder=lambda semantic: None)


@pytest.mark.parametrize(
    "region_ids",
    [[], [0, 1], [2, 1], [1, 1], [1.0, 2.0]],
)
def test_region_id_validation_is_strict(region_ids):
    field = _RegionField(np.ones((1, 2), dtype=np.float32))
    adapter = SurfaceFieldAdapter(field)
    with pytest.raises(ValueError):
        adapter.query_region_ownership(
            np.zeros((1, 3), dtype=np.float32), region_ids=region_ids
        )


def test_region_aware_extractor_reuses_global_topology_and_marks_mixed_faces():
    support = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.1, 0.8],
            [0.2, 0.7],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    field = _RegionField(support)
    extractor = RegionAwareSemanticMeshExtractor(
        field, region_ids=[4, 9], min_region_fraction=0.5, query_chunk_size=64
    )
    assert extractor.global_extractor.field.decode_semantics is False
    assert (
        extractor.global_extractor.field.semantic_ids(
            np.zeros((1, 2), dtype=np.float32), decoder=lambda value: np.array([4])
        )
        is None
    )
    source_mesh = _global_mesh()

    class _GlobalExtractor:
        config = extractor.config

        def __init__(self):
            self.calls = 0

        def extract(self, bounds):
            self.calls += 1
            return source_mesh

    global_extractor = _GlobalExtractor()
    extractor.global_extractor = global_extractor
    result = extractor.extract([-1, -1, -1, 4, 2, 1])

    assert global_extractor.calls == 1
    assert field.ownership_calls == 1
    assert result.global_mesh is source_mesh
    np.testing.assert_array_equal(result.vertex_region_id, [4, 4, 4, 9, 9, 9])
    np.testing.assert_array_equal(result.face_region_id, [4, -1, 9])

    region_four = result.region_view(4)
    assert region_four.faces.shape == (1, 3)
    assert region_four.vertices.shape == (3, 3)
    np.testing.assert_allclose(region_four.vertices, source_mesh.vertices[:3])
    assert region_four.metadata["region_id"] == 4


def test_region_threshold_leaves_weak_vertices_unowned():
    field = _RegionField(np.array([[0.49, 0.2]] * 3, dtype=np.float32))
    extractor = RegionAwareSemanticMeshExtractor(
        field, region_ids=[1, 2], min_region_fraction=0.5, query_chunk_size=64
    )
    mesh = TriangleMesh(
        vertices=np.eye(3, dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    extractor.global_extractor = SimpleNamespace(extract=lambda bounds: mesh)

    result = extractor.extract([0, 0, 0, 1, 1, 1])

    np.testing.assert_array_equal(result.vertex_region_id, [-1, -1, -1])
    np.testing.assert_array_equal(result.face_region_id, [-1])
    np.testing.assert_allclose(result.vertex_region_confidence, 0.49)
