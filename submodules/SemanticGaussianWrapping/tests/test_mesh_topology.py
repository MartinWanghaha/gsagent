from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh import (  # noqa: E402
    ContactGraph,
    compatible_pairs,
    TriangleMesh,
    face_compatibility_mask,
    remove_small_components,
    seam_aware_vertex_clustering,
)
def _mixed_triangle() -> TriangleMesh:
    return TriangleMesh(
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        semantic=np.asarray([[1, 0], [0, 1], [0, 1]], dtype=np.float32),
        semantic_id=np.asarray([0, 1, 1], dtype=np.int32),
        uncertainty=np.zeros(3, dtype=np.float32),
    )


def test_contact_graph_preserves_real_contact_but_blocks_unknown_bridge():
    mesh = _mixed_triangle()
    assert not face_compatibility_mask(mesh, cosine_threshold=0.9)[0]

    graph = ContactGraph.from_edges([(0, 1, 0.95)], threshold=0.8)
    assert face_compatibility_mask(mesh, cosine_threshold=0.9, contact_graph=graph)[0]


def test_unknown_embedding_ids_do_not_bypass_cosine_compatibility():
    labels = np.asarray([-1], dtype=np.int32)
    incompatible = compatible_pairs(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[0.0, 1.0]]),
        labels,
        labels,
        cosine_threshold=0.9,
    )
    compatible = compatible_pairs(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[0.99, 0.01]]),
        labels,
        labels,
        cosine_threshold=0.9,
    )
    assert not incompatible[0]
    assert compatible[0]


def test_seam_aware_clustering_does_not_merge_different_labels():
    mesh = TriangleMesh(
        vertices=np.asarray(
            [[0, 0, 0], [0.01, 0, 0], [0, 1, 0], [0.02, 0, 0], [0.03, 0.2, 0]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [1, 3, 4]], dtype=np.int64),
        semantic=np.asarray([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
        semantic_id=np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        uncertainty=np.zeros(5, dtype=np.float32),
    )
    simplified = seam_aware_vertex_clustering(mesh, voxel_size=0.5)
    label_zero = simplified.vertices[simplified.semantic_id == 0]
    label_one = simplified.vertices[simplified.semantic_id == 1]
    assert len(label_zero) > 0 and len(label_one) > 0
    assert set(simplified.semantic_id.tolist()) == {0, 1}


def test_clustering_preserves_decoded_ids_for_continuous_gaga_embeddings():
    mesh = TriangleMesh(
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        semantic=np.asarray([[0.1, 0.9, 0.2, 0.3]] * 3, dtype=np.float32),
        semantic_id=np.asarray([7, 7, 7], dtype=np.int32),
        uncertainty=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    )
    simplified = seam_aware_vertex_clustering(mesh, voxel_size=0.01)
    assert simplified.semantic_id.tolist() == [7, 7, 7]


def test_small_unique_semantic_instance_is_retained():
    mesh = TriangleMesh(
        vertices=np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [10, 0, 0],
                [11, 0, 0],
                [10, 1, 0],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        semantic=np.asarray([[1, 0]] * 3 + [[0, 1]] * 3, dtype=np.float32),
        semantic_id=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
    )
    cleaned = remove_small_components(mesh, min_faces=10, preserve_semantic_instances=True, min_instance_vertices=3)
    assert len(cleaned.faces) == 2
    assert set(cleaned.semantic_id.tolist()) == {0, 1}


def test_uncertainty_gated_cleanup_does_not_trust_missing_uncertainty():
    mesh = TriangleMesh(
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        semantic_id=np.asarray([7, 7, 7], dtype=np.int32),
    )

    cleaned = remove_small_components(
        mesh,
        min_faces=10,
        preserve_semantic_instances=True,
        min_instance_vertices=3,
        preserve_max_uncertainty=0.25,
    )

    assert len(cleaned.faces) == 0


def test_component_cleanup_keeps_face_region_ids_aligned():
    mesh = TriangleMesh(
        vertices=np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [10, 0, 0],
                [11, 0, 0],
                [10, 1, 0],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [1, 3, 2], [4, 5, 6]], dtype=np.int64),
        face_region_id=np.asarray([7, 7, 0], dtype=np.int32),
    )

    cleaned = remove_small_components(
        mesh,
        min_faces=2,
        preserve_semantic_instances=False,
    )

    assert len(cleaned.faces) == 2
    np.testing.assert_array_equal(cleaned.face_region_id, [7, 7])


def test_face_deduplication_marks_conflicting_region_ownership_as_seam():
    mesh = TriangleMesh(
        vertices=np.asarray(
            [
                [0.00, 0.00, 0],
                [1.00, 0.00, 0],
                [0.00, 1.00, 0],
                [0.01, 0.01, 0],
                [1.01, 0.01, 0],
                [0.01, 1.01, 0],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        face_region_id=np.asarray([4, 9], dtype=np.int32),
    )

    simplified = seam_aware_vertex_clustering(mesh, voxel_size=0.1)

    assert len(simplified.faces) == 1
    np.testing.assert_array_equal(simplified.face_region_id, [-2])
