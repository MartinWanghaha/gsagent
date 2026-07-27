from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh import (  # noqa: E402
    TriangleMesh,
    chamfer_distance,
    compute_mesh_metrics,
    f_score,
    precision_recall_fscore,
    sample_mesh_surface,
)


def test_identical_point_cloud_metrics_are_perfect():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    metrics = compute_mesh_metrics(points, points, threshold=1e-6)
    assert metrics.chamfer == 0.0
    assert metrics.accuracy == 0.0
    assert metrics.completeness == 0.0
    assert metrics.f_score == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_translation_has_expected_symmetric_distance_and_f_score():
    reference = np.asarray([[0, 0, 0], [0, 1, 0]], dtype=np.float64)
    prediction = reference + np.asarray([0.25, 0, 0])
    assert np.isclose(chamfer_distance(prediction, reference), 0.25)
    score, precision, recall = precision_recall_fscore(
        prediction, reference, threshold=0.2
    )
    assert score == precision == recall == 0.0
    score, precision, recall = precision_recall_fscore(
        prediction, reference, threshold=0.3
    )
    assert score == precision == recall == 1.0
    assert f_score(prediction, reference, threshold=0.3) == 1.0


def test_surface_sampling_is_deterministic_and_stays_on_triangle():
    mesh = TriangleMesh(
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    first = sample_mesh_surface(mesh, 1000, seed=7)
    second = sample_mesh_surface(mesh, 1000, seed=7)
    assert np.array_equal(first, second)
    assert np.allclose(first[:, 2], 0.0)
    assert np.all(first[:, :2] >= 0.0)
    assert np.all(first[:, 0] + first[:, 1] <= 1.0 + 1e-12)


def test_surface_sampling_is_identical_across_area_chunk_sizes():
    mesh = TriangleMesh(
        vertices=np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [2, 0, 0],
                [2, 2, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray(
            [
                [0, 1, 2],
                [0, 3, 4],
                [0, 0, 5],
            ],
            dtype=np.int64,
        ),
    )

    one_face_chunks = sample_mesh_surface(
        mesh,
        4096,
        seed=23,
        face_chunk_size=1,
    )
    single_chunk = sample_mesh_surface(
        mesh,
        4096,
        seed=23,
        face_chunk_size=100,
    )

    assert np.array_equal(one_face_chunks, single_chunk)
