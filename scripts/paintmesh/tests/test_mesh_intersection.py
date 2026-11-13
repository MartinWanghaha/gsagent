import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.mesh_intersection import audit_intersections, classify_triangle_pairs


A = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]])
F = np.array([[0, 1, 2]], dtype=np.int64)


def test_transverse_surface_intersection_segment():
    b = np.array([[0., -.5, -1.], [0., -.5, 1.], [0., .5, 0.]])
    r = audit_intersections(A, F, b, F, 1e-9)
    assert r['complete'] and r['counts']['transverse'] == 1
    assert np.array_equal(r['face_pairs'], [[0, 0]])
    segment = r['segments'][0]
    assert np.allclose(segment[:, [0, 2]], 0)
    assert np.allclose(np.sort(segment[:, 1]), [-.5, .5])
    assert not r['containment_checked'] and not r['global_signed_distance_used']


def test_boundary_point_and_coplanar_are_not_transverse():
    cases = [
        (np.array([[0., -.5, 0.], [0., .5, 0.], [0., -.5, 1.]]), 'boundary_touch'),
        (np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 1.]]), 'point_touch'),
        (np.array([[-.2, -.2, 0.], [.2, -.2, 0.], [0., .2, 0.]]), 'coplanar_overlap'),
        (np.array([[-1., -1., 0.], [1., -1., 0.], [0., -2., 0.]]), 'boundary_touch'),
    ]
    for b, expected in cases:
        r = audit_intersections(A, F, b, F, 1e-9)
        assert r['counts'][expected] == 1, (expected, r)
        assert r['counts']['transverse'] == 0


def test_disjoint_and_containment_are_not_mislabeled():
    b = A + [0, 0, 1]
    r = audit_intersections(A, F, b, F)
    assert len(r['face_pairs']) == 0
    # A surface-only audit explicitly does not claim to decide containment.
    assert not r['containment_checked']


def test_contact_cap_recursion_keeps_every_face_pair():
    b = np.array([[0., -.5, -1.], [0., -.5, 1.], [0., .5, 0.]])
    fa, fb = np.repeat(F, 3, axis=0), np.repeat(F, 2, axis=0)
    r = audit_intersections(A, fa, b, fb, 1e-9, max_contacts=2)
    assert r['query_statistics']['contact_cap_subdivisions'] > 0
    assert len(r['face_pairs']) == 6
    assert r['counts']['transverse'] == 6
    assert set(map(tuple, r['face_pairs'])) == {(i, j) for i in range(3) for j in range(2)}


def test_rigid_transform_and_reversed_winding():
    b = np.array([[0., -.5, -1.], [0., -.5, 1.], [0., .5, 0.]])
    q, _ = np.linalg.qr(np.array([[1., 2., 3.], [3., 1., 2.], [2., 3., 1.]]))
    offset = np.array([1e4, -2e4, 3e4])
    r = audit_intersections(A @ q.T+offset, F[:, ::-1], b @ q.T+offset, F)
    assert r['counts']['transverse'] == 1
    expected = np.array([[0., -.5, 0.], [0., .5, 0.]]) @ q.T+offset
    assert np.allclose(np.sort(np.linalg.norm(r['segments'][0][:, None]-expected[None], axis=-1), axis=1)[:, 0], 0, atol=1e-7)


def test_degenerate_pair_is_explicitly_unresolved():
    b = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    r = classify_triangle_pairs(A[None], b[None])
    assert r['classification'][0] == 'degenerate'
