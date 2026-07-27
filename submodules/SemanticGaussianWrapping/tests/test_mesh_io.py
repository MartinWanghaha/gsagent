from __future__ import annotations

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from mesh.io import load_mesh, load_points, write_ply
from mesh.types import TriangleMesh


def _binary_vertices(with_normals: bool = False) -> np.ndarray:
    properties = [(axis, "<f4") for axis in "xyz"]
    if with_normals:
        properties.extend((f"n{axis}", "<f4") for axis in "xyz")
    vertices = np.zeros(3, dtype=properties)
    vertices["x"] = [0.0, 1.0, 0.0]
    vertices["y"] = [0.0, 0.0, 1.0]
    if with_normals:
        vertices["nz"] = 1.0
    return vertices


def _write_binary_ply(path, vertices: np.ndarray, include_face: bool) -> None:
    elements = [PlyElement.describe(vertices, "vertex")]
    if include_face:
        faces = np.empty(1, dtype=[("vertex_indices", "O")])
        faces[0]["vertex_indices"] = np.array([0, 1, 2], dtype=np.int32)
        elements.append(PlyElement.describe(faces, "face"))
    PlyData(elements, text=False, byte_order="<").write(path)
    assert b"format binary_little_endian 1.0" in path.read_bytes()[:80]


def test_binary_little_endian_triangle_mesh_loads_with_normals(tmp_path):
    path = tmp_path / "triangle_binary.ply"
    _write_binary_ply(path, _binary_vertices(with_normals=True), include_face=True)

    mesh = load_mesh(path)

    np.testing.assert_allclose(
        mesh.vertices,
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(mesh.faces, np.array([[0, 1, 2]], dtype=np.int64))
    np.testing.assert_allclose(mesh.normals, np.array([[0, 0, 1]] * 3, dtype=np.float32))
    assert mesh.face_region_id is None


def test_binary_little_endian_point_cloud_loads_without_faces(tmp_path):
    path = tmp_path / "points_binary.ply"
    _write_binary_ply(path, _binary_vertices(), include_face=False)

    mesh = load_mesh(path)

    assert mesh.faces.shape == (0, 3)
    assert mesh.normals is None
    np.testing.assert_allclose(load_points(path), mesh.vertices)


def test_binary_semantic_ply_keeps_custom_vertex_fields(tmp_path):
    expected = TriangleMesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        normals=np.array([[0, 0, 1]] * 3, dtype=np.float32),
        semantic=np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]], dtype=np.float32),
        semantic_id=np.array([4, 7, 4], dtype=np.int32),
        uncertainty=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        face_region_id=np.array([-2], dtype=np.int32),
    )
    path = write_ply(expected, tmp_path / "semantic_binary.ply")

    actual = load_mesh(path)
    face_properties = set(PlyData.read(path)["face"].data.dtype.names or ())

    assert "region_id" in face_properties
    np.testing.assert_allclose(actual.vertices, expected.vertices)
    np.testing.assert_array_equal(actual.faces, expected.faces)
    np.testing.assert_allclose(actual.normals, expected.normals)
    np.testing.assert_allclose(actual.semantic, expected.semantic)
    np.testing.assert_array_equal(actual.semantic_id, expected.semantic_id)
    np.testing.assert_allclose(actual.uncertainty, expected.uncertainty)
    np.testing.assert_array_equal(actual.face_region_id, expected.face_region_id)


def test_triangle_mesh_accepts_residual_and_seam_face_ownership():
    mesh = TriangleMesh(
        vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        face_region_id=np.array([-1, -2], dtype=np.int32),
    )

    np.testing.assert_array_equal(mesh.copy().face_region_id, [-1, -2])
    np.testing.assert_array_equal(mesh.compact().face_region_id, [-1, -2])


def test_binary_ply_roundtrips_residual_seam_and_region_faces(tmp_path):
    expected = TriangleMesh(
        vertices=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 0, 0],
            ],
            dtype=np.float32,
        ),
        faces=np.array(
            [[0, 1, 2], [1, 3, 2], [1, 4, 3]],
            dtype=np.int64,
        ),
        face_region_id=np.array([-2, -1, 725], dtype=np.int32),
    )

    actual = load_mesh(write_ply(expected, tmp_path / "face_regions.ply"))

    np.testing.assert_array_equal(actual.faces, expected.faces)
    np.testing.assert_array_equal(actual.face_region_id, expected.face_region_id)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("vertices", [[np.nan, 0, 0], [1, 0, 0], [0, 1, 0]]),
        ("normals", [[0, 0, np.inf]] * 3),
        ("semantic", [[0, 1], [np.nan, 1], [0, 1]]),
        ("uncertainty", [0, np.inf, 0]),
    ],
)
def test_triangle_mesh_rejects_non_finite_vertex_attributes(attribute, value):
    arguments = {
        "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "faces": np.array([[0, 1, 2]], dtype=np.int64),
    }
    arguments[attribute] = np.asarray(value, dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        TriangleMesh(**arguments)
