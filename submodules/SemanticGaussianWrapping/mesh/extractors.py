"""Marching-cubes and semantic-aware Delaunay/marching-tetra extractors."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np

from .field import SurfaceFieldAdapter, as_field_adapter
from .sampling import GridBlock
from .topology import ContactGraph, compatible_pairs
from .types import SurfaceSamples, TriangleMesh


class MissingOptionalBackend(ImportError):
    """An extraction backend was requested but its optional package is absent."""


def _remove_degenerate(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    distinct = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[distinct]
    if not len(faces):
        return faces
    cross = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    faces = faces[np.linalg.norm(cross, axis=1) > 1e-12]
    if not len(faces):
        return faces
    canonical = np.sort(faces, axis=1)
    _, unique = np.unique(canonical, axis=0, return_index=True)
    return faces[np.sort(unique)]


def orient_faces(mesh: TriangleMesh) -> TriangleMesh:
    if mesh.normals is None or not len(mesh.faces):
        return mesh
    result = mesh.copy()
    face_normal = np.cross(
        result.vertices[result.faces[:, 1]] - result.vertices[result.faces[:, 0]],
        result.vertices[result.faces[:, 2]] - result.vertices[result.faces[:, 0]],
    )
    target = result.normals[result.faces].mean(axis=1)
    flip = np.sum(face_normal * target, axis=1) < 0
    result.faces[flip] = result.faces[flip][:, [0, 2, 1]]
    return result


def mesh_from_vertices_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    field: SurfaceFieldAdapter | object,
    semantic_decoder: Optional[Callable[[Any], Any]] = None,
) -> TriangleMesh:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(vertices) or not len(faces):
        return TriangleMesh.empty()
    faces = _remove_degenerate(vertices, faces)
    if not len(faces):
        return TriangleMesh.empty()
    adapter = as_field_adapter(field)
    samples = adapter.query(vertices)
    mesh = TriangleMesh(
        vertices=vertices,
        faces=faces,
        normals=samples.normal,
        semantic=samples.semantic,
        semantic_id=adapter.semantic_ids(samples.semantic, semantic_decoder),
        uncertainty=samples.uncertainty,
    )
    return orient_faces(mesh)


def marching_cubes_block(
    block: GridBlock,
    field: SurfaceFieldAdapter | object,
    *,
    value: str = "sdf",
    level: float = 0.0,
    backend: str = "skimage",
    semantic_decoder: Optional[Callable[[Any], Any]] = None,
) -> TriangleMesh:
    """Run scikit-image marching cubes on one sampled block."""
    if backend != "skimage":
        raise ValueError("the supported marching-cubes backend is skimage")
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise MissingOptionalBackend(
            "marching cubes requires scikit-image; install it with "
            "`pip install scikit-image` or use --method tetra"
        ) from error
    if value not in {"sdf", "occupancy"}:
        raise ValueError("marching cubes value must be sdf or occupancy")
    volume = np.asarray(block.values(value), dtype=np.float32)
    if not np.isfinite(volume).all() or volume.min() > level or volume.max() < level:
        return TriangleMesh.empty(block.samples.semantic.shape[1])
    if np.isclose(volume.min(), volume.max()):
        return TriangleMesh.empty(block.samples.semantic.shape[1])
    vertices, faces, _, _ = marching_cubes(
        volume,
        level=float(level),
        spacing=tuple(float(value) for value in block.spacing),
        allow_degenerate=False,
    )
    vertices = vertices.astype(np.float32) + block.bounds.minimum[None]
    return mesh_from_vertices_faces(vertices, faces, field, semantic_decoder)


def merge_meshes(meshes: Iterable[TriangleMesh], *, weld_tolerance: float = 1e-6) -> TriangleMesh:
    meshes = [mesh for mesh in meshes if len(mesh.faces)]
    if not meshes:
        return TriangleMesh.empty()
    vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
    offsets = np.cumsum([0] + [len(mesh.vertices) for mesh in meshes[:-1]])
    faces = np.concatenate(
        [mesh.faces + offset for mesh, offset in zip(meshes, offsets)], axis=0
    )
    normals = (
        np.concatenate([mesh.normals for mesh in meshes], axis=0)
        if all(mesh.normals is not None for mesh in meshes)
        else None
    )
    semantic = (
        np.concatenate([mesh.semantic for mesh in meshes], axis=0)
        if all(mesh.semantic is not None for mesh in meshes)
        else None
    )
    uncertainty = (
        np.concatenate([mesh.uncertainty for mesh in meshes], axis=0)
        if all(mesh.uncertainty is not None for mesh in meshes)
        else None
    )
    semantic_id = (
        np.concatenate([mesh.semantic_id for mesh in meshes], axis=0)
        if all(mesh.semantic_id is not None for mesh in meshes)
        else None
    )
    if weld_tolerance <= 0:
        return TriangleMesh(
            vertices,
            faces,
            normals,
            semantic,
            semantic_id=semantic_id,
            uncertainty=uncertainty,
        )

    keys = np.round(vertices / float(weld_tolerance)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    weights = np.bincount(inverse, minlength=count).astype(np.float32)

    def average(values: np.ndarray) -> np.ndarray:
        output = np.zeros((count, values.shape[1]), dtype=np.float64)
        np.add.at(output, inverse, values)
        return (output / weights[:, None]).astype(np.float32)

    welded_vertices = average(vertices)
    welded_normals = None if normals is None else average(normals)
    if welded_normals is not None:
        welded_normals /= np.maximum(
            np.linalg.norm(welded_normals, axis=1, keepdims=True), 1e-8
        )
    welded_semantic = None if semantic is None else average(semantic)
    welded_uncertainty = None
    if uncertainty is not None:
        sums = np.bincount(inverse, weights=uncertainty, minlength=count)
        welded_uncertainty = (sums / weights).astype(np.float32)
    welded_semantic_id = None
    if semantic_id is not None:
        welded_semantic_id = np.full(count, -1, dtype=np.int32)
        for vertex_index in range(count):
            labels = semantic_id[inverse == vertex_index]
            labels = labels[labels >= 0]
            if len(labels):
                values, frequencies = np.unique(labels, return_counts=True)
                welded_semantic_id[vertex_index] = int(values[np.argmax(frequencies)])
    welded_faces = _remove_degenerate(welded_vertices, inverse[faces])
    return TriangleMesh(
        welded_vertices,
        welded_faces,
        welded_normals,
        welded_semantic,
        semantic_id=welded_semantic_id,
        uncertainty=welded_uncertainty,
    ).compact()


def marching_cubes_blocks(
    blocks: Iterable[GridBlock],
    field: SurfaceFieldAdapter | object,
    *,
    value: str = "sdf",
    level: float = 0.0,
    weld_tolerance: Optional[float] = None,
    backend: str = "skimage",
    semantic_decoder: Optional[Callable[[Any], Any]] = None,
) -> TriangleMesh:
    blocks = list(blocks)
    if not blocks:
        return TriangleMesh.empty()
    meshes = [
        marching_cubes_block(
            block,
            field,
            value=value,
            level=level,
            backend=backend,
            semantic_decoder=semantic_decoder,
        )
        for block in blocks
    ]
    if weld_tolerance is None:
        weld_tolerance = min(float(np.min(block.spacing)) for block in blocks) * 1e-4
    return merge_meshes(meshes, weld_tolerance=weld_tolerance)


def _crossing_edges(tetra: np.ndarray, values: np.ndarray, level: float) -> list[tuple[int, int]]:
    inside = values[tetra] <= level
    inside_indices = np.flatnonzero(inside).tolist()
    outside_indices = np.flatnonzero(~inside).tolist()
    if len(inside_indices) in (0, 4):
        return []
    if len(inside_indices) == 1:
        first = inside_indices[0]
        ordered = [(first, other) for other in outside_indices]
    elif len(outside_indices) == 1:
        first = outside_indices[0]
        ordered = [(first, other) for other in inside_indices]
    else:
        first, second = inside_indices
        third, fourth = outside_indices
        # Cyclic order around the quadrilateral intersection.
        ordered = [(first, third), (first, fourth), (second, fourth), (second, third)]
    return [(int(tetra[first]), int(tetra[second])) for first, second in ordered]


def _triangulate_crossings(crossings: list[int]) -> list[tuple[int, int, int]]:
    if len(crossings) == 3:
        return [(crossings[0], crossings[1], crossings[2])]
    if len(crossings) == 4:
        return [
            (crossings[0], crossings[1], crossings[2]),
            (crossings[0], crossings[2], crossings[3]),
        ]
    return []


def marching_tetrahedra(
    samples: SurfaceSamples,
    tetrahedra: np.ndarray,
    field: SurfaceFieldAdapter | object,
    *,
    value: str = "sdf",
    level: float = 0.0,
    cosine_threshold: float = 0.85,
    contact_graph: Optional[ContactGraph] = None,
    max_edge_length: Optional[float] = None,
    semantic_decoder: Optional[Callable[[Any], Any]] = None,
) -> TriangleMesh:
    if value not in {"sdf", "occupancy"}:
        raise ValueError("marching tetra value must be sdf or occupancy")
    scalar = getattr(samples, value)
    adapter = as_field_adapter(field)
    labels = adapter.semantic_ids(samples.semantic, semantic_decoder)
    edge_vertices: dict[tuple[int, int], int] = {}
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []

    for tetra in np.asarray(tetrahedra, dtype=np.int64):
        crossings = _crossing_edges(tetra, scalar, level)
        if len(crossings) not in (3, 4):
            continue
        edge_array = np.asarray(crossings, dtype=np.int64)
        if labels is not None:
            compatible = compatible_pairs(
                samples.semantic[edge_array[:, 0]],
                samples.semantic[edge_array[:, 1]],
                labels[edge_array[:, 0]],
                labels[edge_array[:, 1]],
                cosine_threshold=cosine_threshold,
                contact_graph=contact_graph,
            )
            if not compatible.all():
                continue
        if max_edge_length is not None:
            length = np.linalg.norm(
                samples.points[edge_array[:, 0]] - samples.points[edge_array[:, 1]], axis=1
            )
            if np.any(length > max_edge_length):
                continue

        indices: list[int] = []
        for first, second in crossings:
            edge = (first, second) if first < second else (second, first)
            vertex_index = edge_vertices.get(edge)
            if vertex_index is None:
                first_value, second_value = float(scalar[first]), float(scalar[second])
                denominator = second_value - first_value
                alpha = 0.5 if abs(denominator) < 1e-12 else (level - first_value) / denominator
                alpha = float(np.clip(alpha, 0.0, 1.0))
                point = samples.points[first] * (1.0 - alpha) + samples.points[second] * alpha
                vertex_index = len(vertices)
                edge_vertices[edge] = vertex_index
                vertices.append(point.astype(np.float32))
            indices.append(vertex_index)
        faces.extend(_triangulate_crossings(indices))

    if not faces:
        return TriangleMesh.empty(samples.semantic.shape[1])
    return mesh_from_vertices_faces(
        np.asarray(vertices), np.asarray(faces), adapter, semantic_decoder
    )


def delaunay_marching_tetra(
    samples: SurfaceSamples,
    field: SurfaceFieldAdapter | object,
    **kwargs: object,
) -> TriangleMesh:
    """Build a Delaunay complex, then apply semantic-aware marching tetrahedra."""
    try:
        from scipy.spatial import Delaunay
    except ImportError as error:
        raise MissingOptionalBackend(
            "Delaunay marching tetrahedra requires SciPy; install it with "
            "`pip install scipy` or use --method cubes"
        ) from error
    if len(samples.points) < 4:
        return TriangleMesh.empty(samples.semantic.shape[1])
    try:
        tetrahedra = Delaunay(samples.points).simplices
    except Exception as error:
        raise RuntimeError(
            "Delaunay tetrahedralization failed; check for duplicate/coplanar samples"
        ) from error
    return marching_tetrahedra(samples, tetrahedra, field, **kwargs)
