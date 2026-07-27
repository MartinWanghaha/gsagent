"""PLY/OBJ I/O with explicit per-vertex semantic metadata."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .types import TriangleMesh


def semantic_colors(labels: np.ndarray) -> np.ndarray:
    """Stable, dependency-free RGB colors for integer semantic IDs."""
    labels = np.asarray(labels, dtype=np.uint32).reshape(-1)
    hashed = labels * np.uint32(2654435761)
    colors = np.stack(
        [
            48 + ((hashed >> np.uint32(0)) & np.uint32(191)),
            48 + ((hashed >> np.uint32(8)) & np.uint32(191)),
            48 + ((hashed >> np.uint32(16)) & np.uint32(191)),
        ],
        axis=1,
    )
    return colors.astype(np.uint8)


def write_ply(mesh: TriangleMesh, path: str | Path) -> Path:
    """Write a binary PLY retaining all semantic mesh attributes.

    Meshes produced by the region-conditioned exporter routinely contain
    hundreds of thousands of triangles.  A structured binary payload is both
    substantially faster and smaller than the former line-by-line ASCII
    writer.  Vertex semantics and face-level region ownership use explicit
    properties so downstream geometry tools can consume either independently.
    """
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    semantic_dim = 0 if mesh.semantic is None else mesh.semantic.shape[1]

    properties: list[tuple[str, str]] = [(axis, "<f4") for axis in "xyz"]
    if mesh.normals is not None:
        properties.extend((f"n{axis}", "<f4") for axis in "xyz")
    properties.extend((("semantic_id", "<i4"), ("uncertainty", "<f4")))
    properties.extend((f"semantic_{channel}", "<f4") for channel in range(semantic_dim))
    vertices = np.empty(len(mesh.vertices), dtype=properties)
    for axis, column in zip("xyz", mesh.vertices.T):
        vertices[axis] = column
    if mesh.normals is not None:
        for axis, column in zip("xyz", mesh.normals.T):
            vertices[f"n{axis}"] = column
    vertices["semantic_id"] = (
        np.full(len(mesh.vertices), -1, dtype=np.int32) if mesh.semantic_id is None else mesh.semantic_id
    )
    vertices["uncertainty"] = (
        np.zeros(len(mesh.vertices), dtype=np.float32) if mesh.uncertainty is None else mesh.uncertainty
    )
    if mesh.semantic is not None:
        for channel in range(semantic_dim):
            vertices[f"semantic_{channel}"] = mesh.semantic[:, channel]

    face_properties: list[tuple] = [("vertex_indices", "<i4", (3,))]
    if mesh.face_region_id is not None:
        face_properties.append(("region_id", "<i4"))
    faces = np.empty(len(mesh.faces), dtype=face_properties)
    if len(mesh.faces):
        faces["vertex_indices"] = mesh.faces.astype(np.int32, copy=False)
        if mesh.face_region_id is not None:
            faces["region_id"] = mesh.face_region_id
    PlyData(
        [
            PlyElement.describe(vertices, "vertex"),
            PlyElement.describe(faces, "face"),
        ],
        text=False,
        byte_order="<",
        comments=["SemanticGaussianWrapping semantic mesh"],
    ).write(path)
    return path


def write_obj(mesh: TriangleMesh, path: str | Path) -> Path:
    """Write OBJ with vertex colors and machine-readable semantic comments."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.full(len(mesh.vertices), -1, dtype=np.int32) if mesh.semantic_id is None else mesh.semantic_id
    colors = semantic_colors(np.maximum(labels, 0)).astype(np.float32) / 255.0
    uncertainty = np.zeros(len(mesh.vertices), dtype=np.float32) if mesh.uncertainty is None else mesh.uncertainty
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# SemanticGaussianWrapping semantic mesh\n")
        handle.write("# semantic_vertex_format: index semantic_id uncertainty [embedding...]\n")
        for index, vertex in enumerate(mesh.vertices):
            embedding = ""
            if mesh.semantic is not None:
                embedding = " " + " ".join(str(float(value)) for value in mesh.semantic[index])
            handle.write(f"# semantic_vertex {index + 1} {int(labels[index])} {float(uncertainty[index])}{embedding}\n")
            color = colors[index]
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")
        if mesh.normals is not None:
            for normal in mesh.normals:
                handle.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
        for face in mesh.faces:
            one_based = face + 1
            if mesh.normals is None:
                handle.write(f"f {one_based[0]} {one_based[1]} {one_based[2]}\n")
            else:
                handle.write(
                    f"f {one_based[0]}//{one_based[0]} {one_based[1]}//{one_based[1]} {one_based[2]}//{one_based[2]}\n"
                )
    return path


def export_mesh(mesh: TriangleMesh, path: str | Path) -> Path:
    suffix = Path(path).suffix.lower()
    if suffix == ".ply":
        return write_ply(mesh, path)
    if suffix == ".obj":
        return write_obj(mesh, path)
    raise ValueError("mesh output must end in .ply or .obj")


def _read_ascii_ply(path: Path) -> TriangleMesh:
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")
        vertex_count = face_count = 0
        vertex_properties: list[str] = []
        face_properties: list[tuple[str, bool]] = []
        element = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("truncated PLY header")
            tokens = line.strip().split()
            if tokens[:2] == ["format", "binary_little_endian"] or tokens[:2] == [
                "format",
                "binary_big_endian",
            ]:
                raise ValueError("binary PLY loading is not supported; use ASCII PLY")
            if tokens[:2] == ["element", "vertex"]:
                vertex_count = int(tokens[2])
                element = "vertex"
            elif tokens[:2] == ["element", "face"]:
                face_count = int(tokens[2])
                element = "face"
            elif tokens and tokens[0] == "property":
                if element == "vertex":
                    vertex_properties.append(tokens[-1])
                elif element == "face":
                    face_properties.append((tokens[-1], len(tokens) > 1 and tokens[1] == "list"))
            elif tokens and tokens[0] == "end_header":
                break
        rows = [handle.readline().strip().split() for _ in range(vertex_count)]
        if vertex_count:
            data = np.asarray(rows, dtype=np.float64)
            property_index = {name: index for index, name in enumerate(vertex_properties)}
            vertices = data[:, [property_index[axis] for axis in "xyz"]].astype(np.float32)
            normal_names = ["nx", "ny", "nz"]
            normals = (
                data[:, [property_index[name] for name in normal_names]].astype(np.float32)
                if all(name in property_index for name in normal_names)
                else None
            )
            semantic_names = sorted(
                (name for name in vertex_properties if name.startswith("semantic_") and name.split("_")[-1].isdigit()),
                key=lambda name: int(name.split("_")[-1]),
            )
            semantic = (
                data[:, [property_index[name] for name in semantic_names]].astype(np.float32)
                if semantic_names
                else None
            )
            semantic_id = (
                data[:, property_index["semantic_id"]].astype(np.int32) if "semantic_id" in property_index else None
            )
            uncertainty = (
                data[:, property_index["uncertainty"]].astype(np.float32) if "uncertainty" in property_index else None
            )
        else:
            vertices = np.empty((0, 3), dtype=np.float32)
            normals = semantic = semantic_id = uncertainty = None
        faces = []
        face_region_ids: list[int] = []
        has_face_region_id = any(
            name in {"region_id", "face_region_id"} for name, _ in face_properties
        )
        for _ in range(face_count):
            tokens = handle.readline().strip().split()
            position = 0
            values: dict[str, object] = {}
            for name, is_list in face_properties:
                if is_list:
                    length = int(tokens[position])
                    position += 1
                    values[name] = tokens[position : position + length]
                    position += length
                else:
                    values[name] = tokens[position]
                    position += 1
            indices = values.get("vertex_indices")
            if indices is None:
                raise ValueError("PLY face element has no vertex_indices property")
            if len(indices) != 3:
                raise ValueError("only triangular PLY faces are supported")
            faces.append([int(value) for value in indices])
            if has_face_region_id:
                value = values.get("region_id", values.get("face_region_id"))
                if value is None:
                    raise ValueError("PLY face region property is missing a value")
                face_region_ids.append(int(value))
    return TriangleMesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3),
        normals=normals,
        semantic=semantic,
        semantic_id=semantic_id,
        uncertainty=uncertainty,
        face_region_id=(
            np.asarray(face_region_ids, dtype=np.int32)
            if has_face_region_id
            else None
        ),
    )


def _ply_format(path: Path) -> str:
    """Read only the ASCII PLY header and return its declared encoding."""
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        encoding = None
        header_bytes = 4
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("truncated PLY header")
            header_bytes += len(line)
            if header_bytes > 1024 * 1024:
                raise ValueError("PLY header exceeds the 1 MiB safety limit")
            tokens = line.strip().split()
            if tokens and tokens[0] == b"format":
                if len(tokens) < 3:
                    raise ValueError("malformed PLY format declaration")
                try:
                    encoding = tokens[1].decode("ascii")
                except UnicodeDecodeError as error:
                    raise ValueError("PLY format declaration is not ASCII") from error
            if tokens == [b"end_header"]:
                break
    if encoding is None:
        raise ValueError("PLY header has no format declaration")
    return encoding


def _read_binary_ply(path: Path) -> TriangleMesh:
    """Load binary PLY geometry and every SGW semantic vertex property."""
    from plyfile import PlyData

    payload = PlyData.read(path)
    if "vertex" not in payload:
        raise ValueError("PLY contains no vertex geometry")
    vertex = payload["vertex"].data
    names = set(vertex.dtype.names or ())
    if not set("xyz").issubset(names):
        raise ValueError("PLY vertex element must contain x/y/z")
    vertices = np.column_stack([vertex[axis] for axis in "xyz"]).astype(
        np.float32,
        copy=False,
    )
    normal_names = tuple(f"n{axis}" for axis in "xyz")
    normals = (
        np.column_stack([vertex[name] for name in normal_names]).astype(
            np.float32,
            copy=False,
        )
        if set(normal_names).issubset(names)
        else None
    )
    semantic_names = sorted(
        (name for name in names if name.startswith("semantic_") and name.split("_")[-1].isdigit()),
        key=lambda name: int(name.split("_")[-1]),
    )
    semantic = (
        np.column_stack([vertex[name] for name in semantic_names]).astype(
            np.float32,
            copy=False,
        )
        if semantic_names
        else None
    )
    semantic_id = np.asarray(vertex["semantic_id"], dtype=np.int32) if "semantic_id" in names else None
    uncertainty = np.asarray(vertex["uncertainty"], dtype=np.float32) if "uncertainty" in names else None

    faces = np.empty((0, 3), dtype=np.int64)
    face_region_id = None
    if "face" in payload and "vertex_indices" in (payload["face"].data.dtype.names or ()):
        face_data = payload["face"].data
        face_names = set(face_data.dtype.names or ())
        values = face_data["vertex_indices"]
        if len(values):
            faces = np.asarray(values.tolist(), dtype=np.int64)
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError("only triangular PLY faces are supported")
        region_property = next(
            (
                name
                for name in ("region_id", "face_region_id")
                if name in face_names
            ),
            None,
        )
        if region_property is not None:
            face_region_id = np.asarray(
                face_data[region_property], dtype=np.int32
            )
    return TriangleMesh(
        vertices=vertices,
        faces=faces,
        normals=normals,
        semantic=semantic,
        semantic_id=semantic_id,
        uncertainty=uncertainty,
        face_region_id=face_region_id,
    )


def _read_obj(path: Path) -> TriangleMesh:
    vertices = []
    normals = []
    faces = []
    labels: dict[int, int] = {}
    uncertainty: dict[int, float] = {}
    semantics: dict[int, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[:2] == ["#", "semantic_vertex"] and len(tokens) >= 5:
                index = int(tokens[2]) - 1
                labels[index] = int(tokens[3])
                uncertainty[index] = float(tokens[4])
                semantics[index] = [float(value) for value in tokens[5:]]
            elif tokens[0] == "v":
                vertices.append([float(value) for value in tokens[1:4]])
            elif tokens[0] == "vn":
                normals.append([float(value) for value in tokens[1:4]])
            elif tokens[0] == "f":
                indices = [int(token.split("/")[0]) for token in tokens[1:]]
                if len(indices) < 3:
                    continue
                for corner in range(1, len(indices) - 1):
                    faces.append([indices[0] - 1, indices[corner] - 1, indices[corner + 1] - 1])
    count = len(vertices)
    semantic_dim = max((len(value) for value in semantics.values()), default=0)
    semantic_array = None
    if semantic_dim:
        semantic_array = np.zeros((count, semantic_dim), dtype=np.float32)
        for index, value in semantics.items():
            semantic_array[index, : len(value)] = value
    label_array = None
    if labels:
        label_array = np.full(count, -1, dtype=np.int32)
        for index, value in labels.items():
            label_array[index] = value
    uncertainty_array = None
    if uncertainty:
        uncertainty_array = np.zeros(count, dtype=np.float32)
        for index, value in uncertainty.items():
            uncertainty_array[index] = value
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
        faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3),
        normals=np.asarray(normals, dtype=np.float32) if len(normals) == count else None,
        semantic=semantic_array,
        semantic_id=label_array,
        uncertainty=uncertainty_array,
    )


def load_mesh(path: str | Path) -> TriangleMesh:
    path = Path(path)
    if path.suffix.lower() == ".ply":
        encoding = _ply_format(path)
        if encoding == "ascii":
            return _read_ascii_ply(path)
        if encoding in {"binary_little_endian", "binary_big_endian"}:
            return _read_binary_ply(path)
        raise ValueError(f"unsupported PLY format: {encoding}")
    if path.suffix.lower() == ".obj":
        return _read_obj(path)
    raise ValueError("mesh input must be a .ply or .obj")


def load_points(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        key = "points" if "points" in archive else archive.files[0]
        points = archive[key]
    elif suffix in {".xyz", ".txt", ".pts"}:
        points = np.loadtxt(path, dtype=np.float64)[:, :3]
    elif suffix in {".ply", ".obj"}:
        points = load_mesh(path).vertices
    else:
        raise ValueError("reference points must be .npy, .npz, .xyz, .txt, .ply, or .obj")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("reference point array must have shape [N, >=3]")
    return np.ascontiguousarray(points[:, :3])
