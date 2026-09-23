#!/usr/bin/env python3
"""Create a read-only Inpaint360GS view of an EDGS reconstruction.

EDGS stores the authoritative training options in ``config.yaml``.  Older
``cfg_args`` files can contain placeholder values, so this tool deliberately
does not read them.  The resulting bridge contains only lightweight metadata
and relative symlinks; the source EDGS reconstruction is never copied or
modified.

Example::

    python tools/build_edgs_bridge.py \
        --edgs-model /path/to/edgs/output \
        --iteration 30000 \
        --output /path/to/output/edgs_bridge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import uuid
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

BRIDGE_KIND = "edgs-inpaint360gs-bridge"
BRIDGE_SCHEMA_VERSION = 1
ITERATION_PATTERN = re.compile(r"^iteration_(\d+)$")
MAX_PLY_HEADER_BYTES = 1024 * 1024

_PLY_SCALAR_BYTES = {
    "char": 1,
    "int8": 1,
    "uchar": 1,
    "uint8": 1,
    "short": 2,
    "int16": 2,
    "ushort": 2,
    "uint16": 2,
    "int": 4,
    "int32": 4,
    "uint": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}
_PLY_FLOAT_TYPES = {"float", "float32", "double", "float64"}


class BridgeError(RuntimeError):
    """Raised when an EDGS artifact cannot safely form a bridge."""


@dataclass(frozen=True)
class PlyProperty:
    name: str
    value_type: str
    count_type: str | None = None

    @property
    def is_list(self) -> bool:
        return self.count_type is not None

    def to_manifest(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "value_type": self.value_type,
        }
        if self.count_type is not None:
            result["count_type"] = self.count_type
        return result


@dataclass(frozen=True)
class PlyElement:
    name: str
    count: int
    properties: tuple[PlyProperty, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "properties": [prop.to_manifest() for prop in self.properties],
        }


@dataclass(frozen=True)
class PlyHeader:
    encoding: str
    version: str
    header_bytes: int
    elements: tuple[PlyElement, ...]

    def element(self, name: str) -> PlyElement | None:
        return next(
            (element for element in self.elements if element.name == name), None
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "version": self.version,
            "header_bytes": self.header_bytes,
            "elements": {
                element.name: element.to_manifest() for element in self.elements
            },
        }


def _read_ply_header(path: Path) -> PlyHeader:
    """Read a PLY header without materializing a potentially multi-GB mesh."""

    elements: list[dict[str, Any]] = []
    current_element: dict[str, Any] | None = None
    encoding: str | None = None
    version: str | None = None

    with path.open("rb") as stream:
        first_line = stream.readline()
        if first_line.rstrip(b"\r\n") != b"ply":
            raise BridgeError(f"not a PLY file: {path}")

        while True:
            if stream.tell() > MAX_PLY_HEADER_BYTES:
                raise BridgeError(
                    f"PLY header exceeds {MAX_PLY_HEADER_BYTES} bytes: {path}"
                )
            raw_line = stream.readline()
            if not raw_line:
                raise BridgeError(f"PLY header has no end_header marker: {path}")
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise BridgeError(f"PLY header is not ASCII: {path}") from exc
            if not line or line.startswith("comment ") or line.startswith("obj_info "):
                continue
            if line == "end_header":
                header_bytes = stream.tell()
                break

            tokens = line.split()
            keyword = tokens[0]
            if keyword == "format":
                if len(tokens) != 3:
                    raise BridgeError(
                        f"invalid PLY format declaration in {path}: {line}"
                    )
                encoding, version = tokens[1], tokens[2]
                if encoding not in {
                    "ascii",
                    "binary_little_endian",
                    "binary_big_endian",
                }:
                    raise BridgeError(f"unsupported PLY encoding {encoding!r}: {path}")
            elif keyword == "element":
                if len(tokens) != 3:
                    raise BridgeError(
                        f"invalid PLY element declaration in {path}: {line}"
                    )
                try:
                    count = int(tokens[2])
                except ValueError as exc:
                    raise BridgeError(
                        f"invalid PLY element count in {path}: {line}"
                    ) from exc
                if count < 0:
                    raise BridgeError(f"negative PLY element count in {path}: {line}")
                if any(element["name"] == tokens[1] for element in elements):
                    raise BridgeError(f"duplicate PLY element {tokens[1]!r}: {path}")
                current_element = {
                    "name": tokens[1],
                    "count": count,
                    "properties": [],
                }
                elements.append(current_element)
            elif keyword == "property":
                if current_element is None:
                    raise BridgeError(
                        f"PLY property precedes its element in {path}: {line}"
                    )
                if len(tokens) == 3:
                    value_type, name = tokens[1], tokens[2]
                    count_type = None
                elif len(tokens) == 5 and tokens[1] == "list":
                    count_type, value_type, name = tokens[2], tokens[3], tokens[4]
                    if count_type not in _PLY_SCALAR_BYTES:
                        raise BridgeError(
                            f"unsupported PLY list count type {count_type!r}: {path}"
                        )
                else:
                    raise BridgeError(
                        f"invalid PLY property declaration in {path}: {line}"
                    )
                if value_type not in _PLY_SCALAR_BYTES:
                    raise BridgeError(
                        f"unsupported PLY scalar type {value_type!r}: {path}"
                    )
                properties: list[PlyProperty] = current_element["properties"]
                if any(prop.name == name for prop in properties):
                    raise BridgeError(
                        f"duplicate property {name!r} on element "
                        f"{current_element['name']!r}: {path}"
                    )
                properties.append(PlyProperty(name, value_type, count_type))
            else:
                raise BridgeError(f"unsupported PLY header directive in {path}: {line}")

    if encoding is None or version is None:
        raise BridgeError(f"PLY file has no format declaration: {path}")
    if version != "1.0":
        raise BridgeError(f"unsupported PLY version {version!r}: {path}")
    if not elements:
        raise BridgeError(f"PLY file contains no elements: {path}")

    result_elements = tuple(
        PlyElement(
            name=element["name"],
            count=element["count"],
            properties=tuple(element["properties"]),
        )
        for element in elements
    )
    return PlyHeader(encoding, version, header_bytes, result_elements)


def _property_map(element: PlyElement) -> dict[str, PlyProperty]:
    return {prop.name: prop for prop in element.properties}


def _require_scalar_properties(
    path: Path,
    element: PlyElement,
    names: Iterable[str],
    *,
    floating_point: bool = False,
) -> None:
    properties = _property_map(element)
    missing = [name for name in names if name not in properties]
    if missing:
        raise BridgeError(
            f"{path} is missing required {element.name} properties: "
            + ", ".join(missing)
        )
    invalid = [
        name
        for name in names
        if properties[name].is_list
        or (floating_point and properties[name].value_type not in _PLY_FLOAT_TYPES)
    ]
    if invalid:
        expected = "floating-point scalar" if floating_point else "scalar"
        raise BridgeError(
            f"{path} has non-{expected} {element.name} properties: "
            + ", ".join(invalid)
        )


def _validate_minimum_binary_payload(path: Path, header: PlyHeader) -> None:
    """Reject clearly truncated binary PLYs without parsing list payloads."""

    minimum_bytes = header.header_bytes
    for element in header.elements:
        scalar_stride = 0
        list_count_stride = 0
        for prop in element.properties:
            if prop.count_type is None:
                scalar_stride += _PLY_SCALAR_BYTES[prop.value_type]
            else:
                list_count_stride += _PLY_SCALAR_BYTES[prop.count_type]
        minimum_bytes += element.count * (scalar_stride + list_count_stride)
    actual_bytes = path.stat().st_size
    if actual_bytes < minimum_bytes:
        raise BridgeError(
            f"truncated binary PLY {path}: expected at least {minimum_bytes} bytes, "
            f"found {actual_bytes}"
        )


def validate_gaussian_ply(path: Path, sh_degree: int) -> PlyHeader:
    header = _read_ply_header(path)
    vertex = header.element("vertex")
    if vertex is None or vertex.count <= 0:
        raise BridgeError(f"Gaussian PLY has no vertices: {path}")

    required = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    _require_scalar_properties(path, vertex, required, floating_point=True)

    expected_rest = {f"f_rest_{index}" for index in range(3 * (sh_degree + 1) ** 2 - 3)}
    actual_rest = {
        prop.name for prop in vertex.properties if prop.name.startswith("f_rest_")
    }
    if actual_rest != expected_rest:
        missing = sorted(expected_rest - actual_rest)
        unexpected = sorted(actual_rest - expected_rest)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise BridgeError(
            f"Gaussian PLY SH layout does not match sh_degree={sh_degree}: "
            + "; ".join(details)
        )
    _require_scalar_properties(path, vertex, expected_rest, floating_point=True)

    for prefix, expected_names in (
        ("scale_", {"scale_0", "scale_1", "scale_2"}),
        ("rot_", {"rot_0", "rot_1", "rot_2", "rot_3"}),
    ):
        actual_names = {
            prop.name for prop in vertex.properties if prop.name.startswith(prefix)
        }
        if actual_names != expected_names:
            raise BridgeError(
                f"Gaussian PLY has invalid {prefix.rstrip('_')} layout in {path}: "
                f"expected {sorted(expected_names)}, found {sorted(actual_names)}"
            )

    if header.encoding != "ascii":
        _validate_minimum_binary_payload(path, header)
    elif path.stat().st_size <= header.header_bytes:
        raise BridgeError(f"Gaussian PLY has no ASCII payload: {path}")
    return header


def validate_mesh_ply(path: Path) -> PlyHeader:
    header = _read_ply_header(path)
    vertex = header.element("vertex")
    face = header.element("face")
    if vertex is None or vertex.count <= 0:
        raise BridgeError(f"mesh PLY has no vertices: {path}")
    if face is None or face.count <= 0:
        raise BridgeError(f"mesh PLY has no faces: {path}")
    _require_scalar_properties(path, vertex, ("x", "y", "z"), floating_point=True)
    face_properties = _property_map(face)
    indices = face_properties.get("vertex_indices")
    if indices is None or not indices.is_list:
        raise BridgeError(f"mesh PLY has no list property face.vertex_indices: {path}")

    if header.encoding != "ascii":
        _validate_minimum_binary_payload(path, header)
    elif path.stat().st_size <= header.header_bytes:
        raise BridgeError(f"mesh PLY has no ASCII payload: {path}")
    return header


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise BridgeError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise BridgeError(f"expected a mapping at the root of {path}")
    return document


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeError(f"config.yaml field {name} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise BridgeError(f"config.yaml is missing {location}.{key}")
    value = mapping[key]
    if isinstance(value, str) and "${" in value:
        raise BridgeError(
            f"config.yaml contains unresolved interpolation at {location}.{key}"
        )
    return value


def _config_values(config: Mapping[str, Any], model_path: Path) -> dict[str, Any]:
    gs = _mapping(_required(config, "gs", "root"), "gs")
    dataset = _mapping(_required(gs, "dataset", "gs"), "gs.dataset")
    pipeline = _mapping(gs.get("pipe", {}), "gs.pipe")
    renderer = _mapping(gs.get("renderer", {}), "gs.renderer")

    sh_degree = _required(gs, "sh_degree", "gs")
    if isinstance(sh_degree, bool) or not isinstance(sh_degree, int) or sh_degree < 0:
        raise BridgeError(
            "config.yaml field gs.sh_degree must be a non-negative integer"
        )

    saved_source = _required(dataset, "source_path", "gs.dataset")
    if not isinstance(saved_source, str) or not saved_source.strip():
        raise BridgeError(
            "config.yaml field gs.dataset.source_path must be a non-empty string"
        )
    source_path = Path(saved_source).expanduser()
    if not source_path.is_absolute():
        source_path = model_path / source_path
    source_path = source_path.resolve()
    if not source_path.is_dir():
        raise BridgeError(
            f"dataset source_path from config.yaml does not exist: {source_path}"
        )

    images = _required(dataset, "images", "gs.dataset")
    if not isinstance(images, str) or not images.strip():
        raise BridgeError(
            "config.yaml field gs.dataset.images must be a non-empty string"
        )
    image_path = source_path / images
    if not image_path.is_dir():
        raise BridgeError(
            f"dataset image directory from config.yaml does not exist: {image_path}"
        )

    resolution = _required(dataset, "resolution", "gs.dataset")
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise BridgeError("config.yaml field gs.dataset.resolution must be an integer")

    for key in ("white_background", "eval"):
        value = _required(dataset, key, "gs.dataset")
        if not isinstance(value, bool):
            raise BridgeError(f"config.yaml field gs.dataset.{key} must be boolean")

    data_device = dataset.get("data_device", "cuda")
    if not isinstance(data_device, str) or not data_device:
        raise BridgeError("config.yaml field gs.dataset.data_device must be a string")

    return {
        "sh_degree": sh_degree,
        "source_path_saved": saved_source,
        "source_path": str(source_path),
        "images": images,
        "resolution": resolution,
        "white_background": dataset["white_background"],
        "eval": dataset["eval"],
        "data_device": data_device,
        "depths": dataset.get("depths", ""),
        "train_test_exp": bool(dataset.get("train_test_exp", False)),
        "convert_SHs_python": bool(pipeline.get("convert_SHs_python", False)),
        "compute_cov3D_python": bool(pipeline.get("compute_cov3D_python", False)),
        "debug": bool(pipeline.get("debug", False)),
        "antialiasing": bool(pipeline.get("antialiasing", False)),
        "renderer_backend": renderer.get("backend"),
        "saved_model_path": dataset.get("model_path"),
    }


def available_iterations(model_path: Path) -> list[int]:
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.is_dir():
        return []
    iterations: list[int] = []
    for candidate in point_cloud_root.iterdir():
        match = ITERATION_PATTERN.fullmatch(candidate.name)
        if match and (candidate / "point_cloud.ply").is_file():
            value = int(match.group(1))
            if value > 0:
                iterations.append(value)
    return sorted(set(iterations))


def resolve_iteration(model_path: Path, requested: int) -> int:
    iterations = available_iterations(model_path)
    if not iterations:
        raise BridgeError(
            f"no point_cloud/iteration_<N>/point_cloud.ply found in {model_path}"
        )
    if requested == -1:
        return iterations[-1]
    if requested <= 0:
        raise BridgeError("--iteration must be -1 (latest) or a positive integer")
    if requested not in iterations:
        raise BridgeError(
            f"iteration {requested} is unavailable in {model_path}; "
            f"available: {', '.join(map(str, iterations))}"
        )
    return requested


def resolve_mesh(model_path: Path, iteration: int, requested: Path | None) -> Path:
    if requested is not None:
        mesh_path = requested.expanduser()
        if not mesh_path.is_absolute():
            mesh_path = model_path / mesh_path
        mesh_path = mesh_path.resolve()
        if not mesh_path.is_file():
            raise BridgeError(f"mesh PLY does not exist: {mesh_path}")
        return mesh_path

    mesh_root = model_path / "mesh" / f"ours_{iteration}"
    candidates = (
        mesh_root / "tsdf_fusion_post.ply",
        mesh_root / "tsdf_fusion.ply",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise BridgeError(
        f"no PGSR mesh found for iteration {iteration}; expected one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _artifact_id(
    *, config_sha256: str, gaussian_sha256: str, mesh_sha256: str, iteration: int
) -> str:
    identity = {
        "config_sha256": config_sha256,
        "gaussian_sha256": gaussian_sha256,
        "iteration": iteration,
        "mesh_sha256": mesh_sha256,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _namespace_text(values: Mapping[str, Any]) -> str:
    return str(Namespace(**dict(values))) + "\n"


def _relative_link_target(source: Path, destination: Path) -> str:
    return os.path.relpath(source, start=destination.parent)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_dir() and not destination.is_symlink():
        raise BridgeError(
            f"cannot replace directory with bridge symlink: {destination}"
        )
    relative_target = _relative_link_target(source, destination)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        os.symlink(relative_target, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_existing_manifest(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise BridgeError(f"bridge manifest is not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError(
            f"cannot read existing bridge manifest {path}: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping) or manifest.get("kind") != BRIDGE_KIND:
        raise BridgeError(
            f"{path} is not a {BRIDGE_KIND} manifest; use a new output directory"
        )
    return manifest


def _path_inside(root: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str):
        return None
    candidate = Path(relative)
    if candidate.is_absolute():
        return None
    absolute = Path(os.path.abspath(root / candidate))
    try:
        absolute.relative_to(root)
    except ValueError:
        return None
    return absolute


def _existing_bridge_links(root: Path, manifest: Mapping[str, Any]) -> list[Path]:
    bridge = manifest.get("bridge")
    if not isinstance(bridge, Mapping):
        return []
    links: list[Path] = []
    for key in ("point_cloud", "mesh"):
        candidate = _path_inside(root, bridge.get(key))
        if candidate is not None:
            links.append(candidate)
    return links


def _preflight_output(
    output: Path,
    artifact_id: str,
    *,
    overwrite: bool,
) -> tuple[str, Mapping[str, Any] | None]:
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise BridgeError(f"bridge output must be a directory: {output}")

    manifest_path = output / "bridge_manifest.json"
    existing = _read_existing_manifest(manifest_path) if output.exists() else None
    if existing is None:
        output_is_nonempty = output.exists() and any(output.iterdir())
        if output_is_nonempty and not overwrite:
            raise BridgeError(
                f"bridge output is non-empty and unmanaged: {output}; "
                "choose a new directory or pass --overwrite"
            )
        action = "overwrite-unmanaged" if output_is_nonempty else "create"
        return action, None

    existing_id = existing.get("artifact_id")
    if existing_id == artifact_id:
        return "refresh", existing
    if not overwrite:
        old_edgs = existing.get("edgs", {})
        if isinstance(old_edgs, Mapping):
            old_source = old_edgs.get("model_path", "unknown")
            old_iteration = old_edgs.get("iteration", "unknown")
        else:
            old_source = "unknown"
            old_iteration = "unknown"
        raise BridgeError(
            f"{output} already bridges a different EDGS artifact "
            f"({old_source}, iteration {old_iteration}); pass --overwrite to replace it"
        )

    for stale in _existing_bridge_links(output, existing):
        if (stale.exists() or stale.is_symlink()) and not stale.is_symlink():
            raise BridgeError(
                f"refusing to remove non-symlink from the previous bridge: {stale}"
            )
    return "overwrite", existing


def _remove_stale_links(
    output: Path,
    existing: Mapping[str, Any] | None,
    current_links: Sequence[Path],
) -> None:
    if existing is None:
        return
    current = {Path(os.path.abspath(path)) for path in current_links}
    for stale in _existing_bridge_links(output, existing):
        if stale in current or not stale.is_symlink():
            continue
        stale.unlink()
        parent = stale.parent
        while parent != output:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def build_bridge(
    edgs_model: Path,
    output: Path,
    requested_iteration: int,
    *,
    mesh_path: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    edgs_model = edgs_model.expanduser().resolve()
    output = Path(os.path.abspath(output.expanduser()))
    if not edgs_model.is_dir():
        raise BridgeError(f"EDGS model directory does not exist: {edgs_model}")
    if output == edgs_model:
        raise BridgeError(
            "bridge output must be different from the EDGS model directory"
        )

    config_path = edgs_model / "config.yaml"
    if not config_path.is_file():
        raise BridgeError(f"missing authoritative EDGS config: {config_path}")
    config = _load_yaml_mapping(config_path)
    values = _config_values(config, edgs_model)

    iteration = resolve_iteration(edgs_model, requested_iteration)
    gaussian_path = (
        edgs_model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    ).resolve()
    selected_mesh_path = resolve_mesh(edgs_model, iteration, mesh_path)

    gaussian_header = validate_gaussian_ply(gaussian_path, values["sh_degree"])
    mesh_header = validate_mesh_ply(selected_mesh_path)
    gaussian_vertex = gaussian_header.element("vertex")
    mesh_vertex = mesh_header.element("vertex")
    mesh_face = mesh_header.element("face")
    # The validators above guarantee these elements are present.
    assert (
        gaussian_vertex is not None
        and mesh_vertex is not None
        and mesh_face is not None
    )
    config_sha256 = _sha256(config_path)
    gaussian_sha256 = _sha256(gaussian_path)
    mesh_sha256 = _sha256(selected_mesh_path)
    artifact_id = _artifact_id(
        config_sha256=config_sha256,
        gaussian_sha256=gaussian_sha256,
        mesh_sha256=mesh_sha256,
        iteration=iteration,
    )

    point_cloud_relative = (
        Path("point_cloud") / f"iteration_{iteration}" / "point_cloud.ply"
    )
    mesh_relative = Path("mesh") / f"ours_{iteration}" / selected_mesh_path.name
    point_cloud_link = output / point_cloud_relative
    mesh_link = output / mesh_relative

    action, existing = _preflight_output(output, artifact_id, overwrite=overwrite)
    created_at = datetime.now(timezone.utc).isoformat()
    if existing is not None and existing.get("artifact_id") == artifact_id:
        previous_created_at = existing.get("created_at")
        if isinstance(previous_created_at, str):
            created_at = previous_created_at

    cfg_args = {
        "sh_degree": values["sh_degree"],
        "source_path": values["source_path"],
        "model_path": str(output),
        "resolution": values["resolution"],
        "vanilla_3dgs_path": str(output),
        "object_path": "object_mask",
        "eval": values["eval"],
        "images": values["images"],
        "white_background": values["white_background"],
        "data_device": values["data_device"],
        "n_views": 100,
        "random_init": False,
        "train_split": False,
        "num_classes": -1,
        "init_mode": "sparse",
        "train_distill": False,
        "convert_SHs_python": values["convert_SHs_python"],
        "compute_cov3D_python": values["compute_cov3D_python"],
        "debug": values["debug"],
        # Retained for compatibility with EDGS/Graphdeco consumers.  The
        # Inpaint360GS argument extractor safely ignores fields it does not use.
        "depths": values["depths"],
        "train_test_exp": values["train_test_exp"],
        "antialiasing": values["antialiasing"],
    }

    manifest: dict[str, Any] = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "kind": BRIDGE_KIND,
        # Written only after both symlinks and cfg_args have been committed.
        # Pipeline drivers can therefore use this as the bridge commit marker.
        "complete": True,
        "artifact_id": artifact_id,
        "created_at": created_at,
        "edgs": {
            "model_path": str(edgs_model),
            "saved_model_path": values["saved_model_path"],
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "requested_iteration": requested_iteration,
            "iteration": iteration,
            "available_iterations": available_iterations(edgs_model),
            "renderer_backend": values["renderer_backend"],
            "point_cloud_path": str(gaussian_path),
            "point_cloud_bytes": gaussian_path.stat().st_size,
            "point_cloud_sha256": gaussian_sha256,
            "mesh_path": str(selected_mesh_path),
            "mesh_bytes": selected_mesh_path.stat().st_size,
            "mesh_sha256": mesh_sha256,
        },
        "dataset": {
            "source_path_saved": values["source_path_saved"],
            "source_path": values["source_path"],
            "images": values["images"],
            "resolution": values["resolution"],
            "white_background": values["white_background"],
            "eval": values["eval"],
            "data_device": values["data_device"],
        },
        "gaussian_ply": gaussian_header.to_manifest(),
        "mesh_ply": mesh_header.to_manifest(),
        "bridge": {
            "root": str(output),
            "cfg_args": "cfg_args",
            "point_cloud": point_cloud_relative.as_posix(),
            "point_cloud_link_target": _relative_link_target(
                gaussian_path, point_cloud_link
            ),
            "mesh": mesh_relative.as_posix(),
            "mesh_link_target": _relative_link_target(selected_mesh_path, mesh_link),
        },
    }

    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_symlink(gaussian_path, point_cloud_link)
        _atomic_symlink(selected_mesh_path, mesh_link)
        _atomic_write_text(output / "cfg_args", _namespace_text(cfg_args))
        _remove_stale_links(output, existing, (point_cloud_link, mesh_link))
        _atomic_write_text(
            output / "bridge_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    return {
        "action": action,
        "dry_run": dry_run,
        "artifact_id": artifact_id,
        "iteration": iteration,
        "gaussian_count": gaussian_vertex.count,
        "mesh_vertex_count": mesh_vertex.count,
        "mesh_face_count": mesh_face.count,
        "output": str(output),
        "point_cloud": str(point_cloud_link),
        "mesh": str(mesh_link),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an Inpaint360GS-compatible bridge to one EDGS Gaussian "
            "iteration and its PGSR mesh."
        )
    )
    parser.add_argument(
        "--edgs-model",
        required=True,
        type=Path,
        help="EDGS output directory containing config.yaml and point_cloud/",
    )
    parser.add_argument(
        "--iteration",
        required=True,
        type=int,
        help="positive EDGS iteration, or -1 to resolve the latest valid PLY",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="independent bridge directory to create",
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        help=(
            "optional mesh PLY; relative paths are resolved below --edgs-model. "
            "By default tsdf_fusion_post.ply then tsdf_fusion.ply is selected."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a bridge that points at a different EDGS artifact",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate all inputs and print the plan without creating files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_bridge(
            args.edgs_model,
            args.output,
            args.iteration,
            mesh_path=args.mesh_path,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (BridgeError, OSError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
