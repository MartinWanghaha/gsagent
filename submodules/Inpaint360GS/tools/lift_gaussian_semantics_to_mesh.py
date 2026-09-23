#!/usr/bin/env python3
"""Export instance labels for object-aware Gaussians and an EDGS mesh.

The script deliberately does not import Inpaint360GS or EDGS Python packages.
Those projects contain modules with overlapping names and CUDA extensions.  The
only interchange formats used here are their stable artifacts:

* an Inpaint360GS object-aware Gaussian PLY (``obj_dc_*`` properties),
* the associated 1x1-convolution classifier checkpoint,
* the segmentation ``scene.json``, and
* an EDGS/PGSR triangle mesh PLY.

Mesh vertices receive a confidence-weighted interpolation of nearby Gaussian
object embeddings.  The original classifier is then evaluated on the
interpolated embedding.  Results are written as mmap-friendly ``.npy`` sidecar
arrays; the input mesh is never copied or mutated.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

SCHEMA_VERSION = 1
MANIFEST_NAME = "semantic_manifest.json"
PALETTE_NAME = "palette.json"
COLORED_MESH_NAME = "semantic_mesh.ply"
ARRAY_NAMES = {
    "gaussian_label": "gaussian_instance_id.npy",
    "gaussian_confidence": "gaussian_confidence.npy",
    "vertex_label": "vertex_instance_id.npy",
    "vertex_confidence": "vertex_confidence.npy",
    "face_label": "face_instance_id.npy",
    "face_confidence": "face_confidence.npy",
}
MANAGED_NAMES = frozenset(
    (*ARRAY_NAMES.values(), PALETTE_NAME, COLORED_MESH_NAME, MANIFEST_NAME)
)


@dataclass(frozen=True)
class PlySchema:
    gaussian_count: int
    embedding_names: tuple[str, ...]
    scale_names: tuple[str, ...]
    rotation_names: tuple[str, ...]
    mesh_vertex_count: int
    mesh_face_count: int
    mesh_face_property: str
    mesh_has_normals: bool


@dataclass(frozen=True)
class Classifier:
    weight: np.ndarray
    bias: np.ndarray

    @property
    def num_classes(self) -> int:
        return int(self.weight.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.weight.shape[1])


@dataclass
class ExportStats:
    gaussian_unknown: int = 0
    vertex_unknown: int = 0
    vertex_unsupported: int = 0
    face_unknown: int = 0


class Progress:
    """Small percentage-based progress reporter that avoids per-chunk noise."""

    def __init__(self, label: str, total: int, quiet: bool) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.quiet = quiet
        self.next_percent = 0

    def update(self, completed: int) -> None:
        if self.quiet:
            return
        percent = min(100, int(100 * completed / self.total))
        if percent >= self.next_percent:
            print(f"[{self.label}] {percent:3d}%", flush=True)
            self.next_percent = min(100, percent + 5)


def _log(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


def _indexed_properties(names: Iterable[str], prefix: str) -> tuple[str, ...]:
    indexed: list[tuple[int, str]] = []
    for name in names:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if not suffix.isdigit():
            continue
        indexed.append((int(suffix), name))
    indexed.sort()
    if indexed and [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError(f"PLY properties with prefix {prefix!r} are not contiguous")
    return tuple(name for _, name in indexed)


def _load_classifier(path: Path) -> Classifier:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch only
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, Mapping):
        for wrapper_key in ("state_dict", "classifier", "model"):
            wrapped = checkpoint.get(wrapper_key)
            if isinstance(wrapped, Mapping):
                checkpoint = wrapped
                break
    if not isinstance(checkpoint, Mapping):
        raise ValueError("classifier checkpoint must contain a state dict")

    tensor_items = {
        str(key): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    weight_candidates = [
        value for key, value in tensor_items.items() if key.split(".")[-1] == "weight"
    ]
    bias_candidates = [
        value for key, value in tensor_items.items() if key.split(".")[-1] == "bias"
    ]
    if len(weight_candidates) != 1 or len(bias_candidates) != 1:
        raise ValueError(
            "expected exactly one classifier weight and bias tensor; found "
            f"{len(weight_candidates)} weights and {len(bias_candidates)} biases"
        )

    weight_tensor = weight_candidates[0].detach().cpu().squeeze(-1).squeeze(-1)
    bias_tensor = bias_candidates[0].detach().cpu().reshape(-1)
    if weight_tensor.ndim != 2:
        raise ValueError(
            "classifier weight must have shape [classes, features, 1, 1] or "
            f"[classes, features], got {tuple(weight_candidates[0].shape)}"
        )
    if bias_tensor.shape[0] != weight_tensor.shape[0]:
        raise ValueError("classifier bias length does not match its output channels")

    weight = np.asarray(weight_tensor, dtype=np.float32)
    bias = np.asarray(bias_tensor, dtype=np.float32)
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("classifier contains NaN or infinite values")
    return Classifier(
        weight=np.ascontiguousarray(weight), bias=np.ascontiguousarray(bias)
    )


def _read_scene_info(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    if not isinstance(info, dict):
        raise ValueError("scene info must be a JSON object")
    try:
        num_classes = int(info["num_classes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "scene info must contain a positive integer num_classes"
        ) from error
    if num_classes <= 0:
        raise ValueError("scene num_classes must be positive")
    info["num_classes"] = num_classes
    return info


def _require_binary_ply(path: Path, artifact_name: str) -> None:
    """Reject ASCII PLY before plyfile can materialize a huge artifact."""

    ply_format: str | None = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{artifact_name} is not a PLY file: {path}")
        for _ in range(10_000):
            line = handle.readline()
            if not line:
                break
            if line.startswith(b"format "):
                fields = line.decode("ascii", errors="strict").split()
                if len(fields) >= 2:
                    ply_format = fields[1]
            if line.strip() == b"end_header":
                break
    if ply_format is None:
        raise ValueError(f"{artifact_name} PLY has no valid format declaration")
    if ply_format == "ascii":
        raise ValueError(
            f"{artifact_name} is ASCII PLY, which cannot be memory-mapped safely; "
            "convert it to binary_little_endian first"
        )
    if ply_format not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported {artifact_name} PLY format: {ply_format}")


def _inspect_ply(
    gaussian_path: Path,
    mesh_path: Path,
    classifier: Classifier,
    scene_info: Mapping[str, Any],
    normal_power: float,
) -> tuple[PlySchema, PlyData, PlyData]:
    _require_binary_ply(gaussian_path, "Gaussian")
    _require_binary_ply(mesh_path, "mesh")
    gaussian_ply = PlyData.read(str(gaussian_path), mmap=True)
    try:
        gaussian_vertex = gaussian_ply["vertex"]
    except KeyError as error:
        raise ValueError("Gaussian PLY has no vertex element") from error
    gaussian_names = set(gaussian_vertex.data.dtype.names or ())
    missing = {"x", "y", "z", "opacity"} - gaussian_names
    if missing:
        raise ValueError(f"Gaussian PLY is missing properties: {sorted(missing)}")
    embedding_names = _indexed_properties(gaussian_names, "obj_dc_")
    scale_names = _indexed_properties(gaussian_names, "scale_")
    rotation_names = _indexed_properties(gaussian_names, "rot_")
    if not embedding_names:
        raise ValueError(
            "Gaussian PLY has no obj_dc_* properties; run object-feature "
            "distillation before semantic export"
        )
    if not scale_names:
        raise ValueError("Gaussian PLY has no scale_* properties")
    if normal_power > 0.0 and len(scale_names) != 3:
        raise ValueError(
            "normal weighting requires exactly three Gaussian scale_* properties; "
            f"found {len(scale_names)}"
        )
    if normal_power > 0.0 and len(rotation_names) != 4:
        raise ValueError(
            "normal weighting requires exactly four Gaussian rot_* quaternion "
            f"properties; found {len(rotation_names)}"
        )
    if len(embedding_names) != classifier.embedding_dim:
        raise ValueError(
            f"PLY contains {len(embedding_names)} object channels, but classifier "
            f"expects {classifier.embedding_dim}"
        )
    if int(scene_info["num_classes"]) != classifier.num_classes:
        raise ValueError(
            f"scene.json declares {scene_info['num_classes']} classes, but "
            f"classifier predicts {classifier.num_classes}"
        )
    if gaussian_vertex.count <= 0:
        raise ValueError("Gaussian PLY contains no points")

    mesh_ply = PlyData.read(
        str(mesh_path),
        mmap=True,
        known_list_len={"face": {"vertex_indices": 3, "vertex_index": 3}},
    )
    try:
        mesh_vertex = mesh_ply["vertex"]
        mesh_face = mesh_ply["face"]
    except KeyError as error:
        raise ValueError("mesh PLY must contain vertex and face elements") from error
    mesh_vertex_names = set(mesh_vertex.data.dtype.names or ())
    if not {"x", "y", "z"}.issubset(mesh_vertex_names):
        raise ValueError("mesh vertex element must contain x, y, and z")
    mesh_normal_names = {"nx", "ny", "nz"}
    mesh_has_normals = mesh_normal_names.issubset(mesh_vertex_names)
    if mesh_normal_names & mesh_vertex_names and not mesh_has_normals:
        raise ValueError("mesh must contain all or none of nx, ny, and nz")
    if normal_power > 0.0 and not mesh_has_normals:
        raise ValueError(
            "normal-power is positive, but the mesh has no nx/ny/nz properties; "
            "pass --normal-power 0 to disable normal consistency"
        )
    face_names = set(mesh_face.data.dtype.names or ())
    if "vertex_indices" in face_names:
        face_property = "vertex_indices"
    elif "vertex_index" in face_names:
        face_property = "vertex_index"
    else:
        raise ValueError("mesh face element has no vertex_indices list")
    if mesh_vertex.count <= 0 or mesh_face.count <= 0:
        raise ValueError("mesh must contain at least one vertex and one triangle")

    faces = mesh_face.data[face_property]
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            "only triangle meshes are supported; provide a triangulated EDGS mesh"
        )

    schema = PlySchema(
        gaussian_count=int(gaussian_vertex.count),
        embedding_names=embedding_names,
        scale_names=scale_names,
        rotation_names=rotation_names,
        mesh_vertex_count=int(mesh_vertex.count),
        mesh_face_count=int(mesh_face.count),
        mesh_face_property=face_property,
        mesh_has_normals=mesh_has_normals,
    )
    return schema, gaussian_ply, mesh_ply


def _label_dtype(num_classes: int, unknown_id: int) -> np.dtype:
    if unknown_id < 0:
        raise ValueError("unknown-id must be non-negative")
    if unknown_id < num_classes:
        raise ValueError(
            f"unknown-id {unknown_id} collides with class range [0, {num_classes - 1}]"
        )
    maximum = max(num_classes - 1, unknown_id)
    if maximum <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    raise ValueError("class or unknown ID exceeds uint32 range")


def _smallest_axis_normals(
    vertex: np.ndarray,
    rotation_names: Sequence[str],
    smallest_axis: np.ndarray,
) -> np.ndarray:
    """Match PGSR ``get_smallest_axis`` without importing its CUDA package."""

    quaternion = np.empty((len(vertex), 4), dtype=np.float32)
    for column, name in enumerate(rotation_names):
        quaternion[:, column] = vertex[name]
    norm = np.linalg.norm(quaternion, axis=1)
    if not np.isfinite(quaternion).all() or np.any(norm <= np.finfo(np.float32).eps):
        raise ValueError("Gaussian rotations contain an invalid quaternion")
    quaternion /= norm[:, None]
    w, x, y, z = (quaternion[:, index] for index in range(4))

    normals = np.empty((len(vertex), 3), dtype=np.float32)
    axis_0 = smallest_axis == 0
    axis_1 = smallest_axis == 1
    axis_2 = smallest_axis == 2
    normals[axis_0, 0] = 1.0 - 2.0 * (y[axis_0] ** 2 + z[axis_0] ** 2)
    normals[axis_0, 1] = 2.0 * (x[axis_0] * y[axis_0] + w[axis_0] * z[axis_0])
    normals[axis_0, 2] = 2.0 * (x[axis_0] * z[axis_0] - w[axis_0] * y[axis_0])
    normals[axis_1, 0] = 2.0 * (x[axis_1] * y[axis_1] - w[axis_1] * z[axis_1])
    normals[axis_1, 1] = 1.0 - 2.0 * (x[axis_1] ** 2 + z[axis_1] ** 2)
    normals[axis_1, 2] = 2.0 * (y[axis_1] * z[axis_1] + w[axis_1] * x[axis_1])
    normals[axis_2, 0] = 2.0 * (x[axis_2] * z[axis_2] + w[axis_2] * y[axis_2])
    normals[axis_2, 1] = 2.0 * (y[axis_2] * z[axis_2] - w[axis_2] * x[axis_2])
    normals[axis_2, 2] = 1.0 - 2.0 * (x[axis_2] ** 2 + y[axis_2] ** 2)
    normal_norm = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(
        normal_norm <= np.finfo(np.float32).eps
    ):
        raise ValueError("could not derive finite Gaussian normals")
    normals /= normal_norm[:, None]
    return normals


def _extract_gaussians(
    gaussian_ply: PlyData, schema: PlySchema, use_normals: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    vertex = gaussian_ply["vertex"].data
    count = schema.gaussian_count

    xyz = np.empty((count, 3), dtype=np.float64)
    for column, name in enumerate(("x", "y", "z")):
        xyz[:, column] = vertex[name]

    embeddings = np.empty((count, len(schema.embedding_names)), dtype=np.float32)
    for column, name in enumerate(schema.embedding_names):
        embeddings[:, column] = vertex[name]

    max_log_scale = np.full(count, -np.inf, dtype=np.float32)
    smallest_axis = np.zeros(count, dtype=np.uint8) if use_normals else None
    min_log_scale = np.full(count, np.inf, dtype=np.float32) if use_normals else None
    for axis, name in enumerate(schema.scale_names):
        scale = np.asarray(vertex[name], dtype=np.float32)
        np.maximum(max_log_scale, scale, out=max_log_scale)
        if use_normals:
            assert smallest_axis is not None and min_log_scale is not None
            is_smaller = scale < min_log_scale
            min_log_scale[is_smaller] = scale[is_smaller]
            smallest_axis[is_smaller] = axis
    # The native Gaussian scale activation is exp().  Clipping only protects
    # against malformed artifacts and does not affect normal trained ranges.
    max_scale = np.exp(np.clip(max_log_scale, -30.0, 30.0)).astype(np.float32)

    opacity_logit = np.asarray(vertex["opacity"], dtype=np.float32)
    opacity = 1.0 / (1.0 + np.exp(-np.clip(opacity_logit, -30.0, 30.0)))
    opacity = opacity.astype(np.float32, copy=False)
    normals = (
        _smallest_axis_normals(vertex, schema.rotation_names, smallest_axis)
        if use_normals and smallest_axis is not None
        else None
    )

    for name, array in (
        ("Gaussian positions", xyz),
        ("Gaussian object embeddings", embeddings),
        ("Gaussian scales", max_scale),
        ("Gaussian opacities", opacity),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contain NaN or infinite values")
    if np.allclose(embeddings, 0.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            "all obj_dc_* values are zero; this looks like an undistilled "
            "Gaussian PLY"
        )
    if np.all(np.ptp(embeddings, axis=0) <= 1e-8):
        raise ValueError(
            "obj_dc_* values do not vary between Gaussians; semantic "
            "distillation appears incomplete"
        )
    return xyz, embeddings, max_scale, opacity, normals


def _count_opacity_eligible(
    gaussian_ply: PlyData, opacity_min: float, chunk_size: int
) -> int:
    opacity_logit = gaussian_ply["vertex"].data["opacity"]
    eligible = 0
    for start in range(0, len(opacity_logit), chunk_size):
        stop = min(start + chunk_size, len(opacity_logit))
        value = np.asarray(opacity_logit[start:stop], dtype=np.float32)
        opacity = 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
        eligible += int(np.count_nonzero(opacity >= opacity_min))
    return eligible


def _validate_embedding_signal(
    gaussian_ply: PlyData, schema: PlySchema, chunk_size: int
) -> None:
    """Validate distillation signal without allocating the full embedding."""

    vertex = gaussian_ply["vertex"].data
    minimum = np.full(len(schema.embedding_names), np.inf, dtype=np.float32)
    maximum = np.full(len(schema.embedding_names), -np.inf, dtype=np.float32)
    maximum_absolute = 0.0
    for start in range(0, schema.gaussian_count, chunk_size):
        stop = min(start + chunk_size, schema.gaussian_count)
        for column, name in enumerate(schema.embedding_names):
            value = np.asarray(vertex[name][start:stop], dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError("Gaussian object embeddings contain NaN or infinity")
            minimum[column] = min(minimum[column], float(np.min(value)))
            maximum[column] = max(maximum[column], float(np.max(value)))
            maximum_absolute = max(
                maximum_absolute, float(np.max(np.abs(value), initial=0.0))
            )
    if maximum_absolute <= 1e-8:
        raise ValueError(
            "all obj_dc_* values are zero; this looks like an undistilled "
            "Gaussian PLY"
        )
    if np.all(maximum - minimum <= 1e-8):
        raise ValueError(
            "obj_dc_* values do not vary between Gaussians; semantic "
            "distillation appears incomplete"
        )


def _classify_embeddings(
    embeddings: np.ndarray,
    classifier: Classifier,
    min_confidence: float,
    min_margin: float,
    unknown_id: int,
    label_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(embeddings, dtype=np.float32) @ classifier.weight.T
    logits += classifier.bias[None, :]
    top_label = np.argmax(logits, axis=1)

    logits -= np.max(logits, axis=1, keepdims=True)
    np.exp(logits, out=logits)
    denominator = np.sum(logits, axis=1)
    top_confidence = 1.0 / denominator
    if classifier.num_classes == 1:
        second_confidence = np.zeros_like(top_confidence)
    else:
        logits[np.arange(len(logits)), top_label] = 0.0
        second_confidence = np.max(logits, axis=1) / denominator

    label = top_label.astype(label_dtype, copy=False)
    confidence = top_confidence.astype(np.float32, copy=False)
    valid = (confidence >= min_confidence) & (
        confidence - second_confidence >= min_margin
    )
    label = np.where(valid, label, unknown_id).astype(label_dtype, copy=False)
    return label, confidence


def _open_output_array(
    stage_dir: Path, name: str, dtype: np.dtype, count: int
) -> np.memmap:
    return np.lib.format.open_memmap(
        stage_dir / name,
        mode="w+",
        dtype=dtype,
        shape=(int(count),),
    )


def _export_gaussian_labels(
    stage_dir: Path,
    embeddings: np.ndarray,
    opacity: np.ndarray,
    classifier: Classifier,
    chunk_size: int,
    min_confidence: float,
    min_margin: float,
    opacity_min: float,
    unknown_id: int,
    label_dtype: np.dtype,
    quiet: bool,
) -> int:
    count = len(embeddings)
    labels = _open_output_array(
        stage_dir, ARRAY_NAMES["gaussian_label"], label_dtype, count
    )
    confidence = _open_output_array(
        stage_dir, ARRAY_NAMES["gaussian_confidence"], np.dtype(np.float32), count
    )
    unknown_count = 0
    progress = Progress("Gaussian classification", count, quiet)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        chunk_label, chunk_confidence = _classify_embeddings(
            embeddings[start:stop],
            classifier,
            min_confidence,
            min_margin,
            unknown_id,
            label_dtype,
        )
        low_opacity = opacity[start:stop] < opacity_min
        chunk_label[low_opacity] = unknown_id
        chunk_confidence[low_opacity] = 0.0
        labels[start:stop] = chunk_label
        confidence[start:stop] = chunk_confidence
        unknown_count += int(np.count_nonzero(chunk_label == unknown_id))
        progress.update(stop)
    labels.flush()
    confidence.flush()
    del labels, confidence
    return unknown_count


def _tree_query(
    tree: cKDTree, points: np.ndarray, neighbors: int, workers: int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        distance, index = tree.query(points, k=neighbors, workers=workers)
    except TypeError:  # pragma: no cover - old SciPy compatibility
        distance, index = tree.query(points, k=neighbors)
    if neighbors == 1:
        distance = distance[:, None]
        index = index[:, None]
    return np.asarray(distance), np.asarray(index)


def _export_vertex_labels(
    stage_dir: Path,
    mesh_ply: PlyData,
    schema: PlySchema,
    tree: cKDTree,
    tree_source_indices: np.ndarray | None,
    embeddings: np.ndarray,
    max_scale: np.ndarray,
    opacity: np.ndarray,
    gaussian_normals: np.ndarray | None,
    classifier: Classifier,
    neighbors: int,
    chunk_size: int,
    support_sigma: float,
    scale_floor: float,
    normal_power: float,
    min_confidence: float,
    min_margin: float,
    unknown_id: int,
    label_dtype: np.dtype,
    workers: int,
    quiet: bool,
) -> tuple[int, int]:
    mesh_vertex = mesh_ply["vertex"].data
    count = schema.mesh_vertex_count
    labels = _open_output_array(
        stage_dir, ARRAY_NAMES["vertex_label"], label_dtype, count
    )
    confidence = _open_output_array(
        stage_dir, ARRAY_NAMES["vertex_confidence"], np.dtype(np.float32), count
    )
    unknown_count = 0
    unsupported_count = 0
    progress = Progress("Mesh vertex lifting", count, quiet)

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        size = stop - start
        xyz = np.empty((size, 3), dtype=np.float64)
        for column, name in enumerate(("x", "y", "z")):
            xyz[:, column] = mesh_vertex[name][start:stop]

        chunk_labels = np.full(size, unknown_id, dtype=label_dtype)
        chunk_confidence = np.zeros(size, dtype=np.float32)
        finite = np.isfinite(xyz).all(axis=1)
        mesh_normals: np.ndarray | None = None
        if normal_power > 0.0:
            if gaussian_normals is None:  # guarded by schema validation
                raise RuntimeError("Gaussian normals were not initialized")
            mesh_normals = np.empty((size, 3), dtype=np.float32)
            for column, name in enumerate(("nx", "ny", "nz")):
                mesh_normals[:, column] = mesh_vertex[name][start:stop]
            mesh_normal_norm = np.linalg.norm(mesh_normals, axis=1)
            valid_normal = np.isfinite(mesh_normals).all(axis=1) & (
                mesh_normal_norm > np.finfo(np.float32).eps
            )
            finite &= valid_normal
            mesh_normals[valid_normal] /= mesh_normal_norm[valid_normal, None]
        supported = np.zeros(size, dtype=bool)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            distance, index = _tree_query(tree, xyz[finite_indices], neighbors, workers)
            if tree_source_indices is not None:
                index = tree_source_indices[index]
            neighbor_scale = np.maximum(max_scale[index], scale_floor)
            normalized_distance = distance / neighbor_scale
            in_support = np.isfinite(distance) & (normalized_distance <= support_sigma)
            weight = opacity[index] * np.exp(
                -0.5 * np.square(normalized_distance, dtype=np.float64)
            )
            weight = weight.astype(np.float32, copy=False)
            if normal_power > 0.0:
                assert gaussian_normals is not None and mesh_normals is not None
                normal_similarity = np.abs(
                    np.einsum(
                        "nkd,nd->nk",
                        gaussian_normals[index],
                        mesh_normals[finite_indices],
                        optimize=True,
                    )
                )
                np.clip(normal_similarity, 0.0, 1.0, out=normal_similarity)
                weight *= np.power(normal_similarity, normal_power)
            weight[~in_support] = 0.0
            weight_sum = np.sum(weight, axis=1)
            finite_supported = weight_sum > np.finfo(np.float32).eps

            if np.any(finite_supported):
                selected_index = index[finite_supported]
                selected_weight = weight[finite_supported]
                interpolated = np.einsum(
                    "nk,nkf->nf",
                    selected_weight,
                    embeddings[selected_index],
                    optimize=True,
                )
                interpolated /= weight_sum[finite_supported, None]
                selected_labels, selected_confidence = _classify_embeddings(
                    interpolated,
                    classifier,
                    min_confidence,
                    min_margin,
                    unknown_id,
                    label_dtype,
                )
                output_indices = finite_indices[finite_supported]
                chunk_labels[output_indices] = selected_labels
                chunk_confidence[output_indices] = selected_confidence
                supported[output_indices] = True

        labels[start:stop] = chunk_labels
        confidence[start:stop] = chunk_confidence
        unsupported_count += int(np.count_nonzero(~supported))
        unknown_count += int(np.count_nonzero(chunk_labels == unknown_id))
        progress.update(stop)

    labels.flush()
    confidence.flush()
    del labels, confidence
    return unknown_count, unsupported_count


def _face_consensus(
    vertex_labels: np.ndarray,
    vertex_confidence: np.ndarray,
    unknown_id: int,
    label_dtype: np.dtype,
    min_agreement: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(vertex_labels)
    output_label = np.full(count, unknown_id, dtype=label_dtype)
    output_confidence = np.zeros(count, dtype=np.float32)
    known = vertex_labels != unknown_id

    if min_agreement == 3:
        selected = (
            known[:, 0]
            & known[:, 1]
            & known[:, 2]
            & (vertex_labels[:, 0] == vertex_labels[:, 1])
            & (vertex_labels[:, 0] == vertex_labels[:, 2])
        )
        output_label[selected] = vertex_labels[selected, 0]
    elif min_agreement == 2:
        pair_01 = (
            known[:, 0] & known[:, 1] & (vertex_labels[:, 0] == vertex_labels[:, 1])
        )
        pair_02 = (
            known[:, 0] & known[:, 2] & (vertex_labels[:, 0] == vertex_labels[:, 2])
        )
        pair_12 = (
            known[:, 1] & known[:, 2] & (vertex_labels[:, 1] == vertex_labels[:, 2])
        )
        output_label[pair_01 | pair_02] = vertex_labels[pair_01 | pair_02, 0]
        only_12 = pair_12 & ~(pair_01 | pair_02)
        output_label[only_12] = vertex_labels[only_12, 1]
    else:
        masked_confidence = np.where(known, vertex_confidence, -1.0)
        winner = np.argmax(masked_confidence, axis=1)
        selected = np.any(known, axis=1)
        rows = np.arange(count)[selected]
        output_label[selected] = vertex_labels[rows, winner[selected]]

    selected = output_label != unknown_id
    if np.any(selected):
        agreement = known & (vertex_labels == output_label[:, None])
        # Confidence is support mass across all three corners, not merely the
        # mean of agreeing corners.  Two-corner decisions are thus penalized.
        output_confidence[selected] = (
            np.sum(vertex_confidence[selected] * agreement[selected], axis=1) / 3.0
        )
    return output_label, output_confidence


def _export_face_labels(
    stage_dir: Path,
    mesh_ply: PlyData,
    schema: PlySchema,
    chunk_size: int,
    unknown_id: int,
    label_dtype: np.dtype,
    min_agreement: int,
    quiet: bool,
) -> int:
    vertex_labels = np.load(stage_dir / ARRAY_NAMES["vertex_label"], mmap_mode="r")
    vertex_confidence = np.load(
        stage_dir / ARRAY_NAMES["vertex_confidence"], mmap_mode="r"
    )
    faces = mesh_ply["face"].data[schema.mesh_face_property]
    count = schema.mesh_face_count
    output_labels = _open_output_array(
        stage_dir, ARRAY_NAMES["face_label"], label_dtype, count
    )
    output_confidence = _open_output_array(
        stage_dir, ARRAY_NAMES["face_confidence"], np.dtype(np.float32), count
    )
    unknown_count = 0
    progress = Progress("Mesh face voting", count, quiet)

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        face_indices = np.asarray(faces[start:stop])
        if face_indices.ndim != 2 or face_indices.shape[1] != 3:
            raise ValueError("encountered a non-triangle face")
        if np.any(face_indices < 0) or np.any(face_indices >= schema.mesh_vertex_count):
            raise ValueError("mesh contains a face with an invalid vertex index")
        corner_labels = np.asarray(vertex_labels[face_indices])
        corner_confidence = np.asarray(vertex_confidence[face_indices])
        face_label, face_confidence = _face_consensus(
            corner_labels,
            corner_confidence,
            unknown_id,
            label_dtype,
            min_agreement,
        )
        output_labels[start:stop] = face_label
        output_confidence[start:stop] = face_confidence
        unknown_count += int(np.count_nonzero(face_label == unknown_id))
        progress.update(stop)

    output_labels.flush()
    output_confidence.flush()
    del output_labels, output_confidence, vertex_labels, vertex_confidence
    return unknown_count


def _class_name(scene_info: Mapping[str, Any], class_id: int) -> str:
    for key in ("class_names", "label_names", "instance_names"):
        names = scene_info.get(key)
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            if class_id < len(names):
                value = names[class_id]
                if isinstance(value, Mapping):
                    value = value.get("name")
                if value is not None:
                    return str(value)
        elif isinstance(names, Mapping):
            value = names.get(str(class_id), names.get(class_id))
            if isinstance(value, Mapping):
                value = value.get("name")
            if value is not None:
                return str(value)
    return "background" if class_id == 0 else f"instance_{class_id}"


def _palette(scene_info: Mapping[str, Any], unknown_id: int) -> dict[str, Any]:
    classes = []
    for class_id in range(int(scene_info["num_classes"])):
        if class_id == 0:
            color = [0, 0, 0]
        else:
            hue = (class_id * 0.618033988749895) % 1.0
            saturation = 0.62 + 0.12 * ((class_id * 37) % 3) / 2.0
            value = 0.88 + 0.10 * ((class_id * 53) % 2)
            color = [
                int(round(channel * 255.0))
                for channel in colorsys.hsv_to_rgb(hue, saturation, value)
            ]
        classes.append(
            {
                "id": class_id,
                "name": _class_name(scene_info, class_id),
                "color_rgb": color,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "classes": classes,
        "unknown": {
            "id": unknown_id,
            "name": "unknown",
            "color_rgb": [127, 127, 127],
        },
    }


def _write_colored_mesh(
    path: Path,
    stage_dir: Path,
    mesh_ply: PlyData,
    schema: PlySchema,
    palette: Mapping[str, Any],
    label_dtype: np.dtype,
    chunk_size: int,
    quiet: bool,
) -> None:
    """Stream a binary semantic mesh without materializing the source mesh."""

    label_ply_type = "ushort" if label_dtype == np.dtype(np.uint16) else "uint"
    label_binary_dtype = "<u2" if label_dtype == np.dtype(np.uint16) else "<u4"
    vertex_fields: list[tuple[str, str]] = [
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
    ]
    if schema.mesh_has_normals:
        vertex_fields.extend((("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8")))
    vertex_fields.extend(
        (
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("instance_id", label_binary_dtype),
            ("semantic_confidence", "<f4"),
        )
    )
    vertex_dtype = np.dtype(vertex_fields)
    face_dtype = np.dtype(
        {
            "names": ("count", "vertex_indices"),
            "formats": ("u1", ("<u4", (3,))),
            "offsets": (0, 1),
            "itemsize": 13,
        }
    )

    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment PaintMesh instance labels; .npy sidecars are authoritative",
        f"element vertex {schema.mesh_vertex_count}",
        "property double x",
        "property double y",
        "property double z",
    ]
    if schema.mesh_has_normals:
        header.extend(
            ("property double nx", "property double ny", "property double nz")
        )
    header.extend(
        (
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"property {label_ply_type} instance_id",
            "property float semantic_confidence",
            f"element face {schema.mesh_face_count}",
            "property list uchar uint vertex_indices",
            "end_header",
            "",
        )
    )

    class_colors = np.asarray(
        [entry["color_rgb"] for entry in palette["classes"]], dtype=np.uint8
    )
    unknown_color = np.asarray(palette["unknown"]["color_rgb"], dtype=np.uint8)
    vertex_labels = np.load(stage_dir / ARRAY_NAMES["vertex_label"], mmap_mode="r")
    vertex_confidence = np.load(
        stage_dir / ARRAY_NAMES["vertex_confidence"], mmap_mode="r"
    )
    mesh_vertex = mesh_ply["vertex"].data
    faces = mesh_ply["face"].data[schema.mesh_face_property]

    vertex_progress = Progress("Colored mesh vertices", schema.mesh_vertex_count, quiet)
    face_progress = Progress("Colored mesh faces", schema.mesh_face_count, quiet)
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        for start in range(0, schema.mesh_vertex_count, chunk_size):
            stop = min(start + chunk_size, schema.mesh_vertex_count)
            output = np.empty(stop - start, dtype=vertex_dtype)
            for name in ("x", "y", "z"):
                output[name] = mesh_vertex[name][start:stop]
            if schema.mesh_has_normals:
                for name in ("nx", "ny", "nz"):
                    output[name] = mesh_vertex[name][start:stop]

            labels = np.asarray(vertex_labels[start:stop])
            colors = np.empty((stop - start, 3), dtype=np.uint8)
            colors[:] = unknown_color
            known = labels < len(class_colors)
            colors[known] = class_colors[labels[known]]
            output["red"] = colors[:, 0]
            output["green"] = colors[:, 1]
            output["blue"] = colors[:, 2]
            output["instance_id"] = labels
            output["semantic_confidence"] = vertex_confidence[start:stop]
            output.tofile(handle)
            vertex_progress.update(stop)

        for start in range(0, schema.mesh_face_count, chunk_size):
            stop = min(start + chunk_size, schema.mesh_face_count)
            input_indices = np.asarray(faces[start:stop])
            if np.any(input_indices >= schema.mesh_vertex_count):
                raise ValueError("mesh contains a face with an invalid vertex index")
            output = np.empty(stop - start, dtype=face_dtype)
            output["count"] = 3
            output["vertex_indices"] = input_indices
            output.tofile(handle)
            face_progress.update(stop)
        handle.flush()
        os.fsync(handle.fileno())
    del vertex_labels, vertex_confidence


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, include_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        result["sha256"] = _sha256(path)
    return result


def _output_artifact(
    path: Path, expected_shape: Sequence[int] | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": path.name,
        "size_bytes": int(path.stat().st_size),
    }
    if expected_shape is not None:
        array = np.load(path, mmap_mode="r")
        if tuple(array.shape) != tuple(expected_shape):
            raise RuntimeError(
                f"unexpected output shape for {path.name}: {array.shape}, "
                f"expected {tuple(expected_shape)}"
            )
        result["shape"] = list(array.shape)
        result["dtype"] = str(array.dtype)
        del array
    return result


def _validate_args(args: argparse.Namespace) -> None:
    for label, path in (
        ("Gaussian PLY", args.gaussian_ply),
        ("classifier", args.classifier),
        ("scene info", args.scene_info),
        ("mesh PLY", args.mesh),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.neighbors <= 0:
        raise ValueError("neighbors must be positive")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    for name, value in (
        ("support-sigma", args.support_sigma),
        ("scale-floor", args.scale_floor),
        ("opacity-min", args.opacity_min),
        ("normal-power", args.normal_power),
        ("min-confidence", args.min_confidence),
        ("min-margin", args.min_margin),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if args.support_sigma <= 0.0:
        raise ValueError("support-sigma must be positive")
    if args.scale_floor <= 0.0:
        raise ValueError("scale-floor must be positive")
    if not 0.0 <= args.opacity_min <= 1.0:
        raise ValueError("opacity-min must be in [0, 1]")
    if args.normal_power < 0.0:
        raise ValueError("normal-power must be non-negative")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("min-confidence must be in [0, 1]")
    if not 0.0 <= args.min_margin <= 1.0:
        raise ValueError("min-margin must be in [0, 1]")
    if args.face_min_agreement not in (1, 2, 3):
        raise ValueError("face-min-agreement must be 1, 2, or 3")
    if args.workers == 0 or args.workers < -1:
        raise ValueError("workers must be -1 or a positive integer")


def _managed_existing(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(name for name in MANAGED_NAMES if (output_dir / name).exists())


def _dry_run_plan(
    args: argparse.Namespace,
    schema: PlySchema,
    classifier: Classifier,
    label_dtype: np.dtype,
    opacity_eligible_count: int,
    effective_neighbors: int,
) -> dict[str, Any]:
    label_bytes = label_dtype.itemsize * (
        schema.gaussian_count + schema.mesh_vertex_count + schema.mesh_face_count
    )
    confidence_bytes = np.dtype(np.float32).itemsize * (
        schema.gaussian_count + schema.mesh_vertex_count + schema.mesh_face_count
    )
    colored_vertex_bytes = 3 * 8 + 3 + label_dtype.itemsize + 4
    if schema.mesh_has_normals:
        colored_vertex_bytes += 3 * 8
    colored_mesh_bytes = (
        colored_vertex_bytes * schema.mesh_vertex_count + 13 * schema.mesh_face_count
        if args.write_colored_ply
        else 0
    )
    return {
        "dry_run": True,
        "inputs": {
            "gaussians": str(args.gaussian_ply.resolve()),
            "classifier": str(args.classifier.resolve()),
            "scene_info": str(args.scene_info.resolve()),
            "mesh": str(args.mesh.resolve()),
        },
        "counts": {
            "gaussians": schema.gaussian_count,
            "opacity_eligible_gaussians": opacity_eligible_count,
            "mesh_vertices": schema.mesh_vertex_count,
            "mesh_triangles": schema.mesh_face_count,
            "classes": classifier.num_classes,
            "embedding_dimensions": classifier.embedding_dim,
        },
        "parameters": {
            "neighbors_requested": args.neighbors,
            "neighbors_effective": effective_neighbors,
            "chunk_size": args.chunk_size,
            "support_sigma": args.support_sigma,
            "scale_floor": args.scale_floor,
            "opacity_min": args.opacity_min,
            "normal_power": args.normal_power,
            "min_confidence": args.min_confidence,
            "min_margin": args.min_margin,
            "face_min_agreement": args.face_min_agreement,
            "unknown_id": args.unknown_id,
            "label_dtype": str(label_dtype),
            "write_colored_ply": args.write_colored_ply,
        },
        "output_dir": str(args.output_dir.resolve()),
        "existing_managed_outputs": _managed_existing(args.output_dir),
        "estimated_sidecar_bytes": int(label_bytes + confidence_bytes),
        "estimated_colored_mesh_bytes": int(colored_mesh_bytes),
    }


def _commit(stage_dir: Path, output_dir: Path) -> None:
    # A manifest is the completion marker.  Remove the previous marker before
    # replacing any sidecar so readers never accept a partially replaced set.
    old_manifest = output_dir / MANIFEST_NAME
    if old_manifest.exists():
        old_manifest.unlink()
    for name in (*ARRAY_NAMES.values(), PALETTE_NAME, COLORED_MESH_NAME):
        staged = stage_dir / name
        destination = output_dir / name
        if staged.exists():
            os.replace(staged, destination)
        elif destination.exists():
            destination.unlink()
    os.replace(stage_dir / MANIFEST_NAME, output_dir / MANIFEST_NAME)


def export(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    scene_info = _read_scene_info(args.scene_info)
    classifier = _load_classifier(args.classifier)
    label_dtype = _label_dtype(classifier.num_classes, args.unknown_id)
    schema, gaussian_ply, mesh_ply = _inspect_ply(
        args.gaussian_ply,
        args.mesh,
        classifier,
        scene_info,
        args.normal_power,
    )
    if args.dry_run:
        _validate_embedding_signal(gaussian_ply, schema, args.chunk_size)
        opacity_eligible_count = _count_opacity_eligible(
            gaussian_ply, args.opacity_min, args.chunk_size
        )
        if opacity_eligible_count == 0:
            raise ValueError(
                f"no Gaussian satisfies opacity-min={args.opacity_min}; "
                "lower the threshold"
            )
        effective_neighbors = min(args.neighbors, opacity_eligible_count)
        plan = _dry_run_plan(
            args,
            schema,
            classifier,
            label_dtype,
            opacity_eligible_count,
            effective_neighbors,
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return plan

    existing = _managed_existing(args.output_dir)
    if existing and not args.overwrite:
        raise FileExistsError(
            f"managed outputs already exist in {args.output_dir}: {existing}; "
            "pass --overwrite to replace them"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = args.output_dir / f".semantic-export-{uuid.uuid4().hex}"
    stage_dir.mkdir()
    started = time.monotonic()
    stats = ExportStats()
    try:
        _log("Loading Gaussian attributes...", args.quiet)
        xyz, embeddings, max_scale, opacity, gaussian_normals = _extract_gaussians(
            gaussian_ply, schema, use_normals=args.normal_power > 0.0
        )

        stats.gaussian_unknown = _export_gaussian_labels(
            stage_dir,
            embeddings,
            opacity,
            classifier,
            args.chunk_size,
            args.min_confidence,
            args.min_margin,
            args.opacity_min,
            args.unknown_id,
            label_dtype,
            args.quiet,
        )

        eligible_indices = np.flatnonzero(opacity >= args.opacity_min)
        if len(eligible_indices) == 0:
            raise ValueError(
                f"no Gaussian satisfies opacity-min={args.opacity_min}; "
                "lower the threshold"
            )
        effective_neighbors = min(args.neighbors, len(eligible_indices))
        if len(eligible_indices) == schema.gaussian_count:
            tree_points = xyz
            tree_source_indices = None
        else:
            tree_points = xyz[eligible_indices]
            tree_source_indices = eligible_indices
        _log(
            f"Building cKDTree for {len(eligible_indices):,} opacity-eligible "
            f"Gaussians ({schema.gaussian_count:,} total)...",
            args.quiet,
        )
        tree = cKDTree(tree_points, compact_nodes=True, balanced_tree=True)
        stats.vertex_unknown, stats.vertex_unsupported = _export_vertex_labels(
            stage_dir,
            mesh_ply,
            schema,
            tree,
            tree_source_indices,
            embeddings,
            max_scale,
            opacity,
            gaussian_normals,
            classifier,
            effective_neighbors,
            args.chunk_size,
            args.support_sigma,
            args.scale_floor,
            args.normal_power,
            args.min_confidence,
            args.min_margin,
            args.unknown_id,
            label_dtype,
            args.workers,
            args.quiet,
        )
        opacity_eligible_count = len(eligible_indices)
        del (
            tree,
            tree_points,
            tree_source_indices,
            eligible_indices,
            xyz,
            embeddings,
            max_scale,
            opacity,
            gaussian_normals,
        )

        stats.face_unknown = _export_face_labels(
            stage_dir,
            mesh_ply,
            schema,
            args.chunk_size,
            args.unknown_id,
            label_dtype,
            args.face_min_agreement,
            args.quiet,
        )
        palette = _palette(scene_info, args.unknown_id)
        _write_json(stage_dir / PALETTE_NAME, palette)
        if args.write_colored_ply:
            _write_colored_mesh(
                stage_dir / COLORED_MESH_NAME,
                stage_dir,
                mesh_ply,
                schema,
                palette,
                label_dtype,
                args.chunk_size,
                args.quiet,
            )

        input_artifacts = {
            "gaussian_ply": _artifact(args.gaussian_ply, args.hash_inputs),
            "classifier": _artifact(args.classifier, args.hash_inputs),
            "scene_info": _artifact(args.scene_info, args.hash_inputs),
            "mesh": _artifact(args.mesh, args.hash_inputs),
        }
        counts = {
            "classes": classifier.num_classes,
            "embedding_dimensions": classifier.embedding_dim,
            "gaussians": schema.gaussian_count,
            "opacity_eligible_gaussians": opacity_eligible_count,
            "gaussian_unknown": stats.gaussian_unknown,
            "mesh_vertices": schema.mesh_vertex_count,
            "vertex_unknown": stats.vertex_unknown,
            "vertex_unsupported": stats.vertex_unsupported,
            "mesh_triangles": schema.mesh_face_count,
            "face_unknown": stats.face_unknown,
        }
        rates = {
            "gaussian_unknown": stats.gaussian_unknown / schema.gaussian_count,
            "vertex_unknown": stats.vertex_unknown / schema.mesh_vertex_count,
            "vertex_unsupported": stats.vertex_unsupported / schema.mesh_vertex_count,
            "face_unknown": stats.face_unknown / schema.mesh_face_count,
        }
        outputs = {
            key: _output_artifact(
                stage_dir / name,
                expected_shape=(
                    (
                        schema.gaussian_count
                        if key.startswith("gaussian")
                        else (
                            schema.mesh_vertex_count
                            if key.startswith("vertex")
                            else schema.mesh_face_count
                        )
                    ),
                ),
            )
            for key, name in ARRAY_NAMES.items()
        }
        outputs["palette"] = _output_artifact(stage_dir / PALETTE_NAME)
        if args.write_colored_ply:
            outputs["semantic_mesh"] = _output_artifact(stage_dir / COLORED_MESH_NAME)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "semantic_type": "instance",
            "inputs": input_artifacts,
            "parameters": {
                "neighbors_requested": args.neighbors,
                "neighbors_effective": effective_neighbors,
                "chunk_size": args.chunk_size,
                "support_sigma": args.support_sigma,
                "scale_floor": args.scale_floor,
                "opacity_min": args.opacity_min,
                "normal_power": args.normal_power,
                "min_confidence": args.min_confidence,
                "min_margin": args.min_margin,
                "face_min_agreement": args.face_min_agreement,
                "unknown_id": args.unknown_id,
                "label_dtype": str(label_dtype),
                "workers": args.workers,
                "write_colored_ply": args.write_colored_ply,
                "input_integrity": "sha256" if args.hash_inputs else "size+mtime_ns",
            },
            "algorithm": {
                "gaussian": "softmax(linear(obj_dc))",
                "vertex": (
                    "classify(opacity/normal-weighted Gaussian embedding "
                    "interpolation; isotropic max-scale support)"
                ),
                "face": (
                    f"triangle corner consensus with at least "
                    f"{args.face_min_agreement} agreeing known labels"
                ),
                "face_confidence": "sum(agreeing corner confidence) / 3",
            },
            "counts": counts,
            "unknown_rates": rates,
            "outputs": outputs,
        }
        _write_json(stage_dir / MANIFEST_NAME, manifest)
        _commit(stage_dir, args.output_dir)
        _log(f"Semantic sidecars written to {args.output_dir}", args.quiet)
        return manifest
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lift Inpaint360GS per-Gaussian object embeddings to an EDGS triangle "
            "mesh and export mmap-friendly semantic sidecars."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--scene-info", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument(
        "--support-sigma",
        type=float,
        default=3.0,
        help="reject neighbors farther than this multiple of their maximum scale",
    )
    parser.add_argument(
        "--scale-floor",
        type=float,
        default=1e-8,
        help="minimum activated Gaussian scale used in distance normalization",
    )
    parser.add_argument(
        "--opacity-min",
        type=float,
        default=0.01,
        help="ignore Gaussians whose activated opacity is below this value",
    )
    parser.add_argument(
        "--normal-power",
        type=float,
        default=2.0,
        help=(
            "exponent for abs(mesh normal dot Gaussian smallest-axis normal); "
            "zero disables normal consistency"
        ),
    )
    parser.add_argument("--min-confidence", type=float, default=0.10)
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.02,
        help="minimum top-1 minus top-2 softmax probability",
    )
    parser.add_argument(
        "--face-min-agreement",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="minimum number of triangle corners that must agree",
    )
    parser.add_argument("--unknown-id", type=int, default=65535)
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="cKDTree query workers; -1 uses all available CPU cores",
    )
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="SHA256 all inputs; robust but expensive for large PLY files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate schemas and print the execution plan without writing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace semantic sidecars already present in output-dir",
    )
    parser.add_argument(
        "--write-colored-ply",
        action="store_true",
        help="stream semantic_mesh.ply with label colors and triangle geometry",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        export(args)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
