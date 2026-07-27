"""Transfer learned Gaga labels from Gaussians to a Gaussian Wrapping mesh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from semantic import SemanticHead, semantic_palette


def _read_gaussians(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = PlyData.read(path)
    vertex = data["vertex"].data
    names = set(vertex.dtype.names or ())
    feature_names = [f"obj_dc_{index}" for index in range(16)]
    missing = set(("x", "y", "z", *feature_names)) - names
    if missing:
        raise ValueError(
            f"{path} is not a semantic Gaussian PLY; missing fields: {sorted(missing)}"
        )
    positions = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float32,
        copy=False,
    )
    features = np.column_stack([vertex[name] for name in feature_names]).astype(
        np.float32,
        copy=False,
    )
    return positions, features


def _decode_gaussian_labels(
    features: np.ndarray,
    checkpoint: Path,
) -> tuple[np.ndarray, int]:
    payload = torch.load(checkpoint, map_location="cpu")
    if int(payload.get("semantic_dim", -1)) != 16:
        raise ValueError("Semantic checkpoint does not contain 16D Gaga features")
    num_classes = int(payload["num_classes"])
    head = SemanticHead(16, num_classes).eval()
    head.load_state_dict(payload["head"])
    with torch.no_grad():
        feature_tensor = torch.from_numpy(features).T.contiguous()[None, :, :, None]
        labels = head(feature_tensor).argmax(dim=1).reshape(-1)
    return labels.numpy().astype(np.int64), num_classes


def _nearest_labels(
    vertices: np.ndarray,
    gaussian_positions: np.ndarray,
    gaussian_labels: np.ndarray,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError("scipy is required for semantic mesh transfer") from error
    tree = cKDTree(gaussian_positions)
    distances, indices = tree.query(vertices, k=1, workers=workers)
    return gaussian_labels[indices], distances.astype(np.float32, copy=False)


def _vertex_with_semantics(
    vertex: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    original_names = tuple(vertex.dtype.names or ())
    semantic_fields = {
        "semantic_id": "<u4",
        "red": "u1",
        "green": "u1",
        "blue": "u1",
    }
    descriptor = list(vertex.dtype.descr)
    descriptor.extend(
        (name, dtype)
        for name, dtype in semantic_fields.items()
        if name not in original_names
    )
    result = np.empty(vertex.shape, dtype=descriptor)
    for name in original_names:
        result[name] = vertex[name]
    result["semantic_id"] = labels.astype(np.uint32, copy=False)
    result["red"] = colors[:, 0]
    result["green"] = colors[:, 1]
    result["blue"] = colors[:, 2]
    return result


def export_semantic_mesh(args) -> dict:
    mesh_path = Path(args.mesh)
    semantic_ply = Path(args.semantic_ply)
    checkpoint = Path(args.semantic_checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    gaussian_positions, features = _read_gaussians(semantic_ply)
    gaussian_labels, num_classes = _decode_gaussian_labels(features, checkpoint)
    source = PlyData.read(mesh_path)
    vertex = source["vertex"].data
    vertices = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float32,
        copy=False,
    )
    labels, distances = _nearest_labels(
        vertices,
        gaussian_positions,
        gaussian_labels,
        args.workers,
    )
    colors = semantic_palette(num_classes)[labels]
    semantic_vertex = PlyElement.describe(
        _vertex_with_semantics(vertex, labels, colors),
        "vertex",
    )
    elements = [
        semantic_vertex if element.name == "vertex" else element
        for element in source.elements
    ]
    PlyData(
        elements,
        text=source.text,
        byte_order=source.byte_order,
        comments=source.comments,
        obj_info=source.obj_info,
    ).write(output)

    labels_output = (
        Path(args.labels_output)
        if args.labels_output
        else output.with_suffix(".semantic.npy")
    )
    distances_output = output.with_suffix(".semantic_distance.npy")
    np.save(labels_output, labels)
    np.save(distances_output, distances)
    metadata = {
        "source_mesh": str(mesh_path.resolve()),
        "semantic_gaussians": str(semantic_ply.resolve()),
        "semantic_checkpoint": str(checkpoint.resolve()),
        "num_classes": num_classes,
        "num_vertices": int(vertices.shape[0]),
        "transfer": "nearest_gaussian_center",
        "labels": str(labels_output.resolve()),
        "distances": str(distances_output.resolve()),
    }
    metadata_path = output.with_suffix(".semantic.json")
    with metadata_path.open("w", encoding="utf8") as stream:
        json.dump(metadata, stream, indent=2)
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a Gaussian Wrapping mesh with Gaga semantic labels"
    )
    parser.add_argument("--mesh", required=True, help="Gaussian Wrapping mesh PLY")
    parser.add_argument(
        "--semantic_ply",
        required=True,
        help="Semantic Gaussian PLY containing obj_dc_0..obj_dc_15",
    )
    parser.add_argument("--semantic_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels_output")
    parser.add_argument("--workers", type=int, default=-1)
    export_semantic_mesh(parser.parse_args(sys.argv[1:]))
