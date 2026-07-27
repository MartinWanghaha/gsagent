"""Deterministic CPU geometry metrics for point clouds and triangle meshes."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .types import TriangleMesh


def _points(value: np.ndarray | TriangleMesh, sample_count: int, seed: int) -> np.ndarray:
    if isinstance(value, TriangleMesh):
        return sample_mesh_surface(value, sample_count, seed=seed)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("point clouds must have shape [N, 3]")
    return np.ascontiguousarray(array)


def sample_mesh_surface(
    mesh: TriangleMesh,
    count: int = 100_000,
    *,
    seed: int = 0,
    face_chunk_size: int = 262_144,
) -> np.ndarray:
    """Area-weighted sampling without materializing every face's vertices.

    Only the scalar area vector scales with the source face count. Triangle
    coordinates are gathered for the sampled faces after the deterministic
    categorical draw, which keeps evaluation memory bounded for multi-million
    face meshes.
    """
    if count < 1:
        raise ValueError("sample count must be positive")
    if face_chunk_size < 1:
        raise ValueError("face_chunk_size must be positive")
    if not len(mesh.faces):
        return mesh.vertices.astype(np.float64, copy=True)

    area = np.empty(len(mesh.faces), dtype=np.float64)
    for start in range(0, len(mesh.faces), face_chunk_size):
        stop = min(start + face_chunk_size, len(mesh.faces))
        triangles = mesh.vertices[mesh.faces[start:stop]].astype(
            np.float64,
            copy=False,
        )
        area[start:stop] = 0.5 * np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
    valid = area > 1e-15
    if not valid.any():
        return mesh.vertices.astype(np.float64, copy=True)
    valid_faces = np.flatnonzero(valid)
    probabilities = area[valid_faces] / area[valid_faces].sum()
    rng = np.random.default_rng(seed)
    triangle_index = rng.choice(len(valid_faces), size=count, p=probabilities)
    chosen_faces = mesh.faces[valid_faces[triangle_index]]
    chosen = mesh.vertices[chosen_faces].astype(np.float64, copy=False)
    first = rng.random(count)
    second = rng.random(count)
    reflect = first + second > 1.0
    first[reflect] = 1.0 - first[reflect]
    second[reflect] = 1.0 - second[reflect]
    return (
        chosen[:, 0]
        + first[:, None] * (chosen[:, 1] - chosen[:, 0])
        + second[:, None] * (chosen[:, 2] - chosen[:, 0])
    )


def nearest_distances(
    source: np.ndarray,
    target: np.ndarray,
    *,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """One-way nearest distances on CPU, with a dependency-free fallback."""
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if not len(source):
        return np.empty((0,), dtype=np.float64)
    if not len(target):
        return np.full((len(source),), np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(target).query(source, k=1, workers=1)
        return np.asarray(distances, dtype=np.float64)
    except ImportError:
        output = np.empty((len(source),), dtype=np.float64)
        for start in range(0, len(source), chunk_size):
            chunk = source[start : start + chunk_size]
            minimum_squared = np.full(len(chunk), np.inf, dtype=np.float64)
            target_step = max(1, min(4096, 4_000_000 // max(len(chunk), 1)))
            for target_start in range(0, len(target), target_step):
                difference = chunk[:, None, :] - target[target_start : target_start + target_step]
                minimum_squared = np.minimum(
                    minimum_squared, np.min(np.sum(difference * difference, axis=-1), axis=1)
                )
            output[start : start + len(chunk)] = np.sqrt(minimum_squared)
        return output


def accuracy(prediction: np.ndarray, reference: np.ndarray) -> float:
    distances = nearest_distances(prediction, reference)
    return float(distances.mean()) if len(distances) else (0.0 if not len(reference) else np.inf)


def completeness(prediction: np.ndarray, reference: np.ndarray) -> float:
    distances = nearest_distances(reference, prediction)
    return float(distances.mean()) if len(distances) else (0.0 if not len(prediction) else np.inf)


def chamfer_distance(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    squared: bool = False,
) -> float:
    forward = nearest_distances(prediction, reference)
    backward = nearest_distances(reference, prediction)
    if not len(forward) and not len(backward):
        return 0.0
    if not len(forward) or not len(backward):
        return float("inf")
    if squared:
        forward = forward**2
        backward = backward**2
    return float(0.5 * (forward.mean() + backward.mean()))


def precision_recall_fscore(
    prediction: np.ndarray,
    reference: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Return F-score, precision and recall at a distance threshold."""
    if threshold <= 0:
        raise ValueError("F-score threshold must be positive")
    if not len(prediction) and not len(reference):
        return 1.0, 1.0, 1.0
    if not len(prediction) or not len(reference):
        return 0.0, 0.0, 0.0
    precision = float(np.mean(nearest_distances(prediction, reference) <= threshold))
    recall = float(np.mean(nearest_distances(reference, prediction) <= threshold))
    score = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return float(score), precision, recall


def f_score(prediction: np.ndarray, reference: np.ndarray, threshold: float) -> float:
    """Scalar F-score at ``threshold``; use precision_recall_fscore for all terms."""
    return precision_recall_fscore(prediction, reference, threshold)[0]


@dataclass(frozen=True)
class MeshMetrics:
    chamfer: float
    accuracy: float
    completeness: float
    f_score: float
    precision: float
    recall: float
    threshold: float

    def as_dict(self) -> dict[str, float]:
        return {
            "chamfer": self.chamfer,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "f_score": self.f_score,
            "precision": self.precision,
            "recall": self.recall,
            "threshold": self.threshold,
        }


def compute_mesh_metrics(
    prediction: np.ndarray | TriangleMesh,
    reference: np.ndarray | TriangleMesh,
    *,
    threshold: float = 0.01,
    sample_count: int = 100_000,
    seed: int = 0,
) -> MeshMetrics:
    prediction_points = _points(prediction, sample_count, seed)
    reference_points = _points(reference, sample_count, seed + 1)
    prediction_distances = nearest_distances(prediction_points, reference_points)
    reference_distances = nearest_distances(reference_points, prediction_points)

    if not len(prediction_points) and not len(reference_points):
        accuracy_value = completeness_value = chamfer_value = 0.0
    elif not len(prediction_points) or not len(reference_points):
        accuracy_value = completeness_value = chamfer_value = float("inf")
    else:
        accuracy_value = float(prediction_distances.mean())
        completeness_value = float(reference_distances.mean())
        chamfer_value = 0.5 * (accuracy_value + completeness_value)
    if not len(prediction_points) and not len(reference_points):
        score = precision = recall = 1.0
    elif not len(prediction_points) or not len(reference_points):
        score = precision = recall = 0.0
    else:
        precision = float(np.mean(prediction_distances <= threshold))
        recall = float(np.mean(reference_distances <= threshold))
        score = (
            0.0
            if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
    return MeshMetrics(
        chamfer=chamfer_value,
        accuracy=accuracy_value,
        completeness=completeness_value,
        f_score=score,
        precision=precision,
        recall=recall,
        threshold=float(threshold),
    )


mesh_metrics = compute_mesh_metrics
fscore = f_score
