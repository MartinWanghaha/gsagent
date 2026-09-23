"""COLMAP visibility overlap estimation for MV-RoMa camera planning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from source.vendor import bootstrap_gaussian_splatting


@dataclass(frozen=True)
class OverlapEstimate:
    """Directed view overlap plus provenance used for diagnostics."""

    matrix: np.ndarray
    camera_names: tuple[str, ...]
    visible_point_counts: tuple[int, ...]
    sparse_model_path: Path | None
    provider: str = "colmap_tracks"


def resolve_colmap_sparse_model(
    *,
    image_root: Path | None,
    configured_path: str | Path | None,
) -> Path:
    """Resolve a COLMAP model directory without depending on process cwd."""

    if configured_path is not None:
        text = str(configured_path).strip()
        if text and text.lower() not in {"none", "null"}:
            candidate = Path(text).expanduser().resolve()
            if not candidate.is_dir():
                raise FileNotFoundError(
                    f"configured COLMAP sparse model does not exist: {candidate}"
                )
            return candidate

    if image_root is None:
        raise ValueError(
            "init_wC.mvroma.group_planner.sparse_model_path is required when "
            "init_wC.mvroma.image_root is not set"
        )
    dataset_root = image_root.expanduser().resolve().parent
    candidates = (dataset_root / "sparse" / "0", dataset_root / "sparse")
    for candidate in candidates:
        if (candidate / "images.bin").is_file() or (candidate / "images.txt").is_file():
            return candidate
    raise FileNotFoundError(
        "could not find COLMAP images.bin/images.txt under "
        f"{dataset_root / 'sparse'}; configure group_planner.sparse_model_path"
    )


def _name_aliases(name: str) -> frozenset[str]:
    normalized = str(name).replace("\\", "/").lstrip("./")
    path = PurePosixPath(normalized)
    return frozenset((normalized, path.name, path.stem))


def _match_colmap_images(
    cameras: Sequence[Any], colmap_images: Mapping[int, Any]
) -> list[Any]:
    aliases: dict[str, set[int]] = defaultdict(set)
    for image_id, image in colmap_images.items():
        for alias in _name_aliases(str(image.name)):
            aliases[alias].add(int(image_id))

    matched: list[Any] = []
    used_ids: set[int] = set()
    for camera_index, camera in enumerate(cameras):
        image_name = str(getattr(camera, "image_name", ""))
        candidates: set[int] = set()
        for alias in _name_aliases(image_name):
            candidates.update(aliases.get(alias, ()))
        if len(candidates) != 1:
            raise ValueError(
                "could not uniquely map training camera "
                f"{camera_index} ({image_name!r}) to COLMAP; candidates={sorted(candidates)}"
            )
        image_id = next(iter(candidates))
        if image_id in used_ids:
            raise ValueError(
                f"multiple training cameras map to COLMAP image id {image_id}"
            )
        used_ids.add(image_id)
        matched.append(colmap_images[image_id])
    return matched


def build_colmap_visibility_overlap(
    cameras: Sequence[Any],
    colmap_images: Mapping[int, Any],
    *,
    sparse_model_path: Path = Path("."),
) -> OverlapEstimate:
    """Compute ``O[i,j] = |P_i intersect P_j| / |P_i|`` from point tracks."""

    if len(cameras) < 2:
        raise ValueError("visibility overlap requires at least two cameras")
    matched = _match_colmap_images(cameras, colmap_images)
    observations: list[np.ndarray] = []
    point_to_views: dict[int, list[int]] = defaultdict(list)
    for camera_index, image in enumerate(matched):
        point_ids = np.asarray(image.point3D_ids, dtype=np.int64)
        point_ids = np.unique(point_ids[point_ids >= 0])
        if point_ids.size == 0:
            raise ValueError(
                f"training camera {camera_index} ({image.name!r}) has no "
                "COLMAP 3D observations"
            )
        observations.append(point_ids)
        for point_id in point_ids:
            point_to_views[int(point_id)].append(camera_index)

    camera_count = len(cameras)
    intersections = np.zeros((camera_count, camera_count), dtype=np.int32)
    for view_indices in point_to_views.values():
        indices = np.asarray(view_indices, dtype=np.int64)
        intersections[np.ix_(indices, indices)] += 1

    point_counts = np.asarray(
        [len(values) for values in observations], dtype=np.int64
    )
    overlap = intersections.astype(np.float64) / point_counts[:, None]
    np.fill_diagonal(overlap, 0.0)
    return OverlapEstimate(
        matrix=overlap,
        camera_names=tuple(str(image.name) for image in matched),
        visible_point_counts=tuple(int(value) for value in point_counts),
        sparse_model_path=sparse_model_path.resolve(),
        provider="colmap_tracks",
    )


def build_matcher_visibility_overlap(
    matrix: np.ndarray,
    camera_names: Sequence[str],
) -> OverlapEstimate:
    """Wrap matcher visibility ratios with the same planner contract."""

    overlap = np.asarray(matrix, dtype=np.float64)
    count = len(camera_names)
    if overlap.shape != (count, count):
        raise ValueError("matcher overlap matrix and camera names differ in size")
    if not np.isfinite(overlap).all() or np.any(overlap < 0) or np.any(overlap > 1):
        raise ValueError("matcher overlap values must be finite and in [0,1]")
    overlap = overlap.copy()
    np.fill_diagonal(overlap, 0.0)
    return OverlapEstimate(
        matrix=overlap,
        camera_names=tuple(str(name) for name in camera_names),
        visible_point_counts=tuple(0 for _ in range(count)),
        sparse_model_path=None,
        provider="ufm_visibility",
    )


def estimate_colmap_visibility_overlap(
    cameras: Sequence[Any], sparse_model_path: Path
) -> OverlapEstimate:
    """Load COLMAP image tracks and compute directed visibility overlap."""

    binary_path = sparse_model_path / "images.bin"
    text_path = sparse_model_path / "images.txt"
    bootstrap_gaussian_splatting()
    from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text

    if binary_path.is_file():
        colmap_images = read_extrinsics_binary(str(binary_path))
    elif text_path.is_file():
        colmap_images = read_extrinsics_text(str(text_path))
    else:
        raise FileNotFoundError(
            f"COLMAP model has neither images.bin nor images.txt: {sparse_model_path}"
        )
    return build_colmap_visibility_overlap(
        cameras,
        colmap_images,
        sparse_model_path=sparse_model_path,
    )
