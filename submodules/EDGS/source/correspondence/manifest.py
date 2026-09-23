"""Reproducible, compact artifacts for MV-RoMa correspondence initialization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from .contracts import DirectedPair, PairQuality
from .planning import CameraGroupPlan


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return _plain(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        _plain(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _camera_digest(cameras: Sequence[Any]) -> str:
    records = []
    for index, camera in enumerate(cameras):
        records.append(
            {
                "index": index,
                "name": str(getattr(camera, "image_name", index)),
                "width": int(getattr(camera, "image_width", 0)),
                "height": int(getattr(camera, "image_height", 0)),
                "projection": torch.as_tensor(
                    camera.full_proj_transform
                ).detach().cpu().tolist(),
            }
        )
    return hashlib.sha256(_canonical_bytes(records)).hexdigest()


def _sparse_digest(path_value: Any) -> str | None:
    if path_value is None:
        return None
    root = Path(str(path_value)).expanduser()
    if not root.is_dir():
        return None
    files = []
    for stem in ("cameras", "images", "points3D"):
        for suffix in (".bin", ".txt"):
            candidate = root / f"{stem}{suffix}"
            if candidate.is_file():
                files.append((candidate.name, _sha256_file(candidate)))
                break
    return hashlib.sha256(_canonical_bytes(files)).hexdigest() if files else None


def write_mvroma_artifacts(
    model_path: str | Path,
    *,
    mvroma_config: Any,
    cameras: Sequence[Any],
    overlap_matrix: np.ndarray,
    group_plan: CameraGroupPlan,
    pair_quality: Mapping[DirectedPair, PairQuality],
    diagnostics: Mapping[str, Any],
) -> str:
    """Atomically publish only compact provenance and quality sidecars."""

    output = Path(model_path) / "correspondence_init"
    output.mkdir(parents=True, exist_ok=True)
    config = _plain(mvroma_config)
    planner = config.get("group_planner", {}) if isinstance(config, dict) else {}
    identity = {
        "backend": "mvroma",
        "schema_version": int(config.get("schema_version", 2)),
        "algorithm_version": config.get("algorithm_version", "legacy"),
        "config": config,
        "camera_digest": _camera_digest(cameras),
        "camera_names": [
            str(getattr(camera, "image_name", index))
            for index, camera in enumerate(cameras)
        ],
        "sparse_model_digest": _sparse_digest(planner.get("sparse_model_path")),
        "effective_primary_group_budget": group_plan.primary_group_count,
        "group_plan_digest": hashlib.sha256(
            _canonical_bytes(
                [
                    [group.source_index, list(group.target_indices)]
                    for group in group_plan.groups
                ]
            )
        ).hexdigest(),
    }
    artifact_id = hashlib.sha256(_canonical_bytes(identity)).hexdigest()

    groups_payload = {
        "primary_group_count": group_plan.primary_group_count,
        "groups": [
            {
                "source_index": group.source_index,
                "target_indices": list(group.target_indices),
                "reciprocal": index >= group_plan.primary_group_count,
            }
            for index, group in enumerate(group_plan.groups)
        ],
    }
    _atomic_json(output / "groups.json", groups_payload)
    overlap_temp = output / f".overlap.{os.getpid()}.tmp"
    with overlap_temp.open("wb") as handle:
        np.savez_compressed(handle, overlap=np.asarray(overlap_matrix, dtype=np.float32))
    os.replace(overlap_temp, output / "overlap.npz")

    quality_rows = [
        (
            pair.source_index,
            pair.target_index,
            quality.cycle_inlier_ratio,
            quality.mean_confidence,
            quality.valid_pixels,
        )
        for pair, quality in sorted(pair_quality.items())
    ]
    quality_temp = output / f".pair_quality.{os.getpid()}.tmp"
    with quality_temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            values=np.asarray(quality_rows, dtype=np.float64).reshape(-1, 5),
        )
    os.replace(quality_temp, output / "pair_quality.npz")

    files = {
        "groups": "groups.json",
        "overlap": "overlap.npz",
        "pair_quality": "pair_quality.npz",
    }

    manifest = {
        "kind": "edgs-mvroma-correspondence-init",
        "schema_version": 2,
        "complete": True,
        "artifact_id": artifact_id,
        "identity": identity,
        "diagnostics": _plain(diagnostics),
        "files": files,
        "file_sha256": {
            name: _sha256_file(output / relative)
            for name, relative in files.items()
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    return artifact_id


def write_pgsr_neighbors(
    model_path: str | Path,
    neighbor_map: Mapping[str, Sequence[Any]],
) -> Path:
    output = Path(model_path) / "correspondence_init" / "pgsr_neighbors.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(source): [
            str(getattr(camera, "image_name", camera)) for camera in neighbors
        ]
        for source, neighbors in neighbor_map.items()
    }
    _atomic_json(output, payload)
    return output


def load_pgsr_neighbors(
    model_path: str | Path,
    cameras: Sequence[Any],
) -> dict[str, list[Any]]:
    """Materialize a saved hybrid graph against the current Camera objects."""

    path = Path(model_path) / "correspondence_init" / "pgsr_neighbors.json"
    if not path.is_file():
        raise FileNotFoundError(
            "hybrid MV-RoMa checkpoint resume requires the saved PGSR graph: "
            f"{path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("saved hybrid PGSR graph must be a source-to-target mapping")
    by_name = {
        str(getattr(camera, "image_name", index)): camera
        for index, camera in enumerate(cameras)
    }
    if len(by_name) != len(cameras):
        raise ValueError("current camera image names are not unique")
    if set(payload) != set(by_name):
        raise ValueError("saved hybrid PGSR graph uses a different camera set")
    result = {}
    for source, target_names in payload.items():
        if not isinstance(target_names, list) or not all(
            isinstance(name, str) for name in target_names
        ):
            raise ValueError(f"saved hybrid PGSR targets for {source} are invalid")
        if len(set(target_names)) != len(target_names) or source in target_names:
            raise ValueError(
                f"saved hybrid PGSR targets for {source} contain duplicates or self"
            )
        missing = [name for name in target_names if name not in by_name]
        if missing:
            raise ValueError(
                f"saved hybrid PGSR graph has unknown targets for {source}: {missing}"
            )
        result[source] = [by_name[name] for name in target_names]
    return result
