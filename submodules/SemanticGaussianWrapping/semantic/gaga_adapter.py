"""Adapter for Gaga's view-consistent instance-mask observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


@dataclass(frozen=True)
class SemanticObservation:
    ids: Tensor
    confidence: Tensor
    boundary: Tensor

    def to(self, *args: Any, **kwargs: Any) -> "SemanticObservation":
        return SemanticObservation(
            self.ids.to(*args, **kwargs),
            self.confidence.to(*args, **kwargs),
            self.boundary.to(*args, **kwargs),
        )


def _boundary(ids: Tensor, ignore_label: int) -> Tensor:
    valid = ids != ignore_label
    result = torch.zeros_like(valid)
    dx = (ids[:, 1:] != ids[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    dy = (ids[1:] != ids[:-1]) & valid[1:] & valid[:-1]
    result[:, 1:] |= dx
    result[:, :-1] |= dx
    result[1:] |= dy
    result[:-1] |= dy
    return result.float()


def _decode_color_ids(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array.astype(np.int64, copy=False)
    if array.ndim != 3:
        raise ValueError(f"mask must have shape [H,W] or [H,W,C], got {array.shape}")
    if array.shape[2] == 1:
        return array[..., 0].astype(np.int64, copy=False)
    rgb = array[..., :3].astype(np.int64)
    # This is lossless for ID masks written as 24-bit PNG and deterministic
    # for visualized palette masks.  Black remains Gaga's background id 0.
    return rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]


class GagaObservationAdapter:
    """Read Gaga masks plus optional confidence and boundary maps.

    Gaga's association stage writes ``<image_stem>.png`` and ``info.json`` in
    ``sam_mask`` (or another ``*_mask`` directory).  This adapter also accepts
    ``.npy/.npz/.pt`` observations to avoid format conversion in larger
    preprocessing pipelines.
    """

    _EXTENSIONS = (".png", ".tif", ".tiff", ".npy", ".npz", ".pt", ".pth")

    def __init__(
        self,
        root: str | Path,
        semantic_path: str | Path = "sam_mask",
        confidence_path: str | Path | None = None,
        boundary_path: str | Path | None = None,
        ignore_label: int = -1,
        background_label: int = 0,
        boundary_width: int = 1,
        info_file: str = "info.json",
        background_confidence: float = 0.0,
        compact_labels: bool = True,
    ) -> None:
        self.root = Path(root)
        self.semantic_dir = self._resolve(semantic_path)
        self.confidence_dir = self._resolve(confidence_path)
        self.boundary_dir = self._resolve(boundary_path)
        self.ignore_label = int(ignore_label)
        self.background_label = int(background_label)
        self.boundary_width = max(int(boundary_width), 1)
        self.background_confidence = float(background_confidence)
        self.compact_labels = bool(compact_labels)
        self.raw_to_compact: dict[int, int] = {self.background_label: 0}
        self.compact_to_raw: dict[int, int] = {0: self.background_label}
        self.info: dict[str, Any] = {}
        if self.semantic_dir is not None:
            path = self.semantic_dir / info_file
            if path.is_file():
                with open(path, "r", encoding="utf8") as stream:
                    self.info = json.load(stream)

    def _resolve(self, path: str | Path | None) -> Path | None:
        if path is None or str(path) == "":
            return None
        result = Path(path)
        return result if result.is_absolute() else self.root / result

    @property
    def num_instances(self) -> int | None:
        value = self.info.get("num_mask", self.info.get("num_instances"))
        return None if value is None else int(value)

    @property
    def num_classes(self) -> int:
        expected = self.num_instances
        if expected is not None:
            return max(expected + 1, len(self.compact_to_raw))
        return max(len(self.compact_to_raw), 1)

    def _compact(self, ids: Tensor) -> Tensor:
        if not self.compact_labels:
            return ids
        unique = sorted(int(value) for value in torch.unique(ids).tolist() if value != self.ignore_label)
        expected = self.num_instances
        if expected is not None and unique and min(unique) >= 0 and max(unique) <= expected:
            # Gaga's associated grayscale masks are already globally compact.
            for value in unique:
                self.raw_to_compact[value] = value
                self.compact_to_raw[value] = value
            return ids
        for raw in unique:
            if raw not in self.raw_to_compact:
                compact = len(self.raw_to_compact)
                self.raw_to_compact[raw] = compact
                self.compact_to_raw[compact] = raw
        result = torch.full_like(ids, self.ignore_label)
        for raw in unique:
            result[ids == raw] = self.raw_to_compact[raw]
        return result

    def export_label_mapping(self) -> dict[str, dict[int, int]]:
        return {
            "raw_to_compact": dict(self.raw_to_compact),
            "compact_to_raw": dict(self.compact_to_raw),
        }

    def _find(self, directory: Path | None, image_name: str) -> Path | None:
        if directory is None:
            return None
        name = Path(image_name).name
        candidates = [directory / name]
        stem = Path(name).stem
        candidates.extend(directory / f"{stem}{extension}" for extension in self._EXTENSIONS)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _load(path: Path, keys: Iterable[str]) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix in (".png", ".tif", ".tiff"):
            return np.asarray(Image.open(path))
        if suffix == ".npy":
            return np.load(path, allow_pickle=False)
        if suffix == ".npz":
            archive = np.load(path, allow_pickle=False)
            for key in keys:
                if key in archive:
                    return archive[key]
            if len(archive.files) == 1:
                return archive[archive.files[0]]
            raise KeyError(f"none of {tuple(keys)} found in {path}")
        if suffix in (".pt", ".pth"):
            value = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(value, dict):
                for key in keys:
                    if key in value:
                        value = value[key]
                        break
            if isinstance(value, Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)
        raise ValueError(f"unsupported observation format: {path}")

    @staticmethod
    def _float_map(array: np.ndarray) -> Tensor:
        array = np.asarray(array)
        if array.ndim == 3:
            array = array[..., 0]
        tensor = torch.as_tensor(array.copy() if not array.flags.writeable else array).float()
        if tensor.numel() and tensor.max() > 1.0:
            if np.issubdtype(array.dtype, np.integer):
                scale = float(np.iinfo(array.dtype).max)
            else:
                scale = float(tensor.max().item())
            tensor = tensor / max(scale, 1.0)
        return tensor.clamp(0.0, 1.0)

    def load(
        self,
        image_name: str,
        target_size: tuple[int, int] | None = None,
        required: bool = False,
    ) -> SemanticObservation | None:
        """Load one observation; ``target_size`` is ``(height, width)``."""

        mask_path = self._find(self.semantic_dir, image_name)
        if mask_path is None:
            if required:
                raise FileNotFoundError(f"no Gaga mask for {image_name!r} in {self.semantic_dir}")
            if target_size is None:
                return None
            ids = torch.full(target_size, self.ignore_label, dtype=torch.long)
            return SemanticObservation(ids, torch.zeros(target_size), torch.zeros(target_size))

        mask_array = self._load(mask_path, ("ids", "labels", "mask", "objects"))
        ids = torch.from_numpy(_decode_color_ids(mask_array).copy()).long()
        ids = self._compact(ids)
        confidence_path = self._find(self.confidence_dir, image_name)
        if confidence_path is None:
            confidence = (ids != self.ignore_label).float()
            confidence[ids == 0] = self.background_confidence
        else:
            confidence = self._float_map(
                self._load(confidence_path, ("confidence", "score", "probability"))
            )
            confidence[ids == 0] *= self.background_confidence
        boundary_path = self._find(self.boundary_dir, image_name)
        if boundary_path is None:
            boundary = _boundary(ids, self.ignore_label)
            if self.boundary_width > 1:
                kernel = 2 * self.boundary_width - 1
                boundary = F.max_pool2d(boundary[None, None], kernel, stride=1, padding=kernel // 2)[0, 0]
        else:
            boundary = self._float_map(
                self._load(boundary_path, ("boundary", "edge", "seam"))
            )

        if confidence.shape != ids.shape or boundary.shape != ids.shape:
            raise ValueError(
                f"unaligned Gaga observation {image_name!r}: ids={tuple(ids.shape)}, "
                f"confidence={tuple(confidence.shape)}, boundary={tuple(boundary.shape)}"
            )
        if target_size is not None and tuple(ids.shape) != tuple(target_size):
            ids = F.interpolate(ids[None, None].float(), target_size, mode="nearest")[0, 0].long()
            confidence = F.interpolate(
                confidence[None, None], target_size, mode="bilinear", align_corners=False
            )[0, 0]
            boundary = F.interpolate(
                boundary[None, None], target_size, mode="bilinear", align_corners=False
            )[0, 0]
        confidence = confidence.clamp(0.0, 1.0) * (ids != self.ignore_label).float()
        return SemanticObservation(ids, confidence, boundary.clamp(0.0, 1.0))

    __call__ = load


__all__ = ["GagaObservationAdapter", "SemanticObservation"]
