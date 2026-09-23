"""Lossless temporary storage for merged MV-RoMa directed pair fields."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .config import PairStoreSettings
from .contracts import DirectedPair, ImageSamplingGeometry, PairField


@dataclass
class _StoredPair:
    source_coordinates: torch.Tensor
    target_coordinates: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    source_geometry: ImageSamplingGeometry


class PairFieldStore:
    """Merge pair fields in RAM or a temporary float32 mmap spool.

    ``mmap`` changes only where CPU tensors are backed; both modes execute the
    same comparisons and retain float32 coordinates/confidence.  Files live in
    a ``TemporaryDirectory`` and are never published as model artifacts.
    """

    def __init__(
        self,
        settings: PairStoreSettings,
        *,
        expected_pair_count: int,
    ) -> None:
        if expected_pair_count <= 0:
            raise ValueError("expected_pair_count must be positive")
        self.settings = settings
        self.expected_pair_count = int(expected_pair_count)
        self._mode: str | None = None
        self._pairs: dict[DirectedPair, _StoredPair] = {}
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> "PairFieldStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def mode(self) -> str | None:
        return self._mode

    def _choose_mode(self, height: int, width: int) -> str:
        if self.settings.mode != "auto":
            return self.settings.mode
        # source xy + target xy + confidence + valid byte
        bytes_per_pixel = 4 * 4 + 4 + 1
        estimated = self.expected_pair_count * height * width * bytes_per_pixel
        return (
            "memory"
            if estimated <= self.settings.max_ram_gb * (1024**3)
            else "mmap"
        )

    def _mmap_tensor(
        self,
        key: DirectedPair,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> torch.Tensor:
        if self._temporary is None:
            parent = self.settings.temp_dir
            if parent is not None:
                parent.mkdir(parents=True, exist_ok=True)
            self._temporary = tempfile.TemporaryDirectory(
                prefix="edgs-mvroma-pairs-",
                dir=None if parent is None else str(parent),
            )
        root = Path(self._temporary.name)
        path = root / f"{key.source_index:06d}_{key.target_index:06d}_{name}.bin"
        array = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        return torch.from_numpy(array)

    def _allocate(
        self,
        key: DirectedPair,
        source_coordinates: torch.Tensor,
        source_geometry: ImageSamplingGeometry,
    ) -> _StoredPair:
        height, width = source_coordinates.shape[-2:]
        if self._mode is None:
            self._mode = self._choose_mode(height, width)
        if self._mode == "memory":
            source = source_coordinates.clone()
            target = torch.zeros_like(source)
            confidence = torch.full((height, width), float("-inf"))
            valid = torch.zeros((height, width), dtype=torch.bool)
        else:
            source = self._mmap_tensor(
                key, "source", (2, height, width), np.dtype("float32")
            )
            target = self._mmap_tensor(
                key, "target", (2, height, width), np.dtype("float32")
            )
            confidence = self._mmap_tensor(
                key, "confidence", (height, width), np.dtype("float32")
            )
            valid = self._mmap_tensor(
                key, "valid", (height, width), np.dtype("bool")
            )
            source.copy_(source_coordinates)
            target.zero_()
            confidence.fill_(float("-inf"))
            valid.zero_()
        stored = _StoredPair(
            source_coordinates=source,
            target_coordinates=target,
            confidence=confidence,
            valid=valid,
            source_geometry=source_geometry,
        )
        self._pairs[key] = stored
        return stored

    def merge(
        self,
        key: DirectedPair,
        *,
        source_coordinates: torch.Tensor,
        target_coordinates: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        source_geometry: ImageSamplingGeometry,
    ) -> None:
        """Apply paper Eq.11: keep the highest-confidence valid prediction."""

        source = source_coordinates.detach().to(device="cpu", dtype=torch.float32)
        target = target_coordinates.detach().to(device="cpu", dtype=torch.float32)
        probability = confidence.detach().to(device="cpu", dtype=torch.float32)
        candidate_valid = valid.detach().to(device="cpu", dtype=torch.bool)
        if source.ndim != 3 or source.shape[0] != 2 or target.shape != source.shape:
            raise ValueError("pair coordinates must have shape [2,H,W]")
        if probability.shape != source.shape[-2:] or candidate_valid.shape != probability.shape:
            raise ValueError("pair confidence/valid must have shape [H,W]")

        stored = self._pairs.get(key)
        if stored is None:
            stored = self._allocate(key, source, source_geometry)
        elif (
            stored.source_coordinates.shape != source.shape
            or stored.source_geometry != source_geometry
        ):
            raise ValueError(f"inconsistent source grid for directed pair {key}")

        replace = candidate_valid & (
            ~stored.valid | (probability > stored.confidence)
        )
        stored.target_coordinates[:, replace] = target[:, replace]
        stored.confidence[replace] = probability[replace]
        stored.valid[replace] = True

    def read(self, key: DirectedPair) -> PairField:
        stored = self._pairs[key]
        return PairField(
            source_coordinates=stored.source_coordinates,
            target_coordinates=stored.target_coordinates,
            confidence=stored.confidence,
            valid=stored.valid,
            source_geometry=stored.source_geometry,
        )

    def apply_valid_mask(self, key: DirectedPair, mask: torch.Tensor) -> None:
        stored = self._pairs[key]
        mask_cpu = mask.detach().to(device="cpu", dtype=torch.bool)
        if mask_cpu.shape != stored.valid.shape:
            raise ValueError("valid mask shape differs from stored pair field")
        stored.valid.logical_and_(mask_cpu)

    def __contains__(self, key: DirectedPair) -> bool:
        return key in self._pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def keys(self) -> Iterator[DirectedPair]:
        return iter(self._pairs)

    def close(self) -> None:
        self._pairs.clear()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
