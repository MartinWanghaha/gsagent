"""Load view-consistent Gaga masks without adding segmentation dependencies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class SemanticObservation:
    labels: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor

    def to(self, device: torch.device | str) -> "SemanticObservation":
        return SemanticObservation(
            self.labels.to(device),
            self.valid.to(device),
            self.confidence.to(device),
        )


class GagaObservationStore:
    """Resolve associated Gaga masks by camera image name.

    The store accepts Gaga's PNG masks and optional ``info.json``. Labels are
    resized with nearest-neighbour interpolation to the final camera size.
    """

    _EXTENSIONS = (".png", ".tif", ".tiff", ".npy")

    def __init__(
        self,
        mask_dir: str | Path,
        *,
        info_file: str = "info.json",
        ignore_label: int = -1,
        background_label: int = 0,
        require_all: bool = True,
    ) -> None:
        self.mask_dir = Path(mask_dir)
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Gaga mask directory does not exist: {self.mask_dir}")
        self.ignore_label = int(ignore_label)
        self.background_label = int(background_label)
        self.require_all = bool(require_all)
        self.info: dict = {}
        info_path = self.mask_dir / info_file
        if info_path.is_file():
            with info_path.open("r", encoding="utf8") as stream:
                self.info = json.load(stream)
        raw_count = self.info.get("num_mask", self.info.get("num_instances"))
        self._declared_num_classes = None if raw_count is None else int(raw_count) + 1
        self.disk_ignore_label = self.info.get("ignore_label")
        self.confidence_dir = self.mask_dir / "confidence"
        self.valid_dir = self.mask_dir / "valid"

    @property
    def num_classes(self) -> int | None:
        return self._declared_num_classes

    def _resolve(self, image_name: str) -> Path | None:
        image_path = Path(image_name)
        candidates = [self.mask_dir / image_path.name]
        candidates.extend(
            self.mask_dir / f"{image_path.stem}{suffix}"
            for suffix in self._EXTENSIONS
        )
        matches = [candidate for candidate in candidates if candidate.is_file()]
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous semantic masks for {image_name!r}: {matches}"
            )
        return matches[0] if matches else None

    @staticmethod
    def _decode(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            array = np.load(path, allow_pickle=False)
        else:
            array = np.asarray(Image.open(path))
        if array.ndim == 2:
            return array.astype(np.int64, copy=False)
        if array.ndim == 3 and array.shape[-1] == 1:
            return array[..., 0].astype(np.int64, copy=False)
        if array.ndim == 3 and array.shape[-1] >= 3:
            rgb = array[..., :3].astype(np.int64)
            return rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
        raise ValueError(f"Unsupported mask shape {array.shape} in {path}")

    def load(
        self,
        image_name: str,
        height: int,
        width: int,
    ) -> SemanticObservation:
        path = self._resolve(image_name)
        if path is None:
            if self.require_all:
                raise FileNotFoundError(
                    f"No Gaga mask for {image_name!r} in {self.mask_dir}"
                )
            labels = torch.full(
                (height, width), self.ignore_label, dtype=torch.long
            )
            valid = torch.zeros((height, width), dtype=torch.bool)
            return SemanticObservation(labels, valid, valid.float())

        labels = torch.from_numpy(self._decode(path).copy()).long()
        if self.disk_ignore_label is not None:
            labels[labels == int(self.disk_ignore_label)] = self.ignore_label
        if tuple(labels.shape) != (height, width):
            labels = F.interpolate(
                labels[None, None].float(),
                size=(height, width),
                mode="nearest",
            )[0, 0].long()
        if labels.min().item() < self.ignore_label:
            raise ValueError(f"Invalid negative label in {path}")
        if self.num_classes is not None:
            bad = (labels != self.ignore_label) & (
                (labels < 0) | (labels >= self.num_classes)
            )
            if bad.any():
                raise ValueError(
                    f"Mask {path} contains label {labels[bad].max().item()} "
                    f"outside [0, {self.num_classes - 1}]"
                )
        valid = labels != self.ignore_label
        valid_path = self.valid_dir / f"{Path(image_name).stem}.png"
        if valid_path.is_file():
            disk_valid = torch.from_numpy(
                np.asarray(Image.open(valid_path)).copy()
            ).bool()
            if tuple(disk_valid.shape) != (height, width):
                disk_valid = F.interpolate(
                    disk_valid[None, None].float(),
                    size=(height, width),
                    mode="nearest",
                )[0, 0].bool()
            valid &= disk_valid
            labels[~valid] = self.ignore_label
        confidence_path = self.confidence_dir / f"{Path(image_name).stem}.png"
        if confidence_path.is_file():
            confidence = torch.from_numpy(
                np.asarray(Image.open(confidence_path)).copy()
            ).float()
            if confidence.max().item() > 1:
                confidence = confidence / 255.0
            if tuple(confidence.shape) != (height, width):
                confidence = F.interpolate(
                    confidence[None, None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            confidence = confidence.clamp(0, 1) * valid.float()
        else:
            confidence = valid.float()
        return SemanticObservation(labels, valid, confidence)

    def validate_cameras(self, cameras) -> int:
        observed_max = 0
        for camera in cameras:
            observation = self.load(
                camera.image_name,
                camera.image_height,
                camera.image_width,
            )
            if observation.valid.any():
                observed_max = max(
                    observed_max,
                    int(observation.labels[observation.valid].max().item()),
                )
        inferred = observed_max + 1
        if self.num_classes is not None and inferred > self.num_classes:
            raise ValueError(
                f"Observed {inferred} classes but info.json declares "
                f"{self.num_classes}"
            )
        return self.num_classes or inferred
