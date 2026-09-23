"""Typed tensor contracts for dense and sampled multi-view correspondences.

The MV-RoMa flow is an *absolute* target-image coordinate in ``[-1, 1]``.
Both source and target coordinates in this module use PyTorch's
``align_corners=False`` convention.  Keeping that convention in the type
contract prevents the half-pixel ambiguity that otherwise appears when a
model-resolution flow is consumed by EDGS' projection matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch


@dataclass(frozen=True, order=True)
class DirectedPair:
    """A directed correspondence field key ``source -> target``."""

    source_index: int
    target_index: int

    def __post_init__(self) -> None:
        if self.source_index < 0 or self.target_index < 0:
            raise ValueError("camera indices must be non-negative")
        if self.source_index == self.target_index:
            raise ValueError("a directed pair requires two different cameras")

    def reverse(self) -> "DirectedPair":
        return DirectedPair(self.target_index, self.source_index)


@dataclass(frozen=True)
class ImageSamplingGeometry:
    """Conversion between semantic image coordinates and a stored model grid.

    MV-RoMa may first pad an image to a square.  Its output warp is converted
    back to original-image normalized coordinates, while ``grid_sample`` still
    indexes the square storage grid.  Keeping the affine conversion explicit
    makes reciprocal cycle checks correct for both padded and unpadded inputs.
    """

    original_height: int
    original_width: int
    storage_height: int
    storage_width: int
    square_size: int | None = None
    padding_left: int = 0
    padding_top: int = 0

    def __post_init__(self) -> None:
        dimensions = (
            self.original_height,
            self.original_width,
            self.storage_height,
            self.storage_width,
        )
        if min(dimensions) <= 0:
            raise ValueError("image and storage dimensions must be positive")
        if self.square_size is not None:
            if self.square_size < max(self.original_height, self.original_width):
                raise ValueError("square_size cannot be smaller than the image")
            if min(self.padding_left, self.padding_top) < 0:
                raise ValueError("image padding must be non-negative")

    @classmethod
    def identity(cls, height: int, width: int) -> "ImageSamplingGeometry":
        return cls(height, width, height, width)

    def semantic_to_storage_grid(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Convert original-image normalized x/y to ``grid_sample`` x/y."""

        if coordinates.shape[-1] != 2:
            raise ValueError("coordinates must end in x/y")
        if self.square_size is None:
            return coordinates
        result = coordinates.clone()
        x_original = (coordinates[..., 0] + 1.0) * (self.original_width / 2.0) - 0.5
        y_original = (coordinates[..., 1] + 1.0) * (self.original_height / 2.0) - 0.5
        result[..., 0] = (
            (x_original + self.padding_left + 0.5) * (2.0 / self.square_size) - 1.0
        )
        result[..., 1] = (
            (y_original + self.padding_top + 0.5) * (2.0 / self.square_size) - 1.0
        )
        return result

    def normalized_delta_to_pixels(self, delta: torch.Tensor) -> torch.Tensor:
        """Convert semantic normalized deltas to original-image pixels."""

        if delta.shape[-1] != 2:
            raise ValueError("delta must end in x/y")
        result = delta.clone()
        result[..., 0] *= self.original_width / 2.0
        result[..., 1] *= self.original_height / 2.0
        return result

    def normalized_delta_to_storage_pixels(
        self, delta: torch.Tensor
    ) -> torch.Tensor:
        """Convert semantic deltas to the MV-RoMa output-grid pixel scale."""

        if delta.shape[-1] != 2:
            raise ValueError("delta must end in x/y")
        result = delta.clone()
        if self.square_size is None:
            result[..., 0] *= self.storage_width / 2.0
            result[..., 1] *= self.storage_height / 2.0
        else:
            result[..., 0] *= (
                self.original_width * self.storage_width / (2.0 * self.square_size)
            )
            result[..., 1] *= (
                self.original_height * self.storage_height / (2.0 * self.square_size)
            )
        return result


@dataclass(frozen=True)
class PairField:
    """Merged dense field for one ordered camera pair, stored on CPU."""

    source_coordinates: torch.Tensor
    target_coordinates: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    source_geometry: ImageSamplingGeometry

    def __post_init__(self) -> None:
        if self.source_coordinates.ndim != 3 or self.source_coordinates.shape[0] != 2:
            raise ValueError("source_coordinates must have shape [2,H,W]")
        if self.target_coordinates.shape != self.source_coordinates.shape:
            raise ValueError("source and target coordinates must share [2,H,W]")
        height, width = self.source_coordinates.shape[-2:]
        if self.confidence.shape != (height, width):
            raise ValueError("confidence must have shape [H,W]")
        if self.valid.shape != (height, width) or self.valid.dtype != torch.bool:
            raise ValueError("valid must be boolean with shape [H,W]")


@dataclass(frozen=True)
class PairQuality:
    cycle_inlier_ratio: float
    mean_confidence: float
    valid_pixels: int

    @property
    def score(self) -> float:
        return self.cycle_inlier_ratio * self.mean_confidence


@dataclass(frozen=True)
class MVRoMaInitializationResult:
    """Rich MV result with a legacy three-item iterator for EDGS callers."""

    cameras: list[Any]
    legacy_neighbors: Any
    diagnostics: dict[str, Any]
    overlap_matrix: Any
    pair_quality: Mapping[DirectedPair, PairQuality]
    artifact_id: str | None = None

    def __iter__(self) -> Iterator[Any]:
        yield self.cameras
        yield self.legacy_neighbors
        yield self.diagnostics


def normalized_image_grid(
    height: int,
    width: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a ``[2,H,W]`` x/y grid with ``align_corners=False`` centers."""

    if height <= 0 or width <= 0:
        raise ValueError(f"image dimensions must be positive, got {(height, width)}")
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0)


def normalized_to_pixel(
    coordinates: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Decode x/y coordinates using the ``align_corners=False`` convention."""

    result = coordinates.clone()
    result[..., 0] = (coordinates[..., 0] + 1.0) * (width / 2.0) - 0.5
    result[..., 1] = (coordinates[..., 1] + 1.0) * (height / 2.0) - 0.5
    return result


def pixel_to_normalized(
    coordinates: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Encode x/y pixel centers using the ``align_corners=False`` convention."""

    result = coordinates.clone()
    result[..., 0] = (coordinates[..., 0] + 0.5) * (2.0 / width) - 1.0
    result[..., 1] = (coordinates[..., 1] + 0.5) * (2.0 / height) - 1.0
    return result


@dataclass(frozen=True)
class MultiViewTracks:
    """Sampled tracks ready for multi-view triangulation.

    ``coordinates`` has shape ``[N,V,2]``.  View zero is the source camera and
    the remaining views are targets in the exact order used for inference.
    Coordinates are normalized x/y image coordinates with
    ``align_corners=False`` semantics.
    """

    coordinates: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    sampling_score: torch.Tensor

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 3 or self.coordinates.shape[-1] != 2:
            raise ValueError(
                "coordinates must have shape [N,V,2], got "
                f"{tuple(self.coordinates.shape)}"
            )
        expected = self.coordinates.shape[:2]
        if self.confidence.shape != expected:
            raise ValueError(
                f"confidence must have shape {expected}, got {tuple(self.confidence.shape)}"
            )
        if self.valid.shape != expected or self.valid.dtype != torch.bool:
            raise ValueError("valid must be a bool tensor with shape [N,V]")
        if self.sampling_score.shape != expected[:1]:
            raise ValueError("sampling_score must have shape [N]")


@dataclass(frozen=True)
class DenseMultiViewCorrespondence:
    """One-source/many-target dense MV-RoMa prediction.

    Attributes:
        source_coordinates: ``[2,H,W]`` normalized source coordinate grid.
        target_coordinates: ``[T,2,H,W]`` normalized absolute target coords.
        certainty_logits: ``[T,1,H,W]`` logits produced by MV-RoMa.
        source_valid: ``[H,W]`` mask, primarily used when square padding exists.
        target_valid: ``[T,H,W]`` mask for valid, unpadded target coordinates.
    """

    source_coordinates: torch.Tensor
    target_coordinates: torch.Tensor
    certainty_logits: torch.Tensor
    source_valid: torch.Tensor
    target_valid: torch.Tensor
    source_geometry: ImageSamplingGeometry | None = None
    target_geometries: tuple[ImageSamplingGeometry, ...] | None = None

    def __post_init__(self) -> None:
        if self.source_coordinates.ndim != 3 or self.source_coordinates.shape[0] != 2:
            raise ValueError("source_coordinates must have shape [2,H,W]")
        if self.target_coordinates.ndim != 4 or self.target_coordinates.shape[1] != 2:
            raise ValueError("target_coordinates must have shape [T,2,H,W]")
        targets, _, height, width = self.target_coordinates.shape
        if self.source_coordinates.shape != (2, height, width):
            raise ValueError("source and target correspondence grids must share H,W")
        if self.certainty_logits.shape != (targets, 1, height, width):
            raise ValueError("certainty_logits must have shape [T,1,H,W]")
        if self.source_valid.shape != (height, width) or self.source_valid.dtype != torch.bool:
            raise ValueError("source_valid must be a bool tensor with shape [H,W]")
        if self.target_valid.shape != (targets, height, width):
            raise ValueError("target_valid must have shape [T,H,W]")
        if self.target_valid.dtype != torch.bool:
            raise ValueError("target_valid must be boolean")
        if self.target_geometries is not None and len(self.target_geometries) != targets:
            raise ValueError("target_geometries must contain one item per target")
        tensors = (
            self.source_coordinates,
            self.target_coordinates,
            self.certainty_logits,
        )
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise TypeError("coordinates and certainty logits must be floating point")
        if any(tensor.device != tensors[0].device for tensor in tensors[1:]):
            raise ValueError("dense correspondence tensors must share a device")

    @property
    def target_count(self) -> int:
        return int(self.target_coordinates.shape[0])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return tuple(self.source_coordinates.shape[-2:])

    def sample_tracks(
        self,
        count: int,
        *,
        confidence_threshold: float,
        min_target_views: int,
        generator: torch.Generator | None = None,
    ) -> MultiViewTracks:
        """Sample confidence-weighted source pixels without materializing a cache."""

        if count <= 0:
            raise ValueError("count must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1]")
        if not 1 <= min_target_views <= self.target_count:
            raise ValueError(
                f"min_target_views must be in [1,{self.target_count}], got {min_target_views}"
            )

        probability = self.certainty_logits[:, 0].sigmoid()
        finite = torch.isfinite(self.target_coordinates).all(dim=1) & torch.isfinite(probability)
        in_bounds = (self.target_coordinates.abs() <= 1.0).all(dim=1)
        valid_target = (
            self.target_valid
            & finite
            & in_bounds
            & (probability >= confidence_threshold)
        )
        valid_target &= self.source_valid.unsqueeze(0)

        eligible = valid_target.sum(dim=0) >= min_target_views
        masked_probability = probability.masked_fill(~valid_target, 0.0)
        strongest = masked_probability.topk(min_target_views, dim=0).values
        score = strongest.mean(dim=0).masked_fill(~eligible, 0.0)
        flat_score = score.reshape(-1)
        selectable = int(torch.count_nonzero(flat_score > 0).item())
        sample_count = min(int(count), selectable)

        views = self.target_count + 1
        if sample_count == 0:
            empty_coordinates = self.source_coordinates.new_empty((0, views, 2))
            empty_confidence = self.certainty_logits.new_empty((0, views))
            return MultiViewTracks(
                coordinates=empty_coordinates,
                confidence=empty_confidence,
                valid=torch.empty((0, views), dtype=torch.bool, device=flat_score.device),
                sampling_score=flat_score.new_empty((0,)),
            )

        selected = torch.multinomial(
            flat_score,
            num_samples=sample_count,
            replacement=False,
            generator=generator,
        )
        source = self.source_coordinates.permute(1, 2, 0).reshape(-1, 2)[selected]
        target = (
            self.target_coordinates.permute(2, 3, 0, 1)
            .reshape(-1, self.target_count, 2)[selected]
        )
        target_confidence = probability.permute(1, 2, 0).reshape(-1, self.target_count)[selected]
        sampled_valid = valid_target.permute(1, 2, 0).reshape(-1, self.target_count)[selected]

        source_confidence = torch.ones(
            (sample_count, 1), device=target_confidence.device, dtype=target_confidence.dtype
        )
        source_valid = torch.ones(
            (sample_count, 1), device=sampled_valid.device, dtype=torch.bool
        )
        return MultiViewTracks(
            coordinates=torch.cat((source[:, None], target), dim=1),
            confidence=torch.cat((source_confidence, target_confidence), dim=1),
            valid=torch.cat((source_valid, sampled_valid), dim=1),
            sampling_score=flat_score[selected],
        )
