"""Correspondence backends and geometry used by EDGS initialization."""

from .contracts import (
    DenseMultiViewCorrespondence,
    DirectedPair,
    ImageSamplingGeometry,
    MVRoMaInitializationResult,
    MultiViewTracks,
    PairField,
    PairQuality,
    normalized_image_grid,
    normalized_to_pixel,
    pixel_to_normalized,
)
from .geometry import TriangulationResult, project_row_vector, weighted_multiview_dlt
from .mvroma_backend import init_gaussians_with_mvroma
from .mvroma_runtime import MVRoMaRuntime, MVRoMaSettings

__all__ = [
    "DenseMultiViewCorrespondence",
    "DirectedPair",
    "ImageSamplingGeometry",
    "MVRoMaInitializationResult",
    "MVRoMaRuntime",
    "MVRoMaSettings",
    "MultiViewTracks",
    "PairField",
    "PairQuality",
    "TriangulationResult",
    "init_gaussians_with_mvroma",
    "normalized_image_grid",
    "normalized_to_pixel",
    "pixel_to_normalized",
    "project_row_vector",
    "weighted_multiview_dlt",
]
