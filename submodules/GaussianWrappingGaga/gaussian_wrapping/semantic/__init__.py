"""Gaga-compatible semantic learning for Gaussian Wrapping."""

from .checkpoint import load_semantic_checkpoint, save_semantic_checkpoint
from .head import SemanticHead
from .losses import semantic_cross_entropy, spatial_consistency_loss
from .observations import GagaObservationStore, SemanticObservation
from .palette import gaga_palette, semantic_palette

__all__ = [
    "GagaObservationStore",
    "SemanticHead",
    "SemanticObservation",
    "gaga_palette",
    "semantic_palette",
    "load_semantic_checkpoint",
    "save_semantic_checkpoint",
    "semantic_cross_entropy",
    "spatial_consistency_loss",
]
