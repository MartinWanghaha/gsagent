"""Semantic learning and the Semantic Prior Field for oriented Gaussians."""

from .checkpoint import load_semantic_checkpoint, save_semantic_checkpoint
from .head import SemanticHead
from .losses import semantic_cross_entropy, spatial_consistency_loss
from .observations import GagaObservationStore, SemanticObservation
from .palette import gaga_palette, semantic_palette
from .prior_field import (
    BoundaryWeightCache,
    PriorInstance,
    SemanticPriorField,
    compute_boundary_weight_map,
    head_logits_per_gaussian,
)

__all__ = [
    "BoundaryWeightCache",
    "GagaObservationStore",
    "PriorInstance",
    "SemanticHead",
    "SemanticObservation",
    "SemanticPriorField",
    "compute_boundary_weight_map",
    "gaga_palette",
    "head_logits_per_gaussian",
    "gaga_palette",
    "semantic_palette",
    "load_semantic_checkpoint",
    "save_semantic_checkpoint",
    "semantic_cross_entropy",
    "spatial_consistency_loss",
]
