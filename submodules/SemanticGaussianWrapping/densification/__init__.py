"""Semantic-aware topology control for Gaussian primitives."""

from .controller import DensityController, DensityDecision, DensityReport, TopologyBudget

__all__ = ["DensityController", "DensityDecision", "DensityReport", "TopologyBudget"]
