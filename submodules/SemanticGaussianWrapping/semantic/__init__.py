"""Semantic evidence, geometry policies, and the unified surface field."""

from .gaga_adapter import GagaObservationAdapter, SemanticObservation
from .geometry_policy import (
    EXPERT_NAMES,
    GeometryEvidenceProjector,
    GeometryPolicyOutput,
    SoftGeometryPolicyBank,
)
from .neighbor_index import GaussianNeighborIndex, GaussianSupportAttributes
from .region_membership import SparseRegionMembership


def __getattr__(name):
    # Lazy import avoids a cycle while scene.gaussian_model is importing the
    # policy definitions from this package.
    if name in {
        "GeometryQueryContext",
        "PartitionedSurfaceQueryResult",
        "PointRegionSurfaceQueryResult",
        "RegionOwnershipResult",
        "SemanticSurfaceField",
        "SurfaceQueryContext",
        "SurfaceQueryResult",
    }:
        from .surface_field import (
            GeometryQueryContext,
            PartitionedSurfaceQueryResult,
            PointRegionSurfaceQueryResult,
            RegionOwnershipResult,
            SemanticSurfaceField,
            SurfaceQueryContext,
            SurfaceQueryResult,
        )

        return {
            "GeometryQueryContext": GeometryQueryContext,
            "PartitionedSurfaceQueryResult": PartitionedSurfaceQueryResult,
            "PointRegionSurfaceQueryResult": PointRegionSurfaceQueryResult,
            "RegionOwnershipResult": RegionOwnershipResult,
            "SemanticSurfaceField": SemanticSurfaceField,
            "SurfaceQueryContext": SurfaceQueryContext,
            "SurfaceQueryResult": SurfaceQueryResult,
        }[name]
    raise AttributeError(name)


__all__ = [
    "EXPERT_NAMES",
    "GagaObservationAdapter",
    "GeometryEvidenceProjector",
    "GeometryPolicyOutput",
    "GaussianNeighborIndex",
    "GaussianSupportAttributes",
    "GeometryQueryContext",
    "SemanticObservation",
    "SparseRegionMembership",
    "PartitionedSurfaceQueryResult",
    "PointRegionSurfaceQueryResult",
    "RegionOwnershipResult",
    "SemanticSurfaceField",
    "SoftGeometryPolicyBank",
    "SurfaceQueryContext",
    "SurfaceQueryResult",
]
