"""Semantic-aware surface extraction for Semantic Gaussian Wrapping."""

from .bounds import (
    MeshSupportPolicy,
    gaussian_support_bounds,
    trusted_gaussian_support_bounds,
)

from .extractors import (
    MissingOptionalBackend,
    delaunay_marching_tetra,
    marching_cubes_block,
    marching_cubes_blocks,
    marching_tetrahedra,
    merge_meshes,
)
from .field import (
    FIELD_KEYS,
    FieldContractError,
    SurfaceFieldAdapter,
    as_field_adapter,
)
from .extraction_context import (
    MeshExtractionContext,
    checkpoint_iterations,
    resolve_checkpoint_iteration,
)
from .cameras import MeshCamera
from .io import export_mesh, load_mesh, load_points, write_obj, write_ply
from .metrics import (
    MeshMetrics,
    accuracy,
    chamfer_distance,
    completeness,
    compute_mesh_metrics,
    f_score,
    fscore,
    mesh_metrics,
    nearest_distances,
    precision_recall_fscore,
    sample_mesh_surface,
)
from .pipeline import (
    MeshExtractionConfig,
    RegionAwareSemanticMeshExtractor,
    SemanticMeshExtractor,
)
from .gaussian_pivots import (
    GaussianAdaptivePivotBuilder,
    GaussianPivotConfig,
    GaussianPivotSet,
)
from .opacity_field import (
    OpacityFieldConfig,
    OpacityFieldSamples,
    RefinedFieldRoots,
    RendererOpacityField,
)
from .postprocess import (
    postprocess_mesh,
    recompute_vertex_normals,
    remove_small_components,
    seam_aware_vertex_clustering,
    simplify_to_face_budget,
)
from .sampling import (
    AdaptiveOctreeSampler,
    AdaptiveSamplingConfig,
    BlockedGridSampler,
    Bounds,
    GridBlock,
    OctreeLeaf,
    RefinementDecision,
    refinement_decision,
)
from .topology import (
    ContactGraph,
    compatible_pairs,
    connected_face_components,
    face_compatibility_mask,
    filter_semantic_topology,
    seam_vertices,
    semantic_edge_compatibility,
)
from .types import (
    RegionAwareMesh,
    RegionOwnershipSamples,
    SurfaceSamples,
    TriangleMesh,
)
from .region_atlas import (
    GaussianEvidence,
    RegionAtlas,
    RegionAtlasBuilder,
    RegionAtlasConfig,
    RegionChart,
    RegionMembership,
)
from .region_tetrahedral import (
    ChartSurface,
    RefinedRoots,
    RegionTetrahedralConfig,
    SharedTopology,
    delaunay_chart,
    filter_invalid_root_faces,
    merge_chart_surfaces,
    refine_shared_roots,
)
from .region_wrapping import (
    ALGORITHM,
    SCHEMA_VERSION,
    RegionConditionedGaussianWrappingExtractor,
    RegionGaussianWrappingConfig,
)
from .training_field_extraction import (
    ALGORITHM as TRAINING_FIELD_ALGORITHM,
    SCHEMA_VERSION as TRAINING_FIELD_SCHEMA_VERSION,
    SparseBlockLayout,
    TrainingFieldMeshConfig,
    TrainingFieldMeshExtractor,
)
from .training_field_gaussian_wrapping import (
    ALGORITHM as TRAINING_FIELD_GW_ALGORITHM,
    SCHEMA_VERSION as TRAINING_FIELD_GW_SCHEMA_VERSION,
    TrainingFieldGaussianWrappingConfig,
    TrainingFieldGaussianWrappingExtractor,
)
from .multiview_gaussian_extraction import (
    ALGORITHM as MULTIVIEW_GAUSSIAN_ALGORITHM,
    SCHEMA_VERSION as MULTIVIEW_GAUSSIAN_SCHEMA_VERSION,
    MultiviewGaussianMeshConfig,
    MultiviewGaussianMeshExtractor,
)

__all__ = [name for name in globals() if not name.startswith("_")]
