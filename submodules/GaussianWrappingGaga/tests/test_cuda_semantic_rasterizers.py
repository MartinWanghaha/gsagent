"""GPU integration tests for both native semantic rasterizers."""

from pathlib import Path
import sys

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT / "submodules" / "diff-gaussian-rasterization-semantic"),
    str(PROJECT / "submodules" / "diff-gaussian-rasterization_ours-semantic"),
]


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device is required for rasterizer execution",
)


def test_semantic_extensions_import():
    import diff_gaussian_rasterization_gw_ours_semantic
    import diff_gaussian_rasterization_gw_semantic

    assert diff_gaussian_rasterization_gw_semantic.GaussianRasterizer
    assert diff_gaussian_rasterization_gw_ours_semantic.GaussianRasterizer


def test_semantic_feature_gradient_isolated_from_geometry():
    """The auxiliary backward is intentionally embedding-only.

    A full camera-level smoke test is executed by ``tests/run_gpu_smoke.py``.
    This invariant is also enforced structurally: the semantic CUDA backward
    receives only ranges, means2D/conics as const inputs and exposes only
    ``grad_semantic_features``.
    """

    source = (
        PROJECT
        / "submodules"
        / "diff-gaussian-rasterization-semantic"
        / "cuda_rasterizer"
        / "semantic.h"
    ).read_text(encoding="utf8")
    assert "grad_semantic_features" in source
    assert "dL_dopacity" not in source
    assert "dL_dmeans" not in source
