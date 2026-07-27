"""Static lifecycle contract tests that do not require a CUDA device."""

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MODEL = PROJECT / "gaussian_wrapping" / "scene" / "gaussian_model.py"


def test_semantic_field_covers_ply_optimizer_and_densification():
    source = MODEL.read_text(encoding="utf8")
    required = (
        "obj_dc_",
        '"semantic_features"',
        "get_semantic_features",
        "initialize_semantic_features",
        "new_semantic_features",
        'optimizable_tensors["semantic_features"]',
    )
    for token in required:
        assert token in source


def test_original_gaussian_wrapping_fields_are_preserved():
    source = MODEL.read_text(encoding="utf8")
    for token in (
        "filter_3D",
        "gaussian_features_",
        "base_occupancy_",
        "occupancy_shift_",
    ):
        assert token in source
