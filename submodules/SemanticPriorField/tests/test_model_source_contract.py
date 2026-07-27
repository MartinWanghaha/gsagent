"""Static lifecycle contract tests that do not require a CUDA device."""

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MODEL = PROJECT / "semantic_prior_field" / "scene" / "gaussian_model.py"
TRAIN = PROJECT / "semantic_prior_field" / "train.py"
NORMAL_FIELD = (
    PROJECT
    / "semantic_prior_field"
    / "regularization"
    / "regularizer"
    / "normal_field.py"
)


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


def test_original_semantic_prior_field_fields_are_preserved():
    source = MODEL.read_text(encoding="utf8")
    for token in (
        "filter_3D",
        "gaussian_features_",
        "base_occupancy_",
        "occupancy_shift_",
    ):
        assert token in source


def test_normal_features_are_initialized_before_spf_orientation_graph():
    train_source = TRAIN.read_text(encoding="utf8")
    loop_source = train_source[train_source.index("for iteration in range(") :]

    initialization = loop_source.index("gaussians.reset_normal_features()")
    semantic_prior = loop_source.index("compute_semantic_prior_regularization(")
    normal_field = loop_source.index("compute_normal_field_regularization(")

    assert initialization < semantic_prior < normal_field
    assert "gaussians.reset_normal_features()" not in NORMAL_FIELD.read_text(
        encoding="utf8"
    )
