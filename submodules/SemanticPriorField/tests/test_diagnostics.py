import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "semantic_prior_field"


@pytest.fixture(autouse=True)
def package_path(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE))


def _make_diagnostics(tmp_path):
    from utils.diagnostics import TrainingDiagnostics

    return TrainingDiagnostics(
        str(tmp_path), scalar_interval=10, image_interval=100, snapshot_interval=200
    )


def test_scalar_and_event_streams(tmp_path):
    diagnostics = _make_diagnostics(tmp_path)
    assert diagnostics.wants_scalars(10)
    assert not diagnostics.wants_scalars(11)

    diagnostics.log_scalars(10, {"loss": torch.tensor(0.5), "n_gaussians": 1000})
    diagnostics.log_event(100, "densify_radegs", cloned=5, split=3, pruned=2)

    scalars = [
        json.loads(line)
        for line in (tmp_path / "diagnostics" / "scalars.jsonl").read_text().splitlines()
    ]
    assert scalars == [{"iteration": 10, "loss": 0.5, "n_gaussians": 1000}]

    events = [
        json.loads(line)
        for line in (tmp_path / "diagnostics" / "events.jsonl").read_text().splitlines()
    ]
    assert events[0]["kind"] == "densify_radegs"
    assert events[0]["cloned"] == 5


def test_dump_training_view_writes_images(tmp_path):
    from semantic.head import SemanticHead
    from semantic.observations import SemanticObservation

    diagnostics = _make_diagnostics(tmp_path)
    height, width = 12, 16
    render_pkg = {
        "render": torch.rand(3, height, width),
        "median_depth": torch.rand(1, height, width) + 0.5,
        "normal": torch.nn.functional.normalize(torch.randn(3, height, width), dim=0),
        "semantic_features": torch.randn(16, height, width),
    }
    labels = torch.randint(0, 4, (height, width))
    observation = SemanticObservation(
        labels=labels,
        valid=torch.ones(height, width, dtype=torch.bool),
        confidence=torch.ones(height, width),
    )
    viewpoint_cam = SimpleNamespace(
        image_name="frame_0001",
        image_height=height,
        image_width=width,
        # depth_to_normal needs intrinsics; failure there must not break dumping
    )

    diagnostics.dump_training_view(
        iteration=100,
        viewpoint_cam=viewpoint_cam,
        render_pkg=render_pkg,
        gt_image=torch.rand(3, height, width),
        semantic_head=SemanticHead(16, 4),
        observation=observation,
        ignore_label=-1,
        boundary_weight_map=torch.ones(height, width) * 0.5,
        num_classes=4,
    )

    # The incomplete camera makes the depth-normal error map fail; the
    # decorator isolates that, so either the folder has the core images or
    # (worst case) nothing crashed. Rebuild with a camera-free package to
    # assert the guaranteed subset.
    render_pkg.pop("median_depth")
    render_pkg.pop("normal")
    diagnostics.dump_training_view(
        iteration=200,
        viewpoint_cam=viewpoint_cam,
        render_pkg=render_pkg,
        gt_image=torch.rand(3, height, width),
        semantic_head=SemanticHead(16, 4),
        observation=observation,
        ignore_label=-1,
        boundary_weight_map=torch.ones(height, width) * 0.5,
        num_classes=4,
    )
    image_dir = tmp_path / "diagnostics" / "images" / "iter_000200"
    written = {path.name for path in image_dir.glob("*.png")}
    assert {
        "render.png",
        "gt.png",
        "semantic_pca.png",
        "semantic_pred.png",
        "semantic_error.png",
        "semantic_gt.png",
        "semantic_confidence.png",
        "boundary_weight.png",
    } <= written
    meta = json.loads((image_dir / "meta.json").read_text())
    assert meta["view"] == "frame_0001"


def test_prior_field_dump_once_per_refresh(tmp_path):
    from semantic.head import SemanticHead
    from semantic.prior_field import SemanticPriorField

    diagnostics = _make_diagnostics(tmp_path)
    head = SemanticHead(16, 4)
    with torch.no_grad():
        head.classifier.weight.zero_()
        head.classifier.bias.zero_()
        for class_id in range(4):
            head.classifier.weight[class_id, class_id, 0, 0] = 10.0
    labels = torch.full((256,), 1, dtype=torch.long)
    features = torch.zeros(256, 16)
    features[torch.arange(256), labels] = 1.0
    gaussians = SimpleNamespace(
        get_semantic_features=features,
        get_xyz=torch.randn(256, 3),
        get_scaling=torch.rand(256, 3) * 0.05,
    )
    field = SemanticPriorField(
        {"refresh_interval": 500, "min_instance_gaussians": 64}
    )
    field.refresh(gaussians, head, iteration=7000)

    diagnostics.maybe_dump_prior_field(7000, field)
    diagnostics.maybe_dump_prior_field(7000, field)  # deduplicated
    diagnostics.maybe_dump_prior_field(7001, field)  # not a refresh iteration

    dumps = list((tmp_path / "diagnostics" / "prior_field").glob("*.json"))
    assert len(dumps) == 1
    record = json.loads(dumps[0].read_text())
    assert record["iteration"] == 7000
    assert record["labelled_fraction"] > 0.9
    assert record["instances"][0]["label"] == 1
    assert "prior_type_counts" in record


def test_snapshot_contains_prior_field_state(tmp_path):
    from semantic.head import SemanticHead
    from semantic.prior_field import SemanticPriorField

    diagnostics = _make_diagnostics(tmp_path)
    n_points = 128
    labels = torch.full((n_points,), 1, dtype=torch.long)
    features = torch.zeros(n_points, 16)
    features[torch.arange(n_points), labels] = 1.0
    head = SemanticHead(16, 4)
    with torch.no_grad():
        head.classifier.weight.zero_()
        head.classifier.bias.zero_()
        for class_id in range(4):
            head.classifier.weight[class_id, class_id, 0, 0] = 10.0
    gaussians = SimpleNamespace(
        _xyz=torch.randn(n_points, 3),
        get_opacity=torch.rand(n_points, 1),
        get_scaling=torch.rand(n_points, 3) * 0.05,
        get_semantic_features=features,
        get_xyz=torch.randn(n_points, 3),
        _features_rest=torch.randn(n_points, 15, 3) * 0.01,
        use_gaussian_features=False,
    )
    field = SemanticPriorField({"min_instance_gaussians": 64})
    field.refresh(gaussians, head, iteration=0)

    assert diagnostics.wants_snapshot(200)
    diagnostics.dump_snapshot(200, gaussians, field)

    snapshot = np.load(tmp_path / "diagnostics" / "snapshots" / "iter_000200.npz")
    assert snapshot["xyz"].shape == (n_points, 3)
    assert snapshot["sh_rest_energy"].shape == (n_points,)
    assert snapshot["labels"].shape == (n_points,)
    assert snapshot["prior_type"].shape == (n_points,)
    assert snapshot["densify_multiplier"].shape == (n_points,)
