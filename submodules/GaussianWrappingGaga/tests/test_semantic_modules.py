from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "gaussian_wrapping"


@pytest.fixture(autouse=True)
def package_path(monkeypatch):
    monkeypatch.syspath_prepend(str(PACKAGE))


def test_observation_store_uses_nearest_resize(tmp_path):
    from semantic.observations import GagaObservationStore

    Image.fromarray(np.array([[0, 1], [2, 3]], dtype=np.uint8)).save(
        tmp_path / "frame.png"
    )
    (tmp_path / "info.json").write_text('{"num_mask": 3}', encoding="utf8")
    store = GagaObservationStore(tmp_path)
    observation = store.load("frame.jpg", 4, 4)
    assert observation.labels.tolist() == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ]
    assert observation.valid.all()
    assert store.num_classes == 4


def test_observation_store_reads_robust_ignore_and_confidence(tmp_path):
    from semantic.observations import GagaObservationStore

    labels = np.array([[0, 1], [65_535, 2]], dtype=np.uint16)
    Image.fromarray(labels).save(tmp_path / "frame.png")
    (tmp_path / "confidence").mkdir()
    Image.fromarray(
        np.array([[255, 128], [0, 64]], dtype=np.uint8)
    ).save(tmp_path / "confidence" / "frame.png")
    (tmp_path / "valid").mkdir()
    Image.fromarray(
        np.array([[255, 255], [0, 255]], dtype=np.uint8)
    ).save(tmp_path / "valid" / "frame.png")
    (tmp_path / "info.json").write_text(
        '{"num_mask": 2, "ignore_label": 65535}',
        encoding="utf8",
    )
    observation = GagaObservationStore(tmp_path).load("frame.jpg", 2, 2)
    assert observation.labels.tolist() == [[0, 1], [-1, 2]]
    assert observation.valid.tolist() == [[True, True], [False, True]]
    assert observation.confidence[0, 1].item() == pytest.approx(128 / 255)
    assert observation.confidence[1, 0].item() == 0


def test_semantic_head_and_loss_backpropagate():
    from semantic.head import SemanticHead
    from semantic.losses import semantic_cross_entropy

    features = torch.randn(16, 8, 9, requires_grad=True)
    labels = torch.randint(0, 4, (8, 9))
    head = SemanticHead(16, 4)
    logits = head(features)
    loss = semantic_cross_entropy(logits, labels)
    loss.backward()
    assert features.grad is not None
    assert features.grad.abs().sum() > 0
    assert head.classifier.weight.grad is not None


def test_chunked_semantic_classification_matches_dense(tmp_path):
    from semantic.head import SemanticHead
    from semantic_render import classify_features_chunked

    torch.manual_seed(7)
    features = torch.randn(16, 11, 13)
    head = SemanticHead(16, 9)
    expected = head(features).argmax(dim=0).numpy()
    logits_path = tmp_path / "logits.npy"
    actual = classify_features_chunked(
        head,
        features,
        chunk_size=17,
        logits_path=logits_path,
    )
    assert np.array_equal(actual, expected)
    logits = np.load(logits_path)
    assert logits.shape == (9, 11, 13)
    assert np.array_equal(logits.argmax(axis=0), expected)


def test_gaga_palette_keeps_background_black():
    from semantic.palette import gaga_palette

    colors = gaga_palette(8)
    assert colors.shape == (8, 3)
    assert colors.dtype == np.uint8
    assert colors[0].tolist() == [0, 0, 0]
    assert np.unique(colors, axis=0).shape[0] == 8


def test_semantic_runtime_loads_in_place_cuda_extensions():
    from gaussian_renderer.semantic_runtime import load_semantic_rasterizer

    radegs = load_semantic_rasterizer("radegs")
    ours = load_semantic_rasterizer("ours")
    assert radegs.GaussianRasterizer
    assert ours.GaussianRasterizer


def test_spatial_consistency_is_finite_and_sampled():
    from semantic.losses import spatial_consistency_loss

    features = torch.randn(512, 16, requires_grad=True)
    positions = torch.randn(512, 3)
    loss = spatial_consistency_loss(
        features,
        positions,
        sample_size=64,
        neighbors=4,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None


def test_confusion_metrics():
    from semantic.metrics import confusion_matrix, metrics_from_confusion

    target = torch.tensor([[0, 1], [1, 2]])
    prediction = target.clone()
    matrix = confusion_matrix(prediction, target, 3)
    metrics = metrics_from_confusion(matrix)
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)


def test_hungarian_matching_recovers_permuted_labels():
    from semantic.metrics import (
        confusion_matrix,
        hungarian_permutation,
        metrics_from_confusion,
    )

    target = torch.tensor([0, 0, 1, 1, 2, 2])
    prediction = torch.tensor([2, 2, 0, 0, 1, 1])
    matrix = confusion_matrix(prediction, target, 3)
    prediction_to_target = hungarian_permutation(matrix)
    matched = matrix[:, torch.from_numpy(np.argsort(prediction_to_target))]
    assert prediction_to_target.tolist() == [1, 2, 0]
    assert metrics_from_confusion(matched)["miou"] == pytest.approx(1.0)


def test_semantic_mesh_vertex_schema_preserves_fields():
    from semantic_mesh import _vertex_with_semantics

    vertex = np.zeros(
        2,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("quality", "<f4"),
            ("red", "u1"),
        ],
    )
    vertex["quality"] = (0.25, 0.75)
    labels = np.array([1, 2], dtype=np.int64)
    colors = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    result = _vertex_with_semantics(vertex, labels, colors)
    assert result["quality"].tolist() == pytest.approx([0.25, 0.75])
    assert result["semantic_id"].tolist() == [1, 2]
    assert result["red"].tolist() == [10, 40]
    assert result["green"].tolist() == [20, 50]
    assert result["blue"].tolist() == [30, 60]


def test_semantic_mesh_export_end_to_end(tmp_path):
    from plyfile import PlyData, PlyElement
    from semantic.head import SemanticHead
    from semantic_mesh import export_semantic_mesh

    gaussian_dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    gaussian_dtype += [(f"obj_dc_{index}", "<f4") for index in range(16)]
    gaussians = np.zeros(2, dtype=gaussian_dtype)
    gaussians["x"] = (0.0, 10.0)
    gaussians["obj_dc_0"] = (-1.0, 1.0)
    gaussian_path = tmp_path / "semantic_gaussians.ply"
    PlyData([PlyElement.describe(gaussians, "vertex")]).write(gaussian_path)

    mesh = np.zeros(
        2,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
    )
    mesh["x"] = (0.1, 9.9)
    mesh_path = tmp_path / "mesh.ply"
    PlyData([PlyElement.describe(mesh, "vertex")]).write(mesh_path)

    head = SemanticHead(16, 2)
    with torch.no_grad():
        head.classifier.weight.zero_()
        head.classifier.bias.zero_()
        head.classifier.weight[1, 0, 0, 0] = 1.0
    checkpoint = tmp_path / "semantic.pth"
    torch.save(
        {
            "semantic_dim": 16,
            "num_classes": 2,
            "head": head.state_dict(),
        },
        checkpoint,
    )
    output = tmp_path / "semantic_mesh.ply"
    export_semantic_mesh(
        SimpleNamespace(
            mesh=str(mesh_path),
            semantic_ply=str(gaussian_path),
            semantic_checkpoint=str(checkpoint),
            output=str(output),
            labels_output=None,
            workers=1,
        )
    )
    result = PlyData.read(output)["vertex"].data
    assert result["semantic_id"].tolist() == [0, 1]
    assert output.with_suffix(".semantic.npy").is_file()
    assert output.with_suffix(".semantic_distance.npy").is_file()
    assert output.with_suffix(".semantic.json").is_file()


def test_ply_save_preserves_gw_and_gaga_fields(tmp_path):
    from plyfile import PlyData
    from scene.gaussian_model import GaussianModel

    model = GaussianModel(
        1,
        use_mip_filter=True,
        learn_occupancy=True,
        n_gaussian_features=4,
        semantic_dim=16,
    )
    count = 2
    model._xyz = torch.nn.Parameter(torch.zeros(count, 3))
    model._features_dc = torch.nn.Parameter(torch.zeros(count, 1, 3))
    model._features_rest = torch.nn.Parameter(torch.zeros(count, 3, 3))
    model._opacity = torch.nn.Parameter(torch.zeros(count, 1))
    model._scaling = torch.nn.Parameter(torch.zeros(count, 3))
    model._rotation = torch.nn.Parameter(torch.zeros(count, 4))
    model.filter_3D = torch.zeros(count, 1)
    model._base_occupancy = torch.nn.Parameter(
        torch.zeros(count, 9), requires_grad=False
    )
    model._occupancy_shift = torch.nn.Parameter(torch.zeros(count, 9))
    model._gaussian_features = torch.nn.Parameter(torch.zeros(count, 4))
    model._semantic_features = torch.nn.Parameter(torch.zeros(count, 16))

    output = tmp_path / "point_cloud.ply"
    model.save_ply(str(output))
    names = {item.name for item in PlyData.read(output).elements[0].properties}
    assert "filter_3D" in names
    assert {f"gaussian_features_{index}" for index in range(4)} <= names
    assert {f"obj_dc_{index}" for index in range(16)} <= names
    assert {f"base_occupancy_{index}" for index in range(9)} <= names
