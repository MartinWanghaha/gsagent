import json
from types import SimpleNamespace

import pytest
import torch

import render as render_module
from render import (
    _decode_semantic_labels,
    _ground_truth_image,
    render_set,
    select_iterations,
    select_views,
)


def test_transparent_ground_truth_uses_white_render_background() -> None:
    image = torch.zeros(3, 2, 2)
    image[:, 0, 0] = torch.tensor([0.2, 0.4, 0.6])
    mask = torch.zeros(1, 2, 2)
    mask[:, 0, 0] = 1.0
    camera = SimpleNamespace(original_image=image, gt_mask=mask)

    target = _ground_truth_image(camera, torch.ones(3))
    assert torch.allclose(target[:, 0, 0], image[:, 0, 0])
    assert torch.all(target[:, 0, 1:] == 1.0)
    assert torch.all(target[:, 1, :] == 1.0)


def test_opaque_ground_truth_is_unchanged() -> None:
    image = torch.rand(3, 2, 2)
    camera = SimpleNamespace(original_image=image, gt_mask=None)
    assert _ground_truth_image(camera, torch.ones(3)) is image


def test_render_set_skips_discrete_semantics_for_rgb_only_scene(tmp_path, monkeypatch) -> None:
    height, width = 3, 5
    camera = SimpleNamespace(
        original_image=torch.rand(3, height, width),
        gt_mask=None,
        semantic_ids=None,
    )
    package = {
        "render": torch.rand(3, height, width),
        "semantic": torch.rand(4, height, width),
        "expected_depth": torch.ones(1, height, width),
        "alpha": torch.ones(1, height, width),
        "normal": torch.zeros(3, height, width),
    }
    monkeypatch.setattr(render_module, "render", lambda *args, **kwargs: package)
    gaussians = SimpleNamespace(semantic_decoder=None)

    render_set(
        tmp_path,
        "test",
        7,
        [camera],
        gaussians,
        SimpleNamespace(),
        torch.zeros(3),
        "reference",
        0,
    )

    method = tmp_path / "test" / "ours_7"
    assert not (method / "gt_semantic_id" / "00000.npy").exists()
    assert not (method / "semantic_id").exists()
    metadata = json.loads((method / "metadata.json").read_text(encoding="utf8"))
    assert metadata["has_semantic_ground_truth"] is False
    assert metadata["semantic_label_output"] is False


def test_view_selection_is_repeatable_union_and_preserves_original_indices() -> None:
    views = [
        SimpleNamespace(image_name="frames/left.png"),
        SimpleNamespace(image_name="middle.jpg"),
        SimpleNamespace(image_name="right.png"),
    ]
    selected, indices, matched_indices, matched_names = select_views(
        views,
        requested_indices=[2],
        requested_names=["left", "middle.jpg"],
    )
    assert selected == views
    assert indices == [0, 1, 2]
    assert matched_indices == {2}
    assert matched_names == {"left", "middle.jpg"}


def test_render_subset_uses_original_filename_index(tmp_path, monkeypatch) -> None:
    camera = SimpleNamespace(
        image_name="view-seven.png",
        original_image=torch.rand(3, 2, 3),
        gt_mask=None,
        semantic_ids=None,
    )
    package = {
        "render": torch.rand(3, 2, 3),
        "semantic": torch.rand(4, 2, 3),
        "expected_depth": torch.ones(1, 2, 3),
        "alpha": torch.ones(1, 2, 3),
        "normal": torch.zeros(3, 2, 3),
    }
    monkeypatch.setattr(render_module, "render", lambda *args, **kwargs: package)
    render_set(
        tmp_path,
        "test",
        11,
        [camera],
        SimpleNamespace(semantic_decoder=None),
        SimpleNamespace(),
        torch.zeros(3),
        "reference",
        0,
        view_indices=[7],
        total_views=30,
    )
    method = tmp_path / "test" / "ours_11"
    assert (method / "renders" / "00007.png").is_file()
    assert not (method / "renders" / "00000.png").exists()
    metadata = json.loads((method / "metadata.json").read_text(encoding="utf8"))
    assert metadata["view_indices"] == [7]
    assert metadata["view_names"] == ["view-seven.png"]
    assert metadata["total_views"] == 30


def test_semantic_render_decode_is_exact_and_chunked() -> None:
    class RecordingDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3, 5)
            self.calls = []

        def forward(self, embedding):
            self.calls.append((embedding.shape[0], embedding.dtype))
            return self.linear(embedding)

    torch.manual_seed(4)
    decoder = RecordingDecoder()
    semantic = torch.randn(3, 3, 4)
    expected = decoder.linear(semantic.permute(1, 2, 0).reshape(-1, 3).float()).argmax(-1)
    decoder.calls.clear()
    labels = _decode_semantic_labels(
        SimpleNamespace(semantic_decoder=decoder),
        semantic,
        chunk_size=5,
    )
    assert torch.equal(labels.flatten(), expected)
    assert decoder.calls == [(5, torch.float32), (5, torch.float32), (2, torch.float32)]


def test_semantic_render_decode_requires_decoder() -> None:
    with pytest.raises(RuntimeError, match="semantic decoder"):
        _decode_semantic_labels(
            SimpleNamespace(semantic_decoder=None),
            torch.randn(3, 2, 4),
            chunk_size=5,
        )


def test_render_iteration_selection_supports_repeat_and_all(tmp_path) -> None:
    for iteration in (7000, 12000, 24000):
        (tmp_path / f"chkpnt{iteration}.pth").touch()
    assert select_iterations(tmp_path, [7000, 24000]) == [7000, 24000]
    assert select_iterations(tmp_path) == [24000]
    assert select_iterations(tmp_path, all_iterations=True) == [7000, 12000, 24000]


def test_render_cli_processes_iterations_sequentially(tmp_path, monkeypatch) -> None:
    for iteration in (7, 12):
        (tmp_path / f"chkpnt{iteration}.pth").touch()
    calls = []
    monkeypatch.setattr(
        render_module,
        "_render_iteration",
        lambda _args, iteration: calls.append(iteration),
    )
    render_module.main(["-m", str(tmp_path), "--skip_train", "--all-iterations"])
    assert calls == [7, 12]
