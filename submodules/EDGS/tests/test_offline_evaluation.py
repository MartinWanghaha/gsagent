from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

torch = pytest.importorskip("torch")

import metrics as metrics_script  # noqa: E402
import render as render_script  # noqa: E402


def _save_solid_rgb(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def test_render_uses_saved_config_and_latest_valid_ply(tmp_path):
    model_path = tmp_path / "moved_model"
    source_path = tmp_path / "dataset"
    source_path.mkdir()
    for iteration in (9, 30, 100):
        point_path = model_path / "point_cloud" / f"iteration_{iteration}"
        point_path.mkdir(parents=True)
        (point_path / "point_cloud.ply").touch()
    (model_path / "point_cloud" / "iteration_invalid").mkdir()

    config = OmegaConf.create(
        {
            "gs": {
                "sh_degree": 3,
                "dataset": {
                    "source_path": str(source_path),
                    "model_path": "/old/location",
                    "resolution": 8,
                },
                "pipe": {"debug": False},
            }
        }
    )
    OmegaConf.save(config, model_path / "config.yaml")

    loaded = render_script.load_training_config(model_path)

    assert loaded.gs.dataset.resolution == 8
    assert loaded.gs.dataset.model_path == str(model_path)
    assert render_script.available_iterations(model_path) == [9, 30, 100]
    assert render_script.resolve_iteration(model_path, -1) == 100
    with pytest.raises(FileNotFoundError, match="available iterations: 9, 30, 100"):
        render_script.resolve_iteration(model_path, 42)


class _FakeRenderer:
    backend = "pgsr"

    def render(self, view, gaussians, pipeline, background, **options):
        assert options == {"return_plane": True, "return_depth_normal": False}
        height, width = view.original_image.shape[-2:]
        return {
            "render": torch.full((3, height, width), 0.25),
            "plane_depth": torch.ones(1, height, width),
            "rendered_normal": torch.tensor([0.0, 0.0, 1.0])[:, None, None].expand(
                3, height, width
            ),
        }


def test_render_set_writes_paired_pgsr_directories(tmp_path):
    view = SimpleNamespace(
        image_name="nested/frame.JPG",
        original_image=torch.full((3, 4, 6), 0.75),
    )

    method_path = render_script.render_set(
        model_path=tmp_path,
        split="test",
        iteration=30,
        views=[view],
        renderer=_FakeRenderer(),
        gaussians=object(),
        pipeline=object(),
        background=torch.zeros(3),
        max_depth=5.0,
        use_depth_filter=False,
    )

    expected = {
        "renders/frame.png",
        "gt/frame.png",
        "renders_depth/frame.png",
        "renders_normal/frame.png",
        "render_manifest.json",
    }
    actual = {
        str(path.relative_to(method_path))
        for path in method_path.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    manifest = json.loads((method_path / "render_manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["views"] == {"nested/frame.JPG": "frame.png"}


class _FakeLPIPS(torch.nn.Module):
    def forward(self, rendering, ground_truth):
        return (rendering - ground_truth).abs().mean().reshape(1, 1, 1, 1)


def test_metrics_stream_pairs_and_write_pgsr_schema(tmp_path):
    method_path = tmp_path / "test" / "ours_42"
    _save_solid_rgb(method_path / "renders" / "b.png", 64)
    _save_solid_rgb(method_path / "gt" / "b.png", 128)
    _save_solid_rgb(method_path / "renders" / "a.png", 32)
    _save_solid_rgb(method_path / "gt" / "a.png", 96)

    results = metrics_script.evaluate_scene(
        tmp_path,
        device=torch.device("cpu"),
        lpips_model=_FakeLPIPS(),
    )

    assert list(results) == ["ours_42"]
    assert set(results["ours_42"]) == {
        "SSIM",
        "PSNR",
        "LPIPS",
        "LPIPS_3dgs",
    }
    assert results["ours_42"]["LPIPS"] == pytest.approx(64.0 / 255.0)
    assert results["ours_42"]["LPIPS_3dgs"] == results["ours_42"]["LPIPS"]

    saved_results = json.loads((tmp_path / "results.json").read_text())
    per_view = json.loads((tmp_path / "per_view.json").read_text())
    assert saved_results == results
    assert list(per_view["ours_42"]["SSIM"]) == ["a.png", "b.png"]


def test_metrics_reject_unpaired_images(tmp_path):
    method_path = tmp_path / "test" / "ours_7"
    _save_solid_rgb(method_path / "renders" / "only_render.png", 0)
    (method_path / "gt").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="missing GT: only_render.png"):
        metrics_script.paired_images(method_path)


def test_metrics_reject_incomplete_or_stale_manifest(tmp_path):
    method_path = tmp_path / "test" / "ours_7"
    _save_solid_rgb(method_path / "renders" / "expected.png", 0)
    _save_solid_rgb(method_path / "gt" / "expected.png", 0)
    manifest_path = method_path / "render_manifest.json"
    manifest = {
        "complete": False,
        "num_views": 1,
        "views": {"source.JPG": "expected.png"},
    }
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="render method is incomplete"):
        metrics_script.paired_images(method_path)

    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest))
    _save_solid_rgb(method_path / "renders" / "stale.png", 0)
    _save_solid_rgb(method_path / "gt" / "stale.png", 0)
    with pytest.raises(FileNotFoundError, match="unexpected render: stale.png"):
        metrics_script.paired_images(method_path)
