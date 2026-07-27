from __future__ import annotations

import torch

import model_io
from scene.gaussian_model import SURFACE_INFERENCE_ATTRIBUTES


def test_inference_checkpoint_load_is_cpu_only_and_drops_optimizer(tmp_path, monkeypatch) -> None:
    path = tmp_path / "chkpnt3.pth"
    gaussian_state = {
        "registry": {"tensors": {"xyz": torch.ones(2, 3)}},
        "optimizer": {"state": {0: {"exp_avg": torch.ones(1024)}}},
        "active_sh_degree": 3,
    }
    torch.save(
        {
            "gaussians": gaussian_state,
            "head_optimizer": {"state": {0: {"exp_avg": torch.ones(2048)}}},
            "density": {"gradient_accumulator": torch.ones(4096)},
        },
        path,
    )
    original_load = torch.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(model_io.torch, "load", recording_load)
    inference = model_io.load_inference_gaussian_state(path)

    assert calls == [{"mmap": True, "map_location": "cpu", "weights_only": True}]
    assert inference["optimizer"] is None
    assert inference["registry"]["tensors"]["xyz"].device.type == "cpu"
    assert "head_optimizer" not in inference
    assert "density" not in inference


def test_inference_checkpoint_load_falls_back_when_mmap_is_unsupported(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "legacy.pth"
    calls = []
    checkpoint = {
        "registry": {"tensors": {"xyz": torch.ones(1, 3)}},
        "optimizer": {"large": torch.ones(8)},
    }

    def legacy_load(_path, **kwargs):
        calls.append(kwargs.copy())
        if kwargs.get("mmap"):
            raise TypeError("unexpected keyword argument 'mmap'")
        return checkpoint

    monkeypatch.setattr(model_io.torch, "load", legacy_load)
    inference = model_io.load_inference_gaussian_state(path)
    assert calls == [
        {"mmap": True, "map_location": "cpu", "weights_only": True},
        {"map_location": "cpu", "weights_only": True},
    ]
    assert inference["optimizer"] is None


def test_surface_inference_scope_drops_render_only_registry_tensors(
    tmp_path,
) -> None:
    path = tmp_path / "surface.pth"
    names = (*SURFACE_INFERENCE_ATTRIBUTES, "features_dc", "features_rest")
    tensors = {name: torch.ones(2, 1) for name in names}
    torch.save(
        {
            "gaussians": {
                "registry": {
                    "format_version": 2,
                    "specs": [
                        {"name": name, "trailing_shape": [1]} for name in names
                    ],
                    "tensors": tensors,
                },
                "optimizer": {"unused": torch.ones(1)},
            }
        },
        path,
    )

    inference = model_io.load_inference_gaussian_state(
        path,
        scope="surface",
    )

    assert tuple(inference["registry"]["tensors"]) == SURFACE_INFERENCE_ATTRIBUTES
    assert [item["name"] for item in inference["registry"]["specs"]] == list(
        SURFACE_INFERENCE_ATTRIBUTES
    )
    assert "features_dc" not in inference["registry"]["tensors"]
    assert inference["optimizer"] is None


def test_latest_iteration_prefers_complete_checkpoint_over_orphan_ply(tmp_path) -> None:
    (tmp_path / "chkpnt7.pth").touch()
    orphan = tmp_path / "point_cloud" / "iteration_9"
    orphan.mkdir(parents=True)
    (orphan / "point_cloud.ply").touch()

    assert model_io.resolve_iteration(tmp_path, -1) == 7
    assert model_io.resolve_iteration(tmp_path, 9) == 9
    assert model_io.available_iterations(tmp_path) == [7]

    (tmp_path / "chkpnt7.pth").unlink()
    assert model_io.resolve_iteration(tmp_path, -1) == 9
    assert model_io.available_iterations(tmp_path) == [9]
