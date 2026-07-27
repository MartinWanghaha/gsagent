from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest
import torch

from training.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    resolve_training_configuration,
    restore_rng_state,
    validate_resume_source,
)
from training.engine import SemanticGaussianTrainer
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]


def _config(iterations: int = 10) -> dict:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["optimization"]["iterations"] = iterations
    config["phases"].update(semantic_from=1, joint_from=2, surface_from=3)
    config["surface"].update(topology_from=3, topology_until=iterations)
    config["density"]["max_gaussians"] = 10
    config["logging"]["log_interval"] = 1
    config["runtime"] = {"source_path": "/old/source"}
    return config


def test_resume_uses_checkpoint_config_and_restricts_overrides(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"version": 3, "iteration": 7, "config": _config()}, checkpoint)
    config, state = resolve_training_configuration(
        default_config="unused.yaml",
        config_path=None,
        overrides=[
            "optimization.iterations=12",
            "semantic.region_decode_chunk_size=4096",
            "surface.support_routing_query_chunk=16384",
            "surface.scipy_workers=2",
            "surface.mesh_feedback_scipy_workers=1",
            "logging.log_interval=4",
        ],
        checkpoint_path=checkpoint,
    )
    assert state is not None and config["optimization"]["iterations"] == 12
    assert config["semantic"]["region_decode_chunk_size"] == 4096
    assert config["surface"]["support_routing_query_chunk"] == 16384
    assert config["surface"]["scipy_workers"] == 2
    assert config["surface"]["mesh_feedback_scipy_workers"] == 1
    assert config["logging"]["log_interval"] == 4
    with pytest.raises(ValueError, match="not allowed"):
        resolve_training_configuration(
            default_config="unused.yaml",
            config_path=None,
            overrides=["loss.lambda_semantic=1.0"],
            checkpoint_path=checkpoint,
        )
    with pytest.raises(ValueError, match="not allowed"):
        resolve_training_configuration(
            default_config="unused.yaml",
            config_path=None,
            overrides=["surface.support_candidate_budget=512"],
            checkpoint_path=checkpoint,
        )


def test_resume_checkpoint_load_prefers_cpu_mmap(monkeypatch) -> None:
    calls = []
    checkpoint_state = {"version": 3, "iteration": 7, "config": _config()}

    def recording_load(path, **kwargs):
        calls.append((path, kwargs))
        return checkpoint_state

    monkeypatch.setattr(torch, "load", recording_load)
    config, state = resolve_training_configuration(
        default_config="unused.yaml",
        config_path=None,
        overrides=[],
        checkpoint_path="checkpoint.pth",
    )

    assert state is checkpoint_state
    assert config["optimization"]["iterations"] == 10
    assert calls == [
        (
            "checkpoint.pth",
            {"mmap": True, "map_location": "cpu", "weights_only": True},
        )
    ]


def test_resume_checkpoint_load_falls_back_when_mmap_is_unsupported(monkeypatch) -> None:
    calls = []
    checkpoint_state = {"version": 3, "iteration": 7, "config": _config()}

    def legacy_load(path, **kwargs):
        calls.append((path, kwargs))
        if "mmap" in kwargs:
            raise TypeError("load() got an unexpected keyword argument 'mmap'")
        return checkpoint_state

    monkeypatch.setattr(torch, "load", legacy_load)
    config, state = resolve_training_configuration(
        default_config="unused.yaml",
        config_path=None,
        overrides=[],
        checkpoint_path="checkpoint.pth",
    )

    assert state is checkpoint_state
    assert config["optimization"]["iterations"] == 10
    assert calls == [
        (
            "checkpoint.pth",
            {"mmap": True, "map_location": "cpu", "weights_only": True},
        ),
        (
            "checkpoint.pth",
            {"map_location": "cpu", "weights_only": True},
        ),
    ]


def test_resume_source_mismatch_fails_without_explicit_relocation() -> None:
    with pytest.raises(ValueError, match="source differs"):
        validate_resume_source(_config(), "/new/source")
    validate_resume_source(_config(), "/new/source", allow_relocation=True)


def test_resume_requires_checkpoint_owned_source_path() -> None:
    config = _config()
    config.pop("runtime")
    with pytest.raises(ValueError, match="runtime mapping"):
        validate_resume_source(config, "/new/source")


@pytest.mark.parametrize("version", ["3", 3.0, None])
def test_resume_rejects_non_integer_schema_version(tmp_path, version) -> None:
    checkpoint = tmp_path / "invalid-version.pth"
    torch.save({"version": version, "iteration": 1, "config": _config()}, checkpoint)
    with pytest.raises(ValueError, match="invalid schema version"):
        resolve_training_configuration(
            default_config="unused.yaml",
            config_path=None,
            overrides=[],
            checkpoint_path=checkpoint,
        )


@pytest.mark.parametrize("iteration", [None, "1", 1.0, True, -1])
def test_resume_rejects_invalid_iteration(tmp_path, iteration) -> None:
    checkpoint = tmp_path / "invalid-iteration.pth"
    torch.save({"version": 3, "iteration": iteration, "config": _config()}, checkpoint)
    with pytest.raises(ValueError, match="invalid iteration"):
        resolve_training_configuration(
            default_config="unused.yaml",
            config_path=None,
            overrides=[],
            checkpoint_path=checkpoint,
        )


def test_rng_state_round_trip() -> None:
    random.seed(4)
    np.random.seed(5)
    torch.manual_seed(6)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected


def test_atomic_checkpoint_save_round_trips(tmp_path) -> None:
    path = tmp_path / "run" / "chkpnt7.pth"
    atomic_torch_save({"iteration": 7, "value": torch.arange(4)}, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["iteration"] == 7
    assert torch.equal(state["value"], torch.arange(4))
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "chkpnt7.pth"
    path.write_bytes(b"previous-valid-checkpoint")

    def fail_save(_value, temporary) -> None:
        temporary.write_bytes(b"partial")
        raise RuntimeError("serialization interrupted")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_torch_save({"iteration": 7}, path)
    assert path.read_bytes() == b"previous-valid-checkpoint"
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_trainer_rejects_unknown_future_checkpoint_schema() -> None:
    trainer = object.__new__(SemanticGaussianTrainer)
    with pytest.raises(ValueError, match="newer schema"):
        trainer.load_state_dict({"version": 4}, optimization_config=None)


def test_resume_rejects_unknown_future_checkpoint_before_scene_construction(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "future.pth"
    torch.save({"version": 4, "iteration": 1, "config": _config()}, checkpoint)
    with pytest.raises(ValueError, match="newer schema"):
        resolve_training_configuration(
            default_config="unused.yaml",
            config_path=None,
            overrides=[],
            checkpoint_path=checkpoint,
        )


def test_resume_rejects_legacy_checkpoint_before_scene_construction(tmp_path) -> None:
    checkpoint = tmp_path / "legacy.pth"
    torch.save({"version": 2, "iteration": 1, "config": _config()}, checkpoint)
    with pytest.raises(ValueError, match="region-conditioned training schema"):
        resolve_training_configuration(
            default_config="unused.yaml",
            config_path=None,
            overrides=[],
            checkpoint_path=checkpoint,
        )
