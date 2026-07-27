from __future__ import annotations

from pathlib import Path
import weakref

import pytest
import torch

from model_io import build_neighbor_index, build_surface_field
from scene import GaussianModel
from train import (
    _geometry_evidence_kwargs,
    _mesh_feedback_kwargs,
    _restore_resume_state,
    save_training_iteration,
)
from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_geometry_evidence_propagation_configuration_is_forwarded() -> None:
    config = {
        "semantic": {
            "geometry_evidence_temperature": 0.31,
            "propagation_enabled": False,
            "propagation_samples": 1234,
            "propagation_neighbors": 7,
            "propagation_min_seed_confidence": 0.22,
            "propagation_max_confidence": 0.77,
            "propagation_momentum": 0.4,
            "propagation_decay": 0.98,
            "propagation_semantic_floor": 0.11,
            "propagation_support_sigma": 2.5,
            "propagation_boundary_barrier": 3.5,
        }
    }
    assert _geometry_evidence_kwargs(config) == {
        "temperature": 0.31,
        "propagation_enabled": False,
        "propagation_samples": 1234,
        "propagation_neighbors": 7,
        "propagation_min_seed_confidence": 0.22,
        "propagation_max_confidence": 0.77,
        "propagation_momentum": 0.4,
        "propagation_decay": 0.98,
        "propagation_semantic_floor": 0.11,
        "propagation_support_sigma": 2.5,
        "propagation_boundary_barrier": 3.5,
    }


def test_async_clean_mesh_feedback_configuration_is_forwarded() -> None:
    config = {
        "surface": {
            "enabled": True,
            "mesh_refresh_interval": 321,
            "mesh_feedback_resolution": 80,
            "mesh_feedback_samples": 456,
            "mesh_feedback_async": False,
            "mesh_feedback_snapshot_device": "cpu",
            "mesh_feedback_scipy_workers": 3,
            "mesh_feedback_min_component_faces": 99,
            "mesh_feedback_max_candidate_age": 654,
            "mesh_feedback_max_topology_events": 2,
            "mesh_feedback_max_churn_ratio": 0.02,
            "mesh_feedback_retry_interval": 55,
            "mesh_feedback_blend_iterations": 77,
            "mesh_feedback_gate_probes": 333,
            "mesh_feedback_gate_min_score": 0.71,
            "mesh_feedback_gate_sdf_p90": 2.2,
            "mesh_feedback_gate_normal": 0.61,
            "mesh_feedback_gate_semantic": 0.51,
            "mesh_feedback_min_opacity": 0.06,
            "mesh_feedback_min_confidence": 0.36,
            "mesh_feedback_min_expert_certainty": 0.56,
            "mesh_feedback_match_k": 7,
            "mesh_feedback_match_radius": 2.7,
            "mesh_feedback_match_semantic": 0.52,
            "mesh_feedback_robust_delta": 1.7,
            "mesh_feedback_min_matches": 222,
        },
        "mesh": {
            "padding": 0.03,
            "support_sigma": 2.0,
            "method": "marching_tetrahedra",
            "isovalue": 0.1,
        },
        "loss": {"lambda_mesh": 0.2},
    }
    assert _mesh_feedback_kwargs(config) == {
        "refresh_interval": 321,
        "resolution": 80,
        "sample_points": 456,
        "padding": 0.03,
        "support_sigma": 2.0,
        "method": "marching_tetrahedra",
        "isovalue": 0.1,
        "enabled": True,
        "async_refresh": False,
        "snapshot_device": "cpu",
        "scipy_workers": 3,
        "min_component_faces": 99,
        "max_candidate_age": 654,
        "max_topology_events": 2,
        "max_churn_ratio": 0.02,
        "retry_interval": 55,
        "blend_iterations": 77,
        "gate_probes": 333,
        "gate_min_score": 0.71,
        "gate_sdf_p90": 2.2,
        "gate_normal": 0.61,
        "gate_semantic": 0.51,
        "min_opacity": 0.06,
        "min_confidence": 0.36,
        "min_expert_certainty": 0.56,
        "match_k": 7,
        "match_radius": 2.7,
        "match_semantic": 0.52,
        "robust_delta": 1.7,
        "min_matches": 222,
    }


def test_checkpoint_is_published_before_optional_ply_failure(tmp_path) -> None:
    class FailingScene:
        def save(self, iteration: int) -> None:
            assert (tmp_path / f"chkpnt{iteration}.pth").is_file()
            raise RuntimeError("PLY export failed")

    with pytest.raises(RuntimeError, match="PLY export failed"):
        save_training_iteration(FailingScene(), {"iteration": 4}, tmp_path, 4)
    checkpoint = torch.load(tmp_path / "chkpnt4.pth", map_location="cpu", weights_only=True)
    assert checkpoint["iteration"] == 4


@pytest.mark.parametrize("fail", [False, True])
def test_resume_state_is_released_after_restore(monkeypatch, fail: bool) -> None:
    payload = torch.ones(1024)
    payload_reference = weakref.ref(payload)
    resume_state = {"iteration": 17, "payload": payload}
    del payload
    collect_calls = 0

    def collect() -> int:
        nonlocal collect_calls
        collect_calls += 1
        return 0

    class Trainer:
        def load_state_dict(self, state, optimization) -> int:
            assert state["payload"].numel() == 1024
            assert optimization == "optimizer-config"
            if fail:
                raise RuntimeError("restore failed")
            return int(state["iteration"])

    monkeypatch.setattr("train.gc.collect", collect)
    if fail:
        with pytest.raises(RuntimeError, match="restore failed"):
            _restore_resume_state(Trainer(), resume_state, "optimizer-config")
    else:
        assert _restore_resume_state(Trainer(), resume_state, "optimizer-config") == 17

    assert resume_state == {}
    assert payload_reference() is None
    assert collect_calls == 1


def test_surface_field_receives_neighbor_backend_and_memory_policy() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config["surface"].update(
        neighbor_backend="exact",
        max_distance_bytes=12_345,
        support_routing_query_chunk=987,
        scipy_workers=2,
    )
    model = GaussianModel(device="cpu")
    index = build_neighbor_index(model, config)
    field = build_surface_field(model, config, index)

    assert field.neighbor_backend == "exact"
    assert field.max_distance_bytes == 12_345
    assert field.neighbor_index is index
    assert index.support_routing_query_chunk == 987
    assert index.scipy_workers == 2
    assert field.support_routing_query_chunk == 987
    assert field.scipy_workers == 2
