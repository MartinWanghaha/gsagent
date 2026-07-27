"""Train Semantic Gaussian Wrapping from a COLMAP or Blender scene."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from model_io import (
    build_neighbor_index,
    build_surface_field,
    dataset_namespace,
    optimization_namespace,
    pipeline_namespace,
    resolve_device,
)
from regularization import MeshFeedbackRegularizer
from scene import GaussianModel, Scene
from semantic import GeometryEvidenceProjector
from training import (
    SemanticGaussianTrainer,
    atomic_torch_save,
    resolve_training_configuration,
    validate_resume_source,
)
from utils.config_utils import save_config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mesh_feedback_kwargs(config: dict) -> dict:
    surface = config["surface"]
    mesh = config["mesh"]
    return {
        "refresh_interval": int(surface["mesh_refresh_interval"]),
        "resolution": int(surface["mesh_feedback_resolution"]),
        "sample_points": int(surface["mesh_feedback_samples"]),
        "padding": float(mesh["padding"]),
        "support_sigma": float(mesh["support_sigma"]),
        "method": str(mesh["method"]),
        "isovalue": float(mesh["isovalue"]),
        "enabled": bool(surface["enabled"])
        and float(config["loss"]["lambda_mesh"]) > 0,
        "async_refresh": bool(surface["mesh_feedback_async"]),
        "snapshot_device": str(surface["mesh_feedback_snapshot_device"]),
        "min_component_faces": int(surface["mesh_feedback_min_component_faces"]),
        # The stale-mesh worker shares host CPU resources with the live
        # renderer/surface path.  A single SciPy worker avoids oversubscribing
        # every core while preserving the exact same cKDTree candidates.
        "scipy_workers": int(surface["mesh_feedback_scipy_workers"]),
        "max_candidate_age": int(surface["mesh_feedback_max_candidate_age"]),
        "max_topology_events": int(surface["mesh_feedback_max_topology_events"]),
        "max_churn_ratio": float(surface["mesh_feedback_max_churn_ratio"]),
        "retry_interval": int(surface["mesh_feedback_retry_interval"]),
        "blend_iterations": int(surface["mesh_feedback_blend_iterations"]),
        "gate_probes": int(surface["mesh_feedback_gate_probes"]),
        "gate_min_score": float(surface["mesh_feedback_gate_min_score"]),
        "gate_sdf_p90": float(surface["mesh_feedback_gate_sdf_p90"]),
        "gate_normal": float(surface["mesh_feedback_gate_normal"]),
        "gate_semantic": float(surface["mesh_feedback_gate_semantic"]),
        "min_opacity": float(surface["mesh_feedback_min_opacity"]),
        "min_confidence": float(surface["mesh_feedback_min_confidence"]),
        "min_expert_certainty": float(
            surface["mesh_feedback_min_expert_certainty"]
        ),
        "match_k": int(surface["mesh_feedback_match_k"]),
        "match_radius": float(surface["mesh_feedback_match_radius"]),
        "match_semantic": float(surface["mesh_feedback_match_semantic"]),
        "robust_delta": float(surface["mesh_feedback_robust_delta"]),
        "min_matches": int(surface["mesh_feedback_min_matches"]),
    }


def _geometry_evidence_kwargs(config: dict) -> dict:
    semantic = config["semantic"]
    return {
        "temperature": float(semantic["geometry_evidence_temperature"]),
        "propagation_enabled": bool(semantic["propagation_enabled"]),
        "propagation_samples": int(semantic["propagation_samples"]),
        "propagation_neighbors": int(semantic["propagation_neighbors"]),
        "propagation_min_seed_confidence": float(
            semantic["propagation_min_seed_confidence"]
        ),
        "propagation_max_confidence": float(
            semantic["propagation_max_confidence"]
        ),
        "propagation_momentum": float(semantic["propagation_momentum"]),
        "propagation_decay": float(semantic["propagation_decay"]),
        "propagation_semantic_floor": float(
            semantic["propagation_semantic_floor"]
        ),
        "propagation_support_sigma": float(
            semantic["propagation_support_sigma"]
        ),
        "propagation_boundary_barrier": float(
            semantic["propagation_boundary_barrier"]
        ),
    }


def save_training_iteration(scene, state: dict, output: str | Path, iteration: int) -> None:
    """Publish the self-contained checkpoint before its optional PLY view."""

    output = Path(output)
    atomic_torch_save(state, output / f"chkpnt{iteration}.pth")
    scene.save(iteration)


def _restore_resume_state(
    trainer: SemanticGaussianTrainer,
    resume_state: dict[str, Any],
    optimization: Any,
) -> int:
    """Restore a checkpoint without retaining its CPU serialization buffers.

    ``torch.load(..., map_location="cpu")`` materializes every Gaussian and
    optimizer tensor on the host.  ``trainer.load_state_dict`` copies those
    values into the live model/optimizers, so keeping the source dictionary
    alive after restore needlessly duplicates a large scene and can force the
    training process into swap.  Clear it in ``finally`` so partially failed
    restores release the same memory before the exception propagates.
    """

    try:
        return int(trainer.load_state_dict(resume_state, optimization))
    finally:
        resume_state.clear()
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", "-s", required=True)
    parser.add_argument("--model_path", "-m", required=True)
    parser.add_argument(
        "--config",
        help="fresh-run config; resume uses the resolved config inside its checkpoint",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-source-relocation", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config, resume_state = resolve_training_configuration(
        default_config=Path(__file__).resolve().parent / "configs" / "default.yaml",
        config_path=args.config,
        overrides=args.overrides,
        checkpoint_path=args.checkpoint,
    )
    if resume_state is not None:
        validate_resume_source(
            config,
            args.source_path,
            allow_relocation=args.allow_source_relocation,
        )
    if args.device is not None:
        config["device"] = args.device
    output = Path(args.model_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config["runtime"] = {
        "source_path": str(Path(args.source_path).resolve()),
        "model_path": str(output),
    }
    save_config(config, output / "config.yaml")
    _seed_everything(int(config.get("seed", 0)))
    device = resolve_device(config.get("device", "cuda"))
    dataset = dataset_namespace(config, args.source_path, output, device)
    optimization = optimization_namespace(config)
    pipeline = pipeline_namespace(config)
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.semantic_dim,
        dataset.geometry_experts,
        device,
        confidence_floor=float(
            config.get("semantic", {}).get("confidence_floor", 0.05)
        ),
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(optimization)
    neighbor_index = build_neighbor_index(gaussians, config)
    surface_field = build_surface_field(gaussians, config, neighbor_index)
    mesh_feedback = MeshFeedbackRegularizer(
        surface_field,
        **_mesh_feedback_kwargs(config),
    )
    trainer = SemanticGaussianTrainer(
        scene,
        gaussians,
        pipeline,
        config,
        surface_field=surface_field,
        mesh_feedback=mesh_feedback,
        policy_bank=gaussians.policy_bank,
        evidence_projector=GeometryEvidenceProjector(
            neighbor_index=neighbor_index,
            **_geometry_evidence_kwargs(config),
        ),
        neighbor_index=neighbor_index,
        output_path=output,
    )
    start_iteration = 0
    if resume_state is not None:
        start_iteration = _restore_resume_state(trainer, resume_state, optimization)
        # Drop the now-empty container as well.  The large tensor storages were
        # released inside _restore_resume_state immediately after restoration.
        resume_state = None

    def save(iteration: int, state: dict) -> None:
        save_training_iteration(scene, state, output, iteration)

    save_iterations = config.get("logging", {}).get("save_iterations", [])
    try:
        result = trainer.train(
            start_iteration=start_iteration,
            save_iterations=save_iterations,
            save_callback=save,
        )
    finally:
        # Background mesh extraction owns an executor and possibly a frozen GPU
        # snapshot. Release both on successful completion and on training errors
        # instead of relying on interpreter shutdown ordering.
        mesh_feedback.close()
    if not args.quiet:
        print(
            f"Training complete: iteration={result.iteration}, gaussians={result.gaussian_count}, "
            f"elapsed={result.elapsed_seconds:.1f}s"
        )


if __name__ == "__main__":
    main()
