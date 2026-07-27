#!/usr/bin/env python3
"""Benchmark candidate-first surface consistency on a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_io import load_trained_scene  # noqa: E402
from regularization.surface import (  # noqa: E402
    gaussian_surface_consistency,
    prepare_gaussian_surface_consistency,
)
from semantic import GaussianNeighborIndex, SemanticSurfaceField  # noqa: E402


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "-m", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-points", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--validate-candidate-points",
        type=int,
        default=0,
        help=(
            "Compare bounded candidate routing against an exact all-Gaussian "
            "support scan on three probes per sampled Gaussian."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--support-candidate-budget",
        type=int,
        help="Override the checkpoint's bounded cKDTree shortlist budget.",
    )
    parser.add_argument(
        "--support-routing-query-chunk",
        type=int,
        help="Override the CPU cKDTree routing row batch without changing candidates.",
    )
    parser.add_argument(
        "--scipy-workers",
        type=int,
        help="Override cKDTree workers (-1 means all CPUs).",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if (
        args.sample_points < 1
        or args.warmup < 0
        or args.repeats < 1
        or args.validate_candidate_points < 0
        or (
            args.support_candidate_budget is not None
            and args.support_candidate_budget < 1
        )
        or (
            args.support_routing_query_chunk is not None
            and args.support_routing_query_chunk < 1
        )
        or (
            args.scipy_workers is not None
            and (args.scipy_workers == 0 or args.scipy_workers < -1)
        )
    ):
        parser.error(
            "sample/repeat/routing values must be positive, warmup non-negative, "
            "and scipy-workers must be -1 or positive"
        )
    torch.manual_seed(args.seed)

    bundle = load_trained_scene(
        args.model_path,
        args.iteration,
        args.device,
        with_surface_field=True,
    )
    gaussians = bundle["gaussians"]
    field = bundle["surface_field"]
    index = bundle["neighbor_index"]
    device = bundle["device"]
    if args.support_candidate_budget is not None:
        index.support_candidate_budget = int(args.support_candidate_budget)
        field.support_candidate_budget = int(args.support_candidate_budget)
    if args.support_routing_query_chunk is not None:
        index.support_routing_query_chunk = int(args.support_routing_query_chunk)
        field.support_routing_query_chunk = int(args.support_routing_query_chunk)
    if args.scipy_workers is not None:
        index.scipy_workers = int(args.scipy_workers)
        field.scipy_workers = int(args.scipy_workers)

    _synchronize(device)
    start = time.perf_counter()
    backend = index.refresh(force=True)
    _synchronize(device)
    refresh_seconds = time.perf_counter() - start

    policy_rows: list[int] = []
    hook = field.policy_bank.register_forward_hook(
        lambda _module, inputs, _output: policy_rows.append(int(inputs[0].shape[0]))
    )
    timings: list[float] = []
    peak_memory = None
    try:
        for step in range(args.warmup + args.repeats):
            for parameter in gaussians.parameters():
                parameter.grad = None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _synchronize(device)
            start = time.perf_counter()
            loss, _ = gaussian_surface_consistency(
                gaussians,
                field,
                sample_points=args.sample_points,
            )
            loss.backward()
            _synchronize(device)
            elapsed = time.perf_counter() - start
            if step >= args.warmup:
                timings.append(elapsed)
                if device.type == "cuda":
                    value = int(torch.cuda.max_memory_allocated(device))
                    peak_memory = value if peak_memory is None else max(peak_memory, value)
    finally:
        hook.remove()

    candidate_validation = None
    if args.validate_candidate_points:
        prepared = prepare_gaussian_surface_consistency(
            gaussians,
            sample_points=args.validate_candidate_points,
        )
        assert prepared is not None
        points = prepared.query_points.detach()
        approximate = index.query_support(
            points,
            field.k_neighbors,
            density_scale=field.density_scale,
            minimum_log_support=field.support_log_cutoff,
        )
        exact_index = GaussianNeighborIndex(
            gaussians,
            backend="exact",
            gaussian_chunk_size=index.gaussian_chunk_size,
            query_chunk_size=index.query_chunk_size,
            max_distance_bytes=index.max_distance_bytes,
            support_candidate_budget=index.support_candidate_budget,
        )
        _synchronize(device)
        validation_start = time.perf_counter()
        exact = exact_index.query_support(
            points,
            field.k_neighbors,
            density_scale=field.density_scale,
            minimum_log_support=field.support_log_cutoff,
        )
        _synchronize(device)
        overlap = (exact[:, :, None] == approximate[:, None, :]).any(dim=-1)
        exact_field = SemanticSurfaceField(
            gaussians,
            policy_bank=field.policy_bank,
            k_neighbors=field.k_neighbors,
            query_chunk_size=field.query_chunk_size,
            gaussian_chunk_size=field.gaussian_chunk_size,
            occupancy_iso=field.occupancy_iso,
            density_scale=field.density_scale,
            semantic_decoder=field.semantic_decoder,
            max_distance_bytes=field.max_distance_bytes,
            neighbor_backend="exact",
            neighbor_index=exact_index,
            support_log_cutoff=field.support_log_cutoff,
            support_candidate_budget=index.support_candidate_budget,
        )
        approximate_result = field.query(points)
        exact_result = exact_field.query(points)
        normal_cosine = torch.nn.functional.cosine_similarity(
            approximate_result.normal,
            exact_result.normal,
            dim=-1,
            eps=1e-8,
        ).abs()
        semantic_cosine = torch.nn.functional.cosine_similarity(
            approximate_result.semantic,
            exact_result.semantic,
            dim=-1,
            eps=1e-8,
        )
        candidate_validation = {
            "gaussian_samples": int(args.validate_candidate_points),
            "query_points": int(points.shape[0]),
            "neighbors": int(field.k_neighbors),
            "top1_accuracy": float((exact[:, 0] == approximate[:, 0]).float().mean()),
            "topk_recall": float(overlap.float().mean()),
            "occupancy_mae": float(
                (approximate_result.occupancy - exact_result.occupancy).abs().mean()
            ),
            "sdf_mae": float(
                (approximate_result.sdf - exact_result.sdf).abs().mean()
            ),
            "normal_abs_cosine": float(normal_cosine.mean()),
            "semantic_cosine": float(semantic_cosine.mean()),
            "uncertainty_mae": float(
                (approximate_result.uncertainty - exact_result.uncertainty).abs().mean()
            ),
            "exact_scan_seconds": time.perf_counter() - validation_start,
        }

    result = {
        "model_path": str(Path(args.model_path).resolve()),
        "iteration": int(bundle["iteration"]),
        "device": str(device),
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "sample_points": int(args.sample_points),
        "query_points_per_step": int(3 * args.sample_points),
        "neighbor_backend": backend,
        "support_candidate_budget": int(index.support_candidate_budget),
        "support_routing_query_chunk": int(index.support_routing_query_chunk),
        "scipy_workers": int(index.scipy_workers),
        "index_refresh_seconds": refresh_seconds,
        "step_seconds": timings,
        "step_seconds_median": statistics.median(timings),
        "step_seconds_mean": statistics.fmean(timings),
        "candidate_policy_rows": policy_rows,
        "candidate_validation": candidate_validation,
        "peak_memory_bytes": peak_memory,
    }
    serialized = json.dumps(result, indent=2) + "\n"
    print(serialized, end="")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized, encoding="utf8")


if __name__ == "__main__":
    main()
