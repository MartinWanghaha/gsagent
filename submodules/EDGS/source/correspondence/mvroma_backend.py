"""EDGS Gaussian initialization from direct, dense MV-RoMa predictions."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from source.vendor import bootstrap_gaussian_splatting

from .config import (
    MVRoMaTrainingSettings,
    PaperPostprocessSettings,
    as_bool,
    resolve_group_budget,
)
from .contracts import DirectedPair, MVRoMaInitializationResult, normalized_to_pixel
from .geometry import weighted_multiview_dlt
from .manifest import write_mvroma_artifacts
from .mvroma_runtime import (
    InferenceGroup,
    MVRoMaRuntime,
    MVRoMaSettings,
    _get,
)
from .overlap import (
    OverlapEstimate,
    build_matcher_visibility_overlap,
    estimate_colmap_visibility_overlap,
    resolve_colmap_sparse_model,
)
from .pair_store import PairFieldStore
from .postprocess import (
    apply_reciprocal_cycle_filter,
    build_group_track_candidates,
    merge_dense_group,
)
from .planning import (
    CameraGroup,
    CameraGroupPlan,
    OverlapPlannerSettings,
    plan_overlap_aware_groups,
)
from .seed_fusion import (
    GaussianSeedBatch,
    deduplicate_seed_batches,
    triangulate_source_candidates,
)


@dataclass(frozen=True)
class _SeedBatch:
    xyz: torch.Tensor
    rgb: torch.Tensor
    distance_to_source: torch.Tensor
    reprojection_error: torch.Tensor


def _nested(config: Any, name: str) -> Any:
    value = _get(config, name, None)
    if value is None:
        raise ValueError(f"init_wC.{name} must be configured")
    return value


def _optional_path(config: Any, name: str) -> Path | None:
    value = _get(config, name, None)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return Path(text).expanduser()


def _build_camera_group_plan(
    cameras: Sequence[Any],
    cfg: Any,
    mvroma_config: Any,
    *,
    overlap: OverlapEstimate | None = None,
) -> tuple[OverlapEstimate, CameraGroupPlan]:
    planner_config = _get(mvroma_config, "group_planner", {})
    overlap_provider = str(_get(planner_config, "overlap_provider", "colmap")).lower()
    if overlap_provider not in {"colmap", "colmap_tracks", "ufm_visibility"}:
        raise ValueError(
            "group_planner.overlap_provider must be colmap_tracks or "
            f"ufm_visibility, got {overlap_provider!r}"
        )
    if overlap is None:
        if overlap_provider == "ufm_visibility":
            raise ValueError("ufm_visibility overlap must be computed by the runtime")
        image_root = _optional_path(mvroma_config, "image_root")
        sparse_model_path = resolve_colmap_sparse_model(
            image_root=image_root,
            configured_path=_get(planner_config, "sparse_model_path", None),
        )
        overlap = estimate_colmap_visibility_overlap(cameras, sparse_model_path)

    configured_budget = _get(planner_config, "primary_group_budget", None)
    budget_policy = _get(planner_config, "budget_policy", None)
    if configured_budget is not None:
        primary_group_budget = int(configured_budget)
    elif budget_policy is not None:
        primary_group_budget = resolve_group_budget(
            str(budget_policy),
            len(cameras),
            _get(planner_config, "fixed_budget", None),
        )
    else:
        # Preserve the pre-profile MV adapter's historical auto behavior.
        primary_group_budget = max(len(cameras), int(_get(cfg, "num_refs")))
    settings = OverlapPlannerSettings(
        targets_per_group=min(
            int(_get(planner_config, "targets_per_group", 4)), len(cameras) - 1
        ),
        min_targets_per_group=int(
            _get(planner_config, "min_targets_per_group", 1)
        ),
        primary_group_budget=primary_group_budget,
        overlap_threshold=float(_get(planner_config, "overlap_threshold", 0.05)),
        source_quota_exponent=float(
            _get(planner_config, "source_quota_exponent", 0.75)
        ),
        source_overlap_weight=float(
            _get(planner_config, "source_overlap_weight", 1.0)
        ),
        target_overlap_weight=float(
            _get(planner_config, "target_overlap_weight", 1.0)
        ),
        pair_reuse_penalty=float(
            _get(planner_config, "pair_reuse_penalty", 1.0)
        ),
        augment_reciprocity=as_bool(
            _get(planner_config, "augment_reciprocity", True),
            name="group_planner.augment_reciprocity",
        ),
        allow_partial_groups=as_bool(
            _get(planner_config, "allow_partial_groups", False),
            name="group_planner.allow_partial_groups",
        ),
    )
    return overlap, plan_overlap_aware_groups(overlap.matrix, settings)


def _resolve_image_path(image_root: Path, image_name: str) -> Path:
    relative = Path(str(image_name))
    if relative.is_absolute():
        raise ValueError(f"camera image_name must be relative, got {image_name!r}")
    root = image_root.resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"camera image path escapes image_root: {image_name!r}")
    if candidate.is_file():
        return candidate
    if relative.suffix:
        raise FileNotFoundError(f"camera image does not exist: {candidate}")
    matches = [
        path
        for suffix in (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")
        if (path := candidate.with_suffix(suffix)).is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"could not resolve exactly one image for camera {image_name!r} under {root}"
        )
    return matches[0]


def _write_camera_png(camera: Any, path: Path) -> None:
    image = getattr(camera, "original_image", None)
    if not torch.is_tensor(image) or image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("camera.original_image must have shape [C,H,W]")
    rgb = (
        image[:3]
        .detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")


@contextmanager
def _camera_image_paths(
    cameras: Sequence[Any],
    mvroma_config: Any,
) -> Iterator[list[Path]]:
    """Resolve originals, falling back to temporary lossless camera PNGs."""

    image_root_value = _get(mvroma_config, "image_root", None)
    image_root_text = None if image_root_value is None else str(image_root_value).strip()
    if image_root_text and image_root_text.lower() not in {"none", "null"}:
        image_root = Path(image_root_text).expanduser()
        if not image_root.is_dir():
            raise FileNotFoundError(f"MV-RoMa image_root does not exist: {image_root}")
        paths = [
            _resolve_image_path(image_root, str(camera.image_name)) for camera in cameras
        ]
        yield paths
        return

    print(
        "Warning: init_wC.mvroma.image_root is not set; using temporary lossless "
        "PNGs at the currently loaded EDGS resolution. Set image_root for "
        "full-resolution matching."
    )
    with tempfile.TemporaryDirectory(prefix="edgs-mvroma-") as directory:
        temporary_root = Path(directory)
        paths = []
        for index, camera in enumerate(cameras):
            path = temporary_root / f"{index:06d}.png"
            _write_camera_png(camera, path)
            paths.append(path)
        yield paths


def _make_inference_groups(
    paths: Sequence[Path], camera_groups: Sequence[CameraGroup]
) -> list[InferenceGroup]:
    return [
        InferenceGroup(
            cameras=group,
            source_path=paths[group.source_index],
            target_paths=tuple(paths[index] for index in group.target_indices),
        )
        for group in camera_groups
    ]


def _sample_source_colors(camera: Any, normalized_xy: torch.Tensor) -> torch.Tensor:
    image = getattr(camera, "original_image", None)
    if not torch.is_tensor(image) or image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("camera.original_image must have shape [C,H,W]")
    height, width = int(image.shape[1]), int(image.shape[2])
    pixels = normalized_to_pixel(normalized_xy, height, width).round().long()
    x = pixels[:, 0].clamp(0, width - 1)
    y = pixels[:, 1].clamp(0, height - 1)
    return image[:3, y, x].transpose(0, 1).to(
        device=normalized_xy.device, dtype=torch.float32
    )


def _append_gaussian_seeds(
    gaussians: Any,
    seed_batches: Sequence[Any],
    *,
    device: torch.device,
    scaling_factor: float,
) -> int:
    if not seed_batches:
        raise RuntimeError(
            "MV-RoMa produced no geometrically valid points; inspect confidence, "
            "camera overlap, and reprojection thresholds"
        )
    bootstrap_gaussian_splatting()
    from utils.sh_utils import RGB2SH

    xyz_cpu = torch.cat([batch.xyz for batch in seed_batches])
    rgb_cpu = torch.cat([batch.rgb for batch in seed_batches])
    distance_cpu = torch.cat([batch.distance_to_source for batch in seed_batches])
    count = int(xyz_cpu.shape[0])
    xyz = xyz_cpu.to(device=device)
    rgb = rgb_cpu.to(device=device)
    distance = distance_cpu.to(device=device)

    feature_dc = RGB2SH(rgb).unsqueeze(1)
    feature_rest_shape = tuple(gaussians._features_rest.shape[1:])
    feature_rest = torch.zeros(
        (count, *feature_rest_shape),
        device=device,
        dtype=gaussians._features_rest.dtype,
    )
    # Preserve EDGS correspondence initialization exactly: valid seeds start
    # from opacity logit 0 (activated opacity 0.5), independently of the SfM
    # points' current opacity.
    opacity_shape = tuple(gaussians._opacity.shape[1:])
    opacity = torch.zeros(
        (count, *opacity_shape), device=device, dtype=gaussians._opacity.dtype
    )
    scales = gaussians.scaling_inverse_activation(
        (distance * scaling_factor).clamp_min(1e-8)[:, None].expand(-1, 3)
    )
    base_rotation = gaussians._rotation[-1].detach().to(device)
    rotation = base_rotation.expand(count, *base_rotation.shape).clone()
    radii = torch.zeros(count, device=device, dtype=xyz.dtype)
    gaussians.densification_postfix(
        xyz,
        feature_dc,
        feature_rest,
        opacity,
        scales,
        rotation,
        radii,
    )
    return count


def _planner_diagnostics(
    overlap: OverlapEstimate, group_plan: CameraGroupPlan
) -> dict[str, Any]:
    groups = group_plan.groups
    return {
        "name": "mvroma_overlap_aware",
        "overlap_provider": overlap.provider,
        "sparse_model_path": (
            None if overlap.sparse_model_path is None else str(overlap.sparse_model_path)
        ),
        "primary_groups": group_plan.primary_group_count,
        "reciprocal_groups": group_plan.reciprocal_group_count,
        "targets_per_group_min": min(len(group.target_indices) for group in groups),
        "targets_per_group_max": max(len(group.target_indices) for group in groups),
        "source_quota_min": min(group_plan.source_quotas),
        "source_quota_max": max(group_plan.source_quotas),
        "mean_source_overlap": group_plan.mean_primary_source_overlap,
        "minimum_source_overlap": group_plan.minimum_primary_source_overlap,
        "reciprocity_coverage": group_plan.reciprocity_coverage,
        "support_count_min": min(overlap.visible_point_counts),
        "support_count_max": max(overlap.visible_point_counts),
    }


def _run_legacy_groups(
    runtime: MVRoMaRuntime,
    prepared_groups: Sequence[Any],
    cameras: Sequence[Any],
    cfg: Any,
    mvroma_config: Any,
    diagnostics: dict[str, Any],
) -> list[_SeedBatch]:
    minimum_group_targets = min(
        len(prepared.group.cameras.target_indices) for prepared in prepared_groups
    )
    min_target_views = int(
        _get(mvroma_config, "min_target_views", min(2, minimum_group_targets))
    )
    confidence_threshold = float(_get(mvroma_config, "confidence_threshold", 0.5))
    matches_per_reference = int(_get(cfg, "matches_per_ref"))
    min_angle = float(_get(mvroma_config, "min_triangulation_angle_deg", 0.5))
    reprojection_tolerance = float(_get(cfg, "proj_err_tolerance"))
    seed = int(_get(mvroma_config, "seed", 0))
    seed_batches: list[_SeedBatch] = []

    for group_index, prepared in enumerate(
        tqdm(prepared_groups, desc="MV-RoMa legacy correspondence initialization")
    ):
        if prepared.paired_tracks.shape[1] == 0:
            diagnostics["skipped_empty_groups"] += 1
            continue
        dense = runtime.predict_dense(prepared)
        generator = torch.Generator(device=dense.target_coordinates.device)
        generator.manual_seed(seed + prepared.group.cameras.source_index)
        tracks = dense.sample_tracks(
            matches_per_reference,
            confidence_threshold=confidence_threshold,
            min_target_views=min_target_views,
            generator=generator,
        )
        sampled = int(tracks.coordinates.shape[0])
        diagnostics["sampled_tracks"] += sampled
        if sampled == 0:
            continue
        camera_indices = prepared.group.cameras.all_indices
        projection_matrices = torch.stack(
            [cameras[index].full_proj_transform for index in camera_indices]
        ).to(device=tracks.coordinates.device, dtype=tracks.coordinates.dtype)
        camera_centers = torch.stack(
            [cameras[index].camera_center for index in camera_indices]
        ).to(device=tracks.coordinates.device, dtype=tracks.coordinates.dtype)
        triangulated = weighted_multiview_dlt(
            projection_matrices,
            tracks,
            camera_centers=camera_centers,
            min_views=min_target_views + 1,
            max_reprojection_error=reprojection_tolerance,
            min_triangulation_angle_deg=min_angle,
            reject_outliers=True,
        )
        accepted = triangulated.accepted
        accepted_count = int(accepted.sum().item())
        diagnostics["accepted_before_dedup"] += accepted_count
        if accepted_count:
            xyz = triangulated.points[accepted]
            source_xy = tracks.coordinates[accepted, 0]
            rgb = _sample_source_colors(
                cameras[prepared.group.cameras.source_index], source_xy
            )
            distance = torch.linalg.vector_norm(xyz - camera_centers[0], dim=1)
            errors = triangulated.reprojection_error.masked_fill(
                ~triangulated.valid_observations, float("-inf")
            ).amax(dim=1)
            seed_batches.append(
                _SeedBatch(
                    xyz=xyz.detach().cpu(),
                    rgb=rgb.detach().cpu(),
                    distance_to_source=distance.detach().cpu(),
                    reprojection_error=errors[accepted].detach().cpu(),
                )
            )
        diagnostics["per_group"].append(
            {
                "group": group_index,
                "source_index": prepared.group.cameras.source_index,
                "sampled": sampled,
                "accepted": accepted_count,
            }
        )
        del dense, tracks, triangulated
    return seed_batches


def init_gaussians_with_mvroma(
    gaussians: Any,
    scene: Any,
    cfg: Any,
    device: torch.device | str,
    verbose: bool = False,
    model: torch.nn.Module | None = None,
    prematcher: Any = None,
) -> MVRoMaInitializationResult:
    """Initialize EDGS with configurable legacy or paper MV-RoMa processing."""

    torch_device = torch.device(device)
    mvroma_config = _nested(cfg, "mvroma")
    runtime_settings = MVRoMaSettings.from_config(mvroma_config)
    postprocess = PaperPostprocessSettings.from_config(
        mvroma_config,
        legacy_matches_per_ref=int(_get(cfg, "matches_per_ref")),
    )
    MVRoMaTrainingSettings.from_config(mvroma_config)  # validate before mutation
    cameras = list(scene.getTrainCameras().copy())
    if len(cameras) < 2:
        raise ValueError("MV-RoMa requires at least two training cameras")
    scaling_factor = float(_get(cfg, "scaling_factor"))
    triangulation_config = _get(mvroma_config, "triangulation", {})
    reprojection_tolerance = float(
        _get(
            triangulation_config,
            "max_reprojection_error",
            _get(cfg, "proj_err_tolerance"),
        )
    )
    min_angle = float(
        _get(
            triangulation_config,
            "min_angle_deg",
            _get(mvroma_config, "min_triangulation_angle_deg", 0.5),
        )
    )
    reject_outliers = as_bool(
        _get(triangulation_config, "reject_outliers", True),
        name="triangulation.reject_outliers",
    )

    runtime_kwargs: dict[str, Any] = {}
    if prematcher is not None:
        runtime_kwargs["prematcher_factory"] = lambda _name, _device: prematcher
    if model is not None:
        runtime_kwargs["model_factory"] = lambda _settings, _device: model

    planner_config = _get(mvroma_config, "group_planner", {})
    provider = str(_get(planner_config, "overlap_provider", "colmap")).lower()
    overlap: OverlapEstimate
    group_plan: CameraGroupPlan
    legacy_seed_batches: list[_SeedBatch] = []
    pair_store: PairFieldStore | None = None
    input_xyz = getattr(gaussians, "_xyz", None)
    input_sfm_points = (
        int(input_xyz.shape[0])
        if input_xyz is not None and hasattr(input_xyz, "shape")
        else None
    )
    diagnostics: dict[str, Any] = {
        "backend": "mvroma",
        "postprocess_mode": postprocess.mode,
        "input_sfm_points": input_sfm_points,
        "keep_sfm_points": as_bool(
            _get(cfg, "add_SfM_init", False), name="add_SfM_init"
        ),
        "camera_groups": 0,
        "prepared_groups": 0,
        "predicted_groups": 0,
        "sampled_tracks": 0,
        "accepted_before_dedup": 0,
        "accepted_points": 0,
        "skipped_empty_groups": 0,
        "per_group": [],
    }

    with _camera_image_paths(cameras, mvroma_config) as image_paths:
        with MVRoMaRuntime(
            runtime_settings, torch_device, **runtime_kwargs
        ) as runtime:
            overlap_override = None
            if provider == "ufm_visibility":
                matrix = runtime.estimate_visibility_overlap(
                    image_paths,
                    confidence_threshold=float(
                        _get(planner_config, "visibility_confidence_threshold", 0.3)
                    ),
                    batch_size=int(_get(planner_config, "visibility_batch_size", 4)),
                )
                overlap_override = build_matcher_visibility_overlap(
                    matrix,
                    [str(getattr(camera, "image_name", index)) for index, camera in enumerate(cameras)],
                )
            overlap, group_plan = _build_camera_group_plan(
                cameras,
                cfg,
                mvroma_config,
                overlap=overlap_override,
            )
            groups = list(group_plan.groups)
            if not groups:
                raise RuntimeError("no camera groups were selected for MV-RoMa")
            directed_predictions = sum(len(group.target_indices) for group in groups)
            diagnostics["planned_directed_predictions"] = directed_predictions
            if directed_predictions > 5000:
                print(
                    "Warning: MV-RoMa planned "
                    f"{len(groups)} groups / {directed_predictions} directed "
                    "UFM predictions. This is an expensive large-scene setup; "
                    "use the coverage profile or an explicit fixed group budget "
                    "when full-budget ablation is not required."
                )
            minimum_group_targets = min(len(group.target_indices) for group in groups)
            if postprocess.min_target_views > minimum_group_targets:
                raise ValueError(
                    "postprocess.min_target_views exceeds the smallest camera group"
                )
            diagnostics["camera_groups"] = len(groups)
            diagnostics["group_planner"] = _planner_diagnostics(overlap, group_plan)
            planner_diagnostics = diagnostics["group_planner"]
            print(
                "MV-RoMa overlap-aware planner: "
                f"primary={planner_diagnostics['primary_groups']}, "
                f"reciprocal={planner_diagnostics['reciprocal_groups']}, "
                f"targets={planner_diagnostics['targets_per_group_min']}.."
                f"{planner_diagnostics['targets_per_group_max']}, "
                f"mean overlap={planner_diagnostics['mean_source_overlap']:.3f}, "
                f"reciprocity={planner_diagnostics['reciprocity_coverage']:.1%}"
            )
            inference_groups = _make_inference_groups(image_paths, groups)
            prepared_groups = runtime.prepare_tracks(inference_groups)
            diagnostics["prepared_groups"] = len(prepared_groups)

            if postprocess.mode == "legacy":
                legacy_seed_batches = _run_legacy_groups(
                    runtime,
                    prepared_groups,
                    cameras,
                    cfg,
                    mvroma_config,
                    diagnostics,
                )
            else:
                unique_pairs = {
                    DirectedPair(group.source_index, target)
                    for group in groups
                    for target in group.target_indices
                }
                pair_store = PairFieldStore(
                    postprocess.pair_store,
                    expected_pair_count=len(unique_pairs),
                )
                for group_index, prepared in enumerate(
                    tqdm(prepared_groups, desc="MV-RoMa dense pair prediction")
                ):
                    if prepared.paired_tracks.shape[1] == 0:
                        diagnostics["skipped_empty_groups"] += 1
                        continue
                    dense = runtime.predict_dense(prepared)
                    merge_dense_group(pair_store, prepared.group.cameras, dense)
                    diagnostics["predicted_groups"] += 1
                    diagnostics["per_group"].append(
                        {
                            "group": group_index,
                            "source_index": prepared.group.cameras.source_index,
                            "dense": True,
                        }
                    )
                    del dense

    neighbors = group_plan.neighbor_table()
    pair_quality = {}
    artifact_id = None
    if postprocess.mode == "legacy":
        appended = _append_gaussian_seeds(
            gaussians,
            legacy_seed_batches,
            device=torch_device,
            scaling_factor=scaling_factor,
        )
    else:
        assert pair_store is not None
        try:
            diagnostics["pair_store_mode"] = pair_store.mode
            pair_quality = apply_reciprocal_cycle_filter(
                pair_store,
                max_error_px=postprocess.cycle_threshold_px,
                device=torch_device,
            )
            diagnostics["cycle_valid_pairs"] = sum(
                quality.valid_pixels > 0 for quality in pair_quality.values()
            )
            candidates_by_source: dict[int, list[Any]] = {
                index: [] for index in range(len(cameras))
            }
            candidate_counts = [0 for _ in cameras]
            for group_index, group in enumerate(group_plan.groups):
                candidates = build_group_track_candidates(
                    pair_store,
                    group,
                    group_index=group_index,
                    confidence_threshold=postprocess.confidence_threshold,
                    min_target_views=postprocess.min_target_views,
                    nms_radius_px=postprocess.nms_radius_px,
                    sampling_strategy=postprocess.sampling_strategy,
                    sampling_grid_size_px=postprocess.sampling_grid_size_px,
                    visibility_score_weight=postprocess.visibility_score_weight,
                    max_points=postprocess.seed_budget.per_source,
                )
                if candidates is not None:
                    candidates_by_source[group.source_index].append(candidates)
                    diagnostics["sampled_tracks"] += int(
                        candidates.tracks.coordinates.shape[0]
                    )
                    candidate_counts[group.source_index] += int(
                        candidates.tracks.coordinates.shape[0]
                    )

            source_batches: list[GaussianSeedBatch] = []
            accepted_by_source = [0 for _ in cameras]
            for source_index in tqdm(
                range(len(cameras)), desc="MV-RoMa multi-view triangulation"
            ):
                batch = triangulate_source_candidates(
                    candidates_by_source[source_index],
                    cameras,
                    postprocess.seed_budget,
                    device=torch_device,
                    min_target_views=postprocess.min_target_views,
                    max_reprojection_error=reprojection_tolerance,
                    min_triangulation_angle_deg=min_angle,
                    reject_outliers=reject_outliers,
                    triangulation_batch_size=postprocess.triangulation_batch_size,
                )
                if batch is not None:
                    source_batches.append(batch)
                    accepted_by_source[source_index] = int(batch.xyz.shape[0])
            if not source_batches:
                raise RuntimeError(
                    "MV-RoMa paper postprocessing produced no valid Gaussian seeds"
                )
            diagnostics["accepted_before_dedup"] = sum(
                batch.xyz.shape[0] for batch in source_batches
            )
            diagnostics["candidate_tracks_by_source"] = candidate_counts
            diagnostics["accepted_points_by_source"] = accepted_by_source
            accepted_array = np.asarray(accepted_by_source, dtype=np.int64)
            active_sources = accepted_array > 0
            diagnostics["source_coverage"] = {
                "camera_count": len(cameras),
                "candidate_source_count": sum(
                    count > 0 for count in candidate_counts
                ),
                "accepted_source_count": int(active_sources.sum()),
                "accepted_per_active_source_min": (
                    int(accepted_array[active_sources].min())
                    if active_sources.any()
                    else 0
                ),
                "accepted_per_active_source_median": (
                    float(np.median(accepted_array[active_sources]))
                    if active_sources.any()
                    else 0.0
                ),
                "accepted_per_active_source_max": (
                    int(accepted_array[active_sources].max())
                    if active_sources.any()
                    else 0
                ),
            }
            coverage = diagnostics["source_coverage"]
            print(
                "MV-RoMa source coverage: "
                f"candidates={coverage['candidate_source_count']}/"
                f"{coverage['camera_count']}, accepted="
                f"{coverage['accepted_source_count']}/{coverage['camera_count']}, "
                "accepted/source="
                f"{coverage['accepted_per_active_source_min']}/"
                f"{coverage['accepted_per_active_source_median']:.0f}/"
                f"{coverage['accepted_per_active_source_max']} (min/median/max)"
            )
            fused, voxel_size = deduplicate_seed_batches(
                source_batches,
                postprocess.dedup,
                scaling_factor=scaling_factor,
                total_limit=postprocess.seed_budget.total,
            )
            diagnostics["dedup_voxel_size"] = voxel_size
            appended = _append_gaussian_seeds(
                gaussians,
                [fused],
                device=torch_device,
                scaling_factor=scaling_factor,
            )
        finally:
            pair_store.close()

    diagnostics["accepted_points"] = appended
    model_path = getattr(scene, "model_path", None)
    if model_path is not None:
        artifact_id = write_mvroma_artifacts(
            model_path,
            mvroma_config=mvroma_config,
            cameras=cameras,
            overlap_matrix=overlap.matrix,
            group_plan=group_plan,
            pair_quality=pair_quality,
            diagnostics=diagnostics,
        )
        diagnostics["artifact_id"] = artifact_id
    if verbose:
        print(
            "MV-RoMa initialization: "
            f"sampled={diagnostics['sampled_tracks']}, accepted={appended}, "
            f"groups={diagnostics['camera_groups']}, mode={postprocess.mode}"
        )
    return MVRoMaInitializationResult(
        cameras=cameras,
        legacy_neighbors=neighbors,
        diagnostics=diagnostics,
        overlap_matrix=overlap.matrix,
        pair_quality=pair_quality,
        artifact_id=artifact_id,
    )
