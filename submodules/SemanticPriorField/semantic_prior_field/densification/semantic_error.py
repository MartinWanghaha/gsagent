"""Semantic-error driven density control.

Mirrors the normal-error machinery of ``densification/normal_error.py``:
per-pixel errors are scattered onto the argmax-contributor Gaussian of each
pixel (``render_depth``'s ``gidx``) and normalized by pixel count or
projected splat area. The error signal here is the semantic cross-entropy
between the rendered, classified embedding field and the associated masks.

Gaussians with high semantic error are either straddling an instance
boundary (their embedding is pulled toward two classes) or floaters whose
identity is inconsistent across views — exactly the Gaussians whose
splitting sharpens both RGB edges and mesh boundaries.
"""

import math
from typing import Callable, List

import torch
import torch.nn.functional as F

from arguments import PipelineParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_depth, render_simp
from utils.general_utils import build_rotation


@torch.no_grad()
def compute_semantic_error(
    gaussians: GaussianModel,
    cameras: List[Camera],
    render_func: Callable,
    pipe: PipelineParams,
    semantic_head,
    observations,
    background: torch.Tensor = torch.zeros(3, device="cuda"),
    method: str = "count",  # "count" or "area" or "none"
) -> torch.Tensor:
    """Per-Gaussian average semantic cross-entropy over all training views.

    Returns:
        torch.Tensor: (N_gaussians,) mean semantic error.
    """
    assert method in ["count", "area", "none"], "Invalid method"

    gaussian_errors = torch.zeros_like(gaussians._xyz[:, 0])
    log_num_classes = max(
        torch.log(torch.tensor(float(semantic_head.num_classes))).item(), 1.0
    )

    for i_img in range(len(cameras)):
        # Get the argmax-contributor Gaussian index per pixel
        msv2_render_pkg = render_depth(
            viewpoint_camera=cameras[i_img],
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            culling=None,
        )
        msv2_idx = msv2_render_pkg["gidx"]

        if method == "area":
            msv2_render_pkg_simp = render_simp(
                viewpoint_camera=cameras[i_img],
                pc=gaussians,
                pipe=pipe,
                bg_color=background,
                culling=None,
            )
            gaussians_proj_area = msv2_render_pkg_simp["area_proj"]  # (N_gaussians,)

        # Render the semantic embedding field and classify it. The (C, H, W)
        # logits volume can reach gigabytes for scene class counts in the
        # hundreds, so classification and CE run over pixel chunks.
        render_pkg = render_func(
            viewpoint_camera=cameras[i_img],
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            require_coord=False,
            require_depth=False,
            render_semantics=True,
        )
        semantic_features = render_pkg["semantic_features"]  # (16, H, W)
        height, width = semantic_features.shape[-2:]
        features_flat = semantic_features.reshape(semantic_features.shape[0], -1).t()  # (P, 16)

        observation = observations.load(
            cameras[i_img].image_name,
            height,
            width,
        ).to(features_flat.device)
        labels_flat = observation.labels.flatten()  # (P,)

        head_weight = semantic_head.classifier.weight.view(
            semantic_head.num_classes, semantic_head.semantic_dim
        )
        head_bias = semantic_head.classifier.bias
        per_pixel_error = torch.zeros_like(labels_flat, dtype=torch.float32)
        chunk_size = 65_536
        for start in range(0, features_flat.shape[0], chunk_size):
            end = min(start + chunk_size, features_flat.shape[0])
            chunk_logits = features_flat[start:end] @ head_weight.t() + head_bias  # (c, C)
            per_pixel_error[start:end] = F.cross_entropy(
                chunk_logits,
                labels_flat[start:end],
                ignore_index=observations.ignore_label,
                reduction="none",
            )
        per_pixel_error = per_pixel_error.view(height, width)
        per_pixel_error = per_pixel_error * observation.confidence / log_num_classes

        # Scatter the per-pixel error onto the argmax-contributor Gaussians
        gaussian_errors_i = torch.zeros_like(gaussian_errors)
        gaussian_errors_i.index_add_(0, msv2_idx.flatten(), per_pixel_error.flatten())

        if method == "count":
            gaussian_count = torch.zeros_like(gaussian_errors)
            gaussian_count.index_add_(
                0, msv2_idx.flatten(),
                torch.ones_like(per_pixel_error.flatten()),
            )
            valid_mask = gaussian_count > 0
            gaussian_errors_i = torch.where(
                valid_mask, gaussian_errors_i / gaussian_count, torch.zeros_like(gaussian_errors_i)
            )
        elif method == "area":
            valid_area_mask = gaussians_proj_area > 0
            gaussian_errors_i = torch.where(
                valid_area_mask,
                gaussian_errors_i / gaussians_proj_area,
                torch.zeros_like(gaussian_errors_i),
            )

        gaussian_errors = gaussian_errors + gaussian_errors_i

    return gaussian_errors / len(cameras)


@torch.no_grad()
def densify_semantic_boundary(
    gaussians: GaussianModel,
    cameras: List[Camera],
    render_func: Callable,
    pipe: PipelineParams,
    background: torch.Tensor,
    semantic_head,
    observations,
    config,
    args,
    precomputed_errors=None,
):
    """Split the highest-semantic-error Gaussians along their dominant tangent.

    A Gaussian whose footprint covers two instances receives gradient from
    both classes and keeps a high semantic error. Splitting it tangentially
    (never along its normal) lets the two children specialize on either side
    of the boundary, sharpening the RGB edge and the mesh boundary at once.
    The split preserves volume by halving the scale along the split axis,
    and edits the originals in place to keep Adam moments, following the
    normal-field spoke-splitting convention.
    """
    if (
        precomputed_errors is not None
        and precomputed_errors.shape[0] == gaussians._xyz.shape[0]
    ):
        # Per-iteration conflict statistics from the SPF rasterizer backward:
        # contribution-weighted and accumulated continuously, so no camera
        # sweep is needed.
        semantic_errors = precomputed_errors
    else:
        semantic_errors = compute_semantic_error(
            gaussians=gaussians,
            cameras=cameras,
            render_func=render_func,
            pipe=pipe,
            semantic_head=semantic_head,
            observations=observations,
            background=background,
            method=config["densification_normalization_method"],
        )  # (N_gaussians,)

    errors_quantile = torch.quantile(
        semantic_errors, q=1.0 - config["densification_semantic_errors_quantile"]
    )
    densification_mask = semantic_errors > errors_quantile  # (N_gaussians,)

    # Respect the global Gaussian cap shared with normal-field densification
    if getattr(args, "N_max_gaussians", None) is not None:
        n_current = gaussians._xyz.shape[0]
        n_allowed = args.N_max_gaussians - n_current
        if n_allowed <= 0:
            print("[WARNING] Maximum Number of Gaussians reached. Skipping Semantic Densification.")
            return
        n_selected = densification_mask.sum().item()
        if n_selected > n_allowed:
            candidate_indices = densification_mask.nonzero(as_tuple=True)[0]
            top_indices = candidate_indices[
                semantic_errors[candidate_indices].topk(n_allowed).indices
            ]
            densification_mask = torch.zeros_like(densification_mask)
            densification_mask[top_indices] = True
            print(f"[WARNING] Capping the number of gaussians to {args.N_max_gaussians}.")

    if not densification_mask.any():
        return

    selected = densification_mask.nonzero(as_tuple=True)[0]
    scaling = gaussians.get_scaling[selected]  # (M, 3) activated
    rotation_matrices = build_rotation(gaussians._rotation[selected])  # (M, 3, 3)

    # Choose the dominant tangential axis: the largest-scale local axis,
    # penalized by its alignment with the learned surface normal so the
    # split never happens across the surface.
    if gaussians.use_gaussian_features:
        normals = gaussians.convert_features_to_normals(normalize=True)[selected]  # (M, 3)
        alignment = torch.einsum("mij,mi->mj", rotation_matrices, normals).abs()  # (M, 3)
        axis_scores = scaling * (1.0 - alignment ** 2)
    else:
        axis_scores = scaling
    axis_idx = axis_scores.argmax(dim=-1)  # (M,)

    batch = torch.arange(selected.shape[0], device=selected.device)
    axis_dir = rotation_matrices[batch, :, axis_idx]  # (M, 3)
    axis_sigma = scaling[batch, axis_idx].unsqueeze(-1)  # (M, 1)
    offsets = 0.5 * axis_sigma * axis_dir  # (M, 3)

    # Children positions: original moves +offset, clone gets -offset
    new_xyz = gaussians._xyz[selected] - offsets

    # Halve the extent along the split axis for both children (in place for
    # the originals, inherited by the clones through the mask copy)
    gaussians._scaling[selected, axis_idx] = (
        gaussians._scaling[selected, axis_idx] + math.log(0.5)
    )
    gaussians._xyz[selected] = gaussians._xyz[selected] + offsets

    gaussians.densify_and_clone_from_mask(
        selected_pts_mask=densification_mask,
        new_xyz=new_xyz,
    )


@torch.no_grad()
def prune_identity_unstable_gaussians(
    gaussians: GaussianModel,
    prior_field,
    cameras: List[Camera],
    pipe: PipelineParams,
    background: torch.Tensor,
    confidence_threshold: float = 0.1,
) -> int:
    """Prune Gaussians that are both identity-unstable and non-maximal.

    Floaters receive inconsistent mask supervision across views, so their
    label posterior stays flat (low top1-top2 margin). Requiring them to
    also never be the argmax contributor of any pixel keeps the gate
    conservative: real surface Gaussians win at least one pixel somewhere.

    Returns:
        Number of pruned Gaussians.
    """
    if not prior_field.valid or prior_field.label_confidence is None:
        return 0
    if prior_field.label_confidence.shape[0] != gaussians._xyz.shape[0]:
        return 0

    unstable = prior_field.label_confidence < confidence_threshold

    is_maximal = torch.zeros(
        gaussians._xyz.shape[0], dtype=torch.bool, device=gaussians._xyz.device
    )
    for i_cam in range(len(cameras)):
        render_pkg = render_depth(
            viewpoint_camera=cameras[i_cam],
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            culling=None,
        )
        is_maximal[render_pkg["gidx"].unique()] = True

    prune_mask = unstable & ~is_maximal
    n_pruned = int(prune_mask.sum().item())
    if n_pruned > 0:
        gaussians.prune_points(prune_mask)
    return n_pruned
