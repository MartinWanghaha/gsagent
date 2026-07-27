import os
import sys
import gc
import yaml
from functools import partial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
SUBMODULES_DIR = os.path.join(ROOT_DIR, 'submodules')
sys.path.append(ROOT_DIR)
sys.path.append(SUBMODULES_DIR)
sys.path.append(os.path.join(SUBMODULES_DIR, 'Depth-Anything-V2'))

import torch
from random import randint
from utils.loss_utils import l1_loss, L1_loss_appearance, get_img_grad_weight
from fused_ssim import fused_ssim

from gaussian_renderer import network_gui
from gaussian_renderer import render_imp
from scene import Scene, GaussianModel
from semantic import (
    GagaObservationStore,
    SemanticHead,
    save_semantic_checkpoint,
    semantic_cross_entropy,
    spatial_consistency_loss,
)
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

import numpy as np
import time

from utils.geometry_utils import depth_to_normal, depth_to_normal_with_mask
from utils.log_utils import log_normal_field_training_progress
from regularization.regularizer.normal_field import (
    initialize_normal_field,
    compute_normal_field_regularization,
    reset_normal_field_state_at_next_iteration,
    densify_normal_field,
    prune_non_maximal_gaussians,
)
from regularization.regularizer.depth_order import (
    initialize_depth_order_supervision,
    compute_depth_order_regularization,
)
from regularization.regularizer.multiview import (
    initialize_multiview_regularization,
    compute_multiview_regularization,
)
from regularization.regularizer.mesh_in_the_loop import (
    initialize_mesh_in_the_loop_regularization,
    compute_mesh_in_the_loop_regularization,
    reset_milo_state_at_next_iteration,
)
from regularization.regularizer.semantic_prior import (
    initialize_semantic_prior,
    compute_semantic_prior_regularization,
    reset_semantic_prior_state_at_next_iteration,
    get_boundary_weight_map,
    maybe_refresh_prior_field,
)
from densification.semantic_error import (
    densify_semantic_boundary,
    prune_identity_unstable_gaussians,
)
from utils.diagnostics import TrainingDiagnostics
from semantic.scatter import SemanticStatsAccumulator


def training(
    dataset, opt, pipe,
    testing_iterations, saving_iterations,
    checkpoint_iterations, checkpoint,
    debug_from, args,
    depth_order_config, normal_field_config, multiview_config, milo_config,
    semantic_prior_config,
    log_interval,
):
    # ---Prepare logger--- 
    run = prepare_output_and_logger(dataset, args)
    
    # ---Initialize scene and Gaussians---
    first_iter = 0
    use_mip_filter = not args.disable_mip_filter
    semantic_enabled = args.semantic_masks is not None
    
    if args.use_normal_field:
        if normal_field_config["use_smallest_axis"]:
            n_gaussian_features = 1
        else:
            n_gaussian_features = 4
    else:
        n_gaussian_features = 0
    
    n_pivots_per_gaussian = milo_config["n_pivots_per_gaussian"] if args.milo else 2
    
    gaussians = GaussianModel(
        sh_degree=dataset.sh_degree, 
        use_mip_filter=use_mip_filter, 
        learn_occupancy=True if args.milo else False,
        use_appearance_network=args.decoupled_appearance,
        n_gaussian_features=n_gaussian_features,
        n_pivots_per_gaussian=n_pivots_per_gaussian,
        use_radegs_densification=True,
        use_unbounded_opacity=dataset.use_unbounded_opacity,
        use_exposure_compensation=args.exposure_compensation,
        semantic_dim=16 if semantic_enabled else 0,
    )
    scene = Scene(dataset, gaussians)
    
    if args.exposure_compensation:
        n_cameras = len(scene.getTrainCameras().copy())
        print(f"[INFO] Using exposure compensation for {n_cameras} cameras.")
        gaussians.initialize_exposure_compensation(num_cameras=n_cameras)
    
    gaussians.training_setup(opt)
    print(f"[INFO] Using 3D Mip Filter: {gaussians.use_mip_filter}")
    print(f"[INFO] Using learnable SDF: {gaussians.learn_occupancy}")

    semantic_head = None
    semantic_head_optimizer = None
    semantic_observations = None
    semantic_num_classes = None
    if semantic_enabled:
        semantic_observations = GagaObservationStore(
            args.semantic_masks,
            require_all=not args.allow_missing_semantic_masks,
        )
        semantic_num_classes = (
            args.semantic_num_classes
            or semantic_observations.validate_cameras(scene.getTrainCameras())
        )
        semantic_head = SemanticHead(16, semantic_num_classes).cuda()
        semantic_head_optimizer = torch.optim.Adam(
            semantic_head.parameters(),
            lr=args.semantic_head_lr,
        )
        print(
            f"[INFO] Joint Gaga semantics enabled: {semantic_num_classes} classes, "
            f"weight={args.lambda_semantic}."
        )

    # ---Prepare semantic stats channel (SPF rasterizer) and loss balance---
    semantic_stats = None
    if semantic_enabled and args.sp_stats and args.rasterizer == "ours":
        semantic_stats = SemanticStatsAccumulator()
        print("[INFO] Using the SPF rasterizer stats channel (per-Gaussian "
              "conflict/contribution accumulation in the semantic backward).")
    semantic_log_sigma = None
    if semantic_enabled and args.balance_semantic:
        semantic_log_sigma = torch.zeros(1, device="cuda", requires_grad=True)
        semantic_head_optimizer.add_param_group(
            {"params": [semantic_log_sigma], "lr": 1e-3}
        )
        print("[INFO] Using uncertainty-based semantic loss balancing.")

    # ---Prepare Semantic Prior Field---
    semantic_prior_enabled = args.semantic_prior
    if semantic_prior_enabled and not semantic_enabled:
        raise ValueError(
            "--semantic_prior requires --semantic_masks: the prior field is "
            "derived from the jointly trained semantic embedding."
        )
    if semantic_prior_enabled:
        semantic_prior_state = initialize_semantic_prior(
            scene=scene,
            config=semantic_prior_config,
            observations=semantic_observations,
        )
        print("[INFO] Using Semantic Prior Field.")
        print(f"        > Refresh from iteration {semantic_prior_config['start_iter']} every {semantic_prior_config['refresh_interval']}.")
        print(f"        > Channels: orient={args.sp_orient}, flatten={args.sp_flatten}, sh={args.sp_sh}, "
              f"split={args.sp_split}, prune={args.sp_prune}, budget={args.sp_budget}, boundary={args.sp_boundary}.")

    # ---Prepare diagnostics---
    diagnostics = None
    if args.diagnostics:
        diagnostics = TrainingDiagnostics(
            scene.model_path,
            scalar_interval=args.diag_scalar_interval,
            image_interval=args.diag_image_interval,
            snapshot_interval=args.diag_snapshot_interval,
        )

    if args.use_normal_field:
        print(f"[INFO] Using {n_gaussian_features} learnable Gaussian features.")

    if args.dense_gaussians:
        print("[INFO] Using dense Gaussians.")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    if semantic_enabled and args.semantic_checkpoint:
        semantic_payload = torch.load(args.semantic_checkpoint, map_location="cpu")
        if int(semantic_payload.get("semantic_dim", -1)) != 16:
            raise ValueError("Semantic checkpoint must contain 16D features")
        saved_features = semantic_payload["semantic_features"].to(
            gaussians.get_semantic_features.device
        )
        if saved_features.shape != gaussians.get_semantic_features.shape:
            raise ValueError(
                "Semantic checkpoint Gaussian count does not match the active model: "
                f"{tuple(saved_features.shape)} vs "
                f"{tuple(gaussians.get_semantic_features.shape)}"
            )
        gaussians.get_semantic_features.data.copy_(saved_features)
        semantic_head.load_state_dict(semantic_payload["head"])
        semantic_optimizer_state = semantic_payload.get("optimizer")
        if isinstance(semantic_optimizer_state, dict):
            head_state = semantic_optimizer_state.get("head")
            if head_state is not None:
                semantic_head_optimizer.load_state_dict(head_state)
        print(f"[INFO] Restored Gaga semantic state from {args.semantic_checkpoint}")
        if args.use_normal_field:
            if first_iter > normal_field_config["start_iter"]:
                normal_field_config["start_iter"] = first_iter + 1
        if args.milo:
            if first_iter > milo_config["start_iter"]:
                milo_config["start_iter"] = first_iter + 1
        if args.multiview:
            if first_iter > multiview_config["start_multiview"]:
                multiview_config["start_multiview"] = first_iter + 1
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize culling stats
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    gaussians.init_culling(len(scene.getTrainCameras().copy()))
    
    # Initialize 3D Mip filter
    if use_mip_filter:
        gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())

    # Additional variables
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    all_viewpoints = None
    postfix_dict = {}
    ema_loss_for_log = 0.0
    ema_depth_normal_loss_for_log = 0.0
    
    # ---Prepare Normal Field Optimization---
    if args.use_normal_field:
        print("[INFO] Using Normal Field.")
        normal_field_state = initialize_normal_field(
            scene=scene,
            config=normal_field_config,
        )
        if normal_field_config["reset_normals_after_densification"]:
            print(f"[INFO] Normal features will be reset after densification.")
            print(f"        > Resetting normal directions: {normal_field_config['reset_normal_directions']}.")
            print(f"        > Resetting normal signs: {normal_field_config['reset_normal_signs']}.")
    ema_normal_field_alignment_loss_for_log = 0.0
    ema_front_pivots_visibility_loss_for_log = 0.0
    ema_back_pivots_occlusion_loss_for_log = 0.0
    ema_gaussian_flattening_loss_for_log = 0.0
    ema_sdf_and_normal_field_consistency_loss_for_log = 0.0
    
    # ---Prepare Multiview Regularization---
    if args.multiview:
        print("[INFO] Using multiview regularization.")
        multiview_state = initialize_multiview_regularization(
            scene=scene,
            pipe=pipe,
            kernel_size=0.0,
            multiview_config=multiview_config,
        )
    ema_multiview_loss_for_log = 0.0
    
    # ---Prepare Mesh-In-the-Loop Regularization---
    if args.milo:
        print(f"[INFO] Using mesh-in-the-loop regularization with {n_pivots_per_gaussian} pivots per Gaussian.")
        milo_state = initialize_mesh_in_the_loop_regularization(
            scene=scene,
            gaussians=gaussians,
            milo_config=milo_config,
        )
    ema_mesh_depth_loss_for_log = 0.0
    ema_mesh_normal_loss_for_log = 0.0
    ema_occupied_centers_loss_for_log = 0.0

    # ---Prepare Depth-Order Regularization---    
    if args.depth_order:
        print("[INFO] Using depth order regularization.")
        print(f"        > Using expected depth with depth_ratio {depth_order_config['depth_ratio']} for depth order regularization.")
        if depth_order_config["deactivate_depth_order_after"] > -1:
            print(f"        > Deactivating at iteration {depth_order_config['deactivate_depth_order_after']}.")
        depth_priors = initialize_depth_order_supervision(
            scene=scene,
            config=depth_order_config,
            device='cuda',
        )
    ema_depth_order_loss_for_log = 0.0
        
    # ---Log optimizable param groups---
    print(f"[INFO] Found {len(gaussians.optimizer.param_groups)} optimizable param groups:")
    n_total_params = 0
    for param in gaussians.optimizer.param_groups:
        name = param['name']
        n_params = len(param['params'])
        print(f"\n========== {name} ==========")
        print(f"Learning rate: {param['lr']}")
        print(f"Total number of param groups: {n_params}")
        for param_i in param['params']:
            print(f"   > Shape {param_i.shape}")
            n_total_params = n_total_params + param_i.numel()
    if gaussians.learn_occupancy:
        print(f"\n========== base_occupancy ==========")
        print(f"   > Not learnable")
        print(f"   > Shape {gaussians._base_occupancy.shape}")
    print(f"\nTotal number of optimizable parameters: {n_total_params}\n")
    
    # ---Start optimization loop---    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):   

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_imp(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # ---Update SH degree---
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ---Select random viewpoint---
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_idx_stack = list(range(len(viewpoint_stack)))
            all_viewpoints = scene.getTrainCameras().copy()

        _random_view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_idx = viewpoint_idx_stack.pop(_random_view_idx)
        viewpoint_cam = viewpoint_stack.pop(_random_view_idx)

        # ---Render scene---
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
            
        reg_kick_on = iteration >= args.regularization_from_iter
        normal_field_kick_on = args.use_normal_field and (iteration >= normal_field_config["start_iter"])
        depth_order_kick_on = args.depth_order
        if args.depth_order and depth_order_config["deactivate_depth_order_after"] > -1:
            if iteration == depth_order_config["deactivate_depth_order_after"]:
                print(f"[INFO] Deactivating depth order regularization at iteration {iteration}.")
            depth_order_kick_on = depth_order_kick_on and (iteration < depth_order_config["deactivate_depth_order_after"])
        multiview_kick_on = args.multiview and (iteration >= multiview_config["start_multiview"])
        milo_kick_on = args.milo and (iteration >= milo_config["start_iter"])
        semantic_prior_kick_on = semantic_prior_enabled and (
            iteration >= semantic_prior_config["start_iter"]
        )

        # Initialize the learnable normal field before any consumer builds an
        # autograd graph from it.  In particular, the SPF orientation prior
        # starts on the same iteration by default; resetting the features
        # inside the later normal-field regularizer would mutate a tensor
        # already saved by its normalize backward.
        if (
            normal_field_kick_on
            and iteration == normal_field_config["start_iter"]
        ):
            print("[INFO] Initializing normal features")
            gaussians.reset_normal_features()

        # Boundary weighting needs only the (static) associated masks, so it
        # activates together with the losses it gates.
        boundary_weighting_active = semantic_prior_enabled and args.sp_boundary

        render_depth_in_forward_pass = (
            reg_kick_on 
            or normal_field_kick_on 
            or depth_order_kick_on
            or multiview_kick_on
            or milo_kick_on
        )
        
        # If depth-normal regularization or normal field regularization are active,
        # we use the rasterizer compatible with depth and normal rendering.
        semantic_stats_sink = {} if semantic_stats is not None else None
        render_kwargs = dict(
            require_coord=False,
            require_depth=render_depth_in_forward_pass,
            render_semantics=semantic_enabled,
        )
        if semantic_stats_sink is not None:
            render_kwargs["semantic_stats_sink"] = semantic_stats_sink
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg,
            **render_kwargs,
        )

        # ---Compute losses---
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        gt_image = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.gt_mask is not None:
            alpha_mask = viewpoint_cam.gt_mask.cuda()
            gt_image = gt_image * alpha_mask + bg.unsqueeze(-1).unsqueeze(-1) * (1.0 - alpha_mask)

        # Rendering loss
        if args.decoupled_appearance or args.exposure_compensation:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), padding="valid")
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        semantic_loss = None
        semantic_3d_loss = None
        if semantic_enabled:
            observation = semantic_observations.load(
                viewpoint_cam.image_name,
                viewpoint_cam.image_height,
                viewpoint_cam.image_width,
            ).to("cuda")
            semantic_logits = semantic_head(render_pkg["semantic_features"])
            semantic_loss = semantic_cross_entropy(
                semantic_logits,
                observation.labels,
                confidence=observation.confidence,
                ignore_index=semantic_observations.ignore_label,
            )
            if semantic_log_sigma is not None:
                # Kendall-style uncertainty weighting: the effective semantic
                # weight exp(-s) adapts to the task's noise level.
                loss = loss + (
                    torch.exp(-semantic_log_sigma).squeeze()
                    * args.lambda_semantic * semantic_loss
                    + 0.5 * semantic_log_sigma.squeeze()
                )
            else:
                loss = loss + args.lambda_semantic * semantic_loss
            if (
                args.lambda_semantic_3d > 0
                and iteration % args.semantic_3d_interval == 0
            ):
                semantic_3d_loss = spatial_consistency_loss(
                    gaussians.get_semantic_features,
                    gaussians.get_xyz,
                    sample_size=args.semantic_3d_samples,
                    neighbors=args.semantic_3d_neighbors,
                )
                loss = loss + args.lambda_semantic_3d * semantic_3d_loss
        
        # Depth-Normal Consistency Regularization
        if reg_kick_on:
            if args.mask_depth_normal:
                reg_depth_ratio = 0.6
                
                # Blending that respects the median depthmap validity mask
                depth_blend = torch.where(
                    render_pkg["median_depth"] > 0,  # (1, H, W)
                    (1. - reg_depth_ratio) * render_pkg["expected_depth"] + reg_depth_ratio * render_pkg["median_depth"],  # (1, H, W)
                    render_pkg["median_depth"],  # (1, H, W)
                )
                
                depth_normal, valid_points = depth_to_normal_with_mask(viewpoint_cam, depth_blend)  # (3, H, W), (H, W)
                normal_error_map = 1 - torch.linalg.vecdot(render_pkg["normal"], depth_normal, dim=0)  # (H, W)
                depth_normal_loss = torch.where(valid_points.squeeze(), normal_error_map, torch.zeros_like(normal_error_map))  # (H, W)
                depth_normal_loss = args.lambda_depth_normal * depth_normal_loss

            else:
                rendered_depth_to_normals: torch.Tensor = depth_to_normal(
                    viewpoint_cam, 
                    render_pkg["expected_depth"],  # 1, H, W
                    render_pkg["median_depth"],  # 1, H, W
                )  # 3, H, W or 2, 3, H, W
                rendered_normals: torch.Tensor = render_pkg["normal"]  # 3, H, W
            
                if rendered_depth_to_normals.ndim == 4:
                    # If shape is 2, 3, H, W
                    reg_depth_ratio = 0.6
                    normal_error_map = 1. - (rendered_normals[None] * rendered_depth_to_normals).sum(dim=1)  # 2, H, W
                    depth_normal_loss = args.lambda_depth_normal * (
                        (1. - reg_depth_ratio) * normal_error_map[0]  # (H, W)
                        + reg_depth_ratio * normal_error_map[1]  # (H, W)
                    )  # (H, W)
                else:
                    # If shape is 3, H, W
                    depth_normal_loss = args.lambda_depth_normal * (1 - (rendered_normals * rendered_depth_to_normals).sum(dim=0))  # (H, W)
            
            # Weight by image gradient as in PGSR
            if args.weight_by_img_grad:
                image_weight = get_img_grad_weight(img=gt_image)  # (H, W)
                image_weight = (1.0 - image_weight).clamp(min=0.0, max=1.0)  # (H, W)
                image_weight = image_weight ** 2  # (H, W)
                depth_normal_loss = depth_normal_loss * image_weight  # (H, W)

            # Semantic boundary weighting: depth discontinuities across
            # instance boundaries are legitimate; do not smooth over them.
            if boundary_weighting_active:
                boundary_weight_map = get_boundary_weight_map(
                    semantic_prior_state, viewpoint_cam, depth_normal_loss.device
                )  # (H, W)
                depth_normal_loss = depth_normal_loss * boundary_weight_map  # (H, W)

            depth_normal_loss = depth_normal_loss.mean()
            
            loss = loss + depth_normal_loss
            
        # Min scale regularization (from PGSR)
        if args.use_scale_loss:
            if visibility_filter.sum() > 0:
                min_scaling_loss = torch.sort(
                    gaussians.get_scaling_with_3D_filter[visibility_filter],  # (N_visible_gaussians, 3)
                    dim=-1
                ).values[..., 0]  # (N_visible_gaussians,)
                min_scaling_loss = args.scale_loss_weight * min_scaling_loss.mean()
                loss = loss + min_scaling_loss

        # Semantic Prior Field Regularization
        # > Orientation prior, selective flattening, SH region consistency
        #   and SH outlier decay, all confidence-gated by the prior field.
        if semantic_prior_kick_on:
            semantic_prior_pkg = compute_semantic_prior_regularization(
                iteration=iteration,
                gaussians=gaussians,
                semantic_head=semantic_head,
                config=semantic_prior_config,
                state=semantic_prior_state,
                args=args,
                visibility_filter=visibility_filter,
            )
            semantic_prior_loss = semantic_prior_pkg["semantic_prior_loss"]
            loss = loss + semantic_prior_loss
            
        # Depth Order Regularization
        # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
        # > In the paper, MILo does not rely on any learned prior. 
        if depth_order_kick_on:
            if depth_order_config["depth_ratio"] < 1.:
                depth_for_depth_order = (
                    (1. - depth_order_config["depth_ratio"]) * render_pkg["expected_depth"]
                    + depth_order_config["depth_ratio"] * render_pkg["median_depth"]
                )
            else:
                depth_for_depth_order = render_pkg["median_depth"]
                
            depth_prior_loss, _, do_supervision_depth, lambda_depth_order = compute_depth_order_regularization(
                iteration=iteration,
                rendered_depth=depth_for_depth_order,
                depth_priors=depth_priors,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                config=depth_order_config,
            )
                
            loss = loss + depth_prior_loss
            depth_order_kick_on = lambda_depth_order > 0
            
        # Multiview Regularization
        if multiview_kick_on:
            
            if multiview_render is None:
                multiview_render_pkg = render_pkg
            else:
                multiview_render_pkg = multiview_render(
                    viewpoint_cam, gaussians, pipe, bg,
                    require_coord=False, 
                    require_depth=True,
                )
            
            multiview_pixel_weight = None
            if boundary_weighting_active:
                multiview_pixel_weight = get_boundary_weight_map(
                    semantic_prior_state, viewpoint_cam, render_pkg["render"].device
                )  # (H, W)

            multiview_render_pkg = compute_multiview_regularization(
                iteration=iteration,
                scene=scene,
                render_pkg=multiview_render_pkg,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                render_func=render,
                pipe=pipe,
                background=bg,
                multiview_config=multiview_config,
                multiview_state=multiview_state,
                kernel_size=0.0,
                rasterizer=args.rasterizer,
                pixel_weight=multiview_pixel_weight,
            )
            multiview_loss = multiview_render_pkg["multiview_loss"] * args.multiview_factor
            loss = loss + multiview_loss
        
        # Normal Field Regularization
        if normal_field_kick_on:
            if args.detach_gaussian_rendering:
                detached_render_pkg = {
                    "render": render_pkg["render"].detach(),
                    "median_depth": render_pkg["median_depth"].detach(),
                    "expected_depth": render_pkg["expected_depth"].detach(),
                    "normal": render_pkg["normal"].detach(),
                }
            
            normal_field_render_pkg = compute_normal_field_regularization(
                iteration=iteration,
                render_pkg=detached_render_pkg if args.detach_gaussian_rendering else render_pkg,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                scene=scene,
                pipe=pipe,
                background=bg,
                kernel_size=0.0,
                config=normal_field_config,
                normal_field_state=normal_field_state,
                render_func=partial(render, require_coord=False, require_depth=True),
                args=args,
            )
            normal_field_loss = normal_field_render_pkg["normal_field_loss"]
            loss = loss + normal_field_loss
            
            normal_field_alignment_loss = normal_field_render_pkg["normal_field_alignment_loss"]
            front_pivots_visibility_loss = normal_field_render_pkg["front_pivots_visibility_loss"]
            back_pivots_occlusion_loss = normal_field_render_pkg["back_pivots_occlusion_loss"]
            gaussian_flattening_loss = normal_field_render_pkg["gaussian_flattening_loss"]
            sdf_and_normal_field_consistency_loss = normal_field_render_pkg["sdf_and_normal_field_consistency_loss"]
            
        # Mesh-In-the-Loop Regularization
        if milo_kick_on:
            milo_pkg = compute_mesh_in_the_loop_regularization(
                iteration=iteration,
                train_cameras=all_viewpoints,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                render_pkg=render_pkg,
                gaussians=gaussians,
                pipe=pipe,
                background=bg,
                kernel_size=0.0,
                milo_config=milo_config,
                milo_state=milo_state,
                args=args,
            )
            milo_loss = milo_pkg["milo_loss"]
            loss = loss + milo_loss
            
            mesh_depth_loss = milo_pkg["mesh_depth_loss"]
            mesh_normal_loss = milo_pkg["mesh_normal_loss"]
            occupied_centers_loss = milo_pkg["occupied_centers_loss"]
        
        # ---Backward pass---
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # ---Accumulate semantic stats from the SPF rasterizer backward---
            if semantic_stats is not None and semantic_stats_sink:
                semantic_stats.update_from_sink(semantic_stats_sink)
            # ---Logging---
            (
                postfix_dict,
                ema_loss_for_log, 
                ema_depth_normal_loss_for_log, 
                ema_normal_field_alignment_loss_for_log,
                ema_front_pivots_visibility_loss_for_log,
                ema_back_pivots_occlusion_loss_for_log,
                ema_gaussian_flattening_loss_for_log,
                ema_sdf_and_normal_field_consistency_loss_for_log,
                ema_depth_order_loss_for_log,
                ema_multiview_loss_for_log,
                ema_mesh_depth_loss_for_log,
                ema_mesh_normal_loss_for_log,
                ema_occupied_centers_loss_for_log,
            ) = log_normal_field_training_progress(
                args, iteration, log_interval, progress_bar, run,
                scene, gaussians, pipe, opt, bg,
                viewpoint_idx, viewpoint_cam, render_pkg, 
                normal_field_render_pkg if normal_field_kick_on else None, 
                milo_pkg if milo_kick_on else None,
                do_supervision_depth if depth_order_kick_on else None,
                reg_kick_on, normal_field_kick_on, depth_order_kick_on, multiview_kick_on, milo_kick_on,
                loss, depth_normal_loss if reg_kick_on else None, 
                normal_field_alignment_loss if normal_field_kick_on else None,
                front_pivots_visibility_loss if normal_field_kick_on else None,
                back_pivots_occlusion_loss if normal_field_kick_on else None,
                gaussian_flattening_loss if normal_field_kick_on else None,
                sdf_and_normal_field_consistency_loss if normal_field_kick_on else None,
                depth_prior_loss if depth_order_kick_on else None,
                multiview_loss if multiview_kick_on else None,
                mesh_depth_loss if milo_kick_on else None,
                mesh_normal_loss if milo_kick_on else None,
                occupied_centers_loss if milo_kick_on else None,
                normal_field_config if normal_field_kick_on else None, 
                milo_config if milo_kick_on else None,
                postfix_dict, ema_loss_for_log, ema_depth_normal_loss_for_log, 
                ema_normal_field_alignment_loss_for_log,
                ema_front_pivots_visibility_loss_for_log,
                ema_back_pivots_occlusion_loss_for_log,
                ema_gaussian_flattening_loss_for_log,
                ema_sdf_and_normal_field_consistency_loss_for_log,
                ema_depth_order_loss_for_log, 
                ema_multiview_loss_for_log,
                ema_mesh_depth_loss_for_log,
                ema_mesh_normal_loss_for_log,
                ema_occupied_centers_loss_for_log,
                testing_iterations, saving_iterations, render,
            )

            # ---Diagnostics: scalars, images, prior field, snapshots---
            if diagnostics is not None:
                if diagnostics.wants_scalars(iteration):
                    diag_scalars = {
                        "loss": loss,
                        "l1": Ll1,
                        "ssim": ssim_value,
                        "n_gaussians": gaussians._xyz.shape[0],
                    }
                    if semantic_enabled and semantic_loss is not None:
                        diag_scalars["semantic_ce"] = semantic_loss
                    if semantic_enabled and semantic_3d_loss is not None:
                        diag_scalars["semantic_3d"] = semantic_3d_loss
                    if reg_kick_on:
                        diag_scalars["depth_normal"] = depth_normal_loss
                    if multiview_kick_on:
                        diag_scalars["multiview"] = multiview_loss
                        diag_scalars["multiview_ncc"] = multiview_render_pkg["ncc_loss"]
                        diag_scalars["multiview_geo"] = multiview_render_pkg["geo_loss"]
                    if normal_field_kick_on:
                        diag_scalars["normal_field"] = normal_field_loss
                        diag_scalars["normal_field_alignment"] = normal_field_alignment_loss
                    if milo_kick_on:
                        diag_scalars["milo"] = milo_loss
                    if semantic_prior_kick_on:
                        for diag_key in (
                            "semantic_prior_loss",
                            "orientation_prior_loss",
                            "selective_flatten_loss",
                            "sh_consistency_loss",
                            "sh_decay_loss",
                        ):
                            diag_scalars[diag_key] = semantic_prior_pkg[diag_key]
                    if semantic_stats is not None and semantic_stats.ready:
                        conflict = semantic_stats.conflict_score()
                        if conflict is not None:
                            diag_scalars["semantic_conflict_mean"] = conflict.mean()
                            diag_scalars["semantic_stats_updates"] = semantic_stats.updates
                    if semantic_log_sigma is not None:
                        diag_scalars["semantic_weight_effective"] = (
                            torch.exp(-semantic_log_sigma).squeeze() * args.lambda_semantic
                        )
                    diagnostics.log_scalars(iteration, diag_scalars)

                if diagnostics.wants_images(iteration):
                    diagnostics.dump_training_view(
                        iteration=iteration,
                        viewpoint_cam=viewpoint_cam,
                        render_pkg=render_pkg,
                        gt_image=gt_image,
                        semantic_head=semantic_head,
                        observation=observation if semantic_enabled else None,
                        ignore_label=(
                            semantic_observations.ignore_label if semantic_enabled else -1
                        ),
                        boundary_weight_map=(
                            get_boundary_weight_map(
                                semantic_prior_state, viewpoint_cam, "cpu"
                            )
                            if boundary_weighting_active else None
                        ),
                        num_classes=semantic_num_classes,
                    )

                if semantic_prior_kick_on:
                    diagnostics.maybe_dump_prior_field(
                        iteration, semantic_prior_state["prior_field"]
                    )

                if diagnostics.wants_snapshot(iteration):
                    diagnostics.dump_snapshot(
                        iteration,
                        gaussians,
                        semantic_prior_state["prior_field"] if semantic_prior_enabled else None,
                    )

            if semantic_enabled and iteration in saving_iterations:
                save_semantic_checkpoint(
                    os.path.join(scene.model_path, "semantic"),
                    head=semantic_head,
                    gaussian_model=gaussians,
                    iteration=iteration,
                    num_classes=semantic_num_classes,
                    renderer=args.rasterizer,
                    optimizer={
                        "gaussian": gaussians.optimizer.state_dict(),
                        "head": semantic_head_optimizer.state_dict(),
                    },
                    metadata={
                        "mask_dir": os.path.abspath(args.semantic_masks),
                        "mode": "joint",
                    },
                )
            
            if iteration % 100 == 0:
                if dataset.use_unbounded_opacity:
                    _opacity = gaussians.get_opacity_with_3D_filter.detach()
                    _contribution = gaussians.get_contribution(viewpoint_cam).detach()
                    print(f"Min contribution: {_contribution.min()}, Max contribution: {_contribution.max()}")
                    print(f"Min opacity: {_opacity.min()}, Max opacity: {_opacity.max()}")

            # ---Densification---
            gaussians_have_changed = False
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats_radegs(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if not args.use_max_size_threshold:
                        size_threshold = None
                    # Semantic Prior Field budget reallocation: planar
                    # instances densify less, thin instances densify more.
                    semantic_grad_multipliers = None
                    if semantic_prior_kick_on and args.sp_budget:
                        _prior_field = semantic_prior_state["prior_field"]
                        if (
                            _prior_field.valid
                            and _prior_field.densify_multiplier is not None
                            and _prior_field.densify_multiplier.shape[0] == gaussians._xyz.shape[0]
                        ):
                            semantic_grad_multipliers = _prior_field.densify_multiplier
                    n_cloned, n_split, n_pruned = gaussians.densify_and_prune_radegs(
                        opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold,
                        use_abs_grad=args.use_abs_grad_for_densification,
                        viewpoint_cameras=scene.getTrainCameras().copy(),
                        grad_multipliers=semantic_grad_multipliers,
                    )
                    gaussians_have_changed = True
                    if diagnostics is not None:
                        diagnostics.log_event(
                            iteration, "densify_radegs",
                            cloned=n_cloned, split=n_split, pruned=n_pruned,
                            n_gaussians=gaussians._xyz.shape[0],
                            budget_multipliers_active=semantic_grad_multipliers is not None,
                        )
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )
                    else:
                        gaussians.reset_3D_filter()
                        
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    if diagnostics is not None:
                        diagnostics.log_event(iteration, "opacity_reset")
                    
            # ---Non-maximal pruning---
            if iteration in args.non_maximal_pruning_iterations:
                print(f"[INFO] Pruning non-maximal Gaussians at iteration {iteration+1}.")
                print(f"        > Number of Gaussians before pruning: {gaussians._xyz.shape[0]}.")
                n_before_nonmax_prune = gaussians._xyz.shape[0]
                prune_non_maximal_gaussians(
                    gaussians=gaussians,
                    cameras=scene.getTrainCameras().copy(),
                    pipe=pipe,
                    background=bg,
                )
                gaussians_have_changed = True
                print(f"        > Number of Gaussians after pruning: {gaussians._xyz.shape[0]}.")
                if diagnostics is not None:
                    diagnostics.log_event(
                        iteration, "non_maximal_prune",
                        before=n_before_nonmax_prune,
                        after=gaussians._xyz.shape[0],
                    )
                if use_mip_filter:
                    gaussians.compute_3D_filter(
                        cameras=scene.getTrainCameras().copy()
                    )
                
            # ---Normal Field Densification---
            if args.use_normal_field:
                cond_1 = (
                    normal_field_kick_on 
                    and normal_field_config["use_densification"]
                )
                cond_2 = (
                    (iteration+1 >= normal_field_config["start_iter_densification"])
                    and (iteration+1 <= normal_field_config["end_iter_densification"])
                )
                cond_3 = (
                    (iteration+1 - normal_field_config["start_iter_densification"]) % normal_field_config["densify_every_n_iterations"] == 0
                )
                if cond_1 and cond_2 and cond_3:
                    print(f"[INFO] Densifying normal field at iteration {iteration+1}.")
                    print(f"        > Using normalization method: {normal_field_config['densification_normalization_method']}.")
                    print(f"        > Using normal computed from: {normal_field_config['densification_normal_to_use']}.")
                    print(f"        > Using normal errors quantile: {normal_field_config['densification_normal_errors_quantile']}.")
                    print(f"        > Maintaining constant volume: {normal_field_config['maintain_constant_volume']}.")
                    print(f"        > Number of Gaussians before densification: {gaussians._xyz.shape[0]}.")
                    n_before_spoke_split = gaussians._xyz.shape[0]
                    densify_normal_field(
                        iteration=iteration, 
                        gaussians=gaussians, 
                        cameras=scene.getTrainCameras().copy(),
                        scene=scene, 
                        pipe=pipe, 
                        background=bg, 
                        kernel_size=0.0, 
                        config=normal_field_config,
                        normal_field_state=normal_field_state, 
                        render_func=render, 
                        args=args,
                        maintain_constant_volume=normal_field_config["maintain_constant_volume"],
                    )
                    gaussians_have_changed = True
                    print(f"        > Number of Gaussians after densification: {gaussians._xyz.shape[0]}.")
                    if diagnostics is not None:
                        diagnostics.log_event(
                            iteration, "spoke_split",
                            before=n_before_spoke_split,
                            after=gaussians._xyz.shape[0],
                        )
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )

                    if normal_field_config["reset_normals_after_densification"]:
                        print(f"[INFO] Resetting normal features after densification.")
                        gaussians.reset_normal_features(
                            reset_directions=normal_field_config["reset_normal_directions"],
                            reset_signs=normal_field_config["reset_normal_signs"],
                        )
                        
            # ---Normal field pruning---
            if args.use_normal_field:
                cond_1 = (
                    normal_field_kick_on 
                    and normal_field_config["use_pruning"]
                )
                cond_2 = (
                    (iteration+1 >= normal_field_config["start_iter_pruning"])
                    and (iteration+1 <= normal_field_config["end_iter_pruning"])
                )
                cond_3 = (
                    (iteration+1 - normal_field_config["start_iter_pruning"]) % normal_field_config["prune_every_n_iterations"] == 0
                )
                if cond_1 and cond_2 and cond_3:
                    print(f"[INFO] Pruning non-maximal Gaussians at iteration {iteration+1}.")
                    print(f"        > Number of Gaussians before pruning: {gaussians._xyz.shape[0]}.")
                    n_before_nf_prune = gaussians._xyz.shape[0]
                    prune_non_maximal_gaussians(
                        gaussians=gaussians,
                        cameras=scene.getTrainCameras().copy(),
                        pipe=pipe,
                        background=bg,
                    )
                    gaussians_have_changed = True
                    print(f"        > Number of Gaussians after pruning: {gaussians._xyz.shape[0]}.")
                    if diagnostics is not None:
                        diagnostics.log_event(
                            iteration, "normal_field_prune",
                            before=n_before_nf_prune,
                            after=gaussians._xyz.shape[0],
                        )
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )

            # ---Semantic-error splitting---
            # > Gaussians straddling an instance boundary are split along
            #   their dominant tangential axis (offset from normal-field
            #   densification to avoid same-iteration topology churn).
            if semantic_prior_kick_on and args.sp_split and semantic_prior_config["use_densification"]:
                cond_2 = (
                    (iteration+1 >= semantic_prior_config["start_iter_densification"])
                    and (iteration+1 <= semantic_prior_config["end_iter_densification"])
                )
                cond_3 = (
                    (iteration+1 - semantic_prior_config["start_iter_densification"]) % semantic_prior_config["densify_every_n_iterations"] == 0
                )
                if cond_2 and cond_3:
                    # Prefer the continuously accumulated conflict score from
                    # the SPF rasterizer backward over the episodic full
                    # camera sweep (E5 validation: conflict localizes
                    # boundary-straddling Gaussians ~30x better than chance).
                    stats_scores = None
                    if (
                        semantic_stats is not None
                        and semantic_stats.ready
                        and semantic_stats.updates >= semantic_prior_config.get("stats_min_updates", 100)
                    ):
                        stats_scores = semantic_stats.conflict_score()
                        if (
                            stats_scores is not None
                            and stats_scores.shape[0] != gaussians._xyz.shape[0]
                        ):
                            stats_scores = None
                    split_source = "stats" if stats_scores is not None else "sweep"
                    print(f"[INFO] Semantic-error splitting at iteration {iteration+1} (source: {split_source}).")
                    print(f"        > Number of Gaussians before splitting: {gaussians._xyz.shape[0]}.")
                    n_before_semantic_split = gaussians._xyz.shape[0]
                    densify_semantic_boundary(
                        gaussians=gaussians,
                        cameras=scene.getTrainCameras().copy(),
                        render_func=render,
                        pipe=pipe,
                        background=bg,
                        semantic_head=semantic_head,
                        observations=semantic_observations,
                        config=semantic_prior_config,
                        args=args,
                        precomputed_errors=stats_scores,
                    )
                    gaussians_have_changed = True
                    print(f"        > Number of Gaussians after splitting: {gaussians._xyz.shape[0]}.")
                    if diagnostics is not None:
                        diagnostics.log_event(
                            iteration, "semantic_split",
                            before=n_before_semantic_split,
                            after=gaussians._xyz.shape[0],
                            source=split_source,
                        )
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )

            # ---Identity-stability pruning---
            # > Floaters have flat label posteriors under multi-view mask
            #   supervision; prune them only if they also never win a pixel.
            if semantic_prior_kick_on and args.sp_prune and semantic_prior_config["use_pruning"]:
                if iteration in semantic_prior_config["pruning_iterations"]:
                    if gaussians_have_changed:
                        semantic_prior_state["prior_field"].invalidate()
                    _prior_field = maybe_refresh_prior_field(
                        iteration, gaussians, semantic_head,
                        semantic_prior_config, semantic_prior_state,
                    )
                    print(f"[INFO] Identity-stability pruning at iteration {iteration}.")
                    print(f"        > Number of Gaussians before pruning: {gaussians._xyz.shape[0]}.")
                    n_pruned = prune_identity_unstable_gaussians(
                        gaussians=gaussians,
                        prior_field=_prior_field,
                        cameras=scene.getTrainCameras().copy(),
                        pipe=pipe,
                        background=bg,
                        confidence_threshold=semantic_prior_config["prune_confidence_threshold"],
                    )
                    if n_pruned > 0:
                        gaussians_have_changed = True
                    print(f"        > Number of Gaussians after pruning: {gaussians._xyz.shape[0]}.")
                    if diagnostics is not None:
                        diagnostics.log_event(
                            iteration, "identity_prune",
                            pruned=n_pruned,
                            n_gaussians=gaussians._xyz.shape[0],
                        )
                    if n_pruned > 0 and use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras().copy()
                        )

            # ---Reset Normal field state if Gaussians have changed---
            if normal_field_kick_on and gaussians_have_changed:
                normal_field_state = reset_normal_field_state_at_next_iteration(normal_field_state)

            # ---Reset MILO state if Gaussians have changed---
            if milo_kick_on and gaussians_have_changed:
                milo_state = reset_milo_state_at_next_iteration(milo_state)

            # ---Reset Semantic Prior Field if Gaussians have changed---
            if semantic_prior_enabled and gaussians_have_changed:
                semantic_prior_state = reset_semantic_prior_state_at_next_iteration(semantic_prior_state)

            # ---Reset semantic stats if Gaussians have changed---
            if semantic_stats is not None and gaussians_have_changed:
                semantic_stats.reset()
            
            # ---Update 3D Mip Filter---
            if use_mip_filter and (iteration > opt.densify_until_iter) and (
                (iteration == args.warn_until_iter)
                or (iteration % args.update_mip_filter_every == 0)
            ):
                if iteration < opt.iterations - args.update_mip_filter_every:
                    gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())
                else:
                    print(f"[INFO] Skipping 3D Mip Filter update at iteration {iteration}")

            # ---Optimizer step---
            if iteration < opt.iterations:
                if gaussians.use_appearance_network or gaussians.use_exposure_compensation:
                    gaussians.optimizer.step()
                else:
                    visible = radii>0
                    gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none = True)
                if semantic_head_optimizer is not None:
                    semantic_head_optimizer.step()
                    semantic_head_optimizer.zero_grad(set_to_none=True)

            # ---Save checkpoint---
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")  
                if semantic_enabled:
                    save_semantic_checkpoint(
                        os.path.join(scene.model_path, "semantic"),
                        head=semantic_head,
                        gaussian_model=gaussians,
                        iteration=iteration,
                        num_classes=semantic_num_classes,
                        renderer=args.rasterizer,
                        optimizer={
                            "gaussian": gaussians.optimizer.state_dict(),
                            "head": semantic_head_optimizer.state_dict(),
                        },
                        metadata={
                            "mask_dir": os.path.abspath(args.semantic_masks),
                            "mode": "joint",
                        },
                    )
                
        if iteration % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print('Num of Gaussians: %d'%(gaussians._xyz.shape[0]))
    
    if WANDB_FOUND:
        run.finish()
    
    return 


def prepare_output_and_logger(dataset, args):    
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Create WandB run       
    global WANDB_FOUND
    WANDB_FOUND = (
        WANDB_FOUND
        and (args.wandb_project is not None)
        and (args.wandb_entity is not None)
    )
    if WANDB_FOUND:
        run = wandb.init(
            name=args.wandb_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
        )
    else:
        run=None
        print("[INFO] WandB not found, skipping logging.")
    return run


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # ----- Usual arguments -----
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=-1)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # ----- Rasterization technique -----
    parser.add_argument("--rasterizer", type=str, default="radegs", choices=["radegs", "ours"])

    # ----- Gaga semantic Gaussian learning -----
    parser.add_argument(
        "--semantic_masks",
        type=str,
        default=None,
        help="Directory containing Gaga-associated masks and info.json.",
    )
    parser.add_argument(
        "--semantic_checkpoint",
        type=str,
        default=None,
        help="Optional semantic sidecar used when resuming joint training.",
    )
    parser.add_argument("--semantic_num_classes", type=int, default=None)
    parser.add_argument("--semantic_head_lr", type=float, default=5e-4)
    parser.add_argument("--lambda_semantic", type=float, default=1.0)
    parser.add_argument("--lambda_semantic_3d", type=float, default=0.0)
    parser.add_argument("--semantic_3d_interval", type=int, default=10)
    parser.add_argument("--semantic_3d_samples", type=int, default=10_000)
    parser.add_argument("--semantic_3d_neighbors", type=int, default=5)
    parser.add_argument("--allow_missing_semantic_masks", action="store_true")

    # ----- Semantic Prior Field -----
    # > Live per-Gaussian labels + per-instance geometric proxies guiding
    #   orientation, density, size, SH and mesh extraction. Requires
    #   --semantic_masks (the prior field is derived from the joint
    #   semantic embedding).
    parser.add_argument("--semantic_prior", action="store_true")
    parser.add_argument("--semantic_prior_config", type=str, default="default")
    # Per-channel ablation switches (all soft, all confidence-gated)
    parser.add_argument("--sp_orient", action=BooleanOptionalAction, default=True,
        help="Align learned normals with per-instance proxy normals.")
    parser.add_argument("--sp_flatten", action=BooleanOptionalAction, default=True,
        help="Min-scale flattening restricted to planar/quadric instances.")
    parser.add_argument("--sp_sh", action=BooleanOptionalAction, default=True,
        help="SH region consistency and outlier decay.")
    parser.add_argument("--sp_split", action=BooleanOptionalAction, default=True,
        help="Split Gaussians straddling instance boundaries.")
    parser.add_argument("--sp_prune", action=BooleanOptionalAction, default=True,
        help="Prune identity-unstable non-maximal Gaussians.")
    parser.add_argument("--sp_budget", action=BooleanOptionalAction, default=True,
        help="Per-instance densification threshold multipliers.")
    parser.add_argument("--sp_boundary", action=BooleanOptionalAction, default=True,
        help="Down-weight depth-normal and multiview losses at instance boundaries.")
    parser.add_argument("--sp_stats", action=BooleanOptionalAction, default=True,
        help="Use the SPF rasterizer stats channel (per-Gaussian semantic conflict, ours rasterizer only) to drive boundary splitting.")
    parser.add_argument("--balance_semantic", action=BooleanOptionalAction, default=False,
        help="Kendall-style uncertainty weighting between photometric and semantic losses.")

    # ----- Diagnostics -----
    # > File-based recording of every intermediate quantity under
    #   <model_path>/diagnostics/ for offline architecture analysis.
    parser.add_argument("--diagnostics", action=BooleanOptionalAction, default=True,
        help="Record losses, density events, intermediate images, prior-field state and per-Gaussian snapshots.")
    parser.add_argument("--diag_scalar_interval", type=int, default=10,
        help="Log every active loss term every N iterations (scalars.jsonl).")
    parser.add_argument("--diag_image_interval", type=int, default=1000,
        help="Dump visualizations of the current training view every N iterations.")
    parser.add_argument("--diag_snapshot_interval", type=int, default=5000,
        help="Save a compressed per-Gaussian state snapshot every N iterations (0 = off).")

    # ----- Normal Field -----
    parser.add_argument("--no_normal_field", action="store_true")
    parser.add_argument("--normal_field_config", type=str, default="default_regular_densification")
    # Gaussians management
    parser.add_argument("--dense_gaussians", action="store_true")
    parser.add_argument("--detach_gaussian_rendering", action="store_true")

    # ----- Densification and Simplification -----
    # > Inspired by Mini-Splatting2.
    # > Used for pruning, densification and Gaussian pivots selection.
    parser.add_argument("--N_max_gaussians", type=int, default=None,
        help="Cap Gaussian count during Normal Field Densification. If the next densification would exceed this, only the highest-error Gaussians are added up to the cap. None = no limit.")
    parser.add_argument("--warn_until_iter", type=int, default=3000)
    parser.add_argument("--non_maximal_pruning_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--use_max_size_threshold", action=BooleanOptionalAction, default=False)
    
    # ----- Depth-Normal consistency Regularization -----
    # > Inspired by 2DGS, GOF, RaDe-GS...
    parser.add_argument("--regularization_from_iter", type=int, default = 7_000)
    parser.add_argument("--lambda_depth_normal", type=float, default = 0.05)
    parser.add_argument("--mask_depth_normal", action="store_true")
    
    # ----- Multiview Regularization -----
    parser.add_argument("--multiview", action=BooleanOptionalAction, default=True)
    parser.add_argument("--multiview_config", type=str, default="fast")
    parser.add_argument("--multiview_rasterizer", type=str, default=None)
    parser.add_argument("--multiview_factor", type=float, default=1.0)
    
    # ----- Mesh-In-the-Loop Regularization -----
    parser.add_argument("--milo", action="store_true")
    parser.add_argument("--milo_config", type=str, default="default_regular_densification")
    
    # ----- Depth Order Regularization (Learned Prior) -----
    # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
    # > In the paper, MILo does not rely on any learned prior.
    parser.add_argument("--depth_order", action="store_true")
    parser.add_argument("--depth_order_config", type=str, default="default")

    # ----- 3D Mip Filter -----
    # > Inspired by Mip-Splatting.
    parser.add_argument("--disable_mip_filter", action="store_true", default=False)
    parser.add_argument("--update_mip_filter_every", type=int, default=100)

    # ----- Appearance Network for Exposure-aware loss -----
    # > Inspired by GOF.
    parser.add_argument("--decoupled_appearance", action="store_true")
    # > Inspired by PGSR.
    parser.add_argument("--exposure_compensation", action=BooleanOptionalAction, default=True)
    
    # ----- PGSR losses -----
    parser.add_argument("--use_scale_loss", action="store_true")
    parser.add_argument("--scale_loss_weight", type=float, default=100.0)
    parser.add_argument("--weight_by_img_grad", action="store_true")

    # ----- Logging -----
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    if not -1 in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    args.use_normal_field = not args.no_normal_field
    
    if args.port == -1:
        args.port = np.random.randint(5000, 9000)
        print(f"Using random port: {args.port}")
        
    # Load multiview regularization config
    if args.multiview:
        # Get multiview config file
        multiview_config_file = os.path.join(BASE_DIR, "configs", "multiview", f"{args.multiview_config}.yaml")
        with open(multiview_config_file, "r") as f:
            multiview_config = yaml.safe_load(f)
        print(f"[INFO] Using multiview regularization with config: {args.multiview_config}")
    else:
        multiview_config = None
        
    # Load mesh-in-the-loop regularization config
    if args.milo:
        # Get mesh regularization config file
        milo_config_file = os.path.join(BASE_DIR, "configs", "mesh_in_the_loop", f"{args.milo_config}.yaml")
        with open(milo_config_file, "r") as f:
            milo_config = yaml.safe_load(f)
        print(f"[INFO] Using mesh-in-the-loop regularization with config: {args.milo_config}")
    else:
        milo_config = None
    
    # Load depth order regularization config (not used in MILo paper)
    if args.depth_order:
        # Get depth order config file
        depth_order_config_file = os.path.join(BASE_DIR, "configs", "depth_order", f"{args.depth_order_config}.yaml")
        with open(depth_order_config_file, "r") as f:
            depth_order_config = yaml.safe_load(f)
        print(f"[INFO] Using depth order regularization with config: {args.depth_order_config}")
    else:
        depth_order_config = None
        
    # Load mesh-in-the-loop regularization config
    if args.use_normal_field:
        # Get mesh regularization config file
        normal_field_config_file = os.path.join(BASE_DIR, "configs", "normal_field", f"{args.normal_field_config}.yaml")
        with open(normal_field_config_file, "r") as f:
            normal_field_config = yaml.safe_load(f)
        print(f"[INFO] Using normal field with config: {args.normal_field_config}")
    else:
        normal_field_config = None

    # Load Semantic Prior Field config
    if args.semantic_prior:
        semantic_prior_config_file = os.path.join(BASE_DIR, "configs", "semantic_prior", f"{args.semantic_prior_config}.yaml")
        with open(semantic_prior_config_file, "r") as f:
            semantic_prior_config = yaml.safe_load(f)
        print(f"[INFO] Using Semantic Prior Field with config: {args.semantic_prior_config}")
    else:
        semantic_prior_config = None
    
    # Message for detach_gaussian_rendering
    if args.detach_gaussian_rendering:
        print(f"[INFO] Detaching Gaussian rendering for mesh regularization.")
    
    # Import rendering function
    print(f"[INFO] Using {args.rasterizer} as rasterizer.")
    if args.rasterizer == "radegs":
        from gaussian_renderer.radegs import render_radegs as render
        from gaussian_renderer.radegs import integrate_radegs as integrate
        args.use_abs_grad_for_densification = True
    elif args.rasterizer == "ours":
        from gaussian_renderer.ours import render_ours as render
        from gaussian_renderer.ours import integrate_ours as integrate
        args.use_abs_grad_for_densification = True
        args.mask_depth_normal = True
        print(f"[INFO] Using Ours rasterizer. Setting mask_depth_normal to True.")
    else:
        raise ValueError(f"Invalid rasterizer: {args.rasterizer}")
    
    if args.multiview_rasterizer == "ours":
        print(f"[INFO] Using Ours rasterizer for multiview regularization.")
        from gaussian_renderer.ours import render_ours as multiview_render
    elif args.multiview_rasterizer is None:
        print(f"[INFO] Using default rasterizer for multiview regularization.")
        multiview_render = None
    else:
        raise ValueError(f"Invalid multiview rasterizer: {args.multiview_rasterizer}")
        
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    torch.cuda.synchronize()
    time_start=time.time()
    
    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args,
        depth_order_config,
        normal_field_config,
        multiview_config,
        milo_config,
        semantic_prior_config,
        args.log_interval,
    )

    torch.cuda.synchronize()
    time_end=time.time()
    time_total=time_end-time_start
    print('Training time: %fs'%(time_total))

    time_txt_path=os.path.join(args.model_path, r'time.txt')
    with open(time_txt_path, 'w') as f:  
        f.write(str(time_total)) 

    # All done
    print("\nTraining complete.")
