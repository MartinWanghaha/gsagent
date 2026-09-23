# This file is part of inpaint360gs: Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
# Project page: https://dfki-av.github.io/inpaint360gs/
#
# Copyright 2024-2026 Shaoxiang Wang <shaoxiang.wang@dfki.de>
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
#
# This file contains original research code and modified components from the 
# aforementioned projects. It is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

import numpy as np
import open3d as o3d
from scene import Scene
from plyfile import PlyData, PlyElement
import torch
import os
import tempfile
from pathlib import Path
from os import makedirs, path
from errno import EEXIST
from sklearn.neighbors import KDTree
from gaussian_renderer import render
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from simple_knn._C import distCUDA2

import lpips
from random import randint
from torch import nn
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import json
from tqdm import tqdm
from render import visualize_obj, render_video_func_wriva
from utils.loss_utils import masked_l1_loss, ssim, masked_ssim
from PIL import Image
import torchvision
import cv2
from edit_object_removal import points_inside_convex_hull
from utils.general_utils import safe_state
from utils.pose_utils import generate_ellipse_path
from utils.graphics_utils import getWorld2View2,getProjectionMatrix
from utils.general_utils import PILtoTorch
import copy
from utils.point_utils import project_3d_points,ndc_to_pixel
from utils.virtual_camera_manifest import (
    load_virtual_camera_manifest,
    virtual_views_from_manifest,
)

C0 = 0.28209479177387814

FRAME_COUNT = 30
IMAGE_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def _selected_object_ids(selected_obj_ids, class_count):
    """Normalize object IDs without the historical 8-bit/256-class cutoff."""

    raw_selected = (
        [selected_obj_ids]
        if isinstance(selected_obj_ids, (int, np.integer))
        and not isinstance(selected_obj_ids, bool)
        else (selected_obj_ids or [])
    )
    values = [int(value) for value in raw_selected]
    if not values:
        raise ValueError("select_obj_id must contain at least one object ID")
    invalid_ids = [value for value in values if value < 0 or value >= class_count]
    if invalid_ids:
        raise ValueError(
            f"selected object IDs must be in [0, {class_count - 1}], got {invalid_ids}"
        )
    return values


def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    row_indices = torch.where(rows)[0]
    column_indices = torch.where(cols)[0]
    if row_indices.numel() == 0 or column_indices.numel() == 0:
        return None
    ymin, ymax = row_indices[[0, -1]]
    xmin, xmax = column_indices[[0, -1]]
    
    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    if bbox is None:
        return None
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

# Function to divide image into K x K patches
def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"image is too small for a {K}x{K} patch grid: {H}x{W}")
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)    
    return patches.permute(0, 4, 1, 2, 3)

def construct_list_of_attributes(features_dc,features_rest,scaling,rotation, objects_dc):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]*features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(objects_dc.shape[1]*objects_dc.shape[2]):
            l.append('obj_dc_{}'.format(i))
        return l


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def save_ply(xyz, features_dc, features_rest, opacity, scaling, rotation, objects_dc, path_save):
    """
    
    """
    mkdir_p(os.path.dirname(path_save))

    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scale = scaling.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()
    obj_dc = objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scaling, rotation, objects_dc)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path_save)
    print("The new point cloud are saved at {}".format(path_save))


def get_projected_gaussians(gaussians, viewpoint, supp_ply_path=None): 
    """
    Project 3D Gaussian points to the 2D image plane and filter out points
    that fall outside the image bounds.

    Return:
        p_inside_mask: mask for points inside the image
        p_inside_obj_mask: mask for points inside the object region
    """
    proj_matrix = viewpoint.full_proj_transform

    W = viewpoint.image_width
    H = viewpoint.image_height

    obj_mask = (viewpoint.objects.detach() > 0).to(torch.uint8)
    obj_mask_np = obj_mask.cpu().numpy()
    original_area = np.sum(obj_mask_np)
    if original_area == 0:
        return {
            "p_inside_mask": torch.zeros_like(
                gaussians.get_xyz[:, 0], dtype=torch.bool
            ),
            "p_inside_obj_mask": torch.zeros_like(
                gaussians.get_xyz[:, 0], dtype=torch.bool
            ),
        }
    target_area = int(original_area * 1.10)
    for k in range(3, 101, 2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(obj_mask_np, kernel)
        if np.sum(dilated) >= target_area:
            break 
    obj_mask = torch.from_numpy(dilated).to(device=viewpoint.objects.device).bool()

    p_hom = project_3d_points(gaussians.get_xyz, proj_matrix)  # (N, 4)
    p_hom_z = p_hom[:, 2]

    p_w = 1 / (p_hom[:, 3:] + 1e-8)
    p_proj = p_hom[:, :3] * p_w
    p_proj[:, 0] = ndc_to_pixel(p_proj[:, 0], W)
    p_proj[:, 1] = ndc_to_pixel(p_proj[:, 1], H)
    p_proj = torch.round(p_proj[:, :2]).long()

    p_inside_mask = (p_proj[:, 0] >= 0) & (p_proj[:, 0] < W) & (p_proj[:, 1] >= 0) & (p_proj[:, 1] < H) & (p_hom_z > 0)

    p_proj_inside = p_proj[p_inside_mask]  # (M, 2)
    x_coords, y_coords = p_proj_inside[:, 0], p_proj_inside[:, 1]
    obj_mask_values = obj_mask[y_coords, x_coords]
    
    p_inside_obj_mask = torch.zeros_like(p_inside_mask)    
    p_inside_obj_mask[p_inside_mask] = obj_mask_values

    # --- Spatial Depth Filter ---
    adaptive_threshold = 0.0
    if supp_ply_path is not None:
        support_path = Path(supp_ply_path).expanduser().resolve(strict=True)
        from scipy.spatial import cKDTree
        
        # 1. Load seed points and build index
        supp_ply = PlyData.read(support_path)
        supp_xyz = np.stack([np.asarray(supp_ply.elements[0][axis]) for axis in 'xyz'], axis=1)
        if len(supp_xyz) == 0 or not np.isfinite(supp_xyz).all():
            raise ValueError(f"support PLY must contain finite XYZ points: {support_path}")
        tree = cKDTree(supp_xyz)

        # 2. Define adaptive threshold based on seed distribution (e.g., 3x average STD)
        adaptive_threshold = float(np.std(supp_xyz, axis=0).mean() * 3.0)

        # 3. Query distances for 2D-masked candidates only
        candidate_indices = torch.where(p_inside_obj_mask)[0]
        if len(candidate_indices) > 0 and adaptive_threshold > 0:
            candidate_xyz = gaussians.get_xyz[candidate_indices].detach().cpu().numpy()
            dists, _ = tree.query(candidate_xyz)

            # 4. Refine mask by spatial proximity
            valid_mask = dists < adaptive_threshold
            final_mask = torch.zeros_like(p_inside_obj_mask, dtype=torch.bool)
            final_mask[candidate_indices[valid_mask]] = True
            p_inside_obj_mask = final_mask
   
    p_proj_inside = p_proj[p_inside_mask]
    projected_gaussian = {
        "p_inside_mask": p_inside_mask,       
        "p_inside_obj_mask": p_inside_obj_mask, 
        "gate_mask": obj_mask_np if original_area == 0 else dilated,
        "gate_projection": proj_matrix,
        "gate_distance_threshold": adaptive_threshold,
    }

    return projected_gaussian


def select_inpaint_masks(gaussians, classifier, selected_obj_ids, removal_thresh):
    """Shared, read-only selection for initialization and density reference export."""

    class_count = int(classifier.out_channels)
    selected_values = _selected_object_ids(selected_obj_ids, class_count)
    selected_obj_ids = torch.tensor(
        selected_values, dtype=torch.long, device=gaussians.get_xyz.device
    )
    masks_per_obj = dict()

    # get 3d gaussians idx corresponding to select obj id
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d,dim=0)

        for obj_id in selected_obj_ids:
            obj_id_int = int(obj_id.item())
            obj_prob = prob_obj3d[obj_id_int, :, :]
            mask = obj_prob > removal_thresh
            mask3d = mask.squeeze()
            if not mask3d.any():
                raise ValueError(
                    f"object ID {obj_id_int} selected no Gaussians at "
                    f"removal_thresh={removal_thresh}"
                )

            mask3d_convex, _ = points_inside_convex_hull(
                gaussians._xyz.detach(), mask3d, remove_outliers=True, outlier_factor=1.0
            )
            mask3d = torch.logical_or(mask3d,mask3d_convex)
            mask3d = mask3d.float()[:,None,None]

            masks_per_obj[obj_id_int] = {
                "mask": mask.float()[:,None],
                "mask3d": mask3d,
            }
    return masks_per_obj


@torch.no_grad()
def update_inpaint_density(enabled, iteration, gaussians, viewspace_points,
                           visibility, radii, opt, cameras_extent):
    """Stage 5a only; disabling also skips all density statistics."""
    if not enabled or iteration >= 5000:
        return
    gaussians.max_radii2D[visibility] = torch.max(gaussians.max_radii2D[visibility], radii[visibility])
    gaussians.add_densification_stats(viewspace_points, visibility)
    if iteration > 500 and iteration % 100 == 0:
        gaussians.densify_and_prune_inpaint(opt.densify_grad_threshold, 0.005,
            cameras_extent, 20, gaussians.sub_feature_num)


def finetune_inpaint(args, opt, dataset, model_path, iteration, views, gaussians,
                     pipeline, background, classifier, selected_obj_ids,
                     cameras_extent, removal_thresh, finetune_iteration):
    rgb_densify = getattr(args, 'rgb_densify', True)
    if not isinstance(rgb_densify, bool):
        raise ValueError('rgb_densify must be boolean')
    context_path = getattr(args, 'local_geometry_context', None)
    if context_path:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/paintmesh'))
        from local_geometry_io import read_receipt
        context = read_receipt(context_path, 'paintmesh-rgb-finetune-context')
        if context['parameters'].get('rgb_densify') is not rgb_densify:
            raise ValueError('RGB densification differs from requested context')
    # Direct CLI runs also record their setting before any optimization.
    if getattr(args, 'inpaint_output_ply', None):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/paintmesh'))
        from local_geometry_io import prepare_rgb_policy
        output = Path(args.inpaint_output_ply)
        prepare_rgb_policy(output.with_suffix('.density_policy.json'), output, rgb_densify)
    print(f'Stage 5a clone/split/prune: {rgb_densify}', flush=True)
    masks_per_obj = select_inpaint_masks(gaussians, classifier, selected_obj_ids, removal_thresh)
    if getattr(args, "density_manifest", None):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/paintmesh"))
        from support_density_io import validate_density
        receipt = validate_density(args.density_manifest)
        expected = np.load(receipt["outputs"]["reference"]["path"], allow_pickle=False)["retained_rows"]
        removed = torch.stack([v["mask3d"].bool().reshape(-1) for v in masks_per_obj.values()]).any(0)
        actual = torch.where(~removed)[0].cpu().numpy()
        if not np.array_equal(expected, actual):
            raise ValueError("density reference selection differs from Stage 5a retained rows")

    # initialize gaussians
    gaussians.inpaint_setup(args, opt, masks_per_obj)
    def density_audit(stage):
        if getattr(args, "density_manifest", None):
            from support_density_io import audit_density_xyz
            audit_density_xyz(args.density_manifest, gaussians.get_xyz.detach().cpu().numpy(),
                Path(args.inpaint_output_ply).parent / "density_debug" / f"{stage}.json",
                retained_prefix=gaussians.sub_feature_num)
    density_audit("initialization")

    removal_gaussian = GaussianModel(gaussians.max_sh_degree)
    removal_gaussian._xyz           = nn.Parameter(gaussians._xyz[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._features_dc   = nn.Parameter(gaussians._features_dc[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._features_rest = nn.Parameter(gaussians._features_rest[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._opacity       = nn.Parameter(gaussians._opacity[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._scaling       = nn.Parameter(gaussians._scaling[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._rotation      = nn.Parameter(gaussians._rotation[:gaussians.sub_feature_num].detach().clone())
    removal_gaussian._objects_dc    = nn.Parameter(gaussians._objects_dc[:gaussians.sub_feature_num].detach().clone())

    iterations = finetune_iteration    
    progress_bar = tqdm(range(iterations), desc="Finetuning progress")
    LPIPS = lpips.LPIPS(net='vgg')
    for param in LPIPS.parameters():
        param.requires_grad = False      
    LPIPS.cuda()

    for iteration in range(iterations):
        viewpoint_stack = views.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

        mask2d = viewpoint_cam.objects > 128
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = masked_l1_loss(image, gt_image, ~mask2d)  

        bbox = mask_to_bbox(mask2d)
        lpips_loss = torch.zeros((), dtype=image.dtype, device=image.device)
        if bbox is not None:
            cropped_image = crop_using_bbox(image, bbox)
            cropped_gt_image = crop_using_bbox(gt_image, bbox)
            K = 2
            if cropped_image.shape[-2] >= K and cropped_image.shape[-1] >= K:
                rendering_patches = divide_into_patches(cropped_image[None, ...], K)
                gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)
                if (
                    rendering_patches.shape[-2] >= 32
                    and rendering_patches.shape[-1] >= 32
                ):
                    lpips_loss = LPIPS(
                        rendering_patches.squeeze(0) * 2 - 1,
                        gt_patches.squeeze(0) * 2 - 1,
                    ).mean()
       
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))  + args.lambda_lpips * lpips_loss

        loss.backward()

        update_inpaint_density(rgb_densify, iteration, gaussians, viewspace_point_tensor,
                               visibility_filter, radii, opt, cameras_extent)
                
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)

        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(10)
    progress_bar.close()

    with torch.no_grad():
        density_audit("rgb_before_gate")
        tmp_gaussians = copy.deepcopy(gaussians) 
        support_view_index = int(getattr(args, "fusion_seed_frame", 0))
        if not 0 <= support_view_index < len(views):
            raise ValueError(
                f"fusion_seed_frame {support_view_index} has no matching virtual view"
            )
        projected_gaussian = get_projected_gaussians(
            tmp_gaussians,
            views[support_view_index],
            supp_ply_path=getattr(args, "gate_support_ply", None) or args.supp_ply,
        )
        p_inside_obj_mask = projected_gaussian["p_inside_obj_mask"]

        gaussians._xyz[:gaussians.sub_feature_num]           = removal_gaussian._xyz
        gaussians._features_dc[:gaussians.sub_feature_num]   = removal_gaussian._features_dc
        gaussians._features_rest[:gaussians.sub_feature_num] = removal_gaussian._features_rest
        gaussians._opacity[:gaussians.sub_feature_num]       = removal_gaussian._opacity
        gaussians._scaling[:gaussians.sub_feature_num]       = removal_gaussian._scaling 
        gaussians._rotation[:gaussians.sub_feature_num]      = removal_gaussian._rotation
        gaussians._objects_dc[:gaussians.sub_feature_num]    = removal_gaussian._objects_dc

        fields_to_update = [
        "_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "_objects_dc"
        ]

        for field in fields_to_update:
            getattr(gaussians, field)[p_inside_obj_mask] = getattr(tmp_gaussians, field)[p_inside_obj_mask]

    # save gaussians
    default_output = Path(model_path) / "point_cloud_object_inpaint_virtual" / f"iteration_{iterations}" / "point_cloud.ply"
    output_ply = Path(getattr(args, "inpaint_output_ply", None) or default_output).expanduser().resolve()
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(output_ply))

    # recover surrounding objects back
    target_ids = args.target_id or []
    selected_ids = args.select_obj_id or []
    if args.surrounding_ids and len(target_ids) < len(selected_ids):
        print(f"\nCombine objects{args.surrounding_ids} back.")
        from tools.combine_gaussian_scene import combine_gaussian
        gaussians = combine_gaussian(
            dataset,
            f"_object_inpaint_virtual/iteration_{iterations}/point_cloud.ply",
            pipeline,
            args.surrounding_ids,
            source_iteration=getattr(args, "removal_source_iteration", None) or int(iteration),
            base_ply=output_ply,
            removed_objects_root=getattr(args, "removed_objects_root", None),
            output_ply=output_ply,
            filter_objects=not getattr(args, "skip_surrounding_filter", False),
        )

    # Optional receipt only: no change to the seed, RGB optimizer or gate.
    context_path = getattr(args, "local_geometry_context", None)
    density_audit("rgb_after_gate")
    if context_path:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/paintmesh"))
        from local_geometry_io import save_rgb_receipt
        editable = p_inside_obj_mask.detach().cpu().numpy().astype(bool)
        final_count = len(gaussians.get_xyz)
        if final_count < len(editable):
            raise ValueError("surrounding recovery unexpectedly removed Stage 5a rows")
        # combine_gaussian appends recovered surrounding rows; they are frozen.
        editable = np.pad(editable, (0, final_count - len(editable)), constant_values=False)
        gate = {
            "mask": projected_gaussian["gate_mask"].astype(bool),
            "projection": projected_gaussian["gate_projection"].detach().cpu().numpy(),
            "distance_threshold": np.array(projected_gaussian["gate_distance_threshold"]),
        }
        save_rgb_receipt(context_path, output_ply, editable, gate)

    return gaussians

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, args):
    """
    Args:
        name: "test" or "train" or "inpaint"

    """
    print(f"\nIteration is {iteration}")
    save_folder = os.path.join(model_path, name, "ours{}".format(iteration))

    render_path = os.path.join(save_folder, "renders")
    gts_path = os.path.join(save_folder, "gt")
    depth_path=os.path.join(save_folder, "depth")
    depth_original_path=os.path.join(model_path, name, f"ours{iteration}", "depth")

    makedirs(save_folder, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    with open(os.path.join(save_folder, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)
        pred_obj_mask = torch.argmax(logits,dim=0)
        pred_obj_color_mask = visualize_obj(pred_obj_mask.cpu().numpy().astype(np.uint8))

        gt_objects = view.objects

        if gt_objects == None:
            pass
        else:
            gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))
      
        depth=results["depth_3dgs"].squeeze(0).detach().cpu().numpy()
        np.save(os.path.join(depth_path, view.image_name+".npy"),depth)
        
        if name=="inpaint":
            depth_max = depth.max()
            depth_min = depth.min()
        else:
            depth_max = np.load(os.path.join(depth_original_path, view.image_name+".npy")).max()
            depth_min = np.load(os.path.join(depth_original_path, view.image_name+".npy")).min()

        depth = (depth - depth_min) / (depth_max - depth_min)
        depth = (depth * 255.0).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(depth_path, view.image_name + ".png"), depth)

        pred_obj_mask = pred_obj_mask.cpu().numpy().astype(np.uint8)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))


def _find_virtual_image(directory, image_name):
    root = Path(directory).expanduser().resolve()
    matches = [root / f"{image_name}{suffix}" for suffix in IMAGE_EXTENSIONS]
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"virtual RGB image is missing for {image_name}: {root}")
    if len(matches) > 1:
        raise ValueError(
            f"multiple virtual RGB images match {image_name}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _find_virtual_mask(directory, image_name):
    """Prefer LaMa's explicit mask name, with a legacy-name fallback.

    The color-input directory contains both the source RGB basename and its
    ``*_mask`` companion, so a legacy basename is considered only when the
    explicit mask does not exist.
    """

    root = Path(directory).expanduser().resolve()
    canonical = [
        root / f"{image_name}_mask.png",
        root / f"{image_name}_mask.PNG",
    ]
    legacy = [
        root / f"{image_name}.png",
        root / f"{image_name}.PNG",
    ]
    matches = [path for path in canonical if path.is_file()]
    convention = "canonical"
    if not matches:
        matches = [path for path in legacy if path.is_file()]
        convention = "legacy"
    if not matches:
        raise FileNotFoundError(
            f"virtual mask is missing for {image_name} in {root}; expected "
            f"{image_name}_mask.png (or legacy {image_name}.png)"
        )
    if len(matches) > 1:
        raise ValueError(
            f"multiple {convention} virtual masks match {image_name}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _load_virtual_mask(path, expected_shape):
    mask_path = Path(path).expanduser().resolve(strict=True)
    with Image.open(mask_path) as image:
        labels = np.asarray(image)
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    if labels.ndim != 2 or labels.shape != expected_shape:
        raise ValueError(
            f"virtual mask shape mismatch for {mask_path}: {labels.shape}, "
            f"expected {expected_shape}"
        )
    if not (np.issubdtype(labels.dtype, np.integer) or labels.dtype == np.bool_):
        raise ValueError(f"virtual mask must contain integer labels: {mask_path}")
    mask = labels != 0
    if not mask.any():
        raise ValueError(f"virtual mask is empty: {mask_path}")
    if mask.all():
        raise ValueError(f"virtual mask covers the full frame: {mask_path}")
    return np.ascontiguousarray(mask.astype(np.uint8) * 255)


def _fallback_inpaint_views(views, circle_radius):
    if circle_radius is None or not np.isfinite(circle_radius) or circle_radius <= 0:
        raise ValueError("circle_radius must be finite and positive")
    base_view = views[0]
    poses = generate_ellipse_path(
        views,
        n_frames=FRAME_COUNT,
        is_circle=True,
        circle_radius=circle_radius,
    )
    virtual_views = []
    for index, pose in enumerate(tqdm(poses, desc="\nReplace real virtual camera views")):
        view = copy.deepcopy(base_view)
        view.R = pose[:3, :3].T
        view.T = pose[:3, 3]
        view.world_view_transform = torch.as_tensor(
            getWorld2View2(view.R, view.T, view.trans, view.scale),
            dtype=base_view.world_view_transform.dtype,
            device=base_view.world_view_transform.device,
        ).transpose(0, 1)
        view.projection_matrix = getProjectionMatrix(
            znear=view.znear,
            zfar=view.zfar,
            fovX=view.FoVx,
            fovY=view.FoVy,
        ).to(
            device=base_view.projection_matrix.device,
            dtype=base_view.projection_matrix.dtype,
        ).transpose(0, 1)
        view.full_proj_transform = (
            view.world_view_transform.unsqueeze(0)
            .bmm(view.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        view.image_name = f"{index:05d}"
        virtual_views.append(view)
    return virtual_views


def _prepare_inpaint_views(dataset, scene, args, config):
    from utils.virtual_camera_manifest import require_declared_camera_manifest
    require_declared_camera_manifest(config, getattr(args, "camera_manifest", None))
    train_views = scene.getTrainCameras()
    if not train_views:
        raise ValueError("at least one training camera is required")
    if getattr(args, "camera_manifest", None):
        payload = load_virtual_camera_manifest(
            args.camera_manifest,
            expected_iteration=int(scene.loaded_iter),
        )
        virtual_views = virtual_views_from_manifest(train_views[0], payload)
    else:
        virtual_views = _fallback_inpaint_views(train_views, args.circle_radius)

    virtual_root = Path(
        getattr(args, "virtual_data_root", None) or dataset.source_path
    ).expanduser().resolve()
    configured_mask = config.get("object_path")
    configured_images = config.get("images")
    if not isinstance(configured_mask, str) or not configured_mask:
        raise ValueError("object-inpaint config must define object_path")
    if not isinstance(configured_images, str) or not configured_images:
        raise ValueError("object-inpaint config must define images")
    mask_root = Path(
        getattr(args, "inpaint_mask_dir", None) or virtual_root / configured_mask
    ).expanduser().resolve()
    image_root = Path(
        getattr(args, "inpainted_color_dir", None) or virtual_root / configured_images
    ).expanduser().resolve()

    for view in virtual_views:
        expected_shape = (int(view.image_height), int(view.image_width))
        mask_path = _find_virtual_mask(mask_root, view.image_name)
        mask = _load_virtual_mask(mask_path, expected_shape)
        image_path = _find_virtual_image(image_root, view.image_name)
        with Image.open(image_path) as image_file:
            if image_file.size != (view.image_width, view.image_height):
                raise ValueError(
                    f"virtual RGB size mismatch for {image_path}: {image_file.size}, "
                    f"expected {(view.image_width, view.image_height)}"
                )
            image = image_file.convert("RGB")
            image_tensor = PILtoTorch(image, image_file.size)
        view.objects = torch.from_numpy(mask).to(view.data_device)
        view.original_image = image_tensor[:3, ...].clamp(0.0, 1.0).to(view.data_device)
    return virtual_views


def _concrete_source_iteration(model_path, requested_iteration):
    if requested_iteration is not None and int(requested_iteration) > 0:
        return int(requested_iteration)
    point_cloud_root = Path(model_path).expanduser().resolve() / "point_cloud"
    candidates = []
    if point_cloud_root.is_dir():
        for path in point_cloud_root.iterdir():
            suffix = path.name.removeprefix("iteration_")
            if path.is_dir() and path.name.startswith("iteration_") and suffix.isdecimal():
                candidates.append(int(suffix))
    if not candidates:
        raise ValueError(f"cannot resolve a source iteration from {point_cloud_root}")
    return max(candidates)


def _prepare_temporary_ply(args):
    output_default = (
        Path(args.model_path)
        / "point_cloud_object_inpaint_virtual"
        / f"iteration_{args.finetune_iteration}"
        / "point_cloud.ply"
    )
    output_path = Path(
        getattr(args, "inpaint_output_ply", None) or output_default
    ).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.inpaint_output_ply = str(output_path)
    requested_temporary = getattr(args, "temp_ply", None)
    if requested_temporary:
        temporary = Path(requested_temporary).expanduser().resolve()
        temporary.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            raise FileExistsError(f"temporary PLY already exists: {temporary}")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent, prefix=".inpaint-init-", suffix=".ply"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
    if temporary == output_path:
        raise ValueError("temp_ply and inpaint_output_ply must be different paths")
    args.temp_ply = str(temporary)
    return temporary


def inpaint(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, removal_thresh : float,  finetune_iteration: int, render_video : bool, args, config):
    """
    
    
    """
    scene_json_path = os.path.join(args.config_file)
    with open(scene_json_path, "r") as f:
        mask_info = json.load(f)
    args.circle_radius = mask_info.get("circle_radius")
    print("circle_radius: ", args.circle_radius)

    # 1. load gaussian checkpoint
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    dataset.num_classes = args.num_classes
    print("Num classes: ", dataset.num_classes)
    
    classifier = torch.nn.Conv2d(gaussians.num_objects, dataset.num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    virtual_pose_list = _prepare_inpaint_views(dataset, scene, args, config)

    # 2. inpaint selected object
    gaussians = finetune_inpaint(
        args,
        opt,
        dataset,
        dataset.model_path,
        scene.loaded_iter,
        virtual_pose_list,
        gaussians,
        pipeline,
        background,
        classifier,
        select_obj_id,
        scene.cameras_extent,
        removal_thresh,
        finetune_iteration,
    )
   
    # 3. render new result
    output_iteration = f'_object_inpaint_virtual/iteration_{finetune_iteration}'
    
    if render_video:
        render_video_func_wriva(dataset.source_path, dataset.model_path, output_iteration, scene.getTrainCameras(),
                                gaussians, pipeline, background, classifier, fps = 30)

    with torch.no_grad():
        if not skip_train:
            render_set(dataset.model_path, "train", output_iteration, scene.getTrainCameras(), gaussians, pipeline, background, classifier, args)

        if not skip_test:
            render_set(dataset.model_path, "test", output_iteration, scene.getTestCameras(), gaussians, pipeline, background, classifier, args)

        if "inpaint360" in args.source_path:
            render_set(dataset.model_path, "inpaint", output_iteration, scene.getInpaintCameras(), gaussians, pipeline, background, classifier, args)

# Main Procedure
if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--source_iteration", default=None, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument('--temp_ply', type=str, default=None, help='Run-local temporary Gaussian PLY; a unique path is generated by default.')
    parser.add_argument('--supp_ply', type=str, default=None, help='Explicit fused RGB-D support PLY.')
    parser.add_argument('--init_support_ply', type=str, default=None)
    parser.add_argument('--gate_support_ply', type=str, default=None)
    parser.add_argument('--density_manifest', type=str, default=None)
    parser.add_argument('--disable_rgb_densify', dest='rgb_densify', action='store_false',
                        default=True, help='Disable Stage 5a clone/split/prune and density statistics; retain optimizer and gate.')
    parser.add_argument('--fusion_dir', type=str, default=None, help='Directory containing per-frame fused support PLYs.')
    parser.add_argument('--fusion_seed_frame', type=int, default=4, help='Virtual frame used to initialize inpainted Gaussians.')
    parser.add_argument('--inpaint_output_ply', type=str, default=None, help='Explicit final inpainted Gaussian PLY.')
    parser.add_argument('--local_geometry_context', type=str, default=None,
                        help='Optional PaintMesh Stage 5a receipt request; does not change RGB training.')
    parser.add_argument('--virtual_data_root', type=str, default=None)
    parser.add_argument('--inpainted_color_dir', type=str, default=None)
    parser.add_argument('--inpaint_mask_dir', type=str, default=None)
    parser.add_argument('--camera_manifest', type=str, default=None)
    parser.add_argument('--removal_source_iteration', type=int, default=None)
    parser.add_argument('--removed_objects_root', type=str, default=None)
    parser.add_argument('--skip_surrounding_filter', action='store_true')
    parser.add_argument('--nb_points', type=int, default=100, help='Number of points for the remove_radius_outlier function.')
    parser.add_argument('--threshold', type=float, default=1.0, help='Threshold for the similar_points_tree function.')
    parser.add_argument('--radius', type=float, default=0.1, help='Radius for the remove_radius_outlier function.')
    parser.add_argument("--config_file", type=str, default="config/object_inpaint/inpaint360/doppelherz.json", help="Path to the configuration file")
    args = get_combined_args(parser)

    # Read and parse the configuration file
    with open(args.config_file, 'r') as file:
        config = json.load(file)
    args.removal_thresh = config.get("removal_thresh")
    from utils.virtual_camera_manifest import require_declared_camera_manifest
    require_declared_camera_manifest(config, getattr(args, "camera_manifest", None))
    args.select_obj_id = config.get("select_obj_id")
    args.target_id = config.get("target_id")
    args.surrounding_ids = config.get("surrounding_ids")

    # Respect both the workspace cfg_args value and an explicit ``--images``.
    args.images = getattr(args, "images", None) or "images"
    args.object_path = config.get("object_path")

    args.lambda_dssim = config.get("lambda_dssim")                
    args.finetune_iteration = config.get("finetune_iteration")
    args.opacity_init = config.get("opacity_init", 0.1)
    args.lambda_lpips = config.get("lambda_lpips")         
    requested_source_iteration = getattr(args, "source_iteration", None)
    source_iteration = _concrete_source_iteration(
        args.model_path,
        requested_source_iteration
        if requested_source_iteration is not None
        else args.iteration,
    )
    args.removal_source_iteration = (
        getattr(args, "removal_source_iteration", None) or source_iteration
    )
    frame_count = (load_virtual_camera_manifest(args.camera_manifest)["frame_count"]
                   if getattr(args, "camera_manifest", None) else FRAME_COUNT)
    if not 0 <= args.fusion_seed_frame < frame_count:
        parser.error(f"fusion_seed_frame must be in [0, {frame_count - 1}]")
    fusion_root = Path(
        getattr(args, "fusion_dir", None)
        or Path(args.model_path)
        / "virtual"
        / "ours_object_removal"
        / f"iteration_{source_iteration}"
        / "fused_mask_col_dep_ply"
    ).expanduser().resolve()
    requested_support = getattr(args, "supp_ply", None)
    args.supp_ply = str(
        Path(requested_support).expanduser().resolve()
        if requested_support
        else fusion_root / f"{args.fusion_seed_frame:05d}.ply"
    )
    if not Path(args.supp_ply).is_file():
        parser.error(f"support PLY does not exist: {args.supp_ply}")
    if args.density_manifest:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/paintmesh"))
        from support_density_io import validate_density
        density = validate_density(args.density_manifest)
        if (not args.init_support_ply or not args.gate_support_ply or
                Path(args.init_support_ply).resolve() != Path(density["outputs"]["support"]["path"]) or
                Path(args.gate_support_ply).resolve() != Path(density["inputs"]["support"]["path"])):
            parser.error("density initialization/gate support identity mismatch")
        args.supp_ply = args.init_support_ply
    elif args.init_support_ply or args.gate_support_ply:
        parser.error("separate init/gate support requires a validated density manifest")
    temporary_ply = _prepare_temporary_ply(args)
    torch.cuda.empty_cache()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    try:
        inpaint(model.extract(args), source_iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), args.select_obj_id, args.removal_thresh, args.finetune_iteration, args.render_video, args, config)
    finally:
        temporary_ply.unlink(missing_ok=True)
