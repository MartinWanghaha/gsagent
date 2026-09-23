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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import json
import open3d as o3d
from pathlib import Path
from utils.point_utils import ply_color_fusion
from render import id2rgb

import cv2

from scipy.spatial import Delaunay, QhullError, cKDTree
from render import visualize_obj, render_video_func_wriva
from tools.init_configs import _atomic_write_text


def get_hull_size(filtered_points, percentile=80):

    if len(filtered_points) == 0:
        return 0.0
    
    center = np.mean(filtered_points, axis=0)
    distances = np.linalg.norm(filtered_points - center, axis=1)
    object_radius = np.percentile(distances, percentile)
    
    return object_radius

def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0, expand_ratio=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the 
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.   3xyz
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Smaller values will classify more points as outliers.
    - expand_ratio (float): expand 

    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull 
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()

    if len(masked_points) == 0:
        raise ValueError(
            "cannot build an object hull from an empty mask; lower "
            "removal_thresh or verify the selected instance ID"
        )

    # Remove outliers if the option is selected.
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR)) 
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]            
    else:
        filtered_masked_points = masked_points


    def supports_volume(points):
        return len(points) >= 4 and np.linalg.matrix_rank(
            points - np.mean(points, axis=0, keepdims=True)
        ) >= 3

    # IQR filtering can leave too few points even when the original selection
    # is usable. Prefer the unfiltered selection in that case. If neither set
    # spans a 3D volume, preserve the probability mask instead of asking Qhull
    # to construct an undefined convex volume.
    if not supports_volume(filtered_masked_points) and supports_volume(masked_points):
        filtered_masked_points = masked_points

    if expand_ratio > 1.0:
        center = np.mean(filtered_masked_points, axis=0)
        expanded_points = center + (filtered_masked_points - center) * expand_ratio
    else:
        expanded_points = filtered_masked_points

    # caculate object radius
    object_radius = get_hull_size(filtered_masked_points)

    if not supports_volume(expanded_points):
        return mask.bool().clone(), object_radius

    # QJ perturbs nearly-degenerate input just enough to make Qhull robust.
    # Try the exact hull first because a minimal tetrahedron is valid input but
    # Qhull's joggled mode may request an additional point.
    try:
        delaunay = Delaunay(expanded_points)
    except QhullError:
        try:
            delaunay = Delaunay(expanded_points, qhull_options="QJ")
        except QhullError:
            return mask.bool().clone(), object_radius

    # Determine which points from the original point cloud are inside the convex hull     
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.as_tensor(
        points_inside_hull_mask,
        dtype=torch.bool,
        device=point_cloud.device,
    )

    return inside_hull_tensor_mask, object_radius


def visualize_and_save_ply(points, labels, ply_path, mask=None):
    """
    Args:
      points: (N, 3)
      labels: (N,)
      ply_path: PLY
      mask: (N,) ，
    """
    labels = np.clip(labels, 0, 255).astype(np.int32)

    # ** `id2rgb()` **
    colors = np.array([id2rgb(i) for i in range(256)])  
    point_colors = colors[labels]  

    if point_colors.ndim == 3:  
        point_colors = point_colors.squeeze(1)
    elif point_colors.ndim == 1 or point_colors.shape[1] != 3:
        point_colors = np.tile(point_colors[:, None], (1, 3))

    if points.shape[0] != point_colors.shape[0]:
        print(f" Shape mismatch: points {points.shape[0]} vs colors {point_colors.shape[0]}")
        return

    if mask is None:
        mask = np.ones(points.shape[0], dtype=bool)

    valid_points = points[mask]
    valid_colors = point_colors[mask]

    ply_color_fusion(valid_points, valid_colors, ply_path)

    print(f"✅ color point cloud saved to {ply_path}")

def visualize_rgb_and_save_ply(points, features_dc, ply_path, mask=None):

    C0 = 0.28209479177387814
    if features_dc.dim() == 3:
        features_dc = features_dc.squeeze(1)

    rgb = 0.5 + C0 * features_dc.detach()
    rgb = torch.clamp(rgb, 0.0, 1.0)  
    colors = (rgb * 255).cpu().numpy().astype(np.uint8)
    colors = colors[:, [2, 1, 0]]  

    if mask is None:
        mask = np.ones(points.shape[0], dtype=bool)

    valid_points = points[mask]
    valid_colors = colors[mask]
    print(f"🎨 RGB via SH DC: {valid_points.shape[0]} points → {ply_path}")

    ply_color_fusion(valid_points, valid_colors, ply_path)


def removal_setup(
    model_path,
    iteration,
    gaussians,
    classifier,
    selected_obj_ids,
    removal_thresh,
    skip_debug_ply=False,
):
    """
    Args:
        model_path: Directory containing the semantic Gaussian model.
        iteration: Loaded Gaussian iteration.
        gaussians: Scene Gaussian model to edit in place.
        classifier: Trained semantic classifier.
        selected_obj_ids: Target IDs plus any temporarily removed surroundings.
        removal_thresh: Minimum classifier probability for object membership.
        skip_debug_ply: Do not export diagnostic semantic/RGB point clouds.
    """
    selected_obj_ids = torch.tensor(selected_obj_ids).cuda()

    masks_per_obj = dict()

    
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))    
        prob_obj3d = torch.softmax(logits3d,dim=0)     
        max_indices = prob_obj3d.argmax(dim=0) if not skip_debug_ply else None

        for obj_id in selected_obj_ids:
            obj_id_int = int(obj_id.item())
            obj_prob = prob_obj3d[obj_id, :, :]

            mask = obj_prob > removal_thresh
            # mask = (obj_prob > removal_thresh) | (max_indices == obj_id)
            mask3d = mask.squeeze()

            # here we calculate object_radius
            mask3d_convex, object_radius = points_inside_convex_hull(gaussians._xyz.detach(), mask3d, remove_outliers=True, outlier_factor=1.0)
           
            mask3d = torch.logical_or(mask3d,mask3d_convex)
            mask3d = mask3d.float()[:,None,None]

            masks_per_obj[obj_id_int] = {
            "mask":  mask.float()[:,None],
            "mask3d": mask3d,
            "radius": object_radius
            }

    if len(selected_obj_ids)>1: 
        combined_all_targets_mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device='cuda')
        with torch.no_grad():
            for obj_id in selected_obj_ids:
                obj_id_int = int(obj_id.item())
                current_mask = masks_per_obj[obj_id_int]["mask3d"].squeeze()
                combined_all_targets_mask |= current_mask.bool()
                _, total_radius = points_inside_convex_hull(gaussians._xyz.detach(), combined_all_targets_mask, remove_outliers=True, outlier_factor=1.0)

    ply_vis_folder_path = os.path.join(model_path, "point_cloud_vis")
    if not skip_debug_ply:
        os.makedirs(ply_vis_folder_path, exist_ok=True)
        # Visualize the classifier's most likely object ID for each Gaussian.
        visualize_and_save_ply(
            points=gaussians._xyz.detach().cpu().numpy(), 
            labels=max_indices.detach().cpu().numpy(),
            ply_path=ply_vis_folder_path + "/prob_obj3d_visualization.ply")

    objects_model_dict = gaussians.removal_setup(masks_per_obj)

    if not skip_debug_ply:
        visualize_rgb_and_save_ply(
            points=gaussians.get_xyz.detach().cpu().numpy(),
            features_dc=gaussians._features_dc.detach(),
            ply_path=os.path.join(ply_vis_folder_path, "removed_rgb_visualization.ply"),
        )
    
    # save remaining scene gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud_object_removal/iteration_{}".format(iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # save objects in this scene
    for obj_id, object_gaussian in objects_model_dict.items():
        object_gaussian_path = os.path.join(point_cloud_path, f"point_cloud_{obj_id}.ply")
        object_gaussian.save_ply(object_gaussian_path)
        print(f"The object {obj_id} is saved at {object_gaussian_path} ")
   
    # record target_object_radius, for later virtual radius
    if len(selected_obj_ids) > 1:
        target_object_radius = total_radius
    else:
        target_object_radius = masks_per_obj[selected_obj_ids[0].item()]["radius"]

    return gaussians, objects_model_dict, target_object_radius


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    classifier,
    reference_model_path=None,
):
    """Render removal diagnostics using the semantic model's depth range."""
    iteration_step = iteration.split('_')[-1]
    reference_model_path = reference_model_path or model_path
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    gt_colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred")
    pred_obj_color_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred_color")
    depth_path=os.path.join(model_path, name, "ours{}".format(iteration), "depth")                   
    depth_original_path = os.path.join(
        reference_model_path,
        name,
        "ours_{}".format(iteration_step),
        "depth",
    )
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(pred_obj_color_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)      
        pred_obj_mask = torch.argmax(logits,dim=0)
        pred_obj_color_mask = visualize_obj(pred_obj_mask.cpu().numpy().astype(np.uint8))
        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

        depth = results["depth_3dgs"].squeeze(0).detach().cpu().numpy()
        np.save(os.path.join(depth_path, view.image_name + ".npy"), depth)
        reference_depth_file = os.path.join(
            depth_original_path, view.image_name + ".npy"
        )
        if not os.path.isfile(reference_depth_file):
            raise FileNotFoundError(
                "Removal diagnostic rendering needs the semantic-model reference depth: "
                f"{reference_depth_file}"
            )
        reference_depth = np.load(reference_depth_file, mmap_mode="r")
        depth_max = float(reference_depth.max())
        depth_min = float(reference_depth.min())
        depth_range = depth_max - depth_min
        if depth_range > np.finfo(np.float32).eps:
            depth = np.clip((depth - depth_min) / depth_range, 0.0, 1.0)
        else:
            depth = np.zeros_like(depth)
        del reference_depth
        depth = (depth * 255.0).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(depth_path, view.image_name + ".png"), depth)

        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, view.image_name + ".png"))

        pred_obj_mask = pred_obj_mask.cpu().numpy().astype(np.uint8)
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, view.image_name + ".png"))
        Image.fromarray(pred_obj_color_mask).save(os.path.join(pred_obj_color_path, view.image_name + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

def update_config_radius(removal_config_path, radius):
    """
    Update the target object radius in both removal and inpainting configuration files.

    Args:
        removal_config_path (str): Path to the object removal configuration JSON file.
        radius (float): The new radius value to be set for the target object.
    """
    target_paths = [
        removal_config_path,
        removal_config_path.replace("object_removal", "object_inpaint")
    ]
    
    for path in target_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            data["target_object_radius"] = round(radius, 4)
            
            import re
            json_str = json.dumps(data, indent=4)
            json_str = re.sub(r'\[\s+([\d, \s]+)\s+\]', 
                              lambda m: "[" + re.sub(r'\s+', '', m.group(1)) + "]", 
                              json_str)

            _atomic_write_text(Path(path), json_str)

def combine_gaussian(dataset, iteration, gaussians, objects_model_dict, target_id):
    """ 
    Combine the base scene with specific objects from the dictionary.

    Args:
        dataset: Model parameters.
        iteration (int): Current iteration for path naming.
        gaussians: The base GaussianModel (already contains the background).
        objects_model_dict (dict): Dictionary containing {obj_id: GaussianModel}.
        target_id (list): The list of IDs that were targeted for removal (exclude surrounding object).

    Returns:
        GaussianModel: The combined Gaussian model.
    """
    # here we assume only first object is to be targeted removed 
    
    print(f"Combining scene: Keeping background and merging all objects except ID: {target_id}")

    for obj_id, gaussians_object in objects_model_dict.items():
        if obj_id in target_id:
            continue  
            
        print(f"Merging Object {obj_id} back into the scene...")
        
        with torch.no_grad():
            gaussians._xyz = torch.cat([gaussians._xyz, gaussians_object._xyz], dim=0)
            gaussians._features_dc = torch.cat([gaussians._features_dc, gaussians_object._features_dc], dim=0)
            gaussians._features_rest = torch.cat([gaussians._features_rest, gaussians_object._features_rest], dim=0)
            gaussians._opacity = torch.cat([gaussians._opacity, gaussians_object._opacity], dim=0)
            gaussians._scaling = torch.cat([gaussians._scaling, gaussians_object._scaling], dim=0)
            gaussians._rotation = torch.cat([gaussians._rotation, gaussians_object._rotation], dim=0)

            if hasattr(gaussians, "_objects_dc") and hasattr(gaussians_object, "_objects_dc"):
                gaussians._objects_dc = torch.cat([gaussians._objects_dc, gaussians_object._objects_dc], dim=0)

    combined_gaussian_folder = os.path.join(dataset.model_path, f"point_cloud_object_removal/iteration_{iteration}_removal_target")
    os.makedirs(combined_gaussian_folder, exist_ok=True)
    
    combined_gaussian_path = os.path.join(combined_gaussian_folder, "point_cloud.ply")

    gaussians.save_ply(combined_gaussian_path)
    print(f"✅ Success: Combined scene saved at {combined_gaussian_path}")

    return gaussians


def removal(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    opt: OptimizationParams,
    select_obj_id,
    target_id,
    removal_thresh: float,
    config_file: str,
    render_video: bool = False,
    render_object_videos: bool = False,
    skip_debug_ply: bool = False,
    reference_model_path: str = None,
):
    # 1. load gaussian checkpoint
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    num_classes = dataset.num_classes

    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)  
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth"))) 
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]         
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 2. remove selected object
    gaussians, objects_model_dict, target_object_radius = removal_setup(
        dataset.model_path,
        scene.loaded_iter,
        gaussians,
        classifier,
        select_obj_id,
        removal_thresh,
        skip_debug_ply=skip_debug_ply,
    )

    # scene_json_path = os.path.join(dataset.source_path, "associated_hqsam", "scene.json")
    update_config_radius(config_file, target_object_radius)
    
    # Rendering every isolated object is useful for inspection but expensive;
    # keep it behind an explicit opt-in flag.
    if render_object_videos:
        for obj_id, object_gaussian in objects_model_dict.items():
            loaded_iter = "ours_removed_obj_{}".format(obj_id)   
            render_video_func_wriva(dataset.source_path, dataset.model_path, loaded_iter, scene.getTrainCameras(),
                                    object_gaussian, pipeline, background, classifier, fps = 30)

    # Restore surrounding objects while keeping every actual target removed.
    # removal_setup already persisted the target+surrounding background model;
    # combine_gaussian persists the final target-only removal model separately.
    selected_ids = set(select_obj_id)
    target_ids = set(target_id)
    if selected_ids - target_ids:
        gaussians = combine_gaussian(
            dataset,
            str(scene.loaded_iter),
            gaussians,
            objects_model_dict,
            target_id,
        )

    # --render_video always represents the final result (target removed,
    # surrounding objects restored), independent of whether surrounding IDs
    # were supplied.
    if render_video:
        loaded_iter = "ours_object_removal_{}".format(scene.loaded_iter)
        render_video_func_wriva(dataset.source_path, dataset.model_path, loaded_iter, scene.getTrainCameras(),
                                gaussians, pipeline, background, classifier, fps = 30)

    with torch.no_grad():
        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                "_object_removal/iteration_" + str(scene.loaded_iter),
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                classifier,
                reference_model_path,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                "_object_removal/iteration_" + str(scene.loaded_iter),
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                classifier,
                reference_model_path,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument(
        "--render_object_videos",
        action="store_true",
        help="Render a separate diagnostic video for every selected object (disabled by default).",
    )
    parser.add_argument(
        "--skip_debug_ply",
        action="store_true",
        help="Skip large point_cloud_vis diagnostic PLY files.",
    )
    parser.add_argument(
        "--reference_model_path",
        type=str,
        default=None,
        help=(
            "Semantic-model root containing the original train/test reference depth "
            "used to normalize optional removal diagnostics."
        ),
    )
    parser.add_argument("--config_file", type=str, default="config/object_removal/inpaint360/doppelherz.json", help="Path to the configuration file") 

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Read and parse the configuration file
    with open(args.config_file, 'r') as file:
        config = json.load(file)

    scene_json_path = os.path.join(args.source_path, "associated_hqsam", "scene.json")
    with open(scene_json_path, "r") as f:
        scene_info = json.load(f)
    args.num_classes = scene_info.get("num_classes")
    args.removal_thresh = config.get("removal_thresh")   
    args.select_obj_id = config.get("select_obj_id")
    args.target_id = config.get("target_id")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    removal(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        opt.extract(args),
        args.select_obj_id,
        args.target_id,
        args.removal_thresh,
        args.config_file,
        render_video=args.render_video,
        render_object_videos=args.render_object_videos,
        skip_debug_ply=args.skip_debug_ply,
        reference_model_path=args.reference_model_path,
    )
