# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.point_utils import create_point_cloud, ply_color_fusion, get_intrinsics
from utils.pose_utils import generate_ellipse_path
from utils.graphics_utils import getWorld2View2
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
import json


def id2rgb(id, max_num_obj=np.iinfo(np.uint16).max):
  
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 2.7182818284   
    h = ((id * golden_ratio) % 1)      
    s = 0.5 + (id % 2) * 0.5           
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)   
    if id==0:                              
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    if objects.ndim < 2 or not np.issubdtype(objects.dtype, np.integer):
        raise ValueError(
            f"Expected an integer instance mask, got shape={objects.shape}, "
            f"dtype={objects.dtype}."
        )
    if objects.size and (objects.min() < 0 or objects.max() > np.iinfo(np.uint16).max):
        raise ValueError("Instance mask IDs must fit in uint16.")

    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)             
        rgb_mask[objects == id] = colored_mask      
    return rgb_mask


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    classifier,
    skip_fused_ply=False,
    frame_writer=None,
):
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")           
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    pred_obj_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred_color")
    depth_path=os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    fused_full_col_dep_ply_path=os.path.join(model_path, name, "ours_{}".format(iteration), "fused_full_col_dep_ply") 
   
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_obj_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(pred_obj_color_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    if not skip_fused_ply:
        makedirs(fused_full_col_dep_ply_path, exist_ok=True)

    if not skip_fused_ply:
        print(f"\nSaving per-view fused RGB-D point clouds at {fused_full_col_dep_ply_path}.\n")
    for i, view in enumerate(tqdm(views, desc="Rendering progress")):
        idx = i + 1  
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]              
        rendering_obj = results["render_object"]   
        
        logits = classifier(rendering_obj)         
        pred_obj_mask = torch.argmax(logits,dim=0)
        pred_obj_labels = pred_obj_mask.cpu().numpy()
        pred_obj_color_mask = visualize_obj(pred_obj_labels)
        
        gt_objects = view.objects
        gt_obj_mask = visualize_obj(gt_objects.cpu().numpy())

        depth = results["depth_3dgs"].squeeze(0).detach().cpu().numpy()
       
        np.save(os.path.join(depth_path, view.image_name+".npy"),depth)
        depth_range = depth.max() - depth.min()
        depth_vis = np.zeros_like(depth, dtype=np.uint8)
        if depth_range > 0:
            depth_vis = ((depth - depth.min()) / depth_range * 255.0).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(depth_path, view.image_name + ".png"), depth_vis)
        
        Image.fromarray(gt_obj_mask).save(os.path.join(gt_obj_path, view.image_name + ".png"))
        Image.fromarray(pred_obj_labels.astype(np.uint16, copy=False)).save(
            os.path.join(pred_obj_path, view.image_name + ".png")
        )
        Image.fromarray(pred_obj_color_mask).save(os.path.join(pred_obj_color_path, view.image_name + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        if not skip_fused_ply:
            w2c = np.zeros((4, 4))
            w2c[:3, :3] = view.R.transpose()  # view.R: camera to world
            w2c[:3, 3] = view.T               # view.T: world to camera
            w2c[3, 3] = 1.0
            c2w = np.linalg.inv(w2c)
            intrinsics = get_intrinsics(
                view.image_height, view.image_width, view.FoVx, view.FoVy
            )
            points = create_point_cloud(depth, intrinsics, c2w)
            ply_path = os.path.join(fused_full_col_dep_ply_path, view.image_name + ".ply")
            colors = cv2.imread(os.path.join(gts_path, view.image_name + ".png")).reshape(-1, 3)
            ply_color_fusion(points, colors, ply_path, mask=None)

        if frame_writer is not None:
            frame_writer.write(view, results)



def render_video_func_wriva(source_path, model_path, iteration, views, gaussians, pipeline, background, classifier, fps=30):

    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    print(f"\nThe video will be save in {render_path}")
    makedirs(render_path, exist_ok=True)
    view = views[0]
    render_poses = generate_ellipse_path(views)

    size = (view.original_image.shape[2] * 2, int(view.original_image.shape[1] * 1))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)

    video_images_list = []
    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)

        img = torch.clamp(rendering["render"], min=0., max=1.).cpu()

        rendering_obj = rendering["render_object"] 
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits,dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy()) / 255.
        pred_obj_mask = torch.clamp(torch.tensor(pred_obj_mask), min=0., max=1.).permute(2, 0, 1)

        combined_img = torch.cat([img, pred_obj_mask], dim=2)
        torchvision.utils.save_image(combined_img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (combined_img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1] 
        video_images_list.append(video_img)  

    new_video_images_list = video_images_list

    for video_img in new_video_images_list:
        video_img = video_img[:size[1], :, :]
        final_video.write(video_img)

    final_video.release()


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    render_video: bool,
    skip_fused_ply: bool = False,
):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        scene_json_path = os.path.join(dataset.source_path, dataset.object_path, "scene.json")
        with open(scene_json_path, "r") as f:
            mask_info = json.load(f)
        dataset.num_classes = mask_info.get("num_classes")
        if (
            not isinstance(dataset.num_classes, int)
            or not 1 < dataset.num_classes <= np.iinfo(np.uint16).max
        ):
            raise ValueError(
                f"Invalid num_classes in {scene_json_path}: {dataset.num_classes!r}"
            )
        print("Num classes: ", dataset.num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, dataset.num_classes, kernel_size=1)
        classifier.cuda()
        classifier_path = os.path.join(
            dataset.model_path,
            "point_cloud",
            "iteration_" + str(scene.loaded_iter),
            "classifier.pth",
        )
        try:
            classifier_state = torch.load(
                classifier_path, map_location="cpu", weights_only=True
            )
        except TypeError:  # PyTorch < 2.0 compatibility
            classifier_state = torch.load(classifier_path, map_location="cpu")
        classifier.load_state_dict(classifier_state)
        classifier.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if render_video:
            render_video_func_wriva(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(),
                         gaussians, pipeline, background, classifier, fps = 30)

        if not skip_train:
            render_set(
                dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                gaussians, pipeline, background, classifier, skip_fused_ply,
            )

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            render_set(
                dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                gaussians, pipeline, background, classifier, skip_fused_ply,
            )

        # if "inpaint360" in args.source_path:
        #     from edit_object_inpaint import render_set as render_set_inpaint
        #     print(scene.loaded_iter)
        #     render_set_inpaint(dataset.model_path, "inpaint", scene.loaded_iter, scene.getInpaintCameras(), gaussians, pipeline, background, classifier, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument(
        "--skip_fused_ply",
        action="store_true",
        help="Skip the large per-view RGB-D PLY export while keeping RGB, depth, and semantic renders.",
    )
    args = get_combined_args(parser)

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.render_video,
        args.skip_fused_ply,
    )
