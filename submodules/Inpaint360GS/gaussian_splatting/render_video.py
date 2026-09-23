#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import copy
import cv2

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


# xy circular 
def render_circular_video(model_path, iteration, views, gaussians, pipeline, background, radius=0.5, n_frames=240, fps=30): 
    render_path = os.path.join(model_path, 'video', "ours_circle_{}".format(iteration))
    os.makedirs(render_path, exist_ok=True)

    size = (views[0].original_image.shape[2], int(views[0].original_image.shape[1]))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)

    video_images_list = []
    for idx in tqdm(range(n_frames), desc="Rendering"):
        view = copy.deepcopy(views[0])
        angle = 2 * np.pi * idx / n_frames
        cam = circular_poses(view, radius, angle)
        rendering = render(cam, gaussians, pipeline, background)["render"]

        img = torch.clamp(rendering, min=0., max=1.).cpu()
        torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1] 
        video_images_list.append(video_img) 
   
    for video_img in video_images_list:
        final_video.write(video_img)

    final_video.release()


def render_ellipse_circle_video(model_path, iteration, views, gaussians, pipeline, background, fps=30, is_circle=False):
    """
    
    """
    if is_circle:
        render_path = os.path.join(model_path, 'video', "ours_circle_{}".format(iteration))
    else:
        render_path = os.path.join(model_path, 'video', "ours_ellipse_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]    

    size = (view.original_image.shape[2], int(view.original_image.shape[1]))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)

    poses, _  = generate_ellipse_path(views, n_frames=240, is_circle=is_circle)
    video_images_list = []
    for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)["render"]

        img = torch.clamp(rendering, min=0., max=1.).cpu()
        torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
        video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]  # (1039, 3116, 3)
        video_images_list.append(video_img) 
   
    for video_img in video_images_list:
        final_video.write(video_img)

    final_video.release()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_ellipse_video: bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # by default generate ellipse path, other options include spiral, circular, or other generate_xxx_path function from utils.pose_utils 
        if not skip_ellipse_video:
            render_ellipse_circle_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, is_circle = args.is_circle)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_ellipse_video", action="store_true")
    parser.add_argument("--is_circle", action="store_true")
    parser.add_argument("--skip_gaussians_disturb", action="store_true")
    parser.add_argument("--mean", default=0, type=float)
    parser.add_argument("--std", default=0.03, type=float)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_ellipse_video, args)

    # Render with trajectory. By default ellipse, you can change it to spiral or others trajectory by changing to corresponding function.
    # python render_video.py --source_path data/inpaint360/doppelherz --model_path output/inpaint360/doppelherz --skip_train --skip_test
