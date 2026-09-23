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
import shutil
from scene import Scene
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import json
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_virtual_radius
from render import render_set as render_set_full_scene_stage
# from edit_object_removal import render_set as render_set_removal_stage
import copy
from render import visualize_obj
import torchvision
from PIL import Image
from utils.point_utils import create_point_cloud, ply_color_fusion, get_intrinsics
from utils.virtual_camera_manifest import write_virtual_camera_manifest

try:
    from tools.init_configs import _atomic_write_text
except ModuleNotFoundError:  # Direct ``python tools/virtual_pose.py`` execution.
    from init_configs import _atomic_write_text


def package_tracker_images(renders_dir, tracker_archive=None):
    """Package virtual renders into the requested tracker ZIP archive."""
    if tracker_archive is None:
        assets_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "Segment-and-Track-Anything",
            "assets",
        )
        archive_path = os.path.abspath(os.path.join(assets_dir, "images.zip"))
    else:
        archive_path = os.path.abspath(os.path.expanduser(tracker_archive))
        if not archive_path.endswith(".zip"):
            raise ValueError("--tracker_archive must point to a .zip file")

    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    archive_base = archive_path[:-4]
    generated_path = shutil.make_archive(archive_base, "zip", renders_dir)
    if os.path.abspath(generated_path) != archive_path:
        raise RuntimeError(
            f"Tracker archive was written to {generated_path}, expected {archive_path}"
        )
    return archive_path


def render_set_removal_stage(model_path, name, iteration, views, gaussians, pipeline, background, classifier, frame_writer=None):
    """
    
    """
    print(f"\nIteration is {iteration}")
    iteration_step = iteration.split('_')[-1]
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    gt_colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred")
    pred_obj_color_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred_color")
    depth_path=os.path.join(model_path, name, "ours{}".format(iteration), "depth")               
    depth_original_path=os.path.join(model_path, name, "ours_{}".format(iteration_step), "depth")     
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    makedirs(pred_obj_color_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    fused_vanilla_col_dep_ply_path=os.path.join(model_path, name, "ours{}".format(iteration), "fused_vanilla_col_dep_ply")  
    makedirs(fused_vanilla_col_dep_ply_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)       
        pred_obj_mask = torch.argmax(logits,dim=0)
        pred_obj_color_mask = visualize_obj(pred_obj_mask.cpu().numpy().astype(np.uint8))
        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))
        depth=results["depth_3dgs"].squeeze(0).detach().cpu().numpy()
        np.save(os.path.join(depth_path, view.image_name+".npy"),depth)          

        depth_max = np.load(os.path.join(depth_original_path, view.image_name+".npy")).max()
        depth_min = np.load(os.path.join(depth_original_path, view.image_name+".npy")).min()
        depth = (depth - depth_min) / (depth_max - depth_min)             
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

        w2c = np.zeros((4, 4))
        w2c[:3, :3] = view.R.transpose()    # view.R: camera to world
        w2c[:3, 3] = view.T                 # view.T: world to camera
        w2c[3, 3] = 1.0       
        c2w = np.linalg.inv(w2c)    
        intrinsics = get_intrinsics(view.image_height, view.image_width,view.FoVx,view.FoVy)
        depth = np.load(os.path.join(depth_path, view.image_name +".npy"))
        points = create_point_cloud(depth, intrinsics, c2w)
        colors = cv2.imread(os.path.join(render_path, view.image_name + ".png")).reshape(-1,3)
        ply_path = os.path.join(fused_vanilla_col_dep_ply_path, view.image_name+".ply")
        ply_color_fusion(points, colors, ply_path)
        if frame_writer is not None:
            frame_writer.write(view, results)


def  virtual(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
 
        target_object_physical_radius = args.target_object_radius

        classifier = torch.nn.Conv2d(gaussians.num_objects, dataset.num_classes, kernel_size=1) 
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        views = scene.getTrainCameras()
        view = views[0]
        is_circle=True

        # here we calculate virtual camera pose circle radius
        if args.circle_radius != -1:
            pass
        else:
            args.circle_radius = generate_virtual_radius(views, 
                                    target_object_radius=target_object_physical_radius)

        from utils.pose_utils import generate_virtual_path
        poses, trajectory = generate_virtual_path(
            views, n_frames=args.camera_count, path_type=args.camera_path,
            circle_radius=args.circle_radius, max_elevation_deg=args.hemisphere_max_elevation_deg)
        
        virtual_pose_list = []
        for idx, pose in enumerate(tqdm(poses, desc="Prepare virtual camera pose")):
            view_tmp = copy.deepcopy(view)
            view_tmp.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
            view_tmp.full_proj_transform = (view_tmp.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
            view_tmp.camera_center = view_tmp.world_view_transform.inverse()[3, :3]
            view_tmp.image_name = f"{idx:05d}"
            view_tmp.R = pose[:3, :3].T
            view_tmp.T = pose[:3, 3]

            virtual_pose_list.append(view_tmp)

        if args.camera_manifest is not None:
            from utils.virtual_camera_manifest import (
                build_virtual_camera_manifest, trajectory_diagnostics, trajectory_svg)
            declared = None if args.camera_path == 'circle' and args.camera_count == 30 else trajectory
            candidate = build_virtual_camera_manifest(virtual_pose_list,
                iteration=int(scene.loaded_iter), circle_radius=args.circle_radius, trajectory=declared)
            if Path(args.camera_manifest).exists():
                existing = json.loads(Path(args.camera_manifest).read_text())
                if existing.get("artifact_id") != candidate["artifact_id"]:
                    raise ValueError("virtual camera inputs changed; choose a new removal run")
            manifest_path = write_virtual_camera_manifest(
                args.camera_manifest, virtual_pose_list,
                circle_radius=args.circle_radius, iteration=int(scene.loaded_iter),
                trajectory=declared,
            )
            diagnostics = trajectory_diagnostics(virtual_pose_list, trajectory)
            diagnostics["camera_artifact_id"] = candidate["artifact_id"]
            _atomic_write_text(manifest_path.with_name("camera_trajectory.json"),
                               json.dumps(diagnostics, indent=2) + "\n")
            _atomic_write_text(manifest_path.with_name("camera_trajectory.svg"), trajectory_svg(diagnostics))
            print(f"Virtual camera manifest: {manifest_path}")
        # save the circle radius ration in removal and inpaint config file
        config_paths = [args.config_file,
                        args.config_file.replace("object_removal", "object_inpaint")]
        
        for path in config_paths:
            with open(path, "r") as f:
                scene_info = json.load(f)
            # JSON keeps enough significant digits to reproduce the exact
            # camera path; do not quantize geometry to four decimals.
            scene_info["circle_radius"] = float(args.circle_radius)
            if args.camera_path != "circle" or args.camera_count != 30:
                scene_info["virtual_camera_path"] = args.camera_path
                scene_info["virtual_camera_count"] = args.camera_count
                if args.camera_path == "hemisphere":
                    scene_info["virtual_hemisphere_max_elevation_deg"] = args.hemisphere_max_elevation_deg
            json_str = json.dumps(scene_info, indent=4, ensure_ascii=False)
            
            import re
            json_str = re.sub(
                r'\[\s+([\d, \s.-]+)\s+\]', 
                lambda m: "[" + re.sub(r'\s+', ' ', m.group(1).strip()) + "]", 
                json_str)
            
            _atomic_write_text(Path(path), json_str)

        if getattr(args, "poses_only", False):
            if args.camera_manifest is None:
                raise ValueError("--poses-only requires --camera_manifest")
            return

        # Step 1: Generate the full scene containing all objects
        render_set_full_scene_stage(dataset.model_path, "virtual", scene.loaded_iter, virtual_pose_list, gaussians, pipeline, background, classifier)

        # # Step 2: Generate the background scene with objects(target + surrounding) removed
        step_num = str(scene.loaded_iter)
        load_iteration='_object_removal/iteration_'+step_num
        scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False) # load removal scene
        render_set_removal_stage(dataset.model_path, "virtual", load_iteration, virtual_pose_list, gaussians, pipeline, background, classifier)

        renders_dir = os.path.join(args.model_path, "virtual", f"ours{scene.loaded_iter}", "renders")
        archive_path = package_tracker_images(renders_dir, args.tracker_archive)
        print(f"\n📦 Packaged render results: {renders_dir} -> {archive_path}")

        # Step3: Generate the background scene with objects(only target object) removed
        if len(args.select_obj_id) > 1 and len(args.surrounding_ids) > 0:
            scene = Scene(dataset, gaussians, load_iteration=f'_object_removal/iteration_{step_num}_removal_target', shuffle=False) # load removal scene
            render_set_full_scene_stage(dataset.model_path, "virtual", f'object_removal/iteration_{step_num}_removal_target', virtual_pose_list, gaussians, pipeline, background, classifier)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--camera-path", choices=("circle", "hemisphere"), default="circle")
    parser.add_argument("--camera-count", type=int, default=30,
                        help="Total frames for either path (default 30, minimum 2)")
    parser.add_argument("--hemisphere-max-elevation-deg", type=float, default=85.0)
    parser.add_argument("--poses-only", action="store_true", help="Write exact cameras without rendering or packaging tracker images")
    parser.add_argument("--skip_ellipse_video", action="store_true")
    parser.add_argument("--is_circle", action="store_false")
    parser.add_argument("--circle_radius", default=-1.0, type=float, help="smaller ratio means closer camera to object")
    parser.add_argument("--skip_gaussians_disturb", action="store_true")
    parser.add_argument("--mean", default=0, type=float)
    parser.add_argument("--std", default=0.03, type=float)
    parser.add_argument("--config_file", type=str, default="config/object_removal/inpaint360/doppelherz.json", help="Path to the configuration file")
    parser.add_argument(
        "--tracker_archive",
        type=str,
        default=None,
        help="Optional output .zip for tracker images (default: Segment-and-Track-Anything/assets/images.zip).",
    )
    parser.add_argument(
        "--camera_manifest",
        type=str,
        default=None,
        help="Optional JSON path for the exact virtual cameras used for rendering.",
    )

    args = get_combined_args(parser)  

    from utils.virtual_camera_manifest import check_camera_request, load_virtual_camera_manifest
    check_camera_request(
        load_virtual_camera_manifest(args.camera_manifest)
        if args.camera_manifest and Path(args.camera_manifest).exists() else None,
        args.camera_path, args.camera_count, args.hemisphere_max_elevation_deg)
    if (args.camera_path != "circle" or args.camera_count != 30) and not args.camera_manifest:
        parser.error("nondefault virtual cameras require --camera_manifest")

    with open(args.config_file, 'r') as file:
        config = json.load(file)

    args.select_obj_id = config.get("select_obj_id")
    args.surrounding_ids = config.get("surrounding_ids")
    args.target_object_radius = config.get("target_object_radius")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    virtual(model.extract(args), args.iteration, pipeline.extract(args))

    # # python virtual_pose.py --source_path PATH/TO/DATASET --model_path PATH/TO/MODEL
