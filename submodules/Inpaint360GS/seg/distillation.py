# This file is part of inpaint360gs: Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
# Project page: https://dfki-av.github.io/inpaint360gs/
#
# Copyright 2024-2026 Shaoxiang Wang <shaoxiang.wang@dfki.de>
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0

# Modified from codes in Gaussian-Grouping https://github.com/lkeab/gaussian-grouping 
# and Gaga https://github.com/weijielyu/Gaga/tree/main?tab=readme-ov-file 

# This file contains original research code and modified components from the 
# aforementioned projects. It is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, loss_cls_3d_kl, loss_cls_3d_cosin
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json

def training(dataset, opt, pipe, args, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1)
    # scene = Scene(dataset, gaussians)
    gaussians.training_setup_distill(opt)                                     

    print(f"\n There are {gaussians.get_xyz.shape[0]} points for distillation!!!")

    matched_mask_path = os.path.join(dataset.source_path, dataset.object_path) 
    info = json.load(open(os.path.join(matched_mask_path, "scene.json")))
    print("Info of the mask association process: ", info)
    num_classes = info["num_classes"]

    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    classifier.cuda()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

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
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]

        # Object Loss
        gt_obj = viewpoint_cam.objects.cuda().long()
        logits = classifier(objects)    
        loss_obj_2d = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        loss_obj_2d = loss_obj_2d / torch.log(torch.tensor(num_classes))  
        loss_obj_3d_sim = 0

        if iteration % opt.reg3d_interval == 0: 
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
            feature_vec = gaussians.get_objects.squeeze()

            sim_weight = 0.0005
            _, loss_obj_3d_sim = loss_cls_3d_cosin(gaussians._xyz.squeeze().detach(), feature_vec, prob_obj3d, 
                                                    k=opt.reg3d_k, lambda_val=opt.reg3d_lambda_val, sim_weight=sim_weight, 
                                                    max_points=opt.reg3d_max_points, sample_size=opt.reg3d_sample_size)                                                              
            loss = loss_obj_2d + loss_obj_3d_sim
        else:
            loss = loss_obj_2d

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
          
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if use_wandb:
                training_report(iteration, loss_obj_2d, loss, iter_start.elapsed_time(iter_end), loss_obj_3d_sim)      
          
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians under {}".format(iteration, 
                                                                     os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))))
                scene.save(iteration)
                torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))   

            # Optimizer step
            try:
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    cls_optimizer.step()         
                    cls_optimizer.zero_grad()
            except:
                print("Error in optimizer step")
                pass

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint at {}".format(iteration, os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, loss_obj_2d, loss, elapsed, loss_obj_3d_sim=None):

    wandb.log({"train_loss/total_loss": loss.item(), "train_loss/loss_obj_2d": loss_obj_2d.item(), "train_loss/loss_obj_3d_sim": loss_obj_3d_sim.item(), "iter_time": elapsed, "iter": iteration})

    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config/object_distill/train_distill.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")
    parser.add_argument("--my_debug_tag", action='store_true', default=False, help="Debug tag for my own purpose")

    args = parser.parse_args(sys.argv[1:])

    # Read and parse the configuration file
    with open(args.config_file, 'r') as file:
        config = json.load(file)

    args.reg3d_interval = config.get("reg3d_interval", 50)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)
    args.iterations = int(config.get("iterations", 2000))
    if args.iterations <= 0:
        raise ValueError(f"iterations must be positive, got {args.iterations}")
    args.train_distill = config.get("train_distill", False) 

    # The final iteration is known only after loading the distillation config.
    # Ensure it is saved even when a custom iteration count is configured.
    for iteration_list_name in ("save_iterations", "test_iterations", "checkpoint_iterations"):
        iteration_list = getattr(args, iteration_list_name)
        if args.iterations not in iteration_list:
            iteration_list.append(args.iterations)

    scene_json_path = os.path.join(args.source_path, args.object_path, "scene.json")
    if not os.path.isfile(scene_json_path):
        raise FileNotFoundError(
            f"Mask-association metadata not found: {scene_json_path}. "
            "Check --object_path."
        )
    with open(scene_json_path, "r") as f:
        scene_info = json.load(f)
    args.num_classes = scene_info.get("num_classes")
    if (
        not isinstance(args.num_classes, int)
        or not 1 < args.num_classes <= 65535
    ):
        raise ValueError(f"Invalid num_classes in {scene_json_path}: {args.num_classes!r}")

    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="inpaint360gs")
        wandb.config.args = args
        run_name = "_".join(args.model_path.split("/")[1:])
        wandb.run.name = run_name
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb)

    # All done
    print("\nTraining complete.")
