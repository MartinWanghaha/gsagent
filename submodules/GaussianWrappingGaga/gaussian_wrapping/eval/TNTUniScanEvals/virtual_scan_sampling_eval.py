# This script is a modified version of the TanksAndTemples evaluation toolbox.
# Original source: https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation
# Original license: see https://tanksandtemples.org/license/
# In practiced used code from: https://github.com/Anttwo/MILo/tree/master/milo/eval/tnt
#
# Modifications by Diego Gomez ([2026]): 
#   Done in the context of the paper: From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians
import os
import torch
import numpy as np
import open3d as o3d
import argparse
import trimesh
import copy

from tnt_eval_utils.config import scenes_tau_dict
from tnt_eval_utils.registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
    read_trajectory,
)
from tnt_eval_utils.trajectory_io import CameraPose
from tnt_eval_utils.evaluation import EvaluateHistoSimple, write_color_distances
from tnt_eval_utils.util import make_dir
from tnt_eval_utils.plot import plot_graph
from tnt_eval_utils.help_func import auto_orient_and_center_poses

# Mesh utils
from tnt_eval_utils.crop_box_utils import CropBoxFilter
from general_mesh_utils import (
    ScalableMeshRenderer, 
    MeshRasterizer, 
    Meshes,
    depth_to_view_points,
    transform_points_view_to_world,
    cameraList_from_camInfos,
    frustum_cull_mesh,
    ModelParams,
    sceneLoadTypeCallbacks
)

'''
Auxiliary functions for scan evaluation.
'''

def load_cameras(dataset):
    if os.path.exists(os.path.join(dataset.source_path, "sparse")):
        train_cameras = sceneLoadTypeCallbacks["Colmap"](dataset.source_path, "images", False)
    else:
        assert False, "Could not recognize scene type! The virtual scan sampling evaluation only makes sense for Colmap real-life scenes!"
    print("Loading Training Cameras")
    cameras = cameraList_from_camInfos(train_cameras, 1.0, dataset)
    if len(cameras) == 0:
        raise ValueError("No cameras found")
    return cameras

'''
Load auxiliary information for the scene.
'''

def load_scene(dataset_dir):
    scene = os.path.basename(os.path.normpath(dataset_dir))

    if scene not in scenes_tau_dict:
        print(dataset_dir, scene)
        raise Exception("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    return scene

def load_gt_pcd_and_transform(dataset_dir, scene, traj_path):
    '''
    Returns ground truth with normals and trajectory_transform to align mesh.
    '''
    # put the crop-file, the GT file, the COLMAP SfM log file and
    # the alignment of the according scene in a folder of
    # the same scene name in the dataset_dir
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")

    # this is for groundtruth pointcloud, we can use it
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    gt_filen = os.path.join(dataset_dir, scene + ".ply")

    print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)

    gt_trans = np.loadtxt(alignment)
    print(traj_path)
    traj_to_register = []
    if traj_path.endswith('.npy'):
        ld = np.load(traj_path)
        for i in range(len(ld)):
            traj_to_register.append(CameraPose(meta=None, mat=ld[i]))
    elif traj_path.endswith('.json'): # instant-npg or sdfstudio format
        import json
        with open(traj_path, encoding='UTF-8') as f:
            meta = json.load(f)
        poses_dict = {}
        for i, frame in enumerate(meta['frames']):
            filepath = frame['file_path']
            new_i = int(filepath[13:18]) - 1
            poses_dict[new_i] = np.array(frame['transform_matrix'])
        poses = []
        for i in range(len(poses_dict)):
            poses.append(poses_dict[i])
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, _ = auto_orient_and_center_poses(poses, method='up', center_poses=True)
        scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
        poses[:, :3, 3] *= scale_factor
        poses = poses.numpy()
        for i in range(len(poses)):
            traj_to_register.append(CameraPose(meta=None, mat=poses[i]))
    else:
        traj_to_register = read_trajectory(traj_path)
    print(colmap_ref_logfile)
    gt_traj_col = read_trajectory(colmap_ref_logfile)

    trajectory_transform = trajectory_alignment(None, traj_to_register, gt_traj_col, gt_trans)

    # estimate gt normals
    gt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))

    return gt_pcd, trajectory_transform

# --- Get Predicted Point Cloud From Point Cloud ----

def compute_trajectory_transform_from_pc(pc_o3d, trajectory_transform, gt_pcd, crop_volume, dTau, n_icp_iterations=100, debug_path=None):
    '''
    Given a mesh and ground truth information, we compute the transform needed to align the predicted 
    mesh and the GT point cloud.
    '''
    # Registration refinment in 3 iterations
    if debug_path is not None:
        os.makedirs(debug_path, exist_ok=True)
    print(f"[INFO] First ICP iteration --------------------------------")
    r2 = registration_vol_ds(pc_o3d, gt_pcd, trajectory_transform, crop_volume, dTau,
                             dTau * 80, n_icp_iterations, debug_path=debug_path + ".first_icp.ply" if debug_path is not None else None, already_cropped=False)
    print(f"[INFO] Second ICP iteration --------------------------------")
    r3 = registration_vol_ds(pc_o3d, gt_pcd, r2.transformation, crop_volume, dTau / 2.0,
                             dTau * 20, n_icp_iterations, debug_path=debug_path + ".second_icp.ply" if debug_path is not None else None, already_cropped=False)
    print(f"[INFO] Third ICP iteration --------------------------------")
    r = registration_unif(pc_o3d, gt_pcd, r3.transformation, crop_volume, 
                             2 * dTau, n_icp_iterations, debug_path=debug_path + ".third_icp.ply" if debug_path is not None else None, already_cropped=False)

    trajectory_transform = r.transformation

    if debug_path is not None:
        final_pcd = copy.deepcopy(pc_o3d).transform(trajectory_transform)
        o3d.io.write_point_cloud(debug_path + ".final_icp.ply", final_pcd)

    return trajectory_transform

def get_predicted_pcd_from_mesh_scan(ply_path, gt_pcd, trajectory_transform, vol, dTau, num_surface_samples, debug_path=None, dataset=None, crop_filter=None):
    '''
    Load the predicted point cloud from a file.
    '''
    # Get mesh
    print(f"[DEBUG] Loading predicted mesh: {ply_path}")
    unaligned_mesh = trimesh.load_mesh(ply_path)

    print(f"[DEBUG] Creating Open3D triangle mesh")

    # Load cameras
    cameras = load_cameras(dataset)

    # Iterate over cameras and project the depth to world space: Also filter points outside the crop volume and with depth filter from PGSR
    renderer = ScalableMeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras)
    )
    mesh = Meshes(verts=torch.from_numpy(np.array(unaligned_mesh.vertices)).float().to('cuda'), faces=torch.from_numpy(np.array(unaligned_mesh.faces)).to('cuda'))
    all_world_points = []
    all_normals = []
    for i,cam in enumerate(cameras):
        rendered_image = renderer(
            mesh=frustum_cull_mesh(mesh, cam),  # FIXME: Add znear
            cam_idx=i,
            return_depth=True,
            return_normals=True,
            use_antialiasing=False,  
        )

        depth = rendered_image['depth'].squeeze() # (H, W)
        points = depth_to_view_points(
            cam,
            depth,
        )  # (3, H, W)
        
        points = points.permute(1, 2, 0)  # (H, W, 3)
        points = points.view(-1, 3)  # (H * W, 3)
        world_points = transform_points_view_to_world(points=points.unsqueeze(0), cameras=[cam]).squeeze(0).cpu().contiguous().numpy()  # (H * W, 3)
        normals = rendered_image['normals'].squeeze().view(-1, 3).cpu().contiguous().numpy() # (H * W, 3)
        mask = (depth > 0.).view(-1).cpu().contiguous().numpy()
        #Apply filtering
        world_points = world_points[mask]
        normals = normals[mask]
        if crop_filter is not None:
            world_points, normals = crop_filter.filter_through_pcd(world_points, normals)
        # Add to all world points
        all_world_points.append(world_points)
        all_normals.append(normals)

    # Apply voxel downsampling with finer resolution than the one later
    ## Create point cloud from all world points and normals
    all_world_points = np.concatenate(all_world_points, axis=0)
    all_normals = np.concatenate(all_normals, axis=0)
    unaligned_pcd = o3d.geometry.PointCloud()
    unaligned_pcd.points = o3d.utility.Vector3dVector(all_world_points)
    unaligned_pcd.normals = o3d.utility.Vector3dVector(all_normals)

    ## Apply voxel downsampling
    unaligned_pcd = unaligned_pcd.voxel_down_sample(voxel_size=dTau / 4.0)

    # Compute trajectory transform
    trajectory_transform = compute_trajectory_transform_from_pc(unaligned_pcd, trajectory_transform, gt_pcd, vol, dTau, debug_path=debug_path)

    # Crop and transform point cloud
    s = copy.deepcopy(unaligned_pcd)
    aligned_pcd = s.transform(trajectory_transform)
    aligned_cropped_pcd = vol.crop_point_cloud(aligned_pcd)

    print(f"aligned_cropped_pcd points shape: {np.asarray(aligned_cropped_pcd.points).shape}")
    print(f"aligned_cropped_pcd normals shape: {np.asarray(aligned_cropped_pcd.normals).shape}")

    return aligned_cropped_pcd

'''
Run the evaluation.
'''
def run_evaluation(dataset_dir, traj_path, ply_path, out_dir, num_surface_samples=int(1e6), save_point_clouds=False, dataset=None):
    make_dir(out_dir) # create output directory

    # 1. Load scene, information and GT
    print(f"[INFO] Loading scene, information and GT")
    scene = load_scene(dataset_dir)
    dTau = scenes_tau_dict[scene]
    gt_pcd, trajectory_transform = load_gt_pcd_and_transform(dataset_dir, scene, traj_path)
    ## this crop file is also w.r.t the groundtruth pointcloud, we can use it. Otherwise we have to crop the estimated pointcloud by ourself
    vol = o3d.visualization.read_selection_polygon_volume(os.path.join(dataset_dir, scene + ".json"))
    
    # 2. Load and align the reconstruction
    print(f"[INFO] Loading and aligning the reconstruction")
    crop_filter = CropBoxFilter(os.path.join(dataset_dir, scene + ".json"), transform=trajectory_transform)
    predicted_pcd = get_predicted_pcd_from_mesh_scan(ply_path, gt_pcd, trajectory_transform, vol, dTau, num_surface_samples, debug_path=None, dataset=dataset, crop_filter=crop_filter)

    # 3. Downsample the point clouds into coarse and fine resolutions
    ## We use the fine resolution to compute the fast compute_point_cloud_distance and get precision and recall values.
    voxel_size_fine = dTau / 2.0 
    
    print(f"[INFO] Downsampling the point clouds")

    s_fine = copy.deepcopy(predicted_pcd)
    t_fine = copy.deepcopy(gt_pcd)
    s_fine = s_fine.voxel_down_sample(voxel_size=voxel_size_fine)
    t_fine = t_fine.voxel_down_sample(voxel_size=voxel_size_fine)

    # 4. Evaluate
    ## Compute distances
    print(f"[INFO] Computing distances")
    distance1 = s_fine.compute_point_cloud_distance(t_fine)
    distance2 = t_fine.compute_point_cloud_distance(s_fine)   

    if save_point_clouds:
        source_n_fn = out_dir + "/" + scene + ".precision.ply"
        target_n_fn = out_dir + "/" + scene + ".recall.ply"

        print(f"[Precision file: {source_n_fn}]")
        print(f"[Recall file: {target_n_fn}]")

        print("[ViewDistances] Add color coding to visualize error")
        write_color_distances(source_n_fn, s_fine, distance1, 3 * dTau)

        print("[ViewDistances] Add color coding to visualize error")
        write_color_distances(target_n_fn, t_fine, distance2, 3 * dTau)

    # Histogramms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = EvaluateHistoSimple(
        distance1=distance1,
        distance2=distance2,
        threshold=dTau,
        filename_mvs=out_dir,
        plot_stretch=plot_stretch,
        scene_name=scene,
        verbose=False
    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")

    # Plotting
    plot_graph(
        scene,
        fscore,
        dTau,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model = ModelParams(parser)

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="path to a dataset/scene directory containing X.json, X.ply, ...",
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        required=True,
        help=
        "path to trajectory file. See `convert_to_logfile.py` to create this file.",
    )
    parser.add_argument(
        "--ply-path",
        type=str,
        required=True,
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=
        "output directory, default: an evaluation directory is created in the directory of the ply file",
    )
    parser.add_argument(
        "--num-surface-samples",
        type=int,
        default=int(1e7),
        help="number of surface samples to use for evaluation",
    )
    parser.add_argument(
        "--save_point_clouds", 
        action="store_true",
        help="whether to save the precision and recall point clouds",
    )
    args = parser.parse_args()

    if args.out_dir.strip() == "":
        args.out_dir = os.path.join(os.path.dirname(args.ply_path),
                                    "new_evaluation_scan")

    run_evaluation(
        dataset_dir=args.dataset_dir,
        traj_path=args.traj_path,
        ply_path=args.ply_path,
        out_dir=args.out_dir,
        num_surface_samples=args.num_surface_samples,
        save_point_clouds=args.save_point_clouds,
        dataset=model.extract(args),
    )
