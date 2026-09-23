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

import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from sklearn.cluster import KMeans, DBSCAN


def parse_object_ids(value):
    """Parse legacy ``[1,2]`` and canonical ``1,2`` object lists."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_values = value
    else:
        text = str(value).strip().strip("[]")
        raw_values = text.replace(",", " ").split()
    result = []
    for raw in raw_values:
        object_id = int(raw)
        if object_id <= 0:
            raise ValueError(f"object IDs must be positive, got {object_id}")
        if object_id not in result:
            result.append(object_id)
    return result

def filter_artifacts_by_kmeans(gaussians_object, n_clusters=2):

    xyz_tensor = gaussians_object._xyz
    xyz = xyz_tensor.detach().cpu().numpy()

    if len(xyz) < n_clusters:
        return gaussians_object
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(xyz)
    labels = kmeans.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    main_cluster = unique_labels[np.argmax(counts)]

    keep_mask = (labels == main_cluster)
    keep_indices = torch.nonzero(
        torch.from_numpy(keep_mask).to(xyz_tensor.device)
    ).squeeze(1)

    gaussians_object._xyz = gaussians_object._xyz[keep_indices]
    gaussians_object._features_dc = gaussians_object._features_dc[keep_indices]
    gaussians_object._features_rest = gaussians_object._features_rest[keep_indices]
    gaussians_object._opacity = gaussians_object._opacity[keep_indices]
    gaussians_object._scaling = gaussians_object._scaling[keep_indices]
    gaussians_object._rotation = gaussians_object._rotation[keep_indices]
    gaussians_object._objects_dc = gaussians_object._objects_dc[keep_indices]

    return gaussians_object


def filter_artifacts_by_dbscan(gaussians_object, eps=0.1, min_samples=10):
    xyz_tensor = gaussians_object._xyz
    xyz = xyz_tensor.detach().cpu().numpy()

    if len(xyz) < min_samples:
        return gaussians_object
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = clustering.labels_

    # label = -1 is noise
    if np.all(labels == -1):
        print("⚠️ Warning: all points are labeled as noise!")
        return gaussians_object  #

    valid_mask = labels != -1
    cluster_labels = labels[valid_mask]
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    main_cluster = unique_labels[np.argmax(counts)]

    keep_mask = (labels == main_cluster)
    keep_indices = torch.nonzero(
        torch.from_numpy(keep_mask).to(xyz_tensor.device)
    ).squeeze(1)

    gaussians_object._xyz = gaussians_object._xyz[keep_indices]
    gaussians_object._features_dc = gaussians_object._features_dc[keep_indices]
    gaussians_object._features_rest = gaussians_object._features_rest[keep_indices]
    gaussians_object._opacity = gaussians_object._opacity[keep_indices]
    gaussians_object._scaling = gaussians_object._scaling[keep_indices]
    gaussians_object._rotation = gaussians_object._rotation[keep_indices]
    gaussians_object._objects_dc = gaussians_object._objects_dc[keep_indices]

    return gaussians_object


def _infer_removed_iteration(model_path, object_ids):
    root = Path(model_path) / "point_cloud_object_removal"
    candidates = []
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_dir() or not path.name.startswith("iteration_"):
                continue
            suffix = path.name.removeprefix("iteration_")
            if not suffix.isdecimal():
                continue
            if all((path / f"point_cloud_{object_id}.ply").is_file() for object_id in object_ids):
                candidates.append(int(suffix))
    if len(candidates) != 1:
        raise ValueError(
            "cannot infer one removal source iteration; pass source_iteration "
            f"explicitly (candidates: {sorted(candidates)})"
        )
    return candidates[0]


def combine_gaussian(
    dataset: ModelParams,
    iteration,
    pipeline: PipelineParams,
    object_list,
    *,
    source_iteration=None,
    base_ply=None,
    removed_objects_root=None,
    output_ply=None,
    filter_objects=True,
):
    """ 
    Combine the base scene with multiple removed objects into a single Gaussian model and save the result.

    Args:
        dataset (ModelParams): Model parameters containing SH degree and paths.
        iteration (int): The iteration number to load the base scene.
        pipeline (PipelineParams): Pipeline configuration for rendering.
        object_list (list): A list of object IDs to be re-integrated into the scene.

    Returns:
        GaussianModel: The final combined Gaussian model containing both the base scene and objects.
    """
    del pipeline  # Kept for backwards-compatible calls.
    object_ids = parse_object_ids(object_list)
    if not object_ids:
        raise ValueError("at least one surrounding object ID is required")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        if base_ply is not None:
            base_path = Path(base_ply).expanduser().resolve(strict=True)
            gaussians.load_ply(str(base_path))
            model_path = str(Path(dataset.model_path).expanduser().resolve())
        else:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            model_path = scene.model_path

    if source_iteration is None:
        source_iteration = _infer_removed_iteration(model_path, object_ids)
    if isinstance(source_iteration, bool) or int(source_iteration) <= 0:
        raise ValueError("source_iteration must be a positive integer")
    source_iteration = int(source_iteration)
    object_root = (
        Path(removed_objects_root).expanduser().resolve()
        if removed_objects_root is not None
        else Path(model_path)
        / "point_cloud_object_removal"
        / f"iteration_{source_iteration}"
    )

    for object_id in object_ids:
        gaussians_object = GaussianModel(dataset.sh_degree)
        object_path = object_root / f"point_cloud_{object_id}.ply"
        if not object_path.is_file():
            raise FileNotFoundError(f"removed object checkpoint is missing: {object_path}")
        gaussians_object.load_ply(str(object_path))
        if filter_objects:
            gaussians_object = filter_artifacts_by_dbscan(gaussians_object)

        gaussians._xyz = torch.cat([gaussians._xyz, gaussians_object._xyz], dim=0)
        gaussians._features_dc = torch.cat([gaussians._features_dc, gaussians_object._features_dc], dim=0)
        gaussians._features_rest = torch.cat([gaussians._features_rest, gaussians_object._features_rest], dim=0)
        gaussians._opacity = torch.cat([gaussians._opacity, gaussians_object._opacity], dim=0)
        gaussians._scaling = torch.cat([gaussians._scaling, gaussians_object._scaling], dim=0)
        gaussians._rotation = torch.cat([gaussians._rotation, gaussians_object._rotation], dim=0)
        gaussians._objects_dc = torch.cat([gaussians._objects_dc, gaussians_object._objects_dc], dim=0)
    
    if output_ply is None:
        if base_ply is not None:
            combined_gaussian_path = Path(base_ply).expanduser().resolve()
        else:
            combined_gaussian_path = Path(
                os.path.dirname(model_path + "/point_cloud" + str(iteration))
            ) / "point_cloud.ply"
    else:
        combined_gaussian_path = Path(output_ply).expanduser().resolve()
    combined_gaussian_path.parent.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(combined_gaussian_path))
    print(f"The combined gaussian scene is saved at {combined_gaussian_path}")
    return gaussians


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default='_object_inpaint_virtual/iteration_4999/point_cloud.ply')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--object_list", help="e.g. [11,22]")
    parser.add_argument("--source_iteration", type=int, default=None)
    parser.add_argument("--base_ply", type=str, default=None)
    parser.add_argument("--removed_objects_root", type=str, default=None)
    parser.add_argument("--output_ply", type=str, default=None)
    parser.add_argument("--skip_object_filter", action="store_true")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    # Initialize system state (RNG)s
    safe_state(args.quiet)

    combine_gaussian(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.object_list,
        source_iteration=args.source_iteration,
        base_ply=args.base_ply,
        removed_objects_root=args.removed_objects_root,
        output_ply=args.output_ply,
        filter_objects=not args.skip_object_filter,
    )

    # python tools/combine_gaussian_scene.py -s data/inpaint360/fruits -m output/inpaint360/fruits
