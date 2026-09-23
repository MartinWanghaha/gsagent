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
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.cluster import KMeans
import json
import numpy as np
import torch.nn.functional as F
from arguments import ModelParams, PipelineParams, AssociateParams, get_combined_args
from utils.point_utils import project_3d_points,ndc_to_pixel
from utils.image_utils import convert_instance_mask_to_binary_masks
from tools.vis_obj_color import vis_mask_images

from scene import Scene,GaussianModel


class GaussianMaskAssociator(torch.nn.Module):
    def __init__(self,
                 dataset : ModelParams,
                 associate: AssociateParams,
                 iteration : int,
                 device : torch.device = torch.device("cuda"),
                 ):
        super(GaussianMaskAssociator, self).__init__()
        self.device = device
        # Load pre-trained Gaussians and cameras
        self.gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, self.gaussians, load_iteration=iteration, shuffle=False)
        self.gaussians_xyz = self.gaussians.get_xyz.to(self.device)
        self.viewpoint_stack_train = scene.getTrainCameras()
        self.viewpoint_stack_test = scene.getTestCameras()
        self.viewpoint_stack_unsorted = self.viewpoint_stack_train + self.viewpoint_stack_test
        self.viewpoint_stack = sorted(self.viewpoint_stack_unsorted.copy(), key = lambda x : x.image_name)
        if not self.viewpoint_stack:
            raise ValueError("No cameras are available for mask association.")
        self.patches = associate.patch
        if self.patches <= 0:
            raise ValueError(f"patch must be positive, got {self.patches}")
       
        self.source_path = dataset.source_path
        self.mask_generator = associate.mask_generator
        self.raw_mask_folder = os.path.join(self.source_path, "raw_{0}".format(self.mask_generator))
        if not os.path.exists(self.raw_mask_folder):
            raise FileNotFoundError(f"Raw mask folder does not exist: {self.raw_mask_folder}")
        self.associated_mask_folder = os.path.join(self.source_path, "associated_{0}".format(self.mask_generator))
        self.associated_mask_color_folder = os.path.join(self.source_path, "associated_{0}".format(self.mask_generator)) + "_color"
        os.makedirs(self.associated_mask_folder, exist_ok=True)
        os.makedirs(self.associated_mask_color_folder, exist_ok=True)

        camera_sizes = {
            (camera.image_width, camera.image_height)
            for camera in self.viewpoint_stack
        }
        if len(camera_sizes) != 1:
            raise ValueError(
                "Mask association requires one common camera resolution; "
                f"found {sorted(camera_sizes)}."
            )
        self.W, self.H = next(iter(camera_sizes))
        self.patch_W = self.W // self.patches + 1 if self.W % self.patches != 0 else self.W // self.patches
        self.patch_H = self.H // self.patches + 1 if self.H % self.patches != 0 else self.H // self.patches

        # Masks use the conventional [height, width] layout throughout. Keeping
        # this layout consistent is important because projected points are
        # flattened with the row-major index ``y * width + x`` below.
        self.patch_mask = torch.zeros((self.patches, self.patches, self.H, self.W), dtype=torch.bool, device=self.device)
        for patch_y in range(self.patches):
            for patch_x in range(self.patches):
                self.patch_mask[
                    patch_y,
                    patch_x,
                    patch_y * self.patch_H:(patch_y + 1) * self.patch_H,
                    patch_x * self.patch_W:(patch_x + 1) * self.patch_W,
                ] = True
        self.flatten_patch_mask = self.patch_mask.flatten(start_dim=2)   

        # For mask association
        self.keyobject_database = []
        self.assigned_gaussians = []
        self.num_classes = 0
    
    def update_keyobject_database(self, obj_index: int, foreground_gaussians: torch.Tensor):
        """
        Maintains the key-object database by mapping Gaussian points to unique masks.
        
        This method ensures each Gaussian point is assigned to at most one object.
        If the obj_index matches the current number of classes, a new object entry is created. 
        Otherwise, new points are merged into the existing entry after filtering out points already assigned to other objects.

        Args:
            obj_index (int): Index of the target object (0-indexed).
            foreground_gaussians (torch.Tensor): 1D Tensor of indices representing 
                                                 the detected foreground Gaussian points.
        """
        # Ensure index logic is sound: cannot skip indices when adding new objects
        assert obj_index <= self.num_classes, \
            f"Error: obj_index ({obj_index}) exceeds num_classes ({self.num_classes})"

        if obj_index == self.num_classes:
            # Case 1: Initialize a new object entry in the database
            self.keyobject_database.append(foreground_gaussians)
        else:
            # Case 2: Update existing object. 
            # Filter: Only add points that have not been previously assigned to ANY object.
            is_already_assigned = torch.isin(foreground_gaussians, self.assigned_gaussians)
            new_unassigned_points = foreground_gaussians[~is_already_assigned]
            
            # Merge and ensure the index list remains unique
            updated_indices = torch.cat([self.keyobject_database[obj_index], new_unassigned_points])
            self.keyobject_database[obj_index] = torch.unique(updated_indices)

        # Global Bookkeeping: Update the master list of all assigned Gaussian points
        # to prevent these indices from being claimed by subsequent objects.
        self.assigned_gaussians = torch.unique(torch.cat([self.assigned_gaussians, foreground_gaussians]))


    def associate(self, cam_view):
        foreground_gaussian = self.get_visible_gaussians(cam_view)
        num_classes_cur_view = len(foreground_gaussian)

        self.num_classes = len(self.keyobject_database) if self.keyobject_database else 0
        labels = torch.zeros(num_classes_cur_view, dtype=torch.long, device=self.device)
        for m_idx in range(num_classes_cur_view):
            foreground_gaussian_of_mask = foreground_gaussian[m_idx]         
            num_union = [len(torch.unique(torch.cat([self.keyobject_database[i], foreground_gaussian_of_mask]))) for i in range(self.num_classes)]   
            num_intersection = [len(self.keyobject_database[i]) + len(foreground_gaussian_of_mask) - num_union[i] for i in range(self.num_classes)]  
            num_cur = len(foreground_gaussian_of_mask)                                                                                               

            associate_mode = "IOU_heighlight"
            if associate_mode == "IOU":
                gs_iou = [num_intersection[i] / (num_union[i] + 1e-8) for i in range(self.num_classes)]
                gs_iou = torch.tensor(gs_iou, dtype=torch.float32, device=self.device)
            elif associate_mode == "IOU_heighlight":
                gs_iou = [num_intersection[i] / (num_cur + num_intersection[i] + 1e-8) for i in range(self.num_classes)] 
                gs_iou = torch.tensor(gs_iou, dtype=torch.float32, device=self.device)

            selected_mask = torch.argmax(gs_iou)             
            if gs_iou[selected_mask] < 0.1:     
                selected_mask = self.num_classes              
            self.update_keyobject_database(selected_mask, foreground_gaussian_of_mask)
            labels[m_idx] = selected_mask

            self.num_classes = len(self.keyobject_database) if self.keyobject_database else 0

        return labels
    
    def remap_mask(self, object_labels, segmentation_mask):
        """
        Remaps SAM mask IDs (1..N) to match object_labels (0..N-1).
        Background (0) remains 0.
        """
        if segmentation_mask is None:
            raise ValueError("Cannot remap an empty segmentation mask.")
        if segmentation_mask.ndim != 2 or not np.issubdtype(segmentation_mask.dtype, np.integer):
            raise ValueError(
                f"Expected a 2D integer segmentation mask, got "
                f"shape={segmentation_mask.shape}, dtype={segmentation_mask.dtype}."
            )

        max_mask_id = int(segmentation_mask.max()) if segmentation_mask.size else 0
        if object_labels.shape[0] != max_mask_id:
            raise ValueError(
                "Object labels do not match the segmentation mask IDs: "
                f"got {object_labels.shape[0]} labels for max mask ID {max_mask_id}."
            )
        if (
            object_labels.numel()
            and int(object_labels.max().item()) + 1 >= np.iinfo(np.uint16).max
        ):
            raise ValueError(
                "Associated instance IDs reached 65535, which is reserved for "
                "unknown mesh semantics."
            )

        remapped_mask = np.zeros_like(segmentation_mask, dtype=np.uint16)

        for obj_id, label in enumerate(object_labels):
            remapped_mask[segmentation_mask == obj_id + 1] = label.item() + 1

        return remapped_mask


    def save_scene_info(self):
        
        scene_info = {
            "num_classes": self.num_classes,
            "raw_mask_folder": self.raw_mask_folder,
            "associated_mask_folder": self.associated_mask_folder,
            "patches": self.patches
        }

        scene_info_path = os.path.join(self.associated_mask_folder, "scene.json")
        
        with open(scene_info_path, "w") as f:
            json.dump(scene_info, f, indent=4)
        
        print(f"Scene info saved at: {scene_info_path}")


    def run_mask_association(self):
        for cam_view in tqdm(self.viewpoint_stack):

            if self.num_classes == 0:
                foreground_gaussian = self.get_visible_gaussians(cam_view)
                self.keyobject_database.extend(foreground_gaussian)
                if foreground_gaussian:
                    self.assigned_gaussians = torch.unique(torch.cat(foreground_gaussian))
                else:
                    self.assigned_gaussians = torch.empty(0, dtype=torch.long, device=self.device)
                self.num_classes = len(self.keyobject_database) if self.keyobject_database else 0  
                labels = torch.arange(self.num_classes, dtype=torch.long, device=self.device)
            else:
                labels = self.associate(cam_view.to(self.device))
            
            mask_path = os.path.join(self.raw_mask_folder, cam_view.image_name + ".png")  
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if raw_mask is None:
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            expected_shape = (cam_view.image_height, cam_view.image_width)
            if raw_mask.shape != expected_shape:
                raise ValueError(
                    f"Mask {mask_path} has shape {raw_mask.shape}; expected {expected_shape}. "
                    "Generate masks at the same resolution used by the cameras."
                )
            remapped_mask = self.remap_mask(labels, raw_mask)                         
            remapped_mask_path = os.path.join(self.associated_mask_folder, cam_view.image_name + ".png") 
            if not cv2.imwrite(remapped_mask_path, remapped_mask):
                raise OSError(f"Failed to write associated mask: {remapped_mask_path}")
        
        vis_mask_images(self.associated_mask_folder, self.associated_mask_color_folder)
            
        self.num_classes = len(self.keyobject_database) if self.keyobject_database else 0
        self.num_classes += 1  # Include background class 0.
        if self.num_classes > np.iinfo(np.uint16).max:
            raise ValueError(
                f"Scene contains {self.num_classes} classes; at most 65535 are "
                "supported because label 65535 is reserved for unknown mesh semantics."
            )
        print("Instances in this scene: ", self.num_classes)

        self.save_scene_info()

    def get_projected_gaussians(self, viewpoint): 
        """
        Projects 3D Gaussian centers (self.gaussians_xyz) onto the 2D image plane.
        
        1. Transforms 3D points to 2D image coordinates.
        2. Culls points that fall outside the image boundaries (frustum culling).
        3. Returns the flattened 2D pixel indices and associated projection metadata.
        """
        proj_matrix = viewpoint.full_proj_transform

        # world  → cam
        p_hom = project_3d_points(self.gaussians_xyz, proj_matrix)  
        p_hom_z = p_hom[:, 2]  # depth in z

        p_w = 1 / (p_hom[:, 3:] + 1e-8) 
        p_proj = p_hom[:, :3] * p_w  

        # NDC coordinate → pixel coordinate
        p_proj[:, 0] = ndc_to_pixel(p_proj[:, 0], self.W)
        p_proj[:, 1] = ndc_to_pixel(p_proj[:, 1], self.H)
        p_proj = torch.round(p_proj[:, :2]).long() 

        p_inside_mask = (p_proj[:, 0] >= 0) & (p_proj[:, 0] < self.W) & (p_proj[:, 1] >= 0) & (p_proj[:, 1] < self.H) & (p_hom_z > 0)

        p_proj_inside = p_proj[p_inside_mask]  
        p_proj_inside_indices = p_inside_mask.nonzero(as_tuple=True)[0]  
        p_proj_inside_reverse_mapping = dict(zip(p_proj_inside_indices.cpu().tolist(), range(len(p_proj_inside_indices))))

        p_proj_flatten = p_proj_inside[:, 1] * self.W + p_proj_inside[:, 0]

        # Return all projection data
        projected_gaussian = {
            "p_proj_flatten": p_proj_flatten,                  # Flattened pixel coordinates covered by Gaussians
            "p_proj_inside_indices": p_proj_inside_indices,    # Indices of Gaussians that fall within the screen space
            "p_proj_inside_reverse_mapping": p_proj_inside_reverse_mapping, # Reverse index mapping for in-screen points
            "p_hom_z": p_hom_z                                 # Depth information (z-buffer) for all Gaussians
        }

        return projected_gaussian

    
    def get_binary_mask(self, cam_view):

        mask_path = os.path.join(self.raw_mask_folder, cam_view.image_name + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        if len(mask.shape) != 2:
            raise ValueError(f"Invalid mask shape: {mask.shape} in {mask_path}")
        expected_shape = (cam_view.image_height, cam_view.image_width)
        if mask.shape != expected_shape:
            raise ValueError(f"Invalid mask shape: {mask.shape} in {mask_path}; expected {expected_shape}")

        binary_mask_np = convert_instance_mask_to_binary_masks(mask)
        binary_mask = torch.tensor(binary_mask_np, dtype=torch.bool, device=self.device)

        return binary_mask


    def get_visible_gaussians(self, cam_view): 
        """
        Projects 3D Gaussians onto the 2D image plane and identifies the 
        foreground Gaussians for each object based on the segmentation mask.

        Args:
            cam_view: Camera parameters for the current viewpoint used for 3D-to-2D projection.

        Returns:
            list[torch.Tensor]: A list where each tensor contains the indices of 
                foreground Gaussians for a specific object in the current frame.
        """

        # Project 3D Gaussians to 2D image space
        projected_data = self.get_projected_gaussians(cam_view)
        # Flattened pixel indices occupied by Gaussians (1D representation)
        p_proj_flatten = projected_data["p_proj_flatten"]                
        # Indices of Gaussians that fall within the screen space boundaries
        p_proj_inside_indices = projected_data["p_proj_inside_indices"]  
        # Depth/Homogeneous Z-coordinates for depth testing
        p_hom_z = projected_data["p_hom_z"]                                                 

        instance_mask = self.get_binary_mask(cam_view)  

        obj_mask_flattened = instance_mask.flatten(start_dim=1)  
        visible_gaussians = []
        
        for obj_mask in obj_mask_flattened:
            obj_gaussians = []

            pad_H = (self.patch_H - (self.H % self.patch_H)) % self.patch_H  
            pad_W = (self.patch_W - (self.W % self.patch_W)) % self.patch_W  
            obj_mask_2d = obj_mask.reshape(self.H, self.W)
            obj_mask_padded = F.pad(obj_mask_2d, (0, pad_W, 0, pad_H))
            obj_mask_patched = obj_mask_padded.unfold(0, self.patch_H, self.patch_H).unfold(1, self.patch_W, self.patch_W)

            patch_coverage = obj_mask_patched.sum(dim=(2, 3)) > 0
            active_patches = torch.nonzero(patch_coverage, as_tuple=True) 
            patch_y, patch_x = active_patches[0], active_patches[1]

            for patch_y_idx, patch_x_idx in zip(patch_y, patch_x):
                patch_mask = self.patch_mask[patch_y_idx, patch_x_idx]
                obj_patch_mask = obj_mask & patch_mask.flatten()  

                if obj_patch_mask.sum() == 0:
                    continue

                # Identify Gaussians within the current patch
                gaussian_indices_in_mask = obj_patch_mask[p_proj_flatten].nonzero().squeeze(-1)

                if gaussian_indices_in_mask.shape[0] == 0:
                    continue

                gaussian_indices = p_proj_inside_indices[gaussian_indices_in_mask]  
                depth_values = p_hom_z[gaussian_indices]  

                ### KMeans ###
                if len(depth_values) >= 2:
                    depths_np = depth_values.cpu().numpy().reshape(-1, 1)
                    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(depths_np)
                    cluster_labels = kmeans.labels_

                    foreground_cluster = kmeans.cluster_centers_.argmin()
                    foreground_indices = (cluster_labels == foreground_cluster)

                    foreground_depths = depths_np[foreground_indices].flatten()
                    foreground_gaussians = gaussian_indices[foreground_indices]
                
                    sorted_indices = np.argsort(foreground_depths)
                    num_to_select = max(int(len(sorted_indices) * 0.3), 1)
                    selected_indices = sorted_indices[:num_to_select]
                    
                    obj_gaussians.append(foreground_gaussians[selected_indices])
                else:
                    num_foreground_gaussians = max(int(0.5 * len(gaussian_indices)), 1)
                    obj_gaussians.append(gaussian_indices[torch.argsort(depth_values)[:num_foreground_gaussians]])

            # Store foreground Gaussian indices for the current object
            if obj_gaussians:
                visible_gaussians.append(torch.cat(obj_gaussians))
            else:
                visible_gaussians.append(torch.tensor([], dtype=torch.long, device=self.device))

        return visible_gaussians
    

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--patch", type=int, default=16, help="higher value, smaller object segmentated, but slower")
    parser.add_argument("--mask_generator", default="hqsam", type=str)

    args = get_combined_args(parser)
    associate = AssociateParams(args)

    with torch.no_grad():
        associator = GaussianMaskAssociator(model.extract(args), associate, args.iteration)
        associator.run_mask_association()
