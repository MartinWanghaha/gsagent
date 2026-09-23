# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial import KDTree
import open3d as o3d
from utils.compose_utils import create_from_pcd_our, load_ply_our, similar_points_tree

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """Build covariance matrix from scaling factors and quaternion rotation."""
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid    

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self._objects_dc = torch.empty(0)
        self.num_objects = 16
        self.setup_functions()

    def capture(self):
        """
        
        """
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._objects_dc,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        """
        
        """
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._objects_dc,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc        
        features_rest = self._features_rest   
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_objects(self):
        return self._objects_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """Increment active SH degree up to the configured maximum."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """
        input:
            pcd: point cloud
            spatial_lr_scale: radius scale factor (~1.1x of point cloud radius)
        
        """

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        # random init obj_id now
        fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0],self.num_objects), device="cuda"))
        fused_objects = fused_objects[:,:,None]

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._objects_dc = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True))

    def training_setup(self, training_args):
        """
        Initialize optimizer and learning rates.
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def training_setup_distill(self, training_args):
        """
        Setup optimizer for distillation; only object embeddings are updated.
        """

        l = [
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self._xyz.requires_grad = False
        self._features_dc.requires_grad = False
        self._features_rest.requires_grad = False
        self._opacity.requires_grad = False
        self._scaling.requires_grad = False
        self._rotation.requires_grad = False
    
    def finetune_setup(self, training_args, mask3d):
        # Define a function that applies the mask to the gradients
        def mask_hook(grad):
            return grad * mask3d
        def mask_hook2(grad):
            return grad * mask3d.squeeze(-1)
        

        # Register the hook to the parameter (only once!)
        hook_xyz = self._xyz.register_hook(mask_hook2)
        hook_dc = self._features_dc.register_hook(mask_hook)
        hook_rest = self._features_rest.register_hook(mask_hook)
        hook_opacity = self._opacity.register_hook(mask_hook2)
        hook_scaling = self._scaling.register_hook(mask_hook2)
        hook_rotation = self._rotation.register_hook(mask_hook2)

        self._objects_dc.requires_grad = False

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def removal_setup(self, masks_per_obj):
        """
        Args:
            masks_per_obj: dict[obj_id -> {"mask3d": tensor [N, 1, 1]}]; mask
                entries set to 1/True mark points belonging to that object.
        """
        first_object = next(iter(masks_per_obj))
        combined_mask3d = torch.zeros_like(
            masks_per_obj[first_object]["mask3d"].squeeze(),
            dtype=torch.bool,
        )

        objects_model = {}

        for obj_id, masks in masks_per_obj.items():
            mask3d = masks["mask3d"].squeeze().bool()
            # Convex object masks may overlap. Assign every Gaussian to the
            # first selected object only (targets are inserted before temporary
            # surroundings), so restoring surroundings can neither duplicate
            # points nor reintroduce points owned by a removal target.
            object_mask = mask3d & ~combined_mask3d
            
            sub = GaussianModel(self.max_sh_degree)
            sub._xyz           = nn.Parameter(self._xyz[object_mask].detach(),           requires_grad=True)
            sub._features_dc   = nn.Parameter(self._features_dc[object_mask].detach(),   requires_grad=True)
            sub._features_rest = nn.Parameter(self._features_rest[object_mask].detach(), requires_grad=True)
            sub._opacity       = nn.Parameter(self._opacity[object_mask].detach(),       requires_grad=True)
            sub._scaling       = nn.Parameter(self._scaling[object_mask].detach(),       requires_grad=True)
            sub._rotation      = nn.Parameter(self._rotation[object_mask].detach(),      requires_grad=True)
            sub._objects_dc    = nn.Parameter(self._objects_dc[object_mask].detach(),    requires_grad=True)

            objects_model[obj_id] = sub

            combined_mask3d |= mask3d
        
        combined_mask3d = ~combined_mask3d

        # Extracting subsets using the mask
        xyz_sub = self._xyz[combined_mask3d].detach()
        features_dc_sub = self._features_dc[combined_mask3d].detach()
        features_rest_sub = self._features_rest[combined_mask3d].detach()
        opacity_sub = self._opacity[combined_mask3d].detach()
        scaling_sub = self._scaling[combined_mask3d].detach()
        rotation_sub = self._rotation[combined_mask3d].detach()
        objects_dc_sub = self._objects_dc[combined_mask3d].detach()


        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)


        self._xyz = nn.Parameter(set_requires_grad(xyz_sub, False))
        self._features_dc = nn.Parameter(set_requires_grad(features_dc_sub, False))
        self._features_rest = nn.Parameter(set_requires_grad(features_rest_sub, False))
        self._opacity = nn.Parameter(set_requires_grad(opacity_sub, False))
        self._scaling = nn.Parameter(set_requires_grad(scaling_sub, False))
        self._rotation = nn.Parameter(set_requires_grad(rotation_sub, False))
        self._objects_dc = nn.Parameter(set_requires_grad(objects_dc_sub, False))

        return objects_model

    def inpaint_setup(self, args, training_args, masks_per_obj):

        first_object = next(iter(masks_per_obj))
        combined_mask3d = torch.zeros_like(masks_per_obj[first_object]["mask3d"], dtype=torch.bool)
        objects_model = {}

        for obj_id, masks in masks_per_obj.items():
            mask3d = masks["mask3d"]
            
            sub = GaussianModel(self.max_sh_degree)
            sub._xyz           = nn.Parameter(self._xyz[mask3d.squeeze().bool()].detach(),           requires_grad=True)
            sub._features_dc   = nn.Parameter(self._features_dc[mask3d.squeeze().bool()].detach(),   requires_grad=True)
            sub._features_rest = nn.Parameter(self._features_rest[mask3d.squeeze().bool()].detach(), requires_grad=True)
            sub._opacity       = nn.Parameter(self._opacity[mask3d.squeeze().bool()].detach(),       requires_grad=True)
            sub._scaling       = nn.Parameter(self._scaling[mask3d.squeeze().bool()].detach(),       requires_grad=True)
            sub._rotation      = nn.Parameter(self._rotation[mask3d.squeeze().bool()].detach(),      requires_grad=True)
            sub._objects_dc    = nn.Parameter(self._objects_dc[mask3d.squeeze().bool()].detach(),    requires_grad=True)

            objects_model[obj_id] = sub

            combined_mask3d |= masks_per_obj[obj_id]["mask3d"].bool()
        
        combined_mask3d = ~combined_mask3d.bool().squeeze() 
        
        self.sub_feature_num = combined_mask3d.sum()  

        # Extracting subsets using the mask
        xyz_sub = self._xyz[combined_mask3d].detach()
        features_dc_sub = self._features_dc[combined_mask3d].detach()
        features_rest_sub = self._features_rest[combined_mask3d].detach()
        opacity_sub = self._opacity[combined_mask3d].detach()
        scaling_sub = self._scaling[combined_mask3d].detach()
        rotation_sub = self._rotation[combined_mask3d].detach()
        objects_dc_sub = self._objects_dc[combined_mask3d].detach()

        mask_xyz_values = self._xyz[~combined_mask3d]

        # Add new points with random initialization
        sub_features = {
            'xyz': xyz_sub,
            'features_dc': features_dc_sub,
            'scaling': scaling_sub,
            'objects_dc': objects_dc_sub,
            'features_rest': features_rest_sub,
            'opacity': opacity_sub,
            'rotation': rotation_sub,
        }

        print("\n There are {} points in the after-removal Gaussians.".format(len(xyz_sub)))

        def compose_ply_our(args, sub_features):
            
            supp = args.supp_ply
            nb_points = args.nb_points
            radius = args.radius
            threshold = args.threshold
            processed_supp = args.temp_ply

            plydata = PlyData.read(supp)    
            x = plydata['vertex']['x']
            y = plydata['vertex']['y']
            z = plydata['vertex']['z']
            r = plydata['vertex']['red']
            g = plydata['vertex']['green']
            b = plydata['vertex']['blue']
            points = np.column_stack([x, y, z])
            colors = np.column_stack([r, g, b])
            if len(points)==0 and getattr(args,'density_manifest',None):
                # Validated zero-quota result: preserve background, no dummy splat.
                return tuple(value[:0].detach().clone() for value in (
                    xyz_sub,features_dc_sub,features_rest_sub,opacity_sub,
                    scaling_sub,rotation_sub,objects_dc_sub))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            normalized_colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(normalized_colors)
            # Adaptive supports are hash-validated by the entry point. Sampling quotas
            # must not be reduced by another statistical/radius filter.
            if not getattr(args, "density_manifest", None):
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=4.0)
                pcd = pcd.select_by_index(ind)
            scale_reference=(xyz_sub.cpu().numpy() if getattr(args,'density_manifest',None) and len(points)<4 else None)
            create_from_pcd_our(pcd, processed_supp, args,scale_reference=scale_reference)

            # Load Painted Gaussians and Original Gaussians
            new_xyz, new_features_dc, new_features_extra, new_opacities, new_scales, new_rots, new_objects_dc = load_ply_our(processed_supp, sub_features)
            print("\n There are {} points in the inpainted Gaussians.".format(len(new_xyz)))

            remove_simililar_points = False    # TODO TODO TODO
            if remove_simililar_points:
                # Calculate points near Inpainting Gaussians
                points = xyz_sub 
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points) 
                xyz_similar = similar_points_tree(xyz_sub, new_xyz, threshold)  
                print("\n There are {} similar points in the two point clouds.".format(len(xyz_similar)))

                # Remove the floaters point in the complemented area 
                cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius) 
                real = set(xyz_similar.tolist()).difference(set(ind))  
                ind = list(real)
                print("\n Too many similar points in original Gaussians, so {} floaters were removed from it.".format(len(ind))) 

                indices = np.array([True] * len(xyz_sub)) 
                indices[ind] = False 
                new_xyz = xyz_sub[indices]
                new_features_dc = features_dc_sub[indices]
                new_features_extra = features_rest_sub[indices]
                new_opacities = opacity_sub[indices]
                new_scales = scaling_sub[indices]
                new_rots = rotation_sub[indices]
                new_objects_dc = objects_dc_sub[indices]

            new_xyz = torch.tensor(new_xyz, dtype=torch.float, device="cuda")
            new_features_dc = torch.tensor(new_features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            new_features_extra = torch.tensor(new_features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            new_opacities = torch.tensor(new_opacities, dtype=torch.float, device="cuda")
            new_scales = torch.tensor(new_scales , dtype=torch.float, device="cuda")
            new_rots = torch.tensor(new_rots, dtype=torch.float, device="cuda")
            new_objects_dc = torch.tensor(new_objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()

            return new_xyz, new_features_dc, new_features_extra, new_opacities, new_scales, new_rots, new_objects_dc

        new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, \
        new_rotation, new_objects_dc = compose_ply_our(args, sub_features)


        def set_requires_grad(tensor, requires_grad):
            """Returns a new tensor with the specified requires_grad setting."""
            return tensor.detach().clone().requires_grad_(requires_grad)

        # Construct nn.Parameters with specified gradients
        self._xyz = nn.Parameter(torch.cat([set_requires_grad(xyz_sub, False), set_requires_grad(new_xyz, True)]))
        self._features_dc = nn.Parameter(torch.cat([set_requires_grad(features_dc_sub, False), set_requires_grad(new_features_dc, True)]))
        self._features_rest = nn.Parameter(torch.cat([set_requires_grad(features_rest_sub, False), set_requires_grad(new_features_rest, True)]))
        self._opacity = nn.Parameter(torch.cat([set_requires_grad(opacity_sub, False), set_requires_grad(new_opacity, True)]))
        self._scaling = nn.Parameter(torch.cat([set_requires_grad(scaling_sub, False), set_requires_grad(new_scaling, True)]))
        self._rotation = nn.Parameter(torch.cat([set_requires_grad(rotation_sub, False), set_requires_grad(new_rotation, True)]))
        self._objects_dc = nn.Parameter(torch.cat([set_requires_grad(objects_dc_sub, False), set_requires_grad(new_objects_dc, True)]))

        # for optimize   
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Setup optimizer. Only the new points will have gradients.
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"}, 
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},                     
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},          
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},    
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},    
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}, 
            {'params': [self._objects_dc], 'lr': training_args.feature_lr, "name": "obj_dc"}   # Assuming there's a learning rate for objects_dc in training_args
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step'''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._objects_dc.shape[1]*self._objects_dc.shape[2]):
            l.append('obj_dc_{}'.format(i))
        return l

    def save_ply(self, path):

        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        obj_dc = self._objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):

        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
    
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))             # ['scale_0', 'scale_1', 'scale_2']
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))           #['rot_0', 'rot_1', 'rot_2', 'rot_3']
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])                

        objects_dc = np.zeros((xyz.shape[0], self.num_objects, 1))
        try:
            for idx in range(self.num_objects):
                objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])   
        except:
            print("\nWe do not have object feature for the vanilla 3DGS!!!")
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._objects_dc = nn.Parameter(torch.tensor(objects_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
    
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = optimizable_tensors["obj_dc"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
      
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc):
        """
        
        """
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "obj_dc": new_objects_dc}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)   
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._objects_dc = optimizable_tensors["obj_dc"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")   
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) 

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)                  
        means = torch.zeros((stds.size(0), 3),device="cuda")                   
        samples = torch.normal(mean=means, std=stds)                            
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1) 

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1) 
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))    
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N,1,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        Duplicate points with high gradients to address under-reconstruction.

        Args:
            grads: self.xyz_gradient_accum / self.denom
            grad_threshold: gradient magnitude threshold (e.g., 0.0002)
            scene_extent: scene radius used for density thresholds
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_objects_dc = self._objects_dc[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold
        Args:
            max_grad: gradient threshold indicating under/over reconstruction
            min_opacity: opacity threshold below which points are pruned
            extent: scene radius used for density thresholds
            max_screen_size: optional pixel footprint threshold
        
        
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent) # under reconstruction
        self.densify_and_split(grads, max_grad, extent) # over reconstruction

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def densify_and_prune_inpaint(self, max_grad, min_opacity, extent, max_screen_size, sub_feature_num):
        """
        opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold
        Args:
            max_grad: gradient threshold indicating under/over reconstruction
            min_opacity: opacity threshold below which points are pruned
            extent: scene radius used for density thresholds
            max_screen_size: optional pixel footprint threshold
            sub_feature_num: number of remaining points (not inpainted)
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        def densify_and_clone_inpaint(grads, grad_threshold, scene_extent, sub_feature_num):
            """
            Clone high-gradient points for inpainted regions; existing
            remaining points (first sub_feature_num) are not duplicated.

            Args:
                grads: self.xyz_gradient_accum / self.denom
                grad_threshold: gradient threshold
                scene_extent: scene radius used for density thresholds
                sub_feature_num: number of remaining points
            """
            # Extract points that satisfy the gradient condition
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
            
            selected_pts_mask[:sub_feature_num] = False

            new_xyz = self._xyz[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_objects_dc = self._objects_dc[selected_pts_mask]

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_objects_dc)

        def densify_and_split_inpaint(grads, grad_threshold, scene_extent, sub_feature_num, N=2):
            """
            Split high-gradient points to fix over-reconstruction while keeping
            the remaining points (first sub_feature_num) fixed in count.

            Args:
                grads: self.xyz_gradient_accum / self.denom
                grad_threshold: gradient threshold
                scene_extent: scene radius used for density thresholds
                sub_feature_num: number of remaining points after removal
                N: split factor
            """
            n_init_points = self.get_xyz.shape[0]
            # Extract points that satisfy the gradient condition
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

            selected_pts_mask[:sub_feature_num] = False

            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
            new_objects_dc = self._objects_dc[selected_pts_mask].repeat(N,1,1)

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_objects_dc)

            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter)


        densify_and_clone_inpaint(grads, max_grad, extent, sub_feature_num=sub_feature_num) # under reconstruction
        densify_and_split_inpaint(grads, max_grad, extent, sub_feature_num=sub_feature_num) # over reconstruction

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            prune_mask[:sub_feature_num] = False
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        Accumulate gradient and visibility counts per Gaussian across views.

        Args:
            viewspace_point_tensor: 3D Gaussian positions projected to screen space
            update_filter: bool mask indicating which points are visible in this view
        """

        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
