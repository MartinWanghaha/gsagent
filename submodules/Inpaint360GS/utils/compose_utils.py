import numpy as np
from plyfile import PlyData, PlyElement
import torch
from sklearn.neighbors import KDTree

from simple_knn._C import distCUDA2
from torch import nn
from pathlib import Path


C0 = 0.28209479177387814


def _sh_degree_from_args(args):
    """Resolve SH degree from the loaded model while retaining legacy fallback."""
    value = getattr(args, "sh_degree", 3)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"sh_degree must be a non-negative integer, got {value!r}")
    return value


def _extra_feature_names(plydata, sub_features=None):
    names = [
        prop.name
        for prop in plydata.elements[0].properties
        if prop.name.startswith("f_rest_")
    ]
    names = sorted(names, key=lambda name: int(name.split("_")[-1]))
    if len(names) % 3 != 0:
        raise ValueError(
            f"f_rest property count must be divisible by 3, found {len(names)}"
        )
    expected_names = [f"f_rest_{index}" for index in range(len(names))]
    if names != expected_names:
        raise ValueError("f_rest properties must be contiguous from f_rest_0")
    if sub_features is not None:
        reference = sub_features.get("features_rest")
        if reference is None or reference.ndim != 3:
            raise ValueError("sub_features.features_rest must have shape [N, 3, K]")
        expected_count = int(reference.shape[1] * reference.shape[2])
        if len(names) != expected_count:
            raise ValueError(
                f"temporary PLY has {len(names)} f_rest properties, "
                f"but the source model requires {expected_count}"
            )
    return names


def _load_rest_features(plydata, point_count, names):
    features = np.zeros((point_count, len(names)))
    for index, name in enumerate(names):
        features[:, index] = np.asarray(plydata.elements[0][name])
    return features.reshape((point_count, 3, len(names) // 3))

def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]    # height
    xmin, xmax = torch.where(cols)[0][[0, -1]]    # weight
    
    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

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

def create_from_pcd_our(pcd, path, args, scale_reference=None):
    sh_degree = _sh_degree_from_args(args)
    points = np.asarray(pcd.points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("the fused point cloud must contain at least one XYZ point")
    colors = np.asarray(pcd.colors)
    if colors.shape != points.shape or not np.isfinite(colors).all():
        raise ValueError("the fused point cloud colors must be finite and match XYZ")
    if not np.isfinite(points).all():
        raise ValueError("the fused point cloud XYZ values must be finite")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fused_point_cloud = torch.tensor(points).float().cuda()
    fused_color = RGB2SH(torch.tensor(colors).float().cuda())
    features = torch.zeros(
        (fused_color.shape[0], 3, (sh_degree + 1) ** 2),
        dtype=torch.float32,
        device="cuda",
    )
    features[:, :3, 0 ] = fused_color

    opacity_init = float(args.opacity_init)
    if not 0.0 < opacity_init < 1.0:
        raise ValueError("opacity_init must be strictly between 0 and 1")

    # random init obj_id now
    fused_objects = RGB2SH(torch.rand((fused_point_cloud.shape[0], 16), device="cuda"))
    fused_objects = fused_objects[:,:,None] 

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    if scale_reference is not None and len(pcd.points)<4:
        # distCUDA2 needs three other points. A valid adaptive quota can be
        # 1--3: use retained neighbors for scale only, without adding points.
        from scipy.spatial import cKDTree
        points=np.asarray(pcd.points)
        reference=np.asarray(scale_reference)
        joint=np.concatenate((points,reference),axis=0)
        if len(joint)<2:
            raise ValueError('cannot initialize scale without any neighboring center')
        distances=cKDTree(joint).query(points,k=min(4,len(joint)))[0][:,1:]
        dist2=torch.as_tensor(np.mean(distances**2,axis=1),dtype=torch.float32,device='cuda')
        dist2=torch.clamp_min(dist2,0.0000001)
    else:
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = inverse_sigmoid(opacity_init * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))   #  
    scaling = nn.Parameter(scales.requires_grad_(True))
    rotation = nn.Parameter(rots.requires_grad_(True))
    opacity = nn.Parameter(opacities.requires_grad_(True))
    objects_dc = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(False))
    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scale = scaling.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()
    obj_dc = objects_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()   # Gaussian Grouping

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc, features_rest, scaling, rotation, objects_dc)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output)

def load_ply_our(path, sub_features=None):
    """
    Load a PLY file and optionally initialize its features using KNN-based interpolation 
    from a reference set of sub-features.

    Args:
        path (str): Path to the .ply file containing Gaussian data.
        sub_features (dict, optional): Reference features used for spatial interpolation (KNN).
                                       Expected to contain 'xyz', 'scaling', 'rotation', etc.
    Returns:
        tuple: (xyz, features_dc, features_rest, opacity, scales, rots, objects_dc)
    """
    plydata = PlyData.read(path)
    
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    smarter_ini = sub_features is not None

    if smarter_ini:
        new_features = {}
        source_xyz = sub_features["xyz"].detach().cpu().numpy()
        if len(source_xyz) == 0:
            raise ValueError("cannot initialize inpainted points from an empty source model")
        kdtree = KDTree(source_xyz)
        _, indices = kdtree.query(xyz, k=min(5, len(source_xyz)))
        # Initialize new points for each feature
        for key, feature in sub_features.items():
            # key  'xyz' 'features_dc' 'scaling' 'objects_dc' 'features_rest' 'opacity' 'rotation'
            feature_np = feature.detach().cpu().numpy()
            
            # If we have valid neighbors, calculate the mean of neighbor points
            if feature_np.ndim == 2:
                neighbor_points = feature_np[indices]
            elif feature_np.ndim == 3:
                neighbor_points = feature_np[indices, :, :]
            else:
                raise ValueError(f"Unsupported feature dimension: {feature_np.ndim}")
            new_points_np = np.mean(neighbor_points, axis=1)   # knn feature
            
            # Convert back to tensor
            new_features[key] = new_points_np.astype(feature_np.dtype)

        new_features['xyz'] = xyz
        new_features['opacity'] = opacities
        new_features['features_dc'] = features_dc


        extra_f_names = _extra_feature_names(plydata, sub_features)
        features_rest = _load_rest_features(plydata, len(xyz), extra_f_names)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        Path(path).unlink()
        
        return new_features['xyz'], new_features['features_dc'], features_rest, new_features['opacity'], scales, rots, new_features['objects_dc'].transpose(0, 2, 1)
                                                                                             
    else:
        extra_f_names = _extra_feature_names(plydata)
        features_rest = _load_rest_features(plydata, len(xyz), extra_f_names)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # TODO TODO TODO 
        objects_dc = np.zeros((xyz.shape[0], 16, 1))          
        for idx in range(16):
            objects_dc[:, idx, 0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])

        Path(path).unlink()
        # xyz features_dc features_rest opacities scales rots
        return xyz, features_dc, features_rest, opacities, scales, rots, objects_dc


def similar_points_tree(point_cloud_A, point_cloud_B, threshold=1.0):

    tree = KDTree(point_cloud_B)
    distances, indices = tree.query(point_cloud_A, k=1)
    similar_indices = np.where(distances < threshold)[0]

    return similar_indices
