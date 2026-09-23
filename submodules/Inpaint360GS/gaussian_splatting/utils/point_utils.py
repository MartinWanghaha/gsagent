import numpy as np
import torch

def create_point_cloud(depth_map, intrinsic_matrix, extrinsic_matrix):
  
    H, W = depth_map.shape

    # Create meshgrid for pixel coordinates
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    x, y = np.meshgrid(x, y)

    # Normalize pixel coordinates
    normalized_x = (x - intrinsic_matrix[0, 2]) / intrinsic_matrix[0, 0]    # (x-cx)/fx
    normalized_y = (y - intrinsic_matrix[1, 2]) / intrinsic_matrix[1, 1]
    normalized_z = np.ones_like(x)

     # Homogeneous coordinates in camera frame
    depth_map_reshaped = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
    homogeneous_camera_coords = depth_map_reshaped * np.dstack((normalized_x, 
                                                                normalized_y, 
                                                                normalized_z)) 
    
    # add ones to the last dimention
    ones = np.ones((H, W, 1))
    homogeneous_camera_coords = np.dstack((homogeneous_camera_coords, ones))

    homogeneous_world_coords = homogeneous_camera_coords @ extrinsic_matrix.T

    point_cloud = (homogeneous_world_coords[:, :, :3] / 
                                            homogeneous_world_coords[:, :, 3:])

    point_cloud = point_cloud.reshape(-1, 3)

    return point_cloud

def ply_color_fusion(points, colors, ply_path, mask=None):

    if mask is None:
        mask = np.ones(colors.shape[0], dtype=bool)

    num = np.sum(mask)
    ply_header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''.format(num)
    
    valid_points = points[mask]
    valid_colors = colors[mask]

    lines = np.column_stack((
        valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
        (valid_colors[:, 2]).astype(np.int32),  # Red
        (valid_colors[:, 1]).astype(np.int32),  # Green
        (valid_colors[:, 0]).astype(np.int32),  # Blue
    )).astype(object)

    lines[:, 3:] = lines[:, 3:].astype(np.int32)

    lines_str = ["{:f} {:f} {:f} {:d} {:d} {:d}\n".format(*line) for line in lines]
    
    with open(ply_path, "w") as f:
        f.write(ply_header)
        f.writelines(lines_str)


def get_intrinsics(H, W, fovx, fovy):
    fx = 0.5 * W / np.tan(0.5 * fovx)
    fy = 0.5 * H / np.tan(0.5 * fovy)
    cx = 0.5 * W
    cy = 0.5 * H

    intrinsic_matrix = np.array([[fx, 0,  cx],
                                 [0,  fy, cy],
                                 [0,  0,   1]])
    return intrinsic_matrix


def project_3d_points(point, matrix):
    """Applies a 4x4 transformation matrix to a set of 3D points.

    :param point: (N, 3) 3D points.
    :param matrix: (4, 4) transformation matrix.
    :return: Transformed points in homogeneous coordinates.
    """
    point = torch.cat([point, torch.ones_like(point[:, :1])], dim=1)
    transformed = torch.matmul(point, matrix)
    return transformed


def ndc_to_pixel(ndc_coord: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Convert normalized device coordinates (NDC) to pixel coordinates.

    :param ndc_coord: Tensor of NDC coordinates (range: [-1, 1]).
    :param image_size: Image width or height in pixels.
    :return: Pixel coordinates in the range [0, image_size - 1].
    """
    return (ndc_coord + 1.0) * 0.5 * (image_size - 1)
