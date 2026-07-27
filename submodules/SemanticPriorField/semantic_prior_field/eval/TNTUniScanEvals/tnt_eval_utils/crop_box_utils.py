import os
import torch
import numpy as np
import open3d as o3d

class PointFilter:
    def filter(self, points: torch.Tensor) -> torch.Tensor:
        """
        Filter points.
        Args:
            points: (N, 3) torch tensor in world space.
        Returns:
            mask: (N,) boolean torch tensor where True means keep the point.
        """
        raise NotImplementedError

class TransformedSelectionPolygonVolume:
    def __init__(self, vol, transform):
        self.vol = vol
        if isinstance(transform, torch.Tensor):
            self.transform = transform.detach().cpu().numpy()
        else:
            self.transform = np.array(transform)
        self.inv_transform = np.linalg.inv(self.transform)

    def crop_point_cloud(self, pcd):
        # Transform pcd to volume frame
        pcd.transform(self.transform)
        # Crop
        cropped_pcd = self.vol.crop_point_cloud(pcd)
        # Transform result back to world frame
        cropped_pcd.transform(self.inv_transform)
        return cropped_pcd

def create_o3d_pcd(points: np.ndarray, normals: np.ndarray) -> o3d.geometry.PointCloud:
    # Make sure points and normals are contiguous
    points = np.ascontiguousarray(points, dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

class CropBoxFilter(PointFilter):
    def __init__(self, crop_json_path, transform=None):
        """
        Initialize CropBoxFilter.
        Args:
            crop_json_path: Path to the .json file defining the selection polygon volume.
            transform: Optional (4, 4) transformation matrix to apply to points before checking against the crop box.
                       This is useful if the crop box is in a different coordinate system (e.g. GT space) than the points.
        """
        if not os.path.exists(crop_json_path):
            raise FileNotFoundError(f"Crop file not found: {crop_json_path}")
        
        print(f"[INFO] Loading crop volume from {crop_json_path}")
        self.vol = o3d.visualization.read_selection_polygon_volume(crop_json_path)
        
        self.transform = None
        if transform is not None:
            self.transform = torch.tensor(transform, dtype=torch.float32)

    def filter_through_pcd(self, points: torch.Tensor, normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert Torch to Numpy
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(normals, torch.Tensor):
            normals = normals.cpu().numpy()

        # Maps points to the crop box frame, crops them, and then projects them back
        pcd = create_o3d_pcd(points, normals)
        cropped_pcd = TransformedSelectionPolygonVolume(self.vol, self.transform).crop_point_cloud(pcd)

        return np.asarray(cropped_pcd.points), np.asarray(cropped_pcd.normals)