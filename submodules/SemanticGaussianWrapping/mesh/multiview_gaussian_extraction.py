"""Multiview surface extraction from trained semantic Gaussians.

The topology follows GaussianWrapping's central interpretation of Gaussians as
oriented surface elements, while the visible element set is derived from the
actual calibrated training views.  A small z-buffer rejects internal Gaussian
layers before local PCA normals and screened Poisson reconstruction build the
continuous mesh.  Semantic attributes are queried from the final trained
``SemanticSurfaceField`` only after geometry is fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from .bounds import MeshSupportPolicy
from .training_field_extraction import (
    TrainingFieldMeshConfig,
    TrainingFieldMeshExtractor,
)
from .types import TriangleMesh
from utils.graphics_utils import get_world_to_view


ALGORITHM = "multiview-visible-oriented-semantic-gaussians"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MultiviewGaussianMeshConfig:
    view_count: int = 60
    visibility_width: int = 480
    minimum_visible_views: int = 3
    depth_tolerance_extent_fraction: float = 0.0016
    voxel_extent_fraction: float = 0.005
    normal_neighbors: int = 32
    poisson_depth: int = 9
    poisson_scale: float = 1.05
    poisson_threads: int = 8
    min_opacity: float = 0.05
    min_semantic_confidence: float = 0.35
    require_observation: bool = True
    trim_quantile: float = 0.001
    query_chunk_size: int = 2_048
    semantic_decode_chunk_size: int = 8_192

    def __post_init__(self) -> None:
        for name in (
            "view_count",
            "visibility_width",
            "minimum_visible_views",
            "normal_neighbors",
            "poisson_depth",
            "poisson_threads",
            "query_chunk_size",
            "semantic_decode_chunk_size",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "depth_tolerance_extent_fraction",
            "voxel_extent_fraction",
            "poisson_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_opacity", "min_semantic_confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not 0.0 <= float(self.trim_quantile) < 0.5:
            raise ValueError("trim_quantile must lie in [0,0.5)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _camera_value(camera: Any, *names: str) -> Any:
    for name in names:
        if hasattr(camera, name):
            return getattr(camera, name)
    raise AttributeError(f"camera does not expose any of {names}")


class MultiviewGaussianMeshExtractor:
    """Build continuous topology from multiview-visible trained Gaussians."""

    def __init__(
        self,
        surface_field: Any,
        gaussians: Any,
        semantic_decoder: Any,
        cameras: Sequence[Any],
        scene_extent: float,
        *,
        config: Optional[MultiviewGaussianMeshConfig] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not callable(getattr(surface_field, "query", None)):
            raise TypeError("the trained semantic surface field is required")
        if not callable(semantic_decoder):
            raise TypeError("the trained semantic decoder is required")
        if not cameras:
            raise ValueError("at least one calibrated training camera is required")
        if not math.isfinite(scene_extent) or scene_extent <= 0:
            raise ValueError("scene_extent must be finite and positive")
        self.surface_field = surface_field
        self.gaussians = gaussians
        self.semantic_decoder = semantic_decoder
        self.cameras = list(cameras)
        self.scene_extent = float(scene_extent)
        self.config = config or MultiviewGaussianMeshConfig()
        self.progress_callback = progress_callback

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def _trusted_points(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        policy = MeshSupportPolicy(
            min_opacity=self.config.min_opacity,
            min_semantic_confidence=self.config.min_semantic_confidence,
            require_observation=self.config.require_observation,
            trim_quantile=0.0,
        )
        indices = policy.selected_indices(self.gaussians)
        if not len(indices):
            raise RuntimeError("no Gaussian passes the trusted support policy")
        xyz = (
            self.gaussians.get_xyz.index_select(0, indices)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        lower = np.quantile(xyz, self.config.trim_quantile, axis=0)
        upper = np.quantile(xyz, 1.0 - self.config.trim_quantile, axis=0)
        keep = ((xyz >= lower) & (xyz <= upper)).all(axis=1)
        xyz = np.ascontiguousarray(xyz[keep], dtype=np.float64)
        gaussian_indices = indices[torch.as_tensor(keep, device=indices.device)]
        if not len(xyz):
            raise RuntimeError("robust support trim removed every Gaussian")
        return xyz, gaussian_indices.detach().cpu().numpy(), np.stack((lower, upper))

    def _selected_cameras(self) -> list[Any]:
        count = min(self.config.view_count, len(self.cameras))
        indices = np.linspace(0, len(self.cameras) - 1, count, dtype=int)
        return [self.cameras[index] for index in indices]

    @staticmethod
    def _camera_geometry(camera: Any) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(camera, "R") and hasattr(camera, "T"):
            world_to_view = get_world_to_view(
                np.asarray(camera.R), np.asarray(camera.T)
            ).astype(np.float64)
        else:
            stored = np.asarray(
                _camera_value(camera, "world_view_transform"),
                dtype=np.float64,
            )
            world_to_view = stored.T
        center = np.linalg.inv(world_to_view)[:3, 3]
        return world_to_view, center

    def _visible_surface_points(
        self,
        xyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        cameras = self._selected_cameras()
        counts = np.zeros(len(xyz), dtype=np.uint16)
        view_sum = np.zeros((len(xyz), 3), dtype=np.float32)
        tolerance = (
            self.scene_extent
            * self.config.depth_tolerance_extent_fraction
        )
        for number, camera in enumerate(cameras, 1):
            world_to_view, center = self._camera_geometry(camera)
            camera_xyz = (
                xyz @ world_to_view[:3, :3].T
                + world_to_view[:3, 3]
            )
            depth = camera_xyz[:, 2]
            source_width = int(
                _camera_value(camera, "image_width", "width")
            )
            source_height = int(
                _camera_value(camera, "image_height", "height")
            )
            width = int(self.config.visibility_width)
            height = max(1, int(round(width * source_height / source_width)))
            fov_x = float(_camera_value(camera, "FoVx", "FovX"))
            fov_y = float(_camera_value(camera, "FoVy", "FovY"))
            fx = float(getattr(camera, "Fx", getattr(camera, "fx", 0.0)))
            fy = float(getattr(camera, "Fy", getattr(camera, "fy", 0.0)))
            if not math.isfinite(fx) or fx <= 0.0:
                fx = 0.5 * source_width / math.tan(0.5 * fov_x)
            if not math.isfinite(fy) or fy <= 0.0:
                fy = 0.5 * source_height / math.tan(0.5 * fov_y)
            cx = float(
                getattr(
                    camera,
                    "Cx",
                    getattr(camera, "cx", (source_width - 1) * 0.5),
                )
            )
            cy = float(
                getattr(
                    camera,
                    "Cy",
                    getattr(camera, "cy", (source_height - 1) * 0.5),
                )
            )
            fx *= width / source_width
            fy *= height / source_height
            cx = (cx + 0.5) * width / source_width - 0.5
            cy = (cy + 0.5) * height / source_height - 0.5
            in_front = depth > 0.01
            rows = np.flatnonzero(in_front)
            u = np.rint(
                fx * camera_xyz[rows, 0] / depth[rows] + cx
            ).astype(np.int32)
            v = np.rint(
                fy * camera_xyz[rows, 1] / depth[rows] + cy
            ).astype(np.int32)
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            rows = rows[inside]
            u = u[inside]
            v = v[inside]
            pixel = v * width + u
            z_buffer = np.full(width * height, np.inf, dtype=np.float32)
            np.minimum.at(z_buffer, pixel, depth[rows])
            front = rows[
                depth[rows] <= z_buffer[pixel] + tolerance
            ]
            counts[front] += 1
            direction = center[None] - xyz[front]
            direction /= np.maximum(
                np.linalg.norm(direction, axis=1, keepdims=True),
                1e-8,
            )
            view_sum[front] += direction.astype(np.float32)
            if number == len(cameras) or number % 10 == 0:
                self._progress(
                    f"[multiview] visibility cameras {number}/{len(cameras)}"
                )
        visible = counts >= self.config.minimum_visible_views
        if not np.any(visible):
            raise RuntimeError("multiview z-buffer selected no surface Gaussian")
        directions = view_sum[visible]
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True),
            1e-8,
        )
        return xyz[visible], directions, {
            "visibility_cameras": int(len(cameras)),
            "visible_gaussians": int(np.count_nonzero(visible)),
            "visible_view_count_median": float(np.median(counts[visible])),
            "visible_view_count_p90": float(np.quantile(counts[visible], 0.9)),
            "depth_tolerance": float(tolerance),
        }

    def _poisson(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(points)
        )
        # Colors temporarily transport observation directions through Open3D's
        # deterministic voxel averaging.
        cloud.colors = o3d.utility.Vector3dVector(
            np.clip((directions + 1.0) * 0.5, 0.0, 1.0)
        )
        voxel_size = (
            self.scene_extent * self.config.voxel_extent_fraction
        )
        cloud = cloud.voxel_down_sample(voxel_size)
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamKNN(
                knn=self.config.normal_neighbors
            )
        )
        normals = np.asarray(cloud.normals)
        observed_direction = np.asarray(cloud.colors) * 2.0 - 1.0
        normals[
            np.sum(normals * observed_direction, axis=1) < 0.0
        ] *= -1.0
        cloud.normals = o3d.utility.Vector3dVector(normals)
        cloud.colors = o3d.utility.Vector3dVector(
            np.zeros_like(observed_direction)
        )
        self._progress(
            f"[multiview] Poisson input {len(cloud.points):,} points"
        )
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud,
            depth=self.config.poisson_depth,
            scale=self.config.poisson_scale,
            linear_fit=False,
            n_threads=self.config.poisson_threads,
        )
        mesh = mesh.crop(
            o3d.geometry.AxisAlignedBoundingBox(bounds[0], bounds[1])
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        mesh_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if not len(vertices) or not len(faces):
            raise RuntimeError("Poisson reconstruction produced no mesh")
        return vertices, faces, mesh_normals, {
            "voxel_size": float(voxel_size),
            "poisson_points": int(len(cloud.points)),
            "poisson_depth": int(self.config.poisson_depth),
        }

    def extract(self) -> TriangleMesh:
        xyz, _, bounds = self._trusted_points()
        points, directions, visibility_stats = (
            self._visible_surface_points(xyz)
        )
        vertices, faces, mesh_normals, poisson_stats = self._poisson(
            points, directions, bounds
        )
        helper = TrainingFieldMeshExtractor(
            self.surface_field,
            self.gaussians,
            self.semantic_decoder,
            config=TrainingFieldMeshConfig(
                query_chunk_size=self.config.query_chunk_size,
                semantic_decode_chunk_size=self.config.semantic_decode_chunk_size,
            ),
            progress_callback=self.progress_callback,
        )
        _, semantic, semantic_id, uncertainty, field_stats = (
            helper._vertex_attributes(vertices)
        )
        labels = semantic_id[faces]
        unanimous = np.all(labels == labels[:, :1], axis=1)
        face_region_id = np.where(
            unanimous, labels[:, 0], -2
        ).astype(np.int32)
        metadata = {
            "algorithm": ALGORITHM,
            "geometry_source": "multiview-visible trained Gaussian centers",
            "normal_source": "visible-neighborhood PCA",
            "semantic_source": "SemanticSurfaceField.query",
            "trusted_gaussians": int(len(xyz)),
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "mixed_semantic_faces": int(np.count_nonzero(~unanimous)),
            **visibility_stats,
            **poisson_stats,
            **field_stats,
        }
        return TriangleMesh(
            vertices=vertices,
            faces=faces,
            normals=mesh_normals,
            semantic=semantic,
            semantic_id=semantic_id,
            uncertainty=uncertainty,
            face_region_id=face_region_id,
            metadata=metadata,
        )


__all__ = [
    "ALGORITHM",
    "SCHEMA_VERSION",
    "MultiviewGaussianMeshConfig",
    "MultiviewGaussianMeshExtractor",
]
