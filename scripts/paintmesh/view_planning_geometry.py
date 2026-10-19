"""Unsigned mesh queries and conservative observed-space/path certificates.

No inside/outside assumption: open TSDF walls are valid collision surfaces.
The renderer's similarity transform is retained, including camera-space depth.
"""
from __future__ import annotations

import numpy as np


def world_to_camera(camera):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(camera["R"]).T
    matrix[:3, 3] = camera["T"]
    inverse = np.linalg.inv(matrix)
    inverse[:3, 3] = (inverse[:3, 3] + camera["trans"]) * camera["scale"]
    return np.linalg.inv(inverse)


def camera_center(camera):
    return np.linalg.inv(world_to_camera(camera))[:3, 3]


def near_radius(camera):
    inverse = np.linalg.inv(world_to_camera(camera)[:3, :3])
    z = camera["znear"]
    corners = np.array([[x*np.tan(camera["FoVx"]/2)*z,
                         y*np.tan(camera["FoVy"]/2)*z, z] for x in (-1, 1) for y in (-1, 1)])
    return float(np.linalg.norm(corners @ inverse.T, axis=1).max())


def project(camera, points):
    matrix = world_to_camera(camera)
    xyz = np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]
    z = xyz[:, 2]
    xy = xyz[:, :2] / np.maximum(z[:, None], 1e-15)
    uv = xy / np.tan([camera["FoVx"]/2, camera["FoVy"]/2]) / 2 + .5
    return uv, z


def image_rays(camera):
    h, w = camera["image_height"], camera["image_width"]
    yy, xx = np.mgrid[:h, :w]
    vectors = np.stack(((xx-w/2)/(w/2)*np.tan(camera["FoVx"]/2),
                        (yy-h/2)/(h/2)*np.tan(camera["FoVy"]/2), np.ones_like(xx)), -1)
    vectors = vectors @ np.linalg.inv(world_to_camera(camera)[:3, :3]).T
    lengths = np.linalg.norm(vectors, axis=-1)
    return camera_center(camera), vectors/lengths[..., None], lengths


class MeshScene:
    def __init__(self, vertices, triangles):
        import open3d as o3d
        vertices = np.asarray(vertices)
        triangles = np.asarray(triangles)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)):
            raise ValueError("collision mesh must have finite vertices and triangles")
        if triangles.dtype.kind not in "iu" or triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError("collision mesh has invalid triangle indices")
        self.o3d = o3d
        self.offset = (vertices.min(0)+vertices.max(0))/2
        # The renderer still uses float32 WORLD transforms. Recentering the
        # query mesh improves ray precision but cannot erase that render error.
        self.numeric_floor = float(max(1., np.abs(vertices).max()) * np.finfo(np.float32).eps * 32)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor((vertices-self.offset).astype(np.float32)),
                                 o3d.core.Tensor(triangles.astype(np.uint32)))

    @classmethod
    def from_file(cls, path):
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(path))
        return cls(mesh.vertices, mesh.triangles)

    def distance(self, positions):
        points = (np.asarray(positions, dtype=float).reshape(-1, 3)-self.offset).astype(np.float32)
        return self.scene.compute_distance(self.o3d.core.Tensor(points)).numpy().astype(float)

    def cast(self, origins, directions):
        directions = np.asarray(directions)
        origins = np.broadcast_to(origins, directions.shape)-self.offset
        rays = np.concatenate((origins, directions), -1).astype(np.float32)
        return self.scene.cast_rays(self.o3d.core.Tensor(rays))["t_hit"].numpy().astype(float)


class ObservedFreeSpace:
    """Union of reliable real-camera depth tiles plus validated origin balls.

    Every tile uses ALL pixels, not interpolated sparse rays. Its far plane is
    the minimum corroborated mesh/Gaussian depth minus uncertainty. A query
    ball must fit entirely in one convex tile or origin ball. This deliberately
    rejects uncertain boundaries; it is an estimate from reconstructed evidence.
    """
    def __init__(self):
        self.planes = np.empty((0, 6, 4))
        self.origins = []
        self.origin_radii = []
        self.stats = []

    def contains_ball(self, center, radius):
        center = np.asarray(center)
        if self.origins:
            if np.any(np.linalg.norm(np.asarray(self.origins)-center, axis=1) + radius
                      <= np.asarray(self.origin_radii)):
                return True
        if not len(self.planes):
            return False
        # Plane normals are normalized in WORLD units.
        distances = self.planes[:, :, :3] @ center + self.planes[:, :, 3]
        return bool(np.any(np.all(distances >= radius, axis=1)))

    def add_view(self, scene, camera, depth, alpha, target_radius, config, safe_radius):
        origin, directions, lengths = image_rays(camera)
        mesh_z = scene.cast(origin, directions) / lengths
        depth_scale = np.linalg.norm(world_to_camera(camera)[2, :3])
        tolerance = target_radius * config["depth_agreement_over_target_radius"] * depth_scale
        valid = (np.isfinite(mesh_z) & np.isfinite(depth) & (depth > 0)
                 & (alpha >= config["alpha_min"]) & (np.abs(mesh_z-depth) <= tolerance)
                 & (mesh_z < camera["zfar"]))
        h, w = depth.shape
        if depth.shape != directions.shape[:2] or alpha.shape != depth.shape:
            raise ValueError("observed-space preview/camera dimensions mismatch")
        tile = config["tile_pixels"]
        if tile < 2:
            raise ValueError("free-space tile_pixels must be >= 2")
        a = world_to_camera(camera)
        planes = []
        for y in range(0, h-tile+1, max(1, tile//2)):
            for x in range(0, w-tile+1, max(1, tile//2)):
                region = np.s_[y:y+tile, x:x+tile]
                if not valid[region].all():
                    continue
                z = np.minimum(mesh_z[region], depth[region])
                if np.ptp(z) > target_radius * config["max_depth_spread_over_target_radius"] * depth_scale:
                    continue
                far = float(z.min() - tolerance)
                if far <= camera["znear"]:
                    continue
                # Stay between actual sampled pixel centres (no unobserved rim).
                lo = ((np.array([x, y]) / [w, h]) * 2 - 1) * np.tan([camera["FoVx"]/2, camera["FoVy"]/2])
                hi = ((np.array([x+tile-1, y+tile-1]) / [w, h]) * 2 - 1) * np.tan([camera["FoVx"]/2, camera["FoVy"]/2])
                camera_planes = np.array([[1, 0, -lo[0], 0], [-1, 0, hi[0], 0],
                    [0, 1, -lo[1], 0], [0, -1, hi[1], 0],
                    [0, 0, 1, -camera["znear"]], [0, 0, -1, far]])
                world_planes = camera_planes @ a
                world_planes /= np.linalg.norm(world_planes[:, :3], axis=1)[:, None]
                planes.append(world_planes)
        clearance = float(scene.distance([origin])[0])
        # A real camera origin with corroborated foreground hits is a separate
        # observation anchor; it is still rejected when too close to geometry.
        anchor = len(planes) > 0 and clearance > safe_radius * 2
        if anchor:
            self.origins.append(origin)
            self.origin_radii.append(min(clearance * .5, max(safe_radius * 2, target_radius * .1)))
        if planes:
            if len(self.planes) + len(planes) > config["max_tiles"]:
                raise ValueError("observed-space tile resource limit exceeded; increase explicit max_tiles")
            self.planes = np.concatenate((self.planes, planes))
        self.stats.append(dict(source_image=camera.get("source_image_name", camera["image_name"]),
            reliable_pixels=int(valid.sum()), trusted_tiles=len(planes), anchor=bool(anchor)))


def certify_segment(scene, free_space, start, end, safe_radius, *, margin, min_step, max_depth):
    """1-Lipschitz UDF bound for the WHOLE swept sphere; fail closed.

    The same enclosing midpoint ball must lie in observed space, so neither
    a thin wall nor an unknown interval can be skipped by point sampling.
    """
    stack = [(np.asarray(start, float), np.asarray(end, float), 0)]
    lower, leaves = float("inf"), 0
    while stack:
        left, right, level = stack.pop()
        mid = (left+right)/2
        half = float(np.linalg.norm(right-left)/2)
        distance = float(scene.distance([mid])[0]) - margin
        if distance < safe_radius:
            return None
        if distance-half >= safe_radius and free_space.contains_ball(mid, safe_radius+half):
            lower = min(lower, distance-half)
            leaves += 1
            continue
        if level >= max_depth or half*2 <= min_step:
            return None  # unresolved is not safe
        stack.extend(((left, mid, level+1), (mid, right, level+1)))
    return dict(minimum_mesh_distance_lower_bound=lower, certified_leaves=leaves,
                observed_free=True, swept_sphere=True)
