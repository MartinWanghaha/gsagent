from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from source.pgsr_geometry import (  # noqa: E402
    build_neighbor_map,
    camera_intrinsics,
    camera_rays,
    depth_to_normal,
    image_gray,
    patch_offsets,
    patch_warp,
    points_from_depth,
    sample_depth,
    smallest_axis_normal,
)


class FakeCamera:
    def __init__(self, name="reference", center=(0.0, 0.0, 0.0), size=5):
        self.image_name = name
        self.image_width = size
        self.image_height = size
        self.FoVx = math.pi / 2
        self.FoVy = math.pi / 2
        self.world_view_transform = torch.eye(4)
        self.world_view_transform[3, :3] = -torch.tensor(center)
        self.full_proj_transform = torch.eye(4)
        self.camera_center = torch.tensor(center, dtype=torch.float32)
        self.original_image = torch.zeros(3, size, size)


def test_intrinsics_rays_and_grayscale_are_device_aware():
    camera = FakeCamera(size=4)
    camera.original_image[0].fill_(1.0)

    intrinsics = camera_intrinsics(camera)
    rays = camera_rays(camera)
    gray = image_gray(camera)

    assert intrinsics.device.type == "cpu"
    assert torch.allclose(
        intrinsics,
        torch.tensor(((2.0, 0.0, 2.0), (0.0, 2.0, 2.0), (0.0, 0.0, 1.0))),
    )
    assert rays.shape == (4, 4, 3)
    assert torch.allclose(rays[2, 2], torch.tensor((0.0, 0.0, 1.0)))
    assert gray.shape == (1, 4, 4)
    assert torch.allclose(gray, torch.full_like(gray, 0.299))


def test_depth_unprojection_sampling_and_flat_normal():
    camera = FakeCamera(size=5)
    depth = torch.full((1, 5, 5), 2.0)

    points_world = points_from_depth(camera, depth)
    sampled, valid = sample_depth(
        camera,
        depth,
        torch.tensor(((0.0, 0.0, 2.0), (0.0, 0.0, -1.0))),
    )
    normals = depth_to_normal(camera, depth)

    assert points_world.shape == (25, 3)
    # PGSR/EDGS use (W/2,H/2) as principal point, so the middle integer pixel
    # of an odd-sized image lies half a pixel to its upper-left.
    assert torch.allclose(points_world[12], torch.tensor((-0.4, -0.4, 2.0)))
    assert sampled[0] == pytest.approx(2.0)
    assert valid.tolist() == [True, False]
    assert normals.shape == (3, 5, 5)
    assert torch.allclose(normals[:, 2, 2], torch.tensor((0.0, 0.0, -1.0)))
    assert torch.count_nonzero(normals[:, 0]) == 0


def test_smallest_axis_normal_is_oriented_toward_camera():
    gaussians = SimpleNamespace(
        get_scaling=torch.tensor(((1.0, 0.1, 2.0),)),
        get_rotation=torch.tensor(((1.0, 0.0, 0.0, 0.0),)),
        get_xyz=torch.zeros(1, 3),
    )
    camera = FakeCamera(center=(0.0, -2.0, 0.0))

    normal = smallest_axis_normal(gaussians, camera)

    assert torch.allclose(normal, torch.tensor(((0.0, -1.0, 0.0),)))


def test_patch_helpers_preserve_points_under_identity():
    pixels = torch.tensor([[[10.0, 20.0]]]) + patch_offsets(1, "cpu")
    warped = patch_warp(torch.eye(3), pixels)

    assert pixels.shape == (1, 9, 2)
    assert torch.allclose(warped, pixels)


def test_patch_warp_broadcasts_single_homography_across_batches():
    pixels = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])

    warped = patch_warp(torch.eye(3), pixels)

    assert warped.shape == pixels.shape
    assert torch.allclose(warped, pixels)


def test_neighbor_map_uses_post_split_camera_objects_and_config():
    cameras = [
        FakeCamera("a", (0.0, 0.0, 0.0)),
        FakeCamera("b", (0.2, 0.0, 0.0)),
        FakeCamera("c", (0.5, 0.0, 0.0)),
    ]
    config = {
        "neighbor_num": 1,
        "max_angle_deg": 30.0,
        "min_distance": 0.1,
        "max_distance": 0.4,
    }

    neighbors = build_neighbor_map(cameras, config)

    assert neighbors["a"] == [cameras[1]]
    assert neighbors["b"] == [cameras[0]]
    assert neighbors["c"] == [cameras[1]]
