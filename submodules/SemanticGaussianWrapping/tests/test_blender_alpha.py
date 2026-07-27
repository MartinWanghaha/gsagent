import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scene import camera_from_info
from scene.dataset_readers import readCamerasFromTransforms
from utils.image_utils import composite_background


def test_blender_reader_preserves_foreground_rgb_and_alpha(tmp_path) -> None:
    rgba = np.asarray(
        [
            [[12, 34, 56, 0], [100, 120, 140, 255]],
            [[20, 40, 60, 128], [200, 180, 160, 255]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(rgba).save(tmp_path / "frame.png")
    (tmp_path / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.7,
                "frames": [
                    {
                        "file_path": "frame.png",
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ],
            }
        ),
        encoding="utf8",
    )

    info = readCamerasFromTransforms(
        tmp_path, "transforms_train.json", white_background=True
    )[0]
    assert np.asarray(info.image)[0, 0].tolist() == [12, 34, 56]
    assert np.allclose(info.alpha, rgba[..., 3] / 255.0)

    camera = camera_from_info(
        info,
        SimpleNamespace(resolution=1, data_device="cpu", semantic_ignore_label=-1),
    )
    assert torch.allclose(camera.gt_mask[0], torch.from_numpy(info.alpha))
    assert torch.allclose(
        camera.original_image[:, 0, 0], torch.tensor([12, 34, 56]) / 255.0
    )
    assert camera.resized(1, 1).gt_mask.shape == (1, 1, 1)


def test_deferred_reader_materializes_only_target_resolution(tmp_path) -> None:
    height, width = 12, 16
    rgba = np.arange(height * width * 4, dtype=np.uint8).reshape(height, width, 4)
    Image.fromarray(rgba).save(tmp_path / "frame.png")
    (tmp_path / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.7,
                "frames": [
                    {
                        "file_path": "frame.png",
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ],
            }
        ),
        encoding="utf8",
    )

    eager = readCamerasFromTransforms(
        tmp_path,
        "transforms_train.json",
        white_background=True,
    )[0]
    deferred = readCamerasFromTransforms(
        tmp_path,
        "transforms_train.json",
        white_background=True,
        defer_camera_loading=True,
    )[0]
    assert deferred.image is None
    assert deferred.alpha is None
    assert deferred.semantic_ids is None
    assert deferred.payload_loader is not None

    args = SimpleNamespace(
        resolution=4,
        data_device="cpu",
        semantic_ignore_label=-1,
    )
    eager_camera = camera_from_info(eager, args)
    deferred_camera = camera_from_info(deferred, args)

    assert deferred_camera.original_image.shape == (3, height // 4, width // 4)
    assert torch.equal(deferred_camera.semantic_ids, eager_camera.semantic_ids)
    assert torch.allclose(
        deferred_camera.original_image,
        eager_camera.original_image,
        atol=1e-7,
        rtol=0,
    )
    assert torch.allclose(deferred_camera.gt_mask, eager_camera.gt_mask)
    assert deferred_camera.Fx == pytest.approx(eager_camera.Fx)
    assert deferred_camera.Fy == pytest.approx(eager_camera.Fy)
    assert deferred_camera.Cx == pytest.approx(eager_camera.Cx)
    assert deferred_camera.Cy == pytest.approx(eager_camera.Cy)


def test_random_background_target_uses_preserved_alpha() -> None:
    foreground = torch.tensor([[[0.1]], [[0.2]], [[0.3]]])
    transparent = torch.zeros(1, 1, 1)
    first = composite_background(foreground, transparent, torch.tensor([0.7, 0.4, 0.2]))
    second = composite_background(foreground, transparent, torch.tensor([0.2, 0.6, 0.9]))
    assert torch.allclose(first[:, 0, 0], torch.tensor([0.7, 0.4, 0.2]))
    assert torch.allclose(second[:, 0, 0], torch.tensor([0.2, 0.6, 0.9]))
    assert not torch.allclose(first, second)
