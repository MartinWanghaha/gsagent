"""Camera/mask rounding differences must preserve integer instance IDs."""

from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch


INPAINT = Path(__file__).resolve().parents[3] / "submodules/Inpaint360GS"
sys.path.insert(0, str(INPAINT))
from seg.mask_associate import GaussianMaskAssociator


def associator(tmp_path, mask):
    cv2.imwrite(str(tmp_path / "view.png"), mask)
    obj = GaussianMaskAssociator.__new__(GaussianMaskAssociator)
    torch.nn.Module.__init__(obj)
    obj.device = torch.device("cpu")
    obj.raw_mask_folder = str(tmp_path)
    obj._reported_mask_resizes = set()
    return obj


def test_aligns_one_pixel_difference_without_changing_saved_id_grid(tmp_path):
    mask = np.array([[0, 1, 1, 300, 300], [0, 1, 1, 300, 300]], dtype=np.uint16)
    obj = associator(tmp_path, mask)
    cam = SimpleNamespace(image_name="view", image_height=2, image_width=4)
    binary = obj.get_binary_mask(cam).numpy()
    assert binary.shape == (300, 2, 4)
    assert np.array_equal(binary[299], cv2.resize(mask, (4, 2), interpolation=cv2.INTER_NEAREST) == 300)
    original = obj._read_raw_mask(cam)
    assert np.array_equal(original, mask)
    remapped = obj.remap_mask(torch.arange(300), original)
    assert np.array_equal(remapped, mask)


def test_tiny_last_instance_still_has_a_channel_after_resizing(tmp_path):
    mask = np.zeros((2, 5), dtype=np.uint16)
    mask[:, -1] = 301
    obj = associator(tmp_path, mask)
    cam = SimpleNamespace(image_name="view", image_height=2, image_width=4)
    binary = obj.get_binary_mask(cam)
    assert binary.shape == (301, 2, 4)
    assert not binary.any()
    remapped = obj.remap_mask(torch.arange(301), obj._read_raw_mask(cam))
    assert remapped.max() == 301


def test_rejects_masks_from_another_resolution(tmp_path):
    obj = associator(tmp_path, np.ones((8, 12), dtype=np.uint16))
    cam = SimpleNamespace(image_name="view", image_height=4, image_width=6)
    with pytest.raises(ValueError, match="expected"):
        obj.get_binary_mask(cam)
