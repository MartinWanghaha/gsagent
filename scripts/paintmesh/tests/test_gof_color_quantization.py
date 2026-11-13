from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair_gof_mesh_colors import quantize_rgb, restore_overflow


def test_hdr_channels_saturate_and_in_range_stored_bytes_are_exact():
    values = np.array([[1.05, 1.03, .94], [-.1, .4, .7]], dtype='float32')
    original = (values * 255).astype('uint8')
    corrected = restore_overflow(values, original)
    np.testing.assert_array_equal(corrected, [[255, 255, 239], [0, 102, 178]])
    valid = (values >= 0) & (values <= 1)
    np.testing.assert_array_equal(corrected[valid], original[valid])
    np.testing.assert_array_equal(corrected, quantize_rgb(values))


def test_nonfinite_rgb_cannot_be_exported():
    with pytest.raises(ValueError, match='Nonfinite'):
        restore_overflow(np.array([[np.nan, 0, 1]]), np.zeros((1, 3), dtype='uint8'))
