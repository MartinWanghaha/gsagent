from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repair_come_mesh_colors import unpremultiply


def test_a_constant_material_is_invariant_to_different_compositing_weights():
    material=np.array([[.1,.5,.9],[.1,.5,.9],[.1,.5,.9]],dtype='float32')
    weights=np.array([.01,.53,.99],dtype='float32')
    rgb=material*weights[:,None]
    original=np.rint(rgb*255).astype('uint8')
    fixed=unpremultiply(rgb,weights,original)
    np.testing.assert_array_equal(fixed,np.rint(material*255).astype('uint8'))


def test_unknown_color_keeps_original_and_dark_material_stays_dark():
    rgb=np.array([[0,0,0],[.005,.01,.02],[.8,.9,1.1]],dtype='float32')
    weight=np.array([0,.5,.9],dtype='float32')
    original=np.array([[18,26,99],[1,3,5],[204,230,255]],dtype='uint8')
    result=unpremultiply(rgb,weight,original)
    np.testing.assert_array_equal(result[0],original[0])
    np.testing.assert_array_equal(result[1],[3,5,10])
    assert result[2,2]==255
    with pytest.raises(ValueError,match='Nonfinite'):
        unpremultiply(rgb,np.array([0,np.nan,1]),original)
