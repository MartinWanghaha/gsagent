"""Large GOF mesh IDs must survive the OpenGL adapter without float rounding."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from render_native_colored_meshes import exact_triangle_indices


def test_large_adjacent_vertex_ids_stay_distinct():
    faces=np.array([[2**24,2**24+1,2**24+3]],dtype='int64')
    assert not np.array_equal(faces,faces.astype('float32').astype('int64'))
    indices=exact_triangle_indices(faces,2**24+4)
    assert indices.dtype==np.uint32 and indices.flags.c_contiguous
    np.testing.assert_array_equal(indices,faces)


@pytest.mark.parametrize('faces',[np.array([[-1,0,1]]),np.array([[0,1,3]]),np.array([[0.,1.,2.]])])
def test_invalid_mesh_ids_are_rejected(faces):
    with pytest.raises(ValueError):exact_triangle_indices(faces,3)
