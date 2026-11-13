from pathlib import Path
import sys
import numpy as np
from plyfile import PlyData,PlyElement

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repair_opacity_meshes import filter_faces
from mesh_benchmark.color_export import read_mesh


def test_face_selection_preserves_coordinates_colors_and_retained_face_attributes(tmp_path):
    v=np.array([(0.,0.,0.,1,2,3),(1.,0,0,4,5,6),(0,1.,0,7,8,9),(0,0,1.,10,11,12)],
        dtype=[('x','f8'),('y','f8'),('z','f8'),('red','u1'),('green','u1'),('blue','u1')])
    f=np.array([([0,1,2],2),([0,1,3],7)],dtype=[('vertex_indices','i4',(3,)),('quality','u1')])
    source=tmp_path/'native.ply';target=tmp_path/'validated.ply'
    PlyData([PlyElement.describe(v,'vertex'),PlyElement.describe(f,'face')],text=False).write(source)
    filter_faces(source,target,np.array([False,True]))
    got=read_mesh(target)
    np.testing.assert_array_equal(got['vertex'].data,v)
    np.testing.assert_array_equal(got['face'].data['vertex_indices'],f['vertex_indices'][1:])
    np.testing.assert_array_equal(got['face'].data['quality'],[7])
