"""Preserve original detail and extend through a crop boundary without a seam."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_extension_keeps_original_and_only_removes_covered_new_faces(tmp_path):
    from recompose.io import save_mesh,fingerprint
    from mesh_benchmark.core import identity,write,read
    from mesh_benchmark.extend_fine import extend
    old=tmp_path/'original.ply'
    save_mesh(old,np.array([[-1,-1,4],[1,-1,4],[-1,1,4],[1,1,4]],dtype='float32'),
        np.array([[0,1,2],[1,3,2]]),np.array([[.1,.5,.9]]*4))
    original_bytes=old.read_bytes()
    positions=np.array([[.2,.05,4],[.4,.05,4],[.3,.15,4],
        [2,0,4],[2.2,0,4],[2.1,.2,4],
        [1.004,.1,4],[1.005,.2,4],[1.006,.15,4]],dtype='float32')
    packed=np.empty(9,dtype=[('position','<f4',(3,)),('color','u1',(4,))]);packed['position']=positions;packed['color']=[25,120,240,255]
    vp=tmp_path/'vertices.npy';fp=tmp_path/'faces.npy';part=tmp_path/'part.mesh.json'
    faces=np.arange(9,dtype='<u4').reshape(-1,3);np.save(vp,packed);np.save(fp,faces)
    write(part,dict(format='paintmesh-native-triangles-v1',vertices=fingerprint(vp),indices=fingerprint(fp),
        vertex_identity=identity(vp),index_identity=identity(fp),face_count=3,vertex_count=9,
        bounds=[positions.min(0).tolist(),positions.max(0).tolist()],vertex_stride=16))
    source_vertex_bytes=vp.read_bytes();source_index_bytes=fp.read_bytes()
    result=extend(old,dict(request=dict(voxel=.002,sdf_trunc=.008),mesh_specs=[dict(path=str(part))]),tmp_path/'extension')
    assert result['original_faces_preserved']==2 and result['added_faces']==2
    assert result['redundant_new_faces_removed']==1 and result['faces']==4
    assert result['mesh_specs'][0]['path']==str(old)
    updated=read(result['mesh_specs'][1]['path'])
    np.testing.assert_array_equal(np.load(updated['indices']['path']),faces[1:])
    assert updated['vertices']==fingerprint(vp)
    assert old.read_bytes()==original_bytes and vp.read_bytes()==source_vertex_bytes and fp.read_bytes()==source_index_bytes
    assert result['original_mesh_unchanged'] and result['no_reference_data_used'] and not result['geometry_simplified']


def test_exact_distance_bound_preserves_coverage_decisions_in_empty_aabb_space(tmp_path):
    from recompose.io import save_mesh
    from mesh_benchmark.core import MeshScene
    from mesh_benchmark.extend_fine import covered,prepare_distance_bound
    path=tmp_path/'separated.ply'
    vertices=np.array([[-2,-2,2],[-1.9,-2,2],[-2,-1.9,2],[2,2,4],[2.1,2,4],[2,2.1,4]],dtype='float32')
    save_mesh(path,vertices,np.array([[0,1,2],[3,4,5]]),np.ones((6,3)))
    mesh=MeshScene([dict(path=str(path))])
    q=np.array([[-1.97,-1.97,2.004],[2.03,2.03,4.004],[0,0,3],[-2,-2,2],[1,1,3]],dtype='float32')
    expected=covered(mesh,q);prepare_distance_bound(mesh,path);actual=covered(mesh,q)
    np.testing.assert_array_equal(actual,expected)
    assert actual.tolist()==[True,True,False,True,False]
