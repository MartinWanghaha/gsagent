"""Round-trip source triangles and native appearance through standalone GLB."""
import json
from pathlib import Path
import struct
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.asset_gltf import export_textured_asset


def document(path):
    data=Path(path).read_bytes();length,kind=struct.unpack_from('<II',data,12)
    assert kind==0x4E4F534A
    return json.loads(data[20:20+length])


def nodes(path):
    import trimesh
    scene=trimesh.load(path,force='scene',process=False)
    return {scene.geometry[scene.graph[n][1]].metadata['source_material_id']:
        (scene.graph[n][0],scene.geometry[scene.graph[n][1]]) for n in scene.graph.nodes_geometry}


def test_native_textures_seams_factors_world_transform_and_no_double_shape(tmp_path):
    vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1.]],dtype=float)
    faces=np.array([[0,1,2],[0,2,3],[0,1,4],[1,2,4]])
    materials=np.array([0,0,1,-1],dtype='int32')
    uv=np.array([[[0,0],[1,0],[1,1]],[[.4,.2],[.9,.8],[.1,.9]],
        [[.2,.3],[.8,.3],[.5,.9]],[[0,0],[0,0],[0,0]]],dtype=float)
    tex0=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,0]]],dtype='uint8')
    tex1=np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]],dtype='uint8')
    colors=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1],[.2,.4,.6]])
    source=tmp_path/'asset.npz'
    np.savez(source,vertices=vertices,faces=faces,colors=colors,
        triangle_uv=uv,triangle_material=materials,texture_0=tex0,texture_1=tex1,
        texture_factor_0=np.array([.123456,.789123,.987654]),texture_factor_1=[.5,.6,.7,.4],
        baked_shape_transform=np.diag([9,8,7]))
    transform=np.eye(4);transform[:3,:3]=[[0,-2,0],[2,0,0],[0,0,2]];transform[:3,3]=[2,-3,5]
    target=tmp_path/'new.glb';meta=export_textured_asset(source,target,transform)
    assert meta['face_count']==4 and not meta['shape_transform_reapplied']
    decoded=nodes(target)
    for mid,(T,mesh) in decoded.items():
        selected=faces[materials==mid]
        original=vertices[selected]
        actual=mesh.vertices[mesh.faces]
        # glTF stores float32; both original and transformed face ordering are
        # exact within each material, including two UVs for the same old vertex.
        np.testing.assert_allclose(actual,original,rtol=0,atol=1e-7)
        np.testing.assert_allclose(actual@T[:3,:3].T+T[:3,3],original@transform[:3,:3].T+transform[:3,3],atol=1e-6)
        if mid>=0:
            np.testing.assert_allclose(mesh.visual.uv[mesh.faces],uv[materials==mid],atol=1e-7)
            np.testing.assert_array_equal(np.asarray(mesh.visual.material.baseColorTexture),[tex0,tex1][mid])
        else:
            decoded_colors=mesh.visual.vertex_attributes['color']
            np.testing.assert_array_equal(decoded_colors[mesh.faces][:,:,:3],np.rint(colors[selected]*255).astype('uint8'))
    doc=document(target)
    factors={m['name']:m['pbrMetallicRoughness']['baseColorFactor'] for m in doc['materials']}
    assert factors['paintmesh_material_0']==[.123456,.789123,.987654,1.]
    assert factors['paintmesh_material_1']==[.5,.6,.7,.4]
    for m in doc['meshes']:
        if m['name']!='paintmesh_material_-1':
            assert 'COLOR_0' not in m['primitives'][0]['attributes']
    assert 'KHR_materials_unlit' in doc['extensionsUsed']


def test_vertex_only_asset_keeps_corrected_geometry_without_uv(tmp_path):
    source=tmp_path/'changed.npz';destination=tmp_path/'changed.glb'
    v=np.array([[0,0,.173],[2,0,.1],[0,2,.2]])
    np.savez(source,vertices=v,faces=[[0,1,2]],colors=np.full((3,3),.5),baked_shape_transform=4*np.eye(3))
    export_textured_asset(source,destination)
    T,mesh=nodes(destination)[-1]
    np.testing.assert_allclose(mesh.vertices[mesh.faces],v[[[0,1,2]]],atol=1e-7)
    np.testing.assert_array_equal(T,np.eye(4))
    assert not hasattr(mesh.visual,'uv') or mesh.visual.uv is None


def test_missing_bitmap_fails_without_writing_output(tmp_path):
    source=tmp_path/'missing.npz';destination=tmp_path/'missing.glb'
    np.savez(source,vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],colors=np.ones((3,3)),
        triangle_uv=np.zeros((1,3,2)),triangle_material=[0])
    with pytest.raises(ValueError,match='Missing native bitmap'):
        export_textured_asset(source,destination)
    assert not destination.exists()
