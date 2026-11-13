from pathlib import Path
import copy
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recompose.io import save_mesh,fingerprint,read_json
from recompose.mesh_export import export_mesh_scene
from mesh_benchmark.incremental_mesh_registration import export_registered_mesh_incremental


def tiny_scene(tmp_path):
    background=tmp_path/'background.ply';asset=tmp_path/'asset.npz'
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype='float64');f=np.array([[0,1,2]],dtype='uint32')
    c=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype='float32')
    save_mesh(background,v,f,c);np.savez(asset,vertices=v,faces=f,colors=c)
    return dict(status='needs_review',background=dict(mesh=fingerprint(background),patches=[]),
                objects=[dict(id='8',mesh=fingerprint(asset),object_to_world=np.eye(4).tolist())]),background,asset


def test_incremental_update_keeps_all_background_bindings_and_reads_no_background(tmp_path,monkeypatch):
    scene,background,asset=tiny_scene(tmp_path);world=tmp_path/'world'
    export_mesh_scene(scene,world);old=read_json(world/'mesh_scene.gltf')
    new=tmp_path/'corrected.npz'
    with np.load(asset) as a:
        v=a['vertices'].copy();v[1,2]=.37
        np.savez(new,vertices=v,faces=a['faces'],colors=a['colors'])
    scene['objects'][0]['mesh']=fingerprint(new);scene['objects'][0]['object_to_world'][1][3]=.1
    scene['status']='complete';background.unlink()
    # An unavailable background source must not prevent object-only publication.
    target=export_registered_mesh_incremental(scene,world);actual=read_json(target)
    for key in ['nodes','meshes','buffers']:
        assert actual[key][0]==old[key][0]
    assert actual['bufferViews'][:2]==old['bufferViews'][:2]
    assert actual['accessors'][:3]==old['accessors'][:3]
    object_node=actual['nodes'][1]
    np.testing.assert_array_equal(np.array(object_node['matrix']).reshape(4,4).T,scene['objects'][0]['object_to_world'])
    assert actual['extras']['status']=='complete'
    blob=(world/actual['buffers'][1]['uri']).read_bytes()
    packed=np.frombuffer(blob,dtype=[('position','<f4',(3,)),('color','u1',(4,))],count=3)
    np.testing.assert_array_equal(packed['position'],v.astype('float32'))
    np.testing.assert_array_equal(packed['color'],np.array([[255,0,0,255],[0,255,0,255],[0,0,255,255]],dtype='uint8'))
    np.testing.assert_array_equal(np.frombuffer(blob,dtype='<u4',offset=48),[0,1,2])


def test_missing_existing_gltf_uses_full_export_once(tmp_path):
    scene,_,_=tiny_scene(tmp_path)
    assert export_registered_mesh_incremental(scene,tmp_path/'fresh').exists()


def test_background_or_object_set_changes_fail_before_any_write(tmp_path):
    scene,_,_=tiny_scene(tmp_path);world=tmp_path/'world';export_mesh_scene(scene,world)
    before=(world/'mesh_scene.gltf').read_bytes()
    changed=copy.deepcopy(scene);changed['background']['patches']=[dict(path='never-open-this')]
    with pytest.raises(ValueError,match='Background partition set'):
        export_registered_mesh_incremental(changed,world)
    changed=copy.deepcopy(scene);changed['objects'][0]['id']='9'
    with pytest.raises(ValueError,match='Object node set'):
        export_registered_mesh_incremental(changed,world)
    assert (world/'mesh_scene.gltf').read_bytes()==before
