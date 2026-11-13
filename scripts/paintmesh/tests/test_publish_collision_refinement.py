import copy
import importlib
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import publish_collision_refinement as pub
from recompose.original_gaussians import movement_transform


def test_registration_correction_keeps_original_gaussians_in_world():
    before=np.eye(4);before[:3,:3]*=2;before[:3,3]=[1,2,3]
    target=before.copy();target[:3,3]+=[.04,-.01,.03]
    node=dict(id='8',object_to_world=before.tolist(),original_pose=before.tolist(),gaussian_coordinates='original_world')
    actual=pub.registration_node(node,dict(object_to_world=target.tolist()),dict(path='report'),True)
    np.testing.assert_array_equal(movement_transform(actual),np.eye(4))
    np.testing.assert_array_equal(actual['object_to_world'],target)
    np.testing.assert_array_equal(node['object_to_world'],before)
    assert actual['registration_refinement']['previous_mesh_registration']==before.tolist()


def test_registration_refuses_scale_changes_or_previously_moved_gaussians():
    node=dict(id='8',object_to_world=np.eye(4).tolist(),original_pose=np.eye(4).tolist(),gaussian_coordinates='original_world')
    changed=np.eye(4);changed[:3,:3]*=1.01
    with pytest.raises(ValueError,match='preserve'):
        pub.registration_node(node,dict(object_to_world=changed),{},True)
    moved=copy.deepcopy(node);moved['object_to_world'][0][3]=1
    with pytest.raises(ValueError,match='already moved'):
        pub.registration_node(moved,dict(object_to_world=np.eye(4)),{},True)


def test_changed_mesh_replaces_geometry_and_texture_reference_without_moving_gs(tmp_path):
    old=tmp_path/'old.npz';new=tmp_path/'contact_asset.npz';glb=tmp_path/'contact_asset.glb'
    np.savez(old,vertices=np.zeros((3,3)),faces=np.array([[0,1,2]]))
    np.savez(new,vertices=np.eye(3),faces=np.array([[0,1,2]]))
    glb.write_bytes(b'corrected textured geometry')
    node=dict(id='8',mesh=pub.fingerprint(old),textured_mesh=dict(path='old.glb'),
        object_to_world=np.eye(4).tolist(),original_pose=np.eye(4).tolist(),gaussian_coordinates='original_world')
    actual=pub.registration_node(node,dict(object_to_world=np.eye(4),mesh_path=new,
        corrected_textured_mesh=pub.fingerprint(glb)),{},True)
    assert actual['mesh']==pub.fingerprint(new)
    assert actual['textured_mesh']==pub.fingerprint(glb)
    assert actual['registration_refinement']['generated_mesh_geometry_changed']
    assert actual['registration_refinement']['previous_generated_mesh']==pub.fingerprint(old)
    np.testing.assert_array_equal(movement_transform(actual),np.eye(4))


def test_nearest_depth_composition_preserves_unoccluded_background_bytes():
    bd=np.array([[1.,0.,3.,np.inf]],dtype='float32')
    od=np.array([[2.,4.,2.,0.]],dtype='float32')
    bg=np.arange(12,dtype='uint8').reshape(1,4,3);obj=np.full_like(bg,140)
    depth,rgb=pub.compose_native_arrays(bd,bg,od,obj)
    np.testing.assert_array_equal(depth,np.array([[1.,4.,2.,np.inf]]))
    np.testing.assert_array_equal(rgb[0,[0,3]],bg[0,[0,3]])
    np.testing.assert_array_equal(rgb[0,[1,2]],obj[0,[1,2]])
    np.testing.assert_array_equal(bg,np.arange(12,dtype='uint8').reshape(1,4,3))


def test_unresolved_report_cannot_become_accepted():
    assert pub.validation_object(dict(complete=True,objects={'8':dict(accepted=True)}),'8',{})[1]
    assert not pub.validation_object(dict(complete=True,objects={'8':dict(accepted=True,unresolved=True)}),'8',{})[1]
    assert not pub.validation_object(dict(complete=False,objects={'8':dict(accepted=True)}),'8',{})[1]
    assert not pub.validation_object(dict(complete=True,objects={}), '8', dict(complete=True))[1]


def test_single_state_binding_does_not_mutate_upstream_namespace():
    namespace=dict(STATES=['background','recomposition'])
    exec('def run():\n return tuple(STATES)',namespace)
    fn=namespace['run']
    assert pub._single_state(fn,'recomposition')()==('recomposition',)
    assert fn()==('background','recomposition')


def test_background_retag_preserves_all_numeric_scores_and_image_bytes(tmp_path,monkeypatch):
    native=importlib.import_module('evaluate_native_mesh_rgb')
    root=tmp_path/'work';root.mkdir();scene='bag';common=root/'common'/scene
    graph=tmp_path/'scene.json';pub.write(graph,dict(old=True))
    old_selection=pub.identity(graph)
    request=dict(selection=old_selection,split='same',masks=['same'])
    old_id=pub.digest(request)
    pub.write(common/'complete.json',dict(complete=True,request_id=old_id,request=request))
    geo_request=dict(inputs=[])
    before_metrics={}
    for i,m in enumerate(pub.METHODS):
        req=dict(common=old_id,specs=[m],seed=42,pool_pixels=100,protocol_version=2)
        for state in pub.STATES:
            dest=root/'cache'/m/scene/state
            pub.write(dest/'complete.json',dict(complete=True,request=copy.deepcopy(req),geometry={}))
        geometry=dict(CD_L1=.2+i/100,F1=.8,own_surface=dict(F1=.9),support_id='old-geometric-id')
        pub.write(root/'cache'/m/scene/'background/geometry_metrics.json',geometry)
        geo_request['inputs'].append(dict(method=m,request=copy.deepcopy(req)))
        value=dict(complete=True,mean=dict(PSNR=25.+i,SSIM=.9,LPIPS=.1),per_view=[dict(name='A',PSNR=25.+i)],
            geometry=geometry,request=dict(renderer={},support_id='old-native-id',geometry_support_id='old-geometric-id'))
        pub.write(root/'background'/m/scene/'metrics.json',value)
        before_metrics[m]=copy.deepcopy(value)
    pub.write(common/'background_support/complete.json',dict(complete=True,request=geo_request,request_id='old-geometric-id',coverage=[dict(pixels=12)]))
    pub.write(common/'background_native_support/complete.json',dict(complete=True,request=dict(sources={},masks=['same']),request_id='old-native-id',coverage=[dict(pixels=12)]))
    frozen=common/'held_out/background.png';frozen.parent.mkdir();frozen.write_bytes(b'unchanged-background-image')
    def renderer(root,method,scene,state):
        return dict(request=dict(method=method,geometry=pub.read(root/'cache'/method/scene/state/'complete.json')['request']))
    monkeypatch.setattr(native,'rendering',renderer)
    pub.write(graph,dict(new_pose=True))
    after=pub.retag_background_metadata(root,scene,graph)
    assert after['request_id']!=old_id
    assert after['request_id']==pub.digest(after['request'])
    assert frozen.read_bytes()==b'unchanged-background-image'
    for m in pub.METHODS:
        value=pub.read(root/'background'/m/scene/'metrics.json')
        assert value['mean']==before_metrics[m]['mean']
        assert value['per_view']==before_metrics[m]['per_view']
        assert value['geometry']['CD_L1']==before_metrics[m]['geometry']['CD_L1']
        assert value['geometry']['F1']==before_metrics[m]['geometry']['F1']
        assert value['request']['geometry_support_id']==value['geometry']['support_id']
    geo=pub.read(common/'background_support/complete.json')
    assert geo['request_id']==pub.digest(geo['request'])
    assert geo['coverage']==[dict(pixels=12)]


def test_refresh_evaluation_scores_every_method_unless_strict_reuse_verifies_it(tmp_path,monkeypatch):
    native=importlib.import_module('evaluate_native_mesh_rgb')
    report=importlib.import_module('mesh_benchmark.report')
    root=tmp_path;scene='bag'
    pub.write(root/'protocol.json',dict(config={}))
    marker=root/'cache/paintmesh'/scene/'recomposition/complete.json'
    pub.write(marker,dict(complete=True,geometry_pending_rescore=True))
    monkeypatch.setattr(pub,'_single_state',lambda fn,state:lambda *args,**kwargs:None)
    monkeypatch.setattr(report,'update',lambda root:None)
    checked=[];scored=[]
    def reuse(root,method,scene,state):
        assert state=='recomposition'
        checked.append(method)
        return method in ['2dgs','come']
    monkeypatch.setattr(native,'reuse_unchanged_score',reuse)
    monkeypatch.setattr(native,'score',lambda root,method,scene,state:scored.append(method))
    result=pub.refresh_evaluation(scene,root,figures=False,verify=False)
    assert checked==pub.METHODS
    assert scored==[m for m in pub.METHODS if m not in ['2dgs','come']]
    assert result['composed_methods_rescored']==scored
    assert result['composed_methods_reused_after_exact_checks']==['2dgs','come']
    assert 'geometry_pending_rescore' not in pub.read(marker)
