"""Delivery boundaries: independent mesh nodes and honest candidate status."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recompose.io import save_mesh,read_json,write_json,fingerprint
from recompose.geometry import sim3,transform
from recompose.mesh_export import export_mesh_scene
from recompose.scene import publish,move_object


def test_full_gltf_roundtrip_keeps_background_and_transformed_object(tmp_path):
    import trimesh
    v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);f=np.array([[0,1,2]])
    bg=tmp_path/'bg.ply';asset=tmp_path/'asset.npz'
    save_mesh(bg,v,f,np.tile([.2,.3,.4],(3,1)))
    np.savez(asset,vertices=v,faces=f,colors=np.tile([.8,.7,.6],(3,1)))
    T=sim3(1.5,translation=[2,3,4])
    scene=dict(status='complete',coordinate_convention='COLMAP world',background=dict(mesh=fingerprint(bg)),
               objects=[dict(id='3',mesh=fingerprint(asset),object_to_world=T)])
    path=export_mesh_scene(scene,tmp_path/'world')
    loaded=trimesh.load(path,force='scene',process=False)
    assert {'background','object_3'}<=set(loaded.graph.nodes_geometry)
    matrix,geometry=loaded.graph['object_3']
    np.testing.assert_allclose(matrix,T)
    actual=transform(loaded.geometry[geometry].vertices,matrix)
    np.testing.assert_allclose(actual,transform(v,T))
    doc=read_json(path)
    assert doc['extras']['status']=='complete' and len(doc['meshes'])==2
    for buffer in doc['buffers']:assert (path.parent/buffer['uri']).stat().st_size==buffer['byteLength']


def test_candidate_is_not_published_or_movable(tmp_path,monkeypatch):
    root=tmp_path/'object';root.mkdir();(root/'evidence').mkdir()
    write_json(root/'evidence/observations.json',{})
    for name in ['asset.npz','gaussians.ply','provenance.json','indices.npy']:(root/name).write_bytes(b'test input')
    write_json(root/'pose.json',dict(status='needs_review',reasons=['background support is inconsistent'],object_to_world=np.eye(4)))
    bg=tmp_path/'bg';bg.write_bytes(b'background')
    inputs=dict(target_ids=[1],background_mesh=str(bg),background_gaussians=str(bg))
    objects=[dict(id=1,asset=str(root/'asset.npz'),gaussians=str(root/'gaussians.ply'),gaussian_provenance=str(root/'provenance.json'),gaussian_indices=str(root/'indices.npy'),pose=str(root/'pose.json'),evidence=str(root/'evidence'))]
    monkeypatch.setattr('recompose.scene.validate_scene_placement',lambda scene:dict(accepted=False,objects=[],pairs=[]))
    monkeypatch.setattr('recompose.scene.export_scene',lambda *args:None)
    with pytest.raises(ValueError,match='requires review'):publish(inputs,objects,tmp_path/'published')
    assert not (tmp_path/'published/scene.json').exists()
    publish(inputs,objects,tmp_path/'candidate',review=True)
    scene=read_json(tmp_path/'candidate/scene.json')
    assert scene['complete'] is False and scene['status']=='needs_review'
    with pytest.raises(ValueError,match='Resolve candidate'):move_object(tmp_path/'candidate/scene.json',1,np.eye(4),tmp_path/'moved.json')
    with pytest.raises(ValueError,match='all requested'):publish(inputs,[],tmp_path/'partial',review=True)


def test_isolation_does_not_reinsert_retained_instance_sharing_label():
    from recompose.isolation import classify_removed_surface
    # A repaired removal deletes the box, but keeps a nearby same-ID bottle.
    removed=np.array([[0.,0,0],[0,.1,0],[0,-.1,0]])
    retained=removed+[1,0,0]
    queries=np.array([[.01,0,0],[1.01,0,0],[4,0,0]])
    np.testing.assert_array_equal(classify_removed_surface(queries,removed,retained,.2),[True,False,False])


def test_multi_object_scene_finishes_later_objects_after_rejection(tmp_path,monkeypatch):
    from recompose.pipeline import run_scene
    import recompose.pipeline as pipeline
    config=tmp_path/'config.json'
    write_json(config,dict(inputs_root='inputs',output_root='output',evidence={},generator={},pose={}))
    inputs=dict(run=str(tmp_path/'upstream'),original_mesh=str(tmp_path/'upstream/mesh.ply'),target_ids=[1,2],
                background_mesh='bg',original_gaussians='gs',gaussian_labels='labels',gaussian_confidence='conf',manifest='manifest')
    monkeypatch.setattr(pipeline,'resolve_scene',lambda *args:inputs)
    monkeypatch.setattr(pipeline,'fingerprint',lambda path:dict(path=str(path),sha256='test'))
    monkeypatch.setattr(pipeline,'verify_observations',lambda path:None)
    monkeypatch.setattr(pipeline,'sources',lambda inputs:{})
    monkeypatch.setattr(pipeline,'export_initial',lambda *args:None)
    visited=[]
    def stage(root,name,inputs,outputs,fn):
        if name=='original_gaussians':
            write_json(outputs[0],dict(object_counts={'1':1,'2':1},objects={}));return
        target=int(Path(root).name);visited.append((target,name))
        if name=='align':write_json(outputs[0],dict(status='needs_review' if target==1 else 'accepted',reasons=['support'] if target==1 else []))
    monkeypatch.setattr(pipeline,'stage',stage)
    result=run_scene(tmp_path,config,'example',until='align')
    assert result['state']=='needs_review'
    assert (2,'generate') in visited and (2,'align') in visited
    assert [p['object'] for p in result['pending']]==[1]


def test_isolation_roundtrip_uses_native_ply_and_actual_removal(tmp_path,monkeypatch):
    import open3d as o3d
    from PIL import Image
    from recompose.isolation import isolate_selection
    from recompose.geometry import surface_samples,RayMesh,camera_rays
    m=o3d.geometry.TriangleMesh.create_box(.4,.4,.4)
    v=np.asarray(m.vertices)-[.2,.2,.2];f=np.asarray(m.triangles)
    left=v+[-.5,0,2];right=v+[.5,0,2];vertices=np.r_[left,right];faces=np.r_[f,f+len(v)]
    a=surface_samples(left,f,1000,1)[0];b=surface_samples(right,f,1000,2)[0]
    original=tmp_path/'original.ply';remaining=tmp_path/'remaining.ply'
    save_mesh(original,np.r_[a,b],np.array([[0,1,2]]));save_mesh(remaining,b,np.array([[0,1,2]]))
    labels=tmp_path/'labels.npy';np.save(labels,np.ones(2000,dtype='uint16'))
    removal=tmp_path/'removal.json';write_json(removal,dict(outputs=dict(removed_3dgs=dict(path=str(remaining)))))
    manifest=tmp_path/'manifest.json';write_json(manifest,dict(inputs=dict(removal_manifest=dict(path=str(removal)))))
    masks=tmp_path/'masks';masks.mkdir();K=np.array([[60.,0,32],[0,60,32],[0,0,1.]])
    yy,xx=np.mgrid[:64,:64];rays=camera_rays(K,np.eye(4),np.c_[xx.ravel(),yy.ravel()]);depth,_,_=RayMesh(vertices,faces).cast(rays)
    Image.fromarray(np.isfinite(depth).reshape(64,64).astype('uint16')).save(masks/'train.png')
    root=tmp_path/'scene';proxy=root/'objects/1/evidence/original_proxy.ply';save_mesh(proxy,vertices,faces)
    monkeypatch.setattr('recompose.isolation.original_cameras',lambda *args:[dict(name='train.png',width=64,height=64,K=K,w2c=np.eye(4))])
    inputs=dict(manifest=str(manifest),original_gaussians=str(original),gaussian_labels=str(labels),mask_root=str(masks),source='source',target_ids=[1])
    out=isolate_selection(tmp_path,inputs,root/'target_isolation',64)
    result=np.array(Image.open(Path(out['mask_root'])/'train.png'))
    assert result[32,17]==1 and result[32,47]!=1
    isolated=np.load(out['gaussian_labels']);assert np.all(isolated[:1000]==1) and np.all(isolated[1000:]!=1)
    report=read_json(root/'target_isolation/isolation.json')
    assert report['targets']['1']['removed_centers']==1000
    # Reuse must preserve the exact isolated evidence.
    assert isolate_selection(tmp_path,inputs,root/'target_isolation',64)==out


def test_provenance_cache_detects_same_size_rewrite_and_replacement(tmp_path):
    import os
    from recompose.provenance import Fingerprinter
    hash_file=Fingerprinter();path=tmp_path/'source';path.write_bytes(b'first')
    original=hash_file(path);stamp=path.stat().st_mtime_ns
    path.write_bytes(b'other');os.utime(path,ns=(stamp,stamp))
    changed=hash_file(path);assert changed['sha256']!=original['sha256']
    # Atomic replacement can preserve mtime and size but changes inode/ctime.
    new=tmp_path/'replacement';new.write_bytes(b'third');os.utime(new,ns=(stamp,stamp));new.replace(path)
    replaced=hash_file(path);assert replaced['sha256']!=changed['sha256']


def test_movement_provenance_rejects_modified_contact_evidence(tmp_path):
    from recompose.scene import verify_scene_assets
    file=tmp_path/'geometry';file.write_bytes(b'original')
    scene=dict(background=dict(mesh=fingerprint(file),gaussians=fingerprint(file)),
               objects=[dict(evidence_references=[fingerprint(file)])])
    verify_scene_assets(scene)
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='asset changed'):verify_scene_assets(scene)
