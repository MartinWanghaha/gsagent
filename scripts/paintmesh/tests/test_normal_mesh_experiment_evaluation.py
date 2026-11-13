"""Isolated full-mesh publication must retain all methods, rays and provenance."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate_normal_mesh_experiment as experiment
from mesh_benchmark.core import write,read


def triangle(path,z,color):
    from recompose.io import save_mesh
    vertices=np.array([[-1,-1,z],[1,-1,z],[0,1,z]],dtype='float32')
    save_mesh(path,vertices,np.array([[0,1,2]]),np.broadcast_to(np.array(color,dtype='float32'),(3,3)))
    return dict(path=str(path))


def camera():
    return dict(name='heldout.png',stem='heldout',part='held_out',width=3,height=3,
        K=[[3.,0,1.5],[0,3.,1.5],[0,0,1.]],w2c=np.eye(4).tolist())


def test_stream_render_resumes_after_committed_partition(tmp_path,monkeypatch):
    specs=[triangle(tmp_path/'first.ply',2,[.8,.2,.1]),triangle(tmp_path/'second.ply',3,[.1,.8,.2])]
    original=experiment.MeshScene;calls=[]
    def interrupted(specs):
        calls.append(specs[0]['path'])
        if len(calls)==2:raise RuntimeError('simulated interruption')
        return original(specs)
    monkeypatch.setattr(experiment,'MeshScene',interrupted)
    folder=tmp_path/'resumed'
    with pytest.raises(RuntimeError,match='simulated interruption'):
        experiment.render_specs(folder,specs,[camera()],checkpoint_parts=1)
    assert not (folder/'complete.json').exists()
    assert read(folder/'checkpoint.json')['completed_parts']==1
    resumed=[]
    def tracking(specs):resumed.append(specs[0]['path']);return original(specs)
    monkeypatch.setattr(experiment,'MeshScene',tracking)
    experiment.render_specs(folder,specs,[camera()],checkpoint_parts=1)
    assert resumed==[str(tmp_path/'second.ply')]
    experiment.render_specs(tmp_path/'monolithic',specs,[camera()],checkpoint_parts=32)
    for suffix in ['_depth.npy','_normal.npy']:
        np.testing.assert_array_equal(np.load(folder/('heldout'+suffix)),np.load(tmp_path/'monolithic'/('heldout'+suffix)))
    from PIL import Image
    np.testing.assert_array_equal(np.asarray(Image.open(folder/'heldout.png')),np.asarray(Image.open(tmp_path/'monolithic/heldout.png')))
    # A changed published PNG must be regenerated even though its source mesh
    # and complete marker are unchanged.
    Image.new('RGB',(3,3),(0,0,0)).save(folder/'heldout.png')
    resumed.clear();experiment.render_specs(folder,specs,[camera()],checkpoint_parts=1)
    assert resumed==[s['path'] for s in specs]


def test_all_three_full_extractions_required(tmp_path):
    spec=triangle(tmp_path/'mesh.ply',2,[.8,.2,.1])
    for key in ['A','N']:write(tmp_path/key/'complete.json',dict(complete=True,mesh_specs=[spec],full_scene=True,roi=None))
    with pytest.raises(ValueError,match='R: complete full-scene extraction is not ready'):
        experiment.validate_methods(tmp_path)
    write(tmp_path/'R/complete.json',dict(complete=True,mesh_specs=[spec],full_scene=True,roi=[[-1,-1,-1],[1,1,1]]))
    with pytest.raises(ValueError,match='local-only'):experiment.validate_methods(tmp_path)


def test_composition_respects_foreground_occlusion_and_existing_holes():
    depth=np.array([[1.,3.,0]],dtype='float32');other=np.array([[2.,2.,4]],dtype='float32')
    color=np.full((1,3,3),30,dtype='uint8');new_color=np.full_like(color,180)
    normal=np.broadcast_to([0,0,1.],(1,3,3));new_normal=np.broadcast_to([0,1.,0],(1,3,3))
    z,rgb,n=experiment.compose_arrays(depth,color,normal,other,new_color,new_normal)
    np.testing.assert_array_equal(z,[[1,2,4]])
    np.testing.assert_array_equal(rgb[0,:,0],[30,180,180])
    np.testing.assert_array_equal(n[0],[[0,0,1],[0,1,0],[0,1,0]])


def test_teacher_is_floating_geometry_and_empty_views_stay_explicit():
    c=camera();depth=np.full((3,3),2,dtype='float32')
    normal=np.broadcast_to([0,0,1.],(3,3,3)).copy()
    teacher=dict(depth=depth.copy(),normal=-normal,alpha=np.ones((3,3)))
    allowed=np.ones((3,3),dtype=bool)
    row=experiment.teacher_consistency(c,depth,normal,teacher,allowed,allowed)
    assert row['depth_abs_mean']==0 and row['normal_angle_mean_deg']==0
    assert row['pixels']==9 and row['own_teacher_coverage']==1
    row=experiment.teacher_consistency(c,depth,normal,teacher,np.zeros((3,3),dtype=bool),allowed)
    assert not row['computed'] and row['pixels']==0
    means=experiment.average_rows([dict(PSNR=None,mesh_coverage=0),dict(PSNR=30,mesh_coverage=1)],['PSNR','mesh_coverage'])
    assert means==dict(PSNR=30.,mesh_coverage=.5)
