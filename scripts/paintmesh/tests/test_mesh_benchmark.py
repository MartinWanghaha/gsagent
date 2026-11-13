"""Geometric calibration, missing-surface penalties, and evaluation isolation."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.core import MeshScene,rays,points_from_depth,chamfer,psnr,ssim_map
from recompose.io import save_mesh

def test_triangle_camera_centers_and_occlusion(tmp_path):
    bg=tmp_path/'bg.ply';obj=tmp_path/'obj.ply'
    # An oblique triangle has nonconstant depth. This catches ray normalization,
    # flipped camera axes, and incorrect object-to-world translation.
    save_mesh(bg,np.array([[-2,-2,4],[2,-2,4],[0,2,6.]]),np.array([[0,1,2]]),np.ones((3,3)))
    save_mesh(obj,np.array([[-1,-1,2],[1,-1,2],[0,1,2.]]),np.array([[0,1,2]]),np.array([[1,0,0]]*3))
    camera=dict(width=1,height=1,K=np.array([[1.,0,.5],[0,1,.5],[0,0,1]]),w2c=np.eye(4))
    rr=rays(camera)
    np.testing.assert_allclose(rr,[[0,0,0,0,0,1]])
    far=MeshScene([dict(path=str(bg))]);z,_=far.cast(rr)
    np.testing.assert_allclose(z,[5],atol=1e-5)
    both=MeshScene([dict(path=str(bg)),dict(path=str(obj))]);z,rgb=both.cast(rr,color=True)
    np.testing.assert_allclose(z,[2],atol=1e-6);np.testing.assert_allclose(rgb,[[1,0,0]],atol=1e-6)
    np.testing.assert_allclose(points_from_depth(camera,z),[[0,0,2]])
    np.testing.assert_allclose(both.distance(np.array([[0,0,2],[0,0,1]],dtype='float32')),[0,1],atol=1e-5)

def test_chamfer_half_sum_unsquared():
    a=np.array([[0.,0,0],[2,0,0]]);b=np.array([[0.,0,0]])
    out=chamfer(a,b)
    assert out['CD_L1']==pytest.approx(.5)
    assert out['CD_L2']==pytest.approx(1.)

def test_masked_psnr_no_canvas_dilution_and_mesh_holes_penalized():
    target=np.ones((30,40,3),dtype='float32');pred=target.copy()
    mask=np.zeros((30,40),dtype=bool);mask[:10]=True;pred[:10]=0
    assert psnr(pred,target,mask)==pytest.approx(0)
    assert psnr(pred,target)>4
    assert psnr(target,target)==120
    np.testing.assert_allclose(ssim_map(target,target),1,atol=1e-6)

def test_field_gradients_and_finite_contraction():
    import torch
    from mesh_benchmark.appearance import ColorField
    model=ColorField(np.zeros(3),1.,resolution=8,rank=2)
    points=torch.tensor([[0.,0,0],[5,0,0],[1,1,1]])
    directions=torch.tensor([[0.,0,1]]*3)
    colors=model(points,directions)
    assert colors.shape==(3,3) and torch.isfinite(colors).all()
    (colors.square().mean()+model.tv()).backward()
    assert model.planes.grad is not None and torch.isfinite(model.planes.grad).all()

def test_background_excludes_object_for_all_image_metrics():
    import torch
    from mesh_benchmark.score import image_metrics
    class FakeSpatialLPIPS(torch.nn.Module):
        def forward(self,a,b):return (a-b).square().mean(1,keepdim=True)
    image=np.full((40,50,3),128,dtype='uint8');pred=image.copy()
    allowed=np.ones((40,50),dtype=bool);allowed[10:30,10:30]=False
    pred[~allowed]=0
    out=image_metrics(pred,image,allowed,FakeSpatialLPIPS(),'cpu',True)
    assert out['PSNR']==120 and out['SSIM']==pytest.approx(1.,abs=1e-5) and out['LPIPS']==0

def test_f1_known_precision_recall_and_no_threshold_filtering():
    from mesh_benchmark.core import surface_metrics
    ref=np.array([[0.,0,0],[2,0,0]])
    pred=np.array([[0.,0,0],[100,0,0]])
    out=surface_metrics(pred,ref,extent=1.)
    assert out['precision']==.5 and out['recall']==.5 and out['F1']==.5
    assert out['surface_samples']==2 and out['reference_points']==2
    assert out['CD_L1']>20  # Bad points must remain in the error distribution.

def test_coverage_anchor_selection_uses_real_observations_once():
    from mesh_benchmark.overlap import select_anchors
    # Reference 0 has one covered and one uncovered observation; reference 1
    # has no covered observation. Reference 2 is a background-mask exclusion.
    rid=np.array([0,0,1,2]);cam=np.array([0,1,0,0]);px=np.array([0,1,1,0])
    valid=[np.array([[True,False]]),np.array([[False,False]])]
    eligible=np.array([True,True,False])
    out=select_anchors(rid,cam,px,valid,eligible,seed=0)
    np.testing.assert_array_equal(out,[0])

def test_invalid_pixels_do_not_affect_any_score_or_ssim_windows():
    import torch
    from mesh_benchmark.score import image_metrics
    class FakeSpatialLPIPS(torch.nn.Module):
        def forward(self,a,b):return (a-b).square().mean(1,keepdim=True)
    rng=np.random.default_rng(2)
    target=rng.integers(0,256,(20,30,3),dtype='uint8');pred=target.copy()
    valid=np.zeros((20,30),dtype=bool);valid[4:16,4:16]=True
    pred[~valid]=0;target[~valid]=255
    out=image_metrics(pred,target,valid,FakeSpatialLPIPS(),'cpu')
    assert out['PSNR']==120 and out['SSIM']==pytest.approx(1.,abs=1e-5) and out['LPIPS']==0
    assert image_metrics(pred,target,np.zeros_like(valid),FakeSpatialLPIPS(),'cpu')['PSNR'] is None

def test_completed_mesh_refresh_only_invalidates_changed_scene_scores(tmp_path):
    import evaluate_recomposition_meshes as campaign
    for scene in ['bag','redbull']:
        for state in ['background','recomposition']:
            model=tmp_path/state/'paintmesh'/scene;model.mkdir(parents=True)
            (model/'metrics.json').write_text('{}');(model/'color_field.pt').write_bytes(b'keep field')
            cache=tmp_path/'cache/paintmesh'/scene/state;cache.mkdir(parents=True)
            (cache/'geometry_metrics.json').write_text('{}');(cache/'source_depth.npy').write_bytes(b'keep depth')
            support=tmp_path/'common'/scene/(state+'_support');support.mkdir(parents=True)
            (support/'complete.json').write_text('{}')
    old={'bag':['2dgs','paintmesh'],'redbull':['2dgs','paintmesh']}
    new={'bag':['2dgs','paintmesh'],'redbull':['2dgs','gof','paintmesh']}
    campaign.refresh_cohorts(tmp_path,old,new)
    for state in ['background','recomposition']:
        assert (tmp_path/state/'paintmesh/bag/metrics.json').exists()
        assert not (tmp_path/state/'paintmesh/redbull/metrics.json').exists()
        assert (tmp_path/state/'paintmesh/redbull/color_field.pt').read_bytes()==b'keep field'
        assert (tmp_path/'cache/paintmesh/redbull'/state/'source_depth.npy').exists()
        assert not (tmp_path/'common/redbull'/(state+'_support')).exists()

def test_whole_memory_retry_never_reduces_quality(tmp_path):
    from mesh_benchmark.core import write,read
    from run_whole_mesh_comparison import reduce_fusion_memory
    original=dict(voxel_size=.01,sdf_trunc=.06,grid_radius_resolution=384,
                  max_depth=100.,alpha_min=.5,train_names=['a.jpg','b.jpg'])
    write(tmp_path/'whole_scene_plan.json',dict(scenes={'bag':original}))
    import pytest
    before=(tmp_path/'whole_scene_plan.json').read_bytes()
    with pytest.raises(RuntimeError,match='coarsening is disabled'):
        reduce_fusion_memory(tmp_path,'bag')
    updated=read(tmp_path/'whole_scene_plan.json')['scenes']['bag']
    assert updated==original
    assert (tmp_path/'whole_scene_plan.json').read_bytes()==before

def test_whole_baseline_reuse_canonicalizes_matrices_and_isolates_outputs(tmp_path,monkeypatch):
    from mesh_benchmark import whole,core,cache
    from mesh_benchmark.core import write,read,identity
    old_base=tmp_path/'old';old=old_base/'mesh_comparison';new=tmp_path/'new'
    monkeypatch.setattr(whole,'BASE',old_base);monkeypatch.setattr(core,'SCENES',['bag'])
    monkeypatch.setattr(core,'METHODS',['2dgs','paintmesh'])
    mesh=tmp_path/'mesh.ply';mesh.write_text('mesh identity only')
    monkeypatch.setattr(core,'specs_for',lambda *a:[dict(path=str(mesh))])
    camera=[dict(K=np.eye(3),w2c=np.eye(4))]
    info=dict(cameras=camera,request=dict(masks=[]),request_id='new-reference')
    def fresh_common(output,scene):
        write(output/'common'/scene/'complete.json',info)
        return None,None,info  # Deliberately returns un-serialized numpy matrices.
    monkeypatch.setattr(cache,'common',fresh_common)
    write(old/'common/bag/complete.json',dict(cameras=camera,request=dict(masks=[])))
    write(new/'recomposition/bag/composition/scene.json',{})
    for state in ['background','recomposition']:
        source=old/'cache/2dgs/bag'/state;(source/'depth').mkdir(parents=True)
        (source/'train_pool.npz').write_bytes(b'immutable ray data')
        write(source/'complete.json',dict(complete=True,request=dict(common='old-reference',specs=[dict(identity=identity(mesh),transform=None)])))
        p=old/state/'2dgs/bag';p.mkdir(parents=True);(p/'color_field.pt').write_bytes(b'immutable field')
    whole.reuse_baseline_caches(new)
    for state in ['background','recomposition']:
        assert read(new/'cache/2dgs/bag'/state/'complete.json')['request']['common']=='new-reference'
        assert read(old/'cache/2dgs/bag'/state/'complete.json')['request']['common']=='old-reference'
        assert (new/'cache/2dgs/bag'/state/'train_pool.npz').is_symlink()
        assert (new/state/'2dgs/bag/color_field.pt').read_bytes()==b'immutable field'
        assert (new/state/'2dgs/bag/color_field.pt').stat().st_ino!=(old/state/'2dgs/bag/color_field.pt').stat().st_ino
