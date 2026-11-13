"""Frozen-GS geometry evidence must preserve baseline geometry and reliability."""
from pathlib import Path
import sys
import copy
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_normal_fusion_frames import unit_camera_normals,normal_evidence,compare_reference,validate_sources


def test_camera_normals_are_unit_and_oriented_without_alpha_amplitude_bias():
    depth=np.ones((5,7),dtype='float32');K=np.array([[100.,0,3.5],[0,100.,2.5],[0,0,1.]])
    weighted=np.zeros((3,5,7),dtype='float32');weighted[2]=.12
    derived=np.zeros_like(weighted);derived[2]=-.6;alpha=np.full(depth.shape,.75,dtype='float32')
    a=normal_evidence(depth,K,weighted,derived,alpha)
    b=normal_evidence(depth,K,weighted*4,derived*.5,alpha)
    np.testing.assert_allclose(a['normal'],np.broadcast_to([0,0,-1],(*depth.shape,3)))
    np.testing.assert_allclose(a['confidence'],alpha)
    np.testing.assert_array_equal(a['confidence'],b['confidence'])
    np.testing.assert_allclose(np.linalg.norm(a['normal'],axis=-1),1)
    assert a['normal'].dtype==np.float32 and a['normal_valid'].all()


def test_missing_derived_normals_keep_soft_support_and_invalid_render_normals_do_not():
    depth=np.ones((5,7),dtype='float32');K=np.array([[100.,0,3.5],[0,100.,2.5],[0,0,1.]])
    normal=np.zeros((5,7,3),dtype='float32');normal[...,2]=-1
    derived=normal.copy();derived[1,1]=0;normal[2,2]=[np.inf,0,-1];depth[3,3]=0
    alpha=np.full(depth.shape,.8,dtype='float32');original=depth.copy()
    result=normal_evidence(depth,K,normal,derived,alpha)
    assert result['normal_valid'][1,1] and not result['depth_normal_valid'][1,1]
    assert result['confidence'][1,1]==pytest.approx(.2)
    assert result['confidence'][2,2]==0 and result['confidence'][3,3]==0
    np.testing.assert_array_equal(depth,original)
    assert np.isfinite(result['normal']).all() and np.isfinite(result['confidence']).all()


def test_angular_disagreement_softly_reduces_normal_reliability():
    depth=np.ones((5,7),dtype='float32');K=np.array([[100.,0,3.5],[0,100.,2.5],[0,0,1.]])
    normal=np.broadcast_to([0,0,-1.],(*depth.shape,3)).copy()
    derived=np.broadcast_to([.6,0,-.8],normal.shape).copy()
    result=normal_evidence(depth,K,normal,derived,np.full(depth.shape,.75,dtype='float32'))
    np.testing.assert_allclose(result['normal_agreement'],.8,atol=1e-7)
    np.testing.assert_allclose(result['confidence'],.75*(.25+.75*.8**2),atol=1e-7)


def test_reference_comparison_detects_one_float_step_and_rgb_change():
    base=dict(depth=np.ones((2,2),dtype='float32'),rgb=np.full((2,2,3),120,dtype='uint8'),K=np.eye(3),w2c=np.eye(4))
    fresh={k:v.copy() for k,v in base.items()}
    assert compare_reference(base,fresh)['all_exact']
    fresh['depth'][0,0]=np.nextafter(np.float32(1),np.float32(2))
    check=compare_reference(base,fresh)
    assert not check['all_exact'] and check['depth']['different_values']==1
    assert check['depth']['maximum_absolute_difference']>0
    fresh['rgb'][0,0,1]+=1
    assert not compare_reference(base,fresh)['rgb']['exact']


def test_background_source_uses_its_actual_checkpoint_iteration(tmp_path):
    from mesh_benchmark.core import identity
    model=tmp_path/'model';checkpoint=model/'point_cloud/iteration_5000/point_cloud.ply'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'frozen Gaussian checkpoint')
    config=model/'config.yaml';config.write_text('frozen resolved config')
    rows=[]
    for i in range(85):
        p=tmp_path/(f'{i}.npz');p.write_bytes(b'unchanged RGB-D')
        rows.append(dict(path=str(p),identity=identity(p)))
    parameters=dict(scene='bag',kind='background',iteration=5000,resolution=4,alpha_min=.5,
                    checkpoint=str(checkpoint),model=str(model))
    source=dict(request=dict(parameters=parameters,checkpoint=identity(checkpoint),config=identity(config)),frames=rows)
    assert validate_sources(source,'background')['iteration']==5000
    with pytest.raises(ValueError,match='selected source kind'):validate_sources(source,'stage1')
    source['request']['parameters']['iteration']=10000
    with pytest.raises(ValueError,match='Checkpoint iteration'):validate_sources(source,'background')
