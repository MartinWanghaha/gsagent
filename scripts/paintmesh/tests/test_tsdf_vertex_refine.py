"""Actual mesh geometry, output precision and frozen seams must be preserved."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def fixture(size=9):
    yy,xx=np.mgrid[:size,:size]
    v=np.stack([(xx-(size-1)/2)*.002,(yy-(size-1)/2)*.002,
        .2+3e-5*np.sin(xx)*np.cos(yy)],-1).reshape(-1,3).astype('float32')
    f=[]
    for y in range(size-1):
        for x in range(size-1):
            i=y*size+x;f.extend([[i,i+1,i+size],[i+size+1,i+size,i+1]])
    frame=dict(depth=np.full((32,32),.2,dtype='float32'),
        normal=np.broadcast_to([0.,0.,-1.],(32,32,3)).copy(),
        K=np.array([[80.,0,16],[0,80.,16],[0,0,1.]]),w2c=np.eye(4),
        alpha=np.ones((32,32),dtype='float32'),confidence=np.ones((32,32),dtype='float32'),
        rgb=np.full((32,32,3),[170,120,90],dtype='uint8'))
    return v,np.array(f,dtype='int64'),frame


def cuda():
    import torch
    if not torch.cuda.is_available():pytest.skip('Fused teacher requires CUDA')
    return torch


def test_fused_teacher_matches_planar_camera_geometry_and_uses_every_view():
    torch=cuda()
    from mesh_benchmark.tsdf_vertex_refine import prepare_teacher,query_teacher
    _,_,f=fixture();teacher=prepare_teacher([f,f])
    points=torch.tensor([[0.,0.,.201],[.001,.002,.199],[1.,1.,.2]],device='cuda')
    normals=torch.tensor([[0.,0.,1.]]*3,device='cuda')
    o=query_teacher(points,normals,teacher)
    assert o[:2,14].tolist()==[2.,2.] and o[2,12]==0
    np.testing.assert_allclose(o[:2,5].cpu(),1.,atol=1e-6)
    np.testing.assert_allclose(o[:2,8].cpu(),[.001,-.001],atol=2e-7)
    np.testing.assert_allclose(o[:2,9:12].cpu(),[[0,0,1.]]*2,atol=1e-6)


def test_refinement_improves_depth_and_real_triangle_normals_without_moving_boundary():
    torch=cuda();torch.set_num_threads(2)
    from mesh_benchmark.tsdf_vertex_refine import prepare_teacher,refine_mesh
    v,f,frame=fixture();before=v.copy();faces=f.copy();teacher=prepare_teacher([frame])
    result,r=refine_mesh(v,f,teacher,iterations=10,normal_weight=.1,chunk_faces=31,chunk_vertices=17)
    mask=np.zeros((9,9),bool);mask[1:-1,1:-1]=True;mask=mask.reshape(-1)
    assert np.abs(result[mask,2]-.2).mean()<np.abs(v[mask,2]-.2).mean()
    np.testing.assert_array_equal(result[~mask],v[~mask])
    np.testing.assert_array_equal(v,before);np.testing.assert_array_equal(f,faces)
    assert r['flipped_faces']==0 and r['minimum_projected_area_ratio']>0
    assert r['normal_cosine_error_after']<r['normal_cosine_error_before']
    assert r['moved_vertices']>0 and not r['differentiable_rasterizer_used']


def test_no_observation_preserves_original_float64_positions_exactly():
    cuda()
    from mesh_benchmark.tsdf_vertex_refine import prepare_teacher,refine_mesh
    v,f,frame=fixture();v=v.astype('float64');v[:,0]+=1.234567890123e-7;frame['confidence'][:]=0
    result,r=refine_mesh(v,f,prepare_teacher([frame]),iterations=2,chunk_faces=31)
    np.testing.assert_array_equal(v,result)
    assert result.dtype==v.dtype and r['moved_vertices']==0


def test_two_partitions_freeze_shared_open_edge_vertices():
    cuda()
    from mesh_benchmark.tsdf_vertex_refine import prepare_teacher,refine_mesh
    v,f,frame=fixture();teacher=prepare_teacher([frame]);left=f[(v[f,0]<=0).all(1)]
    right=f[(v[f,0]>=0).all(1)];outputs=[]
    for fi in [left,right]:
        ids,inv=np.unique(fi,return_inverse=True)
        a,r=refine_mesh(v[ids],inv.reshape(-1,3),teacher,iterations=2,chunk_faces=31)
        outputs.append({int(i):q for i,q in zip(ids,a)})
    shared=set(outputs[0])&set(outputs[1]);assert shared
    for index in shared:
        np.testing.assert_array_equal(outputs[0][index],v[index])
        np.testing.assert_array_equal(outputs[1][index],v[index])
