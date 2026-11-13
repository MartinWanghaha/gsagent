"""Geometry tests for the independent normal-aware fusion experiment."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.normal_surface_fusion import (integrate_normal_partition,
    refine_vertices_local,_prepare_frames,_sample_teacher)
from mesh_benchmark.surface_fusion_probe import blocks_for_triangles,triangles_in_roi


def plane_frame(height=14,width=14,a=5.,b=.04,c=-.06):
    yy,xx=np.mgrid[:height,:width]
    K=np.array([[30.,0,width/2],[0,30.,height/2],[0,0,1.]])
    depth=(1/(a+b*xx+c*yy)).astype('float32')
    n=-np.array([b*K[0,0],c*K[1,1],a+b*K[0,2]+c*K[1,2]],dtype='float32')
    n/=np.linalg.norm(n)
    normal=np.broadcast_to(n,(height,width,3)).copy()
    return dict(depth=depth,K=K,w2c=np.eye(4),normal=normal,
        rgb=np.full((height,width,3),[170,120,90],dtype='uint8'),
        alpha=np.ones_like(depth),confidence=np.ones_like(depth)),(a,b,c)


def test_known_slanted_surface_has_correct_camera_facing_sign():
    import torch
    frame,_=plane_frame();f=_prepare_frames([frame],'cpu')[0]
    uv=np.array([[3.25,3.25],[4.75,8.75],[8.25,5.25]])
    ray=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(frame['K']).T
    depth=1/(5+.04*uv[:,0]-.06*uv[:,1])
    surface=ray*depth[:,None]
    n=frame['normal'][0,0]
    query=np.concatenate([surface,surface+n*.003,surface-n*.003]).astype('float32')
    signed,_,weight,_,_=_sample_teacher(torch.tensor(query),f)
    np.testing.assert_allclose(signed.numpy()[:3],0,atol=1e-7)
    np.testing.assert_allclose(signed.numpy()[3:6],.003,atol=1e-7)
    np.testing.assert_allclose(signed.numpy()[6:],-.003,atol=1e-7)
    assert (weight.numpy()>0).all()
    # The alpha-blended normal numerator can have any nonzero magnitude.
    changed=dict(frame,normal=frame['normal']*-0.07)
    other=_sample_teacher(torch.tensor(query),_prepare_frames([changed],'cpu')[0])
    np.testing.assert_allclose(other[0].numpy(),signed.numpy(),atol=1e-7)
    np.testing.assert_allclose(other[2].numpy(),weight.numpy(),atol=1e-7)


def test_invalid_normals_and_depth_edges_cannot_create_blended_layer():
    import torch
    frame,_=plane_frame(10,12,a=5,b=0,c=0)
    frame['depth'][:,6:]=.4;frame['normal'][0,0]=np.nan;frame['normal'][0,1]=0
    data=_prepare_frames([frame],'cpu')[0]
    assert not data['continuity'][:,5].any()
    assert not data['valid'][0,:2].any()
    K=frame['K'];uv=np.array([[5.75,5.25],[6.25,5.25]])
    xyz=(np.c_[uv,np.ones(2)]@np.linalg.inv(K).T*.3).astype('float32')
    signed,_,weight,_,_= _sample_teacher(torch.tensor(xyz),data)
    assert signed[0]<-.09 and weight[0]==0  # occluded behind the nearer surface
    assert signed[1]>.09 and weight[1]>0   # free space before the farther surface
    assert not np.isclose(signed.numpy(),0,atol=.01).any()


def test_total_confidence_does_not_apply_alpha_weight_twice():
    import torch
    frame,_=plane_frame(a=5,b=0,c=0);frame['confidence'][:]=.35
    query=torch.tensor([[0.,0.,.2]],dtype=torch.float32)
    a=_sample_teacher(query,_prepare_frames([frame],'cpu')[0])[2]
    low=dict(frame,alpha=np.full_like(frame['alpha'],.55))
    b=_sample_teacher(query,_prepare_frames([low],'cpu')[0])[2]
    np.testing.assert_allclose(a.numpy(),b.numpy(),atol=1e-8)
    assert a.item()==pytest.approx(.35)
    rejected=dict(frame,alpha=np.full_like(frame['alpha'],.49))
    assert _sample_teacher(query,_prepare_frames([rejected],'cpu')[0])[2].item()==0


def test_normal_fusion_recovers_known_plane_and_native_rgb(tmp_path):
    import torch
    torch.set_num_threads(2)
    from mesh_benchmark.core import MeshScene
    from recompose.io import save_mesh
    frame,_=plane_frame();roi=np.array([[-.06,-.06,.17],[.06,.06,.26]])
    blocks=blocks_for_triangles(triangles_in_roi(frame,roi),roi)
    vertices,faces,color=integrate_normal_partition(blocks,[frame],device='CPU:0',chunk_blocks=7)
    assert len(faces)>0
    path=tmp_path/'plane.ply';save_mesh(path,vertices,faces,color)
    model=MeshScene([dict(path=str(path))])
    uv=np.array([[3.25,3.25],[4.75,8.75],[8.25,5.25],[9.75,9.75]])
    rr=np.c_[np.zeros((len(uv),3)),np.c_[uv,np.ones(len(uv))]@np.linalg.inv(frame['K']).T].astype('float32')
    z,rgb=model.cast(rr,color=True)
    truth=1/(5+.04*uv[:,0]-.06*uv[:,1])
    np.testing.assert_allclose(z,truth,atol=2e-6)
    np.testing.assert_allclose(rgb,np.broadcast_to(np.array([170,120,90])/255.,rgb.shape),atol=1/255.+1e-6)
    # Absent reliable normals must yield no surface, rather than a fallback
    # plane inferred from zero/NaN normal vectors.
    empty=dict(frame,normal=np.zeros_like(frame['normal']))
    vv,ff,cc=integrate_normal_partition(blocks,[empty],device='CPU:0',chunk_blocks=7)
    assert len(ff)==len(vv)==len(cc)==0


def test_vertex_refinement_moves_geometry_with_bounds_and_chunk_invariance():
    import torch
    torch.set_num_threads(2)
    frame,_=plane_frame(20,20,a=5,b=0,c=0)
    yy,xx=np.mgrid[5:15,5:15];K=frame['K']
    directions=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],-1)
    true=(directions*.2).reshape(-1,3).astype('float32')
    vertices=true.copy();vertices[:,2]+=.003*np.where(np.arange(len(vertices))%2,1,-1)
    faces=[]
    for y in range(9):
        for x in range(9):
            i=y*10+x;faces.extend([[i,i+1,i+10],[i+11,i+10,i+1]])
    faces=np.asarray(faces,dtype='int32')
    first,receipt=refine_vertices_local(vertices,[frame],faces=faces,iterations=6,device='CPU:0',chunk_vertices=13)
    second,_=refine_vertices_local(vertices,[frame],faces=faces,iterations=6,device='CPU:0',chunk_vertices=1000)
    assert np.abs(first[:,2]-.2).mean()<np.abs(vertices[:,2]-.2).mean()/10
    np.testing.assert_allclose(first,second,atol=1e-7)
    assert receipt['actual_mesh_edge_tangent_term'] and receipt['faces_unchanged']
    assert receipt['maximum_displacement']<=.008+1e-7
    assert all(r['max_step']<=.001+1e-7 for r in receipt['history'])
    assert receipt['moved_vertices']>0 and not receipt['differentiable_rasterizer_used']


def test_refinement_preserves_vertices_without_reliable_observations():
    frame,_=plane_frame();frame['confidence'][:]=0
    vertices=np.array([[0,0,.2],[.01,0,.2],[0,.01,.2]],dtype='float32')
    result,receipt=refine_vertices_local(vertices,[frame],faces=np.array([[0,1,2]]),device='CPU:0')
    np.testing.assert_array_equal(result,vertices)
    assert receipt['moved_vertices']==0


def test_refinement_rejects_new_triangle_collapse():
    import torch
    torch.set_num_threads(2)
    frame,_=plane_frame(20,20,a=5,b=0,c=0)
    # The third vertex must be off the first edge's interpolation plane. With
    # z=.2 it is already an almost-collinear sliver and projection preserves
    # >10% of its tiny original area, so no rejection is warranted. At .201,
    # the first proposed step genuinely collapses this triangle's area.
    vertices=np.array([[0,0,.197],[.002,0,.203],[.001,1e-8,.201]],dtype='float32')
    faces=np.array([[0,1,2]],dtype='int32')
    result,receipt=refine_vertices_local(vertices,[frame],faces=faces,iterations=6,
        device='CPU:0',edge_weight=0)
    a=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0])
    b=np.cross(result[1]-result[0],result[2]-result[0])
    assert np.dot(a,b)>0 and np.linalg.norm(b)>1e-12
    assert receipt['triangle_flip_and_collapse_protection']
    assert any(r['flipped_or_collapsed_face_attempts']>0 for r in receipt['history'])
