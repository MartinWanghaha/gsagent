from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.depth_tetra_field import (prepare_frames,query_field,make_pivots,
    marching_tetrahedra,refine_crossing_edges,refine_edges)


def plane_frame(z=1.,h=32,w=32):
    return dict(depth=np.full((h,w),z,dtype='float32'),
        normal=np.broadcast_to([0,0,-1.],(h,w,3)).astype('float32').copy(),
        rgb=np.full((h,w,3),[150,90,40],dtype='uint8'),
        alpha=np.ones((h,w),dtype='float32'),confidence=np.ones((h,w),dtype='float32'),
        K=np.array([[100.,0,w/2],[0,100.,h/2],[0,0,1.]]),w2c=np.eye(4))


def anchors(z=1.,n=(0,0,-1.)):
    return dict(points=np.array([[-.04,-.04,z],[.04,-.04,z],[-.04,.04,z],[.04,.04,z]],dtype='float32'),
        normals=np.broadcast_to(n,(4,3)).astype('float32').copy(),scale=np.full(4,.01,dtype='float32'))


def test_known_slanted_plane_sign_is_exact_and_chunk_invariant():
    import torch
    torch.set_num_threads(2)
    f=plane_frame();yy,xx=np.mgrid[:32,:32];a,b,c=1.,.002,-.001
    f['depth']=(1/(a+b*xx+c*yy)).astype('float32')
    normal=-np.array([b*100,c*100,a+b*16+c*16],dtype='float32');normal/=np.linalg.norm(normal)
    f['normal'][:]=normal
    uv=np.array([[11.25,12.75],[18.75,20.25]],dtype='float32')
    rays=np.c_[(uv[:,0]-16)/100,(uv[:,1]-16)/100,np.ones(2)]
    p=(rays/(a+b*uv[:,0]+c*uv[:,1])[:,None]).astype('float32')
    point=np.concatenate([p,p+normal*.005,p-normal*.005])
    reference=np.tile(p,(3,1));normals=np.broadcast_to(normal,point.shape).copy();scales=np.full(len(point),.01)
    aa=dict(points=p,normals=np.broadcast_to(normal,p.shape),scale=np.full(len(p),.01))
    refs=dict(reference_points=reference,reference_normals=normals,reference_scales=scales)
    out=query_field(point,aa,prepare_frames([f]),chunk_points=1,**refs)
    other=query_field(point,aa,prepare_frames([f]),chunk_points=1000,**refs)
    assert out['support'].all()
    np.testing.assert_allclose(out['field'],[0,0,.005,.005,-.005,-.005],atol=2e-7)
    np.testing.assert_array_equal(out['support'],other['support'])
    np.testing.assert_allclose(out['field'],other['field'],atol=2e-7)
    np.testing.assert_array_equal(out['colors'],np.tile([150,90,40],(6,1)))


def test_unknown_and_occluded_space_is_nan_not_synthetic_occupancy():
    f=plane_frame();aa=anchors()
    points=np.array([[0,0,1.01],[0,0,1.5],[3,0,1]],dtype='float32')
    refs=dict(reference_points=np.tile([0,0,1.],(3,1)),reference_normals=np.tile([0,0,-1.],(3,1)),reference_scales=np.full(3,.01))
    out=query_field(points,aa,prepare_frames([f]),**refs)
    np.testing.assert_array_equal(out['support'],[True,False,False])
    assert np.isnan(out['field'][1:]).all()
    assert out['diagnostics']['occluded_observations']>0


def test_thin_opposite_sheets_never_create_middle_surface():
    # One tetrahedron's corners belong to distinct, opposite-oriented sheets.
    points=np.array([[-.02,0,.99],[.02,0,1.01],[0,.02,1.03],[0,-.02,1.05]],dtype='float32')
    refs=np.array([[0,0,1],[0,0,1],[0,0,1.04],[0,0,1.04]],dtype='float32')
    normals=np.array([[0,0,-1],[0,0,-1],[0,0,1],[0,0,1]],dtype='float32')
    r=dict(field=np.array([.01,-.01,-.01,.01]),support=np.ones(4,dtype=bool),
        reference_points=refs,reference_normals=normals,reference_scales=np.full(4,.005))
    out=marching_tetrahedra(points,np.array([[0,1,2,3]]),r)
    assert len(out['faces'])==0 and out['diagnostics']['incompatible_normal_tetrahedra']==1
    # Equally-oriented but distinct parallel layers also cannot be averaged.
    r['reference_normals'][:]=[0,0,-1]
    out=marching_tetrahedra(points,np.array([[0,1,2,3]]),r)
    assert len(out['faces'])==0 and out['diagnostics']['incompatible_layer_tetrahedra']==1


def test_marching_plane_and_fixed_chart_refinement_preserve_real_root():
    f=plane_frame();aa=anchors()
    points=np.array([[-.04,-.04,.98],[.04,-.04,1.02],[-.04,.04,1.02],[.04,.04,1.02]],dtype='float32')
    references=points.copy();references[:,2]=1
    row=query_field(points,aa,prepare_frames([f]),reference_points=references,
        reference_normals=np.tile([0,0,-1.],(4,1)),reference_scales=np.full(4,.01))
    mesh=marching_tetrahedra(points,np.array([[0,1,2,3]]),row)
    assert mesh['faces'].shape==(1,3)
    np.testing.assert_allclose(mesh['vertices'][:,2],1,atol=1e-7)
    refined=refine_crossing_edges(points,mesh,aa,prepare_frames([f]),iterations=7)
    assert refined['valid'].all() and refined['diagnostics']['fixed_chart_during_refinement']
    np.testing.assert_allclose(refined['vertices'][:,2],1,atol=.0002)
    edges=mesh['crossing_edges']
    shard=refine_edges(points[edges[:,0]],points[edges[:,1]],aa,prepare_frames([f]),iterations=7,
        reference_points=mesh['edge_reference_points'],reference_normals=mesh['edge_reference_normals'],
        reference_scales=mesh['edge_reference_scales'])
    np.testing.assert_array_equal(shard['vertices'],refined['vertices'])
    np.testing.assert_array_equal(shard['support'],refined['valid'])
    normal=np.cross(mesh['vertices'][mesh['faces'][:,1]]-mesh['vertices'][mesh['faces'][:,0]],
        mesh['vertices'][mesh['faces'][:,2]]-mesh['vertices'][mesh['faces'][:,0]])
    assert (normal[:,2]<0).all()


def test_duplicate_pivots_with_conflicting_charts_are_explicitly_ambiguous():
    aa=dict(points=np.array([[0,0,1],[0,0,1]],dtype='float32'),
        normals=np.array([[0,0,-1],[0,0,1]],dtype='float32'),scale=np.array([.01,.01]))
    row=make_pivots(aa)
    assert len(row['points'])==3 and len(np.unique(row['points'],axis=0))==3
    assert row['reference_ambiguous'].all()
    assert row['diagnostics']['ambiguous_coordinates_require_unknown']


def test_spherical_patch_uses_depth_and_rotated_world_normals():
    f=plane_frame();yy,xx=np.mgrid[:32,:32];rays=np.stack([(xx-16)/100,(yy-16)/100,np.ones_like(xx)],-1)
    # Front hemisphere of sphere center (0,0,2), radius 1.
    a=(rays*rays).sum(-1);disc=4-a*3
    depth=(2-np.sqrt(disc))/a;pc=rays*depth[...,None]
    f['depth']=depth.astype('float32');f['normal']=(pc-[0,0,2]).astype('float32')
    theta=.3;R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    f['w2c'][:3,:3]=R;f['w2c'][:3,3]=[.1,-.05,.2]
    uv=np.array([[14,15],[17,19]])
    pp=pc[uv[:,1],uv[:,0]];world=(pp-f['w2c'][:3,3])@R
    nn=f['normal'][uv[:,1],uv[:,0]]@R;nn/=np.linalg.norm(nn,axis=1,keepdims=True)
    aa=dict(points=world.astype('float32'),normals=nn.astype('float32'),scale=np.full(2,.01))
    row=query_field(world,aa,prepare_frames([f]),reference_points=world,reference_normals=nn,reference_scales=np.full(2,.01))
    assert row['support'].all()
    np.testing.assert_allclose(row['field'],0,atol=3e-6)
    np.testing.assert_allclose(row['normals'],nn,atol=3e-5)


def test_unknown_corner_cannot_enter_marching_tetrahedra():
    vertices=np.array([[0,0,1],[.01,0,1],[0,.01,1],[0,0,1.01]])
    row=dict(field=np.array([1,-1,1,np.nan]),support=np.array([1,1,1,0],dtype=bool),
        reference_points=np.tile([0,0,1.],(4,1)),reference_normals=np.tile([0,0,-1.],(4,1)),reference_scales=np.full(4,.01))
    mesh=marching_tetrahedra(vertices,np.array([[0,1,2,3]]),row)
    assert len(mesh['faces'])==0 and mesh['diagnostics']['unknown_tetrahedra']==1
