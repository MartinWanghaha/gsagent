"""Refinement must preserve shared coordinates and real geometric orientation."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.normal_mesh_refinement import refine_vertices_anchored,geometry_constraints


def fixture():
    n=12;yy,xx=np.mgrid[:n,:n]
    v=np.stack([(xx-(n-1)/2)*.0015,(yy-(n-1)/2)*.0015,
                .2+2e-5*np.cos(xx*.8)*np.cos(yy*.7)],-1).reshape(-1,3).astype('float32')
    f=[]
    for y in range(n-1):
        for x in range(n-1):
            i=y*n+x;f.extend([[i,i+1,i+n],[i+n+1,i+n,i+1]])
    frame=dict(depth=np.full((32,32),.2,dtype='float32'),
        normal=np.broadcast_to([0,0,-1.],(32,32,3)).copy(),
        K=np.array([[80.,0,16],[0,80.,16],[0,0,1.]]),w2c=np.eye(4),
        alpha=np.ones((32,32),dtype='float32'),confidence=np.ones((32,32),dtype='float32'),
        rgb=np.full((32,32,3),[170,120,90],dtype='uint8'))
    return v,np.asarray(f,dtype='int64'),frame


def test_anchored_refinement_reduces_plane_error_without_modifying_input():
    import torch
    torch.set_num_threads(2)
    v,f,frame=fixture();before=v.copy();indices=f.copy()
    result,receipt=refine_vertices_anchored(v,[frame],f,iterations=3,chunk_vertices=17)
    assert np.abs(result[:,2]-.2).mean()<np.abs(v[:,2]-.2).mean()/20
    np.testing.assert_array_equal(v,before);np.testing.assert_array_equal(f,indices)
    _,radius=geometry_constraints(v,f,.008)
    assert (np.linalg.norm(result.astype('float64')-v,axis=1)<=radius+1e-8).all()
    assert receipt['partition_independent_objective'] and not receipt['partition_global_rollback']
    assert receipt['maximum_displacement']<=.008+1e-8 and receipt['flipped_faces']==0
    assert all(row['max_step']<=.001+1e-7 for row in receipt['history'])


def test_shared_complete_one_rings_match_full_mesh_under_partition_and_permutation():
    import torch
    torch.set_num_threads(2)
    v,f,frame=fixture()
    full,_=refine_vertices_anchored(v,[frame],f,iterations=3,chunk_vertices=200)
    yy,xx=np.mgrid[:12,:12]
    halo=((xx>=2)&(xx<=9)&(yy>=2)&(yy<=9)).reshape(-1)
    kept=f[halo[f].all(1)]
    used,inverse=np.unique(kept,return_inverse=True);local_faces=inverse.reshape(-1,3)
    # Different vertex numbering and opposite face cyclic starts cannot change
    # a shared vertex's trust radius or frozen initial one-ring objective.
    permutation=np.arange(len(used))[::-1];old_to_new=np.empty(len(used),dtype='int64')
    old_to_new[permutation]=np.arange(len(used))
    local_faces=old_to_new[local_faces[:,[1,2,0]]]
    result,_=refine_vertices_anchored(v[used[permutation]],[frame],local_faces,iterations=3,chunk_vertices=7)
    lookup={int(i):j for j,i in enumerate(used[permutation])}
    shared=np.flatnonzero(((xx>=4)&(xx<=7)&(yy>=4)&(yy<=7)).reshape(-1))
    np.testing.assert_allclose(result[[lookup[int(i)] for i in shared]],full[shared],atol=1e-8,rtol=0)


def test_no_teacher_evidence_leaves_every_vertex_exact():
    v,f,frame=fixture();frame['confidence'][:]=0
    result,receipt=refine_vertices_anchored(v,[frame],f)
    np.testing.assert_array_equal(result,v);assert receipt['moved_vertices']==0


def test_analytic_radius_preserves_sliver_orientation_under_adversarial_moves():
    v=np.array([[0,0,0],[1,0,0],[.3,1e-4,0],[.7,-2e-4,0]],dtype='float64')
    f=np.array([[0,1,2],[1,0,3]])
    _,radius=geometry_constraints(v,f,.008)
    _,permuted=geometry_constraints(v,f[:,[2,0,1]],.008)
    np.testing.assert_allclose(radius,permuted,atol=1e-20,rtol=0)
    rng=np.random.default_rng(123)
    old=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    for _ in range(100):
        movement=rng.normal(size=v.shape);movement/=np.linalg.norm(movement,axis=1,keepdims=True)
        q=v+movement*radius[:,None]
        new=np.cross(q[f[:,1]]-q[f[:,0]],q[f[:,2]]-q[f[:,0]])
        ratio=(old*new).sum(1)/(old*old).sum(1)
        assert (ratio>47/64).all()


def test_float32_rounding_at_large_scene_coordinates_respects_tiny_trust_radius():
    import torch
    torch.set_num_threads(2)
    # Here a Float32 coordinate ulp is about 1.9e-6. The sliver's trust radius
    # lies between half and one ulp. An apparently safe Float64 displacement
    # would round to a whole ulp and exceed that radius at mesh export.
    vertices=np.array([[20,20,20],[20.01,20,20],
                       [20.005,20.000002,20.000016]],dtype='float32')
    faces=np.array([[0,1,2]],dtype='int64')
    _,radius=geometry_constraints(vertices,faces,.008)
    coordinate_ulp=float(np.spacing(np.float32(20)))
    assert (radius>coordinate_ulp*.5).all() and (radius<coordinate_ulp).all()
    pose=np.eye(4);pose[:3,3]=[-20,-20,-19.8]
    frame=dict(depth=np.full((32,32),.2,dtype='float32'),
        normal=np.broadcast_to([0,0,-1.],(32,32,3)).copy(),
        K=np.array([[80.,0,16],[0,80.,16],[0,0,1.]]),w2c=pose,
        alpha=np.ones((32,32),dtype='float32'),confidence=np.ones((32,32),dtype='float32'),
        rgb=np.full((32,32,3),[170,120,90],dtype='uint8'))
    before=vertices.copy()
    result,receipt=refine_vertices_anchored(vertices,[frame],faces,iterations=3,
        chunk_vertices=1,max_step=.001,max_displacement=.008)
    displacement=np.linalg.norm(result.astype('float64')-vertices.astype('float64'),axis=1)
    assert (displacement<=radius).all()
    assert receipt['float32_quantized_displacement_bounds_checked']
    assert sum(r['float32_quantized_vertex_moves_rejected'] for r in receipt['history'])>0
    assert all(r['max_step']<=.001 for r in receipt['history'])
    a=np.cross(vertices[1].astype('float64')-vertices[0],vertices[2].astype('float64')-vertices[0])
    b=np.cross(result[1].astype('float64')-result[0],result[2].astype('float64')-result[0])
    assert np.dot(a,b)>0 and receipt['flipped_faces']==0
    np.testing.assert_array_equal(vertices,before)
    # A different processing chunk cannot change which quantized moves pass.
    other,_=refine_vertices_anchored(vertices,[frame],faces,iterations=3,chunk_vertices=100)
    np.testing.assert_array_equal(other,result)
