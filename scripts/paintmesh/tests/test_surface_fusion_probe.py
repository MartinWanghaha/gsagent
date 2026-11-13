"""Continuity must improve under perspective without bridging invalid edges."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.fine_tsdf import blocks_from_depth,contains_rows,integrate_partition
from mesh_benchmark.surface_fusion_probe import (continuous_cells,blocks_for_triangles,
    triangles_in_roi,integrate_sampled)


def test_continuous_allocation_covers_between_far_samples():
    h=w=40;yy,xx=np.mgrid[:h,:w]
    K=np.array([[40.,0,20],[0,40.,0],[0,0,1.]])
    d=np.divide(1.,yy,out=np.zeros((h,w)),where=yy>0)*100
    d[(d>5)|(d<1)]=0
    frame=dict(depth=d.astype('float32'),K=K,w2c=np.eye(4))
    roi=np.array([[-.4,2.49,1],[.4,2.51,5]])
    # Explicit exact local surface triangle: inverse depth interpolation is
    # projectively exact for this noiseless slanted plane.
    tri=triangles_in_roi(frame,roi)
    assert len(tri)
    old=blocks_from_depth(d,K,np.eye(4));new=blocks_for_triangles(tri,roi+np.array([[-.04]*3,[.04]*3]))
    p=np.array([[0,2.5,100/25.25]])
    key=np.floor(p/.032).astype('int32')
    assert not contains_rows(old,key)[0]
    assert contains_rows(new,key)[0]


def test_depth_discontinuity_and_missing_values_disable_interpolation():
    d=np.full((5,5),.2,dtype='float32');d[:,3:]=.4;d[0,0]=0
    valid=continuous_cells(d)
    assert not valid[0,0] and not valid[:,2].any()
    assert valid[1:,0:2].all() and valid[1:,3:].all()


def test_nearest_adapter_matches_native_geometry_and_color(tmp_path):
    import torch
    torch.set_num_threads(2)
    from recompose.io import save_mesh
    from mesh_benchmark.core import MeshScene
    h=w=12;yy,xx=np.mgrid[:h,:w]
    d=(.08+.0001*xx).astype('float32');K=np.array([[30.,0,6],[0,30.,6],[0,0,1.]])
    frame=dict(depth=d,K=K,w2c=np.eye(4),rgb=np.full((h,w,3),[170,120,90],dtype='uint8'))
    active=blocks_from_depth(d,K,np.eye(4))
    rr=np.c_[np.zeros((16,3)),np.c_[np.linspace(-.1,.1,16),np.zeros(16),np.ones(16)]].astype('float32')
    rendered=[]
    for name,method in [('native',lambda:integrate_partition(active,[frame],'CPU:0')),
                        ('adapter',lambda:integrate_sampled(active,[frame],'CPU:0','nearest',chunk_blocks=16))]:
        v,f,c=method();path=tmp_path/(name+'.ply');save_mesh(path,v,f,c)
        model=MeshScene([dict(path=str(path))]);rendered.append(model.cast(rr,color=True))
    np.testing.assert_allclose(rendered[0][0],rendered[1][0],atol=1e-6)
    np.testing.assert_allclose(rendered[0][1],rendered[1][1],atol=1e-7)


def test_inverse_depth_improves_a_noiseless_perspective_surface(tmp_path):
    import torch
    torch.set_num_threads(2)
    from recompose.io import save_mesh
    from mesh_benchmark.core import MeshScene
    h=w=14;yy,xx=np.mgrid[:h,:w];rho=5+.04*xx-.06*yy
    depth=(1/rho).astype('float32');K=np.array([[30.,0,7],[0,30.,7],[0,0,1.]])
    frame=dict(depth=depth,K=K,w2c=np.eye(4),rgb=np.full((h,w,3),[170,120,90],dtype='uint8'))
    roi=np.array([[-.06,-.06,.17],[.06,.06,.26]])
    active=blocks_for_triangles(triangles_in_roi(frame,roi),roi)
    uv=np.array([[3.25,3.25],[4.75,8.75],[8.25,5.25],[9.75,9.75]])
    direction=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T
    rr=np.c_[np.zeros_like(direction),direction].astype('float32')
    truth=1/(5+.04*uv[:,0]-.06*uv[:,1]);errors={}
    for mode in ['nearest','inverse']:
        v,f,c=integrate_sampled(active,[frame],'CPU:0',mode,chunk_blocks=16)
        path=tmp_path/(mode+'.ply');save_mesh(path,v,f,c);model=MeshScene([dict(path=str(path))])
        z,_=model.cast(rr)
        assert (z>0).all()
        errors[mode]=float(np.abs(z-truth).mean())
    assert errors['inverse']<1e-5
    assert errors['inverse']<errors['nearest']/10
