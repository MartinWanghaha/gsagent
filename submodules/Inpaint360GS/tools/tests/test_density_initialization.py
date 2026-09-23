"""Real CUDA initialization consumes validated supports without a second filter."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/paintmesh"))
from support_density_io import write_support
from scene.gaussian_model import GaussianModel


@pytest.mark.skipif(os.environ.get("PAINTMESH_LOCAL_GPU_TEST") != "1", reason="opt-in CUDA initialization")
@pytest.mark.parametrize('count',[0,1,2,144])
def test_density_initializer_preserves_exact_samples_and_background(tmp_path,count):
    model=GaussianModel(0)
    x,y=np.meshgrid(np.linspace(-1,1,5),np.linspace(-1,1,5))
    xyz=torch.tensor(np.column_stack((x.ravel(),y.ravel(),np.full(25,2))),dtype=torch.float32,device="cuda")
    for key,value in dict(_xyz=xyz,_features_dc=torch.zeros(25,1,3,device="cuda"),
        _features_rest=torch.zeros(25,0,3,device="cuda"),_opacity=torch.zeros(25,1,device="cuda"),
        _scaling=torch.full((25,3),-3.,device="cuda"),_rotation=torch.tensor([[1.,0,0,0]]*25,device="cuda"),
        _objects_dc=torch.zeros(25,1,16,device="cuda")).items():
        setattr(model,key,torch.nn.Parameter(value))
    model.spatial_lr_scale=1.
    xx,yy=np.meshgrid(np.linspace(-.2,.2,12),np.linspace(-.2,.2,12))
    points=np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,2)))[:count]
    support=tmp_path/"support.ply"
    write_support(support,dict(xyz=points,rgb=np.full(points.shape,128)))
    args=SimpleNamespace(supp_ply=str(support),temp_ply=str(tmp_path/"init.ply"),
        nb_points=100,radius=.1,threshold=1.,density_manifest="validated-by-entry-point",
        sh_degree=0,opacity_init=.1)
    optim=SimpleNamespace(percent_dense=.01,position_lr_init=1e-4,feature_lr=.001,
        opacity_lr=.01,scaling_lr=.001,rotation_lr=.001)
    removed=torch.zeros(25,1,1,device="cuda");removed[12]=1
    model.inpaint_setup(args,optim,{14:{"mask3d":removed}})
    assert int(model.sub_feature_num)==24
    assert len(model.get_xyz)==24+len(points)
    assert torch.equal(model.get_xyz[:24],xyz[torch.arange(25,device="cuda")!=12])
    assert np.allclose(model.get_xyz[24:].detach().cpu().numpy(),points)
    assert torch.isfinite(model.get_scaling).all()
