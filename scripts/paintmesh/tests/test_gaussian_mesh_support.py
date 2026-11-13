from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.gaussian_support import GaussianSupport


def test_anisotropic_support_is_not_a_nearest_center_heuristic():
    # Many nearer tiny Gaussians cannot hide a farther elongated Gaussian.
    x=np.zeros((40,3));x[:39,1]=.1;x[39]=[5,0,0]
    scales=np.full((40,3),.001);scales[-1]=[10,.001,.001]
    q=np.tile([1.,0,0,0],(40,1));model=GaussianSupport(x,scales,q,np.ones(40)*.8)
    np.testing.assert_array_equal(model.contains([[0,0,0],[0,0,.1]]),[True,False])


def test_rotation_and_native_opacity_threshold():
    q=[[np.sqrt(.5),0,0,np.sqrt(.5)]]
    model=GaussianSupport([[0,0,0]],[[2,.01,.01]],q,[.8])
    np.testing.assert_array_equal(model.contains([[0,3,0],[3,0,0]]),[True,False])
    model=GaussianSupport([[0,0,0],[10,0,0]],[[1,1,1],[1,1,1]],[[1,0,0,0]]*2,[.001,.8])
    np.testing.assert_array_equal(model.contains([[0,0,0],[10,0,0]]),[False,True])


def test_radius_bins_match_exhaustive_ellipsoid_test():
    rng=np.random.default_rng(42)
    centers=rng.normal(size=(200,3));scales=np.exp(rng.normal(-1,1.5,(200,3)))
    q=rng.normal(size=(200,4));opacity=rng.uniform(.001,1,200)
    model=GaussianSupport(centers,scales,q,opacity)
    points=rng.normal(size=(600,3))*5
    delta=points[:,None,:]-model.centers[None,:,:]
    d=np.einsum('pni,nij->pnj',delta,model.rotations)/model.scales[None,:,:]
    expected=(np.sum(d*d,axis=2)<=model.threshold[None,:]).any(1)
    np.testing.assert_array_equal(model.contains(points,chunk=17),expected)
