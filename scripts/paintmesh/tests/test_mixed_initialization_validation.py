"""Guard coordinate conventions and evidence requirements in the screening pilot."""
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "verify_mixed_initialization.py"
spec = importlib.util.spec_from_file_location("mixed_validation", SCRIPT)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def identity_warp(h,w):
    yy,xx=np.mgrid[:h,:w]
    xy=np.stack(((xx+.5)/w*2-1,(yy+.5)/h*2-1),axis=-1)
    return np.concatenate((xy,xy),axis=-1)


def test_warp_pixel_centers_round_trip_different_image_sizes():
    warp=identity_warp(20,30)
    uv=np.array([[20.,15.],[50.,30.],[85.,45.]])
    target,score,cycle,valid=validation.warp_observations(warp,np.ones((20,30)),
        warp,np.ones((20,30)),uv,(60,90),(120,180))
    np.testing.assert_allclose(target,(uv+.5)*2-.5,atol=1e-12)
    np.testing.assert_allclose(cycle,0,atol=1e-12)
    assert valid.all() and (score==1).all()


def test_out_of_bounds_matches_cannot_become_valid_by_clamping():
    warp=identity_warp(20,30)
    uv=np.array([[-2.,15.],[50.,30.],[200.,45.]])
    _,_,_,valid=validation.warp_observations(warp,np.ones((20,30)),
        warp,np.ones((20,30)),uv,(60,90),(60,90))
    np.testing.assert_array_equal(valid,[False,True,False])


def camera(scale,center):
    world=np.eye(4);world[:3,:3]*=scale;world[:3,3]=-scale*np.asarray(center)
    return dict(w2c=world,c2w=np.linalg.inv(world),k=np.diag([700.,700.,1.]))


def test_camera_similarity_scale_does_not_change_world_position_uncertainty():
    points=np.array([[.1,.2,2.],[.5,0.,3.]])
    reference=sum(validation.point_information(points,camera(1.,c)) for c in ((-1,0,0),(1,0,0)))
    scaled=sum(validation.point_information(points,camera(.17,c)) for c in ((-1,0,0),(1,0,0)))
    np.testing.assert_allclose(scaled,reference,rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(validation.sigma_max(scaled),validation.sigma_max(reference),rtol=1e-12)
    singular=validation.point_information(points,camera(1.,(0,0,0)))
    assert np.isinf(validation.sigma_max(singular)).all()


def test_no_rgb_evidence_or_bad_depth_or_bad_conditioning_must_fail():
    n=6
    depth=dict(available=np.full(n,20),agree=np.full(n,19),front=np.zeros(n),known_front=np.zeros(n))
    rgb=dict(edges=np.full(n,3),views=np.ones((n,4),bool),sigma_spacing=np.full(n,.4))
    rgb['edges'][1]=0
    depth['agree'][2]=3
    depth['front'][3]=10
    depth['known_front'][4]=10
    rgb['sigma_spacing'][5]=np.inf
    np.testing.assert_array_equal(validation.select_points(depth,rgb),[True,False,False,False,False,False])


def test_fit_and_check_camera_sets_are_disjoint():
    cameras=[]
    for theta in np.linspace(0,2*np.pi,180,endpoint=False):
        cam=camera(1.,(2*np.cos(theta),2*np.sin(theta),1.))
        cameras.append(cam)
    pairs=validation.choose_pairs(cameras,np.array([[0.,0.,0.],[.01,0.,0.]]),validation.POLICY)
    fit={p[k] for p in pairs if p['role']=='fit' for k in ('a','b')}
    check={p[k] for p in pairs if p['role']=='check' for k in ('a','b')}
    assert fit and check and not fit&check
    assert len(pairs)==24 and all(p['centroid_parallax_degrees']>=5 for p in pairs)
