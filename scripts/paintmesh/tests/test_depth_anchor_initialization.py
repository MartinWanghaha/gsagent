"""Independent geometric controls for the depth-anchor initialization pilot."""
import importlib.util
from pathlib import Path
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'verify_depth_anchor_initialization.py'
spec=importlib.util.spec_from_file_location('anchor_pilot',SCRIPT)
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


def camera(center,scale=1.):
    w=np.eye(4);w[:3,:3]*=scale;w[:3,3]=-scale*np.asarray(center)
    return dict(w2c=w,c2w=np.linalg.inv(w),k=np.array([[500.,0,320],[0,500.,240],[0,0,1]]),w=640,h=480)


def test_rgb_triangulation_recovers_curved_surface_with_scaled_cameras():
    x,y=np.meshgrid(np.linspace(-.3,.3,8),np.linspace(-.2,.2,7))
    points=np.column_stack((x.ravel(),y.ravel(),(2+.7*x*x+.3*np.sin(y*4)).ravel()))
    cameras=[camera(c,.171) for c in ((-.25,0,0),(.3,0,.1),(0,.3,-.1))]
    uv=[pilot.project(points,c)[0] for c in cameras]
    recovered,errors,cov=pilot.triangulate_tracks(cameras,uv)
    np.testing.assert_allclose(recovered,points,atol=1e-9)
    assert errors.max()<1e-8 and (np.linalg.eigvalsh(cov)>0).all()
    for c,observation in zip(cameras,uv):
        _,depth,_=pilot.project(points,c)
        np.testing.assert_allclose(pilot.unproject(observation,depth,c),points,atol=1e-9)


def test_robust_correction_removes_bias_without_flattening_curved_depth():
    rng=np.random.default_rng(17)
    uv=rng.uniform(1,127,(350,2))
    def bias(p):return 3*np.sin(np.pi*p[:,0]/128)*np.sin(np.pi*p[:,1]/128)
    observations=-bias(uv)+rng.normal(0,.03,len(uv))
    corrupted=rng.choice(len(uv),70,replace=False);observations[corrupted]+=10
    q=np.linspace(0,128,33)
    border=np.concatenate([np.column_stack((q,np.full(len(q),v))) for v in (0,128)] +
                          [np.column_stack((np.full(len(q),v),q)) for v in (0,128)])
    coeff,weight=pilot.fit_correction(uv,observations,np.full(len(uv),.2),np.zeros(2),(9,9),16,border)
    check=rng.uniform(2,126,(1000,2));correction=pilot.grid_basis(check,np.zeros(2),(9,9),16)@coeff
    truth=20+2*np.sin(check[:,0]/10)+3*np.cos(check[:,1]/25)
    original=truth+bias(check);corrected=original+correction
    before=np.sqrt(np.mean((original-truth)**2));after=np.sqrt(np.mean((corrected-truth)**2))
    assert after<before*.3,(before,after)
    assert np.median(weight[corrupted])<.1
    # The recovered depth retains its nonplanar high-frequency structure.
    assert np.std(corrected)>2


def test_no_anchors_cannot_invent_a_depth_correction():
    coeff,_=pilot.fit_correction(np.empty((0,2)),np.empty(0),np.empty(0),np.zeros(2),(5,5),16,np.empty((0,2)))
    np.testing.assert_array_equal(coeff,np.zeros(25))


def test_missing_rgb_evidence_or_contradictions_fail_confidence_gate():
    e=dict(edges=np.array([2,0,2,2]),confident=np.array([2,0,10,2]),
           views=np.ones((4,3),bool),sigma_spacing=np.array([.5,.5,.5,2.]))
    np.testing.assert_array_equal(pilot.accepted_rgb(e),[True,False,False,False])


def test_uneditable_background_anchors_cannot_drive_masked_depth_field():
    mask=np.zeros((10,12),bool);mask[3:7,4:9]=True
    uv=np.array([[5.,4.],[2.,4.],[8.8,4.],[5.,6.5],[-1.,4.]])
    np.testing.assert_array_equal(pilot.editable_anchor_rows(mask,uv),[True,False,False,False,False])
