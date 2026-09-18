from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from support_mass import allocate,fit_surfels,sample_mass,load_config,surfel_jacobian
from support_geometry import DensityError, camera, project

CONFIG=Path(__file__).resolve().parents[1]/'configs/support_mass.yaml'


def plane(scale=1., slope=0.):
    cfg = load_config(CONFIG)
    record = dict(image_width=64, image_height=64, FoVx=2*np.arctan(.4), FoVy=2*np.arctan(.4),
                  R=np.eye(3).tolist(), T=[0,0,0], trans=[0,0,0], scale=1.)
    cam = camera(record)
    u = (np.arange(64)-32) / 80
    depth = np.broadcast_to(2*scale/(1-slope*u), (64,64)).copy()
    mask = np.zeros((64,64),bool); mask[20:44,20:44] = True
    frame = dict(camera=cam, depth=depth, removed=depth.copy(), alpha=np.ones_like(depth),
                 mask=mask, rgb=np.full((64,64,3),128,np.uint8))
    x,y = np.meshgrid(np.arange(-.75,.75,.01)*scale, np.arange(-.75,.75,.01)*scale)
    xyz = np.column_stack((x.ravel(), y.ravel(), 2*scale+slope*x.ravel()))
    _, pix, _, _ = project(xyz,cam)
    keep = ~mask[pix[:,1],pix[:,0]]
    ref = dict(xyz=xyz[keep], retained_rows=np.flatnonzero(keep), opacity=np.ones(keep.sum()),
               scale=np.ones((keep.sum(),3)) * .01*scale)
    return cfg, frame, ref


def test_quota_conservation_and_rounding():
    m=np.random.default_rng(4).random(100)*12
    n=allocate(m,0);scaled=m*(round(m.sum())/m.sum())
    assert n.sum()==round(m.sum())
    assert np.all((n==np.floor(scaled))|(n==np.ceil(scaled)))
    assert np.array_equal(n,allocate(m,0))
    assert allocate(np.zeros(12),0).sum()==0


@pytest.mark.parametrize('scale,slope',[(1.,0.),(3.,0.),(1.,.3)])
def test_mass_plane(scale,slope):
    _,frame,ref=plane(scale,slope)
    cfg=load_config(CONFIG)
    data,report,field,q=sample_mass([frame]*2,0,ref,cfg,lambda p:np.ones(len(p),bool))
    assert report['initialized_points']>frame['mask'].sum()
    assert np.allclose(data['xyz'][:,2],2*scale+slope*data['xyz'][:,0],atol=1e-7)
    assert len(data['xyz'])==q['count'].sum()==round(q['mass'].sum())
    assert np.allclose(field['area'].sum(),.6*.6*scale**2,rtol=.1) if slope==0 else True
    assert len(np.unique(data['xyz'].astype('f4'),axis=0))==len(data['xyz'])
    assert .8<report['density_ratio']<1.2  # test expectation, NOT runtime acceptance


@pytest.mark.parametrize('slope',[0.,.3])
def test_scale_and_batch_invariance(slope):
    cfg=load_config(CONFIG);results=[]
    for scale,batch in ((1.,512),(3.,512),(1.,1024)):
        _,f,r=plane(scale,slope)
        results.append(sample_mass([f],0,r,dict(cfg,batch_size=batch),lambda p:np.ones(len(p),bool)))
    assert len({len(x[0]['xyz']) for x in results})==1
    assert np.allclose(results[0][0]['xyz']*3,results[1][0]['xyz'])
    assert np.array_equal(results[0][0]['xyz'],results[2][0]['xyz'])


def test_density_adapts_across_one_surface():
    _,f,r=plane()
    gx=np.rint((r['xyz'][:,0]+.75)/.01).astype(int)
    gy=np.rint((r['xyz'][:,1]+.75)/.01).astype(int)
    keep=(r['xyz'][:,0]<0)|((gx%2==0)&(gy%2==0))
    r={k:v[keep] for k,v in r.items()}
    data,_,_,_=sample_mass([f],0,r,load_config(CONFIG),lambda p:np.ones(len(p),bool))
    assert (data['xyz'][:,0]<-.1).sum()>(data['xyz'][:,0]>.1).sum()*2


def test_original_gate_and_quota_not_discarded():
    _,f,r=plane()
    gate=lambda p:p[:,0]<.013
    data,report,fields,q=sample_mass([f],0,r,load_config(CONFIG),gate)
    assert gate(data['xyz']).all()
    assert len(data['xyz'])==q['count'].sum()
    assert report['gate_excluded_target_mass']>0
    assert (fields['full_area']>=fields['area']).all()
    with pytest.raises(DensityError,match='no seed surface'):
        sample_mass([f],0,r,load_config(CONFIG),lambda p:np.zeros(len(p),bool))


def test_zero_quota_when_background_contribution_suffices(monkeypatch):
    import support_mass
    from support_geometry import backproject
    _,f,r=plane()
    centers=backproject(f['depth'],f['camera'])[f['mask']]
    # Isolate conservation/zero append, independently of density estimation.
    monkeypatch.setattr(support_mass,'reference_field',lambda *args:(
        np.ones_like(f['depth']),{},(centers,np.ones(len(centers)))))
    data,report,_,q=sample_mass([f],0,r,load_config(CONFIG),lambda p:np.ones(len(p),bool))
    assert len(data['xyz'])==q['count'].sum()==0
    assert report['surplus_mass']>0


def write_mass_fixture(root,inputs,xyz):
    """Contract fixture for GPU consumer tests, not a claim of scene quality."""
    from local_geometry_io import write_json,record,seal
    from support_density_io import KIND,write_support
    root.mkdir(parents=True,exist_ok=True)
    n=len(xyz);index=np.arange(n,dtype=np.int64)
    data=dict(xyz=xyz,rgb=np.full((n,3),128),quota_id=index,pixel_id=index,
              spacing=np.ones(n),patch=np.zeros(n,dtype=np.int32))
    write_support(root/'support.ply',data)
    np.savez(root/'samples.npz',**data)
    np.savez(root/'quota.npz',count=np.ones(n,dtype=np.int64),mass=np.ones(n),target=np.ones(n),
             existing=np.zeros(n),pixel_id=index)
    np.savez(root/'field.npz',rho=np.ones(n),area=np.ones(n),pixel_id=index)
    request=seal(KIND+'-request',inputs=inputs,dependencies=[],
        parameters=dict(mode='mass_adaptive',config=load_config(CONFIG),seed_frame=4))
    write_json(root/'request.json',request)
    receipt=seal(KIND,request_id=request['artifact_id'],parameters=request['parameters'],
        inputs=dict(inputs,request=record(root/'request.json')),report=dict(initialized_points=n),
        outputs={k:record(root/name) for k,name in dict(support='support.ply',samples='samples.npz',
                 quota='quota.npz',field='field.npz').items()})
    path=root/'manifest.json';write_json(path,receipt)
    return path


def test_mass_receipt_quota_and_input_tampering(tmp_path):
    from local_geometry_io import record,write_json,read_json,identity
    from support_density_io import validate_density
    source=tmp_path/'source.txt';source.write_text('source')
    path=write_mass_fixture(tmp_path/'density',{'source_ply':record(source)},np.array([[0.,0.,2.],[.1,0.,2.]]))
    validate_density(path)
    quota=path.parent/'quota.npz'
    with np.load(quota) as values:data=dict(values)
    data['mass'][0]=2
    np.savez(quota,**data)
    # Re-sealing container hashes must not disguise an inconsistent quota.
    receipt=read_json(path);receipt['outputs']['quota']=record(quota)
    receipt['artifact_id']=identity({k:v for k,v in receipt.items() if k!='artifact_id'})
    write_json(path,receipt)
    with pytest.raises(ValueError,match='conservation'):validate_density(path)


def test_pixel_model_no_bridge_at_step_and_narrow_mask():
    cfg=load_config(CONFIG);_,frame,ref=plane()
    depth=frame['depth'].copy();depth[:,32:]=4
    ids,uv,slope,_,_=fit_surfels(depth,np.ones_like(depth,bool),1024)
    # Constant-depth half planes do not acquire an interpolated intermediate Z.
    assert np.allclose(slope,0,atol=1e-7)
    frame['mask'][:]=False;frame['mask'][32,20:44]=True
    data,report,_,_=sample_mass([frame],0,ref,cfg,lambda p:np.ones(len(p),bool))
    assert report['valid_hole_pixels']==24
    assert len(data['xyz'])>0


def test_no_reference_resource_and_invalid_depth():
    cfg=load_config(CONFIG);_,f,r=plane()
    with pytest.raises(DensityError,match='resource_limited'):
        sample_mass([f],0,r,dict(cfg,memory_budget_mb=1),lambda p:np.ones(len(p),bool))
    with pytest.raises(DensityError,match='no_reference'):
        sample_mass([dict(f,alpha=np.zeros_like(f['alpha']))],0,r,cfg,lambda p:np.ones(len(p),bool))
    f['depth'][32,32]=np.nan
    with pytest.raises(DensityError,match='undefined depth'):
        sample_mass([f],0,r,cfg,lambda p:np.ones(len(p),bool))


def test_invalid_known_observations_do_not_poison_reference():
    cfg=load_config(CONFIG);_,f,r=plane()
    f['removed'][4:8,4:8]=np.nan
    f['alpha'][4:8,4:8]=np.nan
    with np.errstate(invalid='raise',divide='raise',over='raise'):
        data,report,_,_=sample_mass([f]*2,0,r,cfg,lambda p:np.ones(len(p),bool))
    assert np.isfinite(data['xyz']).all() and report['initialized_points']>0


def test_no_quality_threshold_settings(tmp_path):
    for setting in ('normal_angle_deg: 89','min_coverage: 0','density_ratio: 2','max_points: 123'):
        p=tmp_path/'bad.yaml';p.write_text(setting)
        with pytest.raises(ValueError,match='thresholds'):load_config(p)


def test_camera_roundtrip():
    from support_geometry import backproject
    _,frame,_=plane()
    xyz=backproject(frame['depth'],frame['camera'])
    uv,_,z,_=project(xyz.reshape(-1,3),frame['camera'])
    y,x=np.indices(frame['depth'].shape)
    assert np.allclose(uv,np.column_stack((x.ravel(),y.ravel())))
    assert np.allclose(z,frame['depth'].ravel())


@pytest.mark.parametrize('setting',['mode','algorithm'])
def test_receipt_rejects_unsupported_identity(tmp_path,setting):
    from local_geometry_io import record,write_json,read_json,identity
    from support_density_io import validate_density
    path=write_mass_fixture(tmp_path/'density',{},np.array([[0.,0.,2.]]))
    request_path=path.parent/'request.json'
    request=read_json(request_path)
    if setting=='mode':request['parameters']['mode']='unsupported'
    else:request['parameters']['config']['algorithm']='unsupported'
    request['artifact_id']=identity({k:v for k,v in request.items() if k!='artifact_id'})
    write_json(request_path,request)
    receipt=read_json(path)
    receipt['parameters']=request['parameters']
    receipt['request_id']=request['artifact_id']
    receipt['inputs']['request']=record(request_path)
    receipt['artifact_id']=identity({k:v for k,v in receipt.items() if k!='artifact_id'})
    write_json(path,receipt)
    with pytest.raises(ValueError,match='unsupported density mode|unknown mass sampling algorithm'):
        validate_density(path)


def test_receipt_rejects_mutated_input(tmp_path):
    from local_geometry_io import record
    from support_density_io import validate_density
    source=tmp_path/'source.txt';source.write_text('original')
    path=write_mass_fixture(tmp_path/'density',{'source_ply':record(source)},np.array([[0.,0.,2.]]))
    validate_density(path)
    source.write_text('changed')
    with pytest.raises(ValueError,match='changed'):validate_density(path)
