"""Geometry/visibility/export tests use known synthetic ground truth, not learned guesses."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recompose.geometry import sim3,check_similarity,transform,RayMesh,camera_rays,backproject,project,surface_samples
from recompose.io import save_mesh,ply,write_json,read_json,crop_mesh
from recompose.gaussian import sh_basis,rotate_sh,write_gaussians,transform_gaussians
from recompose.pipeline import stage
from recompose.optimize import PoseProblem,optimize_pose,central_jacobian


def box():
    import open3d as o3d
    m=o3d.geometry.TriangleMesh.create_box(1.,.7,.4)
    v=np.asarray(m.vertices).copy();v-=np.array([.5,.35,0])
    return v,np.asarray(m.triangles).copy()


def lookat(eye,at):
    z=np.array(at)-eye;z/=np.linalg.norm(z);x=np.cross(z,[0,0,1.]);x/=np.linalg.norm(x);y=np.cross(z,x)
    c=np.eye(4);c[:3,:3]=np.stack([x,y,z],axis=1);c[:3,3]=eye;return np.linalg.inv(c)


def test_sim3_and_camera_roundtrip():
    rng=np.random.default_rng(2);p=rng.normal(size=(30,3));T=sim3(2.3,Rotation.from_euler('xyz',[.2,.5,-.1]).as_matrix(),[2,3,4])
    np.testing.assert_allclose(transform(transform(p,T),np.linalg.inv(T)),p,atol=1e-12)
    with pytest.raises(ValueError):check_similarity(np.diag([1.,2,1,1]))
    with pytest.raises(ValueError):check_similarity(np.diag([-1.,1,1,1]))
    K=np.array([[40,0,16],[0,40,16],[0,0,1.]])
    depth=np.full((32,32),2.);xyz=backproject(depth,K,T)
    uv,z=project(xyz.reshape(-1,3),K,T);y,x=np.mgrid[:32,:32]
    np.testing.assert_allclose(uv,np.c_[x.ravel(),y.ravel()],atol=1e-12);np.testing.assert_allclose(z,2.)


def test_inverse_raycast_preserves_depth_under_scale():
    v,f=box();T=sim3(2.,Rotation.from_euler('z',.3).as_matrix(),[0,0,3])
    a=RayMesh(v,f);b=RayMesh(transform(v,T),f)
    K=np.array([[60,0,32],[0,60,32],[0,0,1.]])
    y,x=np.mgrid[:64,:64];r=camera_rays(K,np.eye(4),np.c_[x.ravel(),y.ravel()])
    za,_,_=a.cast(r,T);zb,_,_=b.cast(r);np.testing.assert_allclose(za,zb,atol=1e-5)


def test_sh_rotation_preserves_directional_radiance():
    rng=np.random.default_rng(3);d=rng.normal(size=(100,3));d/=np.linalg.norm(d,axis=1,keepdims=True)
    R=Rotation.from_euler('xyz',[.3,-.8,.5]).as_matrix()
    for degree in range(4):
        coeff=rng.normal(size=(2,(degree+1)**2,3));rot=rotate_sh(coeff,R)
        np.testing.assert_allclose(np.einsum('bi,nic->nbc',sh_basis(d,degree),rot),np.einsum('bi,nic->nbc',sh_basis(d@R,degree),coeff),atol=1e-12)


def test_gaussian_covariance_and_sh_export(tmp_path):
    mu=np.array([[.1,.2,.3]]);scales=np.array([[.1,.2,.03]]);q=np.array([[1.,0,0,0]])
    c=np.random.default_rng(4).normal(size=(1,16,3));src=tmp_path/'local.ply';dst=tmp_path/'world.ply'
    write_gaussians(src,mu,scales,q,np.array([.7]),c)
    R=Rotation.from_euler('xyz',[.4,.2,.8]).as_matrix();T=sim3(2,R,[1,2,3]);transform_gaussians(src,dst,T)
    out=ply(dst)['vertex'].data
    actual_q=np.array([out[f'rot_{i}'][0] for i in range(4)]);actual_R=Rotation.from_quat(actual_q[[1,2,3,0]]).as_matrix()
    actual_s=np.exp([out[f'scale_{i}'][0] for i in range(3)])
    np.testing.assert_allclose(actual_R@np.diag(actual_s**2)@actual_R.T,4*R@np.diag(scales[0]**2)@R.T,atol=1e-7)
    coeff=np.concatenate([np.array([out[f'f_dc_{i}'][0] for i in range(3)])[None],np.array([out[f'f_rest_{i}'][0] for i in range(45)]).reshape(3,15).T])
    np.testing.assert_allclose(coeff,rotate_sh(c,R)[0],atol=1e-6)
    np.testing.assert_allclose(np.array([out[n][0] for n in ['x','y','z']]),transform(mu,T)[0],atol=1e-6)


def test_crop_keeps_crossing_triangles(tmp_path):
    v=np.array([[-2,-2,0],[2,-2,0],[0,3,0]],float);f=np.array([[0,1,2]])
    save_mesh(tmp_path/'x.ply',v,f)
    out,faces,_=crop_mesh(tmp_path/'x.ply',[-.1,-.1,-.1],[.1,.1,.1],100)
    assert len(faces)==1


def test_empty_filtered_roi_recovers_only_matching_tsdf_export(tmp_path):
    import os
    from recompose.io import crop_reconstruction
    raw=tmp_path/'tsdf_fusion.ply';post=tmp_path/'tsdf_fusion_post.ply';gs=tmp_path/'gaussians.ply'
    v=np.array([[-1.,-1,0],[1,-1,0],[0,1,0]]);faces=np.array([[0,1,2]])
    save_mesh(raw,v,faces);save_mesh(post,v+[10,10,10],faces);gs.write_bytes(b'model')
    for p,stamp in [(gs,100),(raw,200),(post,300)]:os.utime(p,(stamp,stamp))
    mesh,path,recovered=crop_reconstruction(post,np.array([-.5]*3),np.array([.5]*3),100,gs)
    assert recovered and path==raw and len(mesh[1])==1
    os.utime(gs,(250,250))
    with pytest.raises(ValueError,match='chronology'):crop_reconstruction(post,np.array([-.5]*3),np.array([.5]*3),100,gs)


def test_receipts_invalidate_corrupt_outputs_and_config(tmp_path):
    dest=tmp_path/'artifact';calls=[]
    def fn():calls.append(1);dest.write_text('verified')
    stage(tmp_path,'s',{'a':1},[dest],fn);stage(tmp_path,'s',{'a':1},[dest],fn);assert len(calls)==1
    dest.write_text('wrong');stage(tmp_path,'s',{'a':1},[dest],fn);assert len(calls)==2
    stage(tmp_path,'s',{'a':2},[dest],fn);assert len(calls)==3


def synthetic(tmp_path):
    root=tmp_path/'evidence';(root/'observations').mkdir(parents=True)
    v,f=box();c=np.tile([.7,.3,.1],(len(v),1));np.savez(tmp_path/'asset.npz',vertices=v,faces=f,colors=c)
    # A nonplanar supporting sheet. The central object's base just touches z=0.
    x,y=np.meshgrid(np.linspace(-2,2,40),np.linspace(-2,2,40));z=.02*np.sin(x)*np.sin(y)
    bg=np.c_[x.ravel(),y.ravel(),z.ravel()];faces=[]
    for i in range(39):
        for j in range(39):
            a=i*40+j;faces.extend([[a,a+1,a+40],[a+1,a+41,a+40]])
    faces=np.array(faces);save_mesh(root/'background_proxy.ply',bg,faces)
    T=sim3(1.,Rotation.from_euler('z',.35).as_matrix(),[.1,-.1,.015])
    object_mesh=RayMesh(v,f);bg_mesh=RayMesh(bg,faces);d=1.3
    points,_,_,_=surface_samples(v,f,500);points=transform(points,T)
    samples,_,_,_=surface_samples(bg,faces,5000)
    np.savez(root/'anchors.npz',points=points,normals=np.zeros_like(points),support_points=samples,support_confidence=np.ones(len(samples)),up=[0,0,1.])
    views=[];K=np.array([[75,0,32],[0,75,32],[0,0,1.]])
    for i,angle in enumerate(np.linspace(0,2*np.pi,5,endpoint=False)):
        w2c=lookat(np.array([3*np.cos(angle),3*np.sin(angle),2.]),[0,0,.2]);yy,xx=np.mgrid[:64,:64]
        rays=camera_rays(K,w2c,np.c_[xx.ravel(),yy.ravel()]);od,_,_=object_mesh.cast(rays,T);bd,_,_=bg_mesh.cast(rays)
        mask=(np.isfinite(od)&(od<=bd+.001)).reshape(64,64);depth=np.minimum(od,bd).reshape(64,64);depth[~np.isfinite(depth)]=0
        rgb=np.full((64,64,3),127,dtype='uint8');rgb[mask]=[178,76,25]
        np.savez(root/'observations'/f'{i:05d}.npz',mask=mask,depth=depth,background_depth=bd.reshape(64,64),K=K,w2c=w2c,rgb=rgb)
        views.append(dict(name=f'{i:05d}.png',original=f'train_{i}.png',validation=i==4,width=64,height=64,K=K,w2c=w2c))
    write_json(root/'observations.json',dict(diameter=d,views=views,up_confidence=1.))
    return root,tmp_path/'asset.npz',T


def config():return read_json(Path(__file__).parents[1]/'configs/recompose.json')


def test_pose_objective_ground_truth_and_occlusion(tmp_path):
    root,asset,T=synthetic(tmp_path);cfg=config()['pose'];cfg.update(surface_samples=300,rays_per_view=1000)
    p=PoseProblem(root,asset,cfg);wrong=T.copy();wrong[:3,3]+=[.3,.1,.2]
    assert p.observation(T)<p.observation(wrong)
    good,views=p.observation(T,validation=True,details=True);assert views[0]['iou']>.98
    fn=lambda x:p.anchor_residual(__import__('recompose.geometry',fromlist=['perturb']).perturb(T,x,p.d))
    assert np.linalg.norm(central_jacobian(fn,np.zeros(7)))>1


def test_full_pose_fit_recovers_perturbed_known_transform(tmp_path,monkeypatch):
    root,asset,T=synthetic(tmp_path);cfg=config()['pose'];cfg.update(surface_samples=500,rays_per_view=900,anchor_iterations=15,refine_candidates=1,pose_iterations=4,physics_iterations=2)
    initial=T.copy();initial[:3,3]+=[.08,-.04,.06]
    monkeypatch.setattr(PoseProblem,'candidates',lambda self,initial=None:[initial])
    r=optimize_pose(root,asset,tmp_path/'pose',cfg,initial)
    np.testing.assert_allclose(np.asarray(r['observation_pose'])[:3,3],T[:3,3],atol=.025)
    assert r['validation'][0]['iou']>.9
    assert list((tmp_path/'pose/review').glob('*.jpg'))


def test_merge_pads_each_color_sh_band_without_shuffling(tmp_path):
    from recompose.gaussian import merge_gaussians
    from recompose.render import load_gaussian_arrays
    paths=[];expected=[]
    for degree in [0,1,2,3]:
        c=np.arange((degree+1)**2*3,dtype=float).reshape(1,-1,3)/10
        p=tmp_path/f'{degree}.ply';paths.append(p)
        write_gaussians(p,np.zeros((1,3)),np.ones((1,3)),np.array([[1.,0,0,0]]),np.array([.7]),c)
        pad=np.zeros((16,3));pad[:c.shape[1]]=c[0];expected.append(pad)
    ranges=merge_gaussians(paths,tmp_path/'merged.ply')
    np.testing.assert_allclose(load_gaussian_arrays(tmp_path/'merged.ply')[-1],expected,atol=1e-6)
    assert [r['start'] for r in ranges]==[0,1,2,3]


@pytest.mark.skipif(__import__('os').environ.get('RECOMPOSE_CUDA_TESTS')!='1',reason='Explicit CUDA opt-in')
def test_cuda_projection_gradients_and_global_occlusion():
    import torch
    from recompose.render import camera_settings,render_layers
    device='cuda:0';torch.cuda.set_device(device)
    K=np.array([[55.,0,23.2],[0,58.,27.8],[0,0,1.]])
    settings=camera_settings(K,np.eye(4),64,64,device)
    mu=torch.tensor([[.12,-.08,2.]],device=device)
    scale=torch.full((1,3),.05,device=device);quat=torch.tensor([[1.,0,0,0]],device=device)
    alpha=torch.tensor([.8],device=device,requires_grad=True);coeff=torch.zeros((1,16,3),device=device)
    color=torch.tensor([[1.,.2,.1]],device=device,requires_grad=True)
    im=render_layers(settings,mu,scale,quat,alpha,coeff,color)
    w=im[0].detach().cpu().numpy();y,x=np.mgrid[:64,:64]
    uv,_=project(mu.cpu().numpy(),K,np.eye(4))
    np.testing.assert_allclose([np.sum(w*x)/w.sum(),np.sum(w*y)/w.sum()],uv[0],atol=.15)
    im.square().mean().backward()
    assert torch.isfinite(alpha.grad).all() and alpha.grad.abs().sum()>0
    assert torch.isfinite(color.grad).all() and color.grad.abs().sum()>0
    # Two layers along the same ray must be depth sorted, not RGB alpha overlaid by object ID.
    with torch.no_grad():
        points=torch.cat([mu,mu*1.5]);colors=torch.tensor([[1.,0,0],[0,0,1.]],device=device)
        image=render_layers(settings,points,scale.repeat(2,1),quat.repeat(2,1),torch.full((2,),.9,device=device),coeff.repeat(2,1,1),colors)
        u,v=np.rint(uv[0]).astype(int);assert image[0,v,u]>image[2,v,u]


def test_movement_rejects_scale_change_and_collision(tmp_path,monkeypatch):
    from recompose.scene import move_object
    scene=dict(objects=[dict(id='1',object_to_world=np.eye(4).tolist())]);path=tmp_path/'scene.json';write_json(path,scene)
    with pytest.raises(ValueError,match='scale'):move_object(path,'1',sim3(2,np.eye(3),[0,0,0]),tmp_path/'moved.json')
    monkeypatch.setattr('recompose.scene.validate_scene_placement',lambda scene:dict(accepted=False))
    with pytest.raises(ValueError,match='rejected'):move_object(path,'1',np.eye(4),tmp_path/'moved.json')
    assert not (tmp_path/'moved.json').exists()


def test_export_preserves_and_verifies_recovered_background_patch(tmp_path):
    from recompose.scene import export_scene
    from recompose.io import fingerprint
    v,f=box();mesh=tmp_path/'bg.ply';patch=tmp_path/'patch.ply';gs=tmp_path/'bg_gaussians.ply'
    save_mesh(mesh,v,f);save_mesh(patch,v+[2,0,0],f)
    write_gaussians(gs,np.zeros((1,3)),np.ones((1,3))*.01,np.array([[1.,0,0,0]]),np.array([.7]),np.zeros((1,16,3)))
    scene=dict(background=dict(mesh=fingerprint(mesh),gaussians=fingerprint(gs),patches=[fingerprint(patch)]),objects=[])
    write_json(tmp_path/'scene.json',scene);export_scene(tmp_path/'scene.json',tmp_path/'world')
    assert read_json(tmp_path/'world/layers.json')['background']['patches'][0]['path']==str(patch)
    patch.write_bytes(b'changed')
    with pytest.raises(ValueError,match='asset changed'):export_scene(tmp_path/'scene.json',tmp_path/'another')


def test_official_sam3d_row_convention_and_anisotropic_shape():
    from recompose.geometry import sam3d_pose_to_world
    v=np.random.default_rng(2).normal(size=(20,3));scale=np.array([1.1,2.3,.6])
    R=Rotation.from_euler('xyz',[.2,-.5,1.]).as_matrix();q=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
    t=np.array([.3,.4,1.2]);w2c=lookat(np.array([3.,2,1.]),[0,0,0])
    shape,T=sam3d_pose_to_world(scale,q,t,w2c)
    A=np.array([[1,0,0],[0,0,-1],[0,1,0]]);B=np.diag([-1.,-1,1])
    expected=transform(((v@A.T)*scale@R+t)@B.T,np.linalg.inv(w2c))
    np.testing.assert_allclose(transform(v@shape.T,T),expected,atol=1e-12)
    check_similarity(T)


def test_uv_texture_sampling_has_correct_vertical_orientation():
    from recompose.texture import sample_colors
    # Top-left red, bottom-left blue. glTF v=0 points at the bitmap's bottom.
    asset=dict(vertices=np.zeros((3,3)),faces=np.array([[0,1,2]]),colors=np.zeros((3,3)),
               triangle_material=np.array([0]),triangle_uv=np.array([[[.25,.25],[.25,.75],[.75,.75]]]),
               texture_0=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]],dtype='uint8'),texture_factor_0=np.ones(3))
    np.testing.assert_allclose(sample_colors(asset,np.array([0,0]),np.array([[1.,0,0],[0,1.,0]])),[[0,0,1],[1,0,0]])


def test_glb_transform_keeps_binary_textures_and_child_nodes(tmp_path):
    import struct
    from recompose.texture import transform_glb
    doc=dict(asset={'version':'2.0'},scene=0,scenes=[{'nodes':[0]}],nodes=[{'name':'canonical'}],buffers=[{'byteLength':4}])
    js=json.dumps(doc).encode();js+=b' '*((-len(js))%4);binary=b'ABCD'
    content=struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',4,0x004E4942)+binary
    (tmp_path/'in.glb').write_bytes(struct.pack('<4sII',b'glTF',2,12+len(content))+content)
    T=sim3(2.,np.eye(3),[1,2,3]);transform_glb(tmp_path/'in.glb',tmp_path/'out.glb',T)
    out=(tmp_path/'out.glb').read_bytes();size=struct.unpack_from('<I',out,12)[0];parsed=json.loads(out[20:20+size])
    np.testing.assert_allclose(np.array(parsed['nodes'][1]['matrix']).reshape(4,4).T,T)
    assert parsed['nodes'][1]['children']==[0] and out[-4:]==binary


