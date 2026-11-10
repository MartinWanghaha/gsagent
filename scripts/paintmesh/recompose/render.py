"""Render unchanged inpaint background plus original deleted object splats."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .io import ply, read_json, write_json, fingerprint
from .geometry import check_similarity, transform
from .gaussian import rotate_sh
from .original_gaussians import movement_transform


def load_gaussian_arrays(path):
    v=ply(path)['vertex'].data;names=v.dtype.names
    mu=np.column_stack([v[k] for k in ['x','y','z']]).astype('float32')
    scale=np.exp(np.column_stack([v[f'scale_{i}'] for i in range(3)]))
    quat=np.column_stack([v[f'rot_{i}'] for i in range(4)]);quat/=np.linalg.norm(quat,axis=1,keepdims=True)
    opacity=1/(1+np.exp(-np.clip(v['opacity'],-30,30)))
    nr=len([n for n in names if n.startswith('f_rest_')]);coeff=np.zeros((len(mu),16,3),dtype='float32')
    coeff[:,0]=np.column_stack([v[f'f_dc_{i}'] for i in range(3)])
    if nr:
        rest=np.column_stack([v[f'f_rest_{i}'] for i in range(nr)]).reshape(len(mu),3,-1).transpose(0,2,1)
        coeff[:,1:1+rest.shape[1]]=rest
    for a in [mu,scale,quat,opacity,coeff]:
        if not np.isfinite(a).all():raise ValueError('Nonfinite Gaussian attributes')
    return mu,scale,quat,opacity,coeff


def camera_settings(K,w2c,width,height,device,degree=3):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    near,far=.001,1000.;K=np.asarray(K);W=np.asarray(w2c)
    P=np.zeros((4,4),dtype='float32');P[0,0]=2*K[0,0]/width;P[1,1]=2*K[1,1]/height
    # CUDA rasterizer uses ((ndc + 1) * dimension - 1) / 2 for pixel centers.
    P[0,2]=(2*K[0,2]+1)/width-1;P[1,2]=(2*K[1,2]+1)/height-1
    P[2,2]=far/(far-near);P[2,3]=-far*near/(far-near);P[3,2]=1
    tensor=lambda a:torch.as_tensor(np.asarray(a).copy(),device=device,dtype=torch.float32)
    args=dict(image_height=int(height),image_width=int(width),tanfovx=width/(2*K[0,0]),tanfovy=height/(2*K[1,1]),
              bg=torch.zeros(3,device=device),scale_modifier=1.,viewmatrix=tensor(W.T),projmatrix=tensor((P@W).T),
              sh_degree=degree,campos=tensor(np.linalg.inv(W)[:3,3]),prefiltered=False,debug=False)
    if 'antialiasing' in GaussianRasterizationSettings._fields:args['antialiasing']=False
    return GaussianRasterizationSettings(**args)


def render_layers(settings,mu,scale,quat,opacity,coeff,colors=None):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizer
    result=GaussianRasterizer(raster_settings=settings)(means3D=mu,
        means2D=torch.zeros_like(mu,requires_grad=torch.is_grad_enabled()),opacities=opacity[:,None],
        scales=scale,rotations=quat,shs=coeff if colors is None else None,colors_precomp=colors)
    return result[0]


def render_composition(scene_path,camera_path,output,device='cuda:0'):
    """Render all instances in one rasterization pass for correct inter-layer occlusion."""
    import torch
    from .scene import verify_scene_assets
    torch.cuda.set_device(torch.device(device));scene=read_json(scene_path);verify_scene_assets(scene)
    if str(camera_path).endswith('.npz'):
        camera=np.load(camera_path);h,w=camera['rgb'].shape[:2];K=camera['K'];w2c=camera['w2c']
    else:
        camera=read_json(camera_path);w,h=camera['width'],camera['height'];K=camera['K'];w2c=camera['w2c']
    arrays=[load_gaussian_arrays(scene['background']['gaussians']['path'])]
    for node in scene['objects']:
        mu,scale,q,alpha,sh=load_gaussian_arrays(node['gaussians']['path']);T=movement_transform(node);s,R,_=check_similarity(T)
        q=(Rotation.from_matrix(R)*Rotation.from_quat(q[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
        arrays.append((transform(mu,T),scale*s,q,alpha,rotate_sh(sh,R)))
    tensors=[torch.as_tensor(np.concatenate([a[i] for a in arrays]),device=device,dtype=torch.float32) for i in range(5)]
    with torch.no_grad():
        rgb=render_layers(camera_settings(K,w2c,w,h,device),*tensors).clamp(0,1).permute(1,2,0).cpu().numpy()
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.rint(rgb*255).astype('uint8')).save(dest)
    write_json(dest.with_suffix('.json'),dict(scene=fingerprint(scene_path),camera=fingerprint(camera_path),width=w,height=h,
        compositing='joint depth-sorted Gaussian rasterization',device=device))

