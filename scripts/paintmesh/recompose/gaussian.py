"""Untrained Gaussian serialization, rigid transforms and lossless composition."""
from pathlib import Path
import os
import tempfile
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
from .geometry import transform, check_similarity
from .io import ply


def sh_basis(d,degree=3):
    """Real SH in the standard 3DGS ordering (direction = camera -> point)."""
    if degree not in range(4):raise ValueError('Supported SH degrees are 0..3')
    x,y,z=np.asarray(d).T;out=[np.full_like(x,.28209479177387814)]
    if degree>=1:out.extend([-.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x])
    if degree>=2:out.extend([1.0925484305920792*x*y,-1.0925484305920792*y*z,.31539156525252005*(2*z*z-x*x-y*y),-1.0925484305920792*x*z,.5462742152960396*(x*x-y*y)])
    if degree>=3:out.extend([-.5900435899266435*y*(3*x*x-y*y),2.890611442640554*x*y*z,-.4570457994644658*y*(4*z*z-x*x-y*y),.3731763325901154*z*(2*z*z-3*x*x-3*y*y),-.4570457994644658*x*(4*z*z-x*x-y*y),1.445305721320277*z*(x*x-y*y),-.5900435899266435*x*(x*x-3*y*y)])
    return np.stack(out,axis=-1)


def rotate_sh(coeff,R):
    """c_world evaluates identically to c_local at R^T d_world; exact through degree 3."""
    degree=round(np.sqrt(coeff.shape[1])-1)
    rng=np.random.default_rng(501);d=rng.normal(size=(128,3));d/=np.linalg.norm(d,axis=1,keepdims=True)
    mapping=np.linalg.lstsq(sh_basis(d,degree),sh_basis(d@R,degree),rcond=None)[0]
    return np.einsum('ij,njc->nic',mapping,coeff)


def write_gaussians(path,mu,scale,rotation,opacity,coeff):
    degree=round(np.sqrt(coeff.shape[1])-1);rest=(degree+1)**2-1
    fields=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(rest*3)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    v=np.zeros(len(mu),dtype=[(f,'f4') for f in fields])
    for i,n in enumerate(['x','y','z']):v[n]=mu[:,i]
    for i in range(3):v[f'f_dc_{i}']=coeff[:,0,i];v[f'scale_{i}']=np.log(np.maximum(scale[:,i],1e-12))
    if rest:
        flat=coeff[:,1:,:].transpose(0,2,1).reshape(len(mu),-1)
        for i in range(rest*3):v[f'f_rest_{i}']=flat[:,i]
    v['opacity']=np.log(np.clip(opacity,1e-6,1-1e-6)/(1-np.clip(opacity,1e-6,1-1e-6)))
    for i in range(4):v[f'rot_{i}']=rotation[:,i]
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(path))


def transform_gaussians(source,destination,T):
    s,R,t=check_similarity(T);v=ply(source)['vertex'].data.copy();names=v.dtype.names
    required=['x','y','z','opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+[f'f_dc_{i}' for i in range(3)]
    if not all(n in names for n in required):raise ValueError('Not a standard 3DGS PLY')
    xyz=transform(np.column_stack([v[n] for n in ['x','y','z']]),T)
    for i,n in enumerate(['x','y','z']):v[n]=xyz[:,i]
    q=np.column_stack([v[f'rot_{i}'] for i in range(4)])
    if np.any(np.linalg.norm(q,axis=1)<1e-8):raise ValueError('Invalid Gaussian quaternion')
    rot=(Rotation.from_matrix(R)*Rotation.from_quat(q[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
    for i in range(4):v[f'rot_{i}']=rot[:,i]
    for i in range(3):v[f'scale_{i}']+=np.log(s)
    nr=len([n for n in names if n.startswith('f_rest_')]);basis=1+nr//3
    if nr%3 or basis not in [1,4,9,16]:raise ValueError('Unsupported SH layout')
    for start in range(0,len(v),32768):
        sl=slice(start,start+32768);dc=np.column_stack([v[f'f_dc_{i}'][sl] for i in range(3)])[:,None,:]
        rest=np.column_stack([v[f'f_rest_{i}'][sl] for i in range(nr)]).reshape(-1,3,basis-1).transpose(0,2,1) if nr else np.empty((len(dc),0,3))
        coeff=rotate_sh(np.concatenate([dc,rest],1),R)
        for i in range(3):v[f'f_dc_{i}'][sl]=coeff[:,0,i]
        flat=coeff[:,1:,:].transpose(0,2,1).reshape(len(dc),nr)
        for i in range(nr):v[f'f_rest_{i}'][sl]=flat[:,i]
    if all(n in names for n in ['nx','ny','nz']):
        normal=np.column_stack([v[n] for n in ['nx','ny','nz']])@R.T
        for i,n in enumerate(['nx','ny','nz']):v[n]=normal[:,i]
    PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(destination))


def merge_gaussians(sources,destination):
    """Stream one viewer-compatible PLY; pad missing SH bands, omit classifier embeddings.

    The editable master remains scene.json. Gaussian ranges retain instance identity.
    """
    arrays=[ply(p)['vertex'].data for p in sources];n=sum(len(a) for a in arrays)
    fields=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    required=['x','y','z','opacity']+[f'f_dc_{i}' for i in range(3)]+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    for a in arrays:
        if not all(f in a.dtype.names for f in required):raise ValueError('Invalid source Gaussian schema')
        nr=len([f for f in a.dtype.names if f.startswith('f_rest_')])
        if nr not in (0,9,24,45):raise ValueError('Unsupported spherical harmonic degree')
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=dest.parent,prefix='.merged-');ranges=[];offset=0
    try:
        with os.fdopen(fd,'wb') as out:
            header='ply\nformat binary_little_endian 1.0\n'+f'element vertex {n}\n'+''.join(f'property float {f}\n' for f in fields)+'end_header\n'
            out.write(header.encode())
            for source,a in zip(sources,arrays):
                ranges.append(dict(source=str(source),start=offset,count=len(a)));offset+=len(a)
                nr=len([f for f in a.dtype.names if f.startswith('f_rest_')]);per_channel=nr//3
                for start in range(0,len(a),65536):
                    src=a[start:start+65536];chunk=np.zeros(len(src),dtype=[(f,'<f4') for f in fields])
                    for f in fields:
                        if f in src.dtype.names and not f.startswith('f_rest_'):chunk[f]=src[f]
                    for channel in range(3):
                        for band in range(per_channel):chunk[f'f_rest_{channel*15+band}']=src[f'f_rest_{channel*per_channel+band}']
                    out.write(chunk.tobytes())
            out.flush();os.fsync(out.fileno())
        os.replace(tmp,dest)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return ranges

