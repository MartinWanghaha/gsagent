"""Column-vector Sim(3), fixed-mesh inverse ray casting, no global signed SDF."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def check_similarity(T):
    T=np.asarray(T,dtype=float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
        raise ValueError('Invalid homogeneous transform')
    s=np.cbrt(np.linalg.det(T[:3,:3]))
    if s<=0 or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3)*s*s,rtol=1e-5,atol=1e-8):
        raise ValueError('Only proper rotations and positive uniform scale are supported')
    return float(s),T[:3,:3]/s,T[:3,3].copy()


def transform(points,T):
    return np.asarray(points)@np.asarray(T)[:3,:3].T+np.asarray(T)[:3,3]


def sim3(scale=1.,rotation=None,translation=None):
    T=np.eye(4);T[:3,:3]=float(scale)*(np.eye(3) if rotation is None else rotation)
    if translation is not None:T[:3,3]=translation
    check_similarity(T);return T


def sam3d_pose_to_world(scale,quaternion_wxyz,translation,w2c):
    """Audited MV-SAM3D GLB -> row-rotation PyTorch3D -> CV -> COLMAP world.

    Return shape anisotropy to bake once, and a true Sim(3) placement matrix.
    """
    scale=np.asarray(scale,dtype=float).reshape(-1)
    if scale.size not in (1,3) or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid model scale')
    scale=np.broadcast_to(scale,(3,));uniform=float(np.prod(scale)**(1/3))
    q=np.asarray(quaternion_wxyz,dtype=float).reshape(4)
    if not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:raise ValueError('Invalid model quaternion')
    R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    A=np.array([[1,0,0],[0,0,-1],[0,1,0]],float);B=np.diag([-1.,-1.,1.])
    shape=A.T@np.diag(scale/uniform)@A
    M=np.eye(4);M[:3,:3]=B@R.T@A*uniform;M[:3,3]=B@np.asarray(translation).reshape(3)
    T=np.linalg.inv(w2c)@M;check_similarity(T)
    return shape,T


def perturb(T,x,diameter,scale=True):
    s,R,t=check_similarity(T)
    return sim3(s*np.exp(x[6]) if scale else s, Rotation.from_rotvec(x[:3]).as_matrix()@R,
                t+np.asarray(x[3:6])*diameter)


def project(points,K,w2c):
    q=transform(points,w2c);p=q@np.asarray(K).T
    return p[:,:2]/np.maximum(p[:,2:3],1e-12),q[:,2]


def backproject(depth,K,w2c):
    h,w=depth.shape;y,x=np.mgrid[:h,:w]
    rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(K).T
    return transform((rays*depth[...,None]).reshape(-1,3),np.linalg.inv(w2c)).reshape(h,w,3)


def camera_rays(K,w2c,uv):
    c2w=np.linalg.inv(w2c); dirs=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T@c2w[:3,:3].T
    return np.c_[np.broadcast_to(c2w[:3,3],dirs.shape),dirs].astype('float32')


def surface_samples(vertices,faces,count=10000,seed=42):
    tri=np.asarray(vertices)[np.asarray(faces)];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area=np.linalg.norm(cross,axis=1);valid=area>1e-14
    if not valid.any():raise ValueError('Mesh has no nondegenerate surface')
    rng=np.random.default_rng(seed);fi=rng.choice(len(tri),count,p=area/area.sum())
    uv=rng.random((count,2));u=np.sqrt(uv[:,0]);b=np.c_[1-u,u*(1-uv[:,1]),u*uv[:,1]]
    return (tri[fi]*b[:,:,None]).sum(1),cross[fi]/area[fi,None],fi,b


class RayMesh:
    def __init__(self,vertices,faces):
        import open3d as o3d
        self.o3d=o3d;self.vertices=np.asarray(vertices);self.faces=np.asarray(faces)
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=4)
        self.scene.add_triangles(o3d.core.Tensor(vertices,dtype=o3d.core.Dtype.Float32),
                                o3d.core.Tensor(faces,dtype=o3d.core.Dtype.UInt32))

    def cast(self,rays,T=None):
        """Directions remain unnormalized, so returned t is camera z-depth even under scale."""
        rays=np.asarray(rays,dtype='float32')
        if T is not None:
            inverse=np.linalg.inv(T);rays=np.c_[transform(rays[:,:3],inverse),rays[:,3:]@inverse[:3,:3].T].astype('float32')
        r=self.scene.cast_rays(self.o3d.core.Tensor(rays),nthreads=4)
        return r['t_hit'].numpy(),r['primitive_ids'].numpy(),r['primitive_normals'].numpy()

    def closest(self,points):
        r=self.scene.compute_closest_points(self.o3d.core.Tensor(np.asarray(points,dtype='float32')),nthreads=4)
        return r['points'].numpy(),r['primitive_normals'].numpy(),r['primitive_ids'].numpy()


def support_up(normals,weights=None):
    """Normal consensus only estimates up; it does not replace the support with a plane."""
    normals=np.asarray(normals);normals=normals[np.isfinite(normals).all(1)]
    if len(normals)<10:raise ValueError('Insufficient support normals')
    cov=normals.T@normals;val,vec=np.linalg.eigh(cov);n=vec[:,-1]
    if (normals@n).sum()<0:n=-n
    return n,float(val[-1]/max(val.sum(),1e-12))
