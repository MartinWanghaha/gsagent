"""Exact finite Gaussian support queries with conservative spatial broad phase.

The opacity cutoff is the native rasterizer's 1/255, not a reference-distance
threshold. Radius bins only accelerate candidate enumeration; every ellipsoid
which can contain a query is checked, including distant, large anisotropic ones.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit
from .color_export import read_mesh,positions


def rotations(quaternions):
    q=np.asarray(quaternions,dtype='float64')
    q=q/np.linalg.norm(q,axis=1,keepdims=True)
    w,x,y,z=q.T;r=np.empty((len(q),3,3),dtype='float64')
    r[:,0,0]=1-2*(y*y+z*z);r[:,0,1]=2*(x*y-w*z);r[:,0,2]=2*(x*z+w*y)
    r[:,1,0]=2*(x*y+w*z);r[:,1,1]=1-2*(x*x+z*z);r[:,1,2]=2*(y*z-w*x)
    r[:,2,0]=2*(x*z-w*y);r[:,2,1]=2*(y*z+w*x);r[:,2,2]=1-2*(x*x+y*y)
    return r


class GaussianSupport:
    def __init__(self,centers,scales,quaternions,opacity,cutoff=1/255):
        centers=np.asarray(centers,dtype='float64');scales=np.asarray(scales,dtype='float64')
        opacity=np.asarray(opacity,dtype='float64').reshape(-1)
        eligible=opacity>=cutoff
        if not eligible.any():raise ValueError('No Gaussian reaches the native opacity cutoff')
        self.centers=centers[eligible];self.scales=scales[eligible];self.opacity=opacity[eligible]
        self.rotations=rotations(np.asarray(quaternions)[eligible]);self.cutoff=float(cutoff)
        self.threshold=2*np.log(self.opacity/cutoff)
        self.radii=self.scales.max(1)*np.sqrt(np.maximum(self.threshold,0))
        self.median_sigma=float(np.median(self.scales.max(1)))
        exponent=np.floor(np.log2(np.maximum(self.radii,1e-20))).astype('int32')
        self.bins=[]
        for value in sorted(np.unique(exponent)):
            ids=np.flatnonzero(exponent==value)
            self.bins.append((ids,cKDTree(self.centers[ids]),float(self.radii[ids].max())))

    @classmethod
    def from_ply(cls,path,cutoff=1/255):
        v=read_mesh(path)['vertex'].data
        scales=np.exp(np.column_stack([v['scale_'+str(i)] for i in range(3)]).astype('float64'))
        filtered=scales
        alpha=expit(np.asarray(v['opacity'],dtype='float64'))
        if 'filter_3D' in v.dtype.names:
            f=np.asarray(v['filter_3D'],dtype='float64')[:,None]
            filtered=np.sqrt(scales*scales+f*f)
            alpha*=np.prod(scales/filtered,axis=1)
        quaternions=np.column_stack([v['rot_'+str(i)] for i in range(4)])
        return cls(positions(v),filtered,quaternions,alpha,cutoff)

    def contains(self,queries,chunk=256):
        points=np.asarray(queries,dtype='float64')
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Invalid support queries')
        answer=np.zeros(len(points),dtype=bool)
        for ids,tree,radius in self.bins:
            unresolved=np.flatnonzero(~answer)
            if not len(unresolved):break
            distance,_=tree.query(points[unresolved],k=1,workers=4 if len(unresolved)>4096 else 1)
            unresolved=unresolved[distance<=radius*(1+1e-12)]
            for begin in range(0,len(unresolved),chunk):
                selected=unresolved[begin:begin+chunk]
                neighbors=tree.query_ball_point(points[selected],r=radius*(1+1e-12),workers=1)
                sizes=np.fromiter(map(len,neighbors),dtype='int64',count=len(neighbors))
                if not sizes.sum():continue
                which=np.repeat(np.arange(len(selected)),sizes)
                local=np.concatenate(neighbors).astype('int64');gaussians=ids[local]
                delta=points[selected[which]]-self.centers[gaussians]
                # R maps principal axes to world; row vectors multiply by R.
                d=np.einsum('ni,nij->nj',delta,self.rotations[gaussians])/self.scales[gaussians]
                inside=np.sum(d*d,axis=1)<=self.threshold[gaussians]*(1+1e-10)+1e-12
                answer[selected[which[inside]]]=True
        return answer
