"""Frozen PGSR observations refine actual TSDF vertices without changing faces.

The data term is a frozen visibility-aware point/plane correspondence quadratic,
not a differentiable rasterization loss.  Triangle normals are differentiated
through their actual three vertex positions.  Boundary vertices can be frozen
to keep independent original TSDF partitions closed at precisely the same seam.
Every original vertex, face and camera is retained. No smoothing is applied.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np


_CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void teacher_kernel(const float* vertices, const float* reference,
 const float* depth, const float* normals, const float* confidence,
 const bool* continuous, const float* poses, const float* intrinsic,
 const int64_t* offsets, const int64_t* sizes, float* output,
 int64_t count, int views, float band) {
 int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=count) return;
 float px=vertices[i*3],py=vertices[i*3+1],pz=vertices[i*3+2];
 float h00=0,h01=0,h02=0,h11=0,h12=0,h22=0,g0=0,g1=0,g2=0;
 float n0=0,n1=0,n2=0,sum=0,error=0,observations=0;
 for(int v=0;v<views;v++) {
  const float* P=poses+v*12; const float* K=intrinsic+v*4;
  float qx=P[0]*px+P[1]*py+P[2]*pz+P[3];
  float qy=P[4]*px+P[5]*py+P[6]*pz+P[7];
  float qz=P[8]*px+P[9]*py+P[10]*pz+P[11];
  if(!(qz>0)) continue;
  float u=K[0]*qx/qz+K[2], vv=K[1]*qy/qz+K[3];
  int H=sizes[v*2],W=sizes[v*2+1];
  if(!(u>=0 && vv>=0 && u<W && vv<H)) continue;
  int x=(int)floorf(u),y=(int)floorf(vv);
  int64_t ix=offsets[v]+(int64_t)y*W+x;
  float d=depth[ix], cf=confidence[ix];
  if(!(d>0 && cf>0)) continue;
  float nx=normals[ix*3],ny=normals[ix*3+1],nz=normals[ix*3+2];
  if(x<W-1 && y<H-1 && continuous[ix]) {
   float a=u-x,b=vv-y;
   float ws[4]={(1-a)*(1-b),a*(1-b),(1-a)*b,a*b};
   int64_t ids[4]={ix,ix+1,ix+W,ix+W+1};
   float rho=0; nx=0;ny=0;nz=0;
   for(int k=0;k<4;k++) {
    rho+=ws[k]/fmaxf(depth[ids[k]],1e-12f);
    nx+=ws[k]*normals[ids[k]*3];
    ny+=ws[k]*normals[ids[k]*3+1];
    nz+=ws[k]*normals[ids[k]*3+2];
   }
   d=1/fmaxf(rho,1e-12f);
  }
  float length=sqrtf(nx*nx+ny*ny+nz*nz);
  if(!(length>1e-8f)) continue;
  nx/=length;ny/=length;nz/=length;
  float rx=(u-K[2])/K[0],ry=(vv-K[3])/K[1];
  float facing=nx*rx+ny*ry+nz;
  if(facing>0) {nx=-nx;ny=-ny;nz=-nz;facing=-facing;}
  float incidence=-facing/sqrtf(rx*rx+ry*ry+1);
  if(!(incidence>=.05f)) continue;
  float signed_distance=nx*(qx-rx*d)+ny*(qy-ry*d)+nz*(qz-d);
  if(!(fabsf(signed_distance)<=band)) continue;
  // Camera row normals transform to world by n_camera @ R.
  float wx=nx*P[0]+ny*P[4]+nz*P[8];
  float wy=nx*P[1]+ny*P[5]+nz*P[9];
  float wz=nx*P[2]+ny*P[6]+nz*P[10];
  float ratio=signed_distance/(band*.5f);
  float weight=cf*incidence/(1+ratio*ratio);
  h00+=weight*wx*wx;h01+=weight*wx*wy;h02+=weight*wx*wz;
  h11+=weight*wy*wy;h12+=weight*wy*wz;h22+=weight*wz*wz;
  g0+=weight*signed_distance*wx;g1+=weight*signed_distance*wy;g2+=weight*signed_distance*wz;
  float orientation=wx*reference[i*3]+wy*reference[i*3+1]+wz*reference[i*3+2];
  float sign=orientation<0?-1.f:1.f;
  n0+=weight*wx*sign;n1+=weight*wy*sign;n2+=weight*wz*sign;
  sum+=weight;error+=weight*signed_distance*signed_distance;observations+=1;
 }
 float den=fmaxf(sum,1e-12f);float nl=sqrtf(n0*n0+n1*n1+n2*n2);
 float* o=output+i*15;
 o[0]=h00/den;o[1]=h01/den;o[2]=h02/den;o[3]=h11/den;o[4]=h12/den;o[5]=h22/den;
 o[6]=g0/den;o[7]=g1/den;o[8]=g2/den;
 o[9]=n0/fmaxf(nl,1e-12f);o[10]=n1/fmaxf(nl,1e-12f);o[11]=n2/fmaxf(nl,1e-12f);
 o[12]=sum;o[13]=error/den;o[14]=observations;
}

torch::Tensor query(torch::Tensor vertices,torch::Tensor reference,
 torch::Tensor depth,torch::Tensor normals,torch::Tensor confidence,
 torch::Tensor continuous,torch::Tensor poses,torch::Tensor intrinsic,
 torch::Tensor offsets,torch::Tensor sizes,double band) {
 TORCH_CHECK(vertices.is_cuda() && vertices.scalar_type()==torch::kFloat32 && vertices.is_contiguous(),"CUDA Float32 vertices required");
 TORCH_CHECK(vertices.dim()==2 && vertices.size(1)==3,"N,3 vertices required");
 auto out=torch::empty({vertices.size(0),15},vertices.options());
 if(vertices.size(0)>0) teacher_kernel<<<(vertices.size(0)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
 vertices.data_ptr<float>(),reference.data_ptr<float>(),depth.data_ptr<float>(),normals.data_ptr<float>(),
 confidence.data_ptr<float>(),continuous.data_ptr<bool>(),poses.data_ptr<float>(),intrinsic.data_ptr<float>(),
 offsets.data_ptr<int64_t>(),sizes.data_ptr<int64_t>(),out.data_ptr<float>(),vertices.size(0),sizes.size(0),band);
 C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
'''
_CPP = '''
#include <torch/extension.h>
torch::Tensor query(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double);
'''
_EXTENSION = None


def native_extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load_inline
        _EXTENSION=load_inline(name='paintmesh_tsdf_vertex_teacher_v1',cpp_sources=_CPP,
            cuda_sources=_CUDA,functions=['query'],extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3'],verbose=False)
    return _EXTENSION


def prepare_teacher(frames,device='cuda:0'):
    """Pack every camera once; accepts dict frames or rows containing ``path``."""
    import torch
    from .normal_surface_fusion import _prepare_frames
    rows=[];sizes=[];offsets=[];depth=[];normals=[];weights=[];continuity=[];poses=[];ks=[];offset=0
    for frame in frames:
        if 'path' in frame:
            with np.load(frame['path']) as a: row={k:a[k].copy() for k in a.files}
        else: row=frame
        # Geometry-only callers intentionally do not load image color arrays.
        # The shared normal validator expects an RGB field, unused here.
        if 'rgb' not in row:
            row=dict(row,rgb=np.zeros((*np.asarray(row['depth']).shape,3),dtype='uint8'))
        # Reuse the established PGSR unit-normal/continuity/validity convention.
        f=_prepare_frames([row],'cpu')[0];h,w=f['depth'].shape
        sizes.append([h,w]);offsets.append(offset);offset+=h*w
        depth.append(f['depth'].numpy().reshape(-1));normals.append(f['normal'].numpy().reshape(-1,3))
        weights.append(f['confidence'].numpy().reshape(-1))
        c=np.zeros((h,w),dtype=bool);c[:-1,:-1]=f['continuity'].numpy();continuity.append(c.reshape(-1))
        pose=np.zeros((3,4),dtype='float32');pose[:,:3]=f['R'].numpy();pose[:,3]=f['t'].numpy();poses.append(pose)
        K=f['K'];ks.append([K[0,0],K[1,1],K[0,2],K[1,2]])
        rows.append(dict(height=h,width=w))
    tensor=lambda a,dtype=None:torch.as_tensor(np.ascontiguousarray(a),device=device,dtype=dtype)
    return dict(depth=tensor(np.concatenate(depth)),normal=tensor(np.concatenate(normals)),
        confidence=tensor(np.concatenate(weights)),continuous=tensor(np.concatenate(continuity)),
        poses=tensor(np.asarray(poses,dtype='float32')),intrinsic=tensor(np.asarray(ks,dtype='float32')),
        offsets=tensor(np.asarray(offsets,dtype='int64')),sizes=tensor(np.asarray(sizes,dtype='int64')),
        frames=len(rows),pixels=offset,device=device)


def query_teacher(vertices,reference_normals,teacher,band=.008):
    return native_extension().query(vertices.float().contiguous(),reference_normals.float().contiguous(),
        teacher['depth'],teacher['normal'],teacher['confidence'],teacher['continuous'],teacher['poses'],
        teacher['intrinsic'],teacher['offsets'],teacher['sizes'],band)


def _geometry(vertices,faces,max_displacement,freeze_boundary,chunk_faces):
    """Initial normals and conservative orientation-preserving vertex radii."""
    import torch
    n=len(vertices);normals=torch.zeros_like(vertices);radii=torch.full((n,),max_displacement,device=vertices.device,dtype=torch.float64)
    for start in range(0,len(faces),chunk_faces):
        f=faces[start:start+chunk_faces];p=vertices[f].double();e1=p[:,1]-p[:,0];e2=p[:,2]-p[:,0]
        cross=torch.linalg.cross(e1,e2);area=torch.linalg.vector_norm(cross,dim=-1)
        lengths=torch.stack([torch.linalg.vector_norm(e1,dim=-1),torch.linalg.vector_norm(e2,dim=-1),
            torch.linalg.vector_norm(p[:,2]-p[:,1],dim=-1)],1)
        L=lengths.sum(1)-lengths.min(1).values
        radius=torch.where(L>0,.95*area/(8*L.clamp_min(1e-30)),torch.zeros_like(L))
        for corner in range(3):
            radii.scatter_reduce_(0,f[:,corner],radius,reduce='amin',include_self=True)
            normals.index_add_(0,f[:,corner],cross.to(normals.dtype))
    boundary=torch.zeros(n,dtype=torch.bool,device=vertices.device)
    if freeze_boundary and len(faces):
        # The integer undirected-edge identity is exact for the present mesh.
        edges=torch.cat([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
        small=edges.min(1).values;large=edges.max(1).values
        keys,counts=torch.unique(small*n+large,return_counts=True)
        open_edges=keys[counts!=2];boundary[open_edges//n]=True;boundary[open_edges%n]=True
        radii[boundary]=0
    normal_length=torch.linalg.vector_norm(normals,dim=-1,keepdim=True)
    # Unreferenced vertices belong to no visible mesh surface and stay exact.
    radii[normal_length[:,0]<=1e-15]=0
    normals=normals/normal_length.clamp_min(1e-15)
    return normals,radii,boundary


def refine_mesh(vertices,faces,teacher,iterations=8,band=.008,max_displacement=.008,
                normal_weight=.1,position_weight=.01,freeze_boundary=True,
                chunk_faces=131072,chunk_vertices=262144):
    """Return original-precision positions and a truthful optimization receipt.

    Camera correspondences and their robust weights are evaluated once for all
    original vertices/cameras.  Iterative optimization differentiates actual
    triangle normals. Boundary freezing preserves each original partition seam;
    the output cannot generate new topology or fill existing mesh holes.
    """
    import torch
    if iterations<0 or band<=0 or max_displacement<=0 or normal_weight<0 or position_weight<0:
        raise ValueError('Invalid mesh refinement settings')
    original=np.asarray(vertices);indices=np.asarray(faces)
    if original.ndim!=2 or original.shape[1]!=3 or not np.isfinite(original).all():raise ValueError('Finite N,3 vertices required')
    if indices.ndim!=2 or indices.shape[1]!=3 or (indices.size and (indices.min()<0 or indices.max()>=len(original))):raise ValueError('Valid triangle indices required')
    device=teacher['device'];started=time.time()
    initial=torch.as_tensor(np.ascontiguousarray(original),device=device)
    tri=torch.as_tensor(np.ascontiguousarray(indices),device=device,dtype=torch.long)
    initial_n,radius,boundary=_geometry(initial,tri,max_displacement,freeze_boundary,chunk_faces)
    query_started=time.time();observations=query_teacher(initial,initial_n,teacher,band)
    torch.cuda.synchronize();query_seconds=time.time()-query_started
    supported=observations[:,12]>0;active=supported&(radius>0)
    radius=torch.where(active,radius,0)
    displacement=torch.nn.Parameter(torch.zeros((len(initial),3),device=device,dtype=torch.float32))
    optimizer=torch.optim.Adam([displacement],lr=.2,eps=1e-12)
    target_n=observations[:,9:12].detach();hist=[];normalizer=max(int(active.sum()),1)
    # All losses are accumulated in face/vertex chunks; no whole-mesh graph.
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True);data_value=normal_value=0.;normal_faces=0
        for start in range(0,len(initial),chunk_vertices):
            stop=min(start+chunk_vertices,len(initial));o=observations[start:stop];u=displacement[start:stop]
            d=u*radius[start:stop,None].float();x,y,z=d.unbind(-1)
            quadratic=o[:,0]*x*x+2*o[:,1]*x*y+2*o[:,2]*x*z+o[:,3]*y*y+2*o[:,4]*y*z+o[:,5]*z*z
            linear=2*(o[:,6:9]*d).sum(-1)
            term=((quadratic+linear)/(band*band)+position_weight*(d*d).sum(-1)/(band*band))
            loss=(term*active[start:stop]).sum()/normalizer
            loss.backward();data_value+=float(loss.detach())
        if normal_weight:
            for start in range(0,len(tri),chunk_faces):
                f=tri[start:start+chunk_faces];p=initial[f]+displacement[f].to(initial.dtype)*radius[f,None].to(initial.dtype)
                actual=torch.linalg.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
                actual=actual/torch.linalg.vector_norm(actual,dim=-1,keepdim=True).clamp_min(1e-20)
                desired=target_n[f].sum(1);desired=desired/torch.linalg.vector_norm(desired,dim=-1,keepdim=True).clamp_min(1e-12)
                # Orient target toward each original triangle, preserving winding.
                with torch.no_grad():
                    old=initial[f];before=torch.linalg.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
                    desired=torch.where(((desired*before).sum(-1)<0)[:,None],-desired,desired)
                    valid=supported[f].all(-1)&active[f].any(-1)&torch.isfinite(desired).all(-1)
                    # Sharp/conflicting teacher normals do not impose a false average tangent.
                    agree=(target_n[f]*desired[:,None]).sum(-1).abs().min(-1).values>=.8660254
                    valid&=agree
                term=1-(actual*desired).sum(-1).clamp(-1,1)
                loss=normal_weight*(term*valid).sum()/max(len(tri),1)
                loss.backward();normal_value+=float(loss.detach());normal_faces+=int(valid.sum())
        optimizer.step()
        with torch.no_grad():
            displacement[~active]=0
            size=torch.linalg.vector_norm(displacement,dim=-1,keepdim=True)
            displacement.div_(size.clamp_min(1.))
            # Enforce the radius on ACTUAL output-precision positions.
            proposed=initial+displacement.to(initial.dtype)*radius[:,None].to(initial.dtype)
            move=proposed.double()-initial.double();safe=torch.linalg.vector_norm(move,dim=-1)<=radius
            displacement[~safe]=0
        hist.append(dict(iteration=iteration+1,data_quadratic_change=data_value,normal_loss=normal_value,
            normal_supported_faces=normal_faces,quantized_moves_rejected=int((~safe).sum())))
    with torch.no_grad():
        result=initial+displacement.to(initial.dtype)*radius[:,None].to(initial.dtype)
        distance=torch.linalg.vector_norm(result.double()-initial.double(),dim=-1)
        initial_squared=final_squared=0.;data_vertices=0
        for start in range(0,len(initial),chunk_vertices):
            stop=min(start+chunk_vertices,len(initial));o=observations[start:stop]
            d=(result[start:stop]-initial[start:stop]).float();x,y,z=d.unbind(-1)
            change=o[:,0]*x*x+2*o[:,1]*x*y+2*o[:,2]*x*z+o[:,3]*y*y+2*o[:,4]*y*z+o[:,5]*z*z
            change+=2*(o[:,6:9]*d).sum(-1)
            good=active[start:stop]
            initial_squared+=float(o[good,13].sum())
            final_squared+=float((o[:,13]+change).clamp_min(0)[good].sum());data_vertices+=int(good.sum())
        flips=0;minimum_ratio=1.;normal_error_before=normal_error_after=0.;scored_faces=0
        for start in range(0,len(tri),chunk_faces):
            f=tri[start:start+chunk_faces];a=initial[f].double();b=result[f].double()
            old=torch.linalg.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);new=torch.linalg.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
            square=(old*old).sum(-1);valid=square>1e-30
            ratio=(old*new).sum(-1)/square.clamp_min(1e-30)
            flips+=int((valid&(ratio<=0)).sum())
            if valid.any():minimum_ratio=min(minimum_ratio,float(ratio[valid].min()))
            desired=target_n[f].sum(1);desired=desired/torch.linalg.vector_norm(desired,dim=-1,keepdim=True).clamp_min(1e-12)
            good=valid&supported[f].all(-1)&active[f].any(-1)
            if good.any():
                old=old/torch.linalg.vector_norm(old,dim=-1,keepdim=True).clamp_min(1e-20)
                new=new/torch.linalg.vector_norm(new,dim=-1,keepdim=True).clamp_min(1e-20)
                normal_error_before+=float((1-(old*desired).sum(-1).abs())[good].sum())
                normal_error_after+=float((1-(new*desired).sum(-1).abs())[good].sum());scored_faces+=int(good.sum())
        if flips:raise ValueError('Mesh refinement introduced flipped faces')
        if (distance>radius).any():raise ValueError('Output-precision movement exceeded analytic trust radius')
        receipt=dict(vertices=len(initial),faces=len(tri),iterations=iterations,faces_unchanged=True,
            original_vertex_precision=str(original.dtype),colors_unchanged=True,source_gaussians_updated=False,
            teacher_views=teacher['frames'],teacher_pixels=teacher['pixels'],camera_views_dropped=0,
            full_frame=True,image_downsampling=False,geometry_simplified=False,
            data_term='frozen robust visible PGSR point-plane correspondences; one fused all-camera pass',
            geometry_normal_term='actual cross-product triangle normal differentiated through vertices',
            differentiable_rasterizer_used=False,global_smoothing=False,held_out_targets_used=False,
            supported_vertices=int(supported.sum()),active_vertices=int(active.sum()),
            boundary_vertices_frozen=int(boundary.sum()),freeze_boundary=freeze_boundary,
            maximum_displacement=float(distance.max()) if len(distance) else 0.,moved_vertices=int((distance>0).sum()),
            flipped_faces=flips,minimum_projected_area_ratio=minimum_ratio,
            teacher_query_seconds=query_seconds,elapsed_seconds=time.time()-started,
            frozen_point_plane_rms_before=float(np.sqrt(initial_squared/max(data_vertices,1))),
            frozen_point_plane_rms_after=float(np.sqrt(final_squared/max(data_vertices,1))),
            normal_cosine_error_before=normal_error_before/max(scored_faces,1),
            normal_cosine_error_after=normal_error_after/max(scored_faces,1),normal_scored_faces=scored_faces,
            band=band,normal_weight=normal_weight,position_weight=position_weight,history=hist)
        array=result.cpu().numpy().copy()
    del optimizer,displacement,observations,initial,tri,initial_n,radius,boundary
    return array,receipt
