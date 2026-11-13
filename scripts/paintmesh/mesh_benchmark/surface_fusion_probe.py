"""Isolated surface-footprint experiment; production fusion is not modified.

World coordinates, pixels, input RGB, voxel size, truncation and unit weights
are identical to fine_tsdf. Only active-block coverage and continuous depth
sampling are experimental. Local continuity never assumes a global plane.
"""
import gc
import numpy as np
from .fine_tsdf import VOXEL,TRUNCATION,BLOCK,unique_rows


def continuous_cells(depth,relative_span=.10,inverse_cross=.005):
    q=np.stack([depth[:-1,:-1],depth[:-1,1:],depth[1:,:-1],depth[1:,1:]],-1)
    good=np.isfinite(q).all(-1)&(q>0).all(-1)
    inv=np.divide(1.,q,out=np.zeros_like(q),where=np.isfinite(q)&(q>0))
    good&=(q.max(-1)-q.min(-1))<=relative_span*np.maximum(q.mean(-1),1e-12)
    good&=np.abs(inv[...,0]+inv[...,3]-inv[...,1]-inv[...,2])<=inverse_cross*np.maximum(inv.mean(-1),1e-12)
    return good


def world_grid(frame):
    depth=frame['depth'];yy,xx=np.mgrid[:depth.shape[0],:depth.shape[1]]
    camera=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(frame['K']).T*depth[...,None]
    T=frame['w2c'];return (camera-T[:3,3])@T[:3,:3]


def triangles_in_roi(frame,roi):
    """Adjacent valid range samples form local, possibly curved surfaces."""
    grid=world_grid(frame);valid=continuous_cells(frame['depth'])
    corners=np.stack([grid[:-1,:-1],grid[:-1,1:],grid[1:,:-1],grid[1:,1:]],-2)
    valid&=(corners.max(-2)>=roi[0]).all(-1)&(corners.min(-2)<=roi[1]).all(-1)
    q=corners[valid]
    return np.concatenate([q[:,[0,1,2]],q[:,[3,2,1]]])


def blocks_for_triangles(triangles,roi,margin=TRUNCATION+VOXEL):
    """Enumerate ALL blocks of each expanded triangle AABB, including its interior.

    Conservative extra blocks change only allocation, never geometry values.
    Batches control RAM; no triangle or intermediate block is sampled away.
    """
    length=VOXEL*BLOCK
    lower=np.floor(np.maximum(triangles.min(1)-margin,roi[0])/length).astype('int32')
    upper=np.floor(np.minimum(triangles.max(1)+margin,roi[1])/length).astype('int32')
    keep=(upper>=lower).all(1);lower=lower[keep];sizes=upper[keep]-lower+1
    result=[]
    for start in range(0,len(lower),64):
        lo=lower[start:start+64];size=sizes[start:start+64];count=np.prod(size.astype('int64'),axis=1)
        # Bound temporary arrays without bounding the spatial footprint.
        for offset in range(0,int(count.max(initial=0)),32768):
            k=np.arange(offset,min(int(count.max()),offset+32768))[None,:]
            x=k%size[:,0:1];y=(k//size[:,0:1])%size[:,1:2];z=k//(size[:,0:1]*size[:,1:2])
            xyz=np.stack(np.broadcast_arrays(x,y,z),-1)+lo[:,None,:]
            result.append(unique_rows(xyz[k<count[:,None]]))
    return unique_rows(np.concatenate(result)) if result else np.empty((0,3),dtype='int32')


def expanded_blocks(frames,base,roi):
    length=VOXEL*BLOCK;expanded=np.asarray(roi)+np.array([[-2*length]*3,[2*length]*3])
    coordinates=base.copy();records=[]
    for i,f in enumerate(frames):
        tri=triangles_in_roi(f,expanded)
        if len(tri):coordinates=unique_rows(np.concatenate([coordinates,blocks_for_triangles(tri,expanded)]))
        records.append(dict(frame=i,continuous_triangles=len(tri),union_blocks=len(coordinates)))
        print('PROBE_FOOTPRINT',i+1,len(frames),'triangles',len(tri),'blocks',len(coordinates),flush=True)
    return coordinates,records


def integrate_sampled(active,frames,device='CPU:0',interpolation='inverse',chunk_blocks=64):
    """Open3D-compatible TSDF updates with an optional continuous depth lookup.

    A nearest mode is provided solely to test the adapter against Open3D.
    RGB lookup, equal observation weights and the .008 band remain unchanged.
    """
    import torch,open3d as o3d
    if interpolation not in ['nearest','inverse']:raise ValueError(interpolation)
    dev=o3d.core.Device(device);torch_device='cuda:0' if device.startswith('CUDA') else 'cpu'
    volume=o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),
        attr_dtypes=(o3d.core.float32,)*3,attr_channels=((1,),(1,),(3,)),voxel_size=VOXEL,
        block_resolution=BLOCK,block_count=max(8,len(active)*2),device=dev)
    coords=o3d.core.Tensor(np.ascontiguousarray(active),o3d.core.int32,dev)
    volume.hashmap().activate(coords);buffer_indices,_=volume.hashmap().find(coords)
    shared=lambda a:torch.utils.dlpack.from_dlpack(a.to_dlpack())
    tsdf=shared(volume.attribute('tsdf')).reshape(-1)
    weight=shared(volume.attribute('weight')).reshape(-1)
    color=shared(volume.attribute('color')).reshape(-1,3)
    tensors=[]
    for f in frames:
        tensors.append(dict(depth=torch.tensor(f['depth'],device=torch_device),
            rgb=torch.tensor(f['rgb'],device=torch_device),
            continuity=torch.tensor(continuous_cells(f['depth']),device=torch_device),
            R=torch.tensor(f['w2c'][:3,:3],dtype=torch.float32,device=torch_device),
            t=torch.tensor(f['w2c'][:3,3],dtype=torch.float32,device=torch_device),K=f['K']))
    with torch.no_grad():
        for start in range(0,len(active),chunk_blocks):
            position,index=volume.voxel_coordinates_and_flattened_indices(buffer_indices[start:start+chunk_blocks])
            xyz=shared(position);idx=shared(index).long()
            value=torch.zeros(len(idx),device=torch_device);weights=torch.zeros_like(value)
            colors=torch.zeros((len(idx),3),device=torch_device)
            for f in tensors:
                q=xyz@f['R'].T+f['t'];z=q[:,2];K=f['K'];h,w=f['depth'].shape
                u=q[:,0]*float(K[0,0])/z+float(K[0,2]);v=q[:,1]*float(K[1,1])/z+float(K[1,2])
                valid=(z>0)&(u>=0)&(u<w)&(v>=0)&(v<h)&torch.isfinite(u)&torch.isfinite(v)
                x=torch.floor(u).long().clamp(0,w-1);y=torch.floor(v).long().clamp(0,h-1)
                depth=f['depth'][y,x]
                if interpolation=='inverse':
                    cx=x.clamp(0,w-2);cy=y.clamp(0,h-2)
                    inside=valid&(x<w-1)&(y<h-1)&f['continuity'][cy,cx]
                    a=(u-cx).clamp(0,1);b=(v-cy).clamp(0,1)
                    rho=((1-a)*(1-b)/f['depth'][cy,cx].clamp_min(1e-12)+
                         a*(1-b)/f['depth'][cy,cx+1].clamp_min(1e-12)+
                         (1-a)*b/f['depth'][cy+1,cx].clamp_min(1e-12)+
                         a*b/f['depth'][cy+1,cx+1].clamp_min(1e-12))
                    depth=torch.where(inside,1/rho.clamp_min(1e-12),depth)
                signed=depth-z;valid&=(depth>0)&torch.isfinite(depth)&(signed>=-TRUNCATION)
                increment=valid.float();next_weight=weights+increment
                denominator=next_weight.clamp_min(1)
                value=(weights*value+torch.where(valid,(signed/TRUNCATION).clamp_max(1),0))/denominator
                sample=f['rgb'][y,x].float()
                colors=(weights[:,None]*colors+sample*increment[:,None])/denominator[:,None]
                weights=next_weight
            tsdf[idx]=value;weight[idx]=weights;color[idx]=colors
            print('PROBE_INTEGRATE',interpolation,min(start+chunk_blocks,len(active)),len(active),flush=True)
            del position,index,xyz,idx,value,weights,colors,q,z,u,v,x,y,depth,signed,valid,sample
    del tensors,tsdf,weight,color
    gc.collect()
    if device.startswith('CUDA'):torch.cuda.empty_cache()
    mesh=volume.extract_triangle_mesh(weight_threshold=0.)
    if 'positions' not in mesh.vertex or len(mesh.vertex.positions)==0:
        result=(np.empty((0,3),dtype='float32'),np.empty((0,3),dtype='int64'),np.empty((0,3),dtype='float32'))
    else:result=tuple(a.cpu().numpy().copy() for a in [mesh.vertex.positions,mesh.triangle.indices,mesh.vertex.colors])
    del mesh,volume,coords,buffer_indices;gc.collect()
    if device.startswith('CUDA'):o3d.core.cuda.synchronize();o3d.core.cuda.release_cache()
    return result


def crop_mesh(vertices,faces,colors,roi):
    if not len(faces):return vertices,faces,colors
    center=vertices[faces].mean(1);keep=((center>=roi[0])&(center<=roi[1])).all(1)
    used,indices=np.unique(faces[keep],return_inverse=True)
    return vertices[used],indices.reshape(-1,3),colors[used]


def topology(vertices,faces,roi):
    import open3d as o3d
    if not len(faces):return dict(vertices=0,faces=0,components=0)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
    _,counts,areas=mesh.cluster_connected_triangles();counts=np.asarray(counts)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    edges,incidence=np.unique(edges,axis=0,return_counts=True)
    boundary=edges[incidence==1];p=vertices[boundary]
    interior=((p>roi[0]+3*VOXEL)&(p<roi[1]-3*VOXEL)).all((1,2))
    largest=int(counts.max(initial=0))
    return dict(vertices=len(vertices),faces=len(faces),components=len(counts),
        tiny_components_under_50_faces=int((counts<50).sum()),largest_component_faces=largest,
        largest_component_fraction=largest/len(faces),boundary_edges=len(boundary),
        internal_boundary_edges=int(interior.sum()),
        internal_boundary_length=float(np.linalg.norm(p[interior,0]-p[interior,1],axis=1).sum()))
