"""Full-scene TSDF on a fixed fine lattice, partitioned without coarsening.

All original RGB-D pixels and fusion cameras participate. Sparse blocks are
allocated from EVERY valid depth pixel. Every partition integrates every frame;
one-block halos give marching cubes the same neighbors as a monolithic volume.
An original triangle has one owner, determined by its world-space centroid.
Partitioning never changes the voxel size, truncation, image or face density.
"""
from pathlib import Path
import gc
import itertools
import os
import time
import numpy as np
from .core import read,write,identity,digest

VOXEL=.002
TRUNCATION=.008
BLOCK=16
DTYPE=np.dtype([('x','<i4'),('y','<i4'),('z','<i4')])


def row_keys(a):
    return np.ascontiguousarray(a,dtype='<i4').view(DTYPE).reshape(-1)


def unique_rows(a):
    return np.unique(row_keys(a)).view('<i4').reshape(-1,3)


def contains_rows(haystack,needles):
    keys=np.sort(row_keys(haystack));queries=row_keys(needles)
    index=np.searchsorted(keys,queries);inside=index<len(keys)
    inside[inside]&=keys[index[inside]]==queries[inside]
    return inside


def blocks_from_depth(depth,K,T,voxel=VOXEL,truncation=TRUNCATION,block=BLOCK):
    """All valid pixels, with enough neighboring blocks for a complete MC cell."""
    yy,xx=np.nonzero(np.isfinite(depth)&(depth>0));z=depth[yy,xx]
    inverse=np.linalg.inv(np.asarray(K,dtype='float64'));W=np.asarray(T,dtype='float64')
    output=[];margin=truncation+voxel;length=voxel*block
    if 2*margin>=length:raise ValueError('The eight-corner allocation needs margin < half a block')
    offsets=np.array(list(itertools.product([-margin,margin],repeat=3)))
    for start in range(0,len(z),50000):
        stop=start+50000
        p=(np.c_[xx[start:stop],yy[start:stop],np.ones(len(z[start:stop]))]@inverse.T)*z[start:stop,None]
        p=(p-W[:3,3])@W[:3,:3]
        coords=np.floor((p[:,None,:]+offsets)/length).astype('int32').reshape(-1,3)
        output.append(unique_rows(coords))
    return unique_rows(np.concatenate(output)) if output else np.empty((0,3),dtype='int32')


def morton_order(coords):
    value=(coords.astype('int64')-coords.min(0)).astype('uint64')
    if value.max()>=2**21:raise ValueError('Scene exceeds the exact 21-bit Morton domain')
    def spread(x):
        x=(x|(x<<32))&np.uint64(0x1f00000000ffff)
        x=(x|(x<<16))&np.uint64(0x1f0000ff0000ff)
        x=(x|(x<<8))&np.uint64(0x100f00f00f00f00f)
        x=(x|(x<<4))&np.uint64(0x10c30c30c30c30c3)
        return (x|(x<<2))&np.uint64(0x1249249249249249)
    code=spread(value[:,0])|(spread(value[:,1])<<1)|(spread(value[:,2])<<2)
    return np.argsort(code,kind='stable')


def partitions(coords,core_blocks=6000,max_blocks=24000):
    ordered=coords[morton_order(coords)];keys=np.sort(row_keys(coords))
    offsets=np.array(list(itertools.product([-1,0,1],repeat=3)),dtype='int32')
    start=0
    while start<len(ordered):
        count=min(core_blocks,len(ordered)-start)
        while True:
            core=ordered[start:start+count]
            expanded=unique_rows((core[:,None,:]+offsets).reshape(-1,3))
            queries=row_keys(expanded);index=np.searchsorted(keys,queries);inside=index<len(keys)
            inside[inside]&=keys[index[inside]]==queries[inside]
            halo=expanded[inside]
            if len(halo)<=max_blocks:break
            if count==1:raise RuntimeError('Partition capacity cannot hold one block and its halo')
            count=max(1,count//2)
        yield core,halo
        start+=count


def retain_owned_faces(vertices,faces,core,voxel=VOXEL,block=BLOCK):
    center=np.asarray(vertices,dtype='float64')[faces].mean(1)
    owners=np.floor(center/(voxel*block)).astype('int32')
    return contains_rows(core,owners)


def discover(folder,records):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    request=dict(frames=[identity(r['path']) for r in records],voxel=VOXEL,truncation=TRUNCATION,
                 block_resolution=BLOCK,allocation='Every valid input pixel; truncation + MC-cell margin')
    marker=folder/'allocation.json';path=folder/'blocks.npy'
    if marker.exists():
        old=read(marker)
        if old['request']!=request:raise ValueError('Fine allocation inputs changed')
        return np.load(path),old
    active=np.empty((0,3),dtype='int32');pixels=0
    for i,record in enumerate(records):
        with np.load(record['path']) as f:
            d=f['depth'];coords=blocks_from_depth(d,f['K'],f['w2c'])
            pixels+=int((np.isfinite(d)&(d>0)).sum())
        active=unique_rows(np.concatenate([active,coords]))
        print('FINE_ALLOCATE',i+1,len(records),'blocks',len(active),'valid_pixels',pixels,flush=True)
    np.save(path,active)
    result=dict(complete=True,request=request,request_id=digest(request),blocks=len(active),valid_pixels=pixels,
        monolithic_attributes_GiB=len(active)*BLOCK**3*20/1024**3,
        quality=dict(voxel=VOXEL,sdf_trunc=TRUNCATION,image_downsampling=False,pixel_stride=1,views_dropped=0))
    write(marker,result);return active,result


def integrate_partition(active,frames,device='CUDA:0'):
    """Native Open3D integration, without modifying the full-frame RGB-D."""
    import open3d as o3d
    dev=o3d.core.Device(device)
    volume=o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),
        attr_dtypes=(o3d.core.float32,o3d.core.float32,o3d.core.float32),attr_channels=((1,),(1,),(3,)),
        voxel_size=VOXEL,block_resolution=BLOCK,block_count=max(8,len(active)*2),device=dev)
    block_coords=o3d.core.Tensor(active,o3d.core.int32,dev)
    for f in frames:
        d=o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(f['depth']),device=dev))
        rgb=o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(f['rgb'],dtype='float32')/255.,device=dev))
        K=o3d.core.Tensor(f['K'].astype('float64'));T=o3d.core.Tensor(f['w2c'].astype('float64'))
        volume.integrate(block_coords,d,rgb,K,T,depth_scale=1.,depth_max=1e6,trunc_voxel_multiplier=4.)
    mesh=volume.extract_triangle_mesh(weight_threshold=0.)
    if 'positions' not in mesh.vertex or 'indices' not in mesh.triangle or len(mesh.vertex.positions)==0:
        vertices=np.empty((0,3),dtype='float32');colors=vertices.copy();faces=np.empty((0,3),dtype='int64')
    else:
        vertices=mesh.vertex.positions.cpu().numpy().copy();colors=mesh.vertex.colors.cpu().numpy().copy()
        faces=mesh.triangle.indices.cpu().numpy().copy()
    del mesh,volume,block_coords;gc.collect()
    if device.startswith('CUDA'):o3d.core.cuda.synchronize()
    return vertices,faces,colors


def fuse(folder,records,device='CUDA:0'):
    import open3d as o3d
    from recompose.io import fingerprint
    folder=Path(folder);coords,allocation=discover(folder,records)
    parts=folder/'parts';parts.mkdir(exist_ok=True)
    request=dict(allocation=allocation['request_id'],voxel=VOXEL,sdf_trunc=TRUNCATION,block=BLOCK,
        core_blocks=6000,max_active_blocks=24000,halo_blocks=1,weight_threshold=0.,
        implementation=identity(Path(__file__)),open3d=o3d.__version__)
    marker=folder/'complete.json'
    if marker.exists():
        previous=read(marker)
        if previous['request']!=request:raise ValueError('Fine TSDF request changed')
        return previous
    frames=[]
    for record in records:
        with np.load(record['path']) as data:
            frames.append({k:data[k].copy() for k in ['depth','rgb','K','w2c']})
    result=[];total_faces=0;total_vertices=0;started=time.time()
    for i,(core,active) in enumerate(partitions(coords)):
        destination=parts/f'{i:05d}.mesh.json';receipt=parts/f'{i:05d}.json'
        signature=digest(dict(request=request,core=core.tolist(),active=active.tolist()))
        if receipt.exists():
            old=read(receipt)
            if old['signature']!=signature:raise ValueError('Partition inputs changed')
            if old['faces'] and old['mesh_identity']!=identity(destination):raise ValueError('Fine partition changed')
            result.append(old);total_faces+=old['faces'];total_vertices+=old['vertices'];continue
        # Disk exhaustion pauses the worker; fidelity is never modified.
        import shutil
        while shutil.disk_usage(folder).free<24*1024**3:
            print('FINE_WAIT_DISK',str(folder),flush=True);time.sleep(60)
        print('FINE_PART_START',i,'core',len(core),'with_halo',len(active),flush=True)
        vertices,faces,colors=integrate_partition(active,frames,device)
        keep=retain_owned_faces(vertices,faces,core);faces=faces[keep]
        if len(faces):
            used,inverse=np.unique(faces,return_inverse=True)
            vertices=vertices[used];colors=colors[used];faces=inverse.reshape(-1,3)
            if colors.max()>1.0001:raise ValueError('Unexpected VBG color range')
            # Native glTF-compatible buffers are also the renderer input.
            # Sharing them avoids a second multi-terabyte mesh copy. This is
            # lossless storage/layout, not a different mesh or a decimation.
            packed=np.empty(len(vertices),dtype=[('position','<f4',(3,)),('color','u1',(4,))])
            packed['position']=vertices;packed['color'][:,:3]=np.rint(np.clip(colors,0,1)*255).astype('uint8');packed['color'][:,3]=255
            vp=parts/f'{i:05d}.vertices.npy';fp=parts/f'{i:05d}.indices.npy'
            np.save(vp,packed);np.save(fp,faces.astype('<u4'))
            write(destination,dict(format='paintmesh-native-triangles-v1',vertices=fingerprint(vp),indices=fingerprint(fp),
                vertex_identity=identity(vp),index_identity=identity(fp),vertex_count=len(vertices),face_count=len(faces),
                bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],vertex_stride=16,
                vertex_color='native TSDF RGB8, opaque RGBA',geometry_simplified=False))
            del packed
        row=dict(complete=True,signature=signature,partition=i,core_blocks=len(core),active_blocks=len(active),
            vertices=len(vertices) if len(faces) else 0,faces=len(faces),
            mesh_identity=identity(destination) if len(faces) else None)
        write(receipt,row);result.append(row);total_faces+=row['faces'];total_vertices+=row['vertices']
        print('FINE_PART_COMPLETE',i,row['vertices'],row['faces'],'total_faces',total_faces,'seconds',time.time()-started,flush=True)
        del vertices,colors,faces;gc.collect()
    result=dict(complete=True,request=request,allocation=allocation,parts=result,vertices=total_vertices,faces=total_faces,
        elapsed=time.time()-started,mesh_specs=[dict(path=r['mesh_identity']['path']) for r in result if r['faces']],
        geometry_simplified=False,connected_components_removed=False,images_downsampled=False)
    write(marker,result);return result
