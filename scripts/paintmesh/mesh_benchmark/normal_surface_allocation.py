"""Conservative full-scene surface coverage without filling huge triangle boxes.

Only activation geometry is subdivided, not an output mesh. Every continuous
source range triangle is covered in full. Splitting long triangles before
enumerating boxes avoids activating the empty volume of a grazing triangle's
large three-dimensional AABB. No pixels, triangles or required blocks are lost.
"""
from pathlib import Path
import gc
import numpy as np
from .core import read,write,identity,digest
from .fine_tsdf import VOXEL,TRUNCATION,BLOCK,unique_rows
from .surface_fusion_probe import continuous_cells,world_grid
from .full_surface_fusion import enumerate_boxes


def triangle_boxes(triangles,max_edge=VOXEL*BLOCK,batch=8192):
    margin=TRUNCATION+VOXEL;length=VOXEL*BLOCK
    for start in range(0,len(triangles),batch):
        stack=[np.asarray(triangles[start:start+batch],dtype='float64')]
        while stack:
            q=stack.pop()
            edges=np.stack([q[:,1]-q[:,0],q[:,2]-q[:,1],q[:,0]-q[:,2]],1)
            lengths=np.max(np.abs(edges),2);longest=lengths.argmax(1)
            done=lengths.max(1)<=max_edge
            if done.any():
                leaf=q[done]
                yield (np.floor((leaf.min(1)-margin)/length).astype('int32'),
                       np.floor((leaf.max(1)+margin)/length).astype('int32'))
            q=q[~done];longest=longest[~done]
            if not len(q):continue
            split=[]
            for edge in range(3):
                selected=q[longest==edge]
                if not len(selected):continue
                a,b,c=edge,(edge+1)%3,(edge+2)%3
                midpoint=(selected[:,a]+selected[:,b])/2
                split.extend([np.stack([selected[:,a],midpoint,selected[:,c]],1),
                              np.stack([midpoint,selected[:,b],selected[:,c]],1)])
            children=np.concatenate(split)
            for i in range(0,len(children),batch):stack.append(children[i:i+batch])


def frame_surface_blocks(frame):
    grid=world_grid(frame);valid=continuous_cells(frame['depth'])
    boxes=[];count=0
    for offsets in [[(0,0),(0,1),(1,0)],[(1,1),(1,0),(0,1)]]:
        tri=np.stack([grid[y:grid.shape[0]-1+y,x:grid.shape[1]-1+x] for y,x in offsets],axis=-2)[valid]
        for lo,hi in triangle_boxes(tri):
            boxes.append(np.c_[lo,hi]);count+=len(lo)
            if count>=100000:
                bounds=np.unique(np.concatenate(boxes),axis=0)
                yield from enumerate_boxes(bounds[:,:3],bounds[:,3:])
                boxes=[];count=0
    if boxes:
        bounds=np.unique(np.concatenate(boxes),axis=0)
        yield from enumerate_boxes(bounds[:,:3],bounds[:,3:])


def allocate(folder,records,baseline_blocks,status_callback=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    request=dict(frames=[identity(r['path']) for r in records],base=identity(baseline_blocks),
        voxel=VOXEL,sdf_trunc=TRUNCATION,block=BLOCK,all_pixels=True,all_frames=True,
        implementation=identity(Path(__file__)),
        rule='Full continuous range triangles; conservative fine surface band; no ROI or spatial clipping')
    marker=folder/'complete.json';file=folder/'blocks.npy'
    if marker.exists():
        previous=read(marker)
        if previous['request']!=request:raise ValueError('Normal-experiment allocation request changed')
        if identity(file)!=previous['block_identity']:raise ValueError('Allocated blocks changed')
        return previous
    checkpoint=folder/'checkpoint.json';rows=[];completed=0
    if checkpoint.exists():
        previous=read(checkpoint)
        if previous['request']!=request:raise ValueError('Partial activation changed')
        rows=previous['frames'];completed=previous['completed_frames'];active=np.load(file)
    else:active=np.load(baseline_blocks).copy()
    for i,r in enumerate(records):
        if i<completed:continue
        if r.get('identity') is not None and identity(r['path'])!=r['identity']:raise ValueError('Range input changed')
        with np.load(r['path']) as a:frame={k:a[k].copy() for k in ['depth','K','w2c']}
        before=len(active);batches=0
        # Accumulate several small batches before a global exact integer union.
        pending=[];pending_count=0
        for extra in frame_surface_blocks(frame):
            pending.append(extra);pending_count+=len(extra)
            if pending_count<4000000:continue
            active=unique_rows(np.concatenate([active,*pending]));pending=[];pending_count=0;batches+=1
            row=dict(frame=i+1,total_frames=len(records),blocks=len(active),batch=batches)
            write(folder/'runtime.json',dict(updated=__import__('time').time(),**row))
            if status_callback:status_callback(row)
            print('NORMAL_SURFACE_ALLOCATION',row,flush=True)
        if pending:active=unique_rows(np.concatenate([active,*pending]))
        rows.append(dict(name=r['name'],blocks=len(active),added=len(active)-before))
        temp=file.with_suffix('.tmp.npy');np.save(temp,active);temp.replace(file)
        write(checkpoint,dict(request=request,completed_frames=i+1,frames=rows))
        if status_callback:status_callback(dict(frame=i+1,total_frames=len(records),blocks=len(active),frame_complete=True))
        print('NORMAL_SURFACE_FRAME_COMPLETE',i+1,len(records),'blocks',len(active),flush=True)
        del frame;gc.collect()
    result=dict(complete=True,request=request,request_id=digest(request),blocks=len(active),
        block_identity=identity(file),frames=rows,full_scene=True,roi=None)
    write(marker,result);return result
