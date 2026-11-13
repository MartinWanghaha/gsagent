"""Complete-frame surface allocation for an isolated bag fusion experiment.

The pilot's conservative triangle AABBs are enumerated exactly. Grouping equal
integer boxes and batching their contents changes storage/work scheduling only;
no input pixel, cell, triangle or covered block is subsampled.
"""
from pathlib import Path
import gc
import numpy as np
from .core import read,write,identity,digest
from .fine_tsdf import VOXEL,TRUNCATION,BLOCK,unique_rows
from .surface_fusion_probe import continuous_cells,world_grid


def enumerate_boxes(lower,upper,batch_entries=1000000):
    lower=np.asarray(lower,dtype='int32');upper=np.asarray(upper,dtype='int32')
    good=(upper>=lower).all(1)
    boxes=np.unique(np.c_[lower[good],upper[good]],axis=0)
    pending=[];pending_count=0
    for start in range(0,len(boxes),4096):
        group=boxes[start:start+4096];size=group[:,3:]-group[:,:3]+1
        counts=np.prod(size.astype('int64'),axis=1)
        for offset in range(0,int(counts.max(initial=0)),128):
            k=np.arange(offset,min(int(counts.max()),offset+128))[None,:]
            x=k%size[:,0:1];y=(k//size[:,0:1])%size[:,1:2];z=k//(size[:,0:1]*size[:,1:2])
            values=np.stack(np.broadcast_arrays(x,y,z),-1)+group[:,None,:3]
            values=values[k<counts[:,None]]
            pending.append(values);pending_count+=len(values)
            if pending_count>=batch_entries:
                yield unique_rows(np.concatenate(pending));pending=[];pending_count=0
    if pending:yield unique_rows(np.concatenate(pending))


def frame_blocks(frame):
    world=world_grid(frame);valid=continuous_cells(frame['depth'])
    length=VOXEL*BLOCK;margin=TRUNCATION+VOXEL
    # The same two triangles/cell as in the tested local probe. Enumerating
    # two AABBs separately avoids an unnecessarily large cell-diagonal box.
    for indices in [[(0,0),(0,1),(1,0)],[(1,1),(1,0),(0,1)]]:
        q=np.stack([world[y:world.shape[0]-1+y,x:world.shape[1]-1+x] for y,x in indices],axis=-2)[valid]
        lower=np.floor((q.min(1)-margin)/length).astype('int32')
        upper=np.floor((q.max(1)+margin)/length).astype('int32')
        yield from enumerate_boxes(lower,upper)


def allocate(folder,records,baseline_blocks):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    request=dict(frames=[identity(r['path']) for r in records],base=identity(baseline_blocks),
        voxel=VOXEL,sdf_trunc=TRUNCATION,all_pixels=True,all_frames=True,
        continuity=dict(relative_span=.10,inverse_cross=.005),implementation=identity(Path(__file__)))
    marker=folder/'complete.json';path=folder/'blocks.npy'
    if marker.exists():
        old=read(marker)
        if old['request']!=request:raise ValueError('Full allocation inputs changed')
        return old
    state=folder/'checkpoint.json';completed=0;rows=[]
    if state.exists():
        old=read(state)
        if old['request']!=request:raise ValueError('Partial allocation inputs changed')
        completed=old['completed_frames'];rows=old['frames'];active=np.load(path)
    else:active=np.load(baseline_blocks).copy()
    for i,record in enumerate(records):
        if i<completed:continue
        with np.load(record['path']) as a:frame={k:a[k] for k in ['depth','K','w2c']}
        before=len(active);batches=0
        for extra in frame_blocks(frame):
            active=unique_rows(np.concatenate([active,extra]));batches+=1
            print('FULL_ALLOCATE_BATCH',i+1,len(records),batches,'blocks',len(active),flush=True)
        row=dict(frame=record['name'],blocks=len(active),added_blocks=len(active)-before)
        rows.append(row)
        temporary=path.with_suffix('.tmp.npy');np.save(temporary,active);temporary.replace(path)
        write(state,dict(request=request,completed_frames=i+1,frames=rows))
        print('FULL_ALLOCATE_FRAME',i+1,len(records),row,flush=True)
        del frame;gc.collect()
    result=dict(complete=True,request=request,request_id=digest(request),blocks=len(active),
        frames=rows,block_file=identity(path),full_scene=True,roi=None)
    write(marker,result);return result


def save_part(folder,index,vertices,faces,colors):
    from recompose.io import fingerprint
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    used,inverse=np.unique(faces,return_inverse=True)
    vertices=vertices[used];colors=colors[used];faces=inverse.reshape(-1,3)
    packed=np.empty(len(vertices),dtype=[('position','<f4',(3,)),('color','u1',(4,))])
    packed['position']=vertices;packed['color'][:,:3]=np.rint(np.clip(colors,0,1)*255).astype('uint8');packed['color'][:,3]=255
    vp=folder/f'{index:05d}.vertices.npy';fp=folder/f'{index:05d}.indices.npy';path=folder/f'{index:05d}.mesh.json'
    np.save(vp,packed);np.save(fp,np.asarray(faces,dtype='<u4'))
    write(path,dict(format='paintmesh-native-triangles-v1',vertices=fingerprint(vp),indices=fingerprint(fp),
        vertex_identity=identity(vp),index_identity=identity(fp),vertex_count=len(vertices),face_count=len(faces),
        bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],vertex_stride=16,
        geometry_simplified=False,vertex_color='Original TSDF RGB8'))
    return dict(path=str(path)),len(vertices),len(faces)
