"""Exact per-frame surface activation with GPU integer sorting and deduplication.

Geometric decisions stay in the already tested Float64 CPU path: original
pixels, continuity gates, world transforms, conservative triangle subdivision,
and floor-to-block calculations are unchanged. GPUs enumerate every integer
block inside each leaf box and sort exact 63-bit keys. The two GPU workers can
therefore allocate independent frames without repeatedly sorting the entire
growing scene on the CPU. No source surface, block, or pixel is sampled away.
"""
import numpy as np

from .fine_tsdf import VOXEL,BLOCK
from .surface_fusion_probe import continuous_cells,world_grid
from .normal_surface_allocation import triangle_boxes

BITS=21
OFFSET=1<<(BITS-1)
MASK=(1<<BITS)-1
MIN_COORD=-OFFSET
MAX_COORD=OFFSET-1


def _integer_coordinates(coordinates):
    values=np.asarray(coordinates)
    if values.ndim!=2 or values.shape[1]!=3:
        raise ValueError('Block coordinates must have shape N,3')
    if not np.issubdtype(values.dtype,np.integer):
        if not np.isfinite(values).all() or not np.equal(values,np.floor(values)).all():
            raise ValueError('Block coordinates must be finite integers')
    if values.size and (values.min()<MIN_COORD or values.max()>MAX_COORD):
        raise ValueError('Block coordinate exceeds exact signed 21-bit domain')
    return np.ascontiguousarray(values,dtype='int64')


def encode_block_keys(coordinates):
    """One-to-one lexicographic int64 key for checked signed int21 triples."""
    values=_integer_coordinates(coordinates)+OFFSET
    return (values[:,0]<<(2*BITS))|(values[:,1]<<BITS)|values[:,2]


def decode_block_keys(keys):
    values=np.asarray(keys)
    if values.ndim!=1 or not np.issubdtype(values.dtype,np.integer):
        raise ValueError('Block keys must be a one-dimensional integer array')
    if values.size and (values.min()<0 or values.max()>np.iinfo('int64').max):
        raise ValueError('Block keys exceed the nonnegative 63-bit key domain')
    values=np.asarray(values,dtype='int64')
    result=np.stack([values>>(2*BITS),(values>>BITS)&MASK,values&MASK],1)-OFFSET
    return np.asarray(result,dtype='int32')


def _box_keys_gpu(lower,upper,device,max_box_entries=1000000):
    """Yield every covered key. CPU floor decisions are never recomputed on GPU."""
    import torch
    lower=_integer_coordinates(lower);upper=_integer_coordinates(upper)
    valid=(upper>=lower).all(1);lower=lower[valid];upper=upper[valid]
    if not len(lower):return
    # Bounding boxes are much smaller than scene block sets. Deduplicating
    # identical leaf boxes is exact and reduces repeated device enumeration.
    boxes=np.unique(np.c_[lower,upper],axis=0)
    sizes=boxes[:,3:]-boxes[:,:3]+1
    counts=np.prod(sizes,axis=1,dtype='int64')
    if np.any(counts<=0):raise ValueError('Integer block box size overflow')
    start=0
    while start<len(boxes):
        # Whole-box batching is an execution bound, not a spatial bound.
        end=min(start+8192,len(boxes))
        while end>start+1 and int(counts[start:end].sum())>max_box_entries:
            end=start+max(1,(end-start)//2)
        lo=torch.as_tensor(boxes[start:end,:3],device=device,dtype=torch.int64)
        size=torch.as_tensor(sizes[start:end],device=device,dtype=torch.int64)
        count=torch.as_tensor(counts[start:end],device=device,dtype=torch.int64)
        largest=int(counts[start:end].max())
        width=max(1,min(largest,max_box_entries//max(1,end-start)))
        for offset in range(0,largest,width):
            k=torch.arange(offset,min(largest,offset+width),device=device,dtype=torch.int64)[None,:]
            x=k%size[:,0:1]+lo[:,0:1]+OFFSET
            y=(k//size[:,0:1])%size[:,1:2]+lo[:,1:2]+OFFSET
            z=k//(size[:,0:1]*size[:,1:2])+lo[:,2:3]+OFFSET
            keys=(x<<(2*BITS))|(y<<BITS)|z
            yield keys[k<count[:,None]]
        start=end


def iter_frame_boxes(frame):
    """Exactly the source range triangles used by frame_surface_blocks."""
    world=world_grid(frame);valid=continuous_cells(frame['depth'])
    for offsets in [[(0,0),(0,1),(1,0)],[(1,1),(1,0),(0,1)]]:
        triangles=np.stack([world[y:world.shape[0]-1+y,x:world.shape[1]-1+x]
            for y,x in offsets],axis=-2)[valid]
        yield from triangle_boxes(triangles)


def gpu_frame_block_keys(frame,device='cuda:0',max_pending_entries=1000000,
                         merge_entries=4000000,progress=None):
    """Return sorted unique keys for an ENTIRE frame's continuous surface band.

    ``device='cpu'`` is available for exact algorithm tests. Production passes
    ``cuda:0`` through CUDA_VISIBLE_DEVICES, so integer enumeration, sort and
    duplicate removal execute on the assigned physical GPU. The largest
    persistent scene union is intentionally kept out of these frame workers.
    """
    import torch
    if max_pending_entries<1 or merge_entries<1:raise ValueError('Positive execution batch sizes required')
    target=torch.device(device)
    if target.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested for exact surface activation but unavailable')
    pending=[];pending_count=0;merged=[];merged_count=0
    active=torch.empty(0,device=target,dtype=torch.int64)
    boxes=0;sorts=0
    def consume():
        nonlocal pending,pending_count,merged,merged_count,active,sorts
        if not pending:return
        # Unique sorts exact integer keys. No tolerance-based welding occurs.
        keys=torch.unique(torch.cat(pending),sorted=True)
        pending=[];pending_count=0;merged.append(keys);merged_count+=keys.numel();sorts+=1
        if merged_count>=merge_entries:
            active=torch.unique(torch.cat([active,*merged]),sorted=True)
            merged=[];merged_count=0
        if progress:progress(dict(leaf_box_batches=boxes,gpu_integer_sorts=sorts,
            committed_keys=active.numel(),pending_unique_keys=merged_count,device=str(target)))
    with torch.no_grad():
        for lower,upper in iter_frame_boxes(frame):
            boxes+=1
            for keys in _box_keys_gpu(lower,upper,target,max_pending_entries):
                pending.append(keys);pending_count+=keys.numel()
                if pending_count>=max_pending_entries:consume()
        consume()
        if merged:active=torch.unique(torch.cat([active,*merged]),sorted=True)
        result=active.cpu().numpy().copy()
    if target.type=='cuda':torch.cuda.synchronize(target)
    return result


def gpu_frame_blocks(frame,device='cuda:0',max_pending_entries=1000000,
                     merge_entries=4000000,progress=None):
    """Return int32[N,3], lexicographically ordered and exactly deduplicated."""
    keys=gpu_frame_block_keys(frame,device,max_pending_entries,merge_entries,progress)
    return decode_block_keys(keys)
