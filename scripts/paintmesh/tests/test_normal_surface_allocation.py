"""Full surface bands preserve measured triangle footprints without volume fill."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.normal_surface_allocation import triangle_boxes,frame_surface_blocks,allocate
from mesh_benchmark.full_surface_fusion import enumerate_boxes
from mesh_benchmark.fine_tsdf import unique_rows,contains_rows,blocks_from_depth


def test_subdivision_covers_the_entire_slanted_triangle_and_band():
    tri=np.array([[[0,0,.2],[.18,.04,.8],[.04,.16,.6]]])
    coordinates=[]
    for lo,hi in triangle_boxes(tri):coordinates.extend(enumerate_boxes(lo,hi))
    blocks=unique_rows(np.concatenate(coordinates))
    samples=[]
    for u in np.linspace(0,1,81):
        for v in np.linspace(0,1-u,41):samples.append(tri[0,0]*(1-u-v)+tri[0,1]*u+tri[0,2]*v)
    import itertools
    offset=np.array(list(itertools.product([-.0099,.0099],repeat=3)))
    points=(np.asarray(samples)[:,None,:]+offset).reshape(-1,3)
    assert contains_rows(blocks,np.floor(points/.032).astype('int32')).all()
    bbox=unique_rows(np.concatenate(list(enumerate_boxes(np.floor((tri.min(1)-.01)/.032).astype('int32'),np.floor((tri.max(1)+.01)/.032).astype('int32')))))
    assert len(blocks)<len(bbox)/2


def test_allocation_preserves_baseline_and_resumes_without_rewriting(tmp_path):
    from mesh_benchmark.core import identity
    h=w=8;K=np.array([[30.,0,4],[0,30.,4],[0,0,1.]])
    d=np.full((h,w),.2,dtype='float32');path=tmp_path/'frame.npz'
    np.savez(path,depth=d,K=K,w2c=np.eye(4))
    base=blocks_from_depth(d,K,np.eye(4));base=np.r_[base,np.array([[99,99,99]],dtype='int32')]
    baseline=tmp_path/'base.npy';np.save(baseline,base)
    frames=[dict(name='fixed',path=str(path),identity=identity(path))]
    done=allocate(tmp_path/'out',frames,baseline)
    current=np.load(tmp_path/'out/blocks.npy');assert contains_rows(current,base).all()
    stamp=(tmp_path/'out/blocks.npy').stat().st_mtime_ns
    assert allocate(tmp_path/'out',frames,baseline)==done
    assert (tmp_path/'out/blocks.npy').stat().st_mtime_ns==stamp
