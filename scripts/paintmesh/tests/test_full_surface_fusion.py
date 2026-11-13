"""The full-scene allocator must preserve the tested pilot's coverage exactly."""
from pathlib import Path
import sys
import itertools
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.full_surface_fusion import enumerate_boxes,frame_blocks,allocate
from mesh_benchmark.fine_tsdf import unique_rows,blocks_from_depth
from mesh_benchmark.surface_fusion_probe import blocks_for_triangles,triangles_in_roi


def test_batched_boxes_include_their_complete_interiors_and_deduplicate():
    lo=np.array([[0,0,0],[0,0,0],[-2,1,0],[2,3,4]])
    hi=np.array([[5,4,2],[5,4,2],[1,3,5],[1,2,3]])
    expected=[]
    for a,b in zip(lo,hi):
        if np.all(a<=b):expected.extend(itertools.product(*[range(x,y+1) for x,y in zip(a,b)]))
    expected=unique_rows(np.asarray(expected,dtype='int32'))
    got=unique_rows(np.concatenate(list(enumerate_boxes(lo,hi,batch_entries=8))))
    np.testing.assert_array_equal(got,expected)


def test_full_surface_cells_equal_the_pilot_without_roi_clipping():
    yy,xx=np.mgrid[:16,:18];d=(1/(5+.02*xx-.03*yy)).astype('float32')
    d[:,10:]*=2;d[0:2,0:2]=0
    K=np.array([[30.,0,9],[0,30.,8],[0,0,1.]])
    frame=dict(depth=d,K=K,w2c=np.eye(4))
    bounds=np.array([[-10,-10,-10],[10,10,10]])
    expected=blocks_for_triangles(triangles_in_roi(frame,bounds),bounds)
    got=unique_rows(np.concatenate(list(frame_blocks(frame))))
    np.testing.assert_array_equal(got,expected)


def test_full_allocation_checkpoint_reuses_all_original_blocks(tmp_path):
    from mesh_benchmark.core import identity,read
    yy,xx=np.mgrid[:10,:12];d=np.full((10,12),.2,dtype='float32');K=np.array([[30.,0,6],[0,30.,5],[0,0,1.]])
    path=tmp_path/'frame.npz';np.savez(path,depth=d,K=K,w2c=np.eye(4))
    records=[dict(name='one',path=str(path),identity=identity(path))]
    base=blocks_from_depth(d,K,np.eye(4));base=np.concatenate([base,np.array([[100,100,100]],dtype='int32')])
    source=tmp_path/'base.npy';np.save(source,base)
    result=allocate(tmp_path/'allocation',records,source)
    assert result['full_scene'] and result['roi'] is None and len(result['frames'])==1
    from mesh_benchmark.fine_tsdf import contains_rows
    assert contains_rows(np.load(tmp_path/'allocation/blocks.npy'),base).all()
    before=(tmp_path/'allocation/blocks.npy').stat().st_mtime_ns
    again=allocate(tmp_path/'allocation',records,source)
    assert again==result and (tmp_path/'allocation/blocks.npy').stat().st_mtime_ns==before
