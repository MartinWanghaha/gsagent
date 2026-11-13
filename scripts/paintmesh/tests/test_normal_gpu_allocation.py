"""GPU activation must exactly match the original full-surface integer union."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.normal_gpu_allocation import (gpu_frame_blocks,gpu_frame_block_keys,
    encode_block_keys,decode_block_keys,MIN_COORD,MAX_COORD,_box_keys_gpu)
from mesh_benchmark.normal_surface_allocation import frame_surface_blocks
from mesh_benchmark.fine_tsdf import unique_rows


def fixture():
    yy,xx=np.mgrid[:15,:18]
    depth=(1/(4+.06*xx-.05*yy)).astype('float32')
    depth[:,11:]*=2;depth[:2,:2]=0
    angle=.37;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=[-.4,.25,-.1]
    return dict(depth=depth,K=np.array([[28.,0,9],[0,29.,7.5],[0,0,1.]]),w2c=T)


def test_signed_integer_keys_are_injective_and_preserve_lexicographic_order():
    rng=np.random.default_rng(7)
    coords=rng.integers(MIN_COORD,MAX_COORD+1,size=(1000,3),dtype='int32')
    coords=np.r_[coords,np.array([[MIN_COORD]*3,[MAX_COORD]*3,[0,-1,1]],dtype='int32')]
    keys=encode_block_keys(coords)
    np.testing.assert_array_equal(decode_block_keys(keys),coords)
    expected=np.unique(coords,axis=0)
    np.testing.assert_array_equal(decode_block_keys(np.unique(keys)),expected)
    assert keys.min()>=0 and keys.max()<=np.iinfo('int64').max
    with pytest.raises(ValueError,match='21-bit'):encode_block_keys([[MAX_COORD+1,0,0]])
    with pytest.raises(ValueError,match='finite integers'):encode_block_keys([[np.nan,0,0]])
    with pytest.raises(ValueError,match='finite integers'):encode_block_keys([[.5,0,0]])
    with pytest.raises(ValueError,match='63-bit'):decode_block_keys(np.array([-1],dtype='int64'))


def test_frame_activation_matches_all_original_boxes_and_is_batch_invariant():
    import torch
    torch.set_num_threads(2)
    frame=fixture();expected=unique_rows(np.concatenate(list(frame_surface_blocks(frame))))
    first=gpu_frame_blocks(frame,device='cpu',max_pending_entries=127,merge_entries=1000)
    second=gpu_frame_blocks(frame,device='cpu',max_pending_entries=100000,merge_entries=400000)
    np.testing.assert_array_equal(first,expected)
    np.testing.assert_array_equal(second,expected)
    np.testing.assert_array_equal(gpu_frame_block_keys(frame,device='cpu'),encode_block_keys(expected))


def test_empty_and_invalid_depths_do_not_allocate_surface():
    frame=fixture();frame['depth'][:]=0
    assert gpu_frame_blocks(frame,device='cpu').shape==(0,3)
    frame['depth'][2,2]=np.nan
    assert gpu_frame_block_keys(frame,device='cpu').shape==(0,)


def test_large_integer_box_is_fully_enumerated_without_truncation():
    import torch,itertools
    lower=np.array([[-3,-4,-2],[0,0,0]],dtype='int32')
    upper=np.array([[4,3,6],[2,1,1]],dtype='int32')
    chunks=list(_box_keys_gpu(lower,upper,torch.device('cpu'),max_box_entries=17))
    got=np.unique(np.concatenate([a.numpy() for a in chunks]))
    expected=[]
    for lo,hi in zip(lower,upper):expected.extend(itertools.product(*[range(a,b+1) for a,b in zip(lo,hi)]))
    np.testing.assert_array_equal(decode_block_keys(got),np.unique(np.asarray(expected),axis=0))


@pytest.mark.skipif(__import__('torch').cuda.is_available() is False,reason='CUDA unavailable')
def test_cuda_integer_sort_matches_cpu_surface():
    import torch
    # Small smoke test only, no production scene allocation or mesh extraction.
    frame=fixture();expected=gpu_frame_blocks(frame,device='cpu')
    actual=gpu_frame_blocks(frame,device='cuda:0',max_pending_entries=101)
    np.testing.assert_array_equal(actual,expected)
