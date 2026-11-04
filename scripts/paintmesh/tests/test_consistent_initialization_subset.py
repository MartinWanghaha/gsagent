"""Protect background and source identities in the opt-in seed rejection ablation."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_consistent_initialization import subset


def data():
    vertex=np.zeros(6,dtype=[(k,'f4') for k in ('x','y','z','opacity','rot_0','custom_feature')])
    for i,k in enumerate(vertex.dtype.names):vertex[k]=np.arange(6)*(i+1)+i*.125
    editable=np.array([False,False,True,True,True,True])
    support=dict(xyz=np.column_stack([vertex[k][editable] for k in 'xyz']),
                 frame=np.arange(4),source_kind=np.array([0,1,0,1]),spacing=np.array(.01))
    return vertex,editable,support


def test_subset_preserves_all_background_fields_and_seed_identity():
    vertex,editable,support=data();before=vertex.tobytes();keep=np.array([False,True,True,False])
    result,mask,observations=subset(vertex,editable,support,keep)
    assert vertex.tobytes()==before
    assert result[:2].tobytes()==vertex[:2].tobytes()
    assert result[2:].tobytes()==vertex[[3,4]].tobytes()
    np.testing.assert_array_equal(mask,[False,False,True,True])
    np.testing.assert_array_equal(observations['frame'],[1,2])
    np.testing.assert_array_equal(observations['xyz'],np.column_stack([result[k][mask] for k in 'xyz']))
    assert observations['spacing']==support['spacing']


def test_empty_filter_or_misaligned_source_is_rejected():
    vertex,editable,support=data()
    with pytest.raises(ValueError,match='No supported'):subset(vertex,editable,support,np.zeros(4,bool))
    support['xyz'][0,0]+=1
    with pytest.raises(ValueError,match='rows differ'):subset(vertex,editable,support,np.ones(4,bool))


def test_interleaved_background_cannot_be_silently_reordered():
    vertex,editable,support=data();editable[[0,2]]=editable[[2,0]]
    with pytest.raises(ValueError,match='unchanged prefix'):subset(vertex,editable,support,np.ones(4,bool))
