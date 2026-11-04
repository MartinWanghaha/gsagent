"""Geometry-safe RGB transformation and non-preempting resource admission."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from difix_targets import padded_rgb,compose
from run_fit_difix import gpu_decision


def test_padding_round_trip_preserves_camera_pixels():
    a=np.arange(477*737*3,dtype=np.uint8).reshape(477,737,3)
    b=padded_rgb(a)
    assert b.shape==(480,744,3)
    np.testing.assert_array_equal(b[:477,:737],a)


def test_hole_compositing_never_changes_known_background():
    a=np.zeros((12,16,3),np.uint8);b=np.full_like(a,123);mask=np.zeros((12,16),bool);mask[3:7,6:9]=True
    c=compose(a,b,mask,'mask')
    np.testing.assert_array_equal(c[mask],b[mask]);np.testing.assert_array_equal(c[~mask],a[~mask])
    np.testing.assert_array_equal(compose(a,b,np.zeros_like(mask),'mask'),a)
    np.testing.assert_array_equal(compose(a,b,mask,'full'),b)
    with pytest.raises(ValueError):compose(a,b[:5],mask,'mask')


def test_other_job_is_not_reason_to_terminate_ours():
    snapshot=dict(free_mib=5000,processes=[dict(pid=12,memory_mib=10000),dict(pid=99,memory_mib=8000)])
    assert gpu_decision(snapshot,{12},running=True) is None
    snapshot['free_mib']=1000
    assert gpu_decision(snapshot,{12},running=True)=='memory_risk'
