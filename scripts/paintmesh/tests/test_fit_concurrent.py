"""Concurrent admission must ignore foreign processes but retain memory checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_fit_concurrent import admission, running_decision


def test_active_foreign_job_does_not_block_fit():
    snapshot = dict(free_mib=13000, utilization=99, processes=[dict(pid=7, memory_mib=11000)])
    assert admission(snapshot, 'inpaint_5_5') is None
    assert admission(snapshot, 'evaluate_5000') is None
    assert running_decision(snapshot, {99}, running=True) is None


def test_low_memory_blocks_admission_and_running_own_job():
    snapshot = dict(free_mib=3000, utilization=0, processes=[])
    assert admission(snapshot, 'inpaint_5_5') is not None
    snapshot['free_mib'] = 1000
    assert running_decision(snapshot, {99}, running=True) == 'memory_risk'


def test_own_limit_does_not_count_other_job_memory():
    snapshot = dict(free_mib=1600, processes=[dict(pid=7, memory_mib=22000),dict(pid=99,memory_mib=100)])
    assert running_decision(snapshot, {99}, running=True) is None
    snapshot['processes'][1]['memory_mib'] = 22500
    assert running_decision(snapshot, {99}, running=True) == 'memory_risk'
