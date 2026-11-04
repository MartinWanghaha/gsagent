"""A faulty peer GPU must neither block monitoring nor fail unrelated scenes."""
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_reviewed_three_groups as scheduler


def test_nvml_queries_are_card_scoped(monkeypatch):
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        assert cmd[:3]==['nvidia-smi','-i','0']
        text='GPU-good, 23000, 1000, 10' if '--query-gpu=' in cmd[3] else '123, 800'
        return subprocess.CompletedProcess(cmd,0,text,'')
    monkeypatch.setattr(scheduler.subprocess,'run',run)
    result=scheduler.isolated_snapshot(0)
    assert len(calls)==2 and result['free_mib']==23000
    assert result['processes']==[{'pid':123,'memory_mib':800}]


def test_faulty_card_is_unavailable_not_empty(monkeypatch):
    monkeypatch.setattr(scheduler.subprocess,'run',lambda cmd,**kw:subprocess.CompletedProcess(cmd,255,'Unknown Error',''))
    with pytest.raises(scheduler.GPUUnavailable,match='Unknown Error'):
        scheduler.isolated_snapshot(1)


def test_cuda_health_required_and_failures_cached(tmp_path,monkeypatch):
    monkeypatch.setattr(scheduler,'JOB',tmp_path)
    monkeypatch.setattr(scheduler,'isolated_snapshot',lambda index:dict(free_mib=24000,processes=[]))
    calls=[]
    def probe(index):calls.append(index);return False,'cuInit: 3 initialization error'
    monkeypatch.setattr(scheduler,'cuda_probe',probe)
    assert not scheduler.health(0)['healthy']
    assert not scheduler.health(0)['healthy']
    assert calls==[0]


def test_hardware_wait_does_not_claim_or_fail_scenes(monkeypatch):
    monkeypatch.setattr(scheduler,'health',lambda index:dict(healthy=False,reason='Xid 79'))
    reports=[];monkeypatch.setattr(scheduler.guard,'report',lambda root:reports.append(root))
    worker=object.__new__(scheduler.Scheduling);worker.gpu=1;worker.root='job';worker.case=None
    states=[];worker.save=lambda **kwargs:states.append(kwargs)
    worker.process=lambda spec:pytest.fail('No scene can start on a faulty GPU')
    worker.cycle()
    assert states[-1]['state']=='waiting_for_hardware'
    assert worker.case is None and reports==['job']


def test_atomic_peak_reservations_do_not_double_count_used_memory(monkeypatch):
    monkeypatch.setattr(scheduler,'alive',lambda pid:True)
    monkeypatch.setattr(scheduler.guard,'process_group_pids',lambda pid:{pid})
    snap=dict(free_mib=21000,processes=[dict(pid=12,memory_mib=3000)])
    assert scheduler.remaining_capacity(snap,[dict(budget_mib=6000,child_pid=12)])==16464
    assert scheduler.remaining_capacity(dict(free_mib=24000,processes=[]),[dict(budget_mib=16000,child_pid=None)])<12000
