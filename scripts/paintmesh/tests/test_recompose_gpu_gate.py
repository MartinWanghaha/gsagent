"""User's strict used-VRAM limit controls launch, independent of total VRAM."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from contextlib import contextmanager
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recompose import pipeline


def test_used_below_five_gib_starts_immediately(monkeypatch):
    run=Mock(return_value=SimpleNamespace(returncode=0,stdout='5119\n'))
    monkeypatch.setattr(pipeline.subprocess,'run',run)
    monkeypatch.setattr(pipeline.time,'sleep',Mock(side_effect=AssertionError('Should launch immediately')))
    pipeline.wait_gpu(1,maximum_used_mib=5120)
    assert '--query-gpu=memory.used' in run.call_args.args[0]


def test_used_exactly_five_gib_waits_and_rechecks(monkeypatch):
    values=[SimpleNamespace(returncode=0,stdout=f'{n}\n') for n in [5120,6000,4251]]
    monkeypatch.setattr(pipeline.subprocess,'run',Mock(side_effect=values))
    sleep=Mock();monkeypatch.setattr(pipeline.time,'sleep',sleep)
    pipeline.wait_gpu(1,maximum_used_mib=5120,timeout=None)
    assert sleep.call_count==2


def test_unreadable_gpu_does_not_launch(monkeypatch):
    monkeypatch.setattr(pipeline.subprocess,'run',Mock(return_value=SimpleNamespace(returncode=1,stdout='')))
    with pytest.raises(RuntimeError,match='used memory must be below'):
        pipeline.wait_gpu(1,maximum_used_mib=5120)


def test_generation_retries_cuda_oom_with_identical_command(tmp_path,monkeypatch):
    @contextmanager
    def slot(*a,**k):yield
    calls=[]
    def run(command,**kwargs):
        calls.append((command,kwargs['env']))
        kwargs['stdout'].write(b'CUDA out of memory\n' if len(calls)==1 else b'complete\n')
        return SimpleNamespace(returncode=1 if len(calls)==1 else 0)
    monkeypatch.setattr(pipeline,'gpu_slot',slot)
    monkeypatch.setattr(pipeline.subprocess,'run',run)
    monkeypatch.setattr(pipeline.time,'sleep',Mock())
    retries=[]
    command=['python','infer.py','--seed','42'];env={'CUDA_VISIBLE_DEVICES':'1'}
    pipeline.generation_with_retry(command,env,tmp_path/'run.log',1,5120,on_oom=retries.append)
    assert calls==[(command,env),(command,env)] and len(retries)==1


def test_old_oom_does_not_hide_new_fatal_error(tmp_path,monkeypatch):
    @contextmanager
    def slot(*a,**k):yield
    log=tmp_path/'run.log';log.write_text('CUDA out of memory in an old attempt\n')
    def run(command,**kwargs):
        kwargs['stdout'].write(b'ValueError: invalid calibrated input\n');return SimpleNamespace(returncode=1)
    monkeypatch.setattr(pipeline,'gpu_slot',slot);monkeypatch.setattr(pipeline.subprocess,'run',run)
    with pytest.raises(pipeline.subprocess.CalledProcessError):
        pipeline.generation_with_retry(['python'],{},log,1,5120)


def test_worker_rechecks_usage_after_imports(tmp_path,monkeypatch):
    from recompose import mvsam_worker
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    values=iter([7000,4251]);sleep=Mock()
    def query(*a,**k):return SimpleNamespace(returncode=0,stdout=str(next(values)))
    monkeypatch.setattr(mvsam_worker.subprocess,'run',query)
    monkeypatch.setattr(mvsam_worker.time,'sleep',sleep)
    mvsam_worker.wait_before_model_allocation(tmp_path/'runtime.json')
    assert sleep.call_count==1
