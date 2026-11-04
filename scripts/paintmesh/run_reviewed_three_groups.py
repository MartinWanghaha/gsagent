#!/usr/bin/env python3
"""One worker per group per GPU, with atomic memory admission between stages."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import run_reviewed_retrack_training as reviewed
import run_fitlama as lama
import run_hard_reference_all as guard
from run_geometry_ablation import REPO, read, write, now, sha, atomic
from run_muti_plan import PYTHON

SCRIPT=Path(__file__).resolve()
BASE=lama.BASE
JOB=REPO/'logs/retrain_three_groups_20261103'
GROUPS=('best_results','fit','fitlama')
RESERVE=1536


class GPUUnavailable(RuntimeError):
    """Hardware/driver availability is not a failed experiment."""


def isolated_snapshot(index):
    # A UUID lookup or an unscoped process query enumerates the faulty peer
    # card on some drivers. Every query must explicitly target a numeric card.
    def query(fields, category):
        out=subprocess.run(['nvidia-smi','-i',str(index),'--query-'+category+'='+fields,
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=12)
        if out.returncode:
            raise GPUUnavailable((out.stdout+' '+out.stderr).strip())
        return out.stdout.strip()
    try:
        fields=[v.strip() for v in query('uuid,memory.free,memory.used,utilization.gpu','gpu').split(',')]
        if len(fields)!=4 or not fields[0].startswith('GPU-'):raise ValueError('Invalid GPU response')
        processes=[]
        for line in query('pid,used_memory','compute-apps').splitlines():
            if not line.strip():continue
            pid,memory=[v.strip() for v in line.split(',')]
            processes.append(dict(pid=int(pid),memory_mib=int(memory)))
        return dict(uuid=fields[0],free_mib=int(fields[1]),used_mib=int(fields[2]),
                    utilization=int(fields[3]),processes=processes)
    except (subprocess.TimeoutExpired,ValueError,OSError) as e:
        raise GPUUnavailable(str(e)) from e


CUDA_PROBE = '''import ctypes
import sys
driver=ctypes.CDLL('libcuda.so.1')
code=driver.cuInit(0)
if code:
    message=ctypes.c_char_p();driver.cuGetErrorString(code,ctypes.byref(message))
    print('cuInit:',code,message.value,flush=True);sys.exit(1)
import torch
x=torch.ones(8,device='cuda');value=(x*x).sum();torch.cuda.synchronize()
assert value.item()==8
print('CUDA allocation and kernel OK',flush=True)
'''


def cuda_probe(index):
    try:
        result=subprocess.run([PYTHON,'-c',CUDA_PROBE],capture_output=True,text=True,timeout=45,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                     PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        return result.returncode==0,(result.stdout+'\n'+result.stderr).strip()[-3000:]
    except (subprocess.TimeoutExpired,OSError) as e:return False,str(e)


def health(index):
    # Cache failed CUDA initialization for 60 seconds, shared by the 3 workers.
    # Recheck NVML every time; a newly fallen card invalidates the success cache.
    with (JOB/f'gpu{index}_health.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);path=JOB/f'gpu{index}_health.json'
        previous=read(path) if path.exists() else {};boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try:snap=isolated_snapshot(index)
        except GPUUnavailable as e:
            value=dict(healthy=False,reason='gpu_query_failed: '+str(e),checked_at=now(),
                       checked_epoch=time.time(),boot_id=boot,gpu=index)
            write(path,value);return value
        age=time.time()-previous.get('checked_epoch',0)
        if previous.get('boot_id')==boot and previous.get('cuda_probed') and age < (120 if previous.get('healthy') else 60):
            return dict(previous,gpu_snapshot=snap)
        okay,message=cuda_probe(index)
        value=dict(healthy=okay,reason=None if okay else 'CUDA initialization/allocation failed',
            probe_output=message,cuda_probed=True,checked_at=now(),checked_epoch=time.time(),
            boot_id=boot,gpu=index,gpu_snapshot=snap)
        write(path,value);return value


def budget(stage):
    if stage in ('official_color','official_depth','normal','inpaint_3_3'):return 6000
    if stage in ('initialize','inpaint_4_4'):return 12000
    if stage in ('train_5000','inpaint_5_5'):return 16000
    if stage=='evaluate_5000':return 5000
    return 10000


def alive(pid):
    try:return bool(Path(f'/proc/{pid}/cmdline').read_bytes())
    except OSError:return False


def remaining_capacity(snap, reservations):
    """Subtract unallocated portions of admitted peaks, not current memory twice."""
    pending=0
    for r in reservations:
        pids=guard.process_group_pids(r['child_pid']) if r.get('child_pid') and alive(r['child_pid']) else set()
        used=sum(p['memory_mib'] for p in snap['processes'] if p['pid'] in pids)
        pending+=max(0,r['budget_mib']-used)
    return snap['free_mib']-pending-RESERVE


class Scheduling:
    def initialize(self, gpu, group, root):
        engine=lama if group=='fitlama' else reviewed
        guard.report=engine.report;guard.check_code=engine.check_code
        guard.GPU_POLICY=engine.POLICY
        guard.gpu_decision=lama.gpu_decision if group=='fitlama' else reviewed.running_decision
        # Do not require a working NVIDIA driver merely to start the supervisor.
        self.root=root;self.gpu=gpu;self.req=read(root/'request.json');self.child=None
        self.uuid=str(gpu);self.case=None;self.state=dict(pid=os.getpid(),gpu=gpu)
        self.group=group
        self.wdir=self.root/'workers'/f'gpu{self.gpu}_{group}'
        self.wdir.mkdir(parents=True,exist_ok=True)
        self.slot=f'{group}_gpu{self.gpu}'
        self.state.update(group=group,slot=self.slot,scheduler=str(JOB))
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
            GPU=str(gpu),PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        self.env.pop('PYTHONPATH',None)
        self.gpu_lock=(BASE/f'.reviewed_retrain_gpu{gpu}.lock').open('a')
        guard.gpu_snapshot=lambda _ignored:isolated_snapshot(self.gpu)

    def ledger(self):return JOB/f'gpu{self.gpu}_reservations.json'

    def locked_entries(self):
        p=self.ledger();d=read(p) if p.exists() else {}
        return {k:r for k,r in d.items() if alive(r['pid'])}

    def save(self, **kw):
        super().save(**kw)
        if not hasattr(self,'slot'):return
        with (JOB/f'gpu{self.gpu}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);d=self.locked_entries()
            if self.slot in d:
                d[self.slot]['child_pid']=self.state.get('child_pid')
                write(self.ledger(),d)
        write(JOB/'workers'/f'{self.slot}.json',dict(self.state,group=self.group,slot=self.slot,root=str(self.root)))

    def release(self):
        with (JOB/f'gpu{self.gpu}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);d=self.locked_entries();d.pop(self.slot,None);write(self.ledger(),d)

    def wait_gpu(self, stage):
        need=budget(stage)
        while True:
            h=health(self.gpu)
            if not h['healthy']:raise GPUUnavailable(h['reason'])
            with (JOB/f'gpu{self.gpu}.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);d=self.locked_entries();d.pop(self.slot,None)
                snap=guard.gpu_snapshot(self.uuid);capacity=remaining_capacity(snap,d.values())
                if capacity>=need:
                    d[self.slot]=dict(pid=os.getpid(),child_pid=None,group=self.group,scene=self.state.get('scene'),
                        stage=stage,budget_mib=need,admitted_at=now())
                    write(self.ledger(),d);return
                write(self.ledger(),d)
            self.save(state='waiting_for_memory',stage=stage,reason='insufficient_unreserved_peak_capacity',
                required_mib=need,unreserved_mib=capacity,reservations=d,gpu_snapshot=snap,child_pid=None)
            guard.report(self.root);time.sleep(15)

    def execute(self,stage,cmd,cwd,expected,gpu=False):
        try:return super().execute(stage,cmd,cwd,expected,gpu=gpu)
        finally:
            if gpu:self.release()

    def has_pending(self):
        for s in self.req['scenes']:
            if self.group!='fitlama' and s['group']!=self.group:continue
            key=s['scene'] if self.group=='fitlama' else s['key']
            if read(self.root/'scenes'/key/'status.json')['state'] not in ('complete','failed','blocked_memory_risk'):
                return True
        return False

    def cycle(self):
        h=health(self.gpu)
        if not h['healthy']:
            self.case=None
            self.save(state='waiting_for_hardware',scene=None,stage=None,reason=h['reason'],
                      health=h,child_pid=None)
            guard.report(self.root);return
        for s in self.req['scenes']:
            if self.group!='fitlama' and s['group']!=self.group:continue
            key=s['scene'] if self.group=='fitlama' else s['key'];case=self.root/'scenes'/key
            with (case/'run.lock').open('a') as sl:
                try:fcntl.flock(sl,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if read(case/'status.json')['state'] in ('complete','failed','blocked_memory_risk'):continue
                try:self.process(s)
                except GPUUnavailable as e:
                    self.save(state='waiting_for_hardware',reason=str(e),child_pid=None)
                    guard.report(self.root);return
                except guard.MemoryRisk as e:
                    self.save(state='blocked_memory_risk',reason=str(e),child_pid=None);traceback.print_exc();guard.report(self.root)
                except Exception as e:
                    # A driver failure during launch must not turn the rest of
                    # the queue into failures. Preserve real model/data errors.
                    h=health(self.gpu)
                    if not h['healthy']:
                        self.save(state='waiting_for_hardware',reason=h['reason'],interrupted_error=str(e),child_pid=None)
                        guard.report(self.root);return
                    self.save(state='failed',reason=str(e),child_pid=None);traceback.print_exc();guard.report(self.root)
        self.case=None
        self.save(state='waiting_for_other_worker',scene=None,stage=None,child_pid=None)

    def run(self):
        with (self.wdir/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN);signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
            while self.has_pending():
                self.cycle()
                if self.has_pending():time.sleep(30)
            self.case=None;self.save(state='queue_finished',scene=None,stage=None,reason=None,child_pid=None);guard.report(self.root)


class ReviewedWorker(Scheduling, reviewed.Worker):
    def __init__(self,gpu,group):
        self.initialize(gpu,group,reviewed.DEFAULT_JOB)


class LamaWorker(Scheduling, lama.Worker):
    def __init__(self,gpu,group):
        self.initialize(gpu,group,BASE/'fitlama')


def launch():
    JOB.mkdir(parents=True,exist_ok=True)
    with (JOB/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reviewed.check_code(read(reviewed.DEFAULT_JOB/'request.json'));lama.check_code(read(BASE/'fitlama/request.json'))
        # Old controllers have no reservation ledger; do not race their next GPU stage.
        for root,script in ((reviewed.DEFAULT_JOB,reviewed.SCRIPT),(BASE/'fitlama',lama.SCRIPT)):
            for state in root.glob('workers/gpu[01]/status.json'):
                d=read(state);pid=d.get('pid');p=Path(f'/proc/{pid}/cmdline')
                if p.exists() and str(script).encode() in p.read_bytes():raise RuntimeError('Stop old controller before handoff: '+str(pid))
        workers={}
        for gpu in (0,1):
            for group in GROUPS:
                slot=f'{group}_gpu{gpu}';state=JOB/'workers'/f'{slot}.json';old=read(state) if state.exists() else {}
                pid=old.get('pid');p=Path(f'/proc/{pid}/cmdline')
                if p.exists() and str(SCRIPT).encode() in p.read_bytes():workers[slot]=dict(pid=pid,state='already_running');continue
                log=JOB/'logs'/f'{slot}.log';log.parent.mkdir(exist_ok=True)
                with log.open('a') as out:
                    child=subprocess.Popen([PYTHON,str(SCRIPT),'--worker',str(gpu),'--group',group],cwd=REPO,
                        stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
                workers[slot]=dict(pid=child.pid,gpu=gpu,group=group,log=str(log))
        write(JOB/'workers.json',workers)
        write(JOB/'scheduler.json',dict(created_at=now(),script=str(SCRIPT),sha256=sha(SCRIPT),groups=list(GROUPS),
            total_tasks=33,workers=6,reserve_mib=RESERVE,budgets_mib={s:budget(s) for s in ('official_color','initialize','train_5000','evaluate_5000','mesh_render')},
            policy='Three group workers per GPU; atomic peak reservations; completed stages retained; never signal unrelated processes'))
        print(workers,flush=True)


def report_loop():
    from refresh_reviewed_publication import refresh
    with (JOB/'reporter.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            reviewed.report(reviewed.DEFAULT_JOB);lama.report(BASE/'fitlama')
            refresh(reviewed.DEFAULT_JOB)
            all_states={}
            roots=(reviewed.DEFAULT_JOB,BASE/'fitlama')
            for root in roots:
                for p in (root/'scenes').glob('*/status.json'):
                    key=p.parent.name if root==reviewed.DEFAULT_JOB else 'fitlama__'+p.parent.name
                    all_states[key]=read(p)
            complete=sum(v['state']=='complete' for v in all_states.values())
            evaluated=read(reviewed.DEFAULT_JOB/'status.json')['evaluated']+read(BASE/'fitlama/status.json')['evaluated']
            write(JOB/'status.json',dict(updated_at=now(),total=33,complete=complete,evaluated=evaluated,
                scenes=all_states,workers={p.stem:read(p) for p in (JOB/'workers').glob('*.json')},
                gpu_health={str(i):read(JOB/f'gpu{i}_health.json') if (JOB/f'gpu{i}_health.json').exists() else None for i in (0,1)}))
            if complete==33:return
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--group',choices=GROUPS)
    p.add_argument('--slot',choices=[f'{group}_gpu{gpu}' for gpu in (0,1) for group in GROUPS],
                   help='Persistent user-service worker, e.g. fitlama_gpu0')
    p.add_argument('--report-loop',action='store_true',help='Refresh all 3 groups and publication tables')
    a=p.parse_args()
    if a.report_loop:
        report_loop();raise SystemExit(0)
    if a.slot:
        if a.worker is not None or a.group is not None:p.error('--slot is exclusive with --worker/--group')
        a.group,gpu=a.slot.rsplit('_gpu',1);a.worker=int(gpu)
    if a.worker is None:launch()
    else:
        if a.group is None:p.error('--worker requires --group')
        JOB.mkdir(parents=True,exist_ok=True);(JOB/'workers').mkdir(exist_ok=True)
        (LamaWorker if a.group=='fitlama' else ReviewedWorker)(a.worker,a.group).run()
