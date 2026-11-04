#!/usr/bin/env python3
"""Resume redbull in fit (GPU0) and fit-fix (GPU1), preserving other scenes."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import run_fit_concurrent as concurrent
import run_fit_difix as fix
from edgs_inpaint_io import record, verify_record

SCRIPT=Path(__file__).resolve()
BASE=fix.BASE
MESH_LOCK=BASE/'fit-fix/recovery_mesh_20261103/run.lock'


class Redbull:
    def execute(self,stage,cmd,cwd,expected,gpu=False):
        verify_record(self.repair['runner'])
        if stage not in ('mesh_render','inpaint_7_8'):
            return super().execute(stage,cmd,cwd,expected,gpu=gpu)
        # TSDF/mesh postprocessing shares host RAM across both GPUs.
        MESH_LOCK.parent.mkdir(exist_ok=True)
        with MESH_LOCK.open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    self.save(state='waiting_for_host_mesh_lock',stage=stage,reason='Serialize mesh memory only; training/evaluation run on both GPUs',child_pid=None)
                    self.report(self.root);time.sleep(15)
            return super().execute(stage,cmd,cwd,expected,gpu=gpu)

    def run_redbull(self):
        self.repair=fix.read(self.root/'redbull_grid_repair_20261103/repair.json')
        self.report=fix.report if self.root.name=='fit-fix' else concurrent.fit.report
        self.wdir=self.root/'workers'/f'redbull_gpu{self.gpu}';self.wdir.mkdir(exist_ok=True)
        case=self.root/'scenes/redbull'
        with (self.wdir/'run.lock').open('a') as lock,(case/'run.lock').open('a') as scene_lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(scene_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN)
            signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
            spec=next(s for s in self.req['scenes'] if s['scene']=='redbull')
            try:self.process(spec)
            except Exception as e:
                self.save(state='failed',child_pid=None,reason=str(e));traceback.print_exc();self.report(self.root);raise


class FitWorker(Redbull,concurrent.ConcurrentWorker):pass
class FixWorker(Redbull,fix.Worker):pass


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=('fit','fit-fix'));args=p.parse_args()
    if args.worker:
        root=BASE/args.worker
        (FitWorker(root,0) if args.worker=='fit' else FixWorker(root,1)).run_redbull()
    else:
        launched={}
        for name,gpu in (('fit',0),('fit-fix',1)):
            root=BASE/name;folder=root/'redbull_grid_repair_20261103'
            verify_record(fix.read(folder/'repair.json')['runner'])
            status=root/f'workers/redbull_gpu{gpu}/status.json'
            if status.exists():
                value=fix.read(status);pid=value.get('pid');path=Path(f'/proc/{pid}/cmdline')
                if path.exists() and str(SCRIPT).encode() in path.read_bytes():
                    launched[name]=dict(pid=pid,state='already_running');continue
            log=folder/f'gpu{gpu}.log'
            with log.open('a') as out:
                child=subprocess.Popen([fix.PYTHON,str(SCRIPT),'--worker',name],cwd=fix.REPO,stdin=subprocess.DEVNULL,
                    stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            launched[name]=dict(pid=child.pid,gpu=gpu,log=str(log))
            fix.write(folder/'launch.json',launched[name])
        print(launched,flush=True)
