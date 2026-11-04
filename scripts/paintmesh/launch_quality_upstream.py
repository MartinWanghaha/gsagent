#!/usr/bin/env python3
"""Detach the reviewed GPU-1 quality-repair queue from the interactive session."""
import fcntl
import subprocess
import sys
from pathlib import Path
from run_geometry_ablation import read, write, now

REPO=Path(__file__).resolve().parents[2]
job=Path(read(REPO/'logs/paintmesh_quality_retrain.latest.json')['job'])
queue=job/'upstream'
assert read(queue/'request.json')['gpu']==1
with (queue/'retrain.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
with (queue/'run.log').open('a') as log:
    child=subprocess.Popen([sys.executable,str(REPO/'scripts/paintmesh/run_quality_retrain.py'),'--job',str(queue),
                            '--wait-pid',(job/'run.pid').read_text().strip()],cwd=REPO,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(queue/'run.pid').write_text(str(child.pid)+'\n')
write(queue/'launch.json',dict(pid=child.pid,time=now(),gpu=1,job=str(queue)))
print('GPU 1 upstream repair queue PID:',child.pid,'logs:',queue,flush=True)
