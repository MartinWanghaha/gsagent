#!/usr/bin/env python3
"""Finish authorized scenes after old workers release locks; preserve generation."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from recompose.io import read_json,write_json
from recompose.report import write_status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-pids',default='');p.add_argument('--scenes',required=True)
    p.add_argument('--reset-align-scenes',default='');p.add_argument('--gpus',default='1,0')
    p.add_argument('--log-dir',required=True)
    p.add_argument('--config',default=str(Path(__file__).with_name('configs')/'recompose.json'))
    a=p.parse_args();repo=Path(__file__).resolve().parents[2];cfg=read_json(a.config);root=repo/cfg['output_root']
    logs=Path(a.log_dir);logs.mkdir(parents=True,exist_ok=True)
    lock=(root/'recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(state,**extra):
        value=dict(state=state,pid=os.getpid(),updated_at=time.time(),scenes=a.scenes.split(','),logs=str(logs),**extra)
        write_json(root/'recovery_status.json',value);print(value,flush=True)
        queue=read_json(root/'queue_status.json') if (root/'queue_status.json').exists() else None
        write_status(root,queue)
    pids=[int(x) for x in a.wait_pids.split(',') if x]
    while True:
        active=[]
        for pid in pids:
            try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:continue
            if b'run_recompose.py' in cmd:active.append(pid)
        if not active:break
        status('waiting_for_existing_workers',active_pids=active);time.sleep(15)
    for scene in filter(None,a.reset_align_scenes.split(',')):
        if scene not in a.scenes.split(','):raise ValueError('Reset scene must be included in recovery')
        for receipt in (root/scene).glob('objects/*/receipts/align.json'):
            receipt.rename(receipt.with_name(f'align.before_contact_fix_{time.time_ns()}.json'))
    command=[sys.executable,str(Path(__file__).with_name('run_recompose.py')),'--config',str(Path(a.config).resolve()),
             'run','--scenes',a.scenes,'--gpus',a.gpus]
    status('running')
    with (logs/'recovery.log').open('a') as f:
        child=subprocess.Popen(command,cwd=repo,stdout=f,stderr=subprocess.STDOUT)
        while True:
            try:code=child.wait(timeout=20);break
            except subprocess.TimeoutExpired:
                queue=read_json(root/'queue_status.json') if (root/'queue_status.json').exists() else None
                write_status(root,queue)
    status('complete' if code==0 else 'some_scenes_need_review',returncode=code)
    return code


if __name__=='__main__':raise SystemExit(main())
