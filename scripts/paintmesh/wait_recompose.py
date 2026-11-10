#!/usr/bin/env python3
"""Wait for complete SAM3D downloads, then run the authorized composition jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from recompose.io import read_json,write_json
from recompose.evidence import SELECTION
from recompose.report import write_status


def checkpoint_candidates(root,requirements):
    """Use exact official payload sizes, including configs; never consume partial weights."""
    expected={Path(x['file']).name:x['size'] for x in requirements['files']
              if x['file'].endswith(('.ckpt','.yaml')) and x['size']}
    found=[];pending=[]
    for config in Path(root).rglob('pipeline.yaml'):
        # ModelScope's fully downloaded models live under ~/.cache. Validate
        # payload sizes instead of excluding an entire cache directory.
        folder=config.parent;missing=[]
        for name,size in expected.items():
            file=folder/name
            if not file.is_file() or file.stat().st_size!=size:missing.append(name)
        if missing:pending.append(dict(path=str(folder),missing_or_partial=missing))
        else:found.append(folder)
    return found,pending


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights-root',default='/home/member/wyc/ckpt')
    p.add_argument('--config',default=str(Path(__file__).with_name('configs')/'recompose.json'))
    p.add_argument('--gpus',default='1,0');p.add_argument('--prepare-pid',type=int)
    p.add_argument('--interval',type=int,default=30);p.add_argument('--log-dir',required=True)
    args=p.parse_args();repo=Path(__file__).resolve().parents[2];cfg=read_json(args.config)
    root=repo/cfg['output_root'];root.mkdir(parents=True,exist_ok=True);logs=Path(args.log_dir);logs.mkdir(parents=True,exist_ok=True)
    lock=(root/'queue.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A composition queue is already active')
    def status(state,**detail):
        value=dict(state=state,pid=os.getpid(),updated_at=time.time(),logs=str(logs),**detail)
        write_json(root/'queue_status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
        write_status(root,value)
    def run(command,name):
        with (logs/name).open('a') as log:
            child=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
            while True:
                try:return child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    write_status(root,read_json(root/'queue_status.json'))
    requirements=read_json(root/'checkpoint_requirements.json');stable=None
    while True:
        candidates,pending=checkpoint_candidates(args.weights_root,requirements)
        if len(candidates)==1:
            snapshot=[(str(f),f.stat().st_size,f.stat().st_mtime_ns) for f in sorted(candidates[0].iterdir()) if f.is_file()]
            if snapshot==stable:break
            stable=snapshot;status('waiting_for_stable_weights',path=str(candidates[0]))
        else:
            stable=None;status('waiting_for_weights',root=args.weights_root,complete_candidates=[str(x) for x in candidates],partial=pending)
        time.sleep(max(10,args.interval))
    # A foreground preparation batch owns scene locks; let it finish first.
    if args.prepare_pid:
        while Path(f'/proc/{args.prepare_pid}/cmdline').exists():
            try:cmd=Path(f'/proc/{args.prepare_pid}/cmdline').read_bytes()
            except FileNotFoundError:break
            if b'run_recompose.py' not in cmd:break
            status('waiting_for_preparation',prepare_pid=args.prepare_pid);time.sleep(max(10,args.interval))
    binder='/home/member/wyc/MV-SAM3D-setup/bind_checkpoints.py'
    if run([cfg['generator']['python'],binder,'--checkpoint-dir',str(candidates[0])],'bind.log'):
        status('checkpoint_binding_failed');return 2
    base=[sys.executable,str(Path(__file__).with_name('run_recompose.py')),'--config',str(Path(args.config).resolve())]
    status('validating_generator')
    if run(base+['audit'],'audit.log'):status('generator_preflight_failed');return 2
    # Establish actual model/VRAM viability on an idle 24 GB card before all scenes.
    first_gpu=args.gpus.split(',')[0];status('running_first_scene',scene='truck',gpu=first_gpu)
    if run(base+['run','--scenes','truck','--gpus',first_gpu],'truck.log'):
        first_status=read_json(root/'truck/status.json')
        if first_status.get('state')!='needs_review':
            status('first_scene_failed',detail=first_status);return 2
        # Real generation succeeded; scene-specific pose failure does not mean
        # that the environment/VRAM is unusable for the other ten scenes.
        status('first_scene_needs_review',detail=first_status)
    status('running_remaining_scenes',gpus=args.gpus)
    scenes=','.join(s for s in SELECTION if s!='truck')
    result=run(base+['run','--scenes',scenes,'--gpus',args.gpus],'remaining.log')
    status('complete' if not result else 'some_scenes_failed',first_scene='truck',remaining_status=str(root/'batch_status.json'))
    return result


if __name__=='__main__':raise SystemExit(main())
