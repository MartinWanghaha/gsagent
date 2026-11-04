#!/usr/bin/env python3
"""Detach the prepared truck run and combined report watcher on the host."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from run_geometry_ablation import REPO, read, write, now


def alive(pid, expected):
    path=Path(f'/proc/{pid}/cmdline')
    return path.exists() and expected.encode() in path.read_bytes()


def main(root):
    script=Path(__file__).with_name('run_soft_reference_pilot.py')
    directory=root/'truck_soft_reference/scenes/truck'
    review=read(directory/'screen/review.json')
    if not review['approved']:raise ValueError('Prepared RGB targets failed visual review')
    state=read(directory/'status.json') if (directory/'status.json').exists() else {}
    if state.get('state')=='complete':raise ValueError('Truck follow-up already completed')
    if state.get('pid') and alive(state['pid'],str(script)):raise ValueError('Truck worker already exists')
    with (directory/'worker.log').open('a') as log:
        worker=subprocess.Popen([sys.executable,str(script),'worker','--root',str(root/'truck_soft_reference')],
            cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'truck_soft_reference/workers.json',dict(truck=dict(pid=worker.pid,gpu=1,launched_at=now())))
    reporter=Path(__file__).with_name('report_reference_optimization.py')
    with (root/'optimization_report.log').open('a') as log:
        monitor=subprocess.Popen([sys.executable,str(reporter),'--root',str(root),'--watch'],
            cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'followup_processes.json',dict(launched_at=now(),truck_worker=worker.pid,reporter=monitor.pid))
    print('Truck GPU 1 worker:',worker.pid,'report watcher:',monitor.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
