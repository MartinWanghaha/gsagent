#!/usr/bin/env python3
"""Resume frozen fit experiments alongside other GPU jobs without idle gating.

Scheduling-only adapter: original fit/fix code and artifact hashes stay intact.
"""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

import run_fit_test_poses as fit
from run_geometry_ablation import read, write, now
from edgs_inpaint_io import record, verify_record

guard = fit.guard
SCRIPT = Path(__file__).resolve()
POLICY = dict(idle_seconds=0, reserve_mib=1536, own_limit_mib=22500,
              sample_seconds=2, disk_free_gib=40,
              foreign_process_policy='ignore_process_presence_and_utilization; never_signal_foreign')
REQUIRED_MIB = dict(virtual_render=6000, track_masks=8000, inpaint_3_3=10000,
                    inpaint_4_4=6000, inpaint_5_5=12000, inpaint_7_8=8000,
                    evaluate_5000=4000)


def admission(snapshot, stage):
    required = REQUIRED_MIB.get(stage, 8000)
    return None if snapshot['free_mib'] >= required else f'need_{required}_MiB_free'


def running_decision(snapshot, own_pids=(), running=False):
    if running:
        own = sum(p['memory_mib'] for p in snapshot['processes'] if p['pid'] in own_pids)
        if snapshot['free_mib'] < POLICY['reserve_mib'] or own >= POLICY['own_limit_mib']:
            return 'memory_risk'
    return None


class ConcurrentWorker(fit.Worker):
    def __init__(self, root, gpu):
        super().__init__(root, gpu)
        self.scheduling = read(root/'scheduling/current.json')
        verify_record(self.scheduling['adapter'])
        if self.scheduling['policy'] != POLICY or self.scheduling['required_mib'] != REQUIRED_MIB:
            raise ValueError('Scheduling policy differs from recorded policy')
        guard.GPU_POLICY = POLICY
        guard.gpu_decision = running_decision

    def save(self, **kw):
        super().save(**dict(kw, scheduling_policy='concurrent_free_memory_only'))

    def wait_gpu(self, stage):
        while True:
            snap = guard.gpu_snapshot(self.uuid)
            reason = admission(snap, stage)
            if reason is None:
                return
            self.save(state='waiting_for_memory', stage=stage, reason=reason,
                      gpu_snapshot=snap, child_pid=None)
            fit.report(self.root)
            time.sleep(10)

    def execute(self, *args, **kwargs):
        verify_record(self.scheduling['adapter'])
        return super().execute(*args, **kwargs)

    def run(self):
        # A worker and a scene remain exclusive within fit, but no whole-GPU lock.
        with (self.wdir/'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, self.stop)
            signal.signal(signal.SIGINT, self.stop)
            self.save(state='starting', child_pid=None)
            while True:
                locked = False
                for spec in self.req['scenes']:
                    case = self.root/'scenes'/spec['scene']
                    with (case/'run.lock').open('a') as scene_lock:
                        try:
                            fcntl.flock(scene_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            locked = True
                            continue
                        if read(case/'status.json')['state'] in ('complete', 'failed', 'blocked_memory_risk'):
                            continue
                        try:
                            self.process(spec)
                        except guard.MemoryRisk as e:
                            self.save(state='blocked_memory_risk', reason=str(e), child_pid=None)
                            traceback.print_exc(); fit.report(self.root)
                        except Exception as e:
                            self.save(state='failed', reason=str(e), child_pid=None)
                            traceback.print_exc(); fit.report(self.root)
                self.case = None
                if not locked:
                    break
                self.save(state='waiting_for_other_fit_worker', scene=None, stage=None, child_pid=None)
                fit.report(self.root); time.sleep(15)
            self.save(state='queue_finished', scene=None, stage=None, child_pid=None)
            fit.report(self.root)


def process_args(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
    except FileNotFoundError:
        return []


def launch(root):
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fit.check_code(read(root/'request.json'))
        old_states = {}
        for gpu in (0, 1):
            state = read(root/f'workers/gpu{gpu}/status.json')
            pid = state.get('pid'); args = process_args(pid)
            if args and str(SCRIPT) in args:
                raise RuntimeError('Concurrent fit already running; leave existing workers intact')
            if args and (str(fit.SCRIPT) not in args or str(root) not in args):
                raise RuntimeError('Worker PID no longer belongs to fit')
            if args and (state.get('child_pid') is not None or not state['state'].startswith('waiting_')):
                raise RuntimeError('Fit has an active child; do not interrupt it with this waiting-worker migration')
            old_states[str(gpu)] = state
        directory = root/'scheduling'/time.strftime('concurrent_%Y%m%d_%H%M%S')
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(SCRIPT, directory/SCRIPT.name)
        entry = dict(created_at=now(), adapter=record(SCRIPT), snapshot=record(directory/SCRIPT.name),
                     policy=POLICY, required_mib=REQUIRED_MIB, old_workers=old_states,
                     original_request=record(root/'request.json'), changes='scheduling only; training configs and inputs unchanged')
        write(directory/'migration.json', entry); write(root/'scheduling/current.json', entry)
        for state in old_states.values():
            pid = state.get('pid')
            if not process_args(pid): continue
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic()+20
        while any(process_args(s.get('pid')) for s in old_states.values()):
            if time.monotonic() > deadline:
                raise RuntimeError('Old fit workers have not exited; no duplicate workers started')
            time.sleep(.2)
        workers = {}
        for gpu in (0, 1):
            log = root/f'logs/gpu{gpu}_concurrent.log'
            with log.open('a') as out:
                child = subprocess.Popen([fit.PYTHON, str(SCRIPT), '--root', str(root), '--worker', str(gpu)],
                    cwd=fit.REPO, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2',
                             OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'))
            workers[str(gpu)] = dict(pid=child.pid, log=str(log), launcher=str(SCRIPT))
        write(root/'workers.json', workers); print(workers, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=fit.BASE/'fit')
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args(); root = args.root.resolve()
    if args.worker is None:
        launch(root)
    else:
        ConcurrentWorker(root, args.worker).run()
