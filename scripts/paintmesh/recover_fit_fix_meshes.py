#!/usr/bin/env python3
"""Finish existing fit-fix raw TSDF meshes without retaining a TSDF volume.

Uses the exact current EDGS post-processing function; no training or metric changes.
"""
import argparse
import ast
import copy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Any
import numpy as np
import run_fit_difix as run
from edgs_inpaint_io import record, verify_record

SCRIPT = Path(__file__).resolve()
ROOT = run.BASE/'fit-fix'
SCENES = ('cube', 'doppelherz', 'car')


def available_mib():
    line = next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
    return int(line.split()[1]) / 1024


def host_guard(done):
    while not done.wait(.5):
        if available_mib() < 6144:
            print('Stopping our post-processing: host MemAvailable below 6144 MiB', flush=True)
            os._exit(75)


def postprocess(scene):
    import open3d as o3d
    raw = ROOT/'scenes'/scene/'inpaint/inpainted_3dgs/mesh/ours_5000/tsdf_fusion.ply'
    dst = raw.with_name('tsdf_fusion_post.ply')
    info = ROOT/'recovery_mesh_20261103'/f'{scene}_postprocess.json'
    request = dict(raw=record(raw), renderer=record(run.EDGS/'render.py'), adapter=record(SCRIPT), clusters=1)
    if info.exists():
        previous = run.read(info)
        if previous['request'] != request: raise ValueError('Recovery inputs changed')
        verify_record(previous['output']); return
    if dst.exists(): raise ValueError('Untracked processed mesh already exists')
    done = threading.Event(); threading.Thread(target=host_guard, args=(done,), daemon=True).start()
    # Load the original function verbatim, avoiding renderer imports and GPU/TSDF allocations.
    module = ast.parse((run.EDGS/'render.py').read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_post_process_mesh')
    context = dict(np=np, copy=copy, Any=Any)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(run.EDGS/'render.py'), 'exec'), context)
    print('Reading existing raw mesh:', raw, flush=True)
    mesh = o3d.io.read_triangle_mesh(str(raw))
    expected = {}
    with raw.open('rb') as f:
        while True:
            line = f.readline().decode().strip()
            if line.startswith('element '):
                _, name, size = line.split(); expected[name] = int(size)
            if line == 'end_header': break
            if not line: raise ValueError('Truncated PLY header')
    if len(mesh.vertices) != expected['vertex'] or len(mesh.triangles) != expected['face']:
        raise ValueError('Incomplete raw mesh; cannot recover')
    print('Raw vertices/faces:', len(mesh.vertices), len(mesh.triangles), flush=True)
    processed = context['_post_process_mesh'](mesh, 1)
    if not len(processed.vertices) or not len(processed.triangles): raise ValueError('Empty processed mesh')
    tmp = dst.with_name('.recovery_tsdf_fusion_post.ply')
    if not o3d.io.write_triangle_mesh(str(tmp), processed): raise RuntimeError('Mesh write failed')
    verify_record(request['raw']); verify_record(request['renderer']); verify_record(request['adapter'])
    tmp.replace(dst)
    run.write(info, dict(complete=True, finished_at=run.now(), request=request, output=record(dst),
                        vertices=len(processed.vertices), faces=len(processed.triangles),
                        operation='Exact EDGS _post_process_mesh(raw, 1); TSDF integration not repeated'))
    done.set(); print('Recovered:', dst, flush=True)


class Recovery(run.Worker):
    def execute(self, stage, cmd, cwd, expected, gpu=False):
        if stage == 'mesh_render':
            while available_mib() < 80000:
                self.save(state='waiting_for_host_memory', stage=stage,
                          reason='Need 80000 MiB MemAvailable for one large mesh', child_pid=None)
                run.report(self.root); time.sleep(15)
            cmd = [run.PYTHON, SCRIPT, '--postprocess', self.case.name]
            cwd = run.REPO; gpu = False
        return super().execute(stage, cmd, cwd, expected, gpu=gpu)

    def recover(self):
        with (self.wdir/'run.lock').open('a') as lock, (self.root/'recovery_mesh_20261103/run.lock').open('a') as hostlock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(hostlock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, self.stop); signal.signal(signal.SIGINT, self.stop)
            for s in self.req['scenes']:
                if s['scene'] not in SCENES: continue
                case = self.root/'scenes'/s['scene']
                with (case/'run.lock').open('a') as scene_lock:
                    fcntl.flock(scene_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if run.read(case/'status.json')['state'] == 'complete': continue
                    # Completed training and evaluation are mandatory; no retraining allowed.
                    for key in ('train_5000', 'evaluate_5000'):
                        for artifact in run.read(case/'receipts'/f'{key}.json')['artifacts']:
                            verify_record(artifact)
                    try: self.process(s)
                    except Exception as e:
                        self.save(state='failed', reason=str(e), child_pid=None)
                        traceback.print_exc(); run.report(self.root)
            self.case = None
            self.save(state='mesh_recovery_finished', scene=None, stage=None, child_pid=None, reason=None)
            run.report(self.root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--postprocess', choices=SCENES)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.postprocess: postprocess(args.postprocess)
    elif args.worker: Recovery(ROOT, 0).recover()
    else:
        folder = ROOT/'recovery_mesh_20261103'; folder.mkdir(exist_ok=True)
        with (folder/'launch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            for g in (0,1):
                state = run.read(ROOT/f'workers/gpu{g}/status.json')
                if Path(f'/proc/{state.get("pid")}/cmdline').exists():
                    raise RuntimeError('Worker PID still exists; inspect before recovery')
            run.check_code(run.read(ROOT/'request.json'))
            run.write(folder/'before.json', {s:run.read(ROOT/'scenes'/s/'status.json') for s in SCENES})
            with (folder/'worker.log').open('a') as out:
                child = subprocess.Popen([run.PYTHON,str(SCRIPT),'--worker'],cwd=run.REPO,
                    stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            run.write(folder/'launch.json', dict(pid=child.pid, at=run.now(), code=record(SCRIPT), scenes=SCENES,
                reason='Kernel-confirmed host OOM; serial existing raw-mesh post-processing; metrics unchanged'))
            print('Mesh recovery PID:', child.pid)
