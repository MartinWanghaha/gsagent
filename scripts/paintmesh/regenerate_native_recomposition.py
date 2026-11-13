#!/usr/bin/env python3
"""Regenerate native-RGB object meshes in place, with independent CPU/GPU queues."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from recompose.evidence import SELECTION
from recompose.io import read_json,write_json
from recompose.report import write_status


def claim_available_scene(jobs,gpu,used_limit,finished_preparing):
    """A busy GPU must not hold a queued scene away from an available peer."""
    from recompose.pipeline import wait_gpu
    try:scene=jobs.get_nowait()
    except queue.Empty:return None,not finished_preparing.is_set()
    try:wait_gpu(gpu,maximum_used_mib=used_limit,timeout=0)
    except RuntimeError:
        jobs.put(scene);jobs.task_done();return None,True
    return scene,True


def array_hashes(path):
    with np.load(path) as data:
        return {k:hashlib.sha256(np.ascontiguousarray(data[k]).tobytes()).hexdigest() for k in data.files}


def snapshot_baseline(root):
    """Keep only comparison metadata, never an old generation fallback."""
    path=root/'native_input_baseline.json'
    if path.exists():return
    objects={}
    for scene in SELECTION:
        for meta in sorted((root/scene).glob('objects/*/evidence/observations.json')):
            d=read_json(meta);obj=meta.parent.parent;pose=read_json(obj/'placement/pose.json')
            objects[f'{scene}/{obj.name}']=dict(
                views=[dict(name=v['name'],original=v['original'],validation=v['validation'],
                            observation=array_hashes(meta.parent/'observations'/Path(v['name']).with_suffix('.npz')))
                       for v in d['views']],
                anchors=array_hashes(meta.parent/'anchors.npz'),
                previous_status=pose['status'],previous_validation=pose['validation'],
                previous_physics=pose.get('physics'))
    write_json(path,dict(created_at=time.time(),objects=objects,source='previous 768-pixel generation run'))


def compare_inputs(root,scene):
    baseline=read_json(root/'native_input_baseline.json')['objects'];checks=[]
    for key,old in baseline.items():
        if not key.startswith(scene+'/'):continue
        target=key.split('/')[1];folder=root/scene/'objects'/target/'evidence';new=read_json(folder/'observations.json')
        if new.get('generation_input_policy')!='native_rgb_official_crop':raise ValueError('Native generation policy is missing')
        new_views=[dict(name=v['name'],original=v['original'],validation=v['validation'],
                        observation=array_hashes(folder/'observations'/Path(v['name']).with_suffix('.npz'))) for v in new['views']]
        if old['views']!=new_views:raise ValueError(f'{key}: selection or pose/validation pixels changed')
        if old['anchors']!=array_hashes(folder/'anchors.npz'):raise ValueError(f'{key}: geometry anchors changed')
        checks.append(dict(object=target,views_unchanged=True,pose_pixels_unchanged=True,anchors_unchanged=True))
    if not checks:raise ValueError(f'No baseline object observations for {scene}')
    write_json(root/scene/'native_input_isolation.json',dict(passed=True,checks=checks))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(Path(__file__).with_name('configs')/'recompose.json'))
    p.add_argument('--gpus',default='0,1');p.add_argument('--log-dir',required=True)
    p.add_argument('--resume',action='store_true',help='Retain prepared inputs, finished scenes and independent running preparation processes')
    p.add_argument('--wait-for-queue',action='store_true',help='Start after the current batch releases its queue lock; useful for OOM recovery')
    a=p.parse_args();repo=Path(__file__).resolve().parents[2];cfg=read_json(a.config)
    root=repo/cfg['output_root'];logs=Path(a.log_dir).resolve();logs.mkdir(parents=True,exist_ok=True)
    gpus=[int(x) for x in a.gpus.split(',')]
    if not gpus or len(set(gpus))!=len(gpus):raise ValueError('GPU IDs must be distinct')
    lock=(root/'queue.lock').open('a')
    if a.wait_for_queue:print('Waiting for the active batch; completed native results will be retained.',flush=True)
    fcntl.flock(lock,fcntl.LOCK_EX if a.wait_for_queue else fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot_baseline(root)
    write_json(root/'active_jobs.json',dict(logs=str(logs),jobs=[dict(task='native RGB regeneration, 11 scenes / 13 objects',pid=os.getpid())]))
    prior=read_json(root/'native_regeneration_status.json') if a.resume and (root/'native_regeneration_status.json').exists() else {}
    jobs=queue.Queue();stop=threading.Event();prepared_all=threading.Event();results=prior.get('scenes',{});mutex=threading.Lock()
    used_limit=cfg['execution']['gpu_max_used_mib']
    poll_interval=cfg['execution']['gpu_poll_interval_seconds']
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    command=[sys.executable,str(Path(__file__).with_name('run_recompose.py')),'--config',str(Path(a.config).resolve()),'run']

    def execute(args,name):
        with (logs/name).open('a' if a.resume else 'w') as log:
            return subprocess.call(args,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)

    def record(scene,**value):
        with mutex:
            results.setdefault(scene,{}).update(value)
            if value.get('phase')!='failed':results[scene].pop('error',None)
            write_json(root/'native_regeneration_status.json',dict(pid=os.getpid(),updated_at=time.time(),logs=str(logs),scenes=results))

    def monitor():
        while not stop.wait(30):write_status(root)

    def prepare_scene(scene):
        record(scene,phase='prepare')
        # A scheduler can be replaced without interrupting CPU raycasting owned
        # by its old child. Let that scene writer finish before resuming it.
        with (root/scene/'run.lock').open('a') as scene_lock:
            while True:
                try:
                    fcntl.flock(scene_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    record(scene,phase='waiting_existing_preparation');time.sleep(10)
            fcntl.flock(scene_lock,fcntl.LOCK_UN)
        code=execute(command+['--scenes',scene,'--gpus',str(gpus[0]),'--until','prepare','--single-worker'],f'{scene}.prepare.log')
        if code:raise RuntimeError(f'Preparation failed: {scene}, exit {code}')
        compare_inputs(root,scene)
        from recompose.mvsam_worker import validate_generation_inputs
        for evidence in (root/scene).glob('objects/*/evidence'):
            _,_,checks=validate_generation_inputs(evidence)
            write_json(evidence/'native_input_validation.json',dict(passed=True,views=checks))
        record(scene,phase='prepared',unchanged_pose_evidence=True);jobs.put(scene)

    def gpu_worker(gpu):
        while True:
            scene,keep_running=claim_available_scene(jobs,gpu,used_limit,prepared_all)
            if not keep_running:return
            if scene is None:
                time.sleep(poll_interval if not jobs.empty() else 1);continue
            try:
                record(scene,phase='generation_alignment_export',gpu=gpu)
                code=execute(command+['--scenes',scene,'--gpus',str(gpu),'--single-worker'],f'{scene}.compose.log')
                state=read_json(root/scene/'status.json')
                if state['state'] not in ('complete','needs_review'):
                    raise RuntimeError(f'{scene}: exit {code}, {state}')
                # Export already completed in run_scene; render actual output without
                # racing the shared all-scenes report between GPU workers.
                snippet=('from recompose.pipeline import gpu_slot; from review_recomposition import review_scene; '
                         f'\nwith gpu_slot({gpu},timeout=None,maximum_used_mib={used_limit}): '
                         f'review_scene({str(root/scene/"composition/scene.json")!r},{gpu})')
                rc=execute([sys.executable,'-c',f'import sys; sys.path.insert(0,{str(Path(__file__).parent)!r}); '+snippet],f'{scene}.review.log')
                if rc:raise RuntimeError(f'{scene}: preview failed with exit {rc}')
                record(scene,phase='finished',state=state['state'],preview_complete=True)
            except Exception as error:
                print(repr(error),flush=True);record(scene,phase='failed',error=str(error))
            finally:jobs.task_done()

    # Existing payloads are overwritten by their owning stages. Mark pending work
    # explicitly so old meshes/previews cannot be reported as new native results.
    scenes=[s for s in SELECTION if not a.resume or results.get(s,{}).get('phase')!='finished']
    if not a.resume:
        for scene in SELECTION:
            write_json(root/scene/'status.json',dict(state='queued',stage='native_input_regeneration'))
            shutil.rmtree(root/scene/'composition/review',ignore_errors=True)
            shutil.rmtree(root/scene/'movement_demo',ignore_errors=True)
        (root/'all_scenes_comparison.jpg').unlink(missing_ok=True)
    write_status(root)
    execute([sys.executable,str(Path(__file__).with_name('review_recomposition.py')),'--root',str(root),'--summary-only'],'summary.initial.log')
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    workers=[threading.Thread(target=gpu_worker,args=(g,),daemon=False) for g in gpus]
    for thread in workers:thread.start()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(prepare_scene,s):s for s in scenes}
        for future in as_completed(futures):
            try:future.result()
            except Exception as error:
                scene=futures[future];print(repr(error),flush=True)
                write_json(root/scene/'status.json',dict(state='failed',stage='native_prepare',error=str(error)))
                record(scene,phase='failed',error=str(error))
    prepared_all.set()
    for thread in workers:thread.join()
    audit=execute([sys.executable,str(Path(__file__).with_name('audit_recomposition.py')),'--root',str(root)],'gaussian_audit.log')
    if read_json(root/'car/status.json').get('state')=='complete':
        # A previous movement demo references the previous generated mesh; recreate.
        from recompose.pipeline import wait_gpu
        wait_gpu(gpus[0],maximum_used_mib=used_limit,timeout=None)
        move=execute([sys.executable,str(Path(__file__).with_name('demo_recomposition_move.py')),'--scene','car','--gpu',str(gpus[0]),'--gpu-max-used-mib',str(used_limit),'--wait-seconds','0'],'car.movement.log')
    else:move=None
    summary=execute([sys.executable,str(Path(__file__).with_name('review_recomposition.py')),'--root',str(root),'--summary-only'],'summary.final.log')
    rows=['# 原生分辨率生成与上轮对照','',
          '只改变生成输入分辨率；留出视角、位姿观测与锚点逐数组核对相同。IoU 是留出轮廓一致性，不是位姿真值。','',
          '| 场景 / 物体 | 上轮 IoU | 本轮 IoU | 差值 | 本轮状态 |','|---|---:|---:|---:|---|']
    for key,old in read_json(root/'native_input_baseline.json')['objects'].items():
        scene,target=key.split('/');previous=float(np.median([v['iou'] for v in old['previous_validation']]))
        if results.get(scene,{}).get('phase')!='finished':
            rows.append(f'| {key} | {previous:.6f} | - | - | 未完成 |');continue
        pose=read_json(root/scene/'objects'/target/'placement/pose.json');now=float(np.median([v['iou'] for v in pose['validation']]))
        rows.append(f'| {key} | {previous:.6f} | {now:.6f} | {now-previous:+.6f} | {pose["status"]} |')
    (root/'native_input_comparison.md').write_text('\n'.join(rows)+'\n')
    stop.set();watcher.join();write_status(root)
    write_json(root/'batch_status.json',[dict(read_json(root/s/'status.json'),scene=s) for s in SELECTION])
    failed=audit or summary or any(v.get('phase')!='finished' for v in results.values())
    write_json(root/'native_regeneration_complete.json',dict(finished_at=time.time(),complete=not bool(failed),
        gaussian_audit_exit=audit,summary_exit=summary,movement_demo_exit=move,scenes=results))
    return int(bool(failed))


if __name__=='__main__':raise SystemExit(main())
