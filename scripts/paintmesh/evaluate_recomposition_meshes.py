#!/usr/bin/env python3
"""Disconnect-safe CPU visibility + memory-gated GPU appearance mesh evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,REPO,SCENES,METHODS,STATES,read,write,specs_for
from mesh_benchmark.layout import work_root,configure_sources

DEFAULT=dict(protocol_version=1,seed=0,appearance_steps=5000,grid_resolution=256,rank=16,
             batch_rays=8192,pool_pixels=32768,gpu_cap_mib=3072,
             minimum_free_gpu_mib=10000,minimum_available_ram_gib=30,
             scoring_protocol_version=2,f1_threshold_fractions=[.005,.01,.02])

def marker_for(output,key):
    method,scene,stage=key
    if stage=='prepare':return output/'cache'/method/scene/'recomposition/complete.json'
    if stage=='support':return output/'common'/scene/'recomposition_support/complete.json'
    return output/stage/method/scene/'metrics.json'

def overwrite_scores(output):
    """Remove only derived scores/support; keep meshes, depths, renders and fields."""
    import shutil
    for state in STATES:
        for method in METHODS:
            for scene in SCENES:
                for p in [output/state/method/scene/'metrics.json',output/'cache'/method/scene/state/'geometry_metrics.json']:
                    if p.exists():p.unlink()
        for scene in SCENES:
            p=output/'common'/scene/(state+'_support')
            if p.exists():shutil.rmtree(p)
    for name in ['protocol.json','adopt_workers.json','adopted_workers.json']:
        p=output/name
        if p.exists():p.unlink()
    print('OVERWRITE_SCORING_V2: mesh/depth/render/color-field artifacts retained',flush=True)

def refresh_cohorts(output,old,new):
    """Rescore only scenes whose completed-mesh membership actually changed."""
    import shutil
    changed=[scene for scene in new if old.get(scene)!=new[scene]]
    for scene in changed:
        for state in STATES:
            support=output/'common'/scene/(state+'_support')
            if support.exists():shutil.rmtree(support)
            for method in METHODS:
                for path in [output/state/method/scene/'metrics.json',output/'cache'/method/scene/state/'geometry_metrics.json']:
                    if path.exists():path.unlink()
    print('REFRESH_COMPLETED_MESH_SCENES',changed,flush=True)

class ExistingWorker:
    """Resume supervising a live worker without launching a duplicate GPU job."""
    def __init__(self,pid,marker):self.pid=pid;self.marker=marker
    def poll(self):
        proc=Path('/proc')/str(self.pid)
        if proc.exists():
            try:
                cmd=(proc/'cmdline').read_bytes()
                if b'evaluate_recomposition_meshes.py' in cmd and b'--worker' in cmd:return None
            except FileNotFoundError:pass
        return 0 if self.marker.exists() and read(self.marker).get('complete') else 1

def free_gpus():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if p.returncode:return []
    return [(int(a),int(b)) for a,b in (x.split(',') for x in p.stdout.strip().splitlines())]

def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])/1024**2
    return 0

def run(args):
    output=work_root(args.output);configure_sources(output);output.mkdir(parents=True,exist_ok=True)
    (output/'logs').mkdir(exist_ok=True)
    lock=open(output/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.overwrite_evaluation:overwrite_scores(output)
    cfg=dict(DEFAULT);cfg.update(appearance_steps=args.appearance_steps)
    scenes=args.scenes.split(',') if args.scenes else SCENES
    methods=args.methods.split(',') if args.methods else METHODS
    if (output/'protocol.json').exists():
        old=read(output/'protocol.json')
        if old['config']!=cfg:raise ValueError('Use a different output for a different protocol')
    cohorts={s:[m for m in methods if specs_for(s,m,'background') is not None] for s in scenes}
    if args.refresh_completed_meshes and (output/'protocol.json').exists():
        refresh_cohorts(output,read(output/'protocol.json').get('cohorts',{}),cohorts)
    protocol=dict(config=cfg,scenes=scenes,methods=methods,cohorts=cohorts,scoring_protocol_version=2,source='existing artifacts, original-state 1/8 baseline holdout',
                  point_reference='COLMAP sparse SfM, not dense independently measured GT',
                  appearance='shared fixed-geometry VM-SH; MILo-inspired lightweight adaptation; no held-out RGB training',
                  leak_notice='Existing PaintMesh geometry/selection/evidence may include these held-out views; not a matched-training-split comparison',
                  reference_state='object present; test* removed-object RGB excluded from combination benchmark',
                  scripts=[str(HERE/'mesh_benchmark'),str(Path(__file__).resolve())])
    write(output/'protocol.json',protocol)
    (output/'campaign.pid').write_text(str(os.getpid())+'\n')
    docs=(HERE/'mesh_benchmark/PROTOCOL.zh-CN.md').read_text()
    (output/'PROTOCOL.zh-CN.md').write_text(docs)
    from mesh_benchmark.report import update
    tasks={};running={};failed=set();started=time.time()
    # Snapshot only completed mesh artifacts. Later completed meshes can be added
    # by rerunning this same command; successful cache/metrics are resumed safely.
    for scene in scenes:
        for method in ['paintmesh']+[m for m in methods if m!='paintmesh']:
            if method not in methods:continue
            if specs_for(scene,method,'background') is None:continue
            tasks[method,scene,'prepare']=dict(status='pending')
            for state in STATES:tasks[method,scene,state]=dict(status='pending')
        if cohorts[scene]:tasks['common',scene,'support']=dict(status='pending')
    adopt=output/'adopt_workers.json'
    if adopt.exists():
        for previous in read(adopt):
            key=previous['method'],previous['scene'],previous['stage']
            if key not in tasks:raise ValueError(f'Cannot adopt out-of-scope worker: {key}')
            method,scene,stage=key
            marker=marker_for(output,key)
            process=ExistingWorker(previous['pid'],marker)
            log=open(output/'logs'/f'{method}__{scene}__{stage}.log','a')
            running[key]=dict(process=process,log=log,gpu=previous.get('gpu'))
            tasks[key].update(status='running',pid=process.pid,gpu=previous.get('gpu'),adopted=True)
            print('ADOPT',key,'pid',process.pid,flush=True)
        adopt.replace(output/'adopted_workers.json')
    while True:
        for key,job in list(running.items()):
            p=job['process'];ret=p.poll()
            if ret is None:continue
            job['log'].close();del running[key]
            method,scene,stage=key
            tasks[key].update(status='complete' if ret==0 else 'failed',exit_code=ret,ended=time.time())
            if ret:failed.add(key)
            print('FINISHED',key,'exit',ret,flush=True)
        cpu_busy=any(k[2] in ['prepare','support'] for k in running)
        for key,t in tasks.items():
            method,scene,stage=key
            marker=marker_for(output,key)
            if marker.exists() and t['status']=='pending':t['status']='complete'
        if not cpu_busy and available_ram()>=cfg['minimum_available_ram_gib']:
            ready=[key for key,t in tasks.items() if key[2]=='support' and t['status']=='pending' and
                   all(tasks[m,key[1],'prepare']['status']=='complete' for m in cohorts[key[1]])]
            if ready:
                key=ready[0];launch(key,None,output,cfg,tasks[key],running);cpu_busy=True
            for key,t in tasks.items():
                if not cpu_busy and key[2]=='prepare' and t['status']=='pending':
                    launch(key,None,output,cfg,t,running);break
        gpus=sorted(free_gpus(),key=lambda x:-x[1])
        used={j.get('gpu') for j in running.values() if j.get('gpu') is not None}
        for gpu,free in gpus:
            if gpu in used or free<cfg['minimum_free_gpu_mib']:continue
            for key,t in tasks.items():
                method,scene,state=key
                if state in ['prepare','support'] or t['status']!='pending':continue
                dependency=tasks['common',scene,'support']['status']
                if dependency=='failed':t['status']='blocked_by_geometry_failure';continue
                if dependency!='complete':continue
                launch(key,gpu,output,cfg,t,running);break
        for key,t in tasks.items():
            if key[2]=='support' and any(tasks[m,key[1],'prepare']['status']=='failed' for m in cohorts[key[1]]) and t['status']=='pending':t['status']='failed'
            elif key[2] not in ['prepare','support'] and tasks['common',key[1],'support']['status']=='failed' and t['status']=='pending':t['status']='blocked_by_geometry_failure'
        write(output/'status.json',dict(pid=os.getpid(),updated=time.time(),started=started,
              running=[dict(method=k[0],scene=k[1],stage=k[2],pid=j['process'].pid,gpu=j.get('gpu')) for k,j in running.items()],
              tasks=[dict(method=k[0],scene=k[1],stage=k[2],**v) for k,v in tasks.items()],
              snapshot_meshes=sum(k[2]=='prepare' for k in tasks),failed=len(failed)))
        update(output)
        if not running and all(t['status'] not in ['pending','running'] for t in tasks.values()):break
        time.sleep(10)
    print('EVALUATION_FINISHED',len(tasks),'tasks; failed',len(failed),flush=True)

def launch(key,gpu,output,cfg,task,running):
    method,scene,stage=key
    command=[sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--worker',stage,
             '--method',method,'--scene',scene,'--appearance-steps',str(cfg['appearance_steps'])]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
    log=open(output/'logs'/f'{method}__{scene}__{stage}.log','w')
    p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
    running[key]=dict(process=p,log=log,gpu=gpu);task.update(status='running',pid=p.pid,gpu=gpu,started=time.time())
    print('START',key,'pid',p.pid,'gpu',gpu,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(BASE/'mesh_comparison'))
    p.add_argument('--scenes');p.add_argument('--methods');p.add_argument('--appearance-steps',type=int,default=5000)
    p.add_argument('--overwrite-evaluation',action='store_true')
    p.add_argument('--refresh-completed-meshes',action='store_true')
    p.add_argument('--worker',choices=['prepare','support',*STATES]);p.add_argument('--method',choices=[*METHODS,'common']);p.add_argument('--scene',choices=SCENES)
    a=p.parse_args();a.output=str(work_root(a.output));configure_sources(a.output)
    if a.worker:
        config=dict(DEFAULT);config['appearance_steps']=a.appearance_steps
        if a.worker=='prepare':
            from mesh_benchmark.cache import prepare
            prepare(a.output,a.method,a.scene,config)
        elif a.worker=='support':
            from mesh_benchmark.overlap import prepare_support
            prepare_support(a.output,a.scene,read(Path(a.output)/'protocol.json')['cohorts'][a.scene],config)
        else:
            from mesh_benchmark.score import score
            score(a.output,a.method,a.scene,a.worker,config,os.environ.get('CUDA_VISIBLE_DEVICES'))
    else:run(a)
