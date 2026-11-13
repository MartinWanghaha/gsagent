#!/usr/bin/env python3
"""Isolated complete bag FSSR pilot; production checkpoints and reports stay read-only."""
from pathlib import Path
import argparse
import copy
import fcntl
import gc
import json
import os
import resource
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,read,write,identity,digest
from mesh_benchmark.layout import work_root

PUBLIC=BASE/'mesh_comparison_whole'
DEFAULT=PUBLIC/'experiments/bag_surfel_fssr_20261112'
CHILDREN=[]


def host_free():
    return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))


def stage(out,label,**details):
    record=dict(complete=label=='complete',stage=label,pid=os.getpid(),updated=time.time(),**details)
    write(out/'status.json',record)
    (out/'STATUS.md').write_text('\n'.join(['# bag：Surfel＋FSSR 全图验证','',
        time.strftime('%Y-%m-%d %H:%M:%S %Z')+'；阶段：'+label,'',
        '85个完整训练视角、原生分辨率、13个全部留出视角；冻结最终3DGS和生成物体位姿。双卡用于证据导出与分帧表面观测处理；FSSR官方八叉树重建核心在CPU执行。',
        '', '本轮先做无UV/无额外顶点优化的最小验证，不覆盖正式全场景结果。', '',
        '详情：`'+str(details)+'`。','',
        '[后台日志](campaign.log) · [方案](PLAN.md) · [完整状态](status.json) · [对比图](figures/README.md)'])+'\n')


def check_stop(out):
    if (out/'STOP_REQUESTED.json').exists():raise KeyboardInterrupt('Explicit stop requested')


def gpu_free():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return dict(tuple(map(int,row.split(','))) for row in p.stdout.splitlines())


def launch(out,label,command,gpu=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=open(out/(label+'.log'),'a')
    child=subprocess.Popen(command,env=env,cwd=HERE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    CHILDREN.append((child,log))
    write(out/'workers'/(label+'.json'),dict(pid=child.pid,label=label,gpu=gpu,command=command,started=time.time()))
    print('FSSR_PILOT_START',label,'gpu',gpu,'pid',child.pid,flush=True)
    return child


def wait_children(out,children,label,progress=None):
    while any(p.poll() is None for p in children):
        check_stop(out)
        for p in children:
            if p.poll() not in [None,0]:raise RuntimeError(f'{label}: worker {p.pid} exited {p.returncode}')
        stage(out,label,workers=[dict(pid=p.pid,running=p.poll() is None) for p in children],progress=progress() if progress else {})
        time.sleep(15)
    for p in children:
        if p.returncode:raise RuntimeError(f'{label}: worker {p.pid} exited {p.returncode}')


def evidence_progress(out):
    result={}
    for folder in ['frames','teacher_views']:
        result[folder]=len(list((out/folder).glob('*.npz')))
    return result


def export_evidence(out):
    jobs={0:('training_geometry',[],'frames'),1:('held_out_geometry',['--teacher-views-only'],'teacher_views')}
    pending={gpu:job for gpu,job in jobs.items() if not (out/job[2]/'complete.json').exists()}
    running=[]
    while pending:
        check_stop(out);free=gpu_free()
        for gpu,(label,flags,folder) in list(pending.items()):
            if free.get(gpu,0)<10240 or host_free()<20*1024**3:continue
            p=launch(out,label,[sys.executable,'-u',str(HERE/'export_normal_fusion_frames.py'),'--output',str(out),'--source-kind','background',*flags],gpu)
            running.append(p);del pending[gpu]
        if pending:
            stage(out,'waiting for safe GPU capacity for geometry export',pending_gpus=list(pending),progress=evidence_progress(out));time.sleep(15)
    wait_children(out,running,'dual GPU full-frame geometry export',lambda:evidence_progress(out))
    frames=read(out/'frames/complete.json');teacher=read(out/'teacher_views/complete.json')
    if len(frames['frames'])!=85 or not frames.get('all_depth_rgb_camera_arrays_exact') or len(teacher['frames'])!=13:
        raise ValueError('Complete exact training evidence and 13 teacher views are required')


def checked_module(out,name):
    path=out/'checks'/(name+'.json')
    while not path.exists():
        check_stop(out);stage(out,'waiting for independent implementation checks',module=name);time.sleep(15)
    row=read(path)
    if not row.get('complete') or row.get('implementation')!=identity(HERE/row['module']):
        raise ValueError('Missing/stale code checks for '+name)


def export_surfels(out):
    checked_module(out,'surfel_export')
    from mesh_benchmark.surfel_export import prepare,finalize
    manifest=prepare(out,read(out/'frames/complete.json'))
    if (out/'surfels/complete.json').exists():return finalize(out)
    running=[]
    for gpu in [0,1]:
        while gpu_free().get(gpu,0)<10240 or host_free()<24*1024**3:
            check_stop(out);stage(out,'waiting for safe GPU capacity for surface samples',gpu=gpu);time.sleep(15)
        running.append(launch(out,'surfel_gpu_'+str(gpu),[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(out),'--surfel-worker',str(gpu)],gpu))
    def progress():
        receipts=[]
        for p in (out/'surfels').rglob('*.json'):
            if p.name in ['manifest.json','complete.json']:continue
            try:r=read(p)
            except (ValueError,OSError):continue
            if r.get('complete') and r.get('points') is not None:receipts.append(r)
        return dict(complete_frame_receipts=len(receipts),point_count=sum(r['points'] for r in receipts))
    wait_children(out,running,'dual GPU full-frame surface sampling and visibility reliability',progress)
    return finalize(out)


def inspect_mesh(path):
    import numpy as np
    from mesh_benchmark.color_export import read_mesh,positions
    mesh=read_mesh(path);vertices=mesh['vertex'].data;faces=np.asarray(mesh['face'].data['vertex_indices'])
    if not len(vertices) or not len(faces):raise ValueError('FSSR returned an empty mesh')
    if faces.shape!=(len(faces),3) or faces.min()<0 or faces.max()>=len(vertices):raise ValueError('Invalid FSSR face indices')
    if not all(k in vertices.dtype.names for k in ['red','green','blue']):raise ValueError('FSSR mesh has no native RGB')
    for start in range(0,len(vertices),262144):
        if not np.isfinite(positions(vertices[start:start+262144])).all():raise ValueError('FSSR mesh has nonfinite coordinates')
    return dict(vertices=len(vertices),faces=len(faces),has_native_color=True,mesh=identity(path),
        mesh_decimation=False,all_components_preserved=True,normal_orientation_not_assumed_globally_closed=True)


def reconstruct(out,samples):
    gate=out/'checks/fssr_official_setup/setup.json'
    while not gate.exists():
        check_stop(out);stage(out,'waiting for official FSSR build and smoke tests');time.sleep(15)
    setup=read(gate)
    if not setup.get('complete') or not setup.get('smoke_geometry_verified'):raise ValueError('Official FSSR build checks did not pass')
    target=out/'mesh/background.ply';target.parent.mkdir(exist_ok=True)
    files=[r['path'] for r in samples['records'] if r['points']]
    if len(samples['records'])!=85 or not samples.get('no_subsampling') or not files:
        raise ValueError('Expected all 85 complete source-frame records')
    for row in samples['records']:
        if identity(row['path'])!=row['identity']:raise ValueError('FSSR sample source changed')
    request=dict(binary=identity(setup['fssrecon']),git_sha=setup['git_sha'],samples=identity(out/'surfels/complete.json'),
        scale_factor=1.0,refine_octree=0,meshclean_used=False,all_components_preserved=True)
    receipt=out/'mesh/complete.json'
    if receipt.exists():
        previous=read(receipt)
        if previous['request']!=request or previous['mesh']!=identity(target):raise ValueError('Existing mesh request changed')
        return previous
    command=[setup['fssrecon'],'--scale-factor=1','--refine-octree=0',*files,str(target)]
    while host_free()<32*1024**3:
        check_stop(out);stage(out,'waiting for CPU memory for FSSR',available_bytes=host_free());time.sleep(15)
    cap=min(48*1024**3,int(host_free()*.7));env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='4')
    # Isolate a pathological reconstruction from host-wide memory exhaustion.
    bounded=['/usr/bin/prlimit','--as='+str(cap),'--',*command]
    write(out/'mesh/invocation.json',dict(command=command,request=request,host_address_space_limit_bytes=cap,started=time.time(),
        threads=8,note='Resource ceiling changes no point, pixel, reconstruction scale or triangle; exhaustion fails explicitly'))
    log=open(out/'fssrecon.log','a');started=time.time()
    p=subprocess.Popen(bounded,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    CHILDREN.append((p,log));write(out/'workers/fssr_cpu.json',dict(pid=p.pid,started=started,command=bounded))
    wait_children(out,[p],'official FSSR adaptive surface reconstruction',lambda:dict(samples=samples['points'],elapsed_seconds=time.time()-started))
    result=dict(complete=True,request=request,elapsed_seconds=time.time()-started,**inspect_mesh(target))
    write(receipt,result);return result


def export_assets(out):
    from recompose.mesh_export import export_mesh_scene
    from recompose.io import fingerprint
    graph=read(PUBLIC/'recomposition/bag/composition/scene.json')
    graph=copy.deepcopy(graph);graph['background'].update(mesh=fingerprint(out/'mesh/background.ply'),patches=[])
    export_mesh_scene(graph,out/'mesh/composed_world');graph['objects']=[]
    export_mesh_scene(graph,out/'mesh/background_world')


def run(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(out/'plan.json')
    if plan['scene']!='bag' or plan['train_views']!=85 or not plan.get('full_scene'):raise ValueError('Unexpected pilot plan')
    check_stop(out);write(out/'execution.json',dict(driver=identity(Path(__file__)),plan=identity(out/'plan.json'),started=time.time(),production_pipeline_modified=False))
    try:
        export_evidence(out)
        samples=export_surfels(out)
        mesh=reconstruct(out,samples)
        stage(out,'exporting complete background and fixed-object composed mesh',mesh_counts=mesh)
        export_assets(out)
        checked_module(out,'surfel_evaluator')
        p=launch(out,'evaluation',[sys.executable,'-u',str(HERE/'evaluate_surfel_mesh_experiment.py'),'--output',str(out)])
        wait_children(out,[p],'full native mesh comparison rendering and evaluation')
        results=read(out/'results.json')
        if not results.get('complete') or results.get('evaluation_views')!=13:raise ValueError('Full comparison remains incomplete')
        stage(out,'complete',mesh_counts=mesh,reports=['reports/background.md','reports/composed.md'],figures='figures/README.md')
    except BaseException as error:
        stage(out,'stopped by user' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error),complete_experiment=False,results_preserved=True)
        raise
    finally:
        for child,log in CHILDREN:
            if child.poll() is None:child.terminate()
        for child,log in CHILDREN:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--surfel-worker');args=parser.parse_args()
    if args.surfel_worker is not None:
        from mesh_benchmark.surfel_export import run_worker
        run_worker(args.output,args.surfel_worker,device='cuda:0')
    else:
        def stop(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        run(args.output)
