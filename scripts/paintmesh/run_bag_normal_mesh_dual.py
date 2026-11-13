#!/usr/bin/env python3
"""Two-GPU scheduling for the isolated, complete bag normal-mesh experiment."""
from pathlib import Path
import argparse
import fcntl
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import read,write,identity,digest
from experiment_bag_normal_mesh import DEFAULT,ROOT,update,host_available,disk_available,wait_checks,export_assets


def allocation_marker(out,index):return Path(out)/'gpu_allocation/frames'/f'{index:03d}.json'


def frame_request(out,index):
    manifest=read(Path(out)/'gpu_allocation/manifest.json');record=manifest['records'][index]
    return dict(frame=record['identity'],name=record['name'],
        algorithm=manifest['algorithm'],implementation=manifest['gpu_module'])


def check_frame(out,index):
    path=allocation_marker(out,index)
    if not path.exists():return False
    row=read(path)
    if row['request']!=frame_request(out,index):raise ValueError('Full-frame allocation request changed')
    if not row.get('complete') or identity(row['coordinates']['path'])!=row['coordinates']:
        raise ValueError('Saved frame activation changed')
    return True


def allocation_worker(out,worker_id):
    import torch
    from mesh_benchmark.normal_gpu_allocation import gpu_frame_blocks
    out=Path(out);manifest=read(out/'gpu_allocation/manifest.json')
    folder=out/'gpu_allocation/frames';folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    free,total=torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(4*1024**3/total)
    for index in range(manifest['completed_seed_frames'],len(manifest['records'])):
        if check_frame(out,index):continue
        with (folder/f'{index:03d}.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if check_frame(out,index):continue
            while True:
                free,_=torch.cuda.mem_get_info()
                if free>=6*1024**3 and host_available()>=16*1024**3 and disk_available(out)>=150*1024**3:break
                write(out/'gpu_allocation'/f'worker_{worker_id}.json',dict(stage='waiting for resources',pid=os.getpid(),frame=index,updated=time.time()))
                time.sleep(15)
            record=manifest['records'][index]
            if identity(record['path'])!=record['identity']:raise ValueError('Original range frame changed')
            state=dict(pid=os.getpid(),frame=index,total_frames=len(manifest['records']),worker_id=worker_id,
                physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),stage='full-frame GPU surface activation',updated=time.time())
            write(out/'gpu_allocation'/f'worker_{worker_id}.json',state)
            print('GPU_SURFACE_FRAME_START',worker_id,index,record['name'],flush=True)
            started=time.time()
            with np.load(record['path']) as a:frame={k:a[k].copy() for k in ['depth','K','w2c']}
            coordinates=gpu_frame_blocks(frame,device='cuda:0')
            destination=folder/f'{index:03d}.npy';temp=destination.with_suffix('.tmp.npy')
            np.save(temp,coordinates);temp.replace(destination)
            write(allocation_marker(out,index),dict(complete=True,request=frame_request(out,index),
                coordinates=identity(destination),blocks=len(coordinates),elapsed=time.time()-started,
                device=torch.cuda.get_device_name(0),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                full_frame=True,image_downsampling=False,all_continuous_surface_cells=True))
            print('GPU_SURFACE_FRAME_COMPLETE',worker_id,index,len(coordinates),'seconds',time.time()-started,flush=True)
            del frame,coordinates;gc.collect();torch.cuda.empty_cache()
    write(out/'gpu_allocation'/f'worker_{worker_id}.json',dict(stage='worker finished',pid=os.getpid(),updated=time.time()))


def prepare_allocation(out):
    out=Path(out);folder=out/'gpu_allocation';folder.mkdir(exist_ok=True)
    marker=folder/'manifest.json'
    if marker.exists():
        manifest=read(marker)
        if manifest['gpu_module']!=identity(HERE/'mesh_benchmark/normal_gpu_allocation.py'):raise ValueError('GPU allocation implementation changed')
        if identity(manifest['seed']['path'])!=manifest['seed']:raise ValueError('Frozen activation seed changed')
        return manifest
    checkpoint=read(out/'allocation/checkpoint.json');source=read(ROOT/'fine_whole/background/bag/frames/complete.json')
    expected=[identity(r['path']) for r in source['frames']]
    if checkpoint['request']['frames']!=expected:raise ValueError('Seed has different RGB-D inputs')
    count=checkpoint['completed_frames']
    if count!=len(checkpoint['frames']):raise ValueError('Incomplete allocation seed receipt')
    seed=out/'allocation/blocks.npy'
    manifest=dict(complete=True,records=source['frames'],completed_seed_frames=count,
        seed=identity(seed),seed_receipt=identity(out/'allocation/checkpoint.json'),
        seed_algorithm=checkpoint['request'],algorithm='Same complete continuous range triangles and conservative .002/.008 surface band; exact integer block union',
        gpu_module=identity(HERE/'mesh_benchmark/normal_gpu_allocation.py'),
        driver=identity(Path(__file__)),full_scene=True,roi=None)
    write(marker,manifest)
    return manifest


def allocation_progress(out):
    manifest=read(Path(out)/'gpu_allocation/manifest.json')
    start=manifest['completed_seed_frames']
    finished=sum(check_frame(out,i) for i in range(start,len(manifest['records'])))
    return dict(seed_frames=start,new_completed_frames=finished,completed_frames=start+finished,total_frames=len(manifest['records']))


def merge_allocation(out):
    from mesh_benchmark.normal_gpu_allocation import encode_block_keys,decode_block_keys
    out=Path(out);folder=out/'gpu_allocation';manifest=read(folder/'manifest.json')
    inputs=[manifest['seed']]
    for i in range(manifest['completed_seed_frames'],len(manifest['records'])):
        if not check_frame(out,i):raise ValueError(f'Frame {i} is not complete')
        inputs.append(read(allocation_marker(out,i))['coordinates'])
    request=dict(manifest=identity(folder/'manifest.json'),inputs=inputs,full_scene=True,
        union='Exact int64 union of all allocated surface blocks',voxel=.002,sdf_trunc=.008)
    marker=folder/'complete.json';destination=folder/'blocks.npy'
    if marker.exists():
        old=read(marker)
        if old['request']!=request or old['block_identity']!=identity(destination):raise ValueError('Full merged allocation changed')
        return old
    if host_available()<24*1024**3:raise RuntimeError('Insufficient host memory for exact full-scene union')
    arrays=[encode_block_keys(np.load(r['path'],mmap_mode='r')) for r in inputs]
    keys=np.unique(np.concatenate(arrays));del arrays
    coordinates=decode_block_keys(keys);del keys;gc.collect()
    temporary=destination.with_suffix('.tmp.npy');np.save(temporary,coordinates);temporary.replace(destination)
    result=dict(complete=True,request=request,request_id=digest(request),block_identity=identity(destination),
        blocks=len(coordinates),frames=len(manifest['records']),full_scene=True,roi=None)
    write(marker,result);return result


def prepare_cached_partitions(out,coordinates,request):
    """Avoid one HDD journal commit per regenerable partition-plan receipt.

    Only the scratch task JSON uses atomic buffered writes. Mesh results,
    paired completion receipts, and the final planning manifest still use the
    original durable writer. Existing plans are checked by the same queue code.
    This changes neither partition arrays nor the algorithm's input identity.
    """
    from mesh_benchmark import normal_dual_queue as queue
    folder=Path(out)/'dual_queue/tasks';original=queue.write
    def cached_write(path,value):
        path=Path(path)
        if path.parent!=folder:return original(path,value)
        path.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
        try:
            with os.fdopen(fd,'w') as handle:handle.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    queue.write=cached_write
    try:return queue.prepare_tasks(out,coordinates,request)
    finally:queue.write=original


class AdoptedWorker:
    def __init__(self,pid,flag):self.pid=pid;self.flag=flag;self.returncode=None
    def poll(self):
        try:command=Path(f'/proc/{self.pid}/cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:command=[]
        if os.fsencode(str(Path(__file__).resolve())) in command and os.fsencode(self.flag) in command:return None
        self.returncode=0
        return 0


def run_pair(out,stage,worker_args,is_complete,progress):
    """Keep one experiment worker per card; existing outside jobs stay running."""
    out=Path(out);running={};failures={};exhausted=set()
    flag='--allocation-worker' if stage=='surface allocation' else '--mesh-worker'
    launches=out/'dual_workers';launches.mkdir(exist_ok=True)
    prefix='allocation' if stage=='surface allocation' else 'mesh'
    for gpu in [0,1]:
        marker=launches/f'{prefix}_{gpu}.json'
        if not marker.exists():continue
        previous=read(marker);process=AdoptedWorker(previous['pid'],flag)
        if process.poll() is None:
            running[gpu]=(process,open(out/previous['log'],'a'))
            print('DUAL_GPU_WORKER_ADOPTED',stage,gpu,process.pid,flush=True)
    while not is_complete():
        for gpu,(process,log) in list(running.items()):
            code=process.poll()
            if code is None:continue
            log.close();del running[gpu]
            if code:
                failures[gpu]=failures.get(gpu,0)+1;exhausted.clear()
            else:exhausted.add(gpu)
            print('DUAL_GPU_WORKER_END',stage,gpu,code,flush=True)
        if any(n>=3 for n in failures.values()):raise RuntimeError(f'{stage}: repeated worker errors; checkpoints retained')
        if not running and not is_complete():exhausted.clear()
        query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        for line in query.stdout.splitlines():
            gpu,free=map(int,line.split(','))
            minimum=6144 if stage=='surface allocation' else 16384
            if gpu not in [0,1] or gpu in running or gpu in exhausted or free<minimum:continue
            if host_available()<32*1024**3 or disk_available(out)<150*1024**3:continue
            args=worker_args(gpu);env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            log=open(out/(('gpu_allocation' if stage=='surface allocation' else 'mesh_gpu')+f'_{gpu}.log'),'a')
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(out),*args],
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            running[gpu]=(process,log)
            write(launches/f'{prefix}_{gpu}.json',dict(pid=process.pid,gpu=gpu,stage=stage,
                log=Path(log.name).name,updated=time.time()))
            print('DUAL_GPU_WORKER_START',stage,gpu,process.pid,flush=True)
        update(out,'dual GPU '+stage,workers={str(g):dict(pid=r[0].pid) for g,r in running.items()},progress=progress())
        time.sleep(15)
    # Let already claimed partitions/frames commit before merging. No workers
    # are terminated, and all filesystem locks release naturally on exit.
    for gpu,(process,log) in running.items():
        while process.poll() is None:time.sleep(5)
        log.close()
        if process.returncode:raise RuntimeError(f'{stage} worker {gpu} failed after completion check')


def run(out):
    from mesh_benchmark.normal_dual_queue import prepare_tasks,progress,finalize
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ['normal_surface_fusion','normal_mesh_refinement','normal_mesh_evaluation','normal_dual_queue']:wait_checks(out,[name])
    evidence=read(out/'frames/complete.json')
    if not evidence.get('all_depth_rgb_camera_arrays_exact') or len(evidence['frames'])!=85:raise ValueError('Frozen normal evidence incomplete')
    execution=dict(driver=identity(Path(__file__)),gpu_allocator=identity(HERE/'mesh_benchmark/normal_gpu_allocation.py'),
        dual_queue=identity(HERE/'mesh_benchmark/normal_dual_queue.py'),geometry_plan=identity(out/'plan.json'),
        gpus=[0,1],all_frames=85,all_eval_views=13,voxel=.002,sdf_trunc=.008,
        production_pipeline_modified=False,full_scene=True,roi=None)
    marker=out/'dual_execution.json'
    if marker.exists() and read(marker)!=execution:raise ValueError('Dual execution inputs changed')
    write(marker,execution)
    manifest=prepare_allocation(out)
    if allocation_progress(out)['completed_frames']<85:
        run_pair(out,'surface allocation',lambda gpu:['--allocation-worker','--worker-id',str(gpu)],
            lambda:allocation_progress(out)['completed_frames']==85,lambda:allocation_progress(out))
    update(out,'exact merge of complete surface activation')
    allocation=merge_allocation(out)
    request=dict(geometry_plan=digest(read(out/'plan.json')),normal_evidence=identity(out/'frames/complete.json'),
        allocation=allocation['request_id'],normal_module=identity(HERE/'mesh_benchmark/normal_surface_fusion.py'),
        refinement_module=identity(HERE/'mesh_benchmark/normal_mesh_refinement.py'),
        voxel=.002,sdf_trunc=.008,full_scene=True,roi=None)
    update(out,'preparing fixed geometry partitions for both GPUs')
    prepare_cached_partitions(out,np.load(out/'gpu_allocation/blocks.npy'),request)
    if not progress(out)['complete']:
        run_pair(out,'normal fusion and refinement',lambda gpu:['--mesh-worker','--worker-id',str(gpu)],
            lambda:progress(out)['complete'],lambda:progress(out))
    finalize(out,request,evidencecount=85)
    update(out,'exporting complete meshes')
    export_assets(out)
    # The independent A renderer may already be running. Wait for its exact
    # source-validated result instead of racing over the same checkpoints.
    launch=out/'control_render_launch.json'
    if launch.exists():
        pid=read(launch)['pid']
        while Path(f'/proc/{pid}').exists():
            try:args=Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:break
            if b'render_control.py' not in args:break
            update(out,'waiting for full control mesh rendering',renderer_pid=pid);time.sleep(15)
    update(out,'rendering all complete scene comparisons and scoring')
    with (out/'evaluation.log').open('a') as log:
        subprocess.run([sys.executable,'-u',str(HERE/'evaluate_normal_mesh_experiment.py'),'--output',str(out)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(out/'results.json')
    if not result.get('complete') or result.get('evaluation_views')!=13:raise ValueError('Full evaluation did not complete')
    update(out,'complete',reports=['reports/background.md','reports/composed.md'],figures='figures/README.md')
    print('DUAL_GPU_BAG_NORMAL_EXPERIMENT_COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--worker-id',type=int,default=0)
    parser.add_argument('--allocation-worker',action='store_true');parser.add_argument('--mesh-worker',action='store_true')
    args=parser.parse_args()
    if args.allocation_worker:allocation_worker(args.output,args.worker_id)
    elif args.mesh_worker:
        from mesh_benchmark.normal_dual_queue import run_worker
        run_worker(args.output,str(args.worker_id))
    else:run(args.output)
