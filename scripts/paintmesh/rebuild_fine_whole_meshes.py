#!/usr/bin/env python3
"""Rebuild stage 1 and inpaint meshes at .002/.008 over the whole scene.

Only scheduling/partition size adapts to resources. No coarse background patch,
mesh decimation, held-out RGB fitting, largest-component pruning or image resize.
"""
from pathlib import Path
import argparse
import fcntl
import gc
import importlib.util
import os
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,REPO,SCENES,read,write,identity,digest
from mesh_benchmark.layout import work_root,configure_sources
from mesh_benchmark.fine_tsdf import fuse,discover,VOXEL,TRUNCATION

PUBLIC=BASE/'mesh_comparison_whole'
ROOT=work_root(PUBLIC)
WORK=ROOT/'fine_whole'


def paths(scene,kind):
    graph=read(PUBLIC/'recomposition'/scene/'composition/scene.json')
    whole=read(ROOT/'models'/scene/'whole_mesh_complete.json')['parameters']
    if kind=='stage1':
        model=Path(graph['input_selection']['original_mesh']).parents[2]
        checkpoint=model/'point_cloud/iteration_30000/point_cloud.ply';iteration=30000
    else:
        checkpoint=Path(graph['background']['gaussians']['path']);model=checkpoint.parents[2]
        iteration=int(checkpoint.parent.name.split('_')[-1])
    return dict(scene=scene,kind=kind,source=graph['input_selection']['source'],model=str(model),
        checkpoint=str(checkpoint),iteration=iteration,train_names=whole['train_names'],max_depth=whole['max_depth'],
        voxel_size=VOXEL,sdf_trunc=TRUNCATION,alpha_min=.5,resolution=4,
        keep_all_connected_components=True,geometry_simplified=False)


def frames(scene,kind):
    params=paths(scene,kind);folder=WORK/kind/scene;folder.mkdir(parents=True,exist_ok=True)
    request=dict(parameters=params,checkpoint=identity(params['checkpoint']),config=identity(Path(params['model'])/'config.yaml'))
    marker=folder/'frames/complete.json'
    if marker.exists():
        old=read(marker)
        if old['request']!=request:raise ValueError('Fine frame request changed')
        assert all(identity(r['path'])==r['identity'] for r in old['frames'])
        return old['frames']
    if kind=='background':
        # Organized directories moved the old paths. Resolve the existing frame
        # by basename and check every file rather than rerendering a stale path.
        prior=ROOT/'models'/scene/'geometry_frames'
        provenance=read(ROOT/'models'/scene/'whole_mesh_complete.json')['request']
        if provenance['checkpoint']!=request['checkpoint'] or provenance['config']!=request['config']:
            raise ValueError('Background cached depth checkpoint/config no longer matches the selected model')
        old=read(prior/'complete.json');actual=[]
        for row in old['frames']:
            path=prior/Path(row['path']).name
            if not path.exists():raise FileNotFoundError(path)
            with np.load(path) as data:assert set(['depth','rgb','K','w2c'])<=set(data.files)
            actual.append(dict(name=row['name'],path=str(path),identity=identity(path)))
        assert {Path(x['name']).stem for x in actual}=={Path(x).stem for x in params['train_names']}
        write(marker,dict(complete=True,request=request,frames=actual,reused_exact_depths=True))
        return actual
    import torch
    torch.set_num_threads(4)
    free,total=torch.cuda.mem_get_info()
    if free<10*1024**3:raise RuntimeError('Insufficient free VRAM for unchanged stage-1 checkpoint')
    torch.cuda.set_per_process_memory_fraction(6*1024**3/total)
    edgs=REPO/'submodules/EDGS';sys.path.insert(0,str(edgs))
    spec=importlib.util.spec_from_file_location('fine_edgs_render',edgs/'render.py')
    render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
    model=Path(params['model']);cfg=render.load_training_config(model,source_path=Path(params['source']),images='images',resolution=4)
    cfg.gs.dataset.data_device='cpu';cfg.gs.dataset.model_path=str(model)
    render.bootstrap_gaussian_splatting()
    from scene import GaussianModel,Scene
    renderer=render.build_renderer({'backend':'pgsr','clamp_rgb':True})
    actual=[];destination=folder/'frames';destination.mkdir(exist_ok=True)
    with torch.no_grad():
        gaussian=GaussianModel(int(cfg.gs.sh_degree));scene_model=Scene(cfg.gs.dataset,gaussian,load_iteration=params['iteration'],shuffle=False)
        wanted={Path(n).stem for n in params['train_names']}
        views=sorted([c for c in scene_model.getTrainCameras() if Path(c.image_name).stem in wanted],key=lambda c:c.image_name)
        if {Path(v.image_name).stem for v in views}!=wanted:raise ValueError('Stage-1 fusion camera mismatch')
        background=torch.zeros(3,device='cuda')
        for i,c in enumerate(views):
            path=destination/(c.image_name+'.npz')
            package=renderer(c,gaussian,cfg.gs.pipe,background,return_plane=True)
            raw=package['plane_depth'].squeeze();alpha=package['rendered_alpha'].squeeze()
            valid=torch.isfinite(raw)&(raw>0)&(raw<=params['max_depth'])&(alpha>=params['alpha_min'])
            depth=torch.where(valid,raw,0).float().cpu().numpy()
            rgb=np.rint(package['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')
            K=render.camera_intrinsics(c,device='cpu').numpy();T=c.world_view_transform.T.detach().cpu().numpy()
            temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,depth=depth,rgb=rgb,K=K,w2c=T);temporary.replace(path)
            actual.append(dict(name=c.image_name,path=str(path),identity=identity(path)))
            print('FINE_STAGE1_RENDER',scene,i+1,len(views),flush=True)
            del package,raw,alpha,valid
    assert identity(params['checkpoint'])==request['checkpoint'],'Stage-1 checkpoint changed during full-scene rendering'
    assert identity(model/'config.yaml')==request['config'],'Stage-1 camera configuration changed during rendering'
    write(marker,dict(complete=True,request=request,frames=actual,reused_exact_depths=False))
    del views,c,gaussian,scene_model,renderer;gc.collect();torch.cuda.empty_cache()
    return actual


def worker(scene,kind,allocate_only=False):
    configure_sources(ROOT);folder=WORK/kind/scene
    folder.mkdir(parents=True,exist_ok=True)
    lock=None
    if not allocate_only:
        lock=open(folder/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (folder/'allocation.lock').open('a') as allocation_lock:
        fcntl.flock(allocation_lock,fcntl.LOCK_EX)
        records=frames(scene,kind);discover(folder,records)
    if not allocate_only:
        # Open3D's caching allocator otherwise retains the peak buffers of
        # earlier partitions. Release only unused GPU memory after each exact
        # integration; every triangle and voxel still uses the same algorithm.
        import open3d as o3d
        import mesh_benchmark.fine_tsdf as tsdf
        original=tsdf.integrate_partition
        def integrate_and_release(*args,**kwargs):
            result=original(*args,**kwargs)
            o3d.core.cuda.release_cache()
            return result
        tsdf.integrate_partition=integrate_and_release
        write(folder/'runtime.json',dict(launcher=identity(Path(__file__)),unused_cuda_cache_released=True,
            geometry_algorithm_unchanged=True,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        fuse(folder,records)


def status(tasks,stage,publication=None):
    for key,row in tasks.items():
        if row['status'] not in ['running','complete']:continue
        kind,scene=key.split('/');folder=WORK/kind/scene
        if (folder/'allocation.json').exists():
            allocation=read(folder/'allocation.json')
            receipts=[read(p) for p in (folder/'parts').glob('[0-9][0-9][0-9][0-9][0-9].json')]
            row.update(total_blocks=allocation['blocks'],completed_blocks=sum(r['core_blocks'] for r in receipts),
                completed_partitions=len(receipts),native_faces=sum(r['faces'] for r in receipts),
                monolithic_GiB=allocation['monolithic_attributes_GiB'])
        row['published']=(WORK/'publication'/scene/'evaluated.json').exists()
    record=dict(complete=stage=='complete',stage=stage,updated=time.time(),pid=os.getpid(),tasks=tasks,
        quality=dict(voxel_size=VOXEL,sdf_trunc=TRUNCATION,all_frames=True,all_depth_pixels=True,
            additional_image_downsampling=False,mesh_decimation=False,automatic_coarsening=False))
    if publication is not None:record['publication']=dict(pid=publication[0].pid,scene=publication[2])
    write(WORK/'status.json',record)
    lines=['# 细体素完整背景重建','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}；阶段：{stage}。','',
        '每场景分别使用第一阶段与inpaint自己的检查点，固定体素0.002、TSDF带宽0.008，保留全部连通块。所有原生1/4图像、训练相机和有效深度像素均参与。按同一全局体素格分区并保留邻域，不采用粗背景拼接。','',
        '| 任务 | 状态 | GPU | 已提取体素块 | 原生面数 | 已更新评估 |','|---|---|---|---:|---:|---|']
    if publication is not None:lines[4:4]=[f'当前正在合并/渲染评估：{publication[2]}；进程{publication[0].pid}。原局部细网格完整保留，新增细网格仅去重重叠面。','']
    for key,row in tasks.items():
        progress=f'{row.get("completed_blocks",0):,}/{row["total_blocks"]:,}' if row.get('total_blocks') else '分配/渲染准备中'
        lines.append(f'| {key} | {row["status"]} | {row.get("gpu","—")} | {progress} | {row.get("native_faces",0):,} | {row.get("published",False)} |')
    (PUBLIC/'docs/FINE_WHOLE_STATUS.md').write_text('\n'.join(lines)+'\n')


class AdoptedWorker:
    def __init__(self,pid,scene,kind):self.pid=pid;self.scene=scene;self.kind=kind
    def poll(self):
        try:
            command=Path(f'/proc/{self.pid}/cmdline').read_bytes()
            if b'rebuild_fine_whole_meshes.py' in command and b'--worker' in command:return None
        except FileNotFoundError:pass
        return 0 if (WORK/self.kind/self.scene/'complete.json').exists() else 1


class AdoptedPublication:
    def __init__(self,pid,scene):self.pid=pid;self.scene=scene;self.returncode=None
    def poll(self):
        try:
            command=Path(f'/proc/{self.pid}/cmdline').read_bytes()
            if b'publish_fine_whole_meshes.py' in command:return None
        except FileNotFoundError:pass
        self.returncode=0 if (WORK/'publication'/self.scene/'evaluated.json').exists() else 1
        return self.returncode


def run(scenes):
    WORK.mkdir(parents=True,exist_ok=True)
    lock=open(WORK/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks={kind+'/'+scene:dict(status='pending') for scene in scenes for kind in ['background','stage1']};active={};publication=None;publication_failed={}
    active_marker=WORK/'active_publication.json'
    if active_marker.exists():
        old=read(active_marker);adopted=AdoptedPublication(old['pid'],old['scene'])
        if adopted.poll() is None:
            publication=(adopted,open(WORK/('publish_'+old['scene']+'.log'),'a'),old['scene'])
            print('FINE_ADOPT_LIVE_PUBLICATION',old['scene'],old['pid'],flush=True)
    previous=read(WORK/'status.json').get('tasks',{}) if (WORK/'status.json').exists() else {}
    for key,row in tasks.items():
        kind,scene=key.split('/')
        if (WORK/kind/scene/'complete.json').exists():row['status']='complete'
        elif previous.get(key,{}).get('status')=='running':
            old=previous[key];adopted=AdoptedWorker(old['pid'],scene,kind)
            if adopted.poll() is None:
                row.update(old);active[key]=(adopted,open(WORK/(kind+'_'+scene+'.log'),'a'),old['gpu'])
                print('FINE_ADOPT_LIVE_WORKER',key,old['pid'],flush=True)
    while True:
        for key,(proc,log,gpu) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();tasks[key].update(status='complete' if code==0 else 'failed',exit_code=code);del active[key]
            print('FINE_TASK_END',key,code,flush=True)
        if publication is not None and publication[0].poll() is not None:
            proc,log,scene=publication;log.close()
            if proc.returncode:
                publication_failed[scene]=dict(exit_code=proc.returncode,log=str(WORK/('publish_'+scene+'.log')))
                write(WORK/'publication_failures.json',publication_failed)
                print('FINE_PUBLICATION_FAILED_EXTRACTION_CONTINUES',scene,proc.returncode,flush=True)
            else:print('FINE_SCENE_PUBLISHED',scene,flush=True)
            publication=None
            if active_marker.exists():active_marker.unlink()
        if publication is None:
            ready=[s for s in scenes if all(tasks[k+'/'+s]['status']=='complete' for k in ['background','stage1'])
                and not (WORK/'publication'/s/'evaluated.json').exists() and s not in publication_failed]
            if ready:
                scene=ready[0];log=open(WORK/('publish_'+scene+'.log'),'a')
                env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
                proc=subprocess.Popen([sys.executable,'-u',str(HERE/'publish_fine_whole_meshes.py'),'--scenes',scene],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                publication=(proc,log,scene);print('FINE_SCENE_PUBLICATION_START',scene,proc.pid,flush=True)
                write(active_marker,dict(pid=proc.pid,scene=scene))
        query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        used={v[2] for v in active.values()}
        available=[int(line.split(',')[0]) for line in query.stdout.splitlines() if int(line.split(',')[1])>=12288 and int(line.split(',')[0]) not in used]
        ram=next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        for gpu in available:
            pending=[k for k,v in tasks.items() if v['status']=='pending']
            if not pending or ram<24:break
            key=pending[0];kind,scene=key.split('/')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            log=open(WORK/(kind+'_'+scene+'.log'),'a')
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--scene',scene,'--kind',kind],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            active[key]=(proc,log,gpu);tasks[key].update(status='running',pid=proc.pid,gpu=gpu)
            print('FINE_TASK_START',key,gpu,proc.pid,flush=True);ram-=12
        status(tasks,'fine extraction',publication)
        if not active and not any(x['status']=='pending' for x in tasks.values()) and publication is None:break
        time.sleep(10)
    if any(x['status']!='complete' for x in tasks.values()):raise RuntimeError('Fine extraction failed; see per-scene logs')
    if publication_failed:
        status(tasks,'extraction complete; publication failures require repair')
        raise RuntimeError(f'All extraction tasks finished, publication failures: {list(publication_failed)}')
    status(tasks,'fine extraction complete; publication pending')
    subprocess.run([sys.executable,'-u',str(HERE/'publish_fine_whole_meshes.py')],check=True)
    status(tasks,'complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--allocate-only',action='store_true')
    p.add_argument('--scene',choices=SCENES);p.add_argument('--kind',choices=['stage1','background']);p.add_argument('--scenes',default=','.join(SCENES))
    p.add_argument('--prepare-backgrounds',action='store_true')
    a=p.parse_args()
    if a.prepare_backgrounds:
        for scene in a.scenes.split(','):
            if not (WORK/'background'/scene/'allocation.json').exists():worker(scene,'background',True)
    elif a.worker:worker(a.scene,a.kind,a.allocate_only)
    else:run(a.scenes.split(','))
