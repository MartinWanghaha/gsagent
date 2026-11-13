#!/usr/bin/env python3
"""Controlled full-scene PGSR surface-anchor/tetrahedron experiment on two GPUs."""
from pathlib import Path
import argparse
import copy
import fcntl
import gc
import os
import signal
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,read,write,identity,digest

PUBLIC=BASE/'mesh_comparison_whole'
DEFAULT=PUBLIC/'experiments/bag_depth_tetra_20261112'
CHILDREN=[]


def stop(out):
    if (Path(out)/'STOP_REQUESTED.json').exists():raise KeyboardInterrupt('Explicit stop request')


def state(out,label,**details):
    write(out/'status.json',dict(complete=label=='complete',stage=label,updated=time.time(),pid=os.getpid(),**details))
    (out/'STATUS.md').write_text('\n'.join(['# bag 深度法线锚定四面体：实时进度','',
        time.strftime('%Y-%m-%d %H:%M:%S %Z')+'；'+label,'',str(details),'',
        '训练与正式全场景结果不改变。85原分辨率训练视角、13全部留出视角。无固定细体素、无任意点数面数上限。',
        '', '[实验方案](PLAN.md) · [后台日志](campaign.log) · [全部对比图](figures/README.md) · [状态](status.json)'])+'\n')


def wait_check(out,name):
    p=out/'checks'/(name+'.json')
    while not p.exists():stop(out);state(out,'waiting for geometry implementation checks',check=name);time.sleep(15)
    r=read(p)
    if not r.get('complete'):raise ValueError('Implementation checks failed: '+name)
    if r.get('implementation') and r.get('module') and identity(HERE/r['module'])!=r['implementation']:
        raise ValueError('Checked implementation changed: '+name)


def launch(out,label,command,gpu=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=open(out/(label+'.log'),'a');p=subprocess.Popen(command,env=env,cwd=HERE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    CHILDREN.append((p,log));write(out/'workers'/(label+'.json'),dict(pid=p.pid,gpu=gpu,command=command,started=time.time()))
    print('DEPTH_TETRA_START',label,gpu,p.pid,flush=True);return p


def wait_for(out,workers,label,progress=None):
    while any(p.poll() is None for p in workers):
        stop(out)
        failed=[(p.pid,p.returncode) for p in workers if p.poll() not in [None,0]]
        if failed:raise RuntimeError(str(failed)+' '+label)
        state(out,label,workers=[dict(pid=p.pid,running=p.poll() is None) for p in workers],progress=progress() if progress else {})
        time.sleep(15)
    if any(p.returncode for p in workers):raise RuntimeError('Failed '+label)


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except (OSError,IndexError):return False


def wait_evidence(out):
    from export_depth_tetra_frames import prepare,progress,finalize
    prepare(out)
    while progress(out)['completed_frames']<85:
        stop(out);workers=[]
        for gpu in [0,1]:
            marker=out/f'evidence_gpu_{gpu}_launch.json'
            previous=read(marker) if marker.exists() else {}
            if alive(previous.get('pid',-1)):workers.append(previous);continue
            p=launch(out,'evidence_gpu_'+str(gpu),[sys.executable,'-u',str(HERE/'export_depth_tetra_frames.py'),'--output',str(out),'--worker',str(gpu)],gpu)
            previous=dict(pid=p.pid,gpu=gpu,started=time.time());write(marker,previous);workers.append(previous)
        state(out,'dual GPU full-frame frozen PGSR geometry export',progress=progress(out),workers=workers);time.sleep(15)
    # Claimed tasks commit before shared workers exit; do not kill valid exports.
    for gpu in [0,1]:
        pid=read(out/f'evidence_gpu_{gpu}_launch.json')['pid']
        while alive(pid):stop(out);time.sleep(2)
    return finalize(out)


def save_arrays(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz')
    np.savez(tmp,**arrays);tmp.replace(path)


def frame_anchors_worker(out,rank):
    from mesh_benchmark.depth_surface_anchors import build_frame_anchors
    out=Path(out);source=read(out/'frames/complete.json')
    for index,row in enumerate(source['frames']):
        stop(out);folder=out/'anchors/frames';folder.mkdir(parents=True,exist_ok=True)
        with (folder/f'{index:03d}.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            receipt=folder/f'{index:03d}.json'
            if receipt.exists():
                old=read(receipt)
                if identity(old['arrays']['path'])!=old['arrays'] or old['source']!=row['identity']:raise ValueError('Anchor frame changed')
                continue
            if identity(row['path'])!=row['identity']:raise ValueError('Source geometry changed')
            with np.load(row['path']) as a:frame={k:a[k].copy() for k in a.files}
            frame['frame_index']=index;arrays,detail=build_frame_anchors(frame)
            path=folder/f'{index:03d}.npz';save_arrays(path,arrays)
            write(receipt,dict(complete=True,arrays=identity(path),detail=detail,index=index,worker=rank,source=row['identity']))
            print('DEPTH_TETRA_ANCHORS',rank,index,len(arrays['positions']),flush=True)


def load_frames(out):
    result=[]
    for row in read(out/'frames/complete.json')['frames']:
        if identity(row['path'])!=row['identity']:raise ValueError('Frozen training evidence changed')
        with np.load(row['path']) as a:result.append({k:a[k].copy() for k in ['depth','normal','alpha','confidence','rgb','K','w2c']})
    return result


def query_worker(out,rank,kind):
    import torch
    from mesh_benchmark.depth_tetra_field import prepare_frames,query_field,refine_crossing_edges
    out=Path(out);torch.set_num_threads(4)
    while True:
        stop(out);free,total=torch.cuda.mem_get_info()
        if free>=4*1024**3:break
        write(out/f'{kind}/worker_{rank}.json',dict(stage='waiting for 4 GiB GPU headroom',pid=os.getpid(),updated=time.time(),free=int(free)))
        time.sleep(15)
    torch.cuda.set_per_process_memory_fraction(3*1024**3/total)
    with np.load(out/'anchors/merged.npz') as a:anchors={k:a[k].copy() for k in a.files}
    prepared=prepare_frames(load_frames(out),device='cuda:0')
    manifest=read(out/kind/'manifest.json')
    for task in manifest['tasks']:
        folder=out/kind/'parts';folder.mkdir(parents=True,exist_ok=True);index=task['index']
        with (folder/f'{index:04d}.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            stop(out);receipt=folder/f'{index:04d}.json'
            expected=dict(input=task['input'],implementation=identity(HERE/'mesh_benchmark/depth_tetra_field.py'),range=[task['start'],task['stop']])
            if receipt.exists():
                previous=read(receipt)
                if previous.get('request')!=expected or identity(previous['arrays']['path'])!=previous['arrays']:
                    raise ValueError('GPU query checkpoint inputs or implementation changed')
                continue
            if identity(task['input']['path'])!=task['input']:raise ValueError('GPU task input changed')
            with np.load(task['input']['path']) as a:arrays={k:a[k].copy() for k in a.files}
            if kind=='query':
                result=query_field(arrays['points'],anchors,prepared,device='cuda:0',
                    reference_points=arrays['reference_points'],reference_normals=arrays['reference_normals'],reference_scales=arrays['reference_scales'])
            else:
                count=len(arrays['left'])
                local=dict(crossing_edges=np.column_stack([np.arange(count),np.arange(count)+count]),
                    edge_reference_points=arrays['reference_points'],edge_reference_normals=arrays['reference_normals'],
                    edge_reference_scales=arrays['reference_scales'])
                result=refine_crossing_edges(np.vstack([arrays['left'],arrays['right']]),local,anchors,prepared,device='cuda:0')
                result['support']=result.pop('valid')
            diagnostics=result.pop('diagnostics',{})
            result={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
            path=folder/f'{index:04d}.npz';save_arrays(path,result)
            write(receipt,dict(complete=True,arrays=identity(path),diagnostics=diagnostics,worker=rank,index=index,range=[task['start'],task['stop']],request=expected))
            print('DEPTH_TETRA_GPU_TASK',kind,rank,index,task['stop']-task['start'],flush=True)


def task_manifest(out,kind,arrays,chunk=131072):
    folder=out/kind;folder.mkdir(exist_ok=True);tasks=[];count=len(next(iter(arrays.values())))
    for i,start in enumerate(range(0,count,chunk)):
        path=folder/'inputs'/f'{i:04d}.npz';part={k:v[start:start+chunk] for k,v in arrays.items()}
        if path.exists():
            with np.load(path) as old:
                if set(old.files)!=set(part) or any(not np.array_equal(old[k],v,equal_nan=True) for k,v in part.items()):
                    raise ValueError('Existing GPU input shard differs from current complete candidate arrays')
        else:save_arrays(path,part)
        tasks.append(dict(index=i,start=start,stop=min(start+chunk,count),input=identity(path)))
    result=dict(complete=True,tasks=tasks,count=count,kind=kind);write(folder/'manifest.json',result);return result


def run_tasks(out,kind):
    tasks=read(out/kind/'manifest.json')['tasks'];workers=[]
    if not tasks:raise ValueError('No supported '+kind+' tasks; refusing an empty experiment stage')
    for gpu in [0,1]:workers.append(launch(out,kind+'_gpu_'+str(gpu),[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(out),'--worker-stage',kind,'--rank',str(gpu)],gpu))
    wait_for(out,workers,'dual GPU '+kind,lambda:dict(completed=len(list((out/kind/'parts').glob('*.json'))),total=len(tasks)))
    collected=[]
    for task in tasks:
        r=read(out/kind/'parts'/f'{task["index"]:04d}.json')
        if identity(r['arrays']['path'])!=r['arrays']:raise ValueError('GPU task arrays changed')
        with np.load(r['arrays']['path']) as a:collected.append({k:a[k].copy() for k in a.files})
    keys=set(collected[0])
    if any(set(a)!=keys for a in collected):raise ValueError('GPU output schemas differ')
    return {k:np.concatenate([a[k] for a in collected],axis=0) for k in keys}


def run(out):
    out=Path(out).resolve();lock=open(out/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        stop(out);write(out/'execution.json',dict(driver=identity(Path(__file__)),plan=identity(out/'plan.json'),started=time.time()))
        wait_evidence(out);wait_check(out,'depth_surface_anchors')
        workers=[launch(out,'anchors_'+str(rank),[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(out),'--worker-stage','anchors','--rank',str(rank)]) for rank in [0,1]]
        wait_for(out,workers,'full-pixel error-validated surface anchoring',lambda:dict(completed=len(list((out/'anchors/frames').glob('*.json'))),total=85))
        wait_check(out,'depth_anchor_merge_native')
        from mesh_benchmark.depth_anchor_merge_native import merge_anchors
        parts=[]
        for index in range(85):
            r=read(out/'anchors/frames'/f'{index:03d}.json')
            if identity(r['arrays']['path'])!=r['arrays']:raise ValueError('Anchor source changed')
            with np.load(r['arrays']['path']) as a:parts.append({k:a[k].copy() for k in a.files})
        state(out,'merging error-bounded same-layer surface observations')
        anchors,detail=merge_anchors(parts);save_arrays(out/'anchors/merged.npz',anchors);write(out/'anchors/complete.json',dict(complete=True,detail=detail,arrays=identity(out/'anchors/merged.npz')))
        del parts;gc.collect();wait_check(out,'depth_tetra_field');wait_check(out,'triangulate_setup')
        from mesh_benchmark.depth_tetra_field import make_pivots,marching_tetrahedra
        anchor_count=len(anchors['positions']);upper_candidates=2*anchor_count
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        write(out/'tetra/candidate_preflight.json',dict(anchors=anchor_count,upper_candidates=upper_candidates,
            conservative_peak_unique_bytes=upper_candidates*320,available_host_bytes=available,no_point_cap=True))
        if upper_candidates*320>available*.6:
            raise MemoryError('Complete pivot generation exceeds safe host budget; no candidate or image has been discarded')
        pivots=make_pivots(anchors,include_centers=False);save_arrays(out/'tetra/pivots.npz',{k:v for k,v in pivots.items() if isinstance(v,np.ndarray)})
        count=len(pivots['points']);estimate=count*2000
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        write(out/'tetra/preflight.json',dict(points=count,conservative_estimated_bytes=estimate,available_host_bytes=available,no_point_cap=True,quality_reduction=False))
        if estimate>available*.6:raise MemoryError('Full candidate triangulation exceeds measured safe host budget; retained exact candidates, no quality reduction')
        save_arrays(out/'tetra/points.npz',dict(points=pivots['points']))
        address_space_limit=int(available*.6)
        write(out/'tetra/triangulate_budget.json',dict(address_space_limit_bytes=address_space_limit,
            estimate_is_heuristic_not_upper_bound=True,all_points_preserved=True,exhaustion_is_explicit_failure=True))
        p=launch(out,'triangulate',['/usr/bin/prlimit','--as='+str(address_space_limit),'--',sys.executable,'-u',str(HERE/'triangulate_depth_anchors.py'),'--input',str(out/'tetra/points.npz'),'--output',str(out/'tetra/cells.npy')])
        wait_for(out,[p],'full sparse CPU Delaunay tetrahedralization',lambda:dict(candidate_points=count))
        query_arrays={k:pivots[k] for k in ['points','reference_points','reference_normals','reference_scales']}
        task_manifest(out,'query',query_arrays);field=run_tasks(out,'query')
        if 'reference_ambiguous' in pivots:
            bad=pivots['reference_ambiguous'];field['support'][bad]=False;field['field'][bad]=np.nan
        save_arrays(out/'query/complete.npz',field)
        state(out,'extracting supported same-layer crossing topology')
        surface=marching_tetrahedra(pivots['points'],np.load(out/'tetra/cells.npy',mmap_mode='r'),field,anchors=anchors)
        edges=surface['crossing_edges'];faces=surface['faces']
        if not len(edges) or not len(faces):raise ValueError('No supported field crossings; topology checks cannot be bypassed')
        write(out/'tetra/topology.json',dict(complete=True,diagnostics=surface.get('diagnostics',{}),vertices=len(edges),faces=len(faces)))
        task_manifest(out,'edges',dict(left=pivots['points'][edges[:,0]],right=pivots['points'][edges[:,1]],
            reference_points=surface['edge_reference_points'],reference_normals=surface['edge_reference_normals'],reference_scales=surface['edge_reference_scales']))
        refined=run_tasks(out,'edges');vertices=refined['vertices'];colors=refined['colors'];supported=refined['support']
        keep=np.all(supported[faces],axis=1);faces=faces[keep]
        if not len(faces):raise ValueError('All triangle edge brackets failed independent root checks')
        used,inverse=np.unique(faces,return_inverse=True);faces=inverse.reshape(-1,3)
        from recompose.io import save_mesh
        save_mesh(out/'mesh/background.ply',vertices[used],faces,colors[used].astype('float32')/255.)
        write(out/'mesh/complete.json',dict(complete=True,vertices=len(used),faces=len(faces),
            mesh=identity(out/'mesh/background.ply'),edge_root_unsupported_faces=int((~keep).sum()),
            raw_topology_faces=len(keep),mesh_decimation=False,source_gaussians_updated=False,uv_texturing=False))
        from recompose.mesh_export import export_mesh_scene
        from recompose.io import fingerprint
        graph=copy.deepcopy(read(PUBLIC/'recomposition/bag/composition/scene.json'));plan=read(out/'plan.json')
        if graph['objects']!=plan['fixed_generated_objects']:raise ValueError('Fixed object source changed')
        graph['background'].update(mesh=fingerprint(out/'mesh/background.ply'),patches=[])
        export_mesh_scene(graph,out/'mesh/composed_world');graph['objects']=[];export_mesh_scene(graph,out/'mesh/background_world')
        wait_check(out,'depth_tetra_evaluation')
        p=launch(out,'evaluation',[sys.executable,'-u',str(HERE/'evaluate_depth_tetra_experiment.py'),'--output',str(out)])
        wait_for(out,[p],'all-method full-frame mesh rendering, reports and scores')
        result=read(out/'results.json')
        if not result.get('complete') or result.get('evaluation_views')!=13:raise ValueError('Final full comparison incomplete')
        state(out,'complete',figures='figures/README.md',reports=['reports/background.md','reports/composed.md'])
    except BaseException as error:
        state(out,'stopped by user' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error),saved_outputs_preserved=True)
        raise
    finally:
        for p,log in CHILDREN:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for p,log in CHILDREN:
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
            log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--worker-stage',choices=['anchors','query','edges']);parser.add_argument('--rank',default='0');args=parser.parse_args()
    if args.worker_stage=='anchors':frame_anchors_worker(args.output,args.rank)
    elif args.worker_stage:query_worker(args.output,args.rank,args.worker_stage)
    else:
        def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);run(args.output)
