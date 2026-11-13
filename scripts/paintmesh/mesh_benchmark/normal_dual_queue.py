"""Exact shared work queue for two-GPU normal-mesh extraction.

Partition planning uses the existing fixed fine lattice and halo algorithm.
File locks claim independent partitions; each claim writes both N and R so a
crash can never publish a half-finished comparison as complete. Only scheduling
is parallel: every relevant original frame, pixel, voxel and triangle remains.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import fcntl
import gc
import os
import re
import shutil
import sys
import time
import numpy as np

from .core import read,write,identity,digest
from .fine_tsdf import VOXEL,TRUNCATION,BLOCK,partitions,retain_owned_faces
from .full_surface_fusion import save_part
from .stream_cache import outside_frustum


def queue_root(output):return Path(output)/'dual_queue'


def array_signature(coordinates):
    import hashlib
    a=np.ascontiguousarray(coordinates,dtype='<i4')
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(memoryview(a)).hexdigest())


def prepare_tasks(output,coordinates,request):
    """Persist the SAME core/halo partitions as ``fine_tsdf.partitions``.

    A complete manifest is the barrier: workers must not claim partial plans.
    Existing per-task plans survive interrupted preparation and are validated
    against their full arrays, not merely their filenames or partition counts.
    """
    output=Path(output);folder=queue_root(output);folder.mkdir(parents=True,exist_ok=True)
    planning=open(folder/'planning.lock','a');fcntl.flock(planning,fcntl.LOCK_EX)
    coords=np.asarray(coordinates,dtype='int32')
    if coords.ndim!=2 or coords.shape[1]!=3 or not len(coords):raise ValueError('Nonempty N,3 global block coordinates required')
    source=dict(request=request,coordinates=array_signature(coords),voxel=VOXEL,sdf_trunc=TRUNCATION,
        partitioning=dict(core_blocks=6000,max_active_blocks=24000,halo_blocks=1),
        queue_implementation=identity(Path(__file__)))
    request_id=digest(source);marker=folder/'manifest.json'
    if marker.exists():
        previous=read(marker)
        if previous['request_id']!=request_id:raise ValueError('Dual queue inputs changed')
        for row in previous['tasks']:
            if identity(row['plan']['path'])!=row['plan']:raise ValueError('Partition plan file changed')
        return previous
    task_folder=folder/'tasks';task_folder.mkdir(exist_ok=True);tasks=[]
    for index,(core,active) in enumerate(partitions(coords)):
        path=task_folder/f'{index:05d}.npz';receipt=path.with_suffix('.json')
        signature=digest(dict(request=request,core=core.tolist(),active=active.tolist()))
        reused=False
        if path.exists() and receipt.exists():
            old=read(receipt)
            if old['signature']!=signature or old['request_id']!=request_id:raise ValueError('Existing partition plan belongs to different inputs')
            with np.load(path) as arrays:
                if not np.array_equal(arrays['core'],core) or not np.array_equal(arrays['active'],active):
                    raise ValueError('Persisted core/halo coordinates changed')
            reused=True
        if not reused:
            temp=path.with_suffix('.tmp.npz');np.savez(temp,core=core,active=active);temp.replace(path)
            write(receipt,dict(signature=signature,request_id=request_id))
        tasks.append(dict(index=index,signature=signature,plan=identity(path),core_blocks=len(core),active_blocks=len(active)))
        if index%50==0:print('NORMAL_DUAL_PLAN',index+1,'core blocks',sum(t['core_blocks'] for t in tasks),len(coords),flush=True)
    result=dict(complete=True,version=1,request_id=request_id,request=request,source=source,tasks=tasks,
        task_count=len(tasks),total_core_blocks=sum(t['core_blocks'] for t in tasks),created=time.time(),
        geometry_algorithm_unchanged=True,full_scene=True,roi=None)
    if result['total_core_blocks']!=len(coords):raise ValueError('Partition plan does not cover the global block array')
    write(marker,result);return result


def manifest(output):
    path=queue_root(output)/'manifest.json'
    result=read(path)
    if not result.get('complete'):raise ValueError('Dual queue planning is incomplete')
    if result['source']['queue_implementation']!=identity(Path(__file__)):
        raise ValueError('Queue implementation changed after planning')
    return result


def _check_native(spec,expected):
    if spec is None:
        if expected:raise ValueError('Nonempty task lacks triangle source')
        return None
    source=read(spec['path'])
    if source['face_count']!=expected:raise ValueError('Triangle count differs from completed receipt')
    for key,stamp in [('vertices','vertex_identity'),('indices','index_identity')]:
        if identity(source[key]['path'])!=source[stamp]:raise ValueError('Native triangle data changed')
    return source


def completed_task(output,task,queue_id,validate_buffers=True):
    path=queue_root(output)/'completed'/f'{task["index"]:05d}.json'
    if not path.exists():return False
    done=read(path)
    if not done.get('complete') or done['request_id']!=queue_id or done['signature']!=task['signature']:
        raise ValueError('Completed queue task request changed')
    counts={}
    for method in ['N','R']:
        receipt=Path(output)/method/'parts'/f'{task["index"]:05d}.json'
        row=read(receipt)
        if identity(receipt)!=done['receipts'][method]:raise ValueError('Committed N/R task receipt changed')
        if not row.get('complete') or row['signature']!=task['signature']:raise ValueError('Wrong N/R partition signature')
        if row['spec']:
            if identity(row['spec']['path'])!=row['spec_identity']:raise ValueError('Native mesh descriptor changed')
        if validate_buffers:_check_native(row['spec'],row['faces'])
        counts[method]=(row['vertices'],row['faces'])
    if counts['N']!=counts['R']:raise ValueError('N/R topology counts differ')
    return True


@contextmanager
def claim_task(output,task,queue_id):
    """Return a claimed task, or None if another worker owns/finished it."""
    folder=queue_root(output)/'locks';folder.mkdir(parents=True,exist_ok=True)
    lock=open(folder/f'{task["index"]:05d}.lock','a')
    acquired=False
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
        except BlockingIOError:yield None;return
        if completed_task(output,task,queue_id,validate_buffers=False):yield None;return
        yield task
    finally:
        if acquired:fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


def relevant_frames(active,frames,max_displacement=TRUNCATION):
    """Skip only entire views whose frusta cannot touch any queried position.

    This includes all VBG positions and possible mesh refinement positions.
    Frustum overlap is only necessary, never an accuracy/angle/distance test.
    A conservative floating-point margin prevents rounding into a skipped view.
    """
    length=VOXEL*BLOCK
    bounds=np.array([np.asarray(active).min(0)*length,(np.asarray(active).max(0)+1)*length],dtype='float64')
    margin=max_displacement+1e-5*(1+float(np.abs(bounds).max()))
    bounds+=np.array([[-margin]*3,[margin]*3])
    selected=[];indices=[]
    for i,frame in enumerate(frames):
        height,width=frame['depth'].shape
        camera=dict(K=frame['K'],w2c=frame['w2c'],width=width,height=height)
        if not outside_frustum(bounds,camera):selected.append(frame);indices.append(i)
    return selected,dict(considered_frames=len(frames),integrated_frames=len(selected),
        frustum_disjoint_frames=len(frames)-len(selected),selected_indices=indices,
        selection='Conservative full active-block AABB plus refinement displacement and numeric margin; only entirely disjoint camera frusta omitted',
        bounds=bounds.tolist(),frame_sampling=False,pixels_subsampled=False)


def wait_resources(output,active_blocks,worker_id,device):
    if not device.startswith('CUDA'):return
    import torch
    needed=max(16*1024**3,active_blocks*BLOCK**3*100+5*1024**3)
    while True:
        free,_=torch.cuda.mem_get_info()
        host=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        disk=shutil.disk_usage(output).free
        if free>=needed and host>=24*1024**3 and disk>=150*1024**3:return
        write(queue_root(output)/'workers'/(worker_id+'.json'),dict(stage='waiting for resources',pid=os.getpid(),
            updated=time.time(),free_vram=int(free),needed_vram=int(needed),host_free=host,disk_free=disk))
        print('NORMAL_DUAL_WAIT_RESOURCES',worker_id,int(free),needed,flush=True);time.sleep(30)


def load_frames(output,expected_frames=85):
    evidence=read(Path(output)/'frames/complete.json')
    if not evidence.get('complete') or len(evidence['frames'])!=expected_frames:raise ValueError('All prescribed normal-evidence frames are required')
    frames=[];fields=['depth','rgb','K','w2c','normal','alpha','confidence']
    for row in evidence['frames']:
        if row.get('identity') and identity(row['path'])!=row['identity']:raise ValueError('Normal-evidence source changed')
        with np.load(row['path']) as arrays:frames.append({key:arrays[key].copy() for key in fields})
    return frames


def compute_task(output,task,queue,frames,worker_id,device='CUDA:0'):
    from .normal_surface_fusion import integrate_normal_partition
    from .normal_mesh_refinement import refine_vertices_anchored
    output=Path(output);path=Path(task['plan']['path'])
    if identity(path)!=task['plan']:raise ValueError('Partition plan changed')
    with np.load(path) as arrays:core=arrays['core'].copy();active=arrays['active'].copy()
    signature=digest(dict(request=queue['request'],core=core.tolist(),active=active.tolist()))
    if signature!=task['signature']:raise ValueError('Partition contents differ from their plan')
    wait_resources(output,len(active),worker_id,device)
    participating,selection=relevant_frames(active,frames)
    started=time.time()
    if participating:
        vertices,faces,colors=integrate_normal_partition(active,participating,device=device,chunk_blocks=64)
        owned=retain_owned_faces(vertices,faces,core)
        moved,refinement=refine_vertices_anchored(vertices,participating,faces=faces,band=TRUNCATION,
            iterations=3,device=device,max_step=.001,max_displacement=TRUNCATION)
        if moved.shape!=vertices.shape or not np.isfinite(moved).all():raise ValueError('Invalid refined positions')
        if len(vertices) and np.max(np.linalg.norm(moved-vertices,axis=1))>TRUNCATION+1e-5:
            raise ValueError('Refinement exceeded displacement limit')
    else:
        vertices=np.empty((0,3),dtype='float32');moved=vertices.copy();faces=np.empty((0,3),dtype='int64')
        colors=vertices.copy();owned=np.empty(0,dtype=bool)
        refinement=dict(iterations=3,vertices=0,faces_unchanged=True,vertex_count_unchanged=True,
            reason='No camera frustum intersects the entire expanded block bounds')
    receipts={};counts={}
    for method,positions in [('N',vertices),('R',moved)]:
        spec=None;nv=nf=0
        if owned.any():spec,nv,nf=save_part(output/method/'parts',task['index'],positions,faces[owned],colors)
        receipt=output/method/'parts'/f'{task["index"]:05d}.json'
        row=dict(complete=True,signature=signature,queue_id=queue['request_id'],spec=spec,
            spec_identity=identity(spec['path']) if spec else None,vertices=nv,faces=nf,
            core_blocks=len(core),halo_blocks=len(active),ownership='Original, unrefined centroid block',
            refinement=refinement if method=='R' else None,frame_selection=selection,
            worker=worker_id,pid=os.getpid(),device=device,physical_cuda_device=os.environ.get('CUDA_VISIBLE_DEVICES'),
            elapsed=time.time()-started)
        write(receipt,row);receipts[method]=identity(receipt);counts[method]=(nv,nf)
    if counts['N']!=counts['R']:raise ValueError('N/R topology differs')
    write(queue_root(output)/'completed'/f'{task["index"]:05d}.json',dict(complete=True,
        request_id=queue['request_id'],signature=signature,index=task['index'],receipts=receipts,
        worker=worker_id,updated=time.time(),faces=counts['N'][1]))
    print('NORMAL_DUAL_TASK_COMPLETE',worker_id,task['index'],'faces',counts['N'][1],flush=True)
    del vertices,moved,faces,colors;gc.collect()
    if device.startswith('CUDA'):
        import torch,open3d as o3d
        torch.cuda.empty_cache();o3d.core.cuda.release_cache()


def run_worker(output,worker_id,device='CUDA:0',expected_frames=85):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',worker_id):raise ValueError('Unsafe worker identifier')
    output=Path(output);queue=manifest(output)
    if device.startswith('CUDA'):
        import torch
        torch.set_num_threads(4)
    frames=load_frames(output,expected_frames)
    status=queue_root(output)/'workers'/(worker_id+'.json');finished=0
    for task in queue['tasks']:
        with claim_task(output,task,queue['request_id']) as claimed:
            if claimed is None:continue
            write(status,dict(stage='running',worker=worker_id,pid=os.getpid(),partition=task['index'],
                completed_by_worker=finished,device=device,physical_cuda_device=os.environ.get('CUDA_VISIBLE_DEVICES'),updated=time.time()))
            try:compute_task(output,task,queue,frames,worker_id,device)
            except Exception as error:
                write(queue_root(output)/'failures'/f'{task["index"]:05d}_{worker_id}.json',dict(
                    partition=task['index'],worker=worker_id,pid=os.getpid(),updated=time.time(),error=repr(error),
                    geometry_quality_changed=False,resumable=True))
                write(status,dict(stage='failed',worker=worker_id,pid=os.getpid(),partition=task['index'],error=repr(error),updated=time.time()))
                raise
            finished+=1
    write(status,dict(stage='complete',worker=worker_id,pid=os.getpid(),completed_by_worker=finished,
        device=device,updated=time.time()))
    print('NORMAL_DUAL_WORKER_COMPLETE',worker_id,finished,flush=True)
    return finished


def progress(output,validate=False):
    queue=manifest(output);completed=0;faces=0
    for task in queue['tasks']:
        path=queue_root(output)/'completed'/f'{task["index"]:05d}.json'
        if path.exists():
            if validate:completed_task(output,task,queue['request_id'])
            row=read(path)
            if row['request_id']!=queue['request_id'] or row['signature']!=task['signature']:raise ValueError('Wrong completed task in progress scan')
            completed+=1;faces+=row['faces']
    workers={p.stem:read(p) for p in (queue_root(output)/'workers').glob('*.json')}
    return dict(complete=completed==queue['task_count'],completed_partitions=completed,total_partitions=queue['task_count'],
        remaining_partitions=queue['task_count']-completed,faces_per_method=faces,workers=workers,request_id=queue['request_id'])


def finalize(output,request=None,evidencecount=85):
    output=Path(output);queue=manifest(output)
    if request is not None and request!=queue['request']:raise ValueError('Finalize request differs from queue')
    evidence=read(output/'frames/complete.json')
    if not evidence.get('complete') or len(evidence['frames'])!=evidencecount:raise ValueError('Complete original-frame evidence required')
    specs={key:[] for key in ['N','R']};faces={key:0 for key in specs};vertices={key:0 for key in specs}
    for task in queue['tasks']:
        if not completed_task(output,task,queue['request_id']):raise ValueError(f'Partition {task["index"]} is not complete in both N/R')
        for method in specs:
            row=read(output/method/'parts'/f'{task["index"]:05d}.json')
            if row['spec']:specs[method].append(row['spec'])
            faces[method]+=row['faces'];vertices[method]+=row['vertices']
    if faces['N']!=faces['R'] or vertices['N']!=vertices['R']:raise ValueError('N/R topology totals differ')
    plan=read(output/'plan.json')
    results={}
    # Both methods are constructed fully before either completion marker is
    # written. Consumers still require all three A/N/R complete markers.
    for method in specs:
        results[method]=dict(complete=True,request=queue['request'],queue_id=queue['request_id'],method=method,
            mesh_specs=specs[method],faces=faces[method],vertices=vertices[method],
            full_scene=True,roi=None,all_input_views=evidencecount,geometry_simplified=False,images_downsampled=False,
            colors_optimized=False,topology_preserved_between_N_and_R=True,source_kind=plan['source_kind'],
            scheduling='Exclusive file-locked shared partition queue; geometry algorithm unchanged')
    for method,row in results.items():write(output/method/'complete.json',row)
    write(queue_root(output)/'complete.json',dict(complete=True,request_id=queue['request_id'],
        task_count=queue['task_count'],faces_per_method=faces,updated=time.time()))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker-id',required=True);parser.add_argument('--device',default='CUDA:0')
    args=parser.parse_args();run_worker(args.output,args.worker_id,args.device)
