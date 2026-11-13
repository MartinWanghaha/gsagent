"""Parallel scheduling cannot lose partitions or publish incomplete N/R pairs."""
from pathlib import Path
import sys
import time
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mesh_benchmark.normal_dual_queue as queue
from mesh_benchmark.core import read,write,identity
from mesh_benchmark.fine_tsdf import partitions


def fixture(tmp_path,count=6005):
    frame=tmp_path/'frame.npz';d=np.full((4,4),2,dtype='float32')
    np.savez(frame,depth=d,rgb=np.full((4,4,3),[170,120,90],dtype='uint8'),
        K=np.array([[3.,0,2],[0,3.,2],[0,0,1.]]),w2c=np.eye(4),
        normal=np.broadcast_to([0,0,-1.],(4,4,3)),alpha=np.ones_like(d),confidence=np.ones_like(d))
    write(tmp_path/'frames/complete.json',dict(complete=True,frames=[dict(name='fixed',path=str(frame),identity=identity(frame))]))
    write(tmp_path/'plan.json',dict(source_kind='background'))
    coords=np.c_[np.arange(count),np.zeros(count),np.zeros(count)].astype('int32')
    request=dict(full_scene=True,voxel=.002,sdf_trunc=.008,source='same immutable inputs')
    return coords,request


def commit_mock(output,task,manifest,frames,worker_id,device='CPU:0'):
    time.sleep(.06)
    spec,nv,nf=queue.save_part(Path(output)/'N/parts',task['index'],
        np.array([[0,0,2],[1,0,2],[0,1,2]],dtype='float32'),np.array([[0,1,2]]),
        np.broadcast_to([.7,.5,.3],(3,3)))
    receipts={}
    for method in ['N','R']:
        if method=='R':spec,nv,nf=queue.save_part(Path(output)/'R/parts',task['index'],
            np.array([[0,0,2],[1,0,2],[0,1,2]],dtype='float32'),np.array([[0,1,2]]),np.broadcast_to([.7,.5,.3],(3,3)))
        path=Path(output)/method/'parts'/f'{task["index"]:05d}.json'
        write(path,dict(complete=True,signature=task['signature'],spec=spec,spec_identity=identity(spec['path']),vertices=nv,faces=nf))
        receipts[method]=identity(path)
    write(queue.queue_root(output)/'completed'/f'{task["index"]:05d}.json',dict(complete=True,
        request_id=manifest['request_id'],signature=task['signature'],index=task['index'],receipts=receipts,
        faces=nf,worker=worker_id))


def test_persisted_plans_equal_original_partitions_and_validate_resume(tmp_path):
    coords,request=fixture(tmp_path)
    result=queue.prepare_tasks(tmp_path,coords,request)
    assert result['total_core_blocks']==len(coords) and result['task_count']==2
    for task,(core,active) in zip(result['tasks'],partitions(coords)):
        with np.load(task['plan']['path']) as saved:
            np.testing.assert_array_equal(saved['core'],core)
            np.testing.assert_array_equal(saved['active'],active)
    before=[Path(row['plan']['path']).stat().st_mtime_ns for row in result['tasks']]
    assert queue.prepare_tasks(tmp_path,coords,request)==result
    assert [Path(row['plan']['path']).stat().st_mtime_ns for row in result['tasks']]==before
    with pytest.raises(ValueError,match='inputs changed'):
        queue.prepare_tasks(tmp_path,coords,dict(request,source='different'))


def test_two_processes_claim_each_partition_once_and_finalize_pair(tmp_path,monkeypatch):
    import multiprocessing
    coords,request=fixture(tmp_path)
    result=queue.prepare_tasks(tmp_path,coords,request)
    context=multiprocessing.get_context('fork');calls=context.Queue()
    def measured(output,task,manifest,frames,worker_id,device='CPU:0'):
        calls.put((task['index'],worker_id));commit_mock(output,task,manifest,frames,worker_id,device)
    monkeypatch.setattr(queue,'compute_task',measured)
    def worker(name):queue.run_worker(tmp_path,name,device='CPU:0',expected_frames=1)
    processes=[context.Process(target=worker,args=(f'gpu{i}',)) for i in range(2)]
    for p in processes:p.start()
    for p in processes:p.join(30);assert p.exitcode==0
    completed=[calls.get(timeout=1) for _ in range(2)]
    assert sorted(index for index,_ in completed)==[0,1]
    assert len({name for _,name in completed})==2
    assert queue.progress(tmp_path,validate=True)['complete']
    final=queue.finalize(tmp_path,request,evidencecount=1)
    assert final['N']['faces']==final['R']['faces']==2
    assert final['N']['vertices']==final['R']['vertices']==6
    assert final['N']['all_input_views']==1 and final['R']['full_scene']
    # Completed work never runs again, on either simulated device.
    monkeypatch.setattr(queue,'compute_task',lambda *a,**k:(_ for _ in ()).throw(AssertionError('duplicate task')))
    assert queue.run_worker(tmp_path,'gpu2',device='CPU:0',expected_frames=1)==0
    path=Path(final['R']['mesh_specs'][0]['path']);meta=read(path)
    np.save(meta['indices']['path'],np.array([[0,2,1]],dtype='<u4'))
    with pytest.raises(ValueError,match='Native triangle data changed'):queue.finalize(tmp_path,request,evidencecount=1)


def test_partial_failure_releases_claim_and_never_finalizes(tmp_path,monkeypatch):
    coords,request=fixture(tmp_path,count=3);result=queue.prepare_tasks(tmp_path,coords,request)
    task=result['tasks'][0]
    def interrupted(*args,**kwargs):
        write(tmp_path/'N/parts/00000.json',dict(complete=True,signature=task['signature'],faces=0,vertices=0,spec=None))
        raise RuntimeError('interrupted before R')
    monkeypatch.setattr(queue,'compute_task',interrupted)
    with pytest.raises(RuntimeError,match='before R'):queue.run_worker(tmp_path,'gpu0',device='CPU:0',expected_frames=1)
    assert not queue.progress(tmp_path)['complete']
    with pytest.raises(ValueError,match='not complete'):queue.finalize(tmp_path,request,evidencecount=1)
    assert not (tmp_path/'N/complete.json').exists() and not (tmp_path/'R/complete.json').exists()
    monkeypatch.setattr(queue,'compute_task',commit_mock)
    assert queue.run_worker(tmp_path,'gpu1',device='CPU:0',expected_frames=1)==1
    assert queue.finalize(tmp_path,request,evidencecount=1)['N']['faces']==1


def test_frustum_culling_preserves_all_potential_observations():
    common=dict(depth=np.ones((4,4),dtype='float32'),K=np.array([[4.,0,2],[0,4.,2],[0,0,1.]]),w2c=np.eye(4))
    away=dict(common,w2c=np.diag([1.,-1.,-1.,1.]))
    active=np.array([[0,0,62],[0,0,63]],dtype='int32')
    frames,receipt=queue.relevant_frames(active,[common,away])
    assert frames==[common] and receipt['considered_frames']==2 and receipt['integrated_frames']==1
    assert receipt['frustum_disjoint_frames']==1 and receipt['selected_indices']==[0]
    from mesh_benchmark.fine_tsdf import integrate_partition
    # A small known visible plane must produce exactly the same geometry when
    # a mathematically disjoint camera frame is omitted.
    rgb=np.full((4,4,3),[170,120,90],dtype='uint8')
    common=dict(common,depth=np.full((4,4),2,dtype='float32'),rgb=rgb)
    away=dict(away,rgb=rgb)
    v1,f1,c1=integrate_partition(active,[common,away],'CPU:0')
    v2,f2,c2=integrate_partition(active,[common],'CPU:0')
    def triangles(v,f):return sorted(tuple(sorted(tuple(np.round(x,7)) for x in t)) for t in v[f])
    assert triangles(v1,f1)==triangles(v2,f2)
    np.testing.assert_allclose(np.sort(c1,axis=0),np.sort(c2,axis=0),atol=1e-7)
