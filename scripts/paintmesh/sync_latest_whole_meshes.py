#!/usr/bin/env python3
"""Replace published objects and rescore mesh geometry with verified cache reuse.

This stage deliberately does not select an RGB renderer. Geometry, source
selection, masks and the full-background meshes remain independently auditable.
"""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,REPO,SCENES,METHODS,STATES,read,write,identity,digest,cameras
from mesh_benchmark.layout import work_root,public_root,configure_sources
from recompose.io import fingerprint

SOURCE=REPO/'output/paintmesh/Inpaint360GS/recomposition'


def same_selection(old,new):
    """Whole extraction replaces only the mesh path in the input selection."""
    omit={'background_mesh'}
    return {k:v for k,v in old.items() if k not in omit}=={k:v for k,v in new.items() if k not in omit}


def preflight(root,scene):
    old=read(root/'recomposition'/scene/'composition/scene.json')
    new=read(SOURCE/scene/'composition/scene.json')
    if not same_selection(old['input_selection'],new['input_selection']):
        raise ValueError(f'Scene selection changed: {scene}; cannot rebind existing reference data')
    extraction=read(root/'models'/scene/'whole_mesh_complete.json')
    if extraction['request']['checkpoint']!=identity(new['background']['gaussians']['path']):
        raise ValueError(f'Background checkpoint changed: {scene}; re-extraction required')
    if Path(extraction['path']).resolve()!=Path(old['background']['mesh']['path']).resolve():
        raise ValueError(f'Published background does not match extraction: {scene}')
    info=read(root/'common'/scene/'complete.json')
    request=info['request']
    if request['split']!=identity(BASE/'_data'/scene/'split.json'):
        raise ValueError(f'Camera split changed: {scene}')
    if any(ref!=identity(ref['path']) for ref in request['masks']):
        raise ValueError(f'Instance masks changed: {scene}')
    retained=[]
    for method in METHODS:
        for state in STATES:
            if method=='paintmesh' and state=='recomposition':continue
            path=root/'cache'/method/scene/state/'complete.json';meta=read(path)
            if meta['request']['common']!=info['request_id']:
                raise ValueError(f'Unmatched common-cache identity: {path}')
            specs=meta['request']['specs']
            if method=='paintmesh':wanted=old['background']['mesh']['path']
            else:
                marker=read(BASE/method/scene/'mesh_complete.json')
                if not marker['complete']:raise ValueError(f'Incomplete baseline: {method}/{scene}')
                wanted=marker.get('geometry_path',marker['path'])
            if len(specs)!=1 or specs[0]['identity']!=identity(wanted) or specs[0]['transform'] is not None:
                raise ValueError(f'Retained mesh geometry changed: {method}/{scene}/{state}')
            retained.append(dict(method=method,state=state,path=str(path)))
    if sorted(n['id'] for n in old['objects'])!=sorted(n['id'] for n in new['objects']):
        raise ValueError(f'Object IDs changed: {scene}')
    for node in new['objects']:
        for key in ['mesh','textured_mesh','pose_report']:
            if fingerprint(node[key]['path'])['sha256']!=node[key]['sha256']:
                raise ValueError(f'Changed generated asset: {scene}/{node["id"]}/{key}')
    return dict(scene=scene,source_scene=fingerprint(SOURCE/scene/'composition/scene.json'),
                source_status=new['status'],retained=retained,old_common=info['request_id'],
                original_objects={n['id']:n['mesh']['sha256'] for n in old['objects']},
                latest_objects={n['id']:n['mesh']['sha256'] for n in new['objects']})


def invalidate_changed_outputs(root,scene):
    """A changed mesh invalidates its field/renders and every common-domain score."""
    for path in [root/'cache/paintmesh'/scene/'recomposition',root/'recomposition/paintmesh'/scene]:
        if path.exists():shutil.rmtree(path)
    for state in STATES:
        support=root/'common'/scene/(state+'_support')
        if support.exists():shutil.rmtree(support)
        support=root/'common'/scene/(state+'_native_support')
        if support.exists():shutil.rmtree(support)
        for method in METHODS:
            for path in [root/state/method/scene/'metrics.json',root/'cache'/method/scene/state/'geometry_metrics.json']:
                if path.exists():path.unlink()


def publish_and_rebind(root,row):
    from mesh_benchmark.whole import publish_whole
    scene=row['scene'];publish_whole(root,scene)
    info_path=root/'common'/scene/'complete.json';info=read(info_path)
    if info['request_id']!=row['old_common']:raise ValueError('Common cache changed during publication')
    request=copy.deepcopy(info['request'])
    request['selection']=identity(root/'recomposition'/scene/'composition/scene.json')
    new_id=digest(request)
    info.update(request=request,request_id=new_id)
    write(info_path,info)
    for item in row['retained']:
        path=Path(item['path']);meta=read(path)
        meta['request']['common']=new_id
        if item['method']=='paintmesh':meta['scene_status']=row['source_status']
        meta['verified_reuse']=dict(reason='Mesh, masks, split and source selection unchanged; only generated objects replaced',
             prior_common=row['old_common'],current_common=new_id,updated=time.time())
        write(path,meta)
    invalidate_changed_outputs(root,scene)
    # Buffers from previously generated objects are no longer part of the scene.
    world=root/'recomposition'/scene/'composition/world'
    active={Path(b['uri']).name for b in read(world/'mesh_scene.gltf')['buffers']}
    for file in (world/'mesh_buffers').glob('object_*.bin'):
        if file.name not in active:file.unlink()
    return dict(new_common=new_id,completed=time.time())


def status(root,manifest,stage):
    manifest.update(stage=stage,pid=os.getpid(),updated=time.time())
    write(root/'sync_latest.json',manifest)
    doc=['# 最新 recomposition 同步','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}；阶段：{stage}。','',
         '覆盖本目录的组合物体、位姿、共同评分域和派生结果。完整背景及四个基线经过身份核对后复用。几何协议保持有效表面 CD/F1；RGB 渲染器口径另外记录。','',
         '| 场景 | 最新来源状态 | 资产同步 | 几何重评估 |','|---|---|---|---|']
    for scene in SCENES:
        row=manifest['scenes'].get(scene,{})
        doc.append(f'| {scene} | {row.get("source_status","-")} | {row.get("published",False)} | {row.get("geometry_complete",False)} |')
    (public_root(root)/'docs/SYNC_LATEST.md').write_text('\n'.join(doc)+'\n')


def worker(root,scene,stage):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    command=[sys.executable,str(HERE/'evaluate_recomposition_meshes.py'),'--output',str(root),'--worker',stage,'--scene',scene]
    if stage=='prepare':command+=['--method','paintmesh']
    log=root/'logs'/f'sync_latest__{scene}__{stage}.log'
    with log.open('w') as stream:
        subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)


def run(output,assets_only=False):
    root=work_root(output);configure_sources(root)
    locks=[]
    for name in ['sync_latest.lock','whole_campaign.lock','campaign.lock']:
        lock=open(root/name,'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    rows=[preflight(root,scene) for scene in SCENES]
    request_id=digest([row['source_scene'] for row in rows])
    path=root/'sync_latest.json'
    manifest=read(path) if path.exists() else dict(request_id=request_id,started=time.time(),scenes={})
    if manifest['request_id']!=request_id:manifest=dict(request_id=request_id,started=time.time(),scenes={})
    for row in rows:
        scene=row['scene']
        if not manifest['scenes'].get(scene,{}).get('published'):
            manifest['scenes'][scene]=row;status(root,manifest,'publishing '+scene)
            print('PUBLISH_LATEST',scene,flush=True)
            row.update(publish_and_rebind(root,row),published=True)
            manifest['scenes'][scene]=row;status(root,manifest,'published '+scene)
    if assets_only:
        status(root,manifest,'latest assets published; geometry queued');return
    for scene in SCENES:
        row=manifest['scenes'][scene]
        if row.get('geometry_complete'):continue
        status(root,manifest,'raycasting '+scene);print('RAYCAST_LATEST',scene,flush=True)
        worker(root,scene,'prepare')
        status(root,manifest,'shared geometry '+scene)
        worker(root,scene,'support');row['geometry_complete']=True
        status(root,manifest,'geometry complete '+scene)
    status(root,manifest,'geometry complete; RGB renderer selection pending')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE/'mesh_comparison_whole')
    parser.add_argument('--assets-only',action='store_true')
    args=parser.parse_args();run(args.output,args.assets_only)
