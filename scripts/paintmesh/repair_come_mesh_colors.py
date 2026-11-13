#!/usr/bin/env python3
"""Undo native CoMe RGB premultiplication using exact native white-field weights."""
import argparse
import fcntl
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,read,write,identity,digest
from mesh_benchmark.color_export import read_mesh,positions,attach_colors
from recompose.io import fingerprint

PUBLIC=BASE/'mesh_comparison_whole'
WORK=PUBLIC/'_internal/workspace/come_color_repair'
POLICY=dict(version=1,source='Native CoMe evaluate_alpha/return_color=True',
    correction='Divide native premultiplied RGB by same-view native constant-white response',
    view_selection='Unchanged official minimum-alpha view; first winner for exact ties',
    unknown='Retain original RGB if no selected camera or color weight is zero',
    reference_data_used=False,geometry_unchanged=True,training_performed=False)


def unpremultiply(rgb,weight,original):
    rgb=np.asarray(rgb);weight=np.asarray(weight)
    if rgb.shape!=original.shape or weight.shape!=(len(rgb),):raise ValueError('Invalid color/weight shape')
    if not np.isfinite(rgb).all() or not np.isfinite(weight).all():raise ValueError('Nonfinite color/weight')
    if np.any(weight<0) or np.any(weight>1.0001):raise ValueError('Native color weight outside [0,1]')
    valid=weight>0
    result=np.array(original,copy=True)
    result[valid]=np.rint(np.clip(rgb[valid]/weight[valid,None],0,1)*255).astype('uint8')
    return result


def native_query(xyz,cameras,gaussian,pipe,settings,official):
    import torch
    points=torch.from_numpy(np.asarray(xyz,dtype='float32')).cuda()
    background=torch.zeros(3,device='cuda');n=len(points)
    minimum=torch.ones(n,device='cuda');selected=torch.full((n,),-1,device='cuda',dtype=torch.int32)
    saved_integrate=official.integrate;lookup={id(c):i for i,c in enumerate(cameras)}
    def recording_integrate(points,alpha,view,*args,**kwargs):
        result=saved_integrate(points,alpha,view,*args,**kwargs)
        wins=alpha<minimum
        selected[wins]=lookup[id(view)];minimum.copy_(torch.minimum(minimum,alpha))
        return result
    official.integrate=recording_integrate
    try:
        with torch.no_grad():
            field_alpha,rgb=official.evaluate_alpha(points,cameras,gaussian,pipe,background,.003,splat_args=settings)
    finally:official.integrate=saved_integrate
    # One extra query per chosen point (not one per point/view): only the
    # actual winning camera is reevaluated under a constant white field.
    weights=torch.zeros(n,device='cuda');white=torch.ones_like(gaussian.get_xyz);maximum_alpha_error=0.
    for i,camera in enumerate(cameras):
        ids=torch.nonzero(selected==i,as_tuple=True)[0]
        if not len(ids):continue
        alpha=torch.ones(len(ids),device='cuda')
        with torch.no_grad():
            result=saved_integrate(points[ids],alpha,camera,gaussian,pipe,background,kernel_size=.003,
                                   splat_args=settings,override_color=white)
        response=result['color_integrated']
        if not torch.isfinite(response).all():raise ValueError('Nonfinite white-field response')
        if float((response-response[:,0:1]).abs().max())>1e-6:raise ValueError('White channels differ')
        weights[ids]=response[:,0]
        maximum_alpha_error=max(maximum_alpha_error,float((alpha-minimum[ids]).abs().max()))
    if maximum_alpha_error>1e-5:raise ValueError('White override changed opacity/point grouping')
    return rgb.cpu().numpy(),weights.cpu().numpy(),dict(maximum_white_field_alpha_difference=maximum_alpha_error,
        no_selected_camera=int((selected<0).sum()),field_alpha_min=float(field_alpha.min()),field_alpha_max=float(field_alpha.max()))


def normalize_native(scene,source,chunk,cameras,gaussian,pipe,settings,official,model,camera_checks):
    folder=WORK/scene;folder.mkdir(parents=True,exist_ok=True)
    lock=open(folder/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request=dict(policy=POLICY,geometry=identity(source),checkpoint=identity(model/'point_cloud/iteration_30000/point_cloud.ply'),
        config=identity(model/'config.json'),split=identity(BASE/'_data'/scene/'split.json'),
        upstream_source=fingerprint(Path(official.__file__)),kernel_size=.003)
    signature=digest(request);complete=folder/'complete.json'
    if complete.exists():
        old=read(complete)
        if old['signature']==signature and old['candidate_mesh']==fingerprint(old['path']):
            print('COME_NORMALIZATION_ALREADY_COMPLETE',scene,flush=True);return old
    mesh=read_mesh(source);vertices=mesh['vertex'].data
    # Validate exact native RGB reproduction and chunk independence before
    # accepting any normalization output. Selection never uses GT or scores.
    sample=np.sort(np.random.default_rng(42).choice(len(vertices),min(2048,len(vertices)),replace=False))
    rgb,weights,proof=native_query(positions(vertices[sample]),cameras,gaussian,pipe,settings,official)
    stored=np.column_stack([vertices[k][sample] for k in ['red','green','blue']])
    replay=np.rint(np.clip(rgb,0,1)*255).astype('uint8')
    replay_error=float(np.abs(replay.astype('float64')-stored).mean())
    if replay_error>.05:raise ValueError(f'Native RGB does not reproduce original mesh: {replay_error}')
    if scene=='cube':
        parts=[native_query(p,cameras,gaussian,pipe,settings,official) for p in np.array_split(positions(vertices[sample]),2)]
        error=float(np.max(np.abs(rgb-np.concatenate([p[0] for p in parts]))))
        weight_error=float(np.max(np.abs(weights-np.concatenate([p[1] for p in parts]))))
        if max(error,weight_error)>1e-5:raise ValueError('Query chunking changed native colors or weights')
        proof.update(chunk_rgb_max_difference=error,chunk_weight_max_difference=weight_error)
    proof.update(native_replay_mean_byte_error=replay_error,white_weight_quantiles=np.quantile(weights,[0,.1,.5,.9,1]).tolist(),
                 vertices=len(sample),camera_checks=camera_checks)
    write(folder/'native_probe.json',proof)
    progress=folder/'progress.json';old=read(progress) if progress.exists() else None
    if old and old['signature']!=signature:raise ValueError('Normalization source changed')
    start=old['completed_vertices'] if old else 0
    colors=np.lib.format.open_memmap(folder/'vertex_colors.npy',mode='r+' if old else 'w+',dtype='uint8',shape=(len(vertices),3))
    weight_cache=np.lib.format.open_memmap(folder/'color_weights.npy',mode='r+' if old else 'w+',dtype='float32',shape=(len(vertices),))
    records=old['chunks'] if old else [];began=time.time()
    while start<len(vertices):
        stop=min(start+chunk,len(vertices));tick=time.time()
        try:rgb,weights,meta=native_query(positions(vertices[start:stop]),cameras,gaussian,pipe,settings,official)
        except Exception as error:
            import torch
            if not isinstance(error,torch.cuda.OutOfMemoryError) or chunk<=65536:raise
            chunk//=2;torch.cuda.empty_cache();print('COME_COLOR_CHUNK_RETRY',scene,chunk,flush=True);continue
        previous=np.column_stack([vertices[k][start:stop] for k in ['red','green','blue']])
        replay=np.rint(np.clip(rgb,0,1)*255).astype('uint8')
        reproduction=float(np.abs(replay.astype('float32')-previous).mean())
        if reproduction>.05:raise ValueError('Native original color mismatch: '+str(reproduction))
        fixed=unpremultiply(rgb,weights,previous);colors[start:stop]=fixed;weight_cache[start:stop]=weights
        colors.flush();weight_cache.flush()
        records.append(dict(start=start,stop=stop,seconds=time.time()-tick,
            zero_weight=int((weights<=0).sum()),changed_vertices=int(np.any(fixed!=previous,axis=1).sum()),
            native_replay_mean_byte_error=reproduction,white_weight_quantiles=np.quantile(weights,[0,.1,.5,.9,1]).tolist(),**meta))
        start=stop;write(progress,dict(signature=signature,request=request,completed_vertices=start,total_vertices=len(vertices),
            chunks=records,updated=time.time(),pid=__import__('os').getpid()))
        print('COME_COLOR_NORMALIZED',scene,start,'/',len(vertices),'seconds',records[-1]['seconds'],flush=True)
    destination=source.with_name(source.stem+'_color_normalized.ply')
    verified=attach_colors(source,destination,colors)
    result=dict(complete=True,scene=scene,method='come',signature=signature,request=request,path=str(destination),
        candidate_mesh=fingerprint(destination),source_geometry=fingerprint(source),geometry_unchanged=True,
        geometry_unchanged_by_colorization=True,training_performed=False,reference_data_used=False,
        changed_vertices=sum(c['changed_vertices'] for c in records),zero_weight_vertices=sum(c['zero_weight'] for c in records),
        native_probe=proof,elapsed=time.time()-began,**verified)
    write(complete,result);print('COME_COLOR_NORMALIZATION_COMPLETE',scene,flush=True)
    return result


def publish(scene):
    from colorize_baseline_meshes import publish as publish_color
    receipt=read(WORK/scene/'complete.json')
    if receipt['candidate_mesh']!=fingerprint(receipt['path']):raise ValueError('Candidate mesh changed')
    source=Path(receipt['source_geometry']['path'])
    detail=dict(color_method=POLICY['correction'],color_normalization_policy=POLICY,
        native_probe=receipt['native_probe'],reference_data_used=False,
        white_field_weights=str(WORK/scene/'color_weights.npy'),zero_weight_vertices=receipt['zero_weight_vertices'],
        vertices=receipt['vertices'],faces=receipt['faces'],coordinates_exact=True,
        all_original_geometry_and_properties_exact=True,color_normalization_receipt=str(WORK/scene/'complete.json'))
    result=publish_color('come',scene,source,Path(receipt['path']),detail)
    out=BASE/'come'/scene;marker=read(out/'mesh_complete.json')
    marker.update(color_normalization=True,color_normalization_receipt=str(WORK/scene/'complete.json'))
    write(out/'mesh_complete.json',marker)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--publish',action='store_true');a=p.parse_args()
    if a.publish:publish(a.scene)
    else:raise SystemExit('Use colorize_baseline_meshes.py --worker --normalize-colors to load native CoMe')
