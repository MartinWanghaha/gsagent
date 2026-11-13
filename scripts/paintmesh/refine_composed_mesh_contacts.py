#!/usr/bin/env python3
"""Correct generated-mesh registration, then independently audit and publish.

Original meshes/Gaussians stay fixed. Completed scenes are published only after
their new world poses have been tested against every environment partition.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,read,write,identity,digest

PUBLIC=BASE/'mesh_comparison_whole'
DEFAULT=PUBLIC/'collision_refinement'


def refine_scene(output,scene,config,algorithm='envelope'):
    from mesh_benchmark.contact_pose_refine import refine_contact_pose
    if algorithm=='envelope':
        from mesh_benchmark.contact_pose_refine import refine_contact_pose_envelope
        refine=refine_contact_pose_envelope
    else:refine=refine_contact_pose
    output=Path(output);folder=output/'scenes'/scene;folder.mkdir(parents=True,exist_ok=True)
    graph_path=PUBLIC/'recomposition'/scene/'composition/scene.json';graph=read(graph_path)
    original_audit=PUBLIC/'collision_audit/scenes'/scene
    with np.load(original_audit/'background_roi.npz') as a:ev,ef=a['vertices'],a['faces']
    objects=[]
    for node in graph['objects']:
        target=folder/'objects'/str(node['id']);target.mkdir(parents=True,exist_ok=True)
        request=dict(graph=identity(graph_path),mesh=identity(node['mesh']['path']),
            environment=identity(original_audit/'background_roi.npz'),
            algorithm=algorithm,config=config,implementation=identity(HERE/'mesh_benchmark/contact_pose_refine.py'))
        marker=target/'report.json'
        if marker.exists():
            old=read(marker)
            if old.get('complete') and old.get('request')==request:
                objects.append(old);continue
            raise ValueError('Existing contact optimization request changed')
        with np.load(node['mesh']['path']) as a:v,f=a['vertices'],a['faces']
        print('CONTACT_REFINE_BEGIN',scene,node['id'],len(v),len(f),len(ef),flush=True)
        T,report=refine(v,f,node['object_to_world'],ev,ef,evidence=node['evidence'],config=config)
        report.update(scene=scene,id=str(node['id']),request=request,
            object_to_world=T.tolist(),original_object_to_world=node['object_to_world'])
        write(marker,report);objects.append(report)
        print('CONTACT_REFINE_COMPLETE',scene,node['id'],report.get('before'),report.get('after'),
            'translation',report.get('translation_fraction'),'observation',report.get('observation_after',{}).get('score'),flush=True)
    # A staging graph changes only object registration. Full-scene audit reads
    # current unchanged environment geometry and these candidate poses.
    candidate=copy.deepcopy(graph)
    for node,report in zip(candidate['objects'],objects):
        node['object_to_world']=report['object_to_world']
        node['original_pose']=report['object_to_world']
    stage=output/'candidate_world/recomposition'/scene/'composition/scene.json'
    write(stage,candidate)
    write(folder/'optimization_complete.json',dict(complete=True,scene=scene,
        original_graph=identity(graph_path),candidate_graph=identity(stage),
        object_reports=[identity(folder/'objects'/str(n['id'])/'report.json') for n in graph['objects']]))
    return scene


def verify_scene(output,scene):
    import audit_composed_mesh_collisions as detector
    output=Path(output)
    detector.PUBLIC=output/'candidate_world'
    result=detector.audit(output/'collision_after',scene)
    old=read(PUBLIC/'collision_audit/scenes'/scene/'audit.json')
    rows={r['id']:r for r in result['objects']};before={r['id']:r for r in old['objects']}
    folder=output/'scenes'/scene
    objects={};updates={}
    for node in read(output/'candidate_world/recomposition'/scene/'composition/scene.json')['objects']:
        oid=str(node['id']);report=read(folder/'objects'/oid/'report.json')
        final=rows[oid];initial=before[oid];nc=final['intersection_counts']['transverse'];bc=initial['intersection_counts']['transverse']
        sheet=report.get('local_sheet_clear',False)
        old_obs=report.get('observation_before');new_obs=report.get('observation_after')
        cfg=report['config'];guard=(old_obs is None or(new_obs['score']<=old_obs['score']+cfg.get('max_observation_score_increase',.03)+1e-8
            and new_obs['median_iou']>=old_obs['median_iou']-cfg.get('max_median_iou_drop',.03)-1e-8))
        reasons=[]
        if nc:reasons.append('Transverse triangle intersections remain in the complete current environment')
        if not guard:reasons.append('Multi-view observation guard did not pass')
        if not sheet:reasons.append('Local support/contact constraints are unresolved')
        if final['penetration'].get('significant_inside_vertices',0):reasons.append('Environment vertices remain inside closed object beyond contact tolerance')
        accepted=not reasons
        objects[oid]=dict(id=oid,accepted=accepted,unresolved=not accepted,reasons=reasons,
            crossings_before=bc,crossings_after=nc,
            crossing_reduction_fraction=float(1-nc/bc) if bc else 0.,
            observation_guard_passed=guard,contact_sheet_clear=sheet,
            full_mesh_collision=identity(output/'collision_after/scenes'/scene/'objects'/oid/'audit.json'),
            local_penetration_before=report.get('before'),local_penetration_after=report.get('after'),
            original_model_scale_preserved=report.get('scale_preserved'),
            background_geometry_unchanged=True,original_gaussians_unchanged=True)
        updates[oid]=dict(object_to_world=report['object_to_world'],report_path=str(folder/'objects'/oid/'report.json'))
    validation=dict(complete=True,scene=scene,objects=objects,object_object_pairs=result['object_object_pairs'],
        all_environment_partitions_checked=True,mesh_simplification=False,original_gaussians_unchanged=True)
    for row in result['object_object_pairs']:
        if row.get('intersection_counts',{}).get('transverse',0):
            for oid in [row['a'],row['b']]:
                objects[oid]['accepted']=False;objects[oid]['unresolved']=True
                objects[oid]['reasons'].append('Object-object collision remains')
    payload=dict(object_updates=updates,validation=validation)
    write(folder/'publication_updates.json',payload)
    return scene


def run(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    config=dict(iterations=12,max_translation_fraction=.06,max_rotation_radians=.10,
        max_observation_score_increase=.05,max_median_iou_drop=.05,verify_intersections=False)
    state=dict(complete=False,stage='optimizing',scenes=args.scenes,configuration=config,
        started=time.time(),pid=os.getpid(),optimized=[],verified=[],published=[],failures=[])
    write(root/'status.json',state)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(refine_scene,root,s,config,args.algorithm):s for s in args.scenes}
        for f in as_completed(jobs):
            scene=jobs[f]
            try:state['optimized'].append(f.result())
            except Exception as e:state['failures'].append(dict(scene=scene,stage='optimization',error=repr(e)))
            state['updated']=time.time();write(root/'status.json',state)
    if state['failures']:raise RuntimeError(state['failures'])
    state['stage']='full_mesh_collision_recheck';write(root/'status.json',state)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(verify_scene,root,s):s for s in args.scenes}
        for f in as_completed(jobs):
            scene=jobs[f]
            try:state['verified'].append(f.result())
            except Exception as e:state['failures'].append(dict(scene=scene,stage='verification',error=repr(e)))
            state['updated']=time.time();write(root/'status.json',state)
    if state['failures']:raise RuntimeError(state['failures'])
    if args.publish:
        state['stage']='publishing_and_evaluating';write(root/'status.json',state)
        for scene in args.scenes:
            subprocess.run([sys.executable,'-u',str(HERE/'publish_collision_refinement.py'),
                '--scene',scene,'--updates',str(root/'scenes'/scene/'publication_updates.json'),
                '--output',str(PUBLIC)],check=True)
            state['published'].append(scene);state['updated']=time.time();write(root/'status.json',state)
    state['stage']='complete';state['complete']=True;state['updated']=time.time();write(root/'status.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--scenes',nargs='+',choices=SCENES,default=SCENES)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--algorithm',choices=['envelope','sheet'],default='envelope')
    p.add_argument('--publish',action='store_true')
    args=p.parse_args()
    if not 1<=args.workers<=3:p.error('workers must be between 1 and 3')
    run(args)


if __name__=='__main__':main()
