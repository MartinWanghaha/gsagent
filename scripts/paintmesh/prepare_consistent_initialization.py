#!/usr/bin/env python3
"""Opt-in subset of existing mixed seeds; all old rows and retained fields unchanged."""
import argparse
import copy
from pathlib import Path
import shutil
import numpy as np
from plyfile import PlyData,PlyElement
from edgs_inpaint_io import (INIT_KIND,validate_init,record,seal,verify_record,write_json,
    read_json,claim,atomic_write,implementation,initialization_config,load_config)


def subset(vertex,editable,support,keep):
    if editable.dtype!=np.bool_ or editable.shape!=(len(vertex),):
        raise ValueError('Invalid old/new ownership')
    old=int((~editable).sum());new=int(editable.sum())
    if editable[:old].any() or not editable[old:].all():
        raise ValueError('Background must be an unchanged prefix')
    if keep.dtype!=np.bool_ or keep.shape!=(new,) or not keep.any():
        raise ValueError('No supported seeds or invalid filter')
    xyz=np.column_stack([vertex[k][editable] for k in 'xyz'])
    if not np.array_equal(xyz,support['xyz']):
        raise ValueError('Support and initialized Gaussian rows differ')
    rows=np.r_[np.arange(old),old+np.flatnonzero(keep)]
    filtered=vertex[rows].copy()
    result={}
    for key,value in support.items():
        if value.ndim==0:result[key]=value.copy()
        elif len(value)==new:result[key]=value[keep].copy()
        else:raise ValueError('Unexpected support row shape: '+key)
    ownership=np.r_[np.zeros(old,dtype=bool),np.ones(int(keep.sum()),dtype=bool)]
    return filtered,ownership,result


def run(baseline,evidence_summary,scene,root,config):
    joint=read_json(baseline/'manifests/edgs_joint_manifest.json')
    initial=validate_init(joint['inputs']['initialization']['path'])
    diagnosis=next(x for x in read_json(evidence_summary) if x['scene']==scene)
    if diagnosis['source']!=joint['inputs']['initialization']:
        raise ValueError('Evidence comes from another initialization')
    cfg=load_config(config)
    if initialization_config(cfg)!=initial['request']['config']:
        raise ValueError('Subset experiment must retain initialization settings')
    if cfg['surface_init']['mode']!='none':raise ValueError('Requires original mixed seeds')
    evidence=verify_record(diagnosis['evidence'])
    thresholds=diagnosis['thresholds']
    if thresholds!=dict(min_angle_deg=5,tolerance_world_spacing=2,min_other_views=2,min_fraction=.6):
        raise ValueError('Unexpected evidence policy')
    with np.load(evidence,allow_pickle=False) as f:
        arrays={k:f[k].copy() for k in f.files}
    keep=(arrays['votes']>=2)&(arrays['votes']>=.6*arrays['checked'])
    if not np.array_equal(keep,arrays['keep']):raise ValueError('Filter differs from declared evidence')
    records=dict(initial['inputs'],proposal_initialization=joint['inputs']['initialization'])
    request=dict(pipeline='edgs-pgsr',inputs=records,implementation=implementation(),
        config=initialization_config(cfg),matcher_weights=initial['request']['matcher_weights'],
        output_root=str((root/'edgs_init').resolve()),manifest=str((root/'manifests/edgs_init_manifest.json').resolve()),
        variant='mixed_seed_consistency_subset',
        filtering=dict(thresholds=thresholds,views=diagnosis['views'],source=diagnosis['source'],
            evidence=diagnosis['evidence'],producer=record(Path(__file__).with_name('diagnose_seed_consistency.py')),
            transform=record(Path(__file__)),uses_test_gt=False,
            meaning='agreement with completed depth, not calibrated true surface confidence'),
        removal_variant='published_target_removed',selection=initial['request']['selection'])
    dest=root/'edgs_init';manifest=root/'manifests/edgs_init_manifest.json'
    if manifest.exists():validate_init(manifest,request);return
    claim(dest,request)
    vertex=PlyData.read(initial['outputs']['ply']['path'],mmap='r')['vertex'].data
    editable=np.load(initial['outputs']['editable_mask']['path'])
    with np.load(initial['outputs']['support']['path']) as f:support={k:f[k].copy() for k in f.files}
    filtered,ownership,support=subset(vertex,editable,support,keep)
    old=int((~editable).sum())
    for key in vertex.dtype.names:
        assert vertex[key][:old].tobytes()==filtered[key][:old].tobytes(),key
    PlyData([PlyElement.describe(filtered,'vertex')],text=False).write(dest/'point_cloud.ply')
    atomic_write(dest/'editable_mask.npy',lambda f:np.save(f,ownership,allow_pickle=False))
    atomic_write(dest/'support.npz',lambda f:np.savez_compressed(f,**support))
    shutil.copyfile(evidence,dest/'filter_evidence.npz')
    report=dict(initial['report'],initialized_points=int(keep.sum()),
        rgb_points=int((support['source_kind']==0).sum()),depth_points=int((support['source_kind']==1).sum()),
        filtering=diagnosis,all_old_fields_unchanged=True,retained_seed_fields_unchanged=True)
    write_json(dest/'diagnostics.json',report);write_json(dest/'config.resolved.json',request['config'])
    outputs={key:record(dest/name) for key,name in (
        ('ply','point_cloud.ply'),('editable_mask','editable_mask.npy'),('support','support.npz'),
        ('diagnostics','diagnostics.json'),('config','config.resolved.json'),('filter_evidence','filter_evidence.npz'))}
    receipt=seal(INIT_KIND,request=request,inputs=records,outputs=outputs,point_count=len(filtered),
        parameters=request['config'],report=report)
    validate_init(receipt)
    write_json(manifest,receipt)
    print(f'Filtered initialization: old={old} (unchanged), new={editable.sum()} -> {keep.sum()}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('baseline','evidence-summary','output-root','config'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scene',required=True);a=p.parse_args()
    run(a.baseline,a.evidence_summary,a.scene,a.output_root,a.config)
