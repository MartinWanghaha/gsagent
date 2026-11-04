#!/usr/bin/env python3
"""Compare new Gaussian shape and verify frozen rows for completed scale ablations."""
import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData
from run_geometry_ablation import read,write,now


def shape(ply,new):
    v=PlyData.read(ply,mmap='r')['vertex'].data
    scales=np.exp(np.column_stack([v['scale_'+str(k)] for k in range(3)]).astype(np.float64))
    ratio=scales.max(1)/scales.min(1)
    return {name:dict(count=int(mask.sum()),aspect_p10_p50_p90=np.quantile(ratio[mask],[.1,.5,.9]).tolist(),
        aspect_under_two_fraction=float((ratio[mask]<2).mean()),
        min_axis_p10_p50_p90=np.quantile(scales[mask].min(1),[.1,.5,.9]).tolist())
        for name,mask in [('new',new),('old',~new)]}


def run(job):
    req=read(job/'request.json');results=[]
    for case in read(job/'experiments.json'):
        if case['name']!='scale_flatten100':continue
        source=next(s for s in req['scenes'] if s['scene']==case['scene'])
        old=Path(source['root']);root=Path(case['output'])
        base=read(old/'manifests/edgs_joint_manifest.json');initial=read(base['inputs']['initialization']['path'])
        own=np.load(initial['outputs']['editable_mask']['path'])
        rows=np.load(old/'edgs_joint/snapshots/iteration_5000/retained_rows.npy')
        before=read(old/'inpainted_3dgs_5000/evaluation/inpaint360_r4/results.json')
        row=dict(scene=case['scene'],baseline=shape(before['inputs']['model']['path'],own[rows]))
        if (root/'manifests/edgs_joint_manifest.json').exists():
            joint=read(root/'manifests/edgs_joint_manifest.json')
            assert joint['inputs']['initialization']==base['inputs']['initialization']
            rows=np.load(joint['outputs']['retained_rows']['path'])
            row['experiment']=shape(joint['outputs']['ply']['path'],own[rows])
            a=read(base['inputs']['local_selection']['path']);b=read(joint['inputs']['local_selection']['path'])
            ta=np.load(a['outputs']['trainable_mask']['path']);tb=np.load(b['outputs']['trainable_mask']['path'])
            assert np.array_equal(ta,tb),'Old-row eligibility changed'
            start=PlyData.read(initial['outputs']['ply']['path'],mmap='r')['vertex'].data
            finish=PlyData.read(joint['outputs']['ply']['path'],mmap='r')['vertex'].data
            original=np.flatnonzero(~tb);lookup=np.full(len(tb),-1,dtype=np.int64);lookup[rows]=np.arange(len(rows))
            assert (lookup[original]>=0).all()
            assert start.dtype==finish.dtype
            for k in start.dtype.names:
                assert start[k][original].tobytes()==finish[k][lookup[original]].tobytes(),k
            row['frozen_verification']=dict(count=len(original),all_retained=True,bitwise_identical=True,old_eligibility_identical=True)
        results.append(row)
        print(row,flush=True)
    write(job/'surface_shape/shape_verification.json',dict(updated_at=now(),scenes=results))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True,type=Path);a=p.parse_args();run(a.job)
