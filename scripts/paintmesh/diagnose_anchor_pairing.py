#!/usr/bin/env python3
"""Training-only pilot: compare compact and historical camera triangles."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'submodules/EDGS'))
from edgs_inpaint_io import verify_targets, read_json, write_json, record
from source.paintmesh_joint_data import make_views, read_targets
from source.paintmesh_edgs_init import camera, RomaMatcher, matcher_weights, sample
from source.paintmesh_surface_init import rgb_anchors


def run(root,output):
    output.mkdir(parents=True,exist_ok=True)
    joint=read_json(root/'manifests/edgs_joint_manifest.json')
    cfg=read_json(root/'edgs_joint/config.resolved.json')
    init=read_json(joint['inputs']['initialization']['path'])
    with np.load(init['outputs']['support']['path']) as f:spacing=float(f['spacing'])
    lama,inputs,_=verify_targets(joint['inputs']['lama']['path'],joint['inputs']['camera']['path'])
    views=make_views(joint['inputs']['camera']['path'],device='cpu')
    cams=[camera(v) for v in views]
    groups={'compact':[(i,i+2,i+4) for i in (0,30,60,90,120,150)],
            'historical':[(8,101,109),(9,102,110),(13,104,112)]}
    frames=sorted({i for triangles in groups.values() for triangle in triangles for i in triangle})
    targets=dict(zip(frames,read_targets(lama,inputs,[views[i] for i in frames])))
    weights=matcher_weights(cfg['matcher'],download=False)
    matcher=RomaMatcher(cfg['matcher'],weights)
    rng=np.random.default_rng(cfg['seed'])
    pairs={};stats={}
    for a,b in sorted({edge for ts in groups.values() for t in ts for edge in combinations(t,2)}):
        path=output/f'pair_{a:03d}_{b:03d}.npz'
        if path.exists():
            with np.load(path) as p:uv,other,score=[p[k] for k in ('uv','other','score')]
            diagnostics=read_json(path.with_suffix('.json'))
        else:
            uv,other,score=matcher(targets[a],targets[b],cfg['init'],rng)
            diagnostics=dict(matcher.last_diagnostics)
            np.savez_compressed(path,uv=uv,other=other,score=score)
            write_json(path.with_suffix('.json'),diagnostics)
        pairs[a,b]=(a,b,uv,other,score);stats[f'{a}/{b}']=diagnostics
        print(a,b,len(score),flush=True)
    results=[]
    for group,triangles in groups.items():
        for triangle in triangles:
            raw=[pairs[e] for e in combinations(triangle,2)]
            anchors,counts=rgb_anchors({'spacing':spacing},cams,cfg['surface_init'],cfg['init'],raw)
            original=np.zeros(len(anchors['xyz']),bool)
            for frame in triangle:
                rows,columns=np.where(anchors['frames']==frame)
                with Image.open(inputs['frames'][views[frame].image_name]['inputs']['mask']['path']) as im:
                    m=np.asarray(im.convert('L'))>0
                original[rows] |= sample(m.astype(float),anchors['uv'][rows,columns],0)>0
            axes=np.array([cams[i]['c2w'][:3,2] for i in triangle]);axes/=np.linalg.norm(axes,axis=1,keepdims=True)
            angles=[float(np.degrees(np.arccos(np.clip(axes[i]@axes[j],-1,1)))) for i,j in combinations(range(3),2)]
            row=dict(group=group,triangle=triangle,pair_matches=[len(p[4]) for p in raw],axis_angles_deg=angles,
                     **counts,anchors_in_original_mask=int(original.sum()))
            results.append(row);print(row,flush=True)
    write_json(output/'summary.json',dict(scene=root.parents[3].name,inputs=joint['inputs'],
        thresholds={'init':cfg['init'],'surface_init':cfg['surface_init']},spacing=spacing,
        seed=cfg['seed'],weights=weights,triangles=results,pair_diagnostics=stats,
        note='Training LaMa views only; no GT; matchability does not establish true surface accuracy.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
