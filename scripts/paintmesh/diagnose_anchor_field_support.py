#!/usr/bin/env python3
"""CPU diagnosis of anchor support; no training or initialization publication."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'submodules/EDGS'))
from edgs_inpaint_io import read_json, write_json, verify_targets
from source.paintmesh_joint_data import make_views, read_targets
from source.paintmesh_edgs_init import camera
from source.paintmesh_surface_init import rgb_anchors, depth_field


def run(root,pilot):
    info=read_json(pilot/'summary.json');cfg=info['thresholds']
    lama,inputs,_=verify_targets(info['inputs']['lama']['path'],info['inputs']['camera']['path'],use_depth=True)
    views=make_views(info['inputs']['camera']['path'],device='cpu');cams=[camera(v) for v in views]
    pairs=[]
    for path in sorted(pilot.glob('pair_*.npz')):
        a,b=map(int,path.stem.split('_')[1:])
        if b-a>4:continue
        with np.load(path) as p:pairs.append((a,b,p['uv'],p['other'],p['score']))
    anchors,counts=rgb_anchors({'spacing':info['spacing']},cams,cfg['surface_init'],cfg['init'],pairs)
    selected=sorted(set(anchors['frames'].ravel().tolist()))
    targets=dict(zip(selected,read_targets(lama,inputs,[views[i] for i in selected],use_depth=True)))
    rows=[]
    for step in (32,8):
        settings=dict(cfg['surface_init'],grid_pixels=step)
        for frame in selected:
            depth,domain,field,stats=depth_field(cams[frame],targets[frame],anchors,frame,info['spacing'],settings)
            with Image.open(inputs['frames'][views[frame].image_name]['inputs']['mask']['path']) as im:original=np.asarray(im)>0
            row=dict(grid_pixels=step,**stats,supported_original_hole_pixels=int((domain&original).sum()),original_hole_pixels=int(original.sum()))
            rows.append(row);print(row,flush=True)
    write_json(pilot/'field_support.json',dict(counts=counts,frames=rows,
        note='Only tests whether original depth-anchor gates support any pixels. No true surface accuracy claimed, no thresholds lowered.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    a=p.parse_args();run(a.root,a.pilot)
