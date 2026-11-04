#!/usr/bin/env python3
"""CPU-only redbull grid-resolution diagnosis using saved accepted RGB anchors.

This does not publish initialization: parent proposal labels are placeholders.
Only grid discretization changes; all confidence/geometry gates stay fixed.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from plyfile import PlyData

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts/paintmesh'),str(REPO/'submodules/EDGS')]
from edgs_inpaint_io import verify_targets,read_json,write_json,record
from source.paintmesh_joint_data import make_views,read_targets
from source import paintmesh_surface_init as surface


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=('fit','fit-fix'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    root=REPO/'output/paintmesh/Inpaint360GS'/args.experiment/'scenes/redbull'
    spec=next(s for s in read_json(root.parents[1]/'request.json')['scenes'] if s['scene']=='redbull')
    archived=root.parents[1]/'redbull_grid_repair_20261103'
    initial=archived/'failed_edgs_init_grid32' if (archived/'failed_edgs_init_grid32').exists() else root/'inpaint/edgs_init'
    config=archived/'config.json' if initial.parent==archived else root/'config.json'
    cfg=read_json(config);old=read_json(initial/'surface_model.json')
    evidence=initial/'surface_evidence.npz'
    with np.load(evidence) as data:anchors={k[7:]:data[k].copy() for k in data.files if k.startswith('anchor_')}
    lama,inputs,_=verify_targets(root/'inpaint/manifests/lama_completion_manifest.json',spec['camera']['path'],use_depth=True)
    views=make_views(spec['camera']['path'],device='cpu');targets=read_targets(lama,inputs,views,use_depth=True)
    for v,t in zip(views,targets):
        with Image.open(inputs['frames'][v.image_name]['inputs']['mask']['path']) as im:t['original_mask']=np.array(im.convert('L'))>0
    removed=read_json(Path(spec['removal_root'])/'removed_3dgs/model_manifest.json')
    vertices=PlyData.read(removed['inputs']['removed_gaussian_ply']['path'],mmap=True)['vertex'].data
    background=np.column_stack([vertices[k] for k in 'xyz'])
    surface.rgb_anchors=lambda *a,**kw:(anchors,old['tracks'])
    support=dict(spacing=np.array(old['spacing']),xyz=anchors['xyz'],source_kind=np.zeros(len(anchors['xyz']),np.int8))
    write_json(args.output/'request.json',dict(source_evidence=record(evidence),original_config=record(config),
        tested_grid_pixels=[32,16,8],no_gt_used=True,scope='CPU initialization evidence only; not published point-cloud initialization'))
    summaries=[]
    for step in (32,16,8):
        settings=copy.deepcopy(cfg['surface_init']);settings['grid_pixels']=step;captured={};started=time.time()
        def capture(model,arrays):captured.update(model=model,arrays=arrays)
        try:
            result,report,model=surface.correct_surface_support(support,background,views,targets,settings,
                seed=cfg['seed'],max_points=None,init_config=cfg['init'],artifact_callback=capture)
            np.savez_compressed(args.output/f'grid_{step}_candidate_geometry.npz',**{k:result[k] for k in ('xyz','frame','pixel_uv','depth_votes','depth_checks','original_mask')})
            error=None
        except ValueError as e:error=str(e)
        model=captured.get('model',{})
        write_json(args.output/f'grid_{step}_surface_model.json',model)
        summary={k:v for k,v in model.items() if k not in ('fields','camera_models','candidate_statistics')}
        summary.update(grid_pixels=step,error=error,elapsed_seconds=time.time()-started,
            anchor_source='saved_accepted_RGB_anchors_from_failed_run; original pair matches not rerun in CPU probe',
            field_observations=sum(f['observations'] for f in model.get('fields',[])),
            field_inliers=sum(f['inliers'] for f in model.get('fields',[])),
            supported_pixels=sum(f['supported_pixels'] for f in model.get('fields',[])))
        summaries.append(summary);write_json(args.output/'summary.json',summaries);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
