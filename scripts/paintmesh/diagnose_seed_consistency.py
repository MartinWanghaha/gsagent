#!/usr/bin/env python3
"""Training-only multiview depth support of existing mixed initialization seeds."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'submodules/EDGS'))
from edgs_inpaint_io import read_json,write_json,verify_targets,record
from source.paintmesh_local_data import make_views
from source.paintmesh_edgs_init import camera,project,sample


def run(job):
    output=job/'seed_consistency_pilot';output.mkdir(exist_ok=True)
    rows=[]
    for source in read_json(job/'request.json')['scenes']:
        if source['scene'] not in ('fruits','garden_toys'):continue
        root=Path(source['root']);joint=read_json(root/'manifests/edgs_joint_manifest.json')
        initial=read_json(joint['inputs']['initialization']['path'])
        with np.load(initial['outputs']['support']['path']) as f:s={k:f[k].copy() for k in f.files}
        lama,inputs,_=verify_targets(joint['inputs']['lama']['path'],joint['inputs']['camera']['path'],use_depth=True)
        views=make_views(joint['inputs']['camera']['path'],device='cpu');cams=[camera(v) for v in views]
        xyz=s['xyz'];gap=float(s['spacing']);centers=np.array([c['c2w'][:3,3] for c in cams])
        selected=np.linspace(0,len(views)-1,24).astype(int)
        votes=np.zeros(len(xyz),np.int32);checked=votes.copy();known_front=votes.copy()
        source_ray=xyz-centers[s['frame']];source_ray/=np.maximum(np.linalg.norm(source_ray,axis=1,keepdims=True),1e-10)
        fractions=[]
        for i in selected:
            uv,z,valid=project(xyz,cams[i]);safe=np.where(valid[:,None],uv,0)
            depth=np.load(lama['frames'][views[i].image_name]['outputs']['depth']['path'])
            with Image.open(inputs['frames'][views[i].image_name]['outputs']['color_mask']['path']) as im:mask=np.asarray(im)>0
            d=sample(depth,safe);inside=sample(mask.astype(float),safe)>.999
            ray=xyz-centers[i];ray/=np.maximum(np.linalg.norm(ray,axis=1,keepdims=True),1e-10)
            angle=np.degrees(np.arccos(np.clip((ray*source_ray).sum(1),-1,1)))
            valid &= (d>0)&(angle>=5)
            tested=valid&inside
            tolerance=2*gap*np.linalg.norm(cams[i]['w2c'][2,:3])
            agree=tested&(np.abs(z-d)<=tolerance)
            checked+=tested;votes+=agree
            known_front+=valid&~inside&(z<d-tolerance)
            fractions.append(dict(frame=int(i),tested=int(tested.sum()),agree=int(agree.sum())))
        keep=(votes>=2)&(votes>=.6*checked)
        row=dict(scene=source['scene'],source=joint['inputs']['initialization'],views=selected.tolist(),
            thresholds=dict(min_angle_deg=5,tolerance_world_spacing=2,min_other_views=2,min_fraction=.6),
            original_count=len(xyz),retained_count=int(keep.sum()),
            source_kinds={str(k):dict(original=int((s['source_kind']==k).sum()),kept=int((keep&(s['source_kind']==k)).sum())) for k in np.unique(s['source_kind'])},
            dropped_to_retained_spacing=None,per_view=fractions,
            note='Consistency against existing completed depth only, not true geometry confidence; known-front counts diagnostic only.')
        if keep.any() and (~keep).any():
            distance=cKDTree(xyz[keep]).query(xyz[~keep],workers=4)[0]/gap
            row['dropped_to_retained_spacing']=np.quantile(distance,[.5,.9,.99]).tolist()
        path=output/(source['scene']+'.npz');np.savez_compressed(path,keep=keep,votes=votes,checked=checked,known_front=known_front)
        row['evidence']=record(path);rows.append(row);print(row,flush=True)
    write_json(output/'summary.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args();run(a.job)
