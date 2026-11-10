#!/usr/bin/env python3
"""Check real merged outputs against every original Gaussian rendering attribute."""
from pathlib import Path
import argparse
import numpy as np
from recompose.io import read_json,write_json,ply
from recompose.evidence import SELECTION,resolve_scene
from recompose.original_gaussians import standard_fields


def compare(source,merged,start,indices=None):
    count=len(source) if indices is None else len(indices)
    nr=sum(n.startswith('f_rest_') for n in source.dtype.names);per_channel=nr//3
    for lo in range(0,count,65536):
        hi=min(lo+65536,count)
        original=source[lo:hi] if indices is None else source[indices[lo:hi]]
        out=merged[start+lo:start+hi]
        for name in standard_fields(source):
            dst=name
            if name.startswith('f_rest_'):
                j=int(name[7:]);channel,band=divmod(j,per_channel);dst=f'f_rest_{channel*15+band}'
            if not np.array_equal(original[name],out[dst]):raise ValueError(f'Original attribute changed: {name} at merged range {start+lo}')
    return count


def main():
    repo=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(repo/'output/paintmesh/Inpaint360GS/recomposition'));args=p.parse_args()
    root=Path(args.root);results=[]
    for scene in SELECTION:
        folder=root/scene;inputs=resolve_scene(repo/'output/paintmesh/Inpaint360GS',scene)
        try:
            info=read_json(folder/'original_gaussians/manifest.json')
            merged=ply(folder/'composition/world/composed_gaussians.ply')['vertex'].data
            bg=ply(inputs['background_gaussians'])['vertex'].data;original=ply(info['sources']['original']['path'])['vertex'].data
            start=compare(bg,merged,0);counts={}
            for target in inputs['target_ids']:
                indices=np.load(folder/'original_gaussians'/str(target)/'original_indices.npy')
                count=compare(original,merged,start,indices);start+=count;counts[str(target)]=count
            if start!=len(merged):raise ValueError('Merged model has extra or missing splats')
            item=dict(scene=scene,passed=True,training=False,background_count=len(bg),original_object_counts=counts,total=len(merged),
                      attributes='XYZ, SH, opacity, scales and quaternion are exactly preserved')
        except Exception as e:item=dict(scene=scene,passed=False,error=str(e))
        results.append(item);print(item,flush=True)
    write_json(root/'original_gaussian_audit.json',dict(passed=all(v['passed'] for v in results),scenes=results))
    return 0 if all(v['passed'] for v in results) else 2


if __name__=='__main__':raise SystemExit(main())
