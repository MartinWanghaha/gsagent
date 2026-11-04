#!/usr/bin/env python3
"""Render original and removed models on fixed evaluation cameras, for diagnosis only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from run_geometry_ablation import read, write, now

REPO = Path(__file__).resolve().parents[2]
PY = '/home/member/.conda/envs/paintmesh/bin/python'


def regions(root):
    values = []
    for path in sorted((root/'renders').glob('*.png')):
        rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)/255
        gt = np.asarray(Image.open(root/'gt'/path.name).convert('RGB'), dtype=np.float64)/255
        mask = np.asarray(Image.open(root/'masks'/path.name).convert('L')) > 127
        error = ((rgb-gt)**2).mean(2)
        values.append(dict(frame=path.name, outside_psnr=float(-10*np.log10(max(error[~mask].mean(),1e-16))),
                           inside_psnr=float(-10*np.log10(max(error[mask].mean(),1e-16)))))
    if not values:
        raise ValueError('No evaluation renders: '+str(root))
    return dict(frame_count=len(values), outside_psnr=float(np.mean([v['outside_psnr'] for v in values])),
                inside_psnr=float(np.mean([v['inside_psnr'] for v in values])), per_view=values)


def run(job):
    request=read(job/'request.json')
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(request['gpu']), CUDA_DEVICE_ORDER='PCI_BUS_ID',
             PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH',None)
    rows=[]
    for scene in request['scenes']:
        name=scene['scene'];original=Path(scene['root']);removal=original.parents[1]
        scene_root=REPO/'output/paintmesh/Inpaint360GS/inpaint360'/name
        row=dict(scene=name, measurements={})
        for stage,model,iteration in [('original',scene_root/'edgs',30000),('removed',removal/'removed_3dgs',2000)]:
            output=job/'background_diagnosis'/name/stage
            log=job/f'diagnosis_{name}_{stage}.log'
            cmd=[PY,str(REPO/'scripts/paintmesh/eval_inpaint360.py'),'run','--model-path',str(model),
                 '--source-path',str(Path('/home/member/wyc/data/Inpaint360GS/inpaint360')/name),
                 '--resolution','4','--iteration',str(iteration),'--output-root',str(output),'--render-only']
            write(job/'diagnosis_status.json',dict(state='running',scene=name,stage=stage,time=now(),log=str(log)))
            with log.open('a') as handle:
                subprocess.run(cmd,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
            row['measurements'][stage]=regions(output)
        row['measurements']['inpaint_5000']=regions(original/'inpainted_3dgs_5000/evaluation/inpaint360_r4')
        rows.append(row);write(job/'background_diagnosis.json',dict(complete=False,scenes=rows))
        print(name, {k:round(v['outside_psnr'],4) for k,v in row['measurements'].items()},flush=True)
    write(job/'background_diagnosis.json',dict(complete=True,scenes=rows))
    write(job/'diagnosis_status.json',dict(state='complete',time=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args()
    try:run(a.job)
    except Exception as exc:
        write(a.job/'diagnosis_status.json',dict(state='failed',error=str(exc),time=now()))
        raise
