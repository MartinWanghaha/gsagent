#!/usr/bin/env python3
"""Paired all-view audit and diagnostic figures for completed reference pilots."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_geometry_ablation import read, write, now
from run_muti_plan import validate_evaluation
from edgs_inpaint_io import record


def rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def stats(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),
        q10=float(np.quantile(a,.1)),q90=float(np.quantile(a,.9)),improved=int((a>1e-6).sum()),
        worse=int((a< -1e-6).sum()),unchanged=int((np.abs(a)<=1e-6).sum()))


def analyze(root):
    request=read(root/'request.json');out=root/'analysis';out.mkdir(exist_ok=True)
    all_results={};records=[];samples=[];fig,axs=plt.subplots(2,2,figsize=(13,8))
    for si,scene in enumerate(('cube','truck')):
        spec=next(s for s in request['scenes'] if s['scene']==scene)
        current=(root/'cube_reference_selected/scenes/cube/A_context' if scene=='cube'
            else root/'truck_soft_reference/scenes/truck')
        models={'Independent':Path(spec['baseline']),'Hard reference':Path(spec['hard_warp'])}
        if scene=='truck':models['Context only']=root/'scenes/truck/A_context'
        models['Current']=current
        evaluations={};step_stats={}
        for step in (5000,10000):
            evaluations[step]={}
            for name,folder in models.items():
                p=folder/f'inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json'
                r=read(p);assert r['complete'];evaluations[step][name]=(p.parent,r);records.append(record(p))
            base_path,base=evaluations[step]['Independent'];names=list(base['per_view']);assert len(names)==base['frame_count']
            for _,r in evaluations[step].values():validate_evaluation(r,base)
            per_method={}
            for name,(path,result) in evaluations[step].items():
                per=[]
                for frame in names:
                    gt=rgb(path/'gt'/f'{frame}.png');m=np.asarray(Image.open(path/'masks'/f'{frame}.png').convert('L'))>127
                    pred=rgb(path/'renders'/f'{frame}.png');bpred=rgb(base_path/'renders'/f'{frame}.png')
                    if not np.array_equal(gt,rgb(base_path/'gt'/f'{frame}.png')):raise ValueError('GT differs')
                    if not np.array_equal(m,np.asarray(Image.open(base_path/'masks'/f'{frame}.png').convert('L'))>127):raise ValueError('Mask differs')
                    err=((pred.astype(float)-gt.astype(float))/255)**2
                    berr=((bpred.astype(float)-gt.astype(float))/255)**2
                    v=result['per_view'][frame];b=base['per_view'][frame]
                    f=dict(scene=scene,step=step,method=name,frame=frame,PSNR=v['PSNR'],masked_PSNR=v['masked_PSNR'],
                        delta_PSNR=v['PSNR']-b['PSNR'],delta_masked_PSNR=v['masked_PSNR']-b['masked_PSNR'],
                        mask_fraction=float(m.mean()),roi_psnr=float(-10*np.log10(max(err[m].mean(),1e-12))),
                        hole_mse=float(err[m].mean()),outside_mse=float(err[~m].mean()),
                        baseline_hole_mse=float(berr[m].mean()),baseline_outside_mse=float(berr[~m].mean()))
                    per.append(f);samples.append(f)
                per_method[name]=dict(metrics=result['metrics'],delta={k:result['metrics'][k]-base['metrics'][k] for k in result['metrics']},
                    delta_PSNR=stats([x['delta_PSNR'] for x in per]),delta_masked_PSNR=stats([x['delta_masked_PSNR'] for x in per]),
                    both_psnr_improved=sum(x['delta_PSNR']>0 and x['delta_masked_PSNR']>0 for x in per),
                    mean_hole_mse=float(np.mean([x['hole_mse'] for x in per])),mean_outside_mse=float(np.mean([x['outside_mse'] for x in per])),
                    mean_true_roi_psnr=float(np.mean([x['roi_psnr'] for x in per])),mean_mask_fraction=float(np.mean([x['mask_fraction'] for x in per])),
                    per_view=per)
                if name!='Independent':
                    ax=axs[si,0 if step==5000 else 1]
                    ax.plot(np.arange(len(names)),[x['delta_masked_PSNR'] for x in per],label=name,linewidth=1.3)
            step_stats[str(step)]=per_method
            ax=axs[si,0 if step==5000 else 1];ax.axhline(0,color='black',linestyle='--',linewidth=.8)
            ax.set_title(f'{scene}: {step} steps, all {len(names)} test views');ax.set_xlabel('Test view index');ax.set_ylabel('Delta masked PSNR (dB)');ax.legend(fontsize=8)
        current_views=step_stats['5000']['Current']['per_view'];ordered=sorted(current_views,key=lambda x:x['delta_masked_PSNR'])
        selected=[]
        for label,item in [('worst delta',ordered[0]),('median delta',ordered[len(ordered)//2]),('best delta',ordered[-1])]:
            selected.append((item['frame'],label))
        names=list(evaluations[5000]['Independent'][1]['per_view'])
        for i in np.linspace(0,len(names)-1,4).astype(int):
            if names[i] not in [x[0] for x in selected]:selected.append((names[i],'uniform index'))
        columns=[('GT',None,None)]+[(name,5000,p) for name,(p,_) in evaluations[5000].items()]+[('Current',10000,evaluations[10000]['Current'][0])]
        g,aa=plt.subplots(len(selected),len(columns),figsize=(3.2*len(columns),2.8*len(selected)),squeeze=False)
        for y,(frame,label) in enumerate(selected):
            basepath=evaluations[5000]['Independent'][0];mask=np.asarray(Image.open(basepath/'masks'/f'{frame}.png').convert('L'))>127
            yy,xx=np.where(mask);y0=max(0,yy.min()-20);y1=min(mask.shape[0],yy.max()+21);x0=max(0,xx.min()-20);x1=min(mask.shape[1],xx.max()+21)
            for x,(name,step,path) in enumerate(columns):
                image=rgb(basepath/'gt'/f'{frame}.png') if name=='GT' else rgb(path/'renders'/f'{frame}.png')
                aa[y,x].imshow(image[y0:y1,x0:x1],interpolation='nearest');aa[y,x].axis('off')
                text=f'{frame} ({label})\nGT' if name=='GT' else f'{name} {step}\nmPSNR={evaluations[step][name][1]["per_view"][frame]["masked_PSNR"]:.3f}'
                aa[y,x].set_title(text,fontsize=9)
        g.suptitle(f'{scene}: same mask bounding box + 20px; selections include worst / median / best current 5000 delta',fontsize=12)
        g.tight_layout(rect=(0,0,1,.96));g.savefig(out/f'{scene}_test_comparison.png',dpi=125);plt.close(g)
        all_results[scene]=dict(frame_count=len(names),steps=step_stats,figure_selection=selected)
    fig.tight_layout();fig.savefig(out/'per_view_masked_psnr_delta.png',dpi=150);plt.close(fig)
    write(out/'paired_analysis.json',dict(created_at=now(),inputs=records,scenes=all_results,
        note='Paired views share one trained model, so frame statistics do not measure training-seed uncertainty; no significance claim.'))
    with (out/'per_view.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    for scene,results in all_results.items():
        for step,methods in results['steps'].items():
            c=methods['Current'];b=methods['Independent']
            print(scene,step,'PSNR',c['delta_PSNR'],'masked',c['delta_masked_PSNR'],
                'hole MSE change%',100*(c['mean_hole_mse']/b['mean_hole_mse']-1),
                'outside MSE change%',100*(c['mean_outside_mse']/b['mean_outside_mse']-1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    analyze(p.parse_args().root.resolve())
