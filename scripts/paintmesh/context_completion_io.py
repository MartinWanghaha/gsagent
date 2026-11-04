#!/usr/bin/env python3
"""Prepare/screen fixed context targets and bind unchanged Gaussian initialization."""
import argparse
import copy
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from run_geometry_ablation import REPO,read,write,now
from edgs_inpaint_io import (record,verify_record,verify_targets,identity,validate_init,
    validate_selection,seal,INIT_KIND,SELECTION_KIND)
from shared_reference_completion import load_scene,consistency

VARIANTS={'A_context':0,'B_context_opt10':10}
PARAMETERS=dict(lr=.002,latent_weight=.01)
PREDICTOR=Path(__file__).with_name('lama_context_predict.py')


def rgb(path):return np.asarray(Image.open(path).convert('RGB'))
def mask(path):return np.asarray(Image.open(path).convert('L'))>0


def case(name,image,mask_path,reference,output,steps):
    return dict(name=name,image=str(image),mask=str(mask_path),reference=str(reference) if reference else None,
                output=str(output),steps=steps)


def prepare_scene(root):
    row=read(root/'source.json');prior=read(root/'prior_guide.json')
    lama,inputs,_=verify_targets(row['inputs']['lama']['path'],row['inputs']['camera']['path'],use_depth=True,use_normal=True)
    if prior['source_lama']!=row['inputs']['lama']:raise ValueError('Reference chosen from another baseline')
    ref=Path(lama['frames'][prior['reference']]['outputs']['color']['path'])
    guided=[s for s,v in prior['frames'].items() if v['mode'] in ('warp_and_lama_residual','fully_supported_warp')]
    preview=[guided[i] for i in np.linspace(0,len(guided)-1,6).astype(int)]
    screen_cases=[]
    for variant,steps in VARIANTS.items():
        dest=root/variant;(dest/'lama/output/color').mkdir(parents=True,exist_ok=True)
        write(dest/'source.json',row);write(dest/'config.json',row['variants'][0]['config'])
        cases=[]
        for stem,frame in lama['frames'].items():
            out=dest/'lama/output/color'/f'{stem}.png'
            source=inputs['frames'][stem]['outputs']
            if stem in guided:
                cases.append(case(stem,source['color']['path'],source['color_mask']['path'],ref,out,steps))
            else:
                if out.exists() and record(out)['sha256']!=frame['outputs']['color']['sha256']:raise ValueError('Unchanged frame differs')
                if not out.exists():shutil.copyfile(frame['outputs']['color']['path'],out)
        write(dest/'prediction_request.json',dict(model_path=str(Path(lama['model']['config']['path']).parent),
            parameters=PARAMETERS,cases=cases,receipt=str(dest/'prediction_receipt.json')))
        for stem in preview:
            source=inputs['frames'][stem]['outputs']
            screen_cases.append(case(f'virtual/{variant}/{stem}',source['color']['path'],source['color_mask']['path'],
                ref,root/'screen/virtual'/variant/f'{stem}.png',steps))
    # Known-background pseudo holes in real TRAIN photographs, never test images.
    source_root=Path(row['source']);photos=sorted(p for p in (source_root/'images').iterdir()
        if p.suffix.lower() in ('.png','.jpg','.jpeg') and not p.stem.lower().startswith('test'))
    ids=row['publish']['parameters']['target_ids'];pseudo=[]
    for index,photo_index in enumerate(np.linspace(0,len(photos)-1,6).astype(int)):
        photo=photos[photo_index];im=Image.open(photo).convert('RGB')
        w,h=round(im.width/4),round(im.height/4);im=im.resize((w,h),Image.Resampling.LANCZOS)
        segmentation=source_root/'associated_hqsam'/f'{photo.stem}.png'
        labels=np.asarray(Image.open(segmentation).resize((w,h),Image.Resampling.NEAREST))
        exclude=binary_dilation(np.isin(labels,ids),iterations=12)
        ys,xs=np.where(exclude);cx=float(xs.mean()) if len(xs) else w/2;cy=float(ys.mean()) if len(ys) else h/2
        bw,bh=round(w*.16),round(h*.18);candidates=[]
        for y in np.linspace(12,h-bh-12,10).astype(int):
            for x in np.linspace(12,w-bw-12,12).astype(int):
                if not exclude[y:y+bh,x:x+bw].any():candidates.append(((x+bw/2-cx)**2+(y+bh/2-cy)**2,int(x),int(y)))
        if not candidates:raise ValueError('No pseudo-hole background region: '+photo.name)
        _,x,y=min(candidates);m=np.zeros((h,w),np.uint8);m[y:y+bh,x:x+bw]=255
        d=root/'screen/inputs';d.mkdir(parents=True,exist_ok=True)
        ip=d/f'{photo.stem}.png';mp=d/f'{photo.stem}_mask.png';im.save(ip);Image.fromarray(m).save(mp)
        pseudo.append(dict(name=photo.stem,image=record(ip),mask=record(mp),original=record(photo),segmentation=record(segmentation),
            excluded_ids=ids,bbox=[x,y,x+bw,y+bh],split='diagnostic' if index<2 else 'validation'))
        for group,steps,reference in [('independent',0,None),('A_context',0,ref),('B_context_opt10',10,ref)]:
            screen_cases.append(case(f'pseudo/{group}/{photo.stem}',ip,mp,reference,root/'screen/pseudo'/group/f'{photo.stem}.png',steps))
    request=dict(model_path=str(Path(lama['model']['config']['path']).parent),parameters=PARAMETERS,
        cases=screen_cases,receipt=str(root/'screen/prediction_receipt.json'))
    write(root/'screen/request.json',request)
    write(root/'screen/input_manifest.json',dict(uses_test_data=False,reference=record(ref),reference_frame=prior['reference'],
        preview_frames=preview,pseudo_cases=pseudo,producer=record(Path(__file__)),parameters=PARAMETERS,
        screen_gate=dict(mean_validation_roi_psnr_drop_max=.5,worst_validation_roi_psnr_drop_max=3.,
            require_visual_review=True),interpretation='Known training-background self-test; not proof for truly unseen holes'))


def verify_predictions(request_path):
    req=read(request_path);receipt=read(req['receipt'])
    if receipt['request_sha256']!=record(request_path)['sha256'] or receipt['code_sha256']!=record(PREDICTOR)['sha256']:
        raise ValueError('Prediction request/code changed')
    records={x['path']:x for x in receipt['outputs']}
    for c in req['cases']:
        p=Path(c['output']);meta=read(p.with_suffix('.json'));item=records[str(p)]
        if (record(p)['sha256']!=item['sha256'] or record(p.with_suffix('.json'))['sha256']!=item['metadata_sha256']
                or meta['signature']['case']!=c or meta['signature']['parameters']!=PARAMETERS):raise ValueError('Prediction identity mismatch')
        for k,h in meta['signature']['inputs'].items():
            if record(c[k])['sha256']!=h:raise ValueError('Prediction input changed')
        trace=meta['trace']
        if trace['optimizer_steps']!=c['steps'] or not trace['outside_mask_byte_exact'] or not trace['network_weights_frozen']:
            raise ValueError('Inactive optimizer or changed known pixels/network')
        if c['steps'] and (len(trace['gradient_norms'])!=c['steps'] or min(trace['gradient_norms'])<=0):raise ValueError('No actual gradient update')
        original=rgb(c['image']);m=mask(c['mask']);out=rgb(p)
        if out.shape!=original.shape or not np.array_equal(out[~m],original[~m]):raise ValueError('Known RGB/dimensions changed')
    return req


def psnr(a,b,m):return float(-10*np.log10(max(float(np.mean((a[m].astype(float)/255-b[m].astype(float)/255)**2)),1e-12)))


def screen(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    req=verify_predictions(root/'screen/request.json');inp=read(root/'screen/input_manifest.json')
    results=[];groups=['independent',*VARIANTS]
    fig,axs=plt.subplots(len(inp['pseudo_cases']),4,figsize=(15,15))
    for i,c in enumerate(inp['pseudo_cases']):
        gt=rgb(verify_record(c['image']));m=mask(verify_record(c['mask']));x0,y0,x1,y1=c['bbox']
        x0=max(0,x0-20);x1=min(gt.shape[1],x1+20);y0=max(0,y0-20);y1=min(gt.shape[0],y1+20)
        scores={};images=[gt]
        for group in groups:
            out=rgb(root/'screen/pseudo'/group/(c['name']+'.png'));scores[group]=psnr(out,gt,m);images.append(out)
        results.append(dict(name=c['name'],split=c['split'],roi_psnr=scores))
        for j,im in enumerate(images):
            axs[i,j].imshow(im[y0:y1,x0:x1]);axs[i,j].axis('off')
            axs[i,j].set_title(f"{c['name']} ({c['split']})\n"+('Known train RGB' if j==0 else f'{groups[j-1]}: {scores[groups[j-1]]:.2f} dB'))
    fig.tight_layout();fig.savefig(root/'screen/pseudo_comparison.png',dpi=140);plt.close(fig)
    row=read(root/'source.json');lama=read(row['inputs']['lama']['path']);lin=read(lama['input_manifest']['path'])
    fig,axs=plt.subplots(len(inp['preview_frames']),3,figsize=(13,16),squeeze=False)
    for i,stem in enumerate(inp['preview_frames']):
        m=mask(lin['frames'][stem]['outputs']['color_mask']['path']);ys,xs=np.where(m)
        y0,y1=max(0,ys.min()-20),min(m.shape[0],ys.max()+21);x0,x1=max(0,xs.min()-20),min(m.shape[1],xs.max()+21)
        paths=[lama['frames'][stem]['outputs']['color']['path']]+[root/'screen/virtual'/v/f'{stem}.png' for v in VARIANTS]
        for j,(group,path) in enumerate(zip(groups,paths)):
            axs[i,j].imshow(rgb(path)[y0:y1,x0:x1]);axs[i,j].axis('off');axs[i,j].set_title(f'{stem}: {group}')
    fig.tight_layout();fig.savefig(root/'screen/virtual_comparison.png',dpi=140);plt.close(fig)
    gates={}
    for group in VARIANTS:
        differences=[r['roi_psnr'][group]-r['roi_psnr']['independent'] for r in results if r['split']=='validation']
        gates[group]=dict(mean_validation_delta=float(np.mean(differences)),worst_validation_delta=float(min(differences)),
            numeric_pass=float(np.mean(differences))>=-.5 and min(differences)>=-3)
        # A and B must share an identical zero-update starting prediction.
    for c in req['cases']:
        if '/B_context_opt10/' in c['name']:
            meta=read(Path(c['output']).with_suffix('.json'))
            other=Path(str(c['output']).replace('/B_context_opt10/','/A_context/')).with_suffix('.json')
            a=read(other)
            if meta['trace']['initial_output_sha256']!=a['trace']['final_output_sha256']:
                raise ValueError('A/B initial predictions differ; latent update is not the only change')
    write(root/'screen/results.json',dict(complete=True,uses_test_data=False,parameters=PARAMETERS,pseudo_cases=results,gates=gates,
        optimizer_updates_verified=True,A_B_initial_predictions_identical=True,
        interpretation='Numerical screen excludes major observed-background failures, still needs visual review'))
    print('Screen:',root.name,gates,flush=True)


def bind(root,variant):
    dest=root/variant;row=read(root/'source.json');review=read(root/'screen/review.json');screen_result=read(root/'screen/results.json')
    if variant not in review['approved_variants'] or not screen_result['gates'][variant]['numeric_pass']:
        raise ValueError('Variant did not pass pre-training screen')
    req=verify_predictions(dest/'prediction_request.json')
    lama,inputs,cams,stems,data=load_scene(row);completed=[]
    for i,stem in enumerate(stems):
        out=rgb(dest/'lama/output/color'/f'{stem}.png')
        if out.shape!=data[i]['rgb'].shape or not np.array_equal(out[~data[i]['mask']],data[i]['original'][~data[i]['mask']]):
            raise ValueError('Completion changed known pixels: '+stem)
        completed.append(out)
    validation=dict(complete=True,variant=variant,reference=read(root/'prior_guide.json')['reference'],
        outside_mask_byte_exact=True,full_original_mask=True,depth_and_normal_unchanged=True,
        optimizer_steps_per_guided_frame=VARIANTS[variant],guided_frames=len(req['cases']),
        consistency=consistency(cams,stems,data,completed))
    write(dest/'completion_validation.json',validation)
    derived=copy.deepcopy(lama);derived['created_at']=now()
    derived['roots']['color_output']=str(dest/'lama/output/color')
    derived['parameters']['shared_reference_context']=dict(mode=variant,reference=validation['reference'],
        parameters=PARAMETERS,steps=VARIANTS[variant],baseline=row['inputs']['lama'],
        input_screen=record(root/'screen/input_manifest.json'),screen_results=record(root/'screen/results.json'),
        visual_review=record(root/'screen/review.json'),prediction_request=record(dest/'prediction_request.json'),
        prediction_receipt=record(dest/'prediction_receipt.json'),validation=record(dest/'completion_validation.json'),
        producer=record(Path(__file__)),predictor=record(PREDICTOR),uses_test_data=False)
    for stem in stems:
        p=dest/'lama/output/color'/f'{stem}.png';derived['frames'][stem]['outputs']['color']=dict(record(p),mtime_ns=p.stat().st_mtime_ns)
    derived['artifact_id']=identity(dict(kind=derived['kind'],schema_version=derived['schema_version'],
        input_artifact_id=derived['input_artifact_id'],model={k:v['sha256'] for k,v in derived['model'].items()},
        parameters=derived['parameters'],outputs={s:{k:v['sha256'] for k,v in f['outputs'].items()} for s,f in derived['frames'].items()}))
    lp=dest/'manifests/lama_completion_manifest.json';write(lp,derived)
    verify_targets(lp,row['inputs']['camera']['path'],use_depth=True,use_normal=True)
    old=validate_init(row['inputs']['initialization']['path'])
    ii=dict(old['inputs'],lama=record(lp),proposal_initialization=row['inputs']['initialization'])
    ir=copy.deepcopy(old['request']);ir.update(inputs=ii,variant='fixed_initialization_rgb_context_ablation',
        output_root=str(dest/'edgs_init'),manifest=str(dest/'manifests/edgs_init_manifest.json'),
        reuse_policy='Reuse every original Gaussian field/support row; initialized with parent independent LaMa')
    initial=seal(INIT_KIND,request=ir,inputs=ii,outputs=old['outputs'],point_count=old['point_count'],
        parameters=old['parameters'],report=old['report']);validate_init(initial)
    ip=dest/'manifests/edgs_init_manifest.json';write(ip,initial)
    new=copy.deepcopy(row);new['inputs'].update(lama=record(lp),initialization=record(ip))
    if 'local_selection' in row['inputs']:
        original=validate_selection(row['inputs']['local_selection']['path'],initial=old)
        si=dict(original['inputs'],lama=record(lp),initialization=record(ip),baseline_selection=row['inputs']['local_selection'])
        sr=copy.deepcopy(original['request']);sr.update(inputs=si,output_root=str(dest/'local_selection'),policy='Identical old/new row eligibility; only RGB targets changed')
        selection=seal(SELECTION_KIND,request=sr,inputs=si,outputs=original['outputs'],initialization_id=initial['artifact_id'])
        validate_selection(selection,initial=initial,config=row['variants'][0]['config']['local_background'])
        sp=dest/'manifests/local_selection_manifest.json';write(sp,selection);new['inputs']['local_selection']=record(sp)
    new['publish']['inputs']['lama_manifest']=record(lp);write(dest/'training_source.json',new)
    print('Targets ready:',root.name,variant,validation,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','screen','bind'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=tuple(VARIANTS))
    a=p.parse_args();r=a.root.resolve()
    if a.action=='prepare':prepare_scene(r)
    elif a.action=='screen':screen(r)
    else:bind(r,a.variant)
