#!/usr/bin/env python3
"""Difix RGB post-completion with auditable original/derived target contracts."""
from __future__ import annotations
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

from run_geometry_ablation import read, write, now, sha
from edgs_inpaint_io import record, verify_record, verify_targets, identity


def rgb(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB')).copy()


def save_png(path, arr):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    Image.fromarray(arr).save(tmp,format='PNG');tmp.replace(path)


def padded_rgb(a):
    h,w=a.shape[:2]
    return np.pad(a,((0,(-h)%8),(0,(-w)%8),(0,0)),mode='reflect')


def compose(original, fixed, mask, scope):
    if fixed.shape!=original.shape or mask.shape!=original.shape[:2]:raise ValueError('RGB/mask dimensions differ')
    if scope=='full':return fixed.copy()
    if scope!='mask':raise ValueError('Unknown Difix compositing scope')
    result=original.copy();result[mask]=fixed[mask];return result


def verify_prediction(req_path):
    req=read(req_path);receipt=read(req['receipt'])
    if receipt['request_sha256']!=sha(req_path):raise ValueError('Difix prediction request changed')
    if receipt['predictor_sha256']!=sha(Path(__file__)):raise ValueError('Difix predictor changed')
    for rec in req['model_files'].values():verify_record(rec)
    verify_record(req['pipeline'])
    for c in req['cases']:
        for key in ('input','mask'):verify_record(c[key])
        metadata=read(c['metadata']);signature=identity(dict(case=c,parameters=req['parameters'],model_files=req['model_files'],pipeline=req['pipeline']))
        if metadata['signature']!=signature:raise ValueError('Difix frame request mismatch')
        verify_record(metadata['raw']);verify_record(metadata['target'])
    return req,receipt


def infer(req_path):
    import torch
    req=read(req_path);p=req['parameters'];sys.path.insert(0,str(Path(req['pipeline']['path']).parent))
    from pipeline_difix import DifixPipeline
    for r in req['model_files'].values():verify_record(r)
    verify_record(req['pipeline']);code=sha(Path(__file__))
    if Path(req['receipt']).exists():verify_prediction(req_path);print('Verified completed Difix inference',flush=True);return
    torch.set_num_threads(2);torch.manual_seed(p['seed']);np.random.seed(p['seed'])
    torch.backends.cudnn.benchmark=False
    pipe=DifixPipeline.from_pretrained(req['model_root'],local_files_only=True,torch_dtype=torch.float32).to('cuda')
    for component in (pipe.unet,pipe.vae,pipe.text_encoder):component.eval();component.requires_grad_(False)
    if not hasattr(pipe.vae.decoder,'skip_conv_1'):raise ValueError('Difix custom skip VAE not loaded')
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.reset_peak_memory_stats();started=time.time();items=[]
    for i,c in enumerate(req['cases']):
        signature=identity(dict(case=c,parameters=p,model_files=req['model_files'],pipeline=req['pipeline']))
        for key in ('input','mask'):verify_record(c[key])
        meta=Path(c['metadata'])
        if meta.exists():
            m=read(meta)
            if m['signature']!=signature:raise ValueError('Existing Difix frame inputs changed')
            verify_record(m['raw']);verify_record(m['target']);items.append(record(meta));continue
        original=rgb(c['input']['path']);mask=np.asarray(Image.open(c['mask']['path']).convert('L'))>0
        padded=padded_rgb(original);h,w=original.shape[:2]
        generator=torch.Generator(device='cuda').manual_seed(p['seed']+i)
        with torch.inference_mode():
            value=pipe(p['prompt'],image=Image.fromarray(padded),height=padded.shape[0],width=padded.shape[1],
                num_inference_steps=1,timesteps=[p['timestep']],guidance_scale=0.,generator=generator,output_type='np').images[0]
        if not np.isfinite(value).all() or value.shape!=padded.shape:raise ValueError('Nonfinite/wrong-shape Difix output')
        fixed=np.rint(np.clip(value[:h,:w],0,1)*255).astype(np.uint8)
        if original.std()>10 and fixed.std()<1:raise ValueError('Collapsed/blank Difix output')
        target=compose(original,fixed,mask,p['scope'])
        if p['scope']=='mask' and not np.array_equal(target[~mask],original[~mask]):raise ValueError('Known pixels changed')
        save_png(c['raw'],fixed);save_png(c['target'],target)
        delta=np.abs(target.astype(float)-original)
        write(meta,dict(complete=True,signature=signature,frame=c['name'],raw=record(c['raw']),target=record(c['target']),
            shape=list(original.shape),padding=[(-h)%8,(-w)%8],seed=p['seed']+i,
            hole_mae_255=float(delta[mask].mean()) if mask.any() else 0.,
            outside_mae_255=float(delta[~mask].mean()) if (~mask).any() else 0.,
            peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
        items.append(record(meta));print(f'Difix {i+1}/{len(req["cases"])} {c["name"]}: peak={torch.cuda.max_memory_allocated()/1024**2:.0f} MiB',flush=True)
    if code!=sha(Path(__file__)):raise ValueError('Difix code changed during inference')
    write(req['receipt'],dict(complete=True,request_sha256=sha(req_path),predictor_sha256=code,frames=items,
        elapsed_seconds=time.time()-started,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        environment={k:importlib.metadata.version(k) for k in ('torch','diffusers','transformers','peft','huggingface-hub','numpy','pillow')}))


def bind(req_path):
    req,receipt=verify_prediction(req_path);p=req['parameters'];lp=Path(req['completion_manifest'])
    if lp.exists():
        v,_,_=verify_targets(lp,req['camera']['path'],use_depth=True,use_normal=True)
        if v['parameters']['difix']['prediction_request']!=record(req_path):raise ValueError('Derived targets changed')
        return
    verify_record(req['parent_lama']);verify_record(req['camera'])
    lama,inputs,_=verify_targets(req['parent_lama']['path'],req['camera']['path'],use_depth=True,use_normal=True)
    derived=copy.deepcopy(lama);derived['created_at']=now();derived['roots']['color_output']=str(Path(req['cases'][0]['target']).parent)
    derived['parameters']['difix']=dict(model_name='nvidia/difix',parameters=p,model_files=req['model_files'],
        parent_lama=req['parent_lama'],prediction_request=record(req_path),prediction_receipt=record(req['receipt']),
        producer=record(Path(__file__)),pipeline=req['pipeline'],test_rgb_used=False,test_masks_used=False,
        geometry_targets='byte-identical parent LaMa depth/normal; Difix is RGB only')
    derived['parameters']['outside_mask_policy']='preserve_input_exactly' if p['scope']=='mask' else 'difix_full_rgb'
    for c in req['cases']:
        t=Path(c['target']);original=rgb(c['input']['path']);target=rgb(t);mask=np.asarray(Image.open(c['mask']['path']).convert('L'))>0
        if p['scope']=='mask' and not np.array_equal(target[~mask],original[~mask]):raise ValueError('Known pixels changed')
        derived['frames'][c['name']]['outputs']['color']=dict(record(t),mtime_ns=t.stat().st_mtime_ns)
    derived['artifact_id']=identity(dict(kind=derived['kind'],schema_version=derived['schema_version'],
        input_artifact_id=derived['input_artifact_id'],model={k:v['sha256'] for k,v in derived['model'].items()},
        parameters=derived['parameters'],outputs={s:{k:v['sha256'] for k,v in f['outputs'].items()} for s,f in derived['frames'].items()}))
    write(lp,derived);verify_targets(lp,req['camera']['path'],use_depth=True,use_normal=True)
    chosen=np.linspace(0,len(req['cases'])-1,min(6,len(req['cases'])),dtype=int)
    page=Image.new('RGB',(1200,len(chosen)*280),'white');draw=ImageDraw.Draw(page)
    for row,j in enumerate(chosen):
        c=req['cases'][j]
        for col,(title,path) in enumerate([('LaMa',c['input']['path']),('Difix raw',c['raw']),('Training target',c['target'])]):
            im=Image.open(path).convert('RGB');im.thumbnail((400,250));page.paste(im,(col*400,row*280+25));draw.text((col*400+5,row*280+5),c['name']+' '+title,fill='black')
    page.save(lp.parent.parent/'difix/preview.jpg',quality=92)
    print('Derived Difix targets validated:',lp,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('infer','bind'));ap.add_argument('--request',type=Path,required=True)
    args=ap.parse_args();globals()[args.action](args.request.resolve())
