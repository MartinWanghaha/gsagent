#!/usr/bin/env python3
"""Isolated LaMa fixed-reference context inference with real single-scale latent updates.

Existing LaMa predictors/refiners and all 3DGS training code remain unchanged.
Only the two encoder feature tensors are optimized, never network parameters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
LAMA_ROOT = REPO / "submodules/Inpaint360GS/LaMa"
sys.path.insert(0, str(LAMA_ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def png(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    Image.fromarray(value).save(temp,format='PNG');temp.replace(path)


def read_rgb(path):return np.asarray(Image.open(path).convert('RGB'),dtype=np.uint8)


def canvas(original, mask, reference=None, device='cpu'):
    """No masked current RGB reaches the encoder; padding is excluded from losses."""
    if original.shape[:2]!=mask.shape or original.ndim!=3 or original.shape[2]!=3:
        raise ValueError('RGB/mask dimensions differ')
    if not mask.any() or mask.all():raise ValueError('Need a nonempty hole and known context')
    h,w=mask.shape;ph=(-h)%8;pw=(-w)%8;wp=w+pw
    rgb=torch.from_numpy(original.copy()).permute(2,0,1)[None].float().to(device)/255
    m=torch.from_numpy(mask.copy())[None,None].float().to(device)
    # Reflect the mask together with RGB: otherwise reflected hidden RGB can
    # leak through pixels incorrectly labelled known in the padding.
    rgb=F.pad(rgb,(0,pw,0,ph),mode='reflect');m=F.pad(m,(0,pw,0,ph),mode='reflect')
    valid=torch.zeros_like(m);valid[:,:,:h,:w]=1
    known=(1-m)*valid;ref_valid=torch.zeros_like(m)
    left=0
    if reference is not None:
        if reference.shape!=original.shape:raise ValueError('Reference must have the target dimensions')
        ref=torch.from_numpy(reference.copy()).permute(2,0,1)[None].float().to(device)/255
        ref=F.pad(ref,(0,pw,0,ph),mode='reflect')
        rgb=torch.cat([ref,rgb],dim=-1)
        m=torch.cat([torch.zeros_like(m),m],dim=-1)
        ref_valid=torch.cat([valid,torch.zeros_like(valid)],dim=-1)
        known=torch.cat([torch.zeros_like(known),known],dim=-1);left=wp
    encoded=torch.cat([rgb*(1-m),m],dim=1)
    return dict(image=rgb,mask=m,encoded=encoded,known=known,reference=ref_valid,
                height=h,width=w,left=left)


def region_l1(pred, target, region):
    return ((pred-target).abs()*region).sum()/(3*region.sum()).clamp_min(1)


def optimize_features(front, rear, batch, steps, lr=.002, latent_weight=.01):
    if steps not in (0,10):raise ValueError('Pilot compares exactly 0 and 10 optimizer steps')
    with torch.no_grad():start=front(batch['encoded'])
    if not isinstance(start,(tuple,list)) or len(start)!=2:raise ValueError('Expected LaMa local/global feature pair')
    origin=tuple(v.detach().clone() for v in start)
    features=tuple(v.detach().clone().requires_grad_(steps>0) for v in start)
    optimizer=torch.optim.Adam(features,lr=lr) if steps else None
    history=[];grad_norms=[]
    def losses(pred):
        known=region_l1(pred,batch['image'],batch['known'])
        ref=region_l1(pred,batch['image'],batch['reference'])
        drift=sum((v-v0).square().mean()/v0.square().mean().clamp_min(1e-4) for v,v0 in zip(features,origin))
        return known+ref+latent_weight*drift,known,ref,drift
    with torch.no_grad():initial_pred=rear(features);_,k,r,_=losses(initial_pred)
    before=dict(known_l1=float(k),reference_l1=float(r))
    for iteration in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred=rear(features);loss,k,r,drift=losses(pred)
        if not torch.isfinite(loss):raise ValueError('Nonfinite latent objective')
        loss.backward()
        if any(v.grad is None or not torch.isfinite(v.grad).all() for v in features):
            raise ValueError('Missing/nonfinite latent gradient')
        norm=float(sum(v.grad.square().sum() for v in features).sqrt())
        if norm<=0:raise ValueError('Zero latent gradient; optimization is not active')
        grad_norms.append(norm)
        history.append(dict(step=iteration+1,total=float(loss),known_l1=float(k),reference_l1=float(r),latent_drift=float(drift)))
        optimizer.step()
    with torch.no_grad():
        # This forward occurs after the tenth update, not before the final update.
        pred=rear(features);_,k,r,drift=losses(pred)
    if not torch.isfinite(pred).all():raise ValueError('Nonfinite RGB prediction')
    trace=dict(requested_steps=steps,optimizer_steps=len(grad_norms),gradient_norms=grad_norms,
        before=before,after=dict(known_l1=float(k),reference_l1=float(r)),
        normalized_latent_drift=float(drift),history=history,
        network_weights_frozen=True,single_scale=True)
    return pred.detach(),initial_pred.detach(),trace


class Engine:
    def __init__(self, model_path):
        from omegaconf import OmegaConf
        from saicinpainting.training.trainers import load_checkpoint
        from saicinpainting.training.modules.ffc import FFCResnetBlock
        from saicinpainting.training.modules.pix2pixhd import ResnetBlock
        cfg=OmegaConf.load(str(model_path/'config.yaml'))
        cfg.training_model.predict_only=True;cfg.visualizer.kind='noop'
        self.model=load_checkpoint(cfg,str(model_path/'models/best.ckpt'),strict=False,map_location='cpu')
        self.model.freeze();self.model.eval()
        if self.model.training or self.model.add_noise_kwargs or not self.model.concat_mask:
            raise ValueError('Unexpected LaMa model mode')
        for p in self.model.parameters():p.requires_grad_(False)
        modules=self.model.generator.model
        first=next(i for i,m in enumerate(modules) if isinstance(m,(FFCResnetBlock,ResnetBlock)))
        self.device=torch.device('cuda:0')
        self.front=modules[:first].to(self.device).eval();self.rear=modules[first:].to(self.device).eval()
        self.versions=[p._version for p in self.model.parameters()]

    def predict(self, case, parameters):
        original=read_rgb(case['image']);mask=np.asarray(Image.open(case['mask']).convert('L'))>0
        reference=read_rgb(case['reference']) if case.get('reference') else None
        if reference is not None and reference.shape!=original.shape:
            reference=np.asarray(Image.fromarray(reference).resize((original.shape[1],original.shape[0]),Image.Resampling.BILINEAR))
        batch=canvas(original,mask,reference,self.device)
        if batch['image'].shape[-1]*batch['image'].shape[-2]>1800000:
            raise ValueError('Context image exceeds pilot pixel budget; no silent resizing')
        pred,initial,trace=optimize_features(self.front,self.rear,batch,case['steps'],**parameters)
        if self.versions!=[p._version for p in self.model.parameters()]:raise ValueError('Network parameter changed')
        if any(p.grad is not None or p.requires_grad for p in self.model.parameters()):raise ValueError('Network gradients enabled')
        results=[]
        for tensor in (pred,initial):
            cropped=tensor[0,:, :batch['height'], batch['left']:batch['left']+batch['width']]
            value=np.clip(cropped.permute(1,2,0).cpu().numpy()*255,0,255).astype(np.uint8)
            final=original.copy();final[mask]=value[mask];results.append(final)
        trace.update(full_original_mask=True,outside_mask_byte_exact=bool(np.array_equal(results[0][~mask],original[~mask])),
            initial_output_sha256=hashlib.sha256(results[1].tobytes()).hexdigest(),
            final_output_sha256=hashlib.sha256(results[0].tobytes()).hexdigest(),
            changed_hole_pixels=int(np.any(results[0]!=results[1],axis=-1)[mask].sum()),hole_pixels=int(mask.sum()),
            input_shape=list(original.shape),context_width=int(batch['image'].shape[-1]))
        return results[0],trace


def run(request_path):
    request=json.loads(Path(request_path).read_text())
    torch.manual_seed(0);np.random.seed(0);torch.backends.cudnn.benchmark=False
    model_path=Path(request['model_path']);engine=Engine(model_path)
    code_sha=sha(Path(__file__));model_sha=sha(model_path/'models/best.ckpt')
    for i,case in enumerate(request['cases']):
        out=Path(case['output']);meta=out.with_suffix('.json')
        signature=dict(case=case,parameters=request['parameters'],code_sha256=code_sha,checkpoint_sha256=model_sha,
            inputs={k:sha(case[k]) for k in ('image','mask','reference') if case.get(k)})
        if meta.exists():
            old=json.loads(meta.read_text())
            if old['signature']!=signature or not out.exists() or old['output_sha256']!=sha(out):raise ValueError('Existing prediction differs: '+str(out))
            continue
        value,trace=engine.predict(case,request['parameters'])
        png(out,value);write(meta,dict(complete=True,signature=signature,trace=trace,output_sha256=sha(out)))
        print(f"Context {i+1}/{len(request['cases'])}: {case['name']} steps={trace['optimizer_steps']} known={trace['before']['known_l1']:.6f}->{trace['after']['known_l1']:.6f} ref={trace['before']['reference_l1']:.6f}->{trace['after']['reference_l1']:.6f}",flush=True)
    write(request['receipt'],dict(complete=True,request_sha256=sha(request_path),code_sha256=code_sha,
        outputs=[dict(path=c['output'],sha256=sha(c['output']),metadata_sha256=sha(Path(c['output']).with_suffix('.json'))) for c in request['cases']]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True)
    run(p.parse_args().request)
