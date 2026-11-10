"""Fit the same color field per fixed mesh, then evaluate RGB on named cameras."""
import gc
from pathlib import Path
import numpy as np
from PIL import Image
from .core import cameras,read,write,psnr,masked_ssim_map

def image_metrics(pred,target,allowed,lpips_model,device,background=False):
    import torch
    a=pred.astype('float32')/255;b=target.astype('float32')/255
    if not allowed.any():return dict(PSNR=None,SSIM=None,LPIPS=None)
    structural=masked_ssim_map(a,b,allowed)
    # Remove target-object appearance from both LPIPS inputs. The same binary mask
    # is shared across methods, then the spatial feature-distance map is averaged
    # only over allowed pixels (no full-canvas masked-PSNR dilution).
    aa=a*allowed[:,:,None];bb=b*allowed[:,:,None]
    t=lambda x:torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0).to(device)*2-1
    with torch.no_grad():
        distance=lpips_model(t(aa),t(bb))[0,0]
        weights=torch.from_numpy(allowed.astype('float32')).to(device)
        lp=float((distance*weights).sum()/weights.sum())
    return dict(PSNR=psnr(a,b,allowed),SSIM=float(structural[allowed].mean()),LPIPS=lp)

def score(output,method,scene,state,config,gpu):
    import torch
    torch.set_num_threads(4)
    device='cuda:0'
    if not torch.cuda.is_available():raise RuntimeError('Color-field worker requires CUDA')
    free,total=torch.cuda.mem_get_info();cap=config['gpu_cap_mib']*1024**2
    if free<cap+2*1024**3:raise RuntimeError('Insufficient GPU headroom before scoring')
    torch.cuda.set_per_process_memory_fraction(cap/total)
    from .appearance import fit,render,ColorField
    root=Path(output);cache=root/'cache'/method/scene/state
    geometry=read(cache/'geometry_metrics.json');metadata=read(cache/'complete.json')
    common=read(root/'common'/scene/'complete.json')
    support_root=root/'common'/scene/(state+'_support');support=read(support_root/'complete.json')
    dest=root/state/method/scene;dest.mkdir(parents=True,exist_ok=True)
    request=dict(config=config,geometry=metadata['request'],state=state,support_id=support['request_id'])
    if (dest/'metrics.json').exists():
        old=read(dest/'metrics.json')
        if old.get('scoring_protocol_version')==2 and old['request']==request:return
    write(dest/'invocation.json',dict(request=request,physical_gpu=gpu,torch=torch.__version__,
                                    geometry_frozen=True,gaussian_renderer_used=False))
    cs=[c for c in cameras(scene) if c['part']=='held_out']
    all_rendered=all((dest/'renders'/(c['stem']+'.png')).exists() for c in cs)
    checkpoint=dest/'color_field.pt';history=[];model=None
    if not all_rendered:
        if checkpoint.exists():
            obj=torch.load(checkpoint,map_location=device)
            keys=['seed','appearance_steps','grid_resolution','rank','batch_rays','pool_pixels']
            if any(obj['config'][k]!=config[k] for k in keys):raise ValueError('Incompatible appearance checkpoint')
            model=ColorField(common['center'],common['extent'],config['grid_resolution'],config['rank']).to(device)
            model.load_state_dict(obj['state_dict']);model.eval();history=obj['history']
        else:
            with np.load(cache/'train_pool.npz') as data:pool={k:data[k] for k in data.files}
            model,history=fit(pool,checkpoint,config,device);del pool;gc.collect()
    import lpips
    criterion=lpips.LPIPS(net='vgg',version='0.1',spatial=True).to(device).eval()
    for p in criterion.parameters():p.requires_grad_(False)
    rows=[];own_rows=[];native=[];native_own=[]
    for c in cs:
        d=np.load(cache/'depth'/(c['stem']+'.npy'))
        rendered=dest/'renders'/(c['stem']+'.png')
        prediction=np.asarray(Image.open(rendered).convert('RGB')) if rendered.exists() else render(model,c,d,device)
        com=root/'common'/scene/'held_out'
        target=np.asarray(Image.open(com/(c['stem']+'_gt.png')).convert('RGB'))
        fg=np.asarray(Image.open(com/(c['stem']+'_foreground.png')))>127
        eligible=~fg if state=='background' else np.ones(fg.shape,dtype=bool)
        own=np.isfinite(d)&(d>0)&eligible
        allowed=np.asarray(Image.open(support_root/(c['stem']+'.png')))>127
        if np.any(allowed&~own):raise ValueError('Common support includes missing mesh pixels')
        m=image_metrics(prediction,target,allowed,criterion,device,state=='background')
        own_m=image_metrics(prediction,target,own,criterion,device,state=='background')
        fraction=float(own.sum()/max(eligible.sum(),1));common_fraction=float(allowed.sum()/max(eligible.sum(),1))
        row=dict(name=c['name'],**m,mesh_coverage=fraction,common_coverage=common_fraction,
                 evaluation_pixels=int(allowed.sum()),own_pixels=int(own.sum()))
        rows.append(row);own_rows.append(dict(name=c['name'],**own_m,pixels=int(own.sum())))
        (dest/'renders').mkdir(exist_ok=True);(dest/'diagnostics').mkdir(exist_ok=True)
        Image.fromarray(prediction).save(rendered)
        Image.fromarray(own.astype('uint8')*255).save(dest/'diagnostics'/(c['stem']+'_coverage.png'))
        Image.fromarray(allowed.astype('uint8')*255).save(dest/'diagnostics'/(c['stem']+'_evaluation_mask.png'))
        error=np.clip(np.mean(np.abs(prediction.astype('float32')-target),axis=2)*3,0,255).astype('uint8')
        error[~allowed]=0;Image.fromarray(error).save(dest/'diagnostics'/(c['stem']+'_error.png'))
        raw=cache/'native_renders'/(c['stem']+'.png')
        if raw.exists():
            raw_image=np.asarray(Image.open(raw))
            native.append(dict(name=c['name'],**image_metrics(raw_image,target,allowed,criterion,device,state=='background')))
            native_own.append(dict(name=c['name'],**image_metrics(raw_image,target,own,criterion,device,state=='background')))
        print('COVERED_MESH_RGB',method,scene,state,row,'own',own_m,flush=True)
    def average(values,keys):
        return {k:float(np.mean([r[k] for r in values if r.get(k) is not None])) if any(r.get(k) is not None for r in values) else None for k in keys}
    keys=['PSNR','SSIM','LPIPS']
    mean=average(rows,[*keys,'mesh_coverage','common_coverage'])
    result=dict(complete=True,scoring_protocol_version=2,request=request,method=method,scene=scene,state=state,
                mean=mean,per_view=rows,own_surface_mean=average(own_rows,keys),own_surface_per_view=own_rows,
                native_color_mean=average(native,keys) if native else None,
                native_color_own_surface_mean=average(native_own,keys) if native_own else None,
                geometry=geometry,quality_status=metadata['scene_status'],
                train_views=common['train_count'],evaluation_views=len(cs),scored_views=sum(r['evaluation_pixels']>0 for r in rows),
                appearance_history=history,reused_existing_renders=all_rendered,
                cuda_peak_mib=torch.cuda.max_memory_allocated()/1024**2,
                notes=['Only positive finite mesh depth pixels scored; background target mask excluded',
                       'Main metrics use cohort common coverage; own coverage metrics separate',
                       'Geometry support never filtered by reconstruction error or F1 threshold',
                       'Shared VM-SH field: lightweight adaptation, not exact MILo protocol',
                       'Existing PaintMesh artifacts may include baseline holdout views'])
    write(dest/'metrics.json',result);print('COMPLETE_V2',method,scene,state,mean,flush=True)
