#!/usr/bin/env python3
"""Export normal evidence for an isolated bag whole-scene meshing experiment.

The checkpoint, 85 training cameras, native image resolution and baseline
depth/RGB are unchanged. Only additional geometric evidence is rendered. This
script never trains Gaussians, reads held-out image targets or edits production
artifacts. CUDA_VISIBLE_DEVICES is selected by the external experiment runner.
"""
from pathlib import Path
import argparse
import fcntl
import gc
import importlib.util
import os
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,REPO,read,write,identity,digest
from mesh_benchmark.layout import work_root

PUBLIC=BASE/'mesh_comparison_whole'
ROOT=work_root(PUBLIC)

def source_file(source_kind):
    if source_kind not in ['background','stage1']:raise ValueError(source_kind)
    return ROOT/'fine_whole'/source_kind/'bag/frames/complete.json'


def unit_camera_normals(normal,depth,K):
    """Return HWC unit normals facing the camera, with explicit invalid flags.

    Alpha already multiplies renderer normal numerators. Unitization removes
    that factor; its magnitude is not reused as an independent confidence.
    """
    depth=np.asarray(depth,dtype='float32');normal=np.asarray(normal,dtype='float32')
    if normal.shape==(3,*depth.shape):normal=np.moveaxis(normal,0,-1)
    if normal.shape!=(*depth.shape,3):raise ValueError('Expected HWC or CHW camera normals')
    finite=np.isfinite(normal).all(-1);safe=np.where(finite[...,None],normal,0)
    length=np.linalg.norm(safe,axis=-1)
    valid=finite&(length>1e-8)&np.isfinite(depth)&(depth>0)
    result=np.zeros_like(safe,dtype='float32')
    result[valid]=safe[valid]/length[valid,None]
    yy,xx=np.mgrid[:depth.shape[0],:depth.shape[1]]
    direction=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(np.asarray(K,dtype='float64')).T
    flip=valid&((result*direction).sum(-1)>0)
    result[flip]*=-1
    return np.ascontiguousarray(result),valid


def normal_evidence(depth,K,rendered_normal,derived_normal,alpha):
    """Construct soft normal reliability without deleting valid depth pixels.

    Invalid depth-derived normals (borders/discontinuities) lower confidence
    to 0.25*alpha. Valid rendered normals remain available in those regions.
    A valid geometric confidence is not a calibrated probability of accuracy.
    """
    depth=np.asarray(depth,dtype='float32');alpha=np.asarray(alpha,dtype='float32').squeeze()
    if alpha.shape!=depth.shape or not np.isfinite(alpha).all():raise ValueError('Invalid alpha shape/range')
    if alpha.min(initial=0)<-1e-6 or alpha.max(initial=0)>1+1e-6:raise ValueError('Alpha outside [0,1]')
    normal,nv=unit_camera_normals(rendered_normal,depth,K)
    derived,dv=unit_camera_normals(derived_normal,depth,K)
    agreement=np.zeros_like(depth,dtype='float32');both=nv&dv
    agreement[both]=np.clip((normal[both]*derived[both]).sum(-1),-1,1)
    weight=np.full_like(depth,.25,dtype='float32')
    weight[both]=.25+.75*np.maximum(agreement[both],0)**2
    confidence=np.clip(alpha,0,1)*weight
    confidence[~nv]=0
    return dict(normal=normal,depth_normal=derived,normal_valid=nv,
        depth_normal_valid=dv,normal_agreement=agreement,alpha=np.clip(alpha,0,1).astype('float32'),
        confidence=np.ascontiguousarray(confidence,dtype='float32'))


def compare_reference(cached,fresh):
    """Record full-precision differences; no silent geometry/color substitution."""
    result={}
    for name in ['depth','rgb','K','w2c']:
        a=np.asarray(cached[name]);b=np.asarray(fresh[name])
        same_shape=a.shape==b.shape
        if not same_shape:
            result[name]=dict(exact=False,cached_shape=list(a.shape),fresh_shape=list(b.shape))
            continue
        d=np.abs(a.astype('float64')-b.astype('float64'))
        result[name]=dict(exact=bool(np.array_equal(a,b)),different_values=int(np.count_nonzero(a!=b)),
            maximum_absolute_difference=float(d.max(initial=0)),mean_absolute_difference=float(d.mean()))
    result['all_exact']=all(v['exact'] for v in result.values())
    return result


def wait_for_cuda(out):
    import torch
    while True:
        free,total=torch.cuda.mem_get_info()
        if free>=10*1024**3:
            torch.cuda.set_per_process_memory_fraction(6*1024**3/total)
            return
        write(out/'normal_frames_status.json',dict(stage='waiting for free VRAM',free_vram_bytes=int(free),
            minimum_free_bytes=10*1024**3,updated=time.time(),pid=os.getpid()))
        print('NORMAL_FRAMES_WAIT_VRAM',int(free),flush=True);time.sleep(30)


def validate_sources(source,source_kind='background'):
    params=source['request']['parameters']
    if params['scene']!='bag' or params['kind']!=source_kind:
        raise ValueError('Expected the existing bag cache for the selected source kind')
    if source_kind=='stage1' and params['iteration']!=30000:raise ValueError('Unexpected stage1 iteration')
    if int(Path(params['checkpoint']).parent.name.split('_')[-1])!=params['iteration']:
        raise ValueError('Checkpoint iteration differs from its cache parameters')
    if params['resolution']!=4 or params['alpha_min']!=.5:raise ValueError('Unexpected baseline image/depth validity convention')
    if len(source['frames'])!=85:raise ValueError('The bag baseline must contain all 85 prescribed training views')
    for key,path in [('checkpoint',params['checkpoint']),('config',Path(params['model'])/'config.yaml')]:
        if identity(path)!=source['request'][key]:raise ValueError(f'Baseline {key} changed')
    for row in source['frames']:
        if identity(row['path'])!=row['identity']:raise ValueError(f'Baseline RGB-D changed: {row["path"]}')
    return params


def export(output,source_kind='background'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    folder=out/'frames';folder.mkdir(exist_ok=True)
    lock=open(folder/'normal_export.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_path=source_file(source_kind);source=read(source_path);params=validate_sources(source,source_kind)
    teacher_marker=out/'teacher_views/complete.json'
    if teacher_marker.exists() and read(teacher_marker)['source_kind']!=source_kind:
        raise ValueError('Training evidence and teacher views must use the same source kind')
    request=dict(version=1,source_kind=source_kind,source_cache=identity(source_path),source_request=source['request'],
        source_frames=[r['identity'] for r in source['frames']],implementation=identity(Path(__file__)),
        fields=['depth','rgb','K','w2c','normal','depth_normal','normal_valid','depth_normal_valid',
                'alpha','normal_agreement','confidence','rendered_distance'],
        depth_and_rgb='Exact existing baseline arrays; fresh render must reproduce them exactly',
        normal_coordinates='Unit camera-space HWC, facing camera, float32',
        confidence='alpha*(0.25+0.75*max(dot(n,depth_n),0)^2); invalid derived normal -> alpha*0.25; invalid rendered normal -> 0',
        additional_depth_filtering=False,train_views=85,training_performed=False)
    request_id=digest(request);marker=folder/'complete.json'
    if marker.exists():
        old=read(marker)
        if old['request']!=request:raise ValueError('Normal export inputs changed')
        for r in old['frames']:
            if identity(r['path'])!=r['identity']:raise ValueError('Exported normal frame changed')
        return old
    import torch
    torch.set_num_threads(4);wait_for_cuda(out)
    edgs=REPO/'submodules/EDGS';sys.path.insert(0,str(edgs))
    spec=importlib.util.spec_from_file_location('normal_evidence_edgs_render',edgs/'render.py')
    render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
    cfg=render.load_training_config(Path(params['model']),source_path=Path(params['source']),images='images',resolution=4)
    cfg.gs.dataset.data_device='cpu';cfg.gs.dataset.model_path=params['model']
    render.bootstrap_gaussian_splatting()
    from scene import GaussianModel,Scene
    renderer=render.build_renderer({'backend':'pgsr','clamp_rgb':True})
    result=[];comparisons=[]
    with torch.no_grad():
        gaussian=GaussianModel(int(cfg.gs.sh_degree));scene_model=Scene(cfg.gs.dataset,gaussian,load_iteration=params['iteration'],shuffle=False)
        wanted={Path(n).stem for n in params['train_names']}
        views=sorted([c for c in scene_model.getTrainCameras() if Path(c.image_name).stem in wanted],key=lambda c:c.image_name)
        if {Path(v.image_name).stem for v in views}!=wanted:raise ValueError('Fusion cameras differ from baseline')
        baseline={Path(r['name']).stem:r for r in source['frames']}
        background=torch.zeros(3,device='cuda')
        for i,camera in enumerate(views):
            base=baseline[Path(camera.image_name).stem];path=folder/(camera.image_name+'.npz');receipt=path.with_suffix('.json')
            frame_request=dict(campaign=request_id,source=base['identity'],camera=camera.image_name)
            if path.exists() and receipt.exists():
                previous=read(receipt)
                if previous['request']!=frame_request or previous['identity']!=identity(path):raise ValueError('Normal frame request changed')
                result.append(dict(name=camera.image_name,path=str(path),identity=previous['identity']))
                comparisons.append(previous['reproduction']);continue
            with np.load(base['path']) as a:cached={k:a[k].copy() for k in ['depth','rgb','K','w2c']}
            package=renderer(camera,gaussian,cfg.gs.pipe,background,return_plane=True,return_depth_normal=True)
            raw=package['plane_depth'].squeeze();alpha=package['rendered_alpha'].squeeze()
            valid=torch.isfinite(raw)&(raw>0)&(raw<=params['max_depth'])&(alpha>=params['alpha_min'])
            K=render.camera_intrinsics(camera,device='cpu').numpy();T=camera.world_view_transform.T.detach().cpu().numpy()
            fresh=dict(depth=torch.where(valid,raw,0).float().cpu().numpy(),
                rgb=np.rint(package['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8'),K=K,w2c=T)
            comparison=compare_reference(cached,fresh);comparison['camera']=camera.image_name
            if not comparison['all_exact']:
                write(folder/(camera.image_name+'_mismatch.json'),dict(request=frame_request,reproduction=comparison))
                raise ValueError(f'Fresh geometry/color differs from baseline; inspect {camera.image_name}_mismatch.json')
            evidence=normal_evidence(cached['depth'],cached['K'],package['rendered_normal'].detach().cpu().numpy(),
                package['depth_normal'].detach().cpu().numpy(),alpha.detach().cpu().numpy())
            evidence['rendered_distance']=package['rendered_distance'].squeeze().float().cpu().numpy()
            temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**cached,**evidence);temporary.replace(path)
            record=dict(name=camera.image_name,path=str(path),identity=identity(path));result.append(record);comparisons.append(comparison)
            write(receipt,dict(complete=True,request=frame_request,identity=record['identity'],reproduction=comparison,
                valid_depth_pixels=int((cached['depth']>0).sum()),valid_normal_pixels=int(evidence['normal_valid'].sum()),
                derived_normal_pixels=int(evidence['depth_normal_valid'].sum()),updated=time.time()))
            write(out/'normal_frames_status.json',dict(stage='rendering normal evidence',finished=i+1,total=len(views),
                updated=time.time(),pid=os.getpid(),all_previous_depth_rgb_exact=True))
            print('NORMAL_FRAME_COMPLETE',camera.image_name,i+1,len(views),'valid_normals',int(evidence['normal_valid'].sum()),flush=True)
            del package,raw,alpha,valid,fresh,cached,evidence;gc.collect()
    validate_sources(source,source_kind)
    record=dict(complete=True,request=request,request_id=request_id,frames=result,comparisons=comparisons,
        all_depth_rgb_camera_arrays_exact=True,full_scene=True,roi=None,normal_coordinates='camera',
        source_kind=source_kind,iteration=params['iteration'],depth_kind='plane_z',depth_unit='scene',
        image_downsampling=False,training_performed=False,updated=time.time())
    write(marker,record);write(out/'normal_frames_status.json',dict(stage='complete',finished=len(result),updated=time.time(),pid=os.getpid()))
    del views,camera,gaussian,scene_model,renderer;gc.collect();torch.cuda.empty_cache()
    print('NORMAL_FRAMES_COMPLETE',len(result),str(marker),flush=True)
    return record


def teacher_camera(metadata):
    """Create only camera geometry; never read a held-out image's RGB bytes."""
    import math,torch
    from types import SimpleNamespace
    from utils.graphics_utils import getProjectionMatrix
    K=np.asarray(metadata['K'],dtype='float64');T=np.asarray(metadata['w2c'],dtype='float32')
    width,height=metadata['width'],metadata['height']
    if not np.allclose(K[:2,2],[width/2,height/2],rtol=0,atol=1e-5):
        raise ValueError('PGSR source camera uses a centered projection; held-out K is not centered')
    fovx=2*math.atan(width/(2*K[0,0]));fovy=2*math.atan(height/(2*K[1,1]))
    view=torch.tensor(T,device='cuda').T
    projection=getProjectionMatrix(znear=.01,zfar=100.,fovX=fovx,fovY=fovy).T.cuda()
    return SimpleNamespace(image_width=width,image_height=height,FoVx=fovx,FoVy=fovy,
        Fx=float(K[0,0]),Fy=float(K[1,1]),Cx=float(K[0,2]),Cy=float(K[1,2]),
        image_name=metadata['stem'],world_view_transform=view,projection_matrix=projection,
        full_proj_transform=view@projection,camera_center=view.inverse()[3,:3],znear=.01,zfar=100.)


def export_teacher_views(output,source_kind='background'):
    """Render frozen-GS geometry at all 13 held-out poses for diagnostics only.

    These views are never used by surface fusion/refinement. Their RGB is GS
    output; neither construction nor weighting reads held-out target photos.
    """
    out=Path(output);out.mkdir(parents=True,exist_ok=True);folder=out/'teacher_views';folder.mkdir(exist_ok=True)
    lock=open(folder/'normal_export.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=source_file(source_kind);source=read(path);params=validate_sources(source,source_kind)
    training_marker=out/'frames/complete.json'
    if training_marker.exists() and read(training_marker)['source_kind']!=source_kind:
        raise ValueError('Teacher views and training evidence must use the same source kind')
    common=ROOT/'common/bag/complete.json';cameras=[c for c in read(common)['cameras'] if c['part']=='held_out']
    if len(cameras)!=13:raise ValueError('Expected all 13 frozen bag held-out poses')
    request=dict(version=1,source_kind=source_kind,source_cache=identity(path),checkpoint=source['request']['checkpoint'],
        config=source['request']['config'],implementation=identity(Path(__file__)),cameras=cameras,
        calibration=identity(common),purpose='Frozen GS geometry teacher for independent evaluation only',
        held_out_target_rgb_read=False,training_performed=False,used_for_fusion=False,used_for_refinement=False)
    request_id=digest(request);marker=folder/'complete.json'
    if marker.exists():
        old=read(marker)
        if old['request']!=request:raise ValueError('Teacher view inputs changed')
        if any(identity(r['path'])!=r['identity'] for r in old['frames']):raise ValueError('Teacher view changed')
        return old
    import torch
    torch.set_num_threads(4);wait_for_cuda(out)
    edgs=REPO/'submodules/EDGS';sys.path.insert(0,str(edgs))
    spec=importlib.util.spec_from_file_location('normal_teacher_edgs_render',edgs/'render.py')
    render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
    cfg=render.load_training_config(Path(params['model']),source_path=Path(params['source']),images='images',resolution=4)
    render.bootstrap_gaussian_splatting()
    from scene import GaussianModel
    renderer=render.build_renderer({'backend':'pgsr','clamp_rgb':True});records=[]
    with torch.no_grad():
        gaussian=GaussianModel(int(cfg.gs.sh_degree));gaussian.load_ply(params['checkpoint'],cfg.gs.dataset.train_test_exp)
        background=torch.zeros(3,device='cuda')
        for i,metadata in enumerate(cameras):
            dest=folder/(metadata['stem']+'.npz');receipt=dest.with_suffix('.json')
            if dest.exists() and receipt.exists():
                old=read(receipt)
                if old['request_id']!=request_id or old['identity']!=identity(dest):raise ValueError('Teacher frame changed')
                records.append(dict(name=metadata['name'],path=str(dest),identity=old['identity']));continue
            camera=teacher_camera(metadata)
            package=renderer(camera,gaussian,cfg.gs.pipe,background,return_plane=True,return_depth_normal=True)
            raw=package['plane_depth'].squeeze();alpha=package['rendered_alpha'].squeeze()
            valid=torch.isfinite(raw)&(raw>0)&(raw<=params['max_depth'])&(alpha>=params['alpha_min'])
            depth=torch.where(valid,raw,0).float().cpu().numpy();K=np.asarray(metadata['K'],dtype='float32')
            rgb=np.rint(package['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')
            evidence=normal_evidence(depth,K,package['rendered_normal'].detach().cpu().numpy(),
                package['depth_normal'].detach().cpu().numpy(),alpha.detach().cpu().numpy())
            evidence['rendered_distance']=package['rendered_distance'].squeeze().float().cpu().numpy()
            temporary=dest.with_suffix('.tmp.npz');np.savez_compressed(temporary,depth=depth,rgb=rgb,K=K,
                w2c=np.asarray(metadata['w2c'],dtype='float32'),**evidence);temporary.replace(dest)
            record=dict(name=metadata['name'],path=str(dest),identity=identity(dest));records.append(record)
            write(receipt,dict(complete=True,identity=record['identity'],request_id=request_id))
            print('NORMAL_TEACHER_VIEW_COMPLETE',metadata['name'],i+1,len(cameras),flush=True)
            del camera,package,raw,alpha,valid,depth,rgb,evidence;gc.collect()
    validate_sources(source,source_kind)
    result=dict(complete=True,request=request,request_id=request_id,frames=records,views=len(records),
        normal_coordinates='camera',source_kind=source_kind,iteration=params['iteration'],depth_kind='plane_z',depth_unit='scene',
        used_for_fusion=False,used_for_refinement=False,
        held_out_target_rgb_read=False,updated=time.time())
    write(marker,result);del gaussian,renderer;gc.collect();torch.cuda.empty_cache()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-kind',choices=['background','stage1'],default='background')
    parser.add_argument('--teacher-views-only',action='store_true');parser.add_argument('--include-held-out',action='store_true')
    args=parser.parse_args()
    if not args.teacher_views_only:export(args.output,args.source_kind)
    if args.teacher_views_only or args.include_held_out:export_teacher_views(args.output,args.source_kind)
