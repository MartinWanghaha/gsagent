#!/usr/bin/env python3
"""Shared two-card full-frame PGSR evidence export for the depth-tetra pilot."""
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
from export_normal_fusion_frames import normal_evidence,compare_reference,validate_sources,teacher_camera

PUBLIC=BASE/'mesh_comparison_whole';ROOT=work_root(PUBLIC)


def prepare(out):
    out=Path(out);folder=out/'frames';folder.mkdir(parents=True,exist_ok=True)
    original=ROOT/'fine_whole/background/bag/frames/complete.json';source=read(original)
    validate_sources(source,'background')
    request=dict(source_manifest=identity(original),source_frames=source['frames'],source_request=source['request'],
        implementation=identity(Path(__file__)),normal_helper=identity(HERE/'export_normal_fusion_frames.py'),
        source_intrinsics=identity(Path(source['request']['parameters']['source'])/'sparse/0/cameras.bin'),
        source_extrinsics=identity(Path(source['request']['parameters']['source'])/'sparse/0/images.bin'),
        train_views=85,full_scene=True,native_resolution=True,normal_coordinates='camera',depth_kind='plane_z')
    result=dict(complete=True,request=request,request_id=digest(request),records=source['frames'],parameters=source['request']['parameters'])
    marker=folder/'manifest.json'
    if marker.exists():
        old=read(marker)
        if old['request']!=request:raise ValueError('Evidence sources changed')
        return old
    write(marker,result);return result


def completed(out,index,plan):
    receipt=Path(out)/'frames/completed'/f'{index:03d}.json'
    if not receipt.exists():return None
    row=read(receipt)
    if row['request_id']!=plan['request_id'] or identity(row['path'])!=row['identity']:raise ValueError('Exported evidence changed')
    return row


def progress(out):
    out=Path(out);plan=read(out/'frames/manifest.json')
    return dict(completed_frames=sum(completed(out,i,plan) is not None for i in range(len(plan['records']))),total_frames=85)


def stop(out):
    if (Path(out)/'STOP_REQUESTED.json').exists():raise KeyboardInterrupt('Pilot stop requested')


def camera_without_image(source_name,cached,params):
    """Reproduce the original camera without decoding unused RGB photos.

    Cached K is Float32. Reconstructing FoV from that rounded matrix changes
    the CUDA projection at some pixels, so preserve original COLMAP Float64
    focal lengths and image sizes when forming the projection matrix. Cached
    world-to-camera and the original native rendered dimensions stay exact.
    """
    import torch
    from scene.colmap_loader import read_intrinsics_binary,read_extrinsics_binary
    from utils.graphics_utils import getProjectionMatrix,focal2fov
    if '_source_cameras' not in params:
        root=Path(params['source'])/'sparse/0'
        intrinsics=read_intrinsics_binary(str(root/'cameras.bin'))
        extrinsics=read_extrinsics_binary(str(root/'images.bin'))
        params['_source_cameras']={Path(row.name).stem:intrinsics[row.camera_id] for row in extrinsics.values()}
    intrinsic=params['_source_cameras'][Path(source_name).stem]
    if intrinsic.model not in ['SIMPLE_PINHOLE','PINHOLE']:raise ValueError('Expected original undistorted pinhole calibration')
    fx=intrinsic.params[0];fy=intrinsic.params[1] if intrinsic.model=='PINHOLE' else fx
    fovx=focal2fov(fx,intrinsic.width);fovy=focal2fov(fy,intrinsic.height)
    height,width=cached['depth'].shape
    camera=teacher_camera(dict(width=width,height=height,K=cached['K'],w2c=cached['w2c'],stem=Path(source_name).stem))
    camera.FoVx=fovx;camera.FoVy=fovy
    camera.projection_matrix=getProjectionMatrix(znear=.01,zfar=100.,fovX=fovx,fovY=fovy).T.cuda()
    camera.full_proj_transform=camera.world_view_transform@camera.projection_matrix
    return camera


def worker(out,worker_id):
    import torch
    out=Path(out);plan=read(out/'frames/manifest.json');params=plan['parameters'];torch.set_num_threads(4)
    while True:
        stop(out);free,total=torch.cuda.mem_get_info()
        if free>=8*1024**3:break
        write(out/'frames/workers'/(str(worker_id)+'.json'),dict(stage='waiting for 8 GiB free VRAM',free_vram=int(free),pid=os.getpid(),updated=time.time()))
        time.sleep(15)
    torch.cuda.set_per_process_memory_fraction(6*1024**3/total)
    edgs=REPO/'submodules/EDGS';sys.path.insert(0,str(edgs))
    spec=importlib.util.spec_from_file_location('tetra_evidence_renderer',edgs/'render.py');render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
    cfg=render.load_training_config(Path(params['model']),source_path=Path(params['source']),images='images',resolution=4)
    cfg.gs.dataset.data_device='cpu';cfg.gs.dataset.model_path=params['model'];render.bootstrap_gaussian_splatting()
    from scene import GaussianModel
    renderer=render.build_renderer({'backend':'pgsr','clamp_rgb':True})
    with torch.no_grad():
        gaussians=GaussianModel(int(cfg.gs.sh_degree));gaussians.load_ply(params['checkpoint'],cfg.gs.dataset.train_test_exp)
        background=torch.zeros(3,device='cuda');finished=0
        for i,source in enumerate(plan['records']):
            stop(out)
            lockpath=out/'frames/claims'/f'{i:03d}.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
            with lockpath.open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if completed(out,i,plan):continue
                if identity(source['path'])!=source['identity']:raise ValueError('Source depth frame changed')
                with np.load(source['path']) as a:cached={k:a[k].copy() for k in ['depth','rgb','K','w2c']}
                camera=camera_without_image(source['name'],cached,params)
                package=renderer(camera,gaussians,cfg.gs.pipe,background,return_plane=True,return_depth_normal=True)
                raw=package['plane_depth'].squeeze();alpha=package['rendered_alpha'].squeeze()
                valid=torch.isfinite(raw)&(raw>0)&(raw<=params['max_depth'])&(alpha>=params['alpha_min'])
                fresh=dict(depth=torch.where(valid,raw,0).float().cpu().numpy(),
                    rgb=np.rint(package['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8'),
                    K=render.camera_intrinsics(camera,device='cpu').numpy(),w2c=camera.world_view_transform.T.detach().cpu().numpy())
                comparison=compare_reference(cached,fresh)
                if not comparison['all_exact']:
                    write(out/'frames'/f'{i:03d}_mismatch.json',comparison);raise ValueError('Frozen PGSR geometry could not reproduce original cache')
                evidence=normal_evidence(cached['depth'],cached['K'],package['rendered_normal'].detach().cpu().numpy(),
                    package['depth_normal'].detach().cpu().numpy(),alpha.detach().cpu().numpy())
                path=out/'frames'/f'{i:03d}.npz';tmp=path.with_suffix('.tmp.npz')
                np.savez(tmp,**cached,**evidence);tmp.replace(path)
                row=dict(complete=True,name=source['name'],path=str(path),identity=identity(path),index=i,
                    request_id=plan['request_id'],reproduction=comparison,worker=str(worker_id),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),updated=time.time())
                write(out/'frames/completed'/f'{i:03d}.json',row);finished+=1
                write(out/'frames/workers'/(str(worker_id)+'.json'),dict(stage='exporting',frame=i,completed_by_worker=finished,pid=os.getpid(),updated=time.time()))
                print('DEPTH_TETRA_FRAME_COMPLETE',worker_id,i,source['name'],flush=True)
                del package,raw,alpha,valid,cached,fresh,evidence;gc.collect();torch.cuda.empty_cache()


def finalize(out):
    out=Path(out);plan=read(out/'frames/manifest.json');records=[]
    for i in range(85):
        row=completed(out,i,plan)
        if row is None:raise ValueError('Evidence incomplete')
        records.append(row)
    result=dict(complete=True,frames=records,request=dict(train_views=85),request_id=plan['request_id'],
        source_kind='background',all_depth_rgb_camera_arrays_exact=True,normal_coordinates='camera',depth_kind='plane_z',
        all_source_frames=True,image_downsampling=False,full_scene=True)
    write(out/'frames/complete.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args.output)
    elif args.finalize:finalize(args.output)
    elif args.worker is not None:worker(args.output,args.worker)
    else:parser.error('Choose --prepare, --finalize or --worker')
