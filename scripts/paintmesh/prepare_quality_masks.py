#!/usr/bin/env python3
"""Correct training-only removal annotations in independent workspaces."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
from run_geometry_ablation import read, write, now, sha

REPO=Path(__file__).resolve().parents[2]


def project(xyz,c):
    w,h=c['image_width'],c['image_height']
    p=xyz@np.asarray(c['R'])+np.asarray(c['T'])
    uv=p[:,:2]/np.maximum(p[:,2:],1e-9)*[w/2/np.tan(c['FoVx']/2),h/2/np.tan(c['FoVy']/2)]+[w/2,h/2]
    valid=(p[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return uv,p[:,2],valid


def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if dst.resolve()!=src.resolve():raise ValueError('Existing different path: '+str(dst))
    else:dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())


def prepare_masks(case,job):
    scene=case['scene'];old=Path(case['source_removal']);root=Path(case['removal_root'])
    if (root/'tracker/quality_correction.json').exists():return
    for name in ('work_model','removal_manifest.json','removed_3dgs','removed_mesh','config'):
        link(old/name,root/name)
    for name in ('virtual_cameras.json','images.zip'):link(old/'tracker'/name,root/'tracker'/name)
    cameras=read(old/'tracker/virtual_cameras.json')['cameras']
    dest=root/'tracker/results/images/images_masks';dest.mkdir(parents=True,exist_ok=True)
    review=root/'tracker/review';review.mkdir(exist_ok=True)
    session=copy.deepcopy(read(old/'tracker/tracking_session.json'));records=[];stats=[];pages=[];started=time.time_ns()
    if scene=='car':
        source=next(s for s in read(job/'request.json')['scenes'] if s['scene']==scene)
        lama_inputs=read(Path(source['root'])/'manifests/lama_input_manifest.json')
        model=REPO/'output/paintmesh/Inpaint360GS/inpaint360'/scene
        v=PlyData.read(old/'work_model/point_cloud/iteration_2000/point_cloud.ply')['vertex'].data
        labels=np.load(model/'semantic_mesh/gaussian_instance_id.npy')
        cone=np.isin(labels,[2,4,5,6])&(v['opacity']>np.log(.1/.9))
        xyz=np.stack([v[k][cone] for k in 'xyz'],1)
    for i,camera in enumerate(cameras):
        name=camera['image_name']+'.png';old_mask=np.asarray(Image.open(old/'tracker/results/images/images_masks'/name))>0
        if scene=='car':
            # Build non-target protection from the scene's own 3D semantic cones.
            uv,z,valid=project(xyz,camera);uv=np.rint(uv).astype(np.int64)
            h,w=old_mask.shape;valid&=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            depth=np.load(old/f'work_model/virtual/ours_2000/depth/{camera["image_name"]}.npy').squeeze()
            idx=np.flatnonzero(valid);d=depth[uv[idx,1],uv[idx,0]]
            idx=idx[(d>0)&(np.abs(z[idx]-d)<.04*np.maximum(d,1e-6))]
            protected=np.zeros((h,w),bool);protected[uv[idx,1],uv[idx,0]]=True
            protected=ndi.binary_closing(ndi.binary_dilation(protected,iterations=3),iterations=2)
            protected=ndi.binary_fill_holes(protected)
            grown=np.asarray(Image.open(lama_inputs['frames'][camera['image_name']]['outputs']['color_mask']['path']).convert('L'))>0
            mask=grown&~ndi.maximum_filter(protected,size=11)
            stats.append(dict(frame=name,old_fraction=float(old_mask.mean()),new_fraction=float(mask.mean()),protected_fraction=float(protected.mean()),removed_protected_overlap=int((grown&protected).sum())))
        else:
            alpha=np.load(old/f'work_model/virtual/ours_object_removal/iteration_2000/alpha/{camera["image_name"]}.npy').squeeze()
            mask=old_mask&(alpha<.95)
            mask=ndi.binary_fill_holes(ndi.binary_closing(mask,iterations=2))
            lab,_=ndi.label(mask);counts=np.bincount(lab.ravel());keep=counts>=50;keep[0]=False;mask=keep[lab]
            stats.append(dict(frame=name,old_fraction=float(old_mask.mean()),new_fraction=float(mask.mean())))
        if not 50<=mask.sum()<mask.size*.4:raise ValueError('Invalid repaired mask '+scene+'/'+name)
        p=dest/name;Image.fromarray(mask.astype('uint8')).save(p);s=p.stat()
        records.append(dict(file=name,size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p)))
        rgb=np.array(Image.open(old/'work_model/virtual/ours_2000/renders'/name).convert('RGB'))
        radius=case['mask_dilation']
        final=cv2.dilate(mask.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
        rgb[final]=(.55*rgb[final]+.45*np.array([255,20,30])).astype('uint8')
        if i%30==0:pages.append(Image.new('RGB',(1500,1200),'white'))
        im=Image.fromarray(rgb);im.thumbnail((300,175));x=i%5*300;y=(i%30)//5*200
        pages[-1].paste(im,(x,y+22));ImageDraw.Draw(pages[-1]).text((x,y),f'{name} {final.mean():.1%}',fill='black')
    for i,im in enumerate(pages):im.save(review/f'quality_{i}.jpg',quality=92)
    session.update(complete=True,status='complete',started_at=now(),started_at_ns=started,completed_at=now(),outputs=dict(masks=records),
                   annotation_correction=dict(source_session=str(old/'tracker/tracking_session.json'),source_sha256=sha(old/'tracker/tracking_session.json'),method=case['changes'],uses_test_GT=False))
    write(root/'tracker/tracking_session.json',session)
    write(root/'tracker/quality_correction.json',dict(complete=True,scene=scene,source=str(old),no_test_GT_used=True,visually_reviewed_all_frames=False,statistics=stats,config=case))


def separate_doppelherz(case,job):
    old=Path(case['source_removal']);root=Path(case['removal_root']);dest=root/'work_model/point_cloud_object_removal/iteration_2000'
    if (root/'target_selection.json').exists():return
    original_path=old/'work_model/point_cloud/iteration_2000/point_cloud.ply'
    target_path=old/'work_model/point_cloud_object_removal/iteration_2000/point_cloud_18.ply'
    original=PlyData.read(original_path)['vertex'].data;target=PlyData.read(target_path)['vertex'].data
    xyz=np.stack([original[k] for k in 'xyz'],1);target_xyz=np.stack([target[k] for k in 'xyz'],1)
    distance,indices=cKDTree(xyz).query(target_xyz,workers=4)
    if distance.max()>1e-7 or len(np.unique(indices))!=len(target):raise ValueError('Cannot identify source target rows')
    # Visual annotation on three ORIGINAL VIRTUAL TRAINING views, not test RGB.
    polygons={0:dict(target=[(265,0),(465,0),(465,180),(285,191)],protected=[(308,273),(439,272),(453,411),(310,418)]),
              60:dict(target=[(164,109),(270,100),(311,322),(208,355),(171,274)],protected=[(444,68),(551,92),(581,202),(441,259),(393,166)]),
              120:dict(target=[(458,151),(564,201),(575,301),(488,399),(363,345),(375,265)],protected=[(213,24),(329,33),(347,167),(298,253),(216,225),(201,104)])}
    cameras=read(old/'tracker/virtual_cameras.json')['cameras'];votes=np.zeros(len(target),int);protect=np.zeros(len(target),int)
    review=root/'target_review';review.mkdir(parents=True,exist_ok=True)
    for i,polys in polygons.items():
        c=cameras[i];uv,_,valid=project(target_xyz,c);pixel=np.rint(uv).astype(int)
        w,h=c['image_width'],c['image_height'];valid&=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
        for key,out in [('target',votes),('protected',protect)]:
            canvas=Image.new('L',(w,h));ImageDraw.Draw(canvas).polygon(polys[key],fill=255);mask=np.asarray(canvas)>0
            ix=np.flatnonzero(valid);out[ix]+=mask[pixel[ix,1],pixel[ix,0]]
        im=Image.open(old/f'work_model/virtual/ours_2000/renders/{i:05d}.png').convert('RGB');draw=ImageDraw.Draw(im)
        draw.polygon(polys['target'],outline='red',width=3);draw.polygon(polys['protected'],outline='lime',width=3);im.save(review/f'annotated_{i:05d}.png')
    chosen=(votes>=2)&(protect==0)
    if not 1000<chosen.sum()<len(target)*.9:raise ValueError('Target/protected separation requires inspection')
    removed=np.zeros(len(original),bool);removed[indices[chosen]]=True
    for name in ('point_cloud','cfg_args','workspace_manifest.json'):link(old/'work_model'/name,root/'work_model'/name)
    if not (root/'config').exists():shutil.copytree(old/'config',root/'config')
    dest.mkdir(parents=True,exist_ok=True)
    for name,data in [('point_cloud.ply',original[~removed]),('point_cloud_18.ply',original[removed])]:PlyData([PlyElement.describe(data,'vertex')],text=False).write(dest/name)
    write(root/'target_selection.json',dict(complete=True,no_test_GT_used=True,source_ply=dict(path=str(original_path),sha256=sha(original_path)),
        source_target=dict(path=str(target_path),sha256=sha(target_path)),method='old removal membership intersected with >=2/3 target silhouette annotations, veto protected carton in any annotated view',
        polygons=polygons,source_target_count=len(target),corrected_target_count=int(removed.sum()),restored_non_target_count=int(len(target)-removed.sum()),
        remaining_count=int((~removed).sum()),outputs={p.name:sha(p) for p in dest.glob('*.ply')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--case',required=True);a=p.parse_args()
    case=next(c for c in read(a.job/'upstream_experiments.json') if c['scene']==a.case)
    if a.case=='doppelherz':separate_doppelherz(case,a.job)
    else:prepare_masks(case,a.job)
