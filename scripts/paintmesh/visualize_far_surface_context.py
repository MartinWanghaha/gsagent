#!/usr/bin/env python3
"""Show the local bag experiment inside the unchanged complete stage-1 mesh.

This is a local geometric replacement, not a full-scene rerun of B or C.
Ray depths handle visibility and reveal existing geometry behind removed ROI
surfaces. Pixels outside the ROI projection retain the published PNG bytes.
"""
from pathlib import Path
import argparse
import gc
import sys
import time
import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,read,write,identity,MeshScene,rays
from mesh_benchmark.layout import work_root

PUBLIC=BASE/'mesh_comparison_whole';ROOT=work_root(PUBLIC)


def ray_box_interval(rr,bounds):
    rr=np.asarray(rr,dtype='float64');lo,hi=np.asarray(bounds,dtype='float64')
    origin=rr[:,:3];direction=rr[:,3:];parallel=np.abs(direction)<1e-12
    a=np.divide(lo-origin,direction,out=np.full_like(origin,-np.inf),where=~parallel)
    b=np.divide(hi-origin,direction,out=np.full_like(origin,np.inf),where=~parallel)
    near=np.minimum(a,b);far=np.maximum(a,b)
    outside=(parallel&((origin<lo)|(origin>hi))).any(1)
    entry=np.maximum(near.max(1),0);exit=far.min(1)
    good=(exit>=entry)&~outside
    return entry,exit,good


def replace_roi(old_depth,old_rgb,local_depth,local_rgb,inside_old,behind_depth,behind_rgb):
    depth=old_depth.copy();image=old_rgb.copy()
    depth[inside_old]=behind_depth[inside_old];image[inside_old]=behind_rgb[inside_old]
    front=(local_depth>0)&((depth==0)|(local_depth<depth))
    depth[front]=local_depth[front];image[front]=local_rgb[front]
    return depth,image


def run(out):
    from plot_mesh_method_comparison import font
    out=Path(out);plan=read(out/'plan.json');roi=np.asarray(plan['roi'])
    summary=read(out/'results.json');selected=read(out/'figures/selection.json')['camera']
    cameras=[c for c in read(ROOT/'common/bag/complete.json')['cameras'] if c['part']=='held_out']
    active={r['camera'] for r in summary['methods']['A']['per_view'] if r['eligible_pixels']}
    cameras=[c for c in cameras if c['name'] in active]
    specs=read(ROOT/'fine_whole/extensions/stage1/bag/complete.json')['mesh_specs']
    gltf=read(PUBLIC/'meshes/bag/stage1_world/mesh_scene.gltf');bounds=[]
    if len(specs)!=len(gltf['meshes']):raise ValueError('Mesh source/export order mismatch')
    for mesh in gltf['meshes']:
        a=gltf['accessors'][mesh['primitives'][0]['attributes']['POSITION']];bounds.append([a['min'],a['max']])
    views=[]
    for c in cameras:
        folder=ROOT/'cache/paintmesh_stage1/bag/background';rr=rays(c)
        z=np.load(folder/'depth'/(c['stem']+'.npy'));rgb=np.asarray(Image.open(folder/'native_renders'/(c['stem']+'.png')).convert('RGB'))
        entry,exit,projection=ray_box_interval(rr,roi)
        xyz=rr[:,:3]+rr[:,3:]*z.reshape(-1,1)
        removed=(z.reshape(-1)>0)&((xyz>=roi[0])&(xyz<=roi[1])).all(1)
        indices=np.flatnonzero(removed);shift=exit[indices]+1e-5
        shifted=rr[indices].copy();shifted[:,:3]+=shift[:,None]*rr[indices,3:]
        views.append(dict(camera=c,rays=rr,depth=z,rgb=rgb,projection=projection,
            removed=removed.reshape(z.shape),indices=indices,shift=shift,shifted=shifted,
            behind=np.zeros(len(indices),dtype='float32'),color=np.zeros((len(indices),3),dtype='float32')))
    parts=0
    for i,(spec,box) in enumerate(zip(specs,bounds)):
        candidates=[]
        for v in views:
            if not len(v['shifted']):candidates.append(np.array([],dtype='int64'));continue
            _,_,hit=ray_box_interval(v['shifted'],box);candidates.append(np.flatnonzero(hit))
        if not any(len(x) for x in candidates):continue
        model=MeshScene([spec]);parts+=1
        for v,ii in zip(views,candidates):
            if not len(ii):continue
            depth,color=model.cast(v['shifted'][ii],color=True)
            keep=(depth>0)&((v['behind'][ii]==0)|(depth<v['behind'][ii]))
            v['behind'][ii[keep]]=depth[keep];v['color'][ii[keep]]=color[keep]
        del model;gc.collect()
        print('FULL_CONTEXT_BACKGROUND',i+1,len(specs),'candidate_parts',parts,flush=True)
    records=[]
    for v in views:
        c=v['camera'];w,h=c['width'],c['height'];folder=out/'full_scene'/c['stem'];folder.mkdir(parents=True,exist_ok=True)
        background=np.zeros((h*w,),dtype='float32');color=np.zeros((h*w,3),dtype='uint8')
        hit=v['behind']>0;indices=v['indices'][hit]
        background[indices]=v['behind'][hit]+v['shift'][hit]
        color[indices]=np.rint(np.clip(v['color'][hit],0,1)*255).astype('uint8')
        background=background.reshape(h,w);color=color.reshape(h,w,3)
        gt=np.asarray(Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_gt.png')).convert('RGB'))
        images=[gt];counts={}
        for key in ['A','B','C']:
            z=np.load(out/key/'renders'/(c['stem']+'_depth.npy'))
            rgb=np.asarray(Image.open(out/key/'renders'/(c['stem']+'.png')).convert('RGB'))
            depth,image=replace_roi(v['depth'],v['rgb'],z,rgb,v['removed'],background,color)
            outside=~v['projection'].reshape(h,w)
            np.testing.assert_array_equal(image[outside],v['rgb'][outside])
            images.append(image);Image.fromarray(image).save(folder/(key+'.png'))
            np.save(folder/(key+'_depth.npy'),depth)
            counts[key]=dict(changed_pixels=int(np.any(image!=v['rgb'],axis=2).sum()),
                outside_roi_pixels_preserved_exact=True)
        Image.fromarray(gt).save(folder/'GT.png')
        yy,xx=np.where(v['projection'].reshape(h,w));box=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
        labels=['GT','A: current extraction','B: surface footprint','C: footprint + inverse depth']
        sheet=Image.new('RGB',(w*4,h+80),(244,246,249));draw=ImageDraw.Draw(sheet)
        draw.text((10,5),f'bag | {c["name"]} | full-frame context of a LOCAL experiment',font=font(20),fill='black')
        for j,(label,pic) in enumerate(zip(labels,images)):
            draw.text((j*w+10,35),label,font=font(19),fill='black');sheet.paste(Image.fromarray(pic),(j*w,70))
        sheet.save(folder/'comparison_full.png')
        marked=sheet.copy();draw=ImageDraw.Draw(marked)
        for j in range(4):
            x0,y0,x1,y1=box;draw.rectangle((j*w+x0,70+y0,j*w+x1,70+y1),outline=(245,30,100),width=2)
        marked.save(folder/'comparison_full_marked.png')
        records.append(dict(camera=c['name'],roi_box=box,methods=counts,full_image=str(folder/'comparison_full.png')))
        if c['name']==selected:
            sheet.save(out/'comparison_full.png');marked.save(out/'comparison_full_marked.png')
        print('FULL_CONTEXT_FIGURE',c['name'],counts,flush=True)
    write(out/'full_scene/manifest.json',dict(complete=True,updated=time.time(),records=records,
        candidate_mesh_parts=parts,selected_camera=selected,reference_depth=identity(ROOT/'cache/paintmesh_stage1/bag/background/depth/IMG_0119.npy'),
        scope='Only the fixed 3D ROI is replaced; all other surfaces use the unchanged published stage1 mesh. B/C are not full-scene reconstructions.',
        local_source='Existing A/B/C mesh ray renders; original native-resolution pixels',
        removed_surface_background='True mesh rays start after the ROI exit; depth-buffer composition handles occlusion',
        production_results_modified=False))
    (out/'full_scene/README.md').write_text('# bag 远景实验的整张图展示\n\n'
        '四列为 GT、原提取 A、表面覆盖 B、连续逆深度积分 C。将局部实验网格替换到现有 stage1 全场景中，按深度处理遮挡，移除局部旧表面后显露后方原网格。区域之外保持原渲染。\n\n'
        '这不是 B/C 的全场景重提取结果；它们只改变固定实验区域。主结果和实验指标未修改。\n\n'+
        '\n'.join(f'- [{r["camera"]}]({Path(r["full_image"]).parent.name}/comparison_full.png) · [区域框]({Path(r["full_image"]).parent.name}/comparison_full_marked.png)' for r in records)+'\n')
    readme=out/'README.md';text=readme.read_text()
    note='\n[整张图对比](comparison_full.png) · [整张图＋实验区域标注](comparison_full_marked.png) · [全部可见视角](full_scene/README.md)\n\n这是局部实验嵌入原 stage1 完整场景的展示，区域之外沿用原网格，不代表 B/C 已全场景重提取。\n'
    if '[整张图对比]' not in text:readme.write_text(text+note)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PUBLIC/'experiments/bag_surface_footprint_20261111')
    run(parser.parse_args().output)
