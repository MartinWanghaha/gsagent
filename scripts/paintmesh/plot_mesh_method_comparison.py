#!/usr/bin/env python3
"""Real mesh RGB/depth figures: four baselines, stage 1, two PaintMesh modes.

All panels use completed triangle renders with their declared appearance.
Views are selected from the GT mask area, independently of method scores.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

HERE=Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,METHODS,read,write,identity,digest,points_from_depth

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
COLS=[('GT',None,None),('2DGS','2dgs','recomposition'),('CoMe','come','recomposition'),
      ('GOF','gof','recomposition'),('Gaussian Wrapping','gaussian_wrapping','recomposition'),
      ('PaintMesh: stage 1','paintmesh_stage1','recomposition'),
      ('PaintMesh: background','paintmesh','background'),('PaintMesh: composed','paintmesh','recomposition')]
PAIRED_LABELS=[('2DGS','2dgs'),('CoMe','come'),('GOF','gof'),('Gaussian Wrapping','gaussian_wrapping'),
               ('PaintMesh: stage 1','paintmesh_stage1'),('PaintMesh','paintmesh')]
BG=(247,248,250);INK=(26,35,48);GREEN=(16,125,91);PINK=(226,64,130)

def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)

def text(draw,xy,value,size=20,color=INK,bold=False):
    draw.text(xy,str(value),font=font(size,bold),fill=color)

def save(image,path,formats=('png',)):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for ext in formats:
        p=path.with_suffix('.'+ext);temp=p.with_name('.'+p.name+'.tmp')
        kw=dict(format='JPEG',quality=94,subsampling=0) if ext=='jpg' else dict(format=ext.upper())
        if ext=='pdf':kw.update(format='PDF',resolution=180.)
        image.save(temp,**kw);temp.replace(p)

def contained(image,size):
    result=Image.new('RGB',size,'white');p=ImageOps.contain(image.convert('RGB'),size,Image.Resampling.LANCZOS)
    result.paste(p,((size[0]-p.width)//2,(size[1]-p.height)//2));return result

def placeholder(size,message):
    im=Image.new('RGB',size,(235,237,240));d=ImageDraw.Draw(im)
    for y in range(0,size[1],32):d.line([(0,y),(min(y*2,size[0]),0)],fill=(228,230,233),width=2)
    lines=message.split('\n');top=size[1]//2-len(lines)*15
    for i,line in enumerate(lines):
        box=d.textbbox((0,0),line,font=font(20));text(d,((size[0]-box[2])//2,top+i*30),line,size=20,color=(105,112,123))
    return im

def choose_view(root,scene):
    info=read(root/'common'/scene/'complete.json');cs=[c for c in info['cameras'] if c['part']=='held_out']
    scored=[]
    for c in cs:
        p=root/'common'/scene/'held_out'/(c['stem']+'_foreground.png')
        m=np.asarray(Image.open(p))>127
        eligible=True
        for state in ['background','recomposition']:
            valid=scoring_mask(root,scene,state,c)
            if valid is not None and valid.sum()<100:eligible=False
        scored.append((float(m.mean()),c,m,eligible))
    scored=[x for x in scored if x[3]] or scored
    _,camera,mask,_=max(scored,key=lambda x:(x[0],x[1]['name']))
    yy,xx=np.where(mask)
    if not len(xx):box=(camera['width']*.25,camera['height']*.25,camera['width']*.75,camera['height']*.75)
    else:box=(xx.min(),yy.min(),xx.max()+1,yy.max()+1)
    x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2
    h=max((y1-y0)*1.6,(x1-x0)*1.6/2,50.);w=h*2
    if w>camera['width']:w=camera['width'];h=w/2
    if h>camera['height']:h=camera['height'];w=h*2
    x0=max(0,min(camera['width']-w,cx-w/2));y0=max(0,min(camera['height']-h,cy-h/2))
    roi=[int(round(x0)),int(round(y0)),int(round(x0+w)),int(round(y0+h))]
    return camera,roi,float(mask.mean())

def sources(root,scene,label,method,state,camera):
    if method is None:
        return dict(label=label,method=None,state='GT',rgb=root/'common'/scene/'held_out'/(camera['stem']+'_gt.png'),metric=None,geometry=None,status='reference RGB')
    if method in ['come','gof'] and read(root/'protocol.json').get('native_support_validation',{}).get('enabled'):
        label+=' (support validated)'
    base=root/state/method/scene;cache=root/'cache'/method/scene/state
    mp=base/'metrics.json';gp=cache/'geometry_metrics.json';rgb=base/'renders'/(camera['stem']+'.png');dp=cache/'depth'/(camera['stem']+'.npy')
    metric=read(mp) if mp.exists() else None;geo=read(gp) if gp.exists() else None
    if metric and metric.get('scoring_protocol_version')!=2:metric=None
    if geo and geo.get('scoring_protocol_version')!=2:geo=None
    mesh_marker=BASE/method/scene/'mesh_complete.json'
    absent='mesh complete; evaluation pending' if method!='paintmesh' and mesh_marker.exists() else 'no completed mesh/evaluation'
    status='complete' if metric and rgb.exists() else ('RGB ready; scoring pending' if rgb.exists() else ('RGB evaluation pending' if dp.exists() else absent))
    return dict(label=label,method=method,state=state,rgb=rgb,depth=dp,metric=metric,geometry=geo,status=status)

def rgb_image(record,size,roi=None):
    if not record['rgb'].exists():return placeholder(size,record['status'].replace('; ','\n'))
    pic=Image.open(record['rgb']).convert('RGB')
    if roi:pic=pic.crop(roi)
    return contained(pic,size)

def view_metric(record,camera):
    if not record['metric']:return None
    return next((r for r in record['metric']['per_view'] if r['name']==camera['name']),None)

def metric_lines(record,camera):
    v=view_metric(record,camera);g=record['geometry']
    first=f'P {v["PSNR"]:.2f}  S {v["SSIM"]:.3f}  L {v["LPIPS"]:.3f}' if v and v.get('PSNR') is not None else ('GT RGB' if record['method'] is None else record['status'])
    second=f'scene CD {g["CD_L1"]:.4g}  F1 {g["F1"]:.4f}' if g and g.get('CD_L1') is not None else ''
    return first,second

def depth_color(depth,near,far,shade=None):
    valid=np.isfinite(depth)&(depth>0)
    t=np.clip((depth-near)/max(far-near,1e-6),0,1)
    # Shared scale, independent of the method and reconstruction error.
    anchors=np.array([[.12,.32,.62],[.16,.68,.74],[.88,.77,.35]])
    rgb=np.stack([np.interp(t,[0,.5,1],anchors[:,c]) for c in range(3)],-1)
    if shade is not None:rgb*=shade[:,:,None]
    rgb[~valid]=[.96,.965,.975]
    return Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype('uint8'))

def reference_depth(root,scene,camera,state):
    with np.load(root/'common'/scene/'reference_points.npz') as a:points=a[state]
    T=np.asarray(camera['w2c']);K=np.asarray(camera['K']);q=points@T[:3,:3].T+T[:3,3]
    pp=q@K.T;valid=q[:,2]>0
    uv=np.floor(pp[:,:2]/np.maximum(pp[:,2:],1e-12)).astype('int64')
    w,h=camera['width'],camera['height']
    valid&=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    z=q[valid,2];pix=uv[valid,1]*w+uv[valid,0]
    image=np.full(w*h,np.inf,dtype='float32');np.minimum.at(image,pix,z)
    from scipy.ndimage import minimum_filter
    image=minimum_filter(image.reshape(h,w),size=3)
    return np.where(np.isfinite(image),image,0).astype('float32'),np.quantile(z,[.02,.98])

def extra_depth(root,out,scene,record,camera):
    """One-view CPU render for a mesh finished after the evaluation snapshot.

    This does not change the evaluator's cohort, masks or reported scores.
    """
    method=record['method']
    if not method or method in ['paintmesh','paintmesh_stage1']:return None
    marker=BASE/method/scene/'mesh_complete.json'
    if not marker.exists():return None
    mesh=read(marker)
    if not mesh.get('complete'):return None
    dest=out/'assets'/scene/'extra';path=dest/(method+'_depth.npy');meta=dest/(method+'_source.json')
    sign=digest(dict(mesh=identity(mesh['path']),camera=camera))
    if path.exists() and meta.exists() and read(meta).get('signature')==sign:return path
    available=next((int(line.split()[1])/1024**2 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')),0)
    if available<30:print('EXTRA_DEPTH_WAIT_RAM',method,scene,available,flush=True);return None
    print('EXTRA_COMPLETED_MESH_DEPTH',method,scene,flush=True)
    from mesh_benchmark.core import MeshScene,rays
    scene_mesh=MeshScene([dict(path=mesh['path'])]);colored=all(x['has_native_color'] for x in scene_mesh.counts)
    depth,rgb=scene_mesh.cast(rays(camera),color=colored)
    dest.mkdir(parents=True,exist_ok=True);np.save(path,depth.reshape(camera['height'],camera['width']))
    if rgb is not None:save(Image.fromarray(np.rint(rgb.reshape(camera['height'],camera['width'],3).clip(0,1)*255).astype('uint8')),dest/(method+'_native.png'))
    write(meta,dict(signature=sign,source_mesh=mesh,depth=str(path),camera=camera['name'],native_colors=colored,
          description='One-view CPU triangle render, outside the existing evaluation cohort; no synthetic metrics assigned'))
    del scene_mesh
    import gc;gc.collect()
    return path

def geometry_image(root,out,scene,record,camera,scale,reference_state='recomposition'):
    method=record['method'];state=record['state'] if method else reference_state
    path=out/'assets'/scene/state/((method or 'reference')+'_geometry.png')
    source=record.get('depth') if method else root/'common'/scene/'reference_points.npz'
    if method and (source is None or not source.exists()):
        source=extra_depth(root,out,scene,record,camera)
        if source is not None:record['depth']=source
    if source is None or not source.exists():return None
    sign=digest(dict(source=identity(source),camera=camera,range=list(scale),kind='mesh-depth-normal-v1'))
    marker=path.with_suffix('.json')
    if path.exists() and marker.exists() and read(marker).get('signature')==sign:return Image.open(path).convert('RGB')
    if method is None:
        depth,_=reference_depth(root,scene,camera,state);image=depth_color(depth,*scale)
    else:
        d=np.load(source);w,h=camera['width'],camera['height'];yy,xx=np.mgrid[:h,:w]
        q=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(camera['K']).T
        q=q*d[:,:,None];dx=np.gradient(q,axis=1);dy=np.gradient(q,axis=0)
        n=np.cross(dx,dy);n/=np.maximum(np.linalg.norm(n,axis=2,keepdims=True),1e-10)
        n*=np.where(n[:,:,2:]>0,-1,1)
        light=np.array([-.3,-.4,-.86]);light/=np.linalg.norm(light)
        shade=.48+.52*np.maximum(0,n@light)
        image=depth_color(d,*scale,shade=shade)
    save(image,path);write(marker,dict(signature=sign,near=float(scale[0]),far=float(scale[1]),
                   source=str(source),description='Mesh depth with approximate depth-gradient lighting; reference is a sparse point-cloud projection'))
    return image

def scoring_mask(root,scene,state,camera):
    native=root/'common'/scene/(state+'_native_support')/(camera['stem']+'.png')
    p=native if native.exists() else root/'common'/scene/(state+'_support')/(camera['stem']+'.png')
    return np.asarray(Image.open(p))>127 if p.exists() else None

def masked_image(image,mask):
    if mask is None:return image
    a=np.asarray(image).copy();a[~mask]=(a[~mask]*.12+np.array([208,211,216])*.88).astype('uint8')
    return Image.fromarray(a)

def draw_scene(root,out,scene,camera,roi,records,scale,state=None,width=None):
    width=max(camera['width'],width or 0)
    columns=records if state is None else [sources(root,scene,'GT',None,None,camera)]+[
             sources(root,scene,('PaintMesh: background' if state=='background' else 'PaintMesh: composed') if m=='paintmesh' else name,m,state,camera)
             for name,m in PAIRED_LABELS]
    n=len(columns);full_h=round(width*2/3);zoom_h=width//2;top=148
    heights=[full_h+95,zoom_h+80,full_h+90]
    canvas=Image.new('RGB',(n*width,top+sum(heights)+45),BG);draw=ImageDraw.Draw(canvas)
    title=f'{scene} | all methods + both PaintMesh modes' if state is None else f'{scene} | {state} | common covered pixels only'
    text(draw,(18,15),title,size=32,bold=True)
    text(draw,(18,58),f'Camera: {camera["name"]}  |  Fixed view: largest target area in GT; no metric-based selection',size=21)
    text(draw,(18,91),'RGB scores: selected view, on each mode\'s common support. Geometry CD/F1: scene-level.',size=20)
    text(draw,(18,118),'P=PSNR (dB), S=SSIM, L=LPIPS. F1 threshold=1% camera radius. BG and composed score domains differ.',size=18,color=(85,92,104))
    for i,record in enumerate(columns):
        x=i*width;accent=GREEN if record['method']=='paintmesh' else INK
        label=record['label'] if record['method'] else ('GT RGB / sparse reference' if state is None else 'GT on same scoring pixels')
        draw.rectangle((x,top,x+width,top+35),fill=accent)
        text(draw,(x+12,top+5),label,size=23,color='white',bold=True)
        original=Image.open(record['rgb']).convert('RGB') if record['rgb'].exists() else None
        pic=original
        if pic is not None and state is not None:pic=masked_image(pic,scoring_mask(root,scene,state,camera))
        y=top+35
        frame=contained(pic,(width,full_h)) if pic else placeholder((width,full_h),record['status'].replace('; ','\n'))
        if state is None and original is not None:
            # Same ROI box on every method, in precisely the same pixel coordinates.
            dr=ImageDraw.Draw(frame);s=min(width/camera['width'],full_h/camera['height'])
            ox=(width-camera['width']*s)/2;oy=(full_h-camera['height']*s)/2
            dr.rectangle(tuple([roi[0]*s+ox,roi[1]*s+oy,roi[2]*s+ox,roi[3]*s+oy]),outline=PINK,width=3)
        canvas.paste(frame,(x,y))
        lines=metric_lines(record,camera)
        text(draw,(x+9,y+full_h+8),lines[0],size=19,color=accent)
        text(draw,(x+9,y+full_h+34),lines[1],size=18,color=(86,94,105))
        y=top+heights[0]+20
        text(draw,(x+10,y),'Target-region zoom (same crop)',size=20,bold=True);y+=30
        zoom=contained(pic.crop(roi),(width,zoom_h)) if pic else placeholder((width,zoom_h),'RGB not available yet')
        canvas.paste(zoom,(x,y))
        y=top+heights[0]+heights[1]+20
        text(draw,(x+10,y),'Sparse reference points' if record['method'] is None else 'Mesh surface: depth + lighting',size=20,bold=True);y+=32
        geom=geometry_image(root,out,scene,record,camera,scale,reference_state=state or 'recomposition')
        if geom is not None and state is not None:
            support=root/'common'/scene/(state+'_support')/(camera['stem']+'.png')
            geom=masked_image(geom,np.asarray(Image.open(support))>127 if support.exists() else None)
        canvas.paste(contained(geom,(width,full_h)) if geom else placeholder((width,full_h),'No completed mesh/depth'),(x,y))
        if record['method']:
            geo=record['geometry'] or {};coverage=view_metric(record,camera)
            caption=f'CD {geo["CD_L1"]:.5g} | F1 {geo["F1"]:.4f}' if geo.get('CD_L1') is not None else 'Geometry score pending'
            if coverage:caption+=f' | own pixels {coverage["mesh_coverage"]:.1%}'
            text(draw,(x+9,y+full_h+9),caption,size=18)
        else:text(draw,(x+9,y+full_h+9),'COLMAP sparse cloud; not dense scan GT',size=18)
        draw.line([(x,top),(x,canvas.height-10)],fill=(220,224,230),width=2)
    text(draw,(16,canvas.height-32),'Geometry colors use one fixed depth range for this camera. Gray in score-matched figures = not evaluated.',size=18)
    return canvas

def overview(root,out,scenes,selections,kind,width=None):
    width=max(max(selections[s]['camera']['width'] for s in scenes),width or 0)
    paired=kind in ['background','recomposition'];n=len(METHODS)+1 if paired else len(COLS)
    fh=round(width*2/3);row_h=fh+91;left=160;top=122
    canvas=Image.new('RGB',(left+n*width,top+len(scenes)*row_h+55),BG);draw=ImageDraw.Draw(canvas)
    title={'rgb':'All scenes | mesh RGB | four baselines + stage 1 + both PaintMesh modes',
           'geometry':'All scenes | mesh surfaces | shared depth color scale per scene',
           'background':'All scenes | background mesh | common covered pixels, target mask excluded',
           'recomposition':'All scenes | composed mesh | common covered pixels'}[kind]
    text(draw,(16,12),title,size=29,bold=True)
    text(draw,(16,55),'Same camera within each scene. P/S/L = selected-view RGB metrics; CD/F1 = scene geometry.',size=21)
    for row,scene in enumerate(scenes):
        camera=selections[scene]['camera'];roi=selections[scene]['roi'];scale=selections[scene]['depth_range']
        cols=COLS if not paired else [('GT',None,None)]+[(label,m,kind) for label,m in PAIRED_LABELS]
        y=top+row*row_h
        pretty=scene.replace('_',' ').split(' ')
        for k,line in enumerate(pretty):text(draw,(10,y+fh//2+k*27),line,size=21,bold=True)
        for i,(label,method,state) in enumerate(cols):
            x=left+i*width
            if row==0:
                draw.rectangle((x,86,x+width,119),fill=GREEN if method=='paintmesh' else INK)
                text(draw,(x+10,91),label,size=21,color='white',bold=True)
            rec=sources(root,scene,label,method,state,camera)
            if kind=='geometry':pic=geometry_image(root,out,scene,rec,camera,scale)
            else:
                pic=Image.open(rec['rgb']).convert('RGB') if rec['rgb'].exists() else None
                if pic is not None and paired:pic=masked_image(pic,scoring_mask(root,scene,kind,camera))
            image=contained(pic,(width,fh)) if pic is not None else placeholder((width,fh),rec['status'].replace('; ','\n'))
            canvas.paste(image,(x,y))
            a,b=metric_lines(rec,camera)
            if kind=='geometry':a=b or ('Sparse COLMAP reference' if method is None else 'Geometry evaluation pending');b=''
            text(draw,(x+8,y+fh+6),a,size=17,color=GREEN if method=='paintmesh' else INK)
            if b:text(draw,(x+8,y+fh+32),b,size=16,color=(83,93,107))
            if rec['metric'] and paired:
                v=view_metric(rec,camera)
                if v:text(draw,(x+8,y+fh+57),f'scored pixels {v.get("common_coverage",0):.1%} | own {v["mesh_coverage"]:.1%}',size=16,color=(83,93,107))
        draw.line([(0,y+row_h-3),(canvas.width,y+row_h-3)],fill=(210,215,223),width=2)
    native=read(root/'protocol.json').get('native_mesh_rgb',{}).get('enabled')
    caption='RGB uses native colored triangle meshes and each project mesh backend; no appearance fitting or Gaussian image substitutions.' if native else 'Unfinished sources are labeled. RGB is actual mesh rendering with the shared appearance field; no Gaussian render substitutions.'
    text(draw,(16,canvas.height-36),caption,size=20)
    return canvas

def input_signature(root,scene,camera):
    paths=[root/'common'/scene/'complete.json',root/'common'/scene/'reference_points.npz',Path(__file__)]
    for state in ['background','recomposition']:
        paths.append(root/'common'/scene/(state+'_support')/(camera['stem']+'.png'))
        paths.append(root/'common'/scene/(state+'_native_support')/(camera['stem']+'.png'))
        for method in METHODS:
            paths += [root/state/method/scene/'metrics.json',root/state/method/scene/'renders'/(camera['stem']+'.png'),
                      root/'cache'/method/scene/state/'depth'/(camera['stem']+'.npy'),root/'cache'/method/scene/state/'geometry_metrics.json']
            if method not in ['paintmesh','paintmesh_stage1']:paths.append(BASE/method/scene/'mesh_complete.json')
    return digest([identity(p) if p.exists() else dict(path=str(p),missing=True) for p in paths])

def build(root,out,scenes):
    out.mkdir(parents=True,exist_ok=True)
    selections={};changed=False;inventory=[];ready=0
    for scene in scenes:
        if not (root/'common'/scene/'complete.json').exists():continue
        camera,roi,area=choose_view(root,scene)
        _,scale=reference_depth(root,scene,camera,'recomposition')
        selections[scene]=dict(camera=camera,roi=roi,target_fraction=area,depth_range=scale.tolist())
        sign=input_signature(root,scene,camera);marker=out/'scenes'/scene/'figure_manifest.json'
        records=[sources(root,scene,*column,camera) for column in COLS]
        record_info=[]
        for record in records:
            m=view_metric(record,camera);geo=record['geometry']
            record_info.append(dict(label=record['label'],method=record['method'],state=record['state'],status=record['status'],
                    rgb_source=str(record['rgb']),depth_source=str(record.get('depth','')),
                    view_metrics=m,scene_geometry=geo))
            ready+=record['status']=='complete'
        inventory.append(dict(scene=scene,camera=camera['name'],records=record_info))
        if marker.exists() and read(marker).get('signature')==sign:
            continue
        print('DRAW',scene,camera['name'],flush=True)
        dest=out/'scenes'/scene
        image=draw_scene(root,out,scene,camera,roi,records,scale)
        save(image,dest/'all_methods',formats=('png','jpg','pdf'))
        for state in ['background','recomposition']:
            image=draw_scene(root,out,scene,camera,roi,records,scale,state=state)
            save(image,dest/state,formats=('png','pdf'))
        for meta,record in zip(record_info,records):meta['depth_source']=str(record.get('depth',''))
        write(marker,dict(signature=sign,selection=selections[scene],records=record_info))
        changed=True
    selected=[s for s in scenes if s in selections]
    write(figure_bucket(out,'metadata')/'selected_views.json',dict(rule='Among held-out cameras with scoring support, maximize GT target-mask fraction; independent of method metric scores',scenes=selections))
    if changed or not (figure_bucket(out,'overview')/'all_scenes_rgb.png').exists():
        for kind in ['rgb','geometry','background','recomposition']:
            print('OVERVIEW',kind,flush=True)
            image=overview(root,out,selected,selections,kind)
            save(image,figure_bucket(out,'overview')/('all_scenes_'+kind),formats=('png','jpg','pdf'))
        for name in ['all_methods','background','recomposition']:
            images=[Image.open(out/'scenes'/s/(name+'.png')).convert('RGB') for s in selected]
            if images:
                temp=figure_bucket(out,'overview')/('.'+name+'_all_scenes.pdf.tmp')
                images[0].save(temp,format='PDF',save_all=True,append_images=images[1:],resolution=180.)
                temp.replace(figure_bucket(out,'overview')/(name+'_all_scenes.pdf'))
            for image in images:image.close()
    protocol=read(root/'protocol.json')
    write(figure_bucket(out,'metadata')/'manifest.json',dict(updated=time.time(),scene_count=len(selected),completed_display_cells=ready,
              columns=[x[0] for x in COLS],source_protocol=str(root/'protocol.json'),source_cohorts=protocol.get('cohorts'),scenes=inventory))
    write_index(out,inventory)
    if protocol.get('native_mesh_rgb',{}).get('enabled'):
        path=out/'README.md';contents=path.read_text()
        contents=contents.replace('RGB 采用本轮 mesh 评估实际输出的统一颜色场渲染；不混入高斯图像或用原生纹理冒充统一外观。',
            'RGB 采用各项目网格后端与原生顶点色，不拟合统一颜色场，不使用高斯图片替代。')
        path.write_text(contents)
    if protocol.get('native_support_validation',{}).get('enabled'):
        path=out/'README.md';contents=path.read_text()
        contents+='\nCoMe / GOF 的当前网格经过同一源高斯有效支撑校验；保留原坐标和原生颜色，剔除无支撑长插值面。不是官方未处理原样输出，详见 [修复协议](../docs/OPACITY_MESH_REPAIR.md)。\n'
        path.write_text(contents)
    return changed

def write_index(out,inventory):
    lines=['# 全方法、全场景 mesh 对比图','',
           '固定八列：GT、2DGS、CoMe、GOF、Gaussian Wrapping、PaintMesh 第一阶段原始mesh、PaintMesh 纯背景、PaintMesh + MV-SAM3D。第一阶段保留原物体，两份同域报告分别按背景/组合mask评分。','',
           '## 全场景总览','',
           '- [RGB 总图](all_scenes_rgb.png) · [PDF](all_scenes_rgb.pdf)',
           '- [表面几何总图](all_scenes_geometry.png) · [PDF](all_scenes_geometry.pdf)',
           '- [纯背景：相同评分区域](all_scenes_background.png)',
           '- [组合 mesh：相同评分区域](all_scenes_recomposition.png)',
           '- [全部场景详细图，多页 PDF](all_methods_all_scenes.pdf)',
           '- [纯背景详细图，多页 PDF](background_all_scenes.pdf)',
           '- [组合 mesh 详细图，多页 PDF](recomposition_all_scenes.pdf)','',
           '## 每个场景的完整画面、目标区域放大和表面图','',
           '| 场景 | 相机 | 八列详细图 | 纯背景同域对比 | 组合 mesh 同域对比 |','|---|---|---|---|---|']
    for row in inventory:
        s=row['scene'];lines.append(f'| {s} | {row["camera"]} | [PNG](scenes/{s}/all_methods.png) · [PDF](scenes/{s}/all_methods.pdf) | [PNG](scenes/{s}/background.png) | [PNG](scenes/{s}/recomposition.png) |')
    lines+=['','## 读图说明','',
       '- 每场景各列严格使用同一相机。选择有评分区域的留出相机中 GT 目标 mask 占比最大的一个；不按 PSNR 或其他方法分数选择。selected_views.json 记录原始文件名、相机、裁剪框、选择规则。',
       '- RGB 采用本轮 mesh 评估实际输出的统一颜色场渲染；不混入高斯图像或用原生纹理冒充统一外观。尚未完成外观的格子明确标为评估中；未完成 mesh 的格子保留缺失状态。',
       '- 八列总览展示完整原始画面，便于观察缺失几何；粉色框为同一个放大区域。两份同域图则把不参与评分的像素显示成灰色。',
       '- 纯背景与组合模式的评分范围不同。八列图中纯背景列的 P/S/L 使用背景支持域，其余重建方法与组合列使用组合支持域；不能跨这两种范围直接排名。公平数值对比请看单独的纯背景/组合图和原评估报告。',
       '- 图上 P/S/L 是所选相机的 PSNR(dB)/SSIM/LPIPS；CD 和 F1 是对应场景的几何汇总，不是该相机独立几何分数。F1 采用统一 1% 相机半径阈值。',
       '- 几何图来自真实 mesh 深度，并用深度梯度近似照明增强表面形状；不是额外生成的几何。每场景所有方法使用同一深度色标。参考列显示真实 COLMAP 稀疏点云投影（点大小3像素），没有伪造稠密 GT mesh。',
       '- 原始有物体留出 RGB 为本图 GT；不把无物体 test* 图像当组合 mesh 真值。已有 PaintMesh 结果可能使用过这些相机，当前为已有结果审计。',
       '- manifest.json 记录各格子的源图片、深度文件、评分状态和指标。后续评估完成后会自动重画受影响场景；这项绘图任务不修改原模型或评估分数。',
       '', '[纯背景评估报告](../background/report.md) · [组合评估报告](../recomposition/report.md) · [评估协议](../PROTOCOL.zh-CN.md)']
    text='\n'.join(lines)+'\n'
    if configuration(out.parent):
        for name in ['all_scenes_rgb','all_scenes_geometry','all_scenes_background','all_scenes_recomposition','all_methods_all_scenes','background_all_scenes','recomposition_all_scenes']:
            text=text.replace(']('+name,'](overview/'+name)
        text=text.replace('selected_views.json','metadata/selected_views.json').replace('manifest.json','metadata/manifest.json')
        text=text.replace('../background/report.md','../reports/background.md').replace('../recomposition/report.md','../reports/recomposition.md').replace('../PROTOCOL.zh-CN.md','../docs/PROTOCOL.zh-CN.md')
    text+='\n- [各方法独立子图](subplots/README.md)\n'
    (out/'README.md').write_text(text)

from mesh_benchmark.layout import work_root,configure_sources,figure_bucket,configuration

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=BASE/'mesh_comparison')
    parser.add_argument('--output',type=Path);parser.add_argument('--scenes',default=','.join(SCENES))
    parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=int,default=45)
    args=parser.parse_args();root=work_root(args.root);configure_sources(root);out=(args.output or root/'figures_all_methods').resolve()
    out.mkdir(parents=True,exist_ok=True);figure_bucket(out,'.runtime').mkdir(exist_ok=True);lock=open(figure_bucket(out,'.runtime')/'figure.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (figure_bucket(out,'.runtime')/'watch.pid').write_text(str(os.getpid())+'\n')
    while True:
        before=time.time()
        try:
            changed=build(root,out,args.scenes.split(','));print('FIGURES_UPDATED' if changed else 'FIGURES_UNCHANGED',time.strftime('%F %T'),flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if not args.watch:raise
        if not args.watch:break
        status=read(root/'status.json')
        pid=status.get('pid');alive=False
        if pid:
            try:alive=b'evaluate_recomposition_meshes.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:pass
        if not alive:
            # A mesh can finish while the original metric snapshot is running.
            # Extend that snapshot once, after it releases its campaign lock.
            refreshed=out/'completed_mesh_refresh.json'
            from mesh_benchmark.core import specs_for
            current={s:[m for m in METHODS if specs_for(s,m,'background') is not None] for s in SCENES}
            previous=read(root/'protocol.json').get('cohorts',{})
            if not refreshed.exists() and current!=previous:
                import subprocess
                env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
                with open(root/'campaign.log','a') as log:
                    process=subprocess.Popen([sys.executable,str(HERE/'evaluate_recomposition_meshes.py'),'--output',str(root),'--refresh-completed-meshes'],
                         stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                write(refreshed,dict(pid=process.pid,started=time.time(),previous=previous,current=current))
                print('REFRESH_ADDITIONAL_COMPLETED_MESHES',process.pid,flush=True)
                time.sleep(10);continue
            print('Evaluation campaign exited; final figures written',flush=True);break
        time.sleep(max(1,args.interval-(time.time()-before)))

if __name__=='__main__':main()
