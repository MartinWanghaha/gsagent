#!/usr/bin/env python3
"""Run a bounded, three-arm bag ROI experiment without changing the main flow."""
from pathlib import Path
import argparse
import gc
import os
import sys
import time
import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,read,write,identity,digest,MeshScene,rays,surface_metrics
from mesh_benchmark.layout import work_root,configure_sources
from mesh_benchmark.fine_tsdf import VOXEL,TRUNCATION,BLOCK,integrate_partition
from mesh_benchmark.surface_fusion_probe import (continuous_cells,world_grid,triangles_in_roi,
    expanded_blocks,integrate_sampled,crop_mesh,topology)

PUBLIC=BASE/'mesh_comparison_whole'
ROOT=work_root(PUBLIC)
LABELS={'A':'Current allocation + nearest depth','B':'Surface footprint + nearest depth','C':'Surface footprint + continuous inverse depth'}


def save_mesh(path,vertices,faces,colors):
    """Same RGB8 rounding as the published native TSDF buffers."""
    from plyfile import PlyData,PlyElement
    vertex=np.empty(len(vertices),dtype=[(n,'<f4') for n in ['x','y','z']]+[(n,'u1') for n in ['red','green','blue']])
    rgb=np.rint(np.clip(colors,0,1)*255).astype('uint8')
    for i,key in enumerate(['x','y','z']):vertex[key]=vertices[:,i]
    for i,key in enumerate(['red','green','blue']):vertex[key]=rgb[:,i]
    face=np.empty(len(faces),dtype=[('vertex_indices','<i4',(3,))]);face['vertex_indices']=faces
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    PlyData([PlyElement.describe(vertex,'vertex'),PlyElement.describe(face,'face')],text=False).write(str(path))


def select_roi(frame,side):
    from scipy.ndimage import label,distance_transform_edt
    d=frame['depth'];mask=continuous_cells(d)&(d[:-1,:-1]>15)&(d[:-1,:-1]<25)
    components,count=label(mask)
    if not count:raise ValueError('First fixed training view has no continuous far surface')
    sizes=np.bincount(components.ravel());sizes[0]=0;component=components==sizes.argmax()
    yy,xx=np.unravel_index(np.argmax(distance_transform_edt(component)),component.shape)
    center=world_grid(frame)[yy,xx]
    return np.array([center-side/2,center+side/2]),dict(training_pixel=[int(xx),int(yy)],
        anchor_depth=float(d[yy,xx]),center=center.tolist(),side_colmap_units=side,
        rule='Largest continuous 15<depth<25 component in the first fixed training camera; center maximizes interior pixel distance. No held-out RGB or metric selection.')


def device_for(blocks):
    import torch
    # The old Open3D allocator/MC and the torch prototype are accounted for
    # separately. Never preempt a running training or production fusion job.
    estimate=len(blocks)*BLOCK**3*100+3*1024**3
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        if free>max(8*1024**3,estimate+4*1024**3):
            torch.cuda.set_per_process_memory_fraction(3*1024**3/total)
            return 'CUDA:0'
    return 'CPU:0'


def render(path,cameras,roi):
    model=MeshScene([dict(path=str(path))]);depths=[];images=[]
    for c in cameras:
        rr=rays(c);z,color=model.cast(rr,color=True)
        p=rr[:,:3]+rr[:,3:]*z[:,None]
        inside=((p>=roi[0])&(p<=roi[1])).all(1)&(z>0)
        z[~inside]=0;color[~inside]=0
        depths.append(z.reshape(c['height'],c['width']))
        images.append(np.rint(np.clip(color.reshape(c['height'],c['width'],3),0,1)*255).astype('uint8'))
    del model;gc.collect();return depths,images


def save_figures(out,cameras,images,reference,eligible,summary):
    from plot_mesh_method_comparison import font
    for i,c in enumerate(cameras):
        valid=eligible[i]
        if not valid.any():continue
        yy,xx=np.where(reference[i]>0)
        crop=[max(0,int(xx.min())-5),max(0,int(yy.min())-5),min(c['width'],int(xx.max())+6),min(c['height'],int(yy.max())+6)]
        width=c['width'];height=c['height'];zh=max(180,crop[3]-crop[1])
        sheet=Image.new('RGB',(4*width,height+zh+125),(244,246,249));draw=ImageDraw.Draw(sheet)
        gt=Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_gt.png')).convert('RGB')
        panels=[('GT',np.asarray(gt)),*[(k,images[k][i]) for k in LABELS]]
        for j,(key,img) in enumerate(panels):
            x=j*width;im=Image.fromarray(img)
            draw.text((x+8,8),key if key=='GT' else LABELS[key],font=font(16),fill='black')
            sheet.paste(im,(x,40))
            # Nearest-neighbor magnification is display-only. Metrics and raw
            # per-method render PNGs retain every original image pixel.
            zoom=im.crop(crop);scale=min(width/zoom.width,zh/zoom.height)
            zoom=zoom.resize((max(1,round(zoom.width*scale)),max(1,round(zoom.height*scale))),Image.Resampling.NEAREST)
            sheet.paste(zoom,(x,40+height))
            if key!='GT':
                row=summary[key]['per_view'][i]
                draw.text((x+8,45+height+zh),f'ROI coverage {row["coverage"]:.1%} | shared pixels {row["common_pixels"]}',font=font(16),fill='black')
        path=out/'figures'/(c['stem']+'.png');path.parent.mkdir(exist_ok=True);sheet.save(path)
    chosen=max(range(len(cameras)),key=lambda i:int(eligible[i].sum()))
    c=cameras[chosen];source=out/'figures'/(c['stem']+'.png')
    if source.exists():
        import shutil
        shutil.copy2(source,out/'comparison.png')
    write(out/'figures/selection.json',dict(camera=c['name'],rule='Maximum fixed source-range ROI pixels, independent of method scores'))


def evaluate(out,roi,anchor,paths):
    import torch,lpips
    from mesh_benchmark.score import image_metrics
    info=read(ROOT/'common/bag/complete.json');cs=[c for c in info['cameras'] if c['part']=='held_out']
    # A triangulated source range patch defines only the ROI coverage domain;
    # it is not geometric ground truth. CD/F1 use independent SfM observations.
    tri=triangles_in_roi(anchor,roi)
    vertices=tri.reshape(-1,3);faces=np.arange(len(vertices)).reshape(-1,3)
    color=np.full_like(vertices,.5);vertices,faces,color=crop_mesh(vertices,faces,color,roi)
    reference_path=out/'source_range_domain.ply';save_mesh(reference_path,vertices,faces,color)
    ref,_=render(reference_path,cs,roi);eligible=[]
    for i,c in enumerate(cs):
        old=np.load(ROOT/'cache/paintmesh_stage1/bag/background/depth'/(c['stem']+'.npy'))
        foreground=np.asarray(Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_foreground.png')))>127
        eligible.append((ref[i]>0)&~foreground&((old==0)|(old>=ref[i]-2*TRUNCATION)))
    depth={};images={}
    for key,path in paths.items():
        depth[key],images[key]=render(path,cs,roi)
        for i,c in enumerate(cs):
            folder=out/key/'renders';folder.mkdir(exist_ok=True)
            Image.fromarray(images[key][i]).save(folder/(c['stem']+'.png'))
            np.save(folder/(c['stem']+'_depth.npy'),depth[key][i])
    shared=[eligible[i]&np.logical_and.reduce([depth[k][i]>0 for k in paths]) for i in range(len(cs))]
    criterion=lpips.LPIPS(net='vgg',version='0.1',spatial=True).to('cpu').eval()
    for p in criterion.parameters():p.requires_grad_(False)
    results={}
    for key in paths:
        rows=[]
        for i,c in enumerate(cs):
            gt=np.asarray(Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_gt.png')).convert('RGB'))
            own=eligible[i]&(depth[key][i]>0)
            scores=image_metrics(images[key][i],gt,shared[i],criterion,'cpu')
            rows.append(dict(camera=c['name'],eligible_pixels=int(eligible[i].sum()),own_pixels=int(own.sum()),
                common_pixels=int(shared[i].sum()),coverage=float(own.sum()/max(eligible[i].sum(),1)),**scores))
        total=sum(r['eligible_pixels'] for r in rows)
        scored=[r for r in rows if r['common_pixels']]
        results[key]=dict(label=LABELS[key],coverage=float(sum(r['own_pixels'] for r in rows)/max(total,1)),
            eligible_pixels=total,common_pixels=sum(r['common_pixels'] for r in rows),
            mean={k:float(np.mean([r[k] for r in scored])) if scored else None for k in ['PSNR','SSIM','LPIPS']},per_view=rows)
        print('PROBE_METRICS',key,results[key]['coverage'],results[key]['mean'],flush=True)
    # Confirm that the cropped A control reproduces the published mesh where
    # that published surface lies inside this ROI, rather than comparing a
    # different extraction by accident.
    errors=[]
    for i,c in enumerate(cs):
        old=np.load(ROOT/'cache/paintmesh_stage1/bag/background/depth'/(c['stem']+'.npy'))
        rr=rays(c);p=rr[:,:3]+rr[:,3:]*old.reshape(-1,1)
        interior=((p>roi[0]+3*VOXEL)&(p<roi[1]-3*VOXEL)).all(1).reshape(old.shape)
        keep=interior&(depth['A'][i]>0)&(old>0)
        if keep.any():errors.extend(np.abs(depth['A'][i][keep]-old[keep]).tolist())
    check=dict(pixels=len(errors),p95=float(np.quantile(errors,.95)) if errors else None,
               maximum=float(np.max(errors)) if errors else None,tolerance=2*VOXEL)
    check['passed']=bool(errors) and check['p95']<2*VOXEL
    if not check['passed']:raise ValueError(f'Control does not reproduce the published mesh: {check}')
    geometry(out,roi,cs,depth,eligible,results)
    save_figures(out,cs,images,ref,eligible,results)
    write(out/'results.json',dict(complete=True,updated=time.time(),methods=results,control_reproduction=check,
        metric_domain='Same source-range visible ROI and common covered pixels; foreground excluded',
        no_image_downsampling=True,view_count=len(cs),geometry_reference='Observed COLMAP points, not independent scan GT'))
    return results,check


def geometry(out,roi,cameras,depth,eligible,results):
    from mesh_benchmark.overlap import observations,select_anchors,prediction_points
    with np.load(ROOT/'common/bag/reference_points.npz') as a:reference=a['background']
    reference=reference[((reference>=roi[0])&(reference<=roi[1])).all(1)]
    obs=observations('bag',cameras,reference)
    shared=[eligible[i]&np.logical_and.reduce([depth[k][i]>0 for k in depth]) for i in range(len(cameras))]
    chosen=select_anchors(obs[:,0],obs[:,1],obs[:,2],shared,np.ones(len(reference),dtype=bool),42,max_points=10**9) if len(obs) else []
    for key in depth:
        result=None
        if len(chosen):
            prediction=prediction_points(cameras,depth[key],chosen,obs)
            result=surface_metrics(prediction,reference[obs[chosen,0]],read(ROOT/'common/bag/complete.json')['extent'])
        results[key].update(geometry=result,roi_reference_points=len(reference),common_reference_observations=len(chosen))


def run(out,side=1.2):
    import torch,open3d as o3d
    torch.set_num_threads(4);configure_sources(ROOT)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(out/'experiment.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache=ROOT/'fine_whole/stage1/bag';source=read(cache/'frames/complete.json')
    frames=[]
    for row in source['frames']:
        if identity(row['path'])!=row['identity']:raise ValueError('Cached RGB-D source changed')
        with np.load(row['path']) as data:frames.append({k:data[k].copy() for k in ['depth','rgb','K','w2c']})
    roi,selection=select_roi(frames[0],side)
    plan=dict(scene='bag',source_kind='stage1',voxel_size=VOXEL,sdf_trunc=TRUNCATION,
        implementation=dict(driver=identity(Path(__file__)),fusion=identity(HERE/'mesh_benchmark/surface_fusion_probe.py')),
        source_receipt=identity(cache/'frames/complete.json'),source_frames=source['frames'],
        roi=roi.tolist(),selection=selection,methods=LABELS,views=len(frames),
        interpolation_gates=dict(relative_depth_span=.10,relative_inverse_depth_cross=.005),
        train_rgb_fitting=False,gaussian_retraining=False,mesh_decimation=False,image_downsampling=False,
        production_pipeline_modified=False,scope='Fixed far ROI pilot, not a full-scene replacement')
    path=out/'plan.json'
    if path.exists() and read(path)!=plan:raise ValueError('Pilot inputs/ROI differ from this output')
    write(path,plan)
    (out/'PLAN.md').write_text('# bag 远景表面连续性独立实验\n\n'
        '只在固定远景区域对照 A/B/C，现有项目流程、检查点和主结果不改变。\n\n'
        f'- 全部 {len(frames)} 个既有训练 RGB-D 视角，原生 1/4 分辨率，体素 0.002、带宽 0.008、原生 RGB、相同单位融合权重。\n'
        '- A：现有分配和最近邻深度；B：连续表面完整覆盖分配、最近邻深度；C：B 加边缘感知逆深度插值。深度断层和无效邻域回到原最近邻采样。\n'
        '- ROI 仅由首个固定训练深度的连续远景区域确定，不按留出视角指标选择。所有既定留出视角参与评价。\n'
        '- 孔洞率对固定源深度曲面域计算，空像素计入缺失；RGB 指标在三组共同有效像素上计算。源深度曲面只界定范围，不作几何真值。CD/F1 使用 ROI 内 COLMAP 的实际可见观测点，并明确点数。\n'
        '- 不平面拟合、不全局平滑、不减面、不缩图，不覆盖主对比图。\n')
    length=VOXEL*BLOCK;padding=2*length
    whole=np.load(cache/'blocks.npy',mmap_mode='r')
    base=np.asarray(whole[((whole+1)*length>=roi[0]-padding).all(1)&(whole*length<=roi[1]+padding).all(1)]).copy()
    del whole
    new_path=out/'surface_blocks.npy'
    if new_path.exists():surface=np.load(new_path)
    else:
        surface,coverage=expanded_blocks(frames,base,roi);np.save(new_path,surface)
        write(out/'allocation.json',dict(base_blocks=len(base),surface_blocks=len(surface),frames=coverage))
    device=device_for(surface)
    write(out/'runtime.json',dict(device=device,pid=os.getpid(),updated=time.time(),
        physical_cuda_device=os.environ.get('CUDA_VISIBLE_DEVICES'),active_blocks=len(surface)))
    paths={}
    for key in LABELS:
        folder=out/key;folder.mkdir(exist_ok=True);path=folder/'mesh.ply';paths[key]=path
        if (folder/'complete.json').exists():continue
        active=base if key=='A' else surface;start=time.time()
        if key=='C':vertices,faces,colors=integrate_sampled(active,frames,device,'inverse')
        else:vertices,faces,colors=integrate_partition(active,frames,device)
        if device.startswith('CUDA'):o3d.core.cuda.release_cache();torch.cuda.empty_cache()
        vertices,faces,colors=crop_mesh(vertices,faces,colors,roi)
        if not len(faces):raise ValueError(f'{key}: no surface in fixed ROI')
        save_mesh(path,vertices,faces,colors)
        stats=topology(vertices,faces,roi)
        write(folder/'complete.json',dict(complete=True,mesh=identity(path),blocks=len(active),
            elapsed=time.time()-start,topology=stats,device=device,mesh_decimation=False))
        print('PROBE_ARM_COMPLETE',key,stats,flush=True)
        del vertices,faces,colors;gc.collect()
    results,check=evaluate(out,roi,frames[0],paths)
    lines=['# bag 远景小实验结果','',
        '这是局部提取消融，主流程与主对比结果未修改。固定 ROI、全部训练和留出相机、相同细体素和原生图像。','',
        '| 组别 | 有效域覆盖率 ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 连通块数 | 内部开口边长 ↓ |','|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:.6f}'
    for key,row in results.items():
        t=read(out/key/'complete.json')['topology'];m=row['mean']
        lines.append('| '+' | '.join([key,fmt(row['coverage']),fmt(m['PSNR']),fmt(m['SSIM']),fmt(m['LPIPS']),str(t['components']),fmt(t['internal_boundary_length'])])+' |')
    lines+=['','[固定视角对比图](comparison.png) · [完整逐视角指标及几何结果](results.json) · [输入与区域选择](plan.json)','',
        f'A 对已发布 mesh 的深度复现：{check}。','',
        'RGB 数值只适用于本 ROI，不能直接并入全场景均值。覆盖域由源深度曲面限定，不是独立真值；连通块更少、覆盖更高也不自动代表几何更准确。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('BAG_SURFACE_PROBE_COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=PUBLIC/'experiments/bag_surface_footprint_20261111')
    p.add_argument('--side',type=float,default=1.2)
    a=p.parse_args();run(a.output,a.side)
