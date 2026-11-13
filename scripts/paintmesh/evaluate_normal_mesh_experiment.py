#!/usr/bin/env python3
"""Full-frame native mesh evaluation for the isolated bag A/N/R experiment.

Only the experiment directory is written. All original triangle partitions,
native pixel rays and frozen generated-object geometry participate. Checkpoints
reduce repeat work after interruptions; they never change resolution or inputs.
"""
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
from mesh_benchmark.stream_cache import outside_frustum

PUBLIC=BASE/'mesh_comparison_whole';ROOT=work_root(PUBLIC)
METHODS=['A','N','R']
LABELS=dict(A='Original complete fine TSDF',N='Depth + normal surface fusion',R='Normal fusion + mesh refinement')


def source_specs(specs):
    result=[]
    for spec in specs:
        record=dict(identity=identity(spec['path']),transform=spec.get('transform'))
        if Path(spec['path']).name.endswith('.mesh.json'):
            meta=read(spec['path'])
            for key,stamp in [('vertices','vertex_identity'),('indices','index_identity')]:
                actual=identity(meta[key]['path'])
                if actual!=meta[stamp]:raise ValueError(f'Native buffer changed: {meta[key]["path"]}')
                record[key]=actual
        result.append(record)
    return result


def cast_native(model,rr):
    """The same original triangle RGB interpolation, plus geometric normals."""
    raw=model.scene.cast_rays(model.o3d.core.Tensor.from_numpy(np.ascontiguousarray(rr,dtype='float32')),nthreads=4)
    z=raw['t_hit'].numpy();good=np.isfinite(z)&(z>0)
    rgb=np.zeros((len(z),3),dtype='float32');normal=np.zeros_like(rgb)
    normal[good]=raw['primitive_normals'].numpy()[good]
    length=np.linalg.norm(normal,axis=1);valid=length>1e-12
    normal[valid]/=length[valid,None]
    geometry=raw['geometry_ids'].numpy();face=raw['primitive_ids'].numpy();uv=raw['primitive_uvs'].numpy()
    from recompose.texture import sample_colors
    for gid,(faces,colors,asset,_) in model.attributes.items():
        keep=good&(geometry==gid)
        if not keep.any() or colors is None:continue
        bary=np.c_[1-uv[keep].sum(1),uv[keep]]
        if asset is not None:rgb[keep]=sample_colors(asset,face[keep],bary)
        else:rgb[keep]=(colors[faces[face[keep]]]*bary[:,:,None]).sum(1)
    return np.where(good,z,0).astype('float32'),rgb,normal


def compose_arrays(depth,rgb,normal,other_depth,other_rgb,other_normal):
    front=np.isfinite(other_depth)&(other_depth>0)&((depth<=0)|(other_depth<depth))
    result=[np.array(a,copy=True) for a in [depth,rgb,normal]]
    for a,b in zip(result,[other_depth,other_rgb,other_normal]):a[front]=b[front]
    return tuple(result)


def atomic_arrays(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.tmp.npz')
    np.savez(temporary,**arrays);os.replace(temporary,path)


def output_files(folder,cameras):
    return [identity(Path(folder)/name) for c in cameras
        for name in [c['stem']+'.png',c['stem']+'_depth.npy',c['stem']+'_normal.npy']]


def render_request(specs,cameras):
    return dict(inputs=source_specs(specs),cameras=cameras,implementation=identity(Path(__file__)),
        backend='PaintMesh Embree native original-triangle RGB and geometric normal',
        full_scene=True,additional_image_downsampling=False,mesh_decimation=False)


def render_specs(folder,specs,cameras,checkpoint_parts=32):
    if not specs or checkpoint_parts<1:raise ValueError('At least one mesh and a positive checkpoint interval are required')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    request=render_request(specs,cameras)
    request_id=digest(request);marker=folder/'complete.json'
    if marker.exists():
        old=read(marker)
        if old.get('complete') and old.get('request_id')==request_id:
            try:current=output_files(folder,cameras)
            except FileNotFoundError:current=None
            if current is not None and old.get('outputs')==current:return old
    marker.unlink(missing_ok=True)
    saved=folder/'checkpoint.json';start=0;previous=None
    if saved.exists():
        old=read(saved)
        if old['request_id']!=request_id:raise ValueError(f'Render checkpoint inputs changed: {folder}')
        if identity(old['arrays']['path'])!=old['arrays']:raise ValueError('Render checkpoint arrays changed')
        previous=Path(old['arrays']['path']);start=old['completed_parts']
        with np.load(previous) as data:arrays={k:data[k].copy() for k in data.files}
    else:
        arrays={}
        for i,c in enumerate(cameras):
            count=c['width']*c['height'];arrays[f'depth_{i}']=np.zeros(count,dtype='float32')
            arrays[f'rgb_{i}']=np.zeros((count,3),dtype='float32');arrays[f'normal_{i}']=np.zeros((count,3),dtype='float32')
    rr=[rays(c) for c in cameras]
    counts=[]
    for j,spec in enumerate(specs):
        if j<start:continue
        mesh=MeshScene([spec])
        if not all(r['has_native_color'] for r in mesh.counts):raise ValueError('Mesh is missing native colors')
        box=mesh.counts[0]['bounds'];counts.extend(mesh.counts)
        for i,c in enumerate(cameras):
            if outside_frustum(box,c):continue
            z,rgb,normal=cast_native(mesh,rr[i]);depth=arrays[f'depth_{i}']
            front=(z>0)&((depth==0)|(z<depth));depth[front]=z[front]
            arrays[f'rgb_{i}'][front]=rgb[front];arrays[f'normal_{i}'][front]=normal[front]
        del mesh;gc.collect()
        print('NORMAL_EXPERIMENT_RENDER',folder.parent.name,j+1,len(specs),flush=True)
        if (j+1)%checkpoint_parts==0 or j+1==len(specs):
            current=folder/f'checkpoint_{j+1:06d}.npz';atomic_arrays(current,arrays)
            write(saved,dict(request_id=request_id,completed_parts=j+1,arrays=identity(current)))
            if previous is not None and previous!=current:previous.unlink(missing_ok=True)
            previous=current
            write(folder/'status.json',dict(updated=time.time(),stage='rendering',completed_parts=j+1,
                  total_parts=len(specs),views=len(cameras)))
    for i,c in enumerate(cameras):
        shape=(c['height'],c['width']);z=arrays[f'depth_{i}'].reshape(shape)
        rgb=np.rint(np.clip(arrays[f'rgb_{i}'],0,1)*255).astype('uint8').reshape(*shape,3)
        np.save(folder/(c['stem']+'_depth.npy'),z)
        np.save(folder/(c['stem']+'_normal.npy'),arrays[f'normal_{i}'].reshape(*shape,3))
        Image.fromarray(rgb).save(folder/(c['stem']+'.png'))
    result=dict(complete=True,request=request,request_id=request_id,views=len(cameras),
        all_triangles_rendered=True,full_scene=True,roi=None,gaussian_rgb_substitution=False,
        outputs=output_files(folder,cameras))
    write(marker,result)
    if previous is not None:previous.unlink(missing_ok=True)
    saved.unlink(missing_ok=True)
    return result


def validate_methods(output):
    output=Path(output);methods={}
    for method in METHODS:
        path=output/method/'complete.json'
        if not path.exists():raise ValueError(f'{method}: complete full-scene extraction is not ready')
        meta=read(path)
        if not meta.get('complete') or not meta.get('mesh_specs') or meta.get('full_scene') is False or meta.get('roi') is not None:
            raise ValueError(f'{method}: incomplete or local-only extraction')
        source_specs(meta['mesh_specs']);methods[method]=meta
    return methods


def eval_cameras():
    configure_sources(ROOT)
    info=read(ROOT/'common/bag/complete.json');cs=[c for c in info['cameras'] if c['part']=='held_out']
    if len(cs)!=info['held_out_count']:raise ValueError('Evaluation camera count changed')
    return cs


def render_method(output,method,cameras=None,checkpoint_parts=32):
    output=Path(output)
    if method not in METHODS:raise ValueError(method)
    meta=read(output/method/'complete.json')
    if not meta.get('complete') or not meta.get('mesh_specs'):raise ValueError('Incomplete upstream mesh')
    return render_specs(output/method/'renders',meta['mesh_specs'],eval_cameras() if cameras is None else cameras,checkpoint_parts)


def render_composed(output,cameras):
    output=Path(output);graph_path=PUBLIC/'recomposition/bag/composition/scene.json';graph=read(graph_path)
    specs=[dict(path=o['mesh']['path'],transform=o['object_to_world']) for o in graph['objects']]
    if not specs:raise ValueError('No fixed generated object meshes for composed evaluation')
    render_specs(output/'objects/renders',specs,cameras)
    for method in METHODS:
        source=output/method/'renders';dest=output/method/'composed_renders';dest.mkdir(exist_ok=True)
        request=dict(environment=read(source/'complete.json')['request_id'],
            objects=read(output/'objects/renders/complete.json')['request_id'],scene=identity(graph_path))
        marker=dest/'complete.json'
        if marker.exists() and read(marker).get('request')==request and read(marker).get('complete'):
            try:current=output_files(dest,cameras)
            except FileNotFoundError:current=None
            if current is not None and read(marker).get('outputs')==current:continue
        marker.unlink(missing_ok=True)
        for c in cameras:
            stem=c['stem'];load=lambda path:(np.load(path/(stem+'_depth.npy')),
                np.asarray(Image.open(path/(stem+'.png')).convert('RGB')),np.load(path/(stem+'_normal.npy')))
            depth,rgb,normal=compose_arrays(*load(source),*load(output/'objects/renders'))
            np.save(dest/(stem+'_depth.npy'),depth);np.save(dest/(stem+'_normal.npy'),normal)
            Image.fromarray(rgb).save(dest/(stem+'.png'))
        write(marker,dict(complete=True,request=request,request_id=digest(request),views=len(cameras),
            frozen_object_geometry=True,frozen_object_pose=True,generated_object_source_status=graph['status'],
            outputs=output_files(dest,cameras)))


def render_folder(output,method,state):
    return Path(output)/method/('composed_renders' if state=='composed' else 'renders')


def average_rows(rows,names):
    return {name:float(np.mean([r[name] for r in rows if r.get(name) is not None]))
        if any(r.get(name) is not None for r in rows) else None for name in names}


def teacher_records(output,cameras):
    path=Path(output)/'teacher_views/complete.json'
    if not path.exists():return {},dict(computed=False,reason='Held-out frozen-3DGS floating depth/normal cache not provided; mesh A is not used as a teacher')
    marker=read(path)
    if not marker.get('complete'):return {},dict(computed=False,reason='Frozen-3DGS teacher cache incomplete')
    records={Path(r['name']).stem:r for r in marker['frames']}
    missing=[c['name'] for c in cameras if c['stem'] not in records]
    if missing:return {},dict(computed=False,reason='Missing teacher views',missing=missing)
    for record in records.values():
        if identity(record['path'])!=record['identity']:raise ValueError('Teacher source changed')
    return records,dict(computed=True,source=identity(path),use='Consistency with frozen 3DGS, not geometric ground truth')


def teacher_consistency(camera,mesh_depth,mesh_normal,teacher,mask,eligible):
    depth=teacher['depth'];normal=teacher['normal']
    if normal.shape!=(*depth.shape,3) and normal.shape==(3,*depth.shape):normal=np.moveaxis(normal,0,-1)
    if normal.shape!=(*depth.shape,3) or mesh_depth.shape!=depth.shape:raise ValueError('Teacher image geometry mismatch')
    alpha=teacher.get('alpha',np.ones(depth.shape,dtype='float32')).squeeze()
    lengths=np.linalg.norm(normal,axis=-1)
    teacher_valid=np.isfinite(depth)&(depth>0)&np.isfinite(normal).all(-1)&(lengths>1e-8)&(alpha>=.5)&eligible
    allowed=mask&teacher_valid
    if not allowed.any():return dict(computed=False,pixels=0,teacher_eligible_pixels=int(teacher_valid.sum()))
    # Native primitive normals are in world coordinates; PGSR normals are in camera coordinates.
    mesh=mesh_normal@np.asarray(camera['w2c'])[:3,:3].T
    mesh_length=np.linalg.norm(mesh,axis=-1)
    mesh=np.divide(mesh,mesh_length[...,None],out=np.zeros_like(mesh),where=mesh_length[...,None]>1e-8)
    normal=np.divide(normal,lengths[...,None],out=np.zeros_like(normal),where=lengths[...,None]>1e-8)
    rr=rays(camera)[:,3:].reshape(*depth.shape,3)@np.asarray(camera['w2c'])[:3,:3].T
    mesh=np.where((np.sum(mesh*rr,axis=-1)>0)[...,None],-mesh,mesh)
    normal=np.where((np.sum(normal*rr,axis=-1)>0)[...,None],-normal,normal)
    cosine=np.clip(np.sum(mesh*normal,axis=-1),-1,1)
    angular=allowed&(mesh_length>1e-8);angle=np.degrees(np.arccos(cosine[angular]))
    error=np.abs(mesh_depth[allowed]-depth[allowed]);relative=error/np.maximum(depth[allowed],1e-8)
    return dict(computed=True,pixels=int(allowed.sum()),teacher_eligible_pixels=int(teacher_valid.sum()),
        own_teacher_coverage=float(((mesh_depth>0)&teacher_valid).sum()/max(teacher_valid.sum(),1)),
        depth_abs_mean=float(error.mean()),depth_abs_median=float(np.median(error)),
        depth_relative_mean=float(relative.mean()),depth_relative_p95=float(np.quantile(relative,.95)),
        normal_pixels=int(angular.sum()),normal_angle_mean_deg=float(angle.mean()) if len(angle) else None,
        normal_angle_median_deg=float(np.median(angle)) if len(angle) else None,
        normal_angle_p95_deg=float(np.quantile(angle,.95)) if len(angle) else None)


def evaluate(output,cameras=None):
    output=Path(output);methods=validate_methods(output);configure_sources(ROOT)
    import torch,lpips
    from mesh_benchmark.score import image_metrics
    from mesh_benchmark.overlap import observations,select_anchors,prediction_points
    cs=eval_cameras() if cameras is None else cameras
    source_kind=read(output/'plan.json').get('source_kind','background')
    states=['background','composed'] if source_kind=='background' else ['background','original_scene']
    teachers,teacher_status=teacher_records(output,cs)
    criterion=lpips.LPIPS(net='vgg',version='0.1',spatial=True).to('cpu').eval();torch.set_num_threads(4)
    for parameter in criterion.parameters():parameter.requires_grad_(False)
    info=read(ROOT/'common/bag/complete.json');results={}
    for state in states:
        sources={m:render_folder(output,m,state) for m in METHODS}
        for method,path in sources.items():
            if not (path/'complete.json').exists() or not read(path/'complete.json').get('complete'):
                raise ValueError(f'Full render not completed: {path}')
            render_meta=read(path/'complete.json')
            if render_meta.get('outputs')!=output_files(path,cs):raise ValueError(f'Rendered files changed: {path}')
            environment=read(output/method/'renders/complete.json')
            if environment['request_id']!=digest(render_request(methods[method]['mesh_specs'],cs)):
                raise ValueError(f'Rendered mesh source changed: {method}')
            if state=='composed':
                objects=read(output/'objects/renders/complete.json')
                graph=PUBLIC/'recomposition/bag/composition/scene.json'
                expected=dict(environment=environment['request_id'],objects=objects['request_id'],scene=identity(graph))
                if render_meta['request']!=expected:raise ValueError('Composed render source changed')
        depth={m:[np.load(path/(c['stem']+'_depth.npy'),mmap_mode='r') for c in cs] for m,path in sources.items()}
        shared=[];eligible=[];domain=output/'support'/state;domain.mkdir(parents=True,exist_ok=True)
        for i,c in enumerate(cs):
            fg=np.asarray(Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_foreground.png')))>127
            allowed=~fg if state=='background' else np.ones(fg.shape,dtype=bool);eligible.append(allowed)
            mask=allowed&np.logical_and.reduce([np.isfinite(depth[m][i])&(depth[m][i]>0) for m in METHODS]);shared.append(mask)
            Image.fromarray(mask.astype('uint8')*255).save(domain/(c['stem']+'.png'))
        with np.load(ROOT/'common/bag/reference_points.npz') as data:reference=data['background' if state=='background' else 'recomposition']
        obs=observations('bag',cs,reference)
        chosen=select_anchors(obs[:,0],obs[:,1],obs[:,2],shared,np.ones(len(reference),dtype=bool),42,max_points=max(1,len(reference))) if len(obs) else np.array([],dtype=int)
        np.savez(domain/'geometry_reference.npz',points=reference[obs[chosen,0]],observations=obs[chosen])
        for method in METHODS:
            rows=[];teacher_rows=[]
            for i,c in enumerate(cs):
                target=np.asarray(Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_gt.png')).convert('RGB'))
                rgb=np.asarray(Image.open(sources[method]/(c['stem']+'.png')).convert('RGB'))
                if rgb.shape!=target.shape or target.shape[:2]!=depth[method][i].shape:raise ValueError('No evaluation image resizing is allowed')
                own=np.isfinite(depth[method][i])&(depth[method][i]>0)&eligible[i]
                score=image_metrics(rgb,target,shared[i],criterion,'cpu')
                rows.append(dict(name=c['name'],**score,eligible_pixels=int(eligible[i].sum()),
                    evaluation_pixels=int(shared[i].sum()),own_pixels=int(own.sum()),
                    mesh_coverage=float(own.sum()/max(eligible[i].sum(),1)),
                    common_coverage=float(shared[i].sum()/max(eligible[i].sum(),1)),
                    status='scored' if shared[i].any() else 'no common covered pixels'))
                # Object geometry is generated independently; do not train or evaluate
                # its normal against the removed-background Gaussian teacher.
                if teachers and state=='background':
                    with np.load(teachers[c['stem']]['path']) as a:teacher={k:a[k] for k in a.files}
                    normal=np.load(sources[method]/(c['stem']+'_normal.npy'),mmap_mode='r')
                    teacher_rows.append(dict(name=c['name'],**teacher_consistency(c,depth[method][i],normal,teacher,shared[i],eligible[i])))
            geometry=None
            if len(chosen):
                pred=prediction_points(cs,depth[method],chosen,obs)
                geometry=surface_metrics(pred,reference[obs[chosen,0]],info['extent'])
            result=dict(complete=True,method=method,state=state,mean=average_rows(rows,['PSNR','SSIM','LPIPS','mesh_coverage','common_coverage']),
                per_view=rows,evaluation_views=len(cs),scored_views=sum(r['evaluation_pixels']>0 for r in rows),
                geometry=geometry,geometry_reference_observations=len(chosen),geometry_eligible_points=len(reference),
                teacher_consistency=dict(status=teacher_status,per_view=teacher_rows,
                    mean=average_rows(teacher_rows,['depth_abs_mean','depth_relative_mean','normal_angle_mean_deg','own_teacher_coverage'])),
                request=dict(render=read(sources[method]/'complete.json')['request_id'],
                    support=digest([identity(domain/(c['stem']+'.png')) for c in cs])),
                gaussian_rgb_substitution=False,appearance_fitting=False,
                appearance_interpretation='A vs N may differ in both geometry and native confidence-weighted TSDF colors; R is designed to preserve N vertex colors while refining coordinates',
                reference_warning='Conditional visible-surface scores against observed sparse COLMAP points, not independent scan ground truth')
            write(output/'metrics'/state/method/'metrics.json',result);results[state+'/'+method]=result
            print('NORMAL_EXPERIMENT_METRICS',state,method,result['mean'],'reference_points',len(chosen),flush=True)
        write(domain/'complete.json',dict(complete=True,methods=METHODS,state=state,
            views=len(cs),geometry_reference_observations=len(chosen),
            inputs={m:read(sources[m]/'complete.json')['request_id'] for m in METHODS},
            selection='Positive finite mesh depth intersection; background excludes target mask; no error/F1 filtering'))
    write_reports(output,states,results,source_kind)
    figures(output,cs,states,results)
    result=dict(complete=True,updated=time.time(),source_kind=source_kind,states=states,
        methods=METHODS,full_scene=True,roi=None,evaluation_views=len(cs),results=results,
        teacher_status=teacher_status,production_results_modified=False,
        gaussian_retraining=False,additional_image_downsampling=False,mesh_decimation=False)
    write(output/'results.json',result)
    print('NORMAL_FULL_SCENE_EVALUATION_COMPLETE',str(output),flush=True)
    return result


def write_reports(output,states,results,source_kind):
    folder=output/'reports';folder.mkdir(exist_ok=True)
    fmt=lambda value:'—' if value is None else f'{value:.6f}'
    for state in states:
        title={'background':'纯背景（排除目标 mask）','composed':'完整背景＋固定 MV-SAM3D 物体','original_scene':'第一阶段完整原场景（保留原物体）'}[state]
        lines=[f'# bag：{title}','',
            'A/N/R 使用同一检查点、输入相机、原生图像尺寸和网格渲染后端。没有减面、颜色场拟合或高斯图片替代。主指标只比较三组都有表面的像素，同时报告自身覆盖率。','',
            'N 的几何置信度权重也用于原生TSDF颜色融合，因此 A 与 N 的RGB变化同时包含几何和颜色融合的影响，不能全归因于几何。R 的设计保留 N 的原生顶点色和拓扑，仅精修实际坐标。几何改进优先由 CD/F1、冻结3DGS深度/法线一致性及覆盖率判断。','',
            '几何参考是 COLMAP 实际观测点，不是独立扫描真值。共享几何支撑由实际特征轨迹和网格正深度决定，不根据距离或误差筛点。CD-L1 为双向最近距离均值的半和；F1 阈值以共同相机半径的比例定义。','',
            '| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 自身覆盖率 ↑ | 共同覆盖率 | CD-L1 ↓ | F1@0.5% ↑ | F1@1% ↑ | F1@2% ↑ | 参考点数 | 有分数视角/全部视角 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for m in METHODS:
            r=results[state+'/'+m];g=r['geometry'];a=r['mean']
            values=[a['PSNR'],a['SSIM'],a['LPIPS'],a['mesh_coverage'],a['common_coverage'],g['CD_L1'] if g else None,
                    *[g['F_scores'][key]['F1'] if g else None for key in ['0.005','0.01','0.02']]]
            lines.append('| '+' | '.join([m+': '+LABELS[m],*[fmt(v) for v in values],str(r['geometry_reference_observations']),f'{r["scored_views"]}/{r["evaluation_views"]}'])+' |')
        if state=='background':
            lines+=['','## 与冻结 3DGS 的深度、法线一致性','',
                '该项测量导出对已有模型的几何保真度，不当作真实几何准确率。仅使用同一检查点的浮点教师深度和法线；共同有效域和背景mask保持一致。','']
            if any(results[state+'/'+m]['teacher_consistency']['per_view'] for m in METHODS):
                lines+=['| 方法 | 深度绝对误差均值 ↓ | 深度相对误差均值 ↓ | 法线角误差均值（度）↓ | 自身教师覆盖率 ↑ |','|---|---:|---:|---:|---:|']
                for m in METHODS:
                    a=results[state+'/'+m]['teacher_consistency']['mean']
                    lines.append('| '+' | '.join([m,*[fmt(a[k]) for k in ['depth_abs_mean','depth_relative_mean','normal_angle_mean_deg','own_teacher_coverage']]])+' |')
            else:lines+=['未计算：未提供完整留出视角浮点教师缓存；不会使用 A mesh 代替 3DGS。']
        lines+=['','[全部逐视角指标](../results.json) · [整图与局部细节对比](../figures/README.md)','',
            '零共同像素的视角保留并标明无评分区域，不计入 RGB 均值；覆盖率仍计入。此独立实验不覆盖正式全方法报告。']
        (folder/(state+'.md')).write_text('\n'.join(lines)+'\n')


def crop_boxes(output,camera):
    """Fixed display crops use target mask and A depth, never score ranking."""
    fg=np.asarray(Image.open(ROOT/'common/bag/held_out'/(camera['stem']+'_foreground.png')))>127
    yy,xx=np.where(fg)
    if len(xx):target=[max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(camera['width'],int(xx.max())+21),min(camera['height'],int(yy.max())+21)]
    else:target=[camera['width']//4,camera['height']//4,3*camera['width']//4,3*camera['height']//4]
    d=np.load(Path(output)/'A/renders'/(camera['stem']+'_depth.npy'),mmap_mode='r')
    valid=np.isfinite(d)&(d>0)&~fg
    if valid.any():
        threshold=np.quantile(d[valid],.75);far=valid&(d>=threshold)
        from scipy.ndimage import uniform_filter
        size=min(128,camera['width'],camera['height']);counts=uniform_filter(far.astype('float64'),size=size,mode='constant')
        y,x=np.unravel_index(np.argmax(counts),counts.shape)
        x0=max(0,min(camera['width']-size,int(x)-size//2));y0=max(0,min(camera['height']-size,int(y)-size//2))
        distant=[x0,y0,x0+size,y0+size]
    else:distant=[0,0,min(128,camera['width']),min(128,camera['height'])]
    return target,distant


def figures(output,cameras,states,results):
    from plot_mesh_method_comparison import font
    output=Path(output);folder=output/'figures';folder.mkdir(exist_ok=True)
    inventory=[]
    for c in cameras:
        target,distant=crop_boxes(output,c);width,height=c['width'],c['height']
        zoom_h=max(256,target[3]-target[1],distant[3]-distant[1])
        for state in states:
            canvas=Image.new('RGB',(4*width,height+2*zoom_h+190),(245,247,250));draw=ImageDraw.Draw(canvas)
            draw.text((10,7),f'bag | {c["name"]} | {state} | full original-resolution mesh rendering',font=font(21),fill='black')
            gt=Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_gt.png')).convert('RGB')
            for j,m in enumerate(['GT',*METHODS]):
                pic=gt if m=='GT' else Image.open(render_folder(output,m,state)/(c['stem']+'.png')).convert('RGB')
                if pic.size!=(width,height):raise ValueError('Unexpected figure downsampling')
                x=j*width;draw.text((x+8,37),m if m=='GT' else m+': '+LABELS[m],font=font(18),fill='black');canvas.paste(pic,(x,70))
                for row,(box,label) in enumerate([(target,'Target-region crop'),(distant,'Far-surface crop')]):
                    y=height+90+row*(zoom_h+30);draw.text((x+8,y),label,font=font(17),fill='black')
                    crop=pic.crop(box);factor=max(1,min(width/crop.width,zoom_h/crop.height))
                    crop=crop.resize((round(crop.width*factor),round(crop.height*factor)),Image.Resampling.NEAREST)
                    canvas.paste(crop,(x,y+25))
                if m!='GT':
                    metric=next(r for r in results[state+'/'+m]['per_view'] if r['name']==c['name'])
                    scores='no common pixels' if metric['PSNR'] is None else f'P {metric["PSNR"]:.2f} S {metric["SSIM"]:.3f} L {metric["LPIPS"]:.3f}'
                    draw.text((x+8,canvas.height-30),scores+f' | coverage {metric["mesh_coverage"]:.1%}',font=font(17),fill='black')
            dest=folder/state;dest.mkdir(exist_ok=True);path=dest/(c['stem']+'.png');canvas.save(path)
            inventory.append(dict(camera=c['name'],state=state,path=str(path),target_crop=target,far_crop=distant,
                crop_rule='Target annotation and top-quartile A-depth 128px region; no metric ranking',native_full_frame_size=[width,height]))
    write(folder/'manifest.json',dict(complete=True,updated=time.time(),views=len(cameras),states=states,records=inventory))
    lines=['# bag 全图 mesh 对比','',
        '每列完整原生图像尺寸，另给目标区域和远景细节。局部放大只用于展示，原始 RGB/深度保存于各方法 renders 中，指标从原始像素计算。','',
        '| 相机 | '+' | '.join(states)+' |','|---|'+'---|'*len(states)]
    for c in cameras:lines.append('| '+c['name']+' | '+' | '.join(f'[PNG]({state}/{c["stem"]}.png)' for state in states)+' |')
    (folder/'README.md').write_text('\n'.join(lines)+'\n')
    index=['# bag 深度＋法线完整场景导出实验','',
        'A：原完整细TSDF；N：深度与法线感知融合；R：法线融合后的小步mesh几何精修。三组全部渲染完整场景，没有局部图像替换。','',
        'A/N 的原生融合颜色可能随置信度权重变化，RGB指标不能单独证明几何改进；N/R 保留同一原生顶点色，R 仅精修坐标。','',
        '原生几何与颜色，冻结3DGS，不使用测试GT进行优化。只有完整 A/N/R 提取、渲染、两份指标和全部视角图片完成后才标记结果完整。','',
        *[f'- [{state} 指标](reports/{state}.md)' for state in states],
        '- [全部整图与局部细节](figures/README.md)','- [逐视角与几何数据](results.json)','',
        '几何参考为实际观测COLMAP稀疏点，指标是共同可见表面条件下的一致性，不能解释为扫描真值完整度。']
    (output/'README.md').write_text('\n'.join(index)+'\n')


def run(output):
    output=Path(output);validate_methods(output);cs=eval_cameras()
    for method in METHODS:render_method(output,method,cs)
    if read(output/'plan.json').get('source_kind','background')=='background':render_composed(output,cs)
    return evaluate(output,cs)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true');parser.add_argument('--score-only',action='store_true')
    args=parser.parse_args()
    if args.render_only and args.score_only:parser.error('Choose at most one partial stage')
    if args.render_only:
        validate_methods(args.output);cs=eval_cameras()
        for method in METHODS:render_method(args.output,method,cs)
        if read(args.output/'plan.json').get('source_kind','background')=='background':render_composed(args.output,cs)
    elif args.score_only:evaluate(args.output)
    else:run(args.output)
