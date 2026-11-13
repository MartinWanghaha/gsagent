#!/usr/bin/env python3
"""Independent full-frame evaluation of one bag depth-anchored tetrahedral colored mesh.

Only the isolated experiment directory is written. Existing baselines retain
their original mesh-renderer RGB/depth, and are rescored on the same visible
surface intersection as the new experiment. Vertex colors are used as stored;
this evaluator does not claim UV texturing or fit an appearance field.
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
from mesh_benchmark.core import BASE,read,write,identity,digest,surface_metrics
from mesh_benchmark.layout import work_root,configure_sources
from evaluate_normal_mesh_experiment import render_specs,output_files,compose_arrays,source_specs

PUBLIC=BASE/'mesh_comparison_whole'
ROOT=work_root(PUBLIC)
DEFAULT=PUBLIC/'experiments/bag_depth_tetra_20261112'
METHODS=['2dgs','come','gof','gaussian_wrapping','paintmesh','depth_tetra']
LABELS=dict(zip(METHODS,['2DGS','CoMe','GOF','Gaussian Wrapping','Published PaintMesh','Depth-anchored tetrahedral (vertex RGB)']))
STATES=['background','composed']


def same_cameras(actual,expected):
    """Reject subtle camera changes even if filenames and image sizes agree."""
    names={c['name']:c for c in actual}
    if len(names)!=len(actual) or set(names)!={c['name'] for c in expected}:
        raise ValueError('Native mesh-renderer camera set differs from the experiment')
    for camera in expected:
        other=names[camera['name']]
        for key in ['stem','width','height']:
            if other[key]!=camera[key]:raise ValueError('Native camera name/image dimensions changed')
        for key in ['K','w2c']:
            if not np.array_equal(other[key],camera[key]):raise ValueError('Native camera calibration changed')


def covered_support(depths,foreground,background):
    foreground=np.asarray(foreground,dtype=bool)
    eligible=~foreground if background else np.ones(foreground.shape,dtype=bool)
    shared=eligible.copy()
    if not depths:raise ValueError('At least one method depth image is required')
    for depth in depths:
        if np.shape(depth)!=foreground.shape:raise ValueError('Depth and evaluation mask resolutions differ')
        shared&=np.isfinite(depth)&(depth>0)
    return eligible,shared


def evaluation_cameras():
    configure_sources(ROOT)
    info=read(ROOT/'common/bag/complete.json')
    cameras=[c for c in info['cameras'] if c['part']=='held_out']
    if not info.get('complete') or len(cameras)!=13 or info['held_out_count']!=13:
        raise ValueError('All 13 original held-out bag views are required')
    same_cameras(cameras,cameras)
    return cameras


def status(output,stage,**detail):
    write(Path(output)/'evaluation_status.json',dict(complete=False,updated=time.time(),pid=os.getpid(),stage=stage,**detail))


def combine_render_folders(background,objects,destination,cameras,scene_identity):
    """Nearest positive finite triangle depth, preserving source RGB/normals."""
    background=Path(background);objects=Path(objects);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    metadata={name:read(folder/'complete.json') for name,folder in [('background',background),('objects',objects)]}
    for name,folder in [('background',background),('objects',objects)]:
        if not metadata[name].get('complete') or metadata[name]['outputs']!=output_files(folder,cameras):
            raise ValueError('Incomplete or changed actual triangle rendering: '+name)
    request=dict(background=metadata['background']['request_id'],objects=metadata['objects']['request_id'],scene=scene_identity,
        implementation=identity(Path(__file__)),composition='Nearest positive finite true triangle depth; fixed generated object and pose')
    marker=destination/'complete.json'
    if marker.exists():
        old=read(marker)
        if old.get('complete') and old.get('request')==request:
            try:current=output_files(destination,cameras)
            except FileNotFoundError:current=None
            if current is not None and old.get('outputs')==current:return old
    marker.unlink(missing_ok=True)
    for camera in cameras:
        stem=camera['stem']
        def load(folder):
            return (np.load(folder/(stem+'_depth.npy')),
                np.asarray(Image.open(folder/(stem+'.png')).convert('RGB')),
                np.load(folder/(stem+'_normal.npy')))
        depth,rgb,normal=compose_arrays(*load(background),*load(objects))
        np.save(destination/(stem+'_depth.npy'),depth);np.save(destination/(stem+'_normal.npy'),normal)
        Image.fromarray(rgb).save(destination/(stem+'.png'))
    row=dict(complete=True,request=request,request_id=digest(request),views=len(cameras),
        frozen_object_geometry=True,frozen_object_pose=True,all_triangles_rendered=True,
        gaussian_rgb_substitution=False,outputs=output_files(destination,cameras))
    write(marker,row);return row


def render_experiment(output,cameras):
    output=Path(output);mesh=output/'mesh/background.ply'
    if not mesh.exists():raise ValueError('Full depth-anchored tetrahedral background mesh is not ready: '+str(mesh))
    graph_path=PUBLIC/'recomposition/bag/composition/scene.json';graph=read(graph_path)
    if not graph.get('complete') or not graph.get('objects'):raise ValueError('Fixed composed object source is incomplete')
    plan=read(output/'plan.json')
    if plan['source_composition_identity']!=identity(graph_path) or plan['fixed_generated_objects']!=graph['objects']:
        raise ValueError('Composition or generated objects differ from the frozen experiment plan')
    object_specs=[dict(path=item['mesh']['path'],transform=item['object_to_world']) for item in graph['objects']]
    frozen=dict(background=identity(mesh),scene=identity(graph_path),objects=source_specs(object_specs),plan=identity(output/'plan.json'),
        cameras=cameras,implementation=identity(Path(__file__)),
        rendering_helper=identity(HERE/'evaluate_normal_mesh_experiment.py'),
        appearance='Stored native vertex RGB and fixed generated-object texture/colors; no UV bake or learned color field')
    status(output,'rendering full depth-anchored tetrahedral background',evaluation_views=13)
    render_specs(output/'renders/background',[dict(path=str(mesh))],cameras)
    status(output,'rendering fixed generated object',evaluation_views=13)
    render_specs(output/'renders/objects',object_specs,cameras)
    combine_render_folders(output/'renders/background',output/'renders/objects',output/'renders/composed',cameras,identity(graph_path))
    if identity(mesh)!=frozen['background'] or identity(graph_path)!=frozen['scene']:
        raise ValueError('Experimental mesh or object placement changed while rendering')
    if source_specs(object_specs)!=frozen['objects']:raise ValueError('Generated object source changed while rendering')
    write(output/'renders/complete.json',dict(complete=True,request=frozen,evaluation_views=13,
        background=identity(output/'renders/background/complete.json'),composed=identity(output/'renders/composed/complete.json')))


def native_sources(output,cameras):
    """Read original native backends; write fingerprint caches only to output."""
    from evaluate_native_mesh_rgb import rendering,verified_file_fingerprint
    output=Path(output);sources={state:{} for state in STATES};stamps=[]
    common=read(ROOT/'common/bag/complete.json')
    for method in METHODS[:4]:
        folder=ROOT/'native_renderers'/method/'bag';marker=folder/'complete.json';meta=read(marker)
        if not meta.get('complete') or meta.get('pilot') or meta.get('views')!=13:
            raise ValueError('Native baseline rendering is incomplete: '+method)
        if not meta.get('all_source_triangles_preserved') or meta['request'].get('gaussian_renderer_used'):
            raise ValueError('Real complete native mesh renders are required')
        same_cameras(meta['request']['cameras'],cameras)
        fingerprint=meta['request']['mesh']
        if verified_file_fingerprint(output,fingerprint['path'])!=fingerprint:
            raise ValueError('Native baseline mesh differs from its renderer source: '+method)
        stamps.append(identity(marker))
        for state in STATES:sources[state][method]=dict(rgb=folder/'renders',depth=folder/'depth',
            backend=meta['request']['backend'],renderer=identity(marker),depth_suffix='.npy')
    for state in STATES:
        original_state='recomposition' if state=='composed' else 'background'
        folder=ROOT/'cache/paintmesh/bag'/original_state;marker=folder/'complete.json';meta=read(marker)
        if not meta.get('complete') or meta['request'].get('common')!=common['request_id']:
            raise ValueError('Published PaintMesh native renderer has different common camera inputs')
        for item in meta['request']['specs']:
            if identity(item['identity']['path'])!=item['identity']:raise ValueError('Published PaintMesh geometry changed')
            path=Path(item['identity']['path'])
            if path.name.endswith('.mesh.json'):
                asset=read(path)
                for key,stamp in [('vertices','vertex_identity'),('indices','index_identity')]:
                    if identity(asset[key]['path'])!=asset[stamp]:raise ValueError('Published native triangle buffer changed')
        actual=rendering(ROOT,'paintmesh','bag',original_state)
        if actual['rgb']!=folder/'native_renders':raise ValueError('Learned color-field RGB is forbidden')
        sources[state]['paintmesh']=dict(rgb=actual['rgb'],depth=actual['depth'],
            backend=actual['request']['backend'],renderer=identity(marker),depth_suffix='.npy')
        stamps.append(identity(marker))
        own=output/'renders'/state;meta=read(own/'complete.json')
        if not meta.get('complete') or meta['outputs']!=output_files(own,cameras):raise ValueError('Experimental full renders are incomplete or changed')
        sources[state]['depth_tetra']=dict(rgb=own,depth=own,backend='PaintMesh actual triangle Embree with native vertex RGB',
            renderer=identity(own/'complete.json'),depth_suffix='_depth.npy')
        stamps.append(identity(own/'complete.json'))
    for state in STATES:
        for method,source in sources[state].items():
            source['files']=[]
            for camera in cameras:
                rgb=source['rgb']/(camera['stem']+'.png');depth=source['depth']/(camera['stem']+source['depth_suffix'])
                if Image.open(rgb).size!=(camera['width'],camera['height']):raise ValueError('Native RGB resolution mismatch: '+str(rgb))
                array=np.load(depth,mmap_mode='r')
                if array.shape!=(camera['height'],camera['width']):raise ValueError('Native depth resolution mismatch: '+str(depth))
                source['files'].extend([identity(rgb),identity(depth)])
            stamps.extend(source['files'])
    return sources,stamps


def average(rows,keys):
    return {key:float(np.mean([r[key] for r in rows if r.get(key) is not None]))
        if any(r.get(key) is not None for r in rows) else None for key in keys}


def figures(output,cameras,sources,results=None):
    """Publish every full-size view immediately, before expensive metrics."""
    from plot_mesh_method_comparison import font
    output=Path(output);inventory=[]
    for state in STATES:
        folder=output/'figures'/state;folder.mkdir(parents=True,exist_ok=True)
        for i,camera in enumerate(cameras):
            width,height=camera['width'],camera['height'];row_height=height+75
            canvas=Image.new('RGB',(4*width,2*row_height+50),(247,248,250));draw=ImageDraw.Draw(canvas)
            draw.text((10,8),'bag | '+camera['name']+' | '+state+' | original-size complete mesh views',font=font(20),fill='black')
            for j,method in enumerate(['GT',*METHODS]):
                x=(j%4)*width;y=50+(j//4)*row_height
                source=ROOT/'common/bag/held_out'/(camera['stem']+'_gt.png') if method=='GT' else sources[state][method]['rgb']/(camera['stem']+'.png')
                image=Image.open(source).convert('RGB')
                if image.size!=(width,height):raise ValueError('Figure images must preserve the native resolution')
                draw.text((x+7,y+2),'GT' if method=='GT' else LABELS[method],font=font(17),fill='black')
                canvas.paste(image,(x,y+25))
                if method!='GT':
                    if results is None:text='metrics pending | native vertex / texture RGB'
                    else:
                        metric=results[state+'/'+method]['per_view'][i]
                        text='no shared valid pixels' if metric['PSNR'] is None else f'P {metric["PSNR"]:.2f}  S {metric["SSIM"]:.3f}  L {metric["LPIPS"]:.3f}'
                        text+=f' | coverage {metric["mesh_coverage"]:.1%}'
                    draw.text((x+7,y+height+31),text,font=font(15),fill='black')
                subplot=output/'subplots'/method/state;subplot.mkdir(parents=True,exist_ok=True)
                image.save(subplot/(camera['stem']+'.png'))
            path=folder/(camera['stem']+'.png');canvas.save(path)
            inventory.append(dict(camera=camera['name'],state=state,path=str(path),native_panel_size=[width,height]))
    write(output/'figures/manifest.json',dict(images_complete=True,metrics_complete=results is not None,
        updated=time.time(),views=13,figure_count=len(inventory),records=inventory,images_resized=False))
    lines=['# bag depth-anchored tetrahedral 全图对比','',
        '26 张完整视角对比已生成。'+('指标已重算。' if results is not None else '指标仍在计算，图片中的 pending 不代表零分。'),'',
        '各子图保持原生尺寸；GT、四个基线、已发布 PaintMesh，以及本次 depth-anchored tetrahedral mesh。使用存储的原生顶点色及生成物体纹理；本阶段未做新的 UV 烘焙。','',
        '| 视角 | 纯背景 | 组合 |','|---|---|---|']
    for camera in cameras:lines.append(f'| {camera["name"]} | [PNG](background/{camera["stem"]}.png) | [PNG](composed/{camera["stem"]}.png) |')
    (output/'figures/README.md').write_text('\n'.join(lines)+'\n')


def score(output,cameras,sources):
    import torch,lpips
    from mesh_benchmark.score import image_metrics
    from mesh_benchmark.overlap import observations,select_anchors,prediction_points
    output=Path(output);status(output,'scoring common visible surfaces',evaluation_views=13)
    torch.set_num_threads(4)
    criterion=lpips.LPIPS(net='vgg',version='0.1',spatial=True).cpu().eval()
    for parameter in criterion.parameters():parameter.requires_grad_(False)
    info=read(ROOT/'common/bag/complete.json');results={}
    for state in STATES:
        depths={method:[np.load(source['depth']/(camera['stem']+source['depth_suffix']),mmap_mode='r') for camera in cameras]
            for method,source in sources[state].items()}
        eligible=[];shared=[];folder=output/'support'/state;folder.mkdir(parents=True,exist_ok=True)
        for i,camera in enumerate(cameras):
            foreground=np.asarray(Image.open(ROOT/'common/bag/held_out'/(camera['stem']+'_foreground.png')))>127
            allowed,mask=covered_support([depths[m][i] for m in METHODS],foreground,state=='background')
            eligible.append(allowed);shared.append(mask)
            Image.fromarray(mask.astype('uint8')*255).save(folder/(camera['stem']+'.png'))
        with np.load(ROOT/'common/bag/reference_points.npz') as data:reference=data['background' if state=='background' else 'recomposition']
        obs=observations('bag',cameras,reference)
        selected=select_anchors(obs[:,0],obs[:,1],obs[:,2],shared,np.ones(len(reference),dtype=bool),42,
            max_points=max(1,len(reference))) if len(obs) else np.empty(0,dtype='int64')
        np.savez(folder/'geometry_reference.npz',points=reference[obs[selected,0]],observations=obs[selected])
        for method in METHODS:
            rows=[]
            for i,camera in enumerate(cameras):
                target=np.asarray(Image.open(ROOT/'common/bag/held_out'/(camera['stem']+'_gt.png')).convert('RGB'))
                rgb=np.asarray(Image.open(sources[state][method]['rgb']/(camera['stem']+'.png')).convert('RGB'))
                own=eligible[i]&np.isfinite(depths[method][i])&(depths[method][i]>0)
                if rgb.shape!=target.shape or rgb.shape[:2]!=own.shape:raise ValueError('No evaluation image resizing is permitted')
                rows.append(dict(name=camera['name'],**image_metrics(rgb,target,shared[i],criterion,'cpu'),
                    eligible_pixels=int(eligible[i].sum()),evaluation_pixels=int(shared[i].sum()),own_pixels=int(own.sum()),
                    mesh_coverage=float(own.sum()/max(1,eligible[i].sum())),
                    common_coverage=float(shared[i].sum()/max(1,eligible[i].sum()))))
            geometry=surface_metrics(prediction_points(cameras,depths[method],selected,obs),reference[obs[selected,0]],info['extent']) if len(selected) else None
            result=dict(complete=True,method=method,label=LABELS[method],state=state,evaluation_views=13,
                scored_views=sum(r['evaluation_pixels']>0 for r in rows),per_view=rows,
                mean=average(rows,['PSNR','SSIM','LPIPS','mesh_coverage','common_coverage']),geometry=geometry,
                geometry_reference_observations=len(selected),geometry_eligible_reference_points=len(reference),
                renderer=sources[state][method]['backend'],gaussian_rgb_substitution=False,appearance_field_fitted=False,
                reference_warning='Visible-surface consistency against actually observed sparse COLMAP SfM points, not an independent dense scan or full-surface completeness')
            results[state+'/'+method]=result;write(output/'metrics'/state/(method+'.json'),result)
            print('DEPTH_TETRA_MESH_METRICS',state,method,result['mean'],'reference_points',len(selected),flush=True)
            gc.collect()
        write(folder/'complete.json',dict(complete=True,methods=METHODS,evaluation_views=13,reference_observations=len(selected),
            selection='Shared positive finite triangle depth; background excludes target mask; actual feature-track observations; no error-based point filtering'))
    return results


def reports(output,results):
    output=Path(output);folder=output/'reports';folder.mkdir(exist_ok=True)
    for state in STATES:
        lines=['# bag：'+('纯背景（目标 mask 内不评分）' if state=='background' else '完整背景与固定生成物体组合'),'',
            '同一 13 个原始留出相机与像素尺寸；四基线使用各自项目的原生网格 RGB/深度，PaintMesh 与本次 mesh 使用真实三角形渲染。原生顶点色/现有物体纹理，无高斯 RGB 替代、无颜色场拟合、无新 UV 烘焙。','',
            '所有分数重新在六个方法的共同正深度覆盖域计算；背景排除相同目标 mask。自身覆盖率同时报告，不能用丢失表面来解释为完整度提高。','',
            '几何参考为 COLMAP 实际特征轨迹中的稀疏 SfM 点，并非独立扫描真值。CD-L1 为双向平均最近距离半和；F1 阈值使用共同相机半径的 0.5%、1%、2%，单位为 COLMAP 场景单位。','',
            '| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 自身覆盖率 ↑ | 共同覆盖率 | CD-L1 ↓ | F1@0.5% ↑ | F1@1% ↑ | F1@2% ↑ | 参考点数 | 评分视角/全部 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for method in METHODS:
            r=results[state+'/'+method];a=r['mean'];g=r['geometry']
            values=[a[key] for key in ['PSNR','SSIM','LPIPS','mesh_coverage','common_coverage']]
            values+=[g['CD_L1'] if g else None,*[g['F_scores'][key]['F1'] if g else None for key in ['0.005','0.01','0.02']]]
            lines.append('| '+' | '.join([LABELS[method],*['—' if v is None else f'{v:.6f}' for v in values],
                str(r['geometry_reference_observations']),f'{r["scored_views"]}/13'])+' |')
        lines+=['','[全部 26 张完整对比图](../figures/README.md) · [逐视角与几何数据](../results.json)','',
            '零共同像素的视角保留并标记无评分域；不算作满分，覆盖率仍计入全部 13 视角。正式项目输出未修改。']
        (folder/(state+'.md')).write_text('\n'.join(lines)+'\n')


def run(output,render_only=False,score_only=False):
    output=Path(output).absolute()
    if not output.resolve().is_relative_to((PUBLIC/'experiments').resolve()):
        raise ValueError('Evaluation may write only an isolated directory inside mesh_comparison_whole/experiments')
    output.mkdir(parents=True,exist_ok=True);configure_sources(ROOT)
    # A stopped/failed rerun must never leave an old global completion flag.
    (output/'evaluation_complete.json').unlink(missing_ok=True)
    (output/'results.json').unlink(missing_ok=True)
    cameras=evaluation_cameras()
    if not score_only:render_experiment(output,cameras)
    sources,stamps=native_sources(output,cameras)
    plan=read(output/'plan.json');graph_path=PUBLIC/'recomposition/bag/composition/scene.json';graph=read(graph_path)
    if plan['source_composition_identity']!=identity(graph_path) or plan['fixed_generated_objects']!=graph['objects']:
        raise ValueError('Composition or generated objects differ from the frozen experiment plan')
    request=dict(implementation=identity(Path(__file__)),mesh=identity(output/'mesh/background.ply'),plan=identity(output/'plan.json'),
        rendered=identity(output/'renders/complete.json'),scene=identity(PUBLIC/'recomposition/bag/composition/scene.json'),
        methods=METHODS,cameras=cameras,inputs=stamps,full_scene=True,additional_downsampling=False,
        production_pipeline_modified=False,appearance='Native vertex colors; no new UV bake')
    # Early preview is an explicit deliverable; metric failures never hide it.
    status(output,'publishing all full-frame comparison images',evaluation_views=13)
    figures(output,cameras,sources)
    write(output/'sources.json',request)
    (output/'README.md').write_text('# bag depth-anchored tetrahedral 全图实验\n\n[26 张完整视角对比图](figures/README.md) · [评估实时状态](evaluation_status.json) · [输入来源](sources.json)\n\n图片已生成；最终指标尚未完成。原生顶点色与现有物体纹理，本阶段未做新的 UV 烘焙。正式项目流程未修改。\n')
    if render_only:
        status(output,'all 26 comparison images ready; metric scoring not requested',images_complete=True,metrics_complete=False)
        return
    results=score(output,cameras,sources)
    for stamp in stamps:
        if identity(stamp['path'])!=stamp:raise ValueError('A render input changed during evaluation: '+stamp['path'])
    if (identity(output/'mesh/background.ply')!=request['mesh'] or identity(PUBLIC/'recomposition/bag/composition/scene.json')!=request['scene']
        or identity(output/'plan.json')!=request['plan']):
        raise ValueError('Experimental mesh or fixed object scene changed during evaluation')
    figures(output,cameras,sources,results);reports(output,results)
    result=dict(complete=True,updated=time.time(),request=request,methods=METHODS,results=results,
        evaluation_views=13,states=STATES,full_scene=True,figure_count=26,production_pipeline_modified=False,
        mesh_decimation=False,additional_image_downsampling=False,uv_baking_performed=False)
    write(output/'results.json',result)
    write(output/'evaluation_complete.json',dict(complete=True,request_id=digest(request),updated=time.time(),evaluation_views=13,figures=26))
    write(output/'evaluation_status.json',dict(complete=True,stage='complete',updated=time.time(),evaluation_views=13,figures=26))
    (output/'README.md').write_text('# bag depth-anchored tetrahedral 全图实验\n\n[26 张完整视角对比图](figures/README.md) · [纯背景报告](reports/background.md) · [组合报告](reports/composed.md) · [逐视角数据](results.json) · [输入来源](sources.json)\n\n完整 mesh 的真实三角形渲染；四个基线保留各自原生网格后端。新旧 PaintMesh 使用存储的原生顶点色与生成物体现有纹理，本阶段未做新的 UV 烘焙。正式项目流程未修改。\n\n几何参考为实际观测的 COLMAP 稀疏点，并非独立扫描真值。指标只比较共同可见区域，同时报告覆盖率。\n')
    print('BAG_DEPTH_TETRA_EVALUATION_COMPLETE',output,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--render-only',action='store_true');parser.add_argument('--score-only',action='store_true')
    args=parser.parse_args()
    if args.render_only and args.score_only:parser.error('Choose at most one partial stage')
    run(args.output,args.render_only,args.score_only)
