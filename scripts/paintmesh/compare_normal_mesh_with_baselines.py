#!/usr/bin/env python3
"""Compare the isolated bag experiment with existing native baseline renders.

No production file is written. Original native pixels/depths are reused only
after matching every camera and the complete mesh-source fingerprint. All
scores are recalculated on a NEW support shared by all seven methods.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time


def wait_for_experiment(output):
    """Wait without importing the heavy scoring stack or taking a GPU."""
    output=Path(output)
    while True:
        path=output/'results.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row.get('complete') and row.get('evaluation_views')==13:return
        time.sleep(30)


def assert_same_cameras(actual,expected):
    import numpy as np
    by_name={c['name']:c for c in actual}
    if len(by_name)!=len(actual) or set(by_name)!={c['name'] for c in expected}:
        raise ValueError('Native renderer camera set differs from experiment')
    for camera in expected:
        other=by_name[camera['name']]
        for key in ['width','height','stem']:
            if other[key]!=camera[key]:raise ValueError('Native image dimensions/name changed')
        for key in ['K','w2c']:
            if not np.array_equal(other[key],camera[key]):raise ValueError('Native camera calibration differs')


def shared_support(depths,foreground,background):
    import numpy as np
    eligible=~foreground if background else np.ones(foreground.shape,dtype=bool)
    valid=eligible.copy()
    for depth in depths:
        if depth.shape!=eligible.shape:raise ValueError('Native depth resolution changed')
        valid&=np.isfinite(depth)&(depth>0)
    return eligible,valid


def run(output):
    import numpy as np
    import torch,lpips
    from PIL import Image,ImageDraw
    from evaluate_normal_mesh_experiment import (PUBLIC,ROOT,eval_cameras,render_folder,
        validate_methods,average_rows,LABELS,output_files)
    from evaluate_native_mesh_rgb import verified_file_fingerprint
    from mesh_benchmark.core import read,write,identity,digest,surface_metrics
    from mesh_benchmark.layout import configure_sources
    from mesh_benchmark.overlap import observations,select_anchors,prediction_points
    from mesh_benchmark.score import image_metrics
    from plot_mesh_method_comparison import font

    output=Path(output);validate_methods(output);configure_sources(ROOT)
    if not read(output/'results.json').get('complete'):raise ValueError('Experiment still incomplete')
    cs=eval_cameras()
    if len(cs)!=13:raise ValueError('All 13 held-out views are required')
    baselines=['2dgs','come','gof','gaussian_wrapping'];methods=[*baselines,'A','N','R']
    labels=dict(zip(baselines,['2DGS','CoMe','GOF','Gaussian Wrapping']));labels.update(LABELS)
    native={};native_identities={}
    for method in baselines:
        folder=ROOT/'native_renderers'/method/'bag';meta=read(folder/'complete.json')
        if not meta.get('complete') or meta.get('pilot') or meta.get('views')!=13:
            raise ValueError('Native baseline rendering is incomplete: '+method)
        if not meta.get('all_source_triangles_preserved') or meta['request'].get('gaussian_renderer_used'):
            raise ValueError('Actual complete colored triangle rendering is required')
        assert_same_cameras(meta['request']['cameras'],cs)
        expected=meta['request']['mesh']
        if verified_file_fingerprint(ROOT,expected['path'])!=expected:
            raise ValueError('Native mesh source differs from its renderer receipt')
        native[method]=folder
        native_identities[method]=identity(folder/'complete.json')
    destination=output/'all_methods';destination.mkdir(exist_ok=True)
    request=dict(implementation=identity(Path(__file__)),experiment=identity(output/'results.json'),
        native_receipts=native_identities,cameras=digest(cs),methods=methods,full_scene=True,
        production_modified=False,protocol='All methods: exact shared positive finite depth; target mask excluded for background; actual COLMAP track observations')
    request_id=digest(request)
    receipt=destination/'complete.json'
    if receipt.exists():
        if read(receipt)['request']!=request:raise ValueError('All-method comparison inputs changed')
        return
    write(destination/'status.json',dict(stage='rescoring all native methods',pid=os.getpid(),updated=time.time(),request=request))
    torch.set_num_threads(4)
    criterion=lpips.LPIPS(net='vgg',version='0.1',spatial=True).cpu().eval()
    for p in criterion.parameters():p.requires_grad_(False)
    info=read(ROOT/'common/bag/complete.json');results={};inventory=[];source_files=[]
    for state in ['background','composed']:
        folders={m:render_folder(output,m,state) for m in ['A','N','R']}
        depths={};images={}
        for method in methods:
            if method in native:
                dp=[native[method]/'depth'/(c['stem']+'.npy') for c in cs]
                rgb=[native[method]/'renders'/(c['stem']+'.png') for c in cs]
            else:
                meta=read(folders[method]/'complete.json')
                if not meta.get('complete') or meta['outputs']!=output_files(folders[method],cs):
                    raise ValueError('Experimental rendering changed')
                dp=[folders[method]/(c['stem']+'_depth.npy') for c in cs]
                rgb=[folders[method]/(c['stem']+'.png') for c in cs]
            source_files.extend(identity(p) for p in [*dp,*rgb])
            depths[method]=[np.load(p,mmap_mode='r') for p in dp];images[method]=rgb
        allowed=[];shared=[];support=destination/'support'/state;support.mkdir(parents=True,exist_ok=True)
        for i,c in enumerate(cs):
            foreground=np.asarray(Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_foreground.png')))>127
            eligible,mask=shared_support([depths[m][i] for m in methods],foreground,state=='background')
            allowed.append(eligible);shared.append(mask)
            Image.fromarray(mask.astype('uint8')*255).save(support/(c['stem']+'.png'))
        with np.load(ROOT/'common/bag/reference_points.npz') as a:
            reference=a['background' if state=='background' else 'recomposition']
        obs=observations('bag',cs,reference)
        chosen=select_anchors(obs[:,0],obs[:,1],obs[:,2],shared,np.ones(len(reference),dtype=bool),42,
            max_points=max(1,len(reference))) if len(obs) else np.empty(0,dtype='int64')
        np.savez(support/'geometry_reference.npz',points=reference[obs[chosen,0]],observations=obs[chosen])
        for method in methods:
            rows=[]
            for i,c in enumerate(cs):
                target=np.asarray(Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_gt.png')).convert('RGB'))
                rgb=np.asarray(Image.open(images[method][i]).convert('RGB'))
                if rgb.shape!=target.shape or rgb.shape[:2]!=depths[method][i].shape:
                    raise ValueError('Full native image resolution is required')
                own=allowed[i]&np.isfinite(depths[method][i])&(depths[method][i]>0)
                rows.append(dict(name=c['name'],**image_metrics(rgb,target,shared[i],criterion,'cpu'),
                    mesh_coverage=float(own.sum()/max(1,allowed[i].sum())),
                    common_coverage=float(shared[i].sum()/max(1,allowed[i].sum())),
                    evaluation_pixels=int(shared[i].sum())))
            geometry=surface_metrics(prediction_points(cs,depths[method],chosen,obs),reference[obs[chosen,0]],info['extent']) if len(chosen) else None
            result=dict(method=method,label=labels[method],state=state,geometry=geometry,per_view=rows,
                mean=average_rows(rows,['PSNR','SSIM','LPIPS','mesh_coverage','common_coverage']),
                evaluation_views=13,scored_views=sum(r['evaluation_pixels']>0 for r in rows),
                geometry_reference_observations=len(chosen),reference_eligible_points=len(reference))
            results[state+'/'+method]=result
            write(destination/'metrics'/state/(method+'.json'),result)
            print('ALL_METHOD_BAG_METRICS',state,method,result['mean'],flush=True)
        figdir=destination/'figures'/state;figdir.mkdir(parents=True,exist_ok=True)
        for i,c in enumerate(cs):
            width,height=c['width'],c['height'];cellh=height+70
            canvas=Image.new('RGB',(4*width,2*cellh+50),(247,248,250));draw=ImageDraw.Draw(canvas)
            draw.text((10,8),'bag | '+c['name']+' | '+state+' | complete native-size mesh views',font=font(20),fill='black')
            for j,method in enumerate(['GT',*methods]):
                x=(j%4)*width;y=(j//4)*cellh+50
                src=ROOT/'common/bag/held_out'/(c['stem']+'_gt.png') if method=='GT' else images[method][i]
                pic=Image.open(src).convert('RGB')
                if pic.size!=(width,height):raise ValueError('Comparison panels cannot resize native frames')
                draw.text((x+7,y+3),'GT' if method=='GT' else labels[method],font=font(17),fill='black')
                canvas.paste(pic,(x,y+25))
                if method!='GT':
                    row=results[state+'/'+method]['per_view'][i]
                    text='no common pixels' if row['PSNR'] is None else f'P {row["PSNR"]:.2f}  S {row["SSIM"]:.3f}  L {row["LPIPS"]:.3f}'
                    draw.text((x+7,y+height+30),text+f' | cov {row["mesh_coverage"]:.1%}',font=font(16),fill='black')
                sub=destination/'subplots'/method/state;sub.mkdir(parents=True,exist_ok=True)
                pic.save(sub/(c['stem']+'.png'))
            path=figdir/(c['stem']+'.png');canvas.save(path)
            inventory.append(dict(camera=c['name'],state=state,path=str(path),native_panel_size=[width,height]))
        lines=['# bag 全方法：'+state,'',
            '四个基线复用各自项目网格渲染器已完成的原生 RGB 与深度；A/N/R 是本次完整实验网格的真实三角形渲染。所有方法重新在同一覆盖交集评分，不混用旧表分数。背景目标 mask 内不计分。',
            '', 'COLMAP 稀疏点仅为观测几何参考，并非独立扫描真值。CD/F1 只在实际特征轨迹所对应的共同可见表面计算，同时保留覆盖率。F1 阈值为相机半径的 0.5%、1%、2%。', '',
            '| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 覆盖率 ↑ | 共同覆盖率 | CD-L1 ↓ | F1@0.5% ↑ | F1@1% ↑ | F1@2% ↑ | 几何参考点 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for method in methods:
            r=results[state+'/'+method];a=r['mean'];g=r['geometry']
            values=[a[k] for k in ['PSNR','SSIM','LPIPS','mesh_coverage','common_coverage']]
            values+=[g['CD_L1'] if g else None,*[g['F_scores'][k]['F1'] if g else None for k in ['0.005','0.01','0.02']]]
            lines.append('| '+' | '.join([labels[method],*['—' if v is None else f'{v:.6f}' for v in values],str(r['geometry_reference_observations'])])+' |')
        lines+=['','[全部整图与独立子图](README.md) · [逐视角数据](results.json)']
        (destination/(state+'.md')).write_text('\n'.join(lines)+'\n')
    for stamp in source_files:
        if identity(stamp['path'])!=stamp:raise ValueError('A render source changed during comparison')
    for method,stamp in native_identities.items():
        if identity(native[method]/'complete.json')!=stamp:raise ValueError('Native rendering was replaced during scoring')
    write(destination/'results.json',dict(complete=True,request=request,methods=methods,results=results,evaluation_views=13))
    write(destination/'figures/manifest.json',dict(complete=True,records=inventory,all_pixels_preserved=True))
    lines=['# bag：四个基线与本次全图实验','',
        '[纯背景指标](background.md) · [组合指标](composed.md) · [逐视角数据](results.json)','',
        '每个子图保持原始宽高；每张含 GT、2DGS、CoMe、GOF、Wrapping，以及 A/N/R 三种完整 mesh。无缩图、无高斯 RGB 替代。保留全部 13 个留出视角，不按指标筛图。','',
        '| 相机 | 纯背景 | 组合 |','|---|---|---|']
    for c in cs:lines.append(f'| {c["name"]} | [PNG](figures/background/{c["stem"]}.png) | [PNG](figures/composed/{c["stem"]}.png) |')
    lines+=['','各方法原尺寸独立图片位于 `subplots/<method>/<state>/`。']
    (destination/'README.md').write_text('\n'.join(lines)+'\n')
    with (output/'README.md').open('a') as handle:handle.write('\n[四个基线＋本次全图实验：统一指标与完整对比图](all_methods/README.md)\n')
    write(receipt,dict(complete=True,request=request,request_id=request_id,updated=time.time(),evaluation_views=13,figures=len(inventory)))
    write(destination/'status.json',dict(complete=True,stage='complete',updated=time.time(),figures=len(inventory)))
    print('BAG_NORMAL_ALL_METHODS_COMPLETE',destination,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.wait:wait_for_experiment(args.output)
    sys.path.insert(0,str(Path(__file__).resolve().parent));run(args.output)
