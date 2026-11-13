#!/usr/bin/env python3
"""Publish the completed geometry stage without relabeling old RGB renderings."""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw,ImageOps

from mesh_benchmark.core import BASE,SCENES,METHODS,STATES,read,write
from mesh_benchmark.layout import work_root,public_root,configure_sources
from mesh_benchmark.report import LABEL


def publish(output):
    root=work_root(output);configure_sources(root);public=public_root(root)
    summary=[]
    for state in STATES:
        title='纯背景 mesh（排除目标 mask）' if state=='background' else '完整背景 + 最新 MV-SAM3D 物体 mesh'
        lines=[f'# {title}','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}。','',
          '**最新几何指标已覆盖。PSNR / SSIM / LPIPS 等待确认各方法 RGB 渲染入口，旧 VM-SH 分数不作为本轮结果。**','',
          '几何参考为有真实观测的 COLMAP 稀疏点云，不是扫描真值。仅在所有方法均有三角形表面的真实观测像素处比较；不按误差筛点。CD-L1 为双向最近距离平均的半和，单位为 COLMAP 单位；不进行各方法独立 ICP 或缩放。','',
          'F1 阈值以共同训练相机半径 R 的比例定义，主表为 1%，同时报告 0.5%、2%。只有覆盖范围内的几何一致性，不能解释为全场景完整度。纯背景进一步排除目标 mask。','',
          '| 场景 | 方法 | CD-L1 ↓ | F1@0.5% ↑ | Precision@1% | Recall@1% | F1@1% ↑ | F1@2% ↑ | 共同像素比例 | 自身覆盖率 | 参考点数 | 组合状态 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        metrics={}
        for scene in SCENES:
            support=read(root/'common'/scene/(state+'_support')/'complete.json')
            graph=read(public/'recomposition'/scene/'composition/scene.json')
            for method in METHODS:
                g=read(root/'cache'/method/scene/state/'geometry_metrics.json')
                assert g['scoring_protocol_version']==2 and g['support_id']==support['request_id']
                metrics[method,scene]=g
                common=float(np.mean([r['common_fraction'] for r in support['coverage']]))
                own=float(np.mean([r['own_fraction'][method] for r in support['coverage']]))
                status=graph['status'] if method=='paintmesh' and state=='recomposition' else '几何已评估'
                f=g['F_scores'];values=[g['CD_L1'],f['0.005']['F1'],g['precision'],g['recall'],g['F1'],f['0.02']['F1'],common,own]
                lines.append('| '+' | '.join([scene,LABEL[method],*[f'{x:.6f}' for x in values],str(g['reference_points']),status])+' |')
                summary.append(dict(scene=scene,method=method,state=state,geometry=g,common_coverage=common,own_coverage=own,
                    rgb_status='pending_renderer_protocol_confirmation',source_status=graph['status'] if method=='paintmesh' else None))
        lines+=['','## 全部 11 场景的宏平均','',
          '| 方法 | 场景数 | 归一化 CD-L1 ↓ | F1@0.5% ↑ | F1@1% ↑ | F1@2% ↑ |',
          '|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            selected=[metrics[method,s] for s in SCENES]
            values=[np.mean([g['CD_L1_normalized'] for g in selected])]+[np.mean([g['F_scores'][k]['F1'] for g in selected]) for k in ['0.005','0.01','0.02']]
            lines.append('| '+' | '.join([LABEL[method],'11',*[f'{v:.6f}' for v in values]])+' |')
        lines+=['','## RGB 与图片来源','',
          '2DGS、GOF 常用官方 RGB 入口渲染高斯；CoMe、Gaussian Wrapping 当前原生 PLY 没有颜色。真实三角网格 RGB 与官方高斯 RGB 属于不同表示，需确认后再填入三项图像指标。',
          '', '[渲染器核对](../docs/RENDERER_AUDIT.zh-CN.md) · [同步进度](../docs/SYNC_LATEST.md) · [最新几何图](../figures_all_methods/geometry_latest/README.md)',
          '', '已有 RGB 总图和统一颜色场结果仍属于旧协议；不能把它们解释为最新官方渲染器结果。当前 needs_review 是生成物体的来源状态，不因完成几何评分而转为验收通过。']
        (public/'reports'/(state+'.md')).write_text('\n'.join(lines)+'\n')
        target=public/'results'/state/'report.md'
        # The public copy lives one directory deeper than reports/.
        text='\n'.join(lines).replace('](../docs/','](../../docs/').replace('](../figures_all_methods/','](../../figures_all_methods/')
        target.write_text(text+'\n')
    write(root/'latest_geometry_results.json',dict(updated=time.time(),scene_count=11,mode_metric_count=len(summary),rgb_complete=False,records=summary))
    return summary


def figures(output,scenes=None):
    from plot_mesh_method_comparison import (choose_view,reference_depth,sources,geometry_image,
        overview,save,font,COLS)
    root=work_root(output);configure_sources(root);public=public_root(root)
    out=public/'figures_all_methods';dest=out/'geometry_latest';dest.mkdir(exist_ok=True)
    requested=set(SCENES if scenes is None else scenes)
    if not requested or not requested.issubset(SCENES):raise ValueError(f'Unknown or empty scene selection: {scenes}')
    manifest=dest/'manifest.json'
    previous=read(manifest) if manifest.exists() else {}
    selections={};records=[r for r in previous.get('subplots',[]) if r['scene'] not in requested]
    for scene in SCENES:
        print('LATEST_GEOMETRY_FIGURE',scene,flush=True)
        camera,roi,area=choose_view(root,scene)
        _,scale=reference_depth(root,scene,camera,'recomposition')
        selections[scene]=dict(camera=camera,roi=roi,target_fraction=area,depth_range=scale.tolist())
        if scene not in requested and manifest.exists():continue
        width=camera['width'];height=camera['height']
        zoom_height=max(240,round((roi[3]-roi[1])*width/max(1,roi[2]-roi[0])))
        sheet=Image.new('RGB',(width*len(COLS),height+zoom_height+140),'white');draw=ImageDraw.Draw(sheet)
        draw.text((10,8),f'{scene} | latest actual triangle geometry | {camera["name"]}',fill='black',font=font(23))
        draw.text((10,40),'Common depth color range; CD/F1 are scene-level. Native-resolution original mesh depths.',fill='black',font=font(17))
        for i,(label,method,state) in enumerate(COLS):
            rec=sources(root,scene,label,method,state,camera)
            image=geometry_image(root,out,scene,rec,camera,scale)
            assert image is not None
            geometry=rec['geometry'];x=i*width
            draw.text((x+8,70),label,fill='black',font=font(19))
            full=ImageOps.contain(image,(width,height),Image.Resampling.LANCZOS)
            sheet.paste(full,(x,100));crop=ImageOps.contain(image.crop(roi),(width,zoom_height),Image.Resampling.LANCZOS)
            sheet.paste(crop,(x,100+height))
            if geometry:draw.text((x+8,105+height+zoom_height),f'CD {geometry["CD_L1"]:.5f}  F1 {geometry["F1"]:.4f}',fill='black',font=font(18))
        save(sheet,dest/(scene+'_all_methods'),formats=('png','jpg','pdf'))
        # Each method has a separate geometry image for each evaluation mode.
        for state in STATES:
            for method in [None,*METHODS]:
                rec=sources(root,scene,'GT' if method is None else LABEL[method],method,state,camera)
                image=geometry_image(root,out,scene,rec,camera,scale,reference_state=state)
                folder=out/'subplots'/(method or 'gt')/state/scene;folder.mkdir(parents=True,exist_ok=True)
                save(image,folder/'geometry',formats=('png',))
                save(image.crop(roi),folder/'geometry_zoom',formats=('png',))
                write(folder/'geometry_source.json',dict(updated=time.time(),camera=camera['name'],roi=roi,
                     depth_source=str(rec.get('depth') or root/'common'/scene/'reference_points.npz'),
                     geometry=rec['geometry'],description='Latest actual mesh depth; no Gaussian or RGB substitution'))
                records.append(dict(scene=scene,method=method or 'gt',state=state,folder=str(folder)))
    image=overview(root,out,SCENES,selections,'geometry')
    draw=ImageDraw.Draw(image)
    draw.rectangle((0,image.height-44,image.width,image.height),fill='white')
    draw.text((16,image.height-35),'Actual triangle geometry and sparse COLMAP reference; native RGB metrics are in the two main reports.',fill='black',font=font(20))
    save(image,out/'overview/all_scenes_geometry',formats=('png','jpg','pdf'))
    write(dest/'manifest.json',dict(updated=time.time(),scenes=selections,subplots=records,
        columns=[x[0] for x in COLS],native_resolution=True,
        rgb_complete=read(root/'protocol.json').get('native_mesh_rgb',{}).get('enabled',False)))
    lines=['# 最新 mesh 几何对比图','',
        '11 个场景，GT 稀疏参考、四个基线、实际 PaintMesh 第一阶段mesh、完整背景及组合两种模式。保留原生图像尺寸；图像是真实三角网格深度的统一伪彩色几何诊断，不是 RGB 渲染分数。','',
        '[全场景总图](../overview/all_scenes_geometry.jpg)','',
        '各方法的 `subplots/<method>/<mode>/<scene>/geometry.png` 和 `geometry_zoom.png` 已更新；RGB 子图、两份指标报告使用当前各项目原生网格后端。第一阶段保留原物体，背景模式仍排除相同目标mask。','']
    lines += [f'- [{s}]({s}_all_methods.jpg)' for s in SCENES]
    (dest/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE/'mesh_comparison_whole')
    parser.add_argument('--skip-figures',action='store_true');args=parser.parse_args()
    publish(args.output)
    if not args.skip_figures:figures(args.output)
