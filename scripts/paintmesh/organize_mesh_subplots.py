#!/usr/bin/env python3
"""Export independent method panels, lossless RGB/geometry crops and provenance.

The source figures and evaluator remain unchanged. The organizer follows their
atomic per-scene manifests, and never treats missing output as a completed render.
"""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image,ImageDraw

from plot_mesh_method_comparison import (SCENES,METHODS,BASE,read,write,identity,digest,
    save,text,masked_image,input_signature)

from mesh_benchmark.layout import work_root,figure_bucket

MODES=['background','recomposition']
LABELS={'gt':'GT reference','2dgs':'2DGS','come':'CoMe','gof':'GOF',
        'gaussian_wrapping':'Gaussian Wrapping','paintmesh_stage1':'PaintMesh stage 1','paintmesh':'PaintMesh'}
FILES=['rgb.png','rgb_zoom.png','rgb_scored.png','rgb_scored_zoom.png',
       'geometry.png','geometry_zoom.png','geometry_scored.png','evaluation_mask.png',
       'target_mask.png','coverage_mask.png','geometry_evaluation_mask.png','panel.png']

def copy_atomic(source,dest):
    if Path(source).resolve()==Path(dest).resolve():raise ValueError('Cannot overwrite a source artifact')
    tmp=dest.with_name('.'+dest.name+'.tmp');shutil.copyfile(source,tmp);tmp.replace(dest)

def export_method(root,figures,out,scene,mode,method,info):
    camera=info['selection']['camera'];roi=info['selection']['roi'];stem=camera['stem']
    is_gt=method=='gt';m=None if is_gt else method
    dest=out/method/mode/scene;dest.mkdir(parents=True,exist_ok=True)
    rgb=root/'common'/scene/'held_out'/(stem+'_gt.png') if is_gt else root/mode/method/scene/'renders'/(stem+'.png')
    geometry=figures/'assets'/scene/mode/((method if not is_gt else 'reference')+'_geometry.png')
    geometry_evaluation=root/'common'/scene/(mode+'_support')/(stem+'.png')
    native_evaluation=root/'common'/scene/(mode+'_native_support')/(stem+'.png')
    evaluation=native_evaluation if native_evaluation.exists() else geometry_evaluation
    target=root/'common'/scene/'held_out'/(stem+'_foreground.png')
    depth=root/'cache'/method/scene/mode/'depth'/(stem+'.npy')
    if not is_gt and not depth.exists():depth=figures/'assets'/scene/'extra'/(method+'_depth.npy')
    metrics_file=root/mode/method/scene/'metrics.json'
    metrics=read(metrics_file) if not is_gt and metrics_file.exists() else None
    rgb_depth=depth
    if metrics and metrics.get('native_mesh_rendering_version'):
        from evaluate_native_mesh_rgb import rendering
        rgb_depth=rendering(root,method,scene,mode)['depth']/(stem+'.npy')
    geometry_metrics=root/'cache'/method/scene/mode/'geometry_metrics.json'
    sheet=figures/'scenes'/scene/(mode+'.png')
    inputs=[rgb,geometry,evaluation,geometry_evaluation,target,sheet,figures/'scenes'/scene/'figure_manifest.json',Path(__file__)]
    if not is_gt:inputs += [depth,rgb_depth,metrics_file,geometry_metrics]
    sig=digest(dict(camera=camera,roi=roi,inputs=[identity(p) if p.exists() else dict(path=str(p),missing=True) for p in inputs]))
    metadata=dest/'source.json'
    if metadata.exists() and read(metadata).get('signature')==sig and (dest/'panel.png').exists():return read(metadata),False
    produced=set()
    if metrics and metrics.get('scoring_protocol_version')!=2:metrics=None
    geo=read(geometry_metrics) if not is_gt and geometry_metrics.exists() else None
    if geo and geo.get('scoring_protocol_version')!=2:geo=None
    view=next((v for v in metrics['per_view'] if v['name']==camera['name']),None) if metrics else None
    status='reference' if is_gt else ('complete' if metrics and rgb.exists() else ('RGB rendered; metrics pending' if rgb.exists() else ('geometry only; RGB pending' if geometry.exists() else 'no completed mesh/evaluation')))
    def copy(source,name):
        if source.exists():copy_atomic(source,dest/name);produced.add(name)
    for source,name in [(rgb,'rgb.png'),(geometry,'geometry.png'),(evaluation,'evaluation_mask.png'),(geometry_evaluation,'geometry_evaluation_mask.png'),(target,'target_mask.png')]:copy(source,name)
    for source,prefix in [(rgb,'rgb'),(geometry,'geometry')]:
        if not source.exists():continue
        mask=geometry_evaluation if prefix=='geometry' else evaluation
        valid=np.asarray(Image.open(mask))>127 if mask.exists() else None
        with Image.open(source) as image:
            if image.size!=(camera['width'],camera['height']):raise ValueError(f'Camera/image size mismatch: {source}')
            save(image.crop(roi),dest/(prefix+'_zoom'));produced.add(prefix+'_zoom.png')
            if valid is not None:
                scored=masked_image(image.convert('RGB'),valid);save(scored,dest/(prefix+'_scored'));produced.add(prefix+'_scored.png')
                if prefix=='rgb':save(scored.crop(roi),dest/'rgb_scored_zoom');produced.add('rgb_scored_zoom.png')
    if not is_gt and rgb_depth.exists():
        d=np.load(rgb_depth,mmap_mode='r');coverage=np.isfinite(d)&(d>0)
        if mode=='background':coverage&=~(np.asarray(Image.open(target))>127)
        save(Image.fromarray(coverage.astype('uint8')*255),dest/'coverage_mask');produced.add('coverage_mask.png')
    # Literal column extraction preserves the exact scores and gray-mask domain
    # of the corresponding paired comparison figure.
    with Image.open(sheet) as image:
        columns=len(METHODS)+1
        if image.width%columns:raise ValueError(f'Unexpected paired figure layout: {sheet}')
        width=image.width//columns;column=0 if is_gt else METHODS.index(method)+1
        box=[column*width,148,(column+1)*width,image.height-45]
        body=image.crop(box).convert('RGB')
    panel=Image.new('RGB',(width,body.height+138),(247,248,250));draw=ImageDraw.Draw(panel)
    text(draw,(12,10),f'{LABELS[method]} | {scene}',size=26,bold=True)
    text(draw,(12,47),f'{mode} | camera {camera["name"]}',size=20)
    text(draw,(12,77),'RGB: selected view. CD/F1: scene summary.',size=18)
    panel.paste(body,(0,102))
    text(draw,(12,panel.height-26),'Gray = excluded scoring pixels. Same camera/crop as the comparison.',size=15)
    save(panel,dest/'panel');produced.add('panel.png')
    for name in FILES:
        if name not in produced and (dest/name).exists():(dest/name).unlink()
    record=dict(signature=sig,method=method,mode=mode,scene=scene,status=status,
                original_image=camera['name'],camera=camera,roi_xyxy=roi,
                source_rgb=str(rgb) if rgb.exists() else None,source_geometry=str(geometry) if geometry.exists() else None,
                source_depth=str(depth) if not is_gt and depth.exists() else None,
                source_rgb_depth=str(rgb_depth) if not is_gt and rgb_depth.exists() else None,
                source_comparison=str(sheet),source_panel_crop=box,source_evaluation_mask=str(evaluation),
                source_geometry_evaluation_mask=str(geometry_evaluation),
                source_foreground_mask=str(target),generated_files=sorted(produced),
                geometry_postprocessing=metrics.get('geometry_postprocessing') if metrics else None,
                view_metrics=view,scene_geometry=geo,metrics_scope='RGB scores: chosen view; geometry: entire scene; common covered region')
    write(metadata,record)
    write(dest/'metrics.json',dict(method=method,mode=mode,scene=scene,view=camera['name'],status=status,
                                  selected_view=view,scene_geometry=geo,
                                  scene_mean=metrics['mean'] if metrics else None))
    (dest/'README.md').write_text(f'# {LABELS[method]} / {mode} / {scene}\n\n相机：{camera["name"]}。状态：{status}。\n\n'
      '- [独立完整子图](panel.png)：与该模式的对比图同一列，含分数和放大图。\n'
      '- `rgb.png`：原尺寸、无标注的真实 mesh 渲染（GT 目录为实拍参考图）；`rgb_zoom.png`：原始像素裁剪，无拉伸。\n'
      '- `geometry.png` / `geometry_zoom.png`：相同视角的表面预览（GT 为稀疏参考点云）。\n'
      '- `rgb_scored.png` / `rgb_scored_zoom.png` / `geometry_scored.png`：与主评分同域，排除区域显示为灰色。\n'
      '- `evaluation_mask.png`：RGB 共同评分域；`geometry_evaluation_mask.png`：固定几何支持域；`coverage_mask.png`：RGB 自身有效域；`target_mask.png`：目标区域。白色表示该 mask 为真。\n'
      '- `metrics.json` / `source.json`：所选视角分数、场景几何分数、原始文件、相机与裁剪坐标。\n\n'
      '仅导出当前存在的原始数据；缺失 RGB 或 mesh 时，panel.png 中保留明确状态，不制造假结果。\n')
    return record,True

def indexes(out,records,waiting):
    lookup={(r['method'],r['mode'],r['scene']):r for r in records}
    lines=['# 每个方法的独立子图','',
       '目录按 **方法 / 评估模式 / 场景** 整理。背景与组合模式分别保存，各场景采用与原总图完全相同的相机和裁剪框。','',
       '| 方法 | 场景索引 | 背景模式 | 组合模式 |','|---|---|---|---|']
    for method,label in LABELS.items():
        lines.append(f'| {label} | [逐场景索引]({method}/README.md) | [{method}/background/]({method}/background/) | [{method}/recomposition/]({method}/recomposition/) |')
        page=[f'# {label} 子图','', '| 场景 | 纯背景子图 | 状态 | 组合子图 | 状态 |','|---|---|---|---|---|']
        for scene in SCENES:
            values=[scene]
            for mode in MODES:
                record=lookup.get((method,mode,scene))
                values += [f'[panel]({mode}/{scene}/panel.png) · [目录说明]({mode}/{scene}/README.md)' if record else '等待总图生成',record['status'] if record else 'pending']
            page.append('| '+' | '.join(values)+' |')
        path=out/method/'README.md';path.parent.mkdir(exist_ok=True);path.write_text('\n'.join(page)+'\n')
    lines+=['','## 每个场景目录中的文件','',
       '| 文件 | 内容 |','|---|---|',
       '| `panel.png` | 从相应模式对比图逐列提取的完整子图，含指标、放大和表面预览 |',
       '| `rgb.png` / `rgb_zoom.png` | 无标注 RGB 原图及原始像素裁剪，可直接用于排版 |',
       '| `geometry.png` / `geometry_zoom.png` | 表面几何预览及相同 ROI 裁剪 |',
       '| `rgb_scored.png` / `rgb_scored_zoom.png` / `geometry_scored.png` | 无效区域显示为灰色的同评分区域版本 |',
       '| `evaluation_mask.png` | 白色为共同评分域 |',
       '| `geometry_evaluation_mask.png` | 白色为固定几何支持域，与 RGB 光栅化支持分别记录 |',
       '| `coverage_mask.png` | 白色为该方法自身有效像素；纯背景已排除目标 mask |',
       '| `target_mask.png` | 白色为目标物体区域，背景评分忽略其内部 |',
       '| `metrics.json` | 所选视角图像指标与场景几何指标，分别标明范围 |',
       '| `source.json` | 原图路径、真实相机、裁剪坐标、总图列坐标和数据状态 |','',
       '示例：`paintmesh/background/bag/panel.png`、`paintmesh/recomposition/bag/panel.png`。','',
       '原图使用独立文件副本，没有连接到原数据的可写硬链接。缺失数据只保留状态和带明确标注的 panel，不生成假 RGB。',
       'GT 的几何图是稀疏 COLMAP 点云投影。几何预览由真实网格深度近似照明，不是另一个生成模型。',
       '',f'当前已整理目录：{len(records)}。等待总图同步的场景：'+(', '.join(waiting) if waiting else '无。'),
       '', '[返回总图索引](../README.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')

def build(figures,out):
    root=work_root(figures.parent);changed=0;waiting=[]
    for scene in SCENES:
        marker=figures/'scenes'/scene/'figure_manifest.json'
        if not marker.exists():waiting.append(scene);continue
        info=read(marker);camera=info['selection']['camera']
        # Avoid mixing a newer evaluator mask/score with an older column panel.
        if info['signature']!=input_signature(root,scene,camera):waiting.append(scene);continue
        if not all((figures/'scenes'/scene/(mode+'.png')).exists() for mode in MODES):waiting.append(scene);continue
        for mode in MODES:
            for method in ['gt',*METHODS]:
                _,updated=export_method(root,figures,out,scene,mode,method,info);changed+=updated
        print('SUBPLOTS',scene,'ready',flush=True)
    records=[read(p) for p in sorted(out.glob('*/*/*/source.json'))]
    write(out/'manifest.json',dict(updated=time.time(),directory_count=len(records),changed=changed,
          waiting_scenes=waiting,source_figures=str(figures),methods=list(LABELS),modes=MODES,scenes=SCENES,records=records))
    indexes(out,records,waiting)
    print('EXPORTED',len(records),'directories; changed',changed,'waiting',waiting,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--figures',type=Path,default=BASE/'mesh_comparison/figures_all_methods')
    p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=30)
    args=p.parse_args();figures=args.figures.resolve();out=figures/'subplots';out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'organizer.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'organizer.pid').write_text(str(os.getpid())+'\n')
    while True:
        before=time.time()
        try:build(figures,out)
        except Exception:
            import traceback;traceback.print_exc()
            if not args.watch:raise
        if not args.watch:break
        running=False
        try:
            pid=int((figure_bucket(figures,'.runtime')/'watch.pid').read_text())
            running=b'plot_mesh_method_comparison.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
        except (FileNotFoundError,ValueError):pass
        if not running:
            print('Figure watcher finished; subplots synchronized',flush=True);break
        time.sleep(max(1,args.interval-(time.time()-before)))

if __name__=='__main__':main()
