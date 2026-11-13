#!/usr/bin/env python3
"""Compare old/coarse and fine mesh renders on the exact same valid pixels."""
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,read,write,identity
from mesh_benchmark.layout import work_root
from audit_paintmesh_blur import image_evidence,decoded

PUBLIC=BASE/'mesh_comparison_whole';ROOT=work_root(PUBLIC);WORK=ROOT/'fine_whole'


def before_after_figure(scene,rows):
    """Keep native pixels visible, including missing geometry; no fitted RGB."""
    from plot_mesh_method_comparison import choose_view,font
    camera,_,_=choose_view(ROOT,scene);w,h=camera['width'],camera['height']
    sheet=Image.new('RGB',(2*w,70+len(rows)*(h+55)),(245,247,250));draw=ImageDraw.Draw(sheet)
    draw.text((12,8),f'{scene} | {camera["name"]} | actual mesh RGB, native pixels',font=font(21),fill='black')
    draw.text((12,40),'Previous extraction',font=font(18),fill='black')
    draw.text((w+12,40),'Fine extraction + observed background',font=font(18),fill='black')
    for i,row in enumerate(rows):
        view=next(r for r in row['views'] if r['camera']==camera['name'])
        y=70+i*(h+55)
        for j,key in enumerate(['old','fine']):
            image=Image.open(view['sources'][key]['path']).convert('RGB')
            if image.size!=(w,h):raise ValueError('Before/after figure cannot resize native renders')
            metrics=view[key]
            draw.text((j*w+12,y+8),f'{row["label"]} | matched pixels: P {metrics["PSNR"]:.2f}, S {metrics["SSIM"]:.3f}',
                      font=font(17),fill='black')
            sheet.paste(image,(j*w,y+55))
    path=PUBLIC/'figures_all_methods/scenes'/scene/'fine_before_after.png'
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');sheet.save(temp,format='PNG');temp.replace(path)
    return identity(path)


def compare(scenes):
    destination=PUBLIC/'docs/FINE_WHOLE_EFFECTS.json'
    records=read(destination).get('scenes',{}) if destination.exists() else {}
    for scene in scenes:
        before=WORK/'before'/scene;rows=[]
        for method,state,label in [('paintmesh_stage1','background','stage1'),('paintmesh','background','background'),('paintmesh','recomposition','composed')]:
            metrics=[]
            cameras=[c for c in read(ROOT/'common'/scene/'complete.json')['cameras'] if c['part']=='held_out']
            for c in cameras:
                stem=c['stem'];oldmask=before/'common'/(state+'_native_support')/(stem+'.png')
                newmask=ROOT/'common'/scene/(state+'_native_support')/(stem+'.png')
                old=np.asarray(Image.open(oldmask))>127;new=np.asarray(Image.open(newmask))>127;valid=old&new
                if not valid.any():raise ValueError('No before/after common pixels')
                gt_path=ROOT/'common'/scene/'held_out'/(stem+'_gt.png');target=np.asarray(Image.open(gt_path).convert('RGB'))
                oldpath=before/'cache'/method/state/'native_renders'/(stem+'.png')
                newpath=ROOT/'cache'/method/scene/state/'native_renders'/(stem+'.png')
                a=decoded(oldpath,target.shape);b=decoded(newpath,target.shape)
                aa=image_evidence(a,target,valid);bb=image_evidence(b,target,valid)
                metrics.append(dict(camera=c['name'],pixels=int(valid.sum()),old=aa,fine=bb,
                    sources={k:identity(p) for k,p in [('old',oldpath),('fine',newpath),('GT',gt_path),('old_mask',oldmask),('fine_mask',newmask)]}))
            def average(which,key):return float(np.mean([r[which][key] for r in metrics]))
            rows.append(dict(label=label,method=method,state=state,views=metrics,
                old_PSNR=average('old','PSNR'),fine_PSNR=average('fine','PSNR'),
                old_SSIM=average('old','SSIM'),fine_SSIM=average('fine','SSIM'),
                old_detail_ratio=average('old','gradient_energy_ratio'),fine_detail_ratio=average('fine','gradient_energy_ratio')))
        before_after_figure(scene,rows)
        records[scene]=rows
    write(destination,dict(complete=True,updated=time.time(),scenes=records,
        scope='Exact intersection of pre-fix and fine six-method native coverage masks; no image resampling',
        caveat='Gradient energy is a detail/noise diagnostic, not accuracy. Main LPIPS/CD/F1 are in the two reports.'))
    lines=['# 细网格重提前后：同一像素对比','',
        '主报告共同域会随完整背景扩大而改变。此表单独取更新前后有效域的交集，对同一GT、同一相机、同一像素比较旧/新原生mesh渲染。没有缩图或按误差选像素；不使用3DGS图片替代mesh。','',
        '| 场景 | 模式 | 旧PSNR | 细网格PSNR | 差值 | 旧SSIM | 细网格SSIM |','|---|---|---:|---:|---:|---:|---:|']
    for scene,rows in records.items():
        for r in rows:lines.append(f'| {scene} | {r["label"]} | {r["old_PSNR"]:.4f} | {r["fine_PSNR"]:.4f} | {r["fine_PSNR"]-r["old_PSNR"]:+.4f} | {r["old_SSIM"]:.4f} | {r["fine_SSIM"]:.4f} |')
    lines+=['','## 同视角原尺寸图片','',
        '三行分别为 stage1、background、composed，左右为旧/新提取。直接展示原尺寸mesh像素和缺失区域；标注分数仅用于同像素核对，不与主报告的新共同域混用。','']
    lines += [f'- [{scene}：旧/新完整画面](../figures_all_methods/scenes/{scene}/fine_before_after.png)' for scene in records]
    lines+=['','逐视角来源与有效像素数见 [JSON](FINE_WHOLE_EFFECTS.json)。完整CD/F1与LPIPS见 [背景报告](../reports/background.md)、[组合报告](../reports/recomposition.md)。']
    (PUBLIC/'docs/FINE_WHOLE_EFFECTS.md').write_text('\n'.join(lines)+'\n')
    print('FINE_MATCHED_PIXELS_COMPARISON_COMPLETE',scenes,flush=True)
