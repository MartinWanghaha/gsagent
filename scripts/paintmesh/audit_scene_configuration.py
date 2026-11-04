#!/usr/bin/env python3
"""Read saved configs, masks and evaluations; compose CPU-only diagnostic figures."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import make_paper_comparison as fig


def regions(results):
    rows = list(results['per_view'].values())
    fraction = np.array([r['mask_fraction'] for r in rows])
    whole = 10 ** (-np.array([r['PSNR'] for r in rows]) / 10)
    masked = 10 ** (-np.array([r['masked_PSNR'] for r in rows]) / 10)
    outside = np.maximum(whole - masked, 1e-16)
    return dict(inside_psnr=float(np.mean(-10*np.log10(masked/fraction))),
                outside_psnr=float(np.mean(-10*np.log10(outside/(1-fraction)))) ,
                outside_error_fraction=float(outside.sum()/whole.sum()),
                mean_mask_fraction=float(fraction.mean()))


def stage_sheet(name, root, inputs, completion, out):
    removal = root.parents[1]
    names = ['00000', '00060', '00120']
    labels = ['Original virtual view', 'After removal', 'LaMa mask overlay', 'LaMa completed RGB']
    cw, ch, margin = 560, 370, 30
    canvas = Image.new('RGB', (4*cw+5*margin, 100+len(names)*(ch+65)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), name+' | same virtual camera within each row', font=fig.font(30,True), fill='black')
    for c,label in enumerate(labels):
        draw.text((margin+c*(cw+margin), 63), label, font=fig.font(23), fill='black')
    for i,frame in enumerate(names):
        inp = inputs['frames'][frame]
        original = fig.rgb(removal/'work_model/virtual/ours_2000/renders'/f'{frame}.png')
        removed = fig.rgb(inp['inputs']['removed_rgb']['path'])
        mask = np.asarray(fig.rgb(inp['outputs']['color_mask']['path']))[:,:,0] > 0
        tint = np.asarray(removed).copy(); tint[mask] = .55*tint[mask]+.45*np.array([255,40,30])
        images = [original, removed, Image.fromarray(tint), fig.rgb(completion['frames'][frame]['outputs']['color']['path'])]
        y=100+i*(ch+65)
        draw.text((margin,y), f"Frame {frame} | mask {100*mask.mean():.1f}%", font=fig.font(22), fill='#333333')
        for c,im in enumerate(images):
            tile,_=fig.fit(im,(cw,ch));canvas.paste(tile,(margin+c*(cw+margin),y+30))
    canvas.save(out/f'{name}_pipeline.jpg',quality=95)
    # Capture contains the same fixed training view 0 at both steps.
    images = [fig.rgb(completion['frames']['00000']['outputs']['color']['path'])]
    for step in (5000,10000):
        with np.load(root/f'edgs_joint/debug/step_{step:06d}.npz') as d:
            images.append(Image.fromarray(np.rint(np.clip(d['rgb'],0,1)*255).astype(np.uint8)))
    canvas=Image.new('RGB',(3*cw+4*margin,ch+90),'white');draw=ImageDraw.Draw(canvas)
    for c,(label,im) in enumerate(zip(['LaMa target | view 0','Rendered | step 5000','Rendered | step 10000'],images)):
        x=margin+c*(cw+margin);draw.text((x,15),label,font=fig.font(23),fill='black')
        tile,_=fig.fit(im,(cw,ch));canvas.paste(tile,(x,55))
    canvas.save(out/f'{name}_training.jpg',quality=95)


def main():
    job=fig.DEFAULT_JOB
    out=job/'configuration_audit_20261027';out.mkdir(exist_ok=True)
    request=fig.read(job/'request.json');report=fig.read(job/'comparison.json');rows=[]
    for scene in request['scenes']:
        name=scene['name'];roots,frames,labels=fig.resolve_scene(job,scene,report)
        root=roots['PaintMesh 5000'].parents[2]
        config=fig.read(root/'edgs_joint/config.resolved.json')
        initial=fig.read(root/'edgs_init/diagnostics.json')
        selection=fig.read(root/'local_selection/selection_summary.json')
        inputs=fig.read(root/'manifests/lama_input_manifest.json')
        completion=fig.read(root/'manifests/lama_completion_manifest.json')
        d5=fig.read(root/'inpainted_3dgs_5000/evaluation/inpaint360_r4/results.json')
        d10=fig.read(root/'inpainted_3dgs/evaluation/inpaint360_r4/results.json')
        assert d5['metrics']==report['metrics']['PaintMesh 5000'][name]
        assert d10['metrics']==report['metrics']['PaintMesh 10000'][name]
        area=lambda k:[v[k]/np.prod(v['shape']) for v in inputs['frames'].values()]
        annotation=root.parents[1]/'tracker/automatic_annotation.json'
        annotation=fig.read(annotation) if annotation.exists() else {}
        diagnostics=[fig.read(p) for p in (root/'edgs_joint/diagnostics').glob('step_*.json')]
        debug={str(s):fig.read(root/f'edgs_joint/debug/step_{s:06d}.json')['metrics'] for s in (5000,10000)}
        row=dict(scene=name,root=str(root),target_ids=scene['target_ids'],
                 dilation=inputs['parameters']['dilation'],config=config,
                 config_source=str(root/'edgs_joint/config.resolved.json'),
                 config_sha256=fig.sha(root/'edgs_joint/config.resolved.json'),
                 trainable_old=selection['trainable_old'],old_count=selection['initial_old'],
                 unfrozen_fraction=selection['trainable_old']/selection['initial_old'],
                 new_count=initial['initialized_points'],depth_points=initial.get('depth_points'),
                 mask_before_mean=float(np.mean(area('mask_foreground_before'))),
                 mask_after_mean=float(np.mean(area('mask_foreground_after'))),
                 all_frames_visually_reviewed=annotation.get('visually_reviewed_all_frames'),
                 recursive_guide=completion['parameters']['recursive_guide'],
                 inputs_complete=inputs['complete'],completion_complete=completion['complete'],
                 diagnostic_count=len(diagnostics),diagnostics_finite=all(np.isfinite(d['total']) for d in diagnostics),
                 debug_view0_metrics=debug,metrics_5000=d5['metrics'],metrics_10000=d10['metrics'],
                 region_5000=regions(d5),region_10000=regions(d10))
        rows.append(row)
        stage_sheet(name,root,inputs,completion,out)
        print(name, 'PSNR',round(d5['metrics']['PSNR'],3),round(d10['metrics']['PSNR'],3),
              'outside',round(row['region_5000']['outside_psnr'],3),round(row['region_10000']['outside_psnr'],3),
              'local_old',round(100*row['unfrozen_fraction'],1),flush=True)
    fig.save_json(out/'inventory.json',dict(created_at=datetime.now().astimezone().isoformat(),scenes=rows))
    lines=['# 全场景实际配置与质量检查', '',
           '[配置位置、质量原因与修复建议](../../../scripts/paintmesh/INPAINT360_CURRENT_CONFIG_AUDIT.zh-CN.md)。', '',
           '读取已保存的输入、配置和评估，只生成 CPU 诊断图，不修改模型、训练输入或指标。cube 使用报告登记的 alpha95 mask 修正版。', '',
           '| 场景 | PSNR 5k | PSNR 10k | 差值 | 局部解冻旧点 | LaMa mask 平均占比 | 流程检查图 | 同训练视角 5k/10k |',
           '|---|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        n=r['scene'];p5=r['metrics_5000']['PSNR'];p10=r['metrics_10000']['PSNR']
        lines.append(f"| {n} | {p5:.4f} | {p10:.4f} | {p10-p5:+.4f} | {100*r['unfrozen_fraction']:.1f}% | {100*r['mask_after_mean']:.1f}% | [原始/移除/mask/LaMa]({n}_pipeline.jpg) | [LaMa/5k/10k]({n}_training.jpg) |")
    lines+=['', '区域误差使用官方测试 mask，从已有 MSE 分解，按区域像素数归一化后逐视角平均。它与论文式 masked PSNR 不同；虚拟训练 mask 面积不可与真实测试 mask 面积直接作覆盖率比较。', '',
            '| 场景 | 洞内 PSNR 5k | 洞内 PSNR 10k | 洞外 PSNR 5k | 洞外 PSNR 10k |', '|---|---:|---:|---:|---:|']
    for r in rows:
        a,b=r['region_5000'],r['region_10000']
        lines.append(f"| {r['scene']} | {a['inside_psnr']:.4f} | {b['inside_psnr']:.4f} | {a['outside_psnr']:.4f} | {b['outside_psnr']:.4f} |")
    lines+=['', '[完整配置、计数、区域误差和原文件路径](inventory.json)。每场景流程图仅抽查 0/60/120 帧，不代表 180 帧均已人工检查。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
