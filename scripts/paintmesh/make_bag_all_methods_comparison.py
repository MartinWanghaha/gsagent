#!/usr/bin/env python3
"""CPU-only, source-verified bag test-view montage including both PaintMesh runs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import make_paper_comparison as fig
from select_joint_psnr_views import native_scores

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
KEYS = ('PSNR','masked_PSNR')
ORDER = ('GT', 'Inpaint360GS', 'Gaussian Grouping', 'AuraFusion360',
         'PaintMesh previous 5000', 'PaintMesh previous 10000',
         'PaintMesh current 5000', 'PaintMesh current 10000')
LABELS = {
    'GT': ('Ground Truth', '真实背景 / 相同测试相机'),
    'Inpaint360GS': ('Inpaint360GS', '本机复现 · 5000 步'),
    'Gaussian Grouping': ('Gaussian Grouping', '本机复现 · 9999 步'),
    'AuraFusion360': ('AuraFusion360', '本机复现 · 10000 步'),
    'PaintMesh previous 5000': ('PaintMesh 上轮 · 5000', 'adaptive180 · dilation=30 椭圆'),
    'PaintMesh previous 10000': ('PaintMesh 上轮 · 10000', 'adaptive180 · dilation=30 椭圆'),
    'PaintMesh current 5000': ('PaintMesh 本轮 · 5000', 'circle30 · dilation=10 方形'),
    'PaintMesh current 10000': ('PaintMesh 本轮 · 10000', 'circle30 · dilation=10 方形'),
}


def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)


def centered(draw,box,text,size=24,bold=False,fill='#17212b'):
    x,y,w=box; f=font(size,bold)
    while draw.textbbox((0,0),text,font=f)[2] > w-14:
        size-=1; f=font(size,bold)
    draw.text((x+(w-draw.textbbox((0,0),text,font=f)[2])/2,y),text,font=f,fill=fill)


def file_record(path):
    path=Path(path)
    return dict(path=str(path.resolve()),sha256=fig.sha(path),size_bytes=path.stat().st_size)


def sources(job,rerun):
    request=fig.read(rerun/'request.json')
    case=next(s for s in request['scenes'] if s['name']=='bag')
    report=fig.read(job/'comparison.json')
    oldreport=fig.read(rerun/'previous_report/comparison.json')
    previous_request=fig.read(rerun/'previous_report/request.json')
    previous_case=next(s for s in previous_request['scenes'] if s['name']=='bag')
    base,_,_=fig.resolve_scene(job,previous_case,oldreport)
    roots={k:v for k,v in base.items() if k!='PaintMesh 5000'}
    metrics={k:report['metrics'][k]['bag'] for k in roots}
    for epoch,root_key,rep in [('previous','source_inpaint',oldreport),('current','inpaint_root',report)]:
        for step in (5000,10000):
            name=f'PaintMesh {epoch} {step}'
            model='inpainted_3dgs_5000' if step==5000 else 'inpainted_3dgs'
            roots[name]=Path(case[root_key])/model/'evaluation/inpaint360_r4'
            for correction in rep.get('report_corrections',[]):
                if correction['scene']=='bag' and correction['method']==f'PaintMesh {step}':
                    if fig.sha(correction['source'])!=correction['sha256']:raise ValueError('Registered evaluation changed')
                    roots[name]=Path(correction['source']).parent
            metrics[name]=rep['metrics'][f'PaintMesh {step}']['bag']
    current=fig.read(roots['PaintMesh current 5000']/'results.json')
    frames=[s+'.png' for s in current['inputs']['frames']]
    expected=set(frames)
    for method,root in roots.items():
        for part in ('renders','gt'):
            if {p.name for p in (root/part).glob('*.png')}!=expected:
                raise ValueError(f'Incomplete frame set: {method}/{part}')
    # Labels describe the actual preparation receipts, not current defaults.
    for epoch in ('previous','current'):
        root=roots[f'PaintMesh {epoch} 5000'].parents[2]
        lama=fig.read(root/'manifests/lama_input_manifest.json')
        shape=lama['parameters'].get('dilation_shape','ellipse')
        assert (lama['parameters']['dilation'],shape)==((30,'ellipse') if epoch=='previous' else (10,'square'))
    return roots,metrics,frames,current


def validate(roots,metrics,frames,current):
    refroot=roots['PaintMesh current 5000']
    masks={Path(m['path']).stem:m for m in current['inputs']['official_masks']}
    rows=[]
    for i,frame in enumerate(frames):
        reference=fig.rgb(refroot/'gt'/frame);size=reference.size
        reference_array=np.asarray(reference,dtype=np.float32)
        m=masks[Path(frame).stem]
        if fig.sha(m['path'])!=m['sha256']:raise ValueError('Evaluation mask changed')
        info=fig.mask_info(refroot/'masks'/frame)
        if info is None:raise ValueError(f'No repair region in {frame}')
        row=dict(frame=frame,size=list(size),crop=fig.crop_box(info['bbox'],size),mask=info,
            gt=file_record(refroot/'gt'/frame),mask_source=file_record(m['path']),sources={},scores={})
        for method,root in roots.items():
            gt=fig.rgb(root/'gt'/frame);prediction=fig.rgb(root/'renders'/frame)
            if gt.size!=prediction.size:raise ValueError(f'Render/GT size differs: {method}/{frame}')
            native_size=gt.size
            if native_size!=size:
                if max(abs(a-b) for a,b in zip(size,native_size))>1:raise ValueError('Resolution mismatch')
                gt=gt.resize(size,Image.Resampling.BICUBIC)
            diff=np.asarray(gt,dtype=np.float32)-reference_array
            mae=float(np.abs(diff).mean());rmse=float(np.sqrt((diff**2).mean()))
            if mae>3 or rmse>8:raise ValueError(f'Mismatched camera/GT: {method}/{frame}')
            row['scores'][method]=native_scores(root,frame,Path(m['path']))
            row['sources'][method]=dict(render=file_record(root/'renders'/frame),gt=file_record(root/'gt'/frame),
                native_size=list(native_size),display_size=list(size),gt_mae_255=mae,gt_rmse_255=rmse)
        rows.append(row)
        if (i+1)%10==0:print(f'Validated {i+1}/{len(frames)} common test views',flush=True)
    for method in roots:
        for key in KEYS:
            value=float(np.mean([r['scores'][method][key] for r in rows]))
            if abs(value-metrics[method][key])>1e-4:
                raise ValueError(f'Saved images/registered score mismatch: {method}/{key}: {value} vs {metrics[method][key]}')
    return rows


def draw_sheet(rows,roots,metrics,width,*,preview=False):
    gap=12;margin=24;columns=len(ORDER)
    full_h=round(width*rows[0]['size'][1]/rows[0]['size'][0])
    crop_h=round(width/1.55)
    row_h=40+full_h+crop_h+38+26
    header=228; footer=100
    w=2*margin+columns*width+(columns-1)*gap
    h=header+len(rows)*row_h+footer
    canvas=Image.new('RGB',(w,h),'#ffffff');draw=ImageDraw.Draw(canvas)
    title=f"bag | {'6 个均匀抽取的 test 视角预览' if preview else '全部 58 个 test 视角'} | 上轮与本轮 PaintMesh 对比"
    draw.text((margin,14),title,font=font(36,True),fill='#17212b')
    draw.text((margin,68),'每行：完整测试视图 + 同一修复区域放大。P = PSNR；mP = masked_PSNR（dB，越高越好）。',font=font(24),fill='#45525e')
    draw.text((margin,102),'上轮：adaptive180 / 椭圆膨胀30；本轮：官方 circle30 / 方形膨胀10。训练算法仍为 PaintMesh。',font=font(23),fill='#45525e')
    for col,method in enumerate(ORDER):
        x=margin+col*(width+gap)
        color='#dcecf9' if 'current' in method else '#e8f1e7' if 'previous' in method else '#edf0f4'
        draw.rounded_rectangle((x,145,x+width,218),radius=8,fill=color)
        centered(draw,(x,150,width),LABELS[method][0],25,True)
        centered(draw,(x,183,width),LABELS[method][1],19)
    y=header
    for index,row in enumerate(rows):
        if index%2==0:draw.rectangle((0,y,w,y+row_h),fill='#f7f8fa')
        draw.text((margin,y+4),f"{row['frame']}    |    mask 占图 {row['mask']['area']*100:.2f}%",font=font(22,True),fill='#28333d')
        images={m:fig.rgb(r/'renders'/row['frame']) for m,r in roots.items()}
        images['GT']=fig.rgb(roots['PaintMesh current 5000']/'gt'/row['frame'])
        for col,method in enumerate(ORDER):
            x=margin+col*(width+gap);im=images[method]
            if im.size!=tuple(row['size']):im=im.resize(tuple(row['size']),Image.Resampling.BICUBIC)
            fy=y+40
            tile,box=fig.fit(im,(width,full_h));canvas.paste(tile,(x,fy))
            px,py,sw,sh=box;xx0,yy0,xx1,yy1=row['crop']
            draw.rectangle((x+px+xx0/im.width*sw,fy+py+yy0/im.height*sh,
                x+px+xx1/im.width*sw,fy+py+yy1/im.height*sh),outline='#ffc928',width=2)
            cy=fy+full_h+8
            crop,_=fig.fit(im.crop(row['crop']),(width,crop_h));canvas.paste(crop,(x,cy))
            draw.rectangle((x,cy,x+width-1,cy+crop_h-1),outline='#ddb227',width=2)
            text='相同 ROI / Ground Truth' if method=='GT' else f"P {row['scores'][method]['PSNR']:.2f}  |  mP {row['scores'][method]['masked_PSNR']:.2f}"
            centered(draw,(x,cy+crop_h+3,width),text,22,method!='GT')
        y+=row_h
        if (index+1)%10==0:print(f'Composed {index+1}/{len(rows)} rows',flush=True)
    draw.text((margin,h-86),'mP：mask 外误差置零后按整幅图归一化的官方式指标；不是仅对洞内像素归一化的 PSNR。',font=font(23),fill='#45525e')
    draw.text((margin,h-48),'相同 test 相机、相同裁剪；源图仅作展示缩放。完整图按文件名排列全部 58 帧，不按方法优劣筛选。',font=font(23),fill='#45525e')
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,default=fig.DEFAULT_JOB)
    p.add_argument('--rerun-job',type=Path,required=True)
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args();job=a.job.resolve();rerun=a.rerun_job.resolve()
    out=(a.output_dir or job/'bag_test_all_methods_circle30_vs_adaptive180_20261030').resolve()
    out.mkdir(parents=True,exist_ok=True)
    roots,metrics,frames,current=sources(job,rerun)
    rows=validate(roots,metrics,frames,current)
    selected=[int(i) for i in np.linspace(0,len(rows)-1,6).round()]
    small=draw_sheet([rows[i] for i in selected],roots,metrics,640,preview=True)
    small.save(out/'bag_comparison_6views.png',compress_level=4)
    preview=small.copy();preview.thumbnail((2600,5200),Image.Resampling.LANCZOS)
    preview.save(out/'bag_comparison_preview.jpg',quality=94,subsampling=0)
    del small,preview;gc.collect()
    complete=draw_sheet(rows,roots,metrics,448)
    dimensions=list(complete.size)
    complete.save(out/'bag_all_58_test_views.png',compress_level=4)
    complete.save(out/'bag_all_58_test_views.jpg',quality=94,subsampling=0)
    del complete;gc.collect()
    with (out/'per_view_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['frame','method','PSNR','masked_PSNR','mask_fraction'])
        writer.writeheader()
        for row in rows:
            for method in roots:writer.writerow(dict(frame=row['frame'],method=method,
                **row['scores'][method],mask_fraction=row['mask']['area']))
    manifest=dict(created_at=datetime.now().astimezone().isoformat(),scene='bag',view_count=len(frames),
        column_count=8,methods=list(ORDER),roots={m:str(r) for m,r in roots.items()},
        report=file_record(job/'comparison.json'),previous_report=file_record(rerun/'previous_report/comparison.json'),
        request=file_record(rerun/'request.json'),full_sheet_size=dimensions,
        preview_frames=[rows[i]['frame'] for i in selected],selection='All 58 test frames; preview uses 6 uniformly spaced frame indices, without metric-based selection.',
        metrics=metrics,records=rows,artifacts={p.name:file_record(p) for p in sorted(out.glob('*')) if p.suffix in ('.jpg','.png','.csv')})
    fig.save_json(out/'manifest.json',manifest)
    lines=['# bag：所有方法与两轮 PaintMesh 的测试视角对比','',
        '8 列依次为 GT、Inpaint360GS、Gaussian Grouping、AuraFusion360、上轮 PaintMesh 5000/10000、本轮 PaintMesh 5000/10000。',
        '采用官方评估的 58 张 test_IMG_* 相机，不使用训练图中按每 8 张留出的另一组验证图。',
        '所有方法的 406 张渲染均核对帧集合、GT、来源哈希；重算两项逐帧指标并验证均值与登记结果一致。',
        '完整长图按顺序包含全部 58 帧，每帧均有完整视图、黄色框及同一 ROI 放大。预览仅按索引均匀抽取 6 帧，未按效果筛选。','',
        '[全部 58 个 test 视角，单张高清 PNG](bag_all_58_test_views.png) · [较小 JPG](bag_all_58_test_views.jpg)','',
        '[6 视角高清预览 PNG](bag_comparison_6views.png) · [预览 JPG](bag_comparison_preview.jpg)','',
        '| 方法/运行 | 全测试集 PSNR | 全测试集 masked_PSNR |','|---|---:|---:|']
    for method in ORDER[1:]:lines.append(f"| {method} | {metrics[method]['PSNR']:.6f} | {metrics[method]['masked_PSNR']:.6f} |")
    lines+=['','mPSNR 使用官方的 mask 外置零、全图归一化定义。展示图不做生成式修复或美化；只进行统一排版、缩放、裁剪和标注。',
        '本轮/上轮配置与源路径见 [manifest.json](manifest.json)，所有帧指标见 [CSV](per_view_metrics.csv)。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(out),views=len(frames),columns=8,dimensions=dimensions,
        files={p.name:p.stat().st_size for p in out.glob('*.png')}),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
