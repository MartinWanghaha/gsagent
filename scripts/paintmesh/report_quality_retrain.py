#!/usr/bin/env python3
"""Summarize completed repairs, queue state and reproducible diagnostic figures."""
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run_geometry_ablation import read, write, now, sha
from run_quality_retrain import report


def run(job):
    report(job)
    request=read(job/'request.json');comparison=read(job/'comparison.json')
    queues={}
    for sub in ['']+sorted(p.parent.name for p in job.glob('*/request.json')):
        p=job/sub/'status.json'
        if p.exists():queues[sub or 'fixed_inputs']=read(p)
    write(job/'overall_status.json',dict(updated_at=now(),queues=queues,
        completed=sum(r['state']=='complete' for r in comparison['scenes']),registered=len(comparison['scenes']),
        failed=sum(r['state']=='failed' for r in comparison['scenes']),
        running=sum(r['state']=='running' for r in comparison['scenes']),
        note='An old queue failure may be superseded by an explicitly recorded retry; inspect doppel_retry for current doppelherz status.'))
    baselines=job/'method_baselines.json'
    if not baselines.exists():
        source=Path(request['baseline_job'])/'comparison.json'
        write(baselines,dict(source=str(source),sha256=sha(source),snapshot=read(source)))
    metrics=read(baselines)['snapshot']['metrics']
    lines=['# 本轮修复与三方法的本机复现结果','',
           '这里只对比本服务器已完成的实验；不是论文公布值。原基线中的 mask 修正和 Aura/redbull 初始化变体标记继续适用。每行均为相同场景完整测试集的全图 PSNR。','',
           '| 场景 | 实验 | PaintMesh 原 5000 | 本轮 5000 | Inpaint360GS | Gaussian Grouping | AuraFusion360 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in comparison['scenes']:
        if r['state']!='complete':continue
        numbers=[metrics[m][r['scene']]['PSNR'] for m in ('Inpaint360GS','Gaussian Grouping','AuraFusion360')]
        lines.append(f"| {r['scene']} | {r['experiment']} | {r['baseline']['PSNR']:.4f} | {r['metrics']['PSNR']:.4f} | "+' | '.join(f'{v:.4f}' for v in numbers)+' |')
    lines+=['','本轮完整七项指标与差值在 [comparison.json](comparison.json)；此表未把不同实验的最佳指标拼接成一个模型，也未把逐场景事后最佳值包装成固定配置平均值。']
    (job/'comparison_with_methods.md').write_text('\n'.join(lines)+'\n')
    folder=job/'before_after';folder.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    records=[]
    for row in comparison['scenes']:
        if row['state']!='complete':continue
        source=next(s for s in request['scenes'] if s['scene']==row['scene'])
        old=Path(source['root'])/'inpainted_3dgs_5000/evaluation/inpaint360_r4'
        new=Path(row['output'])/'inpainted_3dgs/evaluation/inpaint360_r4'
        target=folder/f"{row['scene']}_{row['experiment']}.jpg"
        if target.exists():continue
        values=[]
        for p in sorted((old/'renders').glob('*.png')):
            a=np.array(Image.open(p).convert('RGB'),dtype=float)/255
            b=np.array(Image.open(old/'gt'/p.name).convert('RGB'),dtype=float)/255
            values.append((-10*np.log10(max(np.mean((a-b)**2),1e-16)),p.name))
        values.sort();selected=[('baseline worst',values[0][1]),('baseline median',values[len(values)//2][1])]
        canvas=Image.new('RGB',(1440,1120),'white');draw=ImageDraw.Draw(canvas)
        for index,(reason,name) in enumerate(selected):
            mask=np.array(Image.open(old/'masks'/name).convert('L'))>127
            yy,xx=np.where(mask);box=(max(0,int(xx.min())-35),max(0,int(yy.min())-35),min(mask.shape[1],int(xx.max())+36),min(mask.shape[0],int(yy.max())+36))
            paths=[old/'renders'/name,new/'renders'/name,old/'gt'/name]
            for col,(label,p) in enumerate(zip(('Baseline 5000','Repair 5000','GT (evaluation only)'),paths)):
                im=Image.open(p).convert('RGB');full=im.copy();ImageDraw.Draw(full).rectangle(box,outline='red',width=3)
                full.thumbnail((470,290));x=col*480;y=index*560
                canvas.paste(full,(x,y+60));draw.text((x+4,y+6),label,font=font,fill='black')
                draw.text((x+4,y+30),name+' / '+reason,font=font,fill='black')
                crop=im.crop(box);crop.thumbnail((470,205));canvas.paste(crop,(x,y+350))
        canvas.save(target,quality=93)
        records.append(dict(scene=row['scene'],experiment=row['experiment'],image=str(target),views=selected,
                            selection='worst and median FULL-image PSNR of unchanged baseline, not chosen by repair gain'))
    existing=read(folder/'index.json') if (folder/'index.json').exists() else []
    write(folder/'index.json',existing+records)
    lines=['# 修复前后对照','', '每组选择基线全图 PSNR 最差和中位数的两个测试视角；不是按新实验提升最多挑图。每个视角包含同相机基线、修复、GT，以及同范围局部裁剪。定量结论仍以完整测试集为准。','']
    lines += [f"- [{r['scene']} / {r['experiment']}]({Path(r['image']).name})" for r in existing+records]
    (folder/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        run(a.job)
        active=False
        for path in a.job.glob('*/run.pid'):
            try:
                command=Path('/proc')/path.read_text().strip()/'cmdline'
                active |= 'run_quality_retrain.py' in command.read_bytes().decode(errors='replace')
            except FileNotFoundError:pass
        if not a.watch or not active:break
        time.sleep(30)
