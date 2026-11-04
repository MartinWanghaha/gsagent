#!/usr/bin/env python3
"""Rank matched held-out views for explicitly labeled PaintMesh showcases."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
from scipy.stats import rankdata
from skimage.metrics import structural_similarity

import make_paper_comparison as fig

OURS='PaintMesh 5000'
PROTOCOL=('按用户要求选择 PaintMesh 5000 优势展示视角，而非代表性样本。基于统一显示网格，计算完整评估 mask 内按像素数归一化的 PSNR、'
    '统一 ROI 的 SSIM，以及相对最强其他方法的优势；这两个局部指标仅供选图，不是论文式 masked 指标。'
    '候选排除修复区过小、边界截断或 PaintMesh 全图质量处于本场景最低 20% 的视角。'
    '排名=35% 洞内 PSNR 优势百分位+25% ROI SSIM 优势百分位+25%自身洞内 PSNR百分位+15%自身 ROI SSIM百分位。'
    '先提供排名靠前且序列位置分散的候选，人工检查后确定展示顺序；四种方法使用相同相机和完整修复区域的统一裁剪。'
    '不修改训练、渲染图像或完整测试集的评估结果。')


def psnr(error):return float(-10*np.log10(max(float(error),1e-12)))


def score_view(roots,frame,stat):
    images,checks=fig.aligned_images(roots,frame)
    gt=np.asarray(images['Ground Truth'],dtype=np.float32)/255
    with Image.open(roots[OURS]/'masks'/frame) as im:mask=np.asarray(im.convert('L'))>127
    box=fig.crop_box(stat['bbox'],images['Ground Truth'].size)
    target=images['Ground Truth'].crop(box);scale=min(1,256/target.width)
    size=(max(7,round(target.width*scale)),max(7,round(target.height*scale)))
    crop_gt=np.asarray(target.resize(size,Image.Resampling.BICUBIC))
    scores={}
    for method in fig.METHODS:
        a=np.asarray(images[method],dtype=np.float32)/255;err=np.mean((a-gt)**2,axis=2)
        roi=np.asarray(images[method].crop(box).resize(size,Image.Resampling.BICUBIC))
        scores[method]=dict(region_psnr=psnr(err[mask].mean()),full_psnr=psnr(err.mean()),
            roi_ssim=float(structural_similarity(crop_gt,roi,channel_axis=2,data_range=255)))
    others=[scores[m] for m in fig.METHODS if m!=OURS]
    return dict(frame=frame,mask=stat,crop=box,scores=scores,
        psnr_margin=scores[OURS]['region_psnr']-max(v['region_psnr'] for v in others),
        ssim_margin=scores[OURS]['roi_ssim']-max(v['roi_ssim'] for v in others),sources=checks)


def separated(rows,n,frame_count):
    result=[];distance=max(2,round(.08*frame_count))
    for row in rows:
        if all(abs(row['index']-r['index'])>=distance for r in result):result.append(row)
        if len(result)==n:return result
    for row in rows:
        if row not in result:result.append(row)
        if len(result)==n:return result
    return result


def candidates_image(scene,rows,roots,path):
    cw=340;gap=8;left=24;header=100;rh=280
    canvas=Image.new('RGB',(left+5*cw+4*gap+24,header+len(rows)*rh+45),'white');d=ImageDraw.Draw(canvas)
    d.text((left,12),scene+' | candidate repair-region crops',font=fig.font(27,True),fill='black')
    for i,name in enumerate(fig.COLS):fig.text_center(d,(left+i*(cw+gap)+cw/2,60),name,fig.font(19,True))
    for row_index,row in enumerate(rows):
        y=header+row_index*rh
        title=f"{row_index+1}. {row['frame']}  |  own ROI PSNR {row['scores'][OURS]['region_psnr']:.2f}  |  margin vs best: {row['psnr_margin']:+.2f} dB, SSIM {row['ssim_margin']:+.3f}"
        d.text((left,y),title,font=fig.font(20),fill='#333333')
        images,_=fig.aligned_images(roots,row['frame'])
        for c,name in enumerate(fig.COLS):
            tile,_=fig.fit(images[name].crop(row['crop']),(cw,230));canvas.paste(tile,(left+c*(cw+gap),y+34))
    d.text((left,canvas.height-30),'Candidates selected to favor PaintMesh. Scores are selection diagnostics, not full-test / paper masked metrics.',font=fig.font(17),fill='#555555')
    canvas.save(path,quality=95)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,default=fig.DEFAULT_JOB);args=p.parse_args()
    job=args.job.resolve();out=job/'qualitative_pm5000_showcase/selection';out.mkdir(parents=True,exist_ok=True)
    report=fig.read(job/'comparison.json');req=fig.read(job/'request.json');selection={};count=0
    for scene in req['scenes']:
        name=scene['name'];roots,frames,labels=fig.resolve_scene(job,scene,report);rows=[]
        for i,f in enumerate(frames):
            stat=fig.mask_info(roots[OURS]/'masks'/f)
            if stat is None:continue
            row=score_view(roots,f,stat);row['index']=i;rows.append(row)
        count+=len(rows)
        min_area=max(.001,float(np.quantile([r['mask']['area'] for r in rows],.25)))
        min_full=float(np.quantile([r['scores'][OURS]['full_psnr'] for r in rows],.2))
        eligible=[r for r in rows if r['mask']['area']>=min_area and not r['mask']['touches_edge'] and r['scores'][OURS]['full_psnr']>=min_full]
        if len(eligible)<3:
            eligible=[r for r in rows if r['mask']['area']>=min_area and r['scores'][OURS]['full_psnr']>=min_full]
        if len(eligible)<3:raise ValueError('Too few visible high-quality candidates: '+name)
        features=[(r['psnr_margin'],r['ssim_margin'],r['scores'][OURS]['region_psnr'],r['scores'][OURS]['roi_ssim']) for r in eligible]
        ranks=rankdata(np.asarray(features),axis=0)/len(eligible);ranks=ranks@np.array([.35,.25,.25,.15])
        for r,score in zip(eligible,ranks):r['showcase_rank_score']=float(score)
        ranked=sorted(eligible,key=lambda r:r['showcase_rank_score'],reverse=True)
        chosen=separated(ranked,3,len(frames));review=separated(ranked,6,len(frames))
        selection[name]=dict(frames=[r['frame'] for r in chosen],candidates=[r['frame'] for r in review],
            source_test_frames=len(frames),scored_frames=len(rows),eligible_frames=len(eligible),
            min_area=min_area,min_full_psnr=min_full,manually_reviewed=False)
        fig.save_json(out/f'{name}_scores.json',dict(scene=name,rows=rows,selection=selection[name]))
        candidates_image(name,review,roots,out/f'{name}_candidates.jpg')
        print(name,'all',len(rows),'eligible',len(eligible),'chosen',[(r['frame'],round(r['psnr_margin'],2),round(r['ssim_margin'],3)) for r in chosen],flush=True)
    fig.save_json(out/'selected_frames.json',dict(created_at=datetime.now().astimezone().isoformat(),protocol=PROTOCOL,
        checkpoint=5000,scored_test_views=count,scenes=selection))
    lines=['# PaintMesh 优势视角选图依据','',PROTOCOL,'',
        '展示图使用当前登记的修正版；完整测试集 PSNR/SSIM/LPIPS/FID 均不变。即使某场景没有明显优势，仍展示真实结果，不修改图像或混用不同相机。','',
        '| 场景 | 测试视角数 | 候选检查图 | 全部选图分数 |','|---|---:|---|---|']
    for scene,s in selection.items():lines.append(f"| {scene} | {s['source_test_frames']} | [候选](./{scene}_candidates.jpg) | [JSON](./{scene}_scores.json) |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('selection complete',count,out/'selected_frames.json',flush=True)


if __name__=='__main__':main()
