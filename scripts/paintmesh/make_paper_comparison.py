#!/usr/bin/env python3
"""Compose scientific comparison figures from recorded test renders, on CPU."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/home/member/wyc')
DEFAULT_JOB = BASE/'gsagent/logs/paintmesh_full_r4_gpu1_20261025_105925'
METHODS = ('Inpaint360GS', 'Gaussian Grouping', 'AuraFusion360', 'PaintMesh 5000')
COLS = (*METHODS, 'Ground Truth')
DUAL_PSNR_KINDS = ('joint_psnr', 'relative_joint_psnr')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
PAPER = 'https://arxiv.org/html/2511.06457v1#S4.F6'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def font(size, bold=False): return ImageFont.truetype(BOLD if bold else FONT, size)
def rgb(path):
    with Image.open(path) as im: return im.convert('RGB')
def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def native_folder(source, scene, workspace):
    keys = [key for key in read(source)['per_scene_results']
            if key.split('/inpaint/')[0].split('/')[-1] == scene]
    if len(keys) != 1: raise ValueError(f'Ambiguous render source: {scene}: {source}')
    key = Path(keys[0])
    return key if key.is_absolute() else workspace/key


def resolve_scene(job, scene, report):
    name = scene['name']
    pm = Path(scene['inpaint_root'])/'inpainted_3dgs_5000/evaluation/inpaint360_r4'
    roots = {
        'Inpaint360GS': BASE/f'Inpaint360GS/output/paper_r4_all_scenes/scenes/{name}/workspace/output/inpaint360/{name}/inpaint/ours_object_inpaint_virtual/iteration_5000',
        'Gaussian Grouping': BASE/f'gaussian-grouping/output/inpaint360/{name}/inpaint/ours_gaussian_grouping/iteration_9999',
        'AuraFusion360': BASE/f'AuraFusion360_official/output/inpaint360/{name}/inpaint/ours_aura/iteration_10000',
        'PaintMesh 5000': pm,
    }
    labels = {}
    for item in report.get('report_corrections', []):
        if item['scene'] != name or item['method'] not in METHODS: continue
        source = Path(item['source'])
        if sha(source) != item['sha256']: raise ValueError('Registered metrics changed: '+str(source))
        if item['format'] == 'paintmesh': root = source.parent
        else:
            workspace = next(p for p in source.parents if p.name == 'workspace')
            root = native_folder(source, name, workspace)
        roots[item['method']] = root
        labels[item['method']] = item['label']
    frames = read(roots['PaintMesh 5000']/'results.json')['inputs']['frames']
    expected = {f+'.png' for f in frames}
    for method, root in roots.items():
        for part in ('renders', 'gt'):
            actual = {p.name for p in (root/part).glob('*.png')}
            if actual != expected:
                raise ValueError(f'{name}/{method}/{part}: missing={expected-actual}, extra={actual-expected}')
    return roots, sorted(expected), labels


def mask_info(path):
    with Image.open(path) as im: mask = np.asarray(im.convert('L')) > 127
    yy, xx = np.where(mask)
    if not len(xx): return None
    h, w = mask.shape
    return dict(area=float(mask.mean()), bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                touches_edge=bool(xx.min()==0 or yy.min()==0 or xx.max()==w-1 or yy.max()==h-1))


def crop_box(bbox, size):
    w,h=size; x0,y0,x1,y1=bbox;cx=(x0+x1)/2;cy=(y0+y1)/2
    cw=max((x1-x0)*1.35,w*.25);ch=max((y1-y0)*1.35,h*.25)
    if cw/ch<1.55: cw=ch*1.55
    else: ch=cw/1.55
    cw=min(w,int(np.ceil(cw)));ch=min(h,int(np.ceil(ch)))
    x=max(0,min(w-cw,round(cx-cw/2)));y=max(0,min(h-ch,round(cy-ch/2)))
    return [x,y,x+cw,y+ch]


def select_frames(frames, masks):
    stats={f:mask_info(masks/f) for f in frames}
    valid=[f for f in frames if stats[f] is not None]
    if len(valid)<3: raise ValueError('Need at least three visible repair-region views')
    eligible=[f for f in valid if not stats[f]['touches_edge']] or valid
    target=float(np.quantile([stats[f]['area'] for f in eligible],.75))
    main=min(eligible,key=lambda f:(abs(stats[f]['area']-target),abs(valid.index(f)-(len(valid)-1)/2)))
    chosen=[main]
    for q in (.2,.8):
        target_i=round(q*(len(valid)-1))
        chosen.append(min((f for f in valid if f not in chosen),key=lambda f:abs(valid.index(f)-target_i)))
    return chosen,stats


def aligned_images(roots, frame):
    gt=rgb(roots['PaintMesh 5000']/'gt'/frame);size=gt.size;reference=np.asarray(gt,dtype=np.float32)
    images={};checks={}
    for method,root in roots.items():
        view=rgb(root/'renders'/frame);own_gt=rgb(root/'gt'/frame)
        if view.size!=own_gt.size: raise ValueError('Render/GT size differs: '+method)
        original_size=view.size
        if original_size!=size:
            if max(abs(a-b) for a,b in zip(size,original_size))>1: raise ValueError('Unexpected resolution mismatch')
            view=view.resize(size,Image.Resampling.BICUBIC)
            own_gt=own_gt.resize(size,Image.Resampling.BICUBIC)
        diff=np.asarray(own_gt,dtype=np.float32)-reference
        mae=float(np.abs(diff).mean());rmse=float(np.sqrt(np.mean(diff**2)))
        if mae>3 or rmse>8: raise ValueError(f'GT/camera alignment failed: {method}/{frame}: {mae}/{rmse}')
        checks[method]=dict(native_size=original_size,display_size=size,gt_mae_255=mae,gt_rmse_255=rmse,
            display_resized=original_size!=size,render=str(root/'renders'/frame),render_sha256=sha(root/'renders'/frame))
        images[method]=view
    images['Ground Truth']=gt
    return images,checks


def text_center(draw, xy, text, f, fill='#171717'):
    x,y=xy;b=draw.textbbox((0,0),text,font=f);draw.text((x-(b[2]-b[0])/2,y),text,font=f,fill=fill)


def fit(im,size):
    out=Image.new('RGB',size,'white')
    scale=min(size[0]/im.width,size[1]/im.height)
    scaled=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.BICUBIC)
    x=(size[0]-scaled.width)//2;y=(size[1]-scaled.height)//2;out.paste(scaled,(x,y));return out,(x,y,scaled.width,scaled.height)


def score_label(record, method):
    if record.get('selection_kind') in DUAL_PSNR_KINDS:
        scores = record['per_view_metrics'][method]
        label = f"PSNR {scores['PSNR']:.2f} | mPSNR {scores['masked_PSNR']:.2f} dB"
    elif record.get('selection_kind') == 'full_psnr':
        label = f"PSNR {record['per_view_metrics'][method]['PSNR']:.2f} dB"
    else:
        label = f"FID {record['metrics'][method]['FID']:.2f}"
    return label + (' *' if method in record['labels'] else '') + (' †' if method in record.get('evaluation_issues', {}) else '')


def draw_score(draw, record, method, x, y, width, height):
    label = score_label(record, method)
    f = font(22); b = draw.textbbox((0, 0), label, font=f); tw = b[2] - b[0]
    draw.rectangle((x+width-tw-14, y+height-36, x+width, y+height), fill='white')
    draw.text((x+width-tw-7, y+height-32), label, font=f, fill='black')


def overview(records,cell_width=720,crops=False):
    margin=28;gap=12;left=195;header=85;row_gap=16
    issue_names=sorted({f"{method}/{r['scene']}" for r in records for method in r.get('evaluation_issues', {})})
    heights=[round(cell_width*(r['size'][1]/r['size'][0] if not crops else .54)) for r in records]
    width=left+5*cell_width+4*gap+margin;height=header+sum(heights)+(len(records)-1)*row_gap+(132 if issue_names else 100)
    canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    for c,name in enumerate(COLS):text_center(draw,(left+c*(cell_width+gap)+cell_width/2,25),name,font(28,True))
    y=header
    for rec,rh in zip(records,heights):
        scene=rec['scene'];draw.text((14,y+10),scene.replace('_','\n'),font=font(25,True),fill='#171717')
        draw.text((14,y+rh-30),rec['frame'].replace('test_IMG_','').replace('.png',''),font=font(18),fill='#666666')
        for c,name in enumerate(COLS):
            im=rec['images'][name];x=left+c*(cell_width+gap)
            if crops:
                tile,_=fit(im.crop(rec['crop']),(cell_width,rh));canvas.paste(tile,(x,y))
            else:
                tile,placement=fit(im,(cell_width,rh));canvas.paste(tile,(x,y));px,py,sw,sh=placement
                x0,y0,x1,y1=rec['crop']
                if name=='Ground Truth':
                    draw.rectangle((x+px+x0/im.width*sw,y+py+y0/im.height*sh,x+px+x1/im.width*sw,y+py+y1/im.height*sh),outline='#f4cf24',width=4)
                inset=im.crop(rec['crop']);iw=int(cell_width*.41);ih=round(iw*inset.height/inset.width)
                inset=inset.resize((iw,ih),Image.Resampling.BICUBIC)
                ix=x+4;iy=y+rh-ih-4;canvas.paste(inset,(ix,iy));draw.rectangle((ix-2,iy-2,ix+iw+1,iy+ih+1),outline='white',width=3)
            if name!='Ground Truth':
                draw_score(draw, rec, name, x, y, cell_width, rh)
        y+=rh+row_gap
    note=('Views selected to highlight PaintMesh; not a representative sample. FID remains the all-test-view scene score.'
          if records[0].get('showcase') else 'Same held-out camera and ROI across columns. FID is the scene-level evaluation score, not a per-view score.')
    if records[0].get('selection_kind') == 'full_psnr':
        note='Highest PaintMesh full-image PSNR view per scene; selected showcase. Labels: per-view PSNR at each method\'s native evaluation size.'
    if records[0].get('selection_kind') in DUAL_PSNR_KINDS:
        note='Jointly high PaintMesh PSNR / masked PSNR views; selected showcase. mPSNR: outside-mask pixels zeroed, averaged over the full image.'
    if records[0].get('selection_kind') == 'relative_joint_psnr':
        note='Selected to favor PaintMesh: high own scores, then gaps to the strongest competitors. Not representative; mPSNR uses zero outside mask.'
    draw.text((left,height-68),note,font=font(22),fill='#444444')
    draw.text((left,height-36),'* Registered corrected run; AuraFusion360/redbull uses the initializer-corrected variant. PaintMesh: 5,000-step snapshot.',font=font(21),fill='#444444')
    if issue_names:
        draw.text((left,height-100),'† TARGET MISMATCH: '+', '.join(issue_names)+'. Extra non-target removed; comparison pending correction.',font=font(22,True),fill='#ad2020')
    return canvas


def scene_sheet(records):
    cw=620;gap=12;left=45;width=left+5*cw+4*gap+30;header=140
    heights=[round(cw*r['size'][1]/r['size'][0])+round(cw*.48)+90 for r in records]
    issues=records[0].get('evaluation_issues', {})
    canvas=Image.new('RGB',(width,header+sum(heights)+(117 if issues else 85)),'white');draw=ImageDraw.Draw(canvas)
    title='  |  PaintMesh showcase - matched test views' if records[0].get('showcase') else '  |  matched test views'
    if records[0].get('selection_kind') == 'full_psnr':
        title='  |  Top 3 PaintMesh full-image PSNR views'
    if records[0].get('selection_kind') in DUAL_PSNR_KINDS:
        title='  |  Top 3 PaintMesh joint PSNR / masked PSNR views'
    if records[0].get('selection_kind') == 'relative_joint_psnr':
        title='  |  PaintMesh relative-advantage showcase'
    draw.text((left,20),records[0]['scene']+title,font=font(35,True),fill='black')
    for c,name in enumerate(COLS):text_center(draw,(left+c*(cw+gap)+cw/2,85),name+(' *' if name in records[0]['labels'] else '')+(' †' if name in issues else ''),font(26,True))
    y=header
    for rank,(r,rh) in enumerate(zip(records,heights),1):
        prefix=f'Rank {rank}  /  ' if r.get('selection_kind') in ('full_psnr', *DUAL_PSNR_KINDS) else ''
        row_title=prefix+r['frame']+'   /   full view + identical ROI below'
        if r.get('selection_kind') == 'relative_joint_psnr':
            row_title+=f"   /   gaps to best rivals: {r['relative_selection']['psnr_gap']:+.2f} / {r['relative_selection']['masked_psnr_gap']:+.2f} dB"
            if not r['relative_selection']['high_own_scores']:
                row_title+='   [supplementary: below joint median gate]'
        draw.text((left,y),row_title,font=font(23),fill='#444444');y+=40
        fh=round(cw*r['size'][1]/r['size'][0]);zh=round(cw*.48)
        for c,name in enumerate(COLS):
            x=left+c*(cw+gap);im=r['images'][name];tile,(px,py,sw,sh)=fit(im,(cw,fh));canvas.paste(tile,(x,y))
            x0,y0,x1,y1=r['crop'];draw.rectangle((x+px+x0/im.width*sw,y+py+y0/im.height*sh,x+px+x1/im.width*sw,y+py+y1/im.height*sh),outline='#e3b71b',width=3)
            if r.get('selection_kind') in ('full_psnr', *DUAL_PSNR_KINDS) and name != 'Ground Truth':
                draw_score(draw, r, name, x, y, cw, fh)
            tile,_=fit(im.crop(r['crop']),(cw,zh));canvas.paste(tile,(x,y+fh+14))
        y+=fh+zh+50
    note=('Views selected to highlight PaintMesh, not a representative sample. Same camera, crop and scaling; no image synthesis.'
          if records[0].get('showcase') else 'Same camera / same crop / same display scaling. Existing 1/4-resolution test renders; no image synthesis.')
    if records[0].get('selection_kind') == 'full_psnr':
        note='Ranked by PaintMesh full-image PSNR. Labels: per-view native PSNR, not ROI PSNR. Selected views; full-test averages unchanged.'
    if records[0].get('selection_kind') in DUAL_PSNR_KINDS:
        note='Joint percentile ranking. mPSNR: zero outside mask, then full-image metric; not crop PSNR. Full-test averages unchanged.'
    if records[0].get('selection_kind') == 'relative_joint_psnr':
        note='Selected to favor PaintMesh against the strongest rivals, after a high-own-score gate. mPSNR is not crop PSNR. Full-test averages unchanged.'
    draw.text((left,canvas.height-60),note,font=font(22),fill='#444444')
    draw.text((left,canvas.height-30),'* Corrected run recorded in manifest.json. AuraFusion360/redbull: initializer-corrected variant.',font=font(21),fill='#444444')
    if issues:
        draw.text((left,canvas.height-94),'† TARGET MISMATCH: '+', '.join(issues)+'. Extra non-target removed; comparison pending correction.',font=font(22,True),fill='#ad2020')
    return canvas


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=Path,default=DEFAULT_JOB)
    parser.add_argument('--selection-json',type=Path,help='Explicitly selected showcase views; label this selection in figures')
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
    job=args.job.resolve();out=args.output_dir or job/'qualitative_pm5000';out.mkdir(parents=True,exist_ok=True)
    if args.selection_json and out.resolve()==(job/'qualitative_pm5000').resolve():
        raise ValueError('Keep the method-independent figure set; use a different showcase output directory')
    selection=read(args.selection_json) if args.selection_json else None
    high_psnr=bool(selection and selection.get('kind')=='full_psnr')
    joint_psnr=bool(selection and selection.get('kind') in DUAL_PSNR_KINDS)
    relative_psnr=bool(selection and selection.get('kind')=='relative_joint_psnr')
    for d in ('scenes','pages','aligned_crops'): (out/d).mkdir(exist_ok=True)
    report=read(job/'comparison.json');request=read(job/'request.json');manifest=[];primary=[];sheets=[]
    issue_file=job/'evaluation_issues.json'
    evaluation_issues=read(issue_file)['issues'] if issue_file.exists() else []
    active_issues={(r['scene'],r['method']):r for r in evaluation_issues if r.get('status')=='confirmed_target_mismatch_pending_repair'}
    if joint_psnr and selection['report_sha256'] != sha(job/'comparison.json'):
        raise ValueError('The report changed after joint view selection; rerun selection first')
    (out/'comparison_source.json').write_bytes((job/'comparison.json').read_bytes())
    for scene in request['scenes']:
        roots,frames,labels=resolve_scene(job,scene,report);chosen,stats=select_frames(frames,roots['PaintMesh 5000']/'masks');records=[]
        if selection:
            chosen=selection['scenes'][scene['name']]['frames']
            if len(chosen)!=3 or len(set(chosen))!=3 or any(fn not in frames or stats[fn] is None for fn in chosen):
                raise ValueError('Invalid showcase frame selection for '+scene['name'])
        for fn in chosen:
            images,checks=aligned_images(roots,fn);crop=crop_box(stats[fn]['bbox'],images['Ground Truth'].size)
            record=dict(scene=scene['name'],frame=fn,size=images['Ground Truth'].size,crop=crop,images=images,labels=labels,showcase=bool(selection),
                        metrics={m:report['metrics'][m][scene['name']] for m in METHODS})
            record['evaluation_issues']={m:active_issues[scene['name'],m] for m in METHODS if (scene['name'],m) in active_issues}
            if high_psnr or joint_psnr:
                record.update(selection_kind=selection['kind'],per_view_metrics=selection['scenes'][scene['name']]['per_frame_metrics'][fn])
            if relative_psnr:
                record['relative_selection']=next(r for r in selection['scenes'][scene['name']]['selected_ranks'] if r['frame']==fn)
            records.append(record)
            manifest.append(dict(scene=scene['name'],frame=fn,primary=fn==chosen[0],crop=crop,mask=stats[fn],
                gt=str(roots['PaintMesh 5000']/'gt'/fn),mask_source=str(roots['PaintMesh 5000']/'masks'/fn),
                sources=checks,correction_labels=labels,per_view_metrics=record.get('per_view_metrics'),
                relative_selection=record.get('relative_selection'),evaluation_issues=record['evaluation_issues']))
        primary.append(records[0]);sheet=scene_sheet(records);sheet.save(out/'scenes'/f"{scene['name']}.png",dpi=(300,300));sheets.append(sheet)
        overview([records[0]],crops=True).save(out/'aligned_crops'/f"{scene['name']}.png",dpi=(300,300))
        print(scene['name'],chosen,'verified and drawn',flush=True)
    all_image=overview(primary);all_image.save(out/'all_scenes.png',dpi=(300,300))
    preview=all_image.copy();preview.thumbnail((1700,3400),Image.Resampling.LANCZOS);preview.save(out/'all_scenes_preview.jpg',quality=93)
    crop_image=overview(primary,crops=True);crop_image.save(out/'all_scenes_crops.png',dpi=(300,300))
    pages=[]
    for i,offset in enumerate(range(0,len(primary),4),1):
        page=overview(primary[offset:offset+4]);page.save(out/'pages'/f'comparison_{i:02}.png',dpi=(300,300));pages.append(page)
    pages[0].save(out/'comparison_all_scenes.pdf',save_all=True,append_images=pages[1:],resolution=200,quality=95,subsampling=0)
    sheets[0].save(out/'comparison_multiview.pdf',save_all=True,append_images=sheets[1:],resolution=200,quality=95,subsampling=0)
    save_json(out/'manifest.json',dict(created_at=datetime.now().astimezone().isoformat(),paper_reference=PAPER,
        report_source=str(out/'comparison_source.json'),report_sha256=sha(out/'comparison_source.json'),original_report=str(job/'comparison.json'),methods=list(METHODS),scene_count=len(primary),view_count=len(manifest),
        selection=(dict(showcase=True,description=selection['protocol'],source=str(args.selection_json.resolve()),sha256=sha(args.selection_json)) if selection else 'Primary: non-border mask area closest to scene 75th percentile; extra views: chronological 20%/80% of nonempty masks, without duplicates. No method outputs/metrics used in selection.'),
        display='PGSR evaluation resolution; at most one-pixel native size differences resampled with bicubic for display only. GT identity checked for every selected view. Crops and insets identical across methods. No enhancement or generation.',
        evaluation_issues_source=({'path':str(issue_file),'sha256':sha(issue_file)} if issue_file.exists() else None),records=manifest))
    intro=('按 PaintMesh 5000 的逐帧全图 PSNR 严格降序排序，每个场景选择前三名，总览使用第一名。黄色框与局部图显示统一裁剪区域；图片角落标注的是该方法该视角的全图 PSNR（dB），不是局部 PSNR。高 PSNR 视角展示不代表全测试集平均表现，完整测试集指标不变。'
        if high_psnr else ('按用户要求优先选择 PaintMesh 5000 的修复区域质量及相对表现较好的视角。这是定性展示选集，不代表全测试集平均水平。全部方法使用同一相机和裁剪；完整测试指标不变。' if selection else '每个场景选取 3 个相同测试视角。主视角仅根据评估 mask 的可见面积选择；补充视角按序列位置选择，没有使用方法得分筛选。')+'黄色框为统一裁剪区域，角落 FID 是整个场景的指标，不是该帧指标。')
    suffix='：PaintMesh 高 PSNR 视角展示' if high_psnr else ('：PaintMesh 优势视角展示' if selection else '')
    if joint_psnr:
        intro=selection['protocol']+' 图片角落同时显示该视角 PSNR 与 mPSNR（masked_PSNR）；局部图角落仍是原视角指标，不是裁剪区域重新计算的分数。'
        suffix='：PSNR 与 masked_PSNR 联合择优展示'
    if relative_psnr:
        suffix='：自身双指标较高、相对优势优先展示'
    lines=['# 四方法全场景定性对照'+suffix,'', '列顺序：Inpaint360GS、Gaussian Grouping、AuraFusion360、PaintMesh 5000、GT。',
        '', '[全场景高清总览](all_scenes.png) · [修复区域总览](all_scenes_crops.png) · [三页 PDF](comparison_all_scenes.pdf) · [11 场景多视角 PDF](comparison_multiview.pdf)',
        '', '![全部场景预览](all_scenes_preview.jpg)', '',
        intro,
        '所有内容来自服务器实际渲染。使用当前 comparison.md 登记的修复结果，带 * 的方法/场景为修正版，其中 AuraFusion360/redbull 改过初始化。',
        'Inpaint360GS 在部分场景的四分之一尺寸与其他方法相差 1 像素，仅制图时统一尺寸；每个入选视角已核对 GT、记录尺寸和对齐误差，评估文件保持原样。',
        '',f'排版参考 [Inpaint360GS Figure 6]({PAPER})。图像不是论文原文公布的渲染。','',
        '| 场景 | 三视角完整对照 | 主视角局部对照 |','|---|---|---|']
    for r in primary: lines.append(f"| {r['scene']} | [{r['scene']}](scenes/{r['scene']}.png) | [ROI](aligned_crops/{r['scene']}.png) |")
    if active_issues:
        lines[2:2]=['**对比问题：带 † 的方法误删了非目标物体，删除任务不一致，不能用来证明同目标任务中的方法优势，待修复重评。** [问题记录](../evaluation_issues.json)', '']
    lines+=['','复现脚本：`scripts/paintmesh/make_paper_comparison.py`；视角、裁剪和文件校验记录： [manifest.json](manifest.json)。']
    if selection:lines+=['','[选图依据与逐视角分数](selection/README.md) · [保留的统一规则选图](../qualitative_pm5000/README.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    # Static, offline index for conveniently opening the existing figure assets.
    cards=''.join(f'<section><h2>{html.escape(r["scene"])}</h2><a href="scenes/{r["scene"]}.png"><img loading="lazy" src="scenes/{r["scene"]}.png"></a></section>' for r in primary)
    notice='<p>Views selected to highlight PaintMesh; not a representative sample. Full-test metrics unchanged.</p>' if selection else ''
    if high_psnr:
        notice='<p>Top 3 full-image PSNR views per scene, ranked by PaintMesh 5000. Per-view PSNR labels; full-test averages unchanged.</p>'
    if joint_psnr:
        notice='<p>Top 3 joint PSNR / masked PSNR views per scene, ranked by the lower of the two PaintMesh percentiles. Selected showcase; full-test averages unchanged.</p>'
    if relative_psnr:
        notice='<p>Selected to favor PaintMesh: require both own scores at or above the scene medians, then maximize gaps to the strongest competitors. This is an explicitly biased showcase, not a representative sample. Supplementary views below the gate are labeled.</p>'
    if active_issues:
        notice+='<p style="color:#ad2020"><strong>† Target mismatch: extra non-target objects removed. The marked comparison is pending correction; see evaluation_issues.json.</strong></p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Inpaint360GS four-method comparison</title><style>body{max-width:1800px;margin:32px auto;font:16px sans-serif;background:#f5f5f5;color:#222}section{background:white;margin:20px 0;padding:20px}img{width:100%}a{color:#234e91}</style><h1>Inpaint360GS · Gaussian Grouping · AuraFusion360 · PaintMesh 5000</h1>'+notice+'<p>11 scenes / 3 matched test views each / GT at right. * = corrected run; Aura redbull = initializer-corrected variant.</p><p><a href="comparison_all_scenes.pdf">Overview PDF</a> · <a href="comparison_multiview.pdf">Multi-view PDF</a> · <a href="README.md">Protocol and sources</a></p>'+cards)
    print(out,flush=True)


if __name__=='__main__': main()
