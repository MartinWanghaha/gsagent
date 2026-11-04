#!/usr/bin/env python3
"""Collect historical PSNR-best completed PaintMesh models and diagnose images on CPU."""
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
BASE = REPO.parent
SCENES = REPO / 'output/paintmesh/Inpaint360GS/inpaint360'
OUT = REPO / 'output/paintmesh/Inpaint360GS/best_results'
REPORT = REPO / 'logs/paintmesh_full_r4_gpu1_20261025_105925/comparison.json'
KEYS = ['PSNR', 'masked_PSNR', 'SSIM', 'masked_SSIM', 'LPIPS', 'masked_LPIPS', 'FID']
OTHER = {'Inpaint360GS': 'Inpaint360GS', 'Gaussian Grouping': 'gaussian-grouping', 'AuraFusion360': 'AuraFusion360_official'}
SKIP = {'lama','work_model','edgs_init','edgs_joint','local_selection','point_cloud',
        'point_cloud_object_removal','point_cloud_object_inpaint','removed_3dgs','removed_mesh',
        'tracker','config','configs','virtual','mesh','edgs','edgs_bridge','semantic_3dgs',
        'train','test','video','renders','gt','depth','normal','visualizations','debug','diagnostics','wandb'}
POLICY = '每场景从全部保留的完整 r4 测试评估中，按全图 PSNR 降序选择一套模型；同分再按 masked_PSNR。全部七项指标来自该模型，不拼接指标。属于测试集事后择优的历史实验合集，不是统一配置、统一预算或独立验证集选出的实验。'


def read(p): return json.loads(Path(p).read_text())
def write(p, d): Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''): h.update(chunk)
    return h.hexdigest()
def record(p): return dict(path=str(Path(p).resolve()), sha256=digest(p))
def link(p, target):
    target = Path(target).resolve(strict=True)
    if os.path.lexists(str(p)):
        if p.is_symlink() and p.resolve() == target: return
        raise ValueError('Existing output entry differs: '+str(p))
    p.symlink_to(os.path.relpath(target, p.parent), target_is_directory=target.is_dir())
def csvwrite(p, rows):
    with p.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def psnr(mse): return -10*math.log10(max(float(mse), 1e-16))


def candidates(report):
    result=[]
    for parent, dirs, files in os.walk(SCENES, followlinks=False):
        p=Path(parent); rel=p.relative_to(SCENES)
        if p.name=='inpaint360_r4' and p.parent.name=='evaluation' and 'results.json' in files:
            d=read(p/'results.json'); reasons=[]; name=rel.parts[0]; inp=d.get('inputs',{})
            if not d.get('complete'): reasons.append('incomplete')
            if inp.get('resolution')!=4 or d.get('protocol',{}).get('evaluation_resolution')!=4: reasons.append('not r4')
            if d.get('frame_count')!=report['test_frame_counts'].get(name): reasons.append('different test count')
            if d.get('protocol',{}).get('crop')!=1.0 or d.get('protocol',{}).get('resize')!=1.0: reasons.append('different crop/resize')
            if inp.get('scene')!=name: reasons.append('scene mismatch')
            if not inp.get('training_exclude_test_images'): reasons.append('test exclusion not declared')
            if any(not math.isfinite(d.get('metrics',{}).get(k,float('nan'))) for k in KEYS): reasons.append('invalid metrics')
            if not Path(inp.get('model',{}).get('path','/missing')).is_file(): reasons.append('missing PLY')
            expected={n+'.png' for n in inp.get('frames',[])}
            for sub in ['renders','gt','masks']:
                if {f.name for f in (p/sub).glob('*.png')}!=expected: reasons.append('incomplete '+sub)
            result.append(dict(scene=name, result=str(p/'results.json'), metrics=d.get('metrics'),
                               iteration=inp.get('iteration'), eligible=not reasons, excluded_reasons=reasons))
            dirs[:]=[]; continue
        dirs[:]=[n for n in dirs if n not in SKIP and not (p/n).is_symlink() and len(rel.parts)<11]
    return result


def load_run(row):
    result=Path(row['result']); d=read(result)
    model=Path(d['inputs']['model']['path'])
    if digest(model)!=d['inputs']['model']['sha256']: raise ValueError('Evaluated PLY hash mismatch')
    root=next(p for p in result.parents if (p/'manifests/edgs_joint_manifest.json').exists())
    joint=read(root/'manifests/edgs_joint_manifest.json'); cfg=joint['request']['config']
    if not joint.get('complete'): raise ValueError('Incomplete joint manifest')
    receipts={}
    for name in ['camera','initialization','lama','local_selection']:
        if name not in joint['inputs']: continue
        src=joint['inputs'][name]; p=Path(src['path'])
        if digest(p)!=src['sha256']: raise ValueError('Input manifest changed: '+name)
        receipts[name]=read(p)
    camera=receipts['camera']; lama=receipts['lama']
    lip=lama['input_manifest']; lip=Path(lip['path'] if isinstance(lip,dict) else lip)
    lama_input=read(lip)
    init=receipts['initialization']; init_info=read(init['outputs']['diagnostics']['path'])
    sel=receipts.get('local_selection'); sel_info=read(sel['outputs']['summary']['path']) if sel else {}
    params=lama_input['parameters']; frames=list(lama_input['frames'].values())
    path_type=camera.get('trajectory',{}).get('type')
    if not path_type:
        path_type='hemisphere' if 'max_elevation_deg' in camera.get('trajectory',{}) else 'adaptive' if 'continuous_path' in camera.get('trajectory',{}) else 'undeclared'
    details=dict(iteration=d['inputs']['iteration'], training_budget=cfg.get('iterations'),
                 virtual_frames=camera['frame_count'], trajectory=path_type,
                 mask_dilation=params['dilation'], mask_shape=params.get('dilation_shape','ellipse'),
                 virtual_mask_mean_fraction=float(np.mean([f['mask_foreground_after']/np.prod(f['shape']) for f in frames])),
                 recursive_guide=lama['parameters']['recursive_guide'], init_use_depth=cfg['init']['use_depth'],
                 init_use_normal=cfg['init']['use_normal'], init_point_cap=cfg['init'].get('max_points'),
                 surface_init=cfg.get('surface_init',{}).get('mode','none'),
                 freeze_background=cfg.get('freeze_background','not recorded'), local_background=cfg.get('local_background'),
                 trainable_old=sel_info.get('trainable_old'), initial_old=sel_info.get('initial_old'),
                 trainable_old_fraction=(sel_info['trainable_old']/sel_info['initial_old']) if sel_info else None,
                 initialized_new_points=init_info.get('initialized_points'), depth_points=init_info.get('depth_points'),
                 pruning=cfg.get('pruning'), adaptive_density=cfg.get('adaptive_density'), loss=cfg.get('loss'))
    return dict(row=row, result=d, root=root, model=model, joint=joint, config=cfg,
                receipts=receipts, lama_input=lama_input, details=details)


def font(s): return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',s)
def fit(im, wh):
    scale=min(wh[0]/im.width,wh[1]/im.height)
    im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
    tile=Image.new('RGB',wh,'white'); tile.paste(im,((wh[0]-im.width)//2,(wh[1]-im.height)//2)); return tile
def rgb(p):
    with Image.open(p) as im: return im.convert('RGB')


def image_regions(folder, frames, mask_source, expected):
    rows=[]
    for stem in frames:
        a=np.asarray(rgb(folder/'renders'/f'{stem}.png'),dtype=np.float64)/255
        b=np.asarray(rgb(folder/'gt'/f'{stem}.png'),dtype=np.float64)/255
        if a.shape!=b.shape: raise ValueError('Render/GT dimensions differ')
        with Image.open(mask_source[stem]) as im:
            mask=np.asarray(im.convert('L').resize((a.shape[1],a.shape[0]),Image.Resampling.NEAREST))>127
        q=float(mask.mean()); err=np.mean((a-b)**2,axis=2); total=float(err.mean())
        hole=float(err[mask].sum()/err.size); outside=total-hole
        if not 0<q<1: raise ValueError('Empty/full benchmark mask')
        rows.append(dict(frame=stem,PSNR=psnr(total),masked_PSNR=psnr(hole),
                    inside_psnr=psnr(hole/q),outside_psnr=psnr(outside/(1-q)),mask_fraction=q,
                    full_mse=total,inside_mse_contribution=hole,outside_mse_contribution=outside,
                    rgb_mean_bias_hole_255=float(np.linalg.norm((a-b)[mask].mean(axis=0)*255)),
                    width=a.shape[1],height=a.shape[0]))
    summary={k:float(np.mean([r[k] for r in rows])) for k in ['PSNR','masked_PSNR','inside_psnr','outside_psnr','mask_fraction','full_mse','inside_mse_contribution','outside_mse_contribution','rgb_mean_bias_hole_255']}
    summary['outside_error_fraction']=summary['outside_mse_contribution']/summary['full_mse']
    summary['stored_psnr_difference']=summary['PSNR']-expected['PSNR']
    summary['stored_masked_psnr_difference']=summary['masked_PSNR']-expected['masked_PSNR']
    if abs(summary['stored_psnr_difference'])>1e-4 or abs(summary['stored_masked_psnr_difference'])>1e-4:
        raise ValueError('PNG/recorded PSNR mismatch: '+str(folder)+' '+str(summary))
    return dict(summary=summary,per_view=rows)


def figures(out, name, roots, measurements, masks):
    pm=measurements['PaintMesh best']['per_view']
    ranked=sorted(pm,key=lambda r:r['PSNR'])
    selected=[('median full PSNR',ranked[len(ranked)//2]['frame']),
              ('worst masked PSNR',min(pm,key=lambda r:r['masked_PSNR'])['frame'])]
    methods=['GT','PaintMesh best',*OTHER]
    cw,ch=460,300
    canvas=Image.new('RGB',(5*cw,2*(ch*2+85)),'white'); draw=ImageDraw.Draw(canvas)
    for row,(reason,stem) in enumerate(selected):
        y=row*(ch*2+85)
        ref=rgb(roots['PaintMesh best']/'gt'/f'{stem}.png')
        with Image.open(masks[stem]) as im: m=np.asarray(im.convert('L').resize(ref.size,Image.Resampling.NEAREST))>127
        yy,xx=np.where(m);box=(max(0,int(xx.min())-25),max(0,int(yy.min())-25),min(ref.width,int(xx.max())+26),min(ref.height,int(yy.max())+26))
        for col,method in enumerate(methods):
            im=ref.copy() if method=='GT' else rgb(roots[method]/'renders'/f'{stem}.png').resize(ref.size,Image.Resampling.BICUBIC)
            full=im.copy();ImageDraw.Draw(full).rectangle(box,outline='red',width=3)
            x=col*cw
            draw.text((x+8,y+5),method,font=font(22),fill='black')
            draw.text((x+8,y+33),stem+' | '+reason,font=font(12),fill='black')
            canvas.paste(fit(full,(cw-8,ch)),(x+4,y+60));canvas.paste(fit(im.crop(box),(cw-8,ch)),(x+4,y+ch+62))
            if method!='GT':
                r=next(r for r in measurements[method]['per_view'] if r['frame']==stem)
                draw.text((x+8,y+ch*2+62),f"PSNR {r['PSNR']:.2f} | mPSNR {r['masked_PSNR']:.2f}",font=font(16),fill='black')
    canvas.save(out/'comparison.jpg',quality=94)
    write(out/'figure_selection.json',dict(views=selected,rule='Median full PSNR and worst masked PSNR of selected PaintMesh; not selected for relative advantage',note='Display uses same ROI, native metric evaluation keeps original dimensions.'))


def pipeline_figure(out, run):
    inp=run['lama_input'];completion=run['receipts']['lama'];frames=sorted(inp['frames'])
    indices=[0,len(frames)//3,2*len(frames)//3]
    cw,ch=480,330; canvas=Image.new('RGB',(4*cw,3*(ch+50)),'white');draw=ImageDraw.Draw(canvas)
    for row,index in enumerate(indices):
        stem=frames[index];f=inp['frames'][stem];removed=Path(f['inputs']['removed_rgb']['path'])
        orig=Path(f['inputs']['reference_depth']['path']).parent.parent/'renders'/removed.name
        if not orig.exists(): raise FileNotFoundError('Missing original virtual RGB: '+str(orig))
        with Image.open(f['outputs']['color_mask']['path']) as im:mask=np.asarray(im.convert('L'))>127
        rem=rgb(removed);arr=np.array(rem);arr[mask]=arr[mask]*.55+np.array([255,30,30])*.45
        ims=[rgb(orig),rem,Image.fromarray(arr),rgb(completion['frames'][stem]['outputs']['color']['path'])]
        for col,(label,im) in enumerate(zip(['Original virtual','After removal','LaMa mask','LaMa RGB target'],ims)):
            draw.text((col*cw+5,row*(ch+50)+5),label+' / '+stem,font=font(18),fill='black')
            canvas.paste(fit(im,(cw-6,ch)),(col*cw+3,row*(ch+50)+40))
    canvas.save(out/'pipeline.jpg',quality=94)
    step=run['details']['iteration'];debug=run['root']/f'edgs_joint/debug/step_{step:06d}.npz'
    if debug.exists():
        with np.load(debug) as z: trained=Image.fromarray(np.rint(np.clip(z['rgb'],0,1)*255).astype(np.uint8))
        meta=debug.with_suffix('.json');metadata=read(meta) if meta.exists() else {}
        view_index=run['config'].get('debug',{}).get('view_index',0)
        stem=str(metadata.get('image_name',frames[view_index]))
        if stem not in completion['frames']:raise ValueError('Debug target frame unavailable')
        target=rgb(completion['frames'][stem]['outputs']['color']['path'])
        with np.load(debug) as z: prediction=z['rgb'].astype(np.float64)
        truth=np.asarray(target,dtype=np.float64)/255
        with Image.open(inp['frames'][stem]['outputs']['color_mask']['path']) as im: hole=np.asarray(im.convert('L'))>127
        if prediction.shape!=truth.shape or hole.shape!=truth.shape[:2]:raise ValueError('Debug/target dimensions differ')
        error=(prediction-truth)**2
        write(out/'training_fit.json',dict(scene=out.name,step=step,frame=stem,
            target='LaMa training RGB; not test GT',masked_region_PSNR=psnr(error[hole].mean()),
            full_PSNR=psnr(error.mean()),masked_region_MAE255=float(np.abs(prediction-truth)[hole].mean()*255),debug=str(debug)))
        plate=Image.new('RGB',(1200,460),'white');dr=ImageDraw.Draw(plate)
        for col,(label,im) in enumerate([('LaMa target / '+stem,target),('Training debug / step '+str(step),trained)]):
            dr.text((col*600+10,10),label,font=font(22),fill='black');plate.paste(fit(im,(590,400)),(col*600+5,50))
        plate.save(out/'training_view.jpg',quality=94)


def main():
    OUT.mkdir(exist_ok=True)
    report=read(REPORT); all_rows=candidates(report)
    selected=[];all_regions={}; all_comp=[]
    for name in report['common_scenes']:
        options=sorted([r for r in all_rows if r['scene']==name and r['eligible']],key=lambda r:(r['metrics']['PSNR'],r['metrics']['masked_PSNR']),reverse=True)
        run=None
        for row in options:
            try: run=load_run(row);break
            except (ValueError,FileNotFoundError,StopIteration) as exc:
                row['eligible']=False;row['excluded_reasons'].append(str(exc));print(name,'REJECT',exc,flush=True)
        if run is None:raise RuntimeError('No verified result for '+name)
        row=run['row']; row['selected']=True;target=OUT/name;target.mkdir(exist_ok=True)
        result=Path(row['result']);evalroot=result.parent;root=run['root'];joint=run['joint']
        for sub,src in {'run':root,'model':run['model'].parents[2],'point_cloud.ply':run['model'],
                        'evaluation':evalroot,'results.json':result,'renders':evalroot/'renders',
                        'gt':evalroot/'gt','masks':evalroot/'masks','training_config.json':root/'edgs_joint/config.resolved.json',
                        'joint_manifest.json':root/'manifests/edgs_joint_manifest.json',
                        'camera.json':joint['inputs']['camera']['path'],'lama_manifest.json':joint['inputs']['lama']['path'],
                        'initialization_manifest.json':joint['inputs']['initialization']['path']}.items():link(target/sub,src)
        d=run['result'];frames=d['inputs']['frames'];masks={Path(v['path']).stem:Path(v['path']) for v in d['inputs']['official_masks']}
        roots={'PaintMesh best':evalroot,**{m:BASE/repo/'output'/name for m,repo in OTHER.items()}}
        measurements={}
        for method,folder in roots.items():
            expected=row['metrics'] if method=='PaintMesh best' else report['metrics'][method][name]
            measurements[method]=image_regions(folder,frames,masks,expected)
            all_comp.append(dict(scene=name,method=method,**expected))
        all_regions[name]=measurements
        write(target/'region_diagnosis.json',measurements)
        detail=dict(scene=name,selection_policy=POLICY,result_source=record(result),ply_sha256_verified=d['inputs']['model']['sha256'],
                    run_root=str(root),configuration=run['details'],full_config=run['config'],
                    metrics=row['metrics'],frame_count=d['frame_count'],candidate_count=len(options),
                    previous_report_metrics=report['metrics']['PaintMesh 5000'][name])
        write(target/'selection.json',detail);selected.append(detail)
        figures(target,name,roots,measurements,masks);pipeline_figure(target,run)
        print(name,'selected',row['iteration'],f"PSNR {row['metrics']['PSNR']:.4f}",root,flush=True)
    write(OUT/'candidates.json',all_rows)
    write(OUT/'selection.json',dict(created_at=datetime.now().astimezone().isoformat(),policy=POLICY,source_report=record(REPORT),candidate_count=len(all_rows),scenes=selected))
    write(OUT/'region_diagnosis.json',all_regions);csvwrite(OUT/'comparison.csv',all_comp)
    write(OUT/'training_fit.json',[read(OUT/s['scene']/'training_fit.json') for s in selected if (OUT/s['scene']/'training_fit.json').exists()])
    aggregates={}
    for method in ['PaintMesh best',*OTHER]:
        vals=[r for r in all_comp if r['method']==method];weights=[report['test_frame_counts'][r['scene']] for r in vals]
        aggregates[method]={k:float(np.average([r[k] for r in vals],weights=None if k=='FID' else weights)) for k in KEYS}
    write(OUT/'comparison.json',dict(policy=POLICY,per_scene=all_comp,frame_weighted_average=aggregates))
    header=['# PaintMesh 历史最佳结果合集','',POLICY,'',
            '所有保留实验参与候选扫描；已否定并删除的 circle30/dilation10 结果不在候选中。每场景 PLY 哈希、训练输入 manifest、完整测试图片以及重新从 PNG 计算的 PSNR/masked_PSNR 已核对。',
            '图像仅用于 CPU 诊断，没有重训、没有合成补全图。下面各指标来自相同场景完整测试集。','',
            '| 场景 | 步数 | 相机 | 表面初始化 | PSNR | masked_PSNR | 渲染对照 |', '|---|---:|---|---|---:|---:|---|']
    for s in selected:
        n=s['scene'];c=s['configuration'];m=s['metrics'];header.append(f"| [{n}]({n}/) | {c['iteration']} | {c['trajectory']} {c['virtual_frames']} | {c['surface_init']} | {m['PSNR']:.4f} | {m['masked_PSNR']:.4f} | [同相机对比]({n}/comparison.jpg) |")
    header+=['','## 使用入口','',
             '`<scene>/point_cloud.ply`：选中快照；`renders/gt/masks`：测试产物；`model`：该快照模型目录；`run`：完整原始实验。',
             '`training_config.json`：该运行完整配置，`selection.json` 同时记录选择快照的步数；例如 10000 步运行选用 5000 步快照时，两者不混淆。',
             '`camera.json`、`lama_manifest.json`、`initialization_manifest.json`：输入溯源；`pipeline.jpg`：同一虚拟相机的删除与补全过程；`training_view.jpg`（如有）：已保存训练调试图。',
             '这些大产物使用相对符号链接汇总，不复制或修改原始数据。要复制到另一台机器需同时复制目标数据或解引用链接。',
             '[全部候选与排除原因](candidates.json) · [全指标 CSV](comparison.csv) · [区域误差诊断](region_diagnosis.json) · [逐场景原因分析](DIAGNOSIS.zh-CN.md)','',
             '## 完整七项指标','', '| 场景 | 方法 | '+' | '.join(KEYS)+' |','|---|---|'+'---:|'*len(KEYS)]
    for r in all_comp:header.append('| '+r['scene']+' | '+r['method']+' | '+' | '.join(f'{r[k]:.6f}' for k in KEYS)+' |')
    header+=['','## 全部 425 测试帧汇总（FID 为场景平均）','', '| 方法 | '+' | '.join(KEYS)+' |','|---|'+'---:|'*len(KEYS)]
    for m,v in aggregates.items():header.append('| '+m+' | '+' | '.join(f'{v[k]:.6f}' for k in KEYS)+' |')
    header+=['','AuraFusion360/redbull 使用已登记的初始化修正版，其他修复记录见三个方法 output/EVALUATION.md。旧总对比表保持原样，本合集另外记录历史择优范围。']
    (OUT/'README.md').write_text('\n'.join(header)+'\n')
    for s in selected:
        p=OUT/s['scene'];(p/'README.md').write_text(f"# {s['scene']}\n\n{POLICY}\n\n[选择与实际配置](selection.json) · [原始指标](results.json) · [测试对照](comparison.jpg) · [输入流程](pipeline.jpg) · [区域误差](region_diagnosis.json)\n\n最终高斯：[point_cloud.ply](point_cloud.ply)，原实验：[run](run/)，完整来源见 selection.json。\n")
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
