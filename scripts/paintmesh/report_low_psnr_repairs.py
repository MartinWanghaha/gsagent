#!/usr/bin/env python3
"""Read-only metrics audit; separates baselines from isolated corrected variants."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

BASE=Path('/home/member/wyc')
AUDIT=BASE/'gsagent/logs/low_psnr_audit_20261026'
PM_JOB=BASE/'gsagent/logs/paintmesh_full_r4_gpu1_20261025_105925'
PM_FIX=BASE/'gsagent/output/paintmesh/Inpaint360GS/inpaint360/cube/removal/full_r4_adaptive180_holealpha95_20261026/inpaint/full_r4_adaptive180_holealpha95_20261026'
METRICS=('PSNR','masked_PSNR','SSIM','masked_SSIM','LPIPS','masked_LPIPS','FID')
NATIVE=('PSNR_full','PSNR_masked','SSIM_full','SSIM_masked','LPIPS_full','LPIPS_masked','FID')
SCENES=('bag','car','cone_red','cone_yellow','cube','doppelherz','fruits','garden_toys','redbull','toys','truck')

def read(path):return json.loads(path.read_text())
def optional(path):return read(path) if path.is_file() else {}
def atomic(path,text):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(text);tmp.replace(path)
def save(path,obj):atomic(path,json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def native(path):
    result={}
    for key,value in optional(path).get('per_scene_results',{}).items():
        scene=key.split('/inpaint/')[0].split('/')[-1]
        if scene not in SCENES:continue
        if all(k in value and math.isfinite(value[k]) for k in NATIVE):
            result[scene]=dict(zip(METRICS,(value[k] for k in NATIVE)))
    return result

def gg_repairs():
    # The vendor evaluator replaces its aggregate JSON on each single-scene
    # invocation. Each completed scene has its own preserved result snapshot.
    root=BASE/'gaussian-grouping/output/low_psnr_repair_20261026/workspace/output/inpaint360'
    result={}
    for scene in SCENES:
        result.update({s:v for s,v in native(root/scene/'repair_results.json').items() if s==scene})
    return result

def report():
    sources={
        'Inpaint360GS':BASE/'Inpaint360GS/output/paper_r4_all_scenes/comparison/summary.json',
        'Gaussian Grouping':BASE/'gaussian-grouping/output/inpaint360/all_scene_evaluation_results.json',
        'AuraFusion360':BASE/'AuraFusion360_official/output/inpaint360/all_scene_evaluation_results.json',
        'PaintMesh':PM_JOB/'comparison.json'}
    pm_summary=read(sources['PaintMesh'])
    pm=pm_summary.get('original_metrics',pm_summary['metrics'])
    original={
        'Inpaint360GS':{r['scene']:r['metrics'] for r in read(sources['Inpaint360GS'])['per_scene']},
        'Gaussian Grouping':native(sources['Gaussian Grouping']),
        'AuraFusion360':native(sources['AuraFusion360']),
        'PaintMesh 5000':pm['PaintMesh 5000'],
        'PaintMesh 10000':pm['PaintMesh 10000']}
    source_records={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in sources.items()}
    repairs={
        'Inpaint360GS':native(BASE/'Inpaint360GS/output/low_psnr_repair_20261026/scenes/car/workspace/output/inpaint360/all_scene_evaluation_results.json'),
        'Gaussian Grouping':gg_repairs(),
        'AuraFusion360':native(BASE/'AuraFusion360_official/output/low_psnr_repair_20261026/workspace/output/inpaint360/all_scene_evaluation_results.json'),
        'PaintMesh 5000':{},'PaintMesh 10000':{}}
    for step,name in ((5000,'inpainted_3dgs_5000'),(10000,'inpainted_3dgs')):
        value=optional(PM_FIX/name/'evaluation/inpaint360_r4/results.json')
        if value.get('complete'):repairs[f'PaintMesh {step}']['cube']=value['metrics']
    lows=[dict(method=m,scene=s,PSNR=v['PSNR']) for m,rows in original.items() if m!='PaintMesh 5000' for s,v in rows.items() if v['PSNR']<20]
    scheduled={('Inpaint360GS','car'),('Gaussian Grouping','cone_red'),('Gaussian Grouping','garden_toys'),('AuraFusion360','redbull'),('PaintMesh 10000','cube')}
    common=sorted(set.intersection(*(set(rows) for rows in original.values())))
    macro={m:{k:sum(rows[s][k] for s in common)/len(common) for k in METRICS} for m,rows in original.items()} if common else {}
    # Select the registered correction for each repaired case, never the best
    # score across runs. Baseline tables and source artifacts remain separate.
    latest={m:dict(rows,**repairs[m]) for m,rows in original.items()}
    latest_lows=[dict(method=m,scene=s,PSNR=v['PSNR']) for m,rows in latest.items() if m!='PaintMesh 5000' for s,v in rows.items() if v['PSNR']<20]
    result=dict(updated_at=datetime.now().astimezone().isoformat(),threshold=20,selection='final full-image PSNR',
        original=original,corrected_variants=repairs,original_sources=source_records,low_psnr=lows,
        latest_registered_results_low_psnr=latest_lows,
        common_baseline_scenes=common,common_baseline_scene_macro=macro,
        additional_low_cases_needing_review=[r for r in lows if (r['method'],r['scene']) not in scheduled],
        repaired_still_below_20=[dict(method=m,scene=s,PSNR=v['PSNR']) for m,rows in repairs.items() if m!='PaintMesh 5000' for s,v in rows.items() if v['PSNR']<20],
        missing_baselines={m:[s for s in SCENES if s not in rows] for m,rows in original.items()},
        gpu1_gg_status=optional(AUDIT/'gg_repair_status.json'),gpu1_remaining_status=optional(AUDIT/'remaining_repair_status.json'),
        gg_background_diagnosis_status=optional(AUDIT/'gg_background_diagnosis_status.json'),
        gg_background_diagnosis=optional(AUDIT/'gg_cone_red_background/diagnosis.json'),
        note='Repairs never replace baseline metrics. Changed initializer in Aura is a method variant. Test GT and masks only used for evaluation and diagnosis.')
    if not (AUDIT/'baseline_snapshot.json').exists():save(AUDIT/'baseline_snapshot.json',dict(original=original,sources=source_records,created_at=result['updated_at']))
    save(AUDIT/'comparison.json',result)
    lines=['# PSNR < 20 排查与修复对照','',f"更新：{result['updated_at']}",'',
        '筛选最终检查点的全图 PSNR。原始结果保留，修正实验单独列出；“待完成”不是零分，也不表示修复失败。',
        '测试 GT 和评估 mask 只用于诊断/评估，未用于生成训练目标、选择点或修正训练 mask。',
        '', '[诊断依据与修复范围](AUDIT.zh-CN.md)','',
        '| 方法 | 已有最终指标 | 尚未完成场景 |','|---|---:|---|']
    for m,rows in original.items():lines.append(f"| {m} | {len(rows)}/11 | {', '.join(result['missing_baselines'][m]) or '无'} |")
    lines+=['','采用已登记修正版后，当前仍低于 20 的最终结果：'+('; '.join(f"{r['method']}/{r['scene']} = {r['PSNR']:.6f}" for r in latest_lows) or '无')+'。原始低分保留在后文，不能据此误判为尚未续跑。']
    lines+=['',f"共同完成的 {len(common)} 个场景：{', '.join(common)}。以下只统计原始基线，修复版单独列出。",'',
        '| 方法 | PSNR ↑ | masked PSNR ↑ | SSIM ↑ | masked SSIM ↑ | LPIPS ↓ | masked LPIPS ↓ | 场景 FID 均值 ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,v in macro.items():lines.append(f'| {m} | '+' | '.join(f'{v[k]:.6f}' for k in METRICS)+' |')
    lines+=['','| 方法 | 场景 | 原始 PSNR | 修正后 PSNR | 差值 |','|---|---|---:|---:|---:|']
    for r in lows:
        v=repairs.get(r['method'],{}).get(r['scene'])
        lines.append(f"| {r['method']} | {r['scene']} | {r['PSNR']:.6f} | "+(f"{v['PSNR']:.6f} | {v['PSNR']-r['PSNR']:+.6f} |" if v else '待完成 | — |'))
    lines+=['','## 全部七项指标（原始 → 修正）','',
        '| 方法/场景/版本 | PSNR ↑ | masked PSNR ↑ | SSIM ↑ | masked SSIM ↑ | LPIPS ↓ | masked LPIPS ↓ | FID ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    cases=[(r['method'],r['scene']) for r in lows]+[('PaintMesh 5000','cube')]
    for m,s in cases:
        for label,rows in [('原始',original),('修正',repairs)]:
            v=rows[m].get(s)
            lines.append(f'| {m}/{s}/{label} | '+(' | '.join(f'{v[k]:.6f}' for k in METRICS) if v else '待完成 | — | — | — | — | — | —')+' |')
    lines+=['','## GPU 1 后台状态','',f"- Gaussian Grouping：`{result['gpu1_gg_status']}`",f"- 后续队列：`{result['gpu1_remaining_status']}`",'',
        '指标使用同一 1/4 分辨率测试集及官方 mask，crop=1、resize=1。Aura 的初始化修正版属于方法变体，不能标为未经修改的官方复现。',
        '本表不混算缺失场景，不用修正结果覆盖原始基线，也不把各场景 FID 平均值称为 pooled FID。']
    if result['additional_low_cases_needing_review']:lines+=['',f"新增待诊断低分场景：{result['additional_low_cases_needing_review']}"]
    if result['repaired_still_below_20']:lines+=['',f"修正后仍低于 20，需进一步区分背景重建误差与补全过程问题：{result['repaired_still_below_20']}"]
    if result['gg_background_diagnosis_status']:lines+=['',f"GG/cone_red 分阶段背景诊断：`{result['gg_background_diagnosis_status']}`",'','[修正前后对照图](GG_cone_red_recheck.jpg)']
    atomic(AUDIT/'comparison.md','\n'.join(lines)+'\n')
    return result

if __name__=='__main__':
    value=report();print(json.dumps(dict(low_psnr=value['low_psnr'],report=str(AUDIT/'comparison.md')),ensure_ascii=False))
