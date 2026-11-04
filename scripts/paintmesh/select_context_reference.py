#!/usr/bin/env python3
"""Select one fixed reference using only the predeclared diagnostic train holes."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess

import numpy as np

from context_completion_io import case, mask, psnr, rgb, verify_predictions, PARAMETERS
from edgs_inpaint_io import record, verify_record
from run_geometry_ablation import read, write, now, atomic, REPO
from run_reference_context_pilot import codes, PYTHON, report


def prepare(parent, scene):
    source = parent / 'scenes' / scene
    dest = parent / 'reference_selection' / scene
    request_path = dest / 'prediction_request.json'
    if request_path.exists():
        raise ValueError('Reference search already prepared; use the recorded request')
    inp = read(source / 'screen/input_manifest.json')
    diagnostic = [c for c in inp['pseudo_cases'] if c['split'] == 'diagnostic']
    if len(diagnostic) != 2:
        raise ValueError('Expected exactly the two predeclared diagnostic photos')
    prior = read(source / 'prior_guide.json')
    row = read(source / 'source.json')
    lama = read(verify_record(row['inputs']['lama']))
    stems = sorted(lama['frames'])
    candidates = [stems[c['index']] for c in prior['candidates']]
    if len(candidates) != len(set(candidates)) or prior['reference'] not in candidates:
        raise ValueError('Invalid candidate set')
    cases = []
    for ref in candidates:
        reference = verify_record(lama['frames'][ref]['outputs']['color'])
        for c in diagnostic:
            cases.append(case(f'{ref}/{c["name"]}', verify_record(c['image']),
                verify_record(c['mask']), reference, dest/'predictions'/ref/(c['name']+'.png'), 0))
    prediction = dict(model_path=read(source/'screen/request.json')['model_path'],
        parameters=PARAMETERS, cases=cases, receipt=str(dest/'prediction_receipt.json'))
    write(request_path, prediction)
    write(dest/'selection_policy.json', dict(created_at=now(), scene=scene,
        producer=record(Path(__file__)), parent_input=record(source/'screen/input_manifest.json'),
        diagnostic_cases=diagnostic, candidate_frames=candidates,
        selection='Maximum mean ROI PSNR on the two diagnostic photos; lexical frame ID breaks ties',
        uses_validation_for_ranking=False, uses_test_data=False,
        validation_gate_unchanged=inp['screen_gate'], parameters=PARAMETERS))
    print(request_path)


def score(parent, scene):
    dest = parent / 'reference_selection' / scene
    policy = read(dest/'selection_policy.json')
    verify_record(policy['producer'])
    verify_predictions(dest/'prediction_request.json')
    rows = []
    for ref in policy['candidate_frames']:
        values = []
        for c in policy['diagnostic_cases']:
            gt = rgb(verify_record(c['image'])); m = mask(verify_record(c['mask']))
            out = rgb(dest/'predictions'/ref/(c['name']+'.png'))
            independent = rgb(parent/'scenes'/scene/'screen/pseudo/independent'/(c['name']+'.png'))
            values.append(dict(name=c['name'], roi_psnr=psnr(out,gt,m),
                independent_roi_psnr=psnr(independent,gt,m)))
        rows.append(dict(reference=ref, diagnostic=values,
            mean_roi_psnr=float(np.mean([x['roi_psnr'] for x in values])),
            mean_delta=float(np.mean([x['roi_psnr']-x['independent_roi_psnr'] for x in values]))))
    rows.sort(key=lambda r:(-r['mean_roi_psnr'],r['reference']))
    result = dict(created_at=now(), selected_reference=rows[0]['reference'], ranking=rows,
        policy=record(dest/'selection_policy.json'), predictions=record(dest/'prediction_receipt.json'),
        uses_validation_for_ranking=False, uses_test_data=False)
    output = dest/'selection.json'
    if output.exists() and read(output)['selected_reference'] != result['selected_reference']:
        raise ValueError('Selection changed')
    if not output.exists(): write(output,result)
    lines = ['# cube 固定参考重选','','只用预先划定的两张诊断训练照片排序。其余四张训练照片用于筛查，test 不参与选择。',
        '',f'选定参考：{rows[0]["reference"]}。筛查门槛保持不变；若验证失败，不继续搜索验证集最优帧。',
        '', '| 参考 | 诊断 ROI PSNR | 比无参考补全 Δ |','|---|---:|---:|']
    lines += [f'| {r["reference"]} | {r["mean_roi_psnr"]:.6f} | {r["mean_delta"]:+.6f} |' for r in rows]
    atomic(dest/'README.md','\n'.join(lines)+'\n')
    print('Selected',rows[0])


def launch(parent, scene):
    selection_path = parent/'reference_selection'/scene/'selection.json'
    selection = read(selection_path)
    verify_record(selection['policy']); verify_record(selection['predictions'])
    source = parent/'scenes'/scene
    root = parent/f'{scene}_reference_selected'
    if root.exists(): raise ValueError('Selected reference experiment already exists')
    req = copy.deepcopy(read(parent/'request.json'))
    if req['code'] != codes(): raise ValueError('Parent implementation changed')
    spec = copy.deepcopy(next(x for x in req['scenes'] if x['scene']==scene))
    dest = root/'scenes'/scene; dest.mkdir(parents=True)
    shutil.copyfile(source/'source.json', dest/'source.json')
    previous = read(source/'prior_guide.json')
    lama = read(previous['source_lama']['path']); stems = sorted(lama['frames'])
    chosen = selection['selected_reference']
    # The IO adapter's legacy mode label identifies guided frames only. This experiment
    # has no RGB warping, and records the actual context operation explicitly.
    eligible = {s for s,v in previous['frames'].items()
        if v['mode'] in ('warp_and_lama_residual','fully_supported_warp')}
    eligible.add(previous['reference']); eligible.discard(chosen)
    prior = dict(created_at=now(), source_lama=previous['source_lama'], camera=previous['camera'],
        reference=chosen, reference_index=stems.index(chosen),
        parent_guide=record(source/'prior_guide.json'), selection=record(selection_path),
        policy='Fixed full-mask context; legacy frame mode only adapts eligibility; no hard RGB warp',
        frames={s:dict(mode='warp_and_lama_residual' if s in eligible else 'unchanged',
            actual_operation='full_mask_fixed_context' if s in eligible else 'retain_independent') for s in stems})
    write(dest/'prior_guide.json',prior)
    spec.update(source=record(dest/'source.json'),prior_guide=record(dest/'prior_guide.json'))
    req.update(created_at=now(),root=str(root),scenes=[spec],reference_selection=record(selection_path),
        parent_experiment=record(parent/'request.json'))
    write(root/'request.json',req)
    for name,target in [('independent_control',spec['baseline']),('hard_warp_control',spec['hard_warp'])]:
        (dest/name).symlink_to(target,target_is_directory=True)
    atomic(root/'TRAINING_PLAN.zh-CN.md',f'''# {scene} 参考重选验证

承接 [原方案](../TRAINING_PLAN.zh-CN.md)。本子实验唯一新增因素为参考帧重选：{previous['reference']} → {chosen}。
候选沿用原有 18 个帧，仅由两张诊断训练照片的补全 ROI PSNR 排序；没有使用验证照片或 test 进行排序。
固定选定帧后，在原四张验证训练照片和六个虚拟视角上检查 A/B。数字门槛和视觉检查不变。
二维筛查通过才训练 10000 步，保存并评估 5000/10000；几何目标、初始点云、损失、剪枝和冻结策略沿用原对照。
此处的二维 ROI PSNR 只说明已知背景自测，不等于新物体洞内的正确性。
两个场景的最终结论需要联合父目录 truck 的实际 test 评估。

[参考排序](../reference_selection/{scene}/README.md) · [状态](scenes/{scene}/status.json) · [指标](comparison.md)
''')
    atomic(root/'README.md',f'# {scene} 固定参考重选\n\n[方案](TRAINING_PLAN.zh-CN.md) · [状态](scenes/{scene}/status.json) · [指标](comparison.md)\n')
    report(root)
    command=[PYTHON,str(Path(__file__).with_name('run_reference_context_pilot.py')),'--root',str(root),'--worker',scene]
    with (dest/'worker.log').open('a') as log:
        child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'workers.json',{scene:dict(pid=child.pid,gpu=spec['gpu'])})
    print(root,child.pid)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','score','launch'))
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--scene',choices=('cube','truck'),required=True)
    a=p.parse_args();globals()[a.action](a.parent.resolve(),a.scene)
