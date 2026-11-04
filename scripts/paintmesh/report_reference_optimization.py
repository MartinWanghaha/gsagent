#!/usr/bin/env python3
"""Report all fixed-reference pilot branches, including rejected/pending runs."""
import argparse
import fcntl
from pathlib import Path
import time

from run_geometry_ablation import METRICS, read, write, atomic, now
from run_muti_plan import validate_evaluation
from edgs_inpaint_io import record


def report(root):
    request=read(root/'request.json');rows=[];states=[];pending=[]
    variants=[]
    for scene in request['scenes']:
        s=scene['scene'];base=Path(scene['baseline'])
        variants += [(s,'independent',base,None),(s,'hard_reference',Path(scene['hard_warp']),None)]
        for v in ('A_context','B_context_opt10'):
            variants.append((s,'context_ref_initial/'+v,root/'scenes'/s/v,root/'scenes'/s/'screen/review.json'))
    chosen=root/'cube_reference_selected/scenes/cube'
    for v in ('A_context','B_context_opt10'):
        variants.append(('cube','context_ref_00073/'+v,chosen/v,chosen/'screen/review.json'))
    variants.append(('truck','half_reference',root/'truck_soft_reference/scenes/truck',None))
    for path in (root/'scenes/truck',root/'scenes/cube',chosen,root/'truck_soft_reference/scenes/truck'):
        p=path/'status.json'
        states.append(dict(folder=str(path.relative_to(root)),**(read(p) if p.exists() else dict(state='not_started'))))
    lines=['# 共享参考补全：全部优化分支','','更新：'+now(),'',
        '所有评估使用匹配对照与完整 test 集，5000 为主要节点，10000 为稳定性复核。无结果不填数；未通过筛查组仍保留记录。',
        '', '| 场景 | 步数 | 组 | '+' | '.join(METRICS)+' | ΔPSNR | Δmasked_PSNR |',
        '|---|---:|---|'+'---:|'*(len(METRICS)+2)]
    for s,name,folder,review in variants:
        base=Path(next(x for x in request['scenes'] if x['scene']==s)['baseline'])
        rejected=review and review.exists() and name.split('/')[-1] not in read(review)['approved_variants']
        for step in (5000,10000):
            p=folder/f'inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json'
            if not p.exists():
                pending.append(dict(scene=s,variant=name,step=step,state='screen_rejected' if rejected else 'pending'));continue
            baseline=read(base/f'inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json');result=read(p)
            validate_evaluation(result,baseline);m=result['metrics'];d={k:m[k]-baseline['metrics'][k] for k in METRICS}
            rows.append(dict(scene=s,variant=name,step=step,metrics=m,delta=d,result=record(p)))
            lines.append(f'| {s} | {step} | {name} | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+f' | {d["PSNR"]:+.6f} | {d["masked_PSNR"]:+.6f} |')
    lines+=['','## 后台任务','', '| 分支 | 状态 | 阶段 | 进度 |','|---|---|---|---|']
    for state in states:
        lines.append('| '+state['folder']+' | '+state['state']+' | '+str(state.get('stage',''))+' | '+str(state.get('progress') or '')+' |')
    lines+=['','## 尚无结果的组','','| 场景 | 组 | 步数 | 原因 |','|---|---|---:|---|']
    lines += [f'| {p["scene"]} | {p["variant"]} | {p["step"]} | {p["state"]} |' for p in pending]
    lines += ['','cube 重选参考与 truck 减弱引导是两个诊断分支；即便各自改善，也不等于同一个统一方法已在两场景验证成功。未安排全场景训练。',
        '','[本轮追加方案](OPTIMIZATION_PLAN.zh-CN.md) · [最初上下文试验](TRAINING_PLAN.zh-CN.md)','']
    write(root/'optimization_comparison.json',dict(updated_at=now(),rows=rows,pending=pending,states=states,all_scene_started=False,
        limitation='Scene-specific diagnostic branches; not a shared-policy generalization result'))
    atomic(root/'optimization_comparison.md','\n'.join(lines))
    active=[s for s in states if s['folder'] in ('cube_reference_selected/scenes/cube','truck_soft_reference/scenes/truck')]
    return len(active)==2 and all(s['state'] in ('complete','failed','screen_rejected','stopped') for s in active)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    root=a.root.resolve()
    with (root/'optimization_report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            done=report(root)
            if done or not a.watch:break
            time.sleep(30)
