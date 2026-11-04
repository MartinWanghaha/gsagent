#!/usr/bin/env python3
"""Keep publication descriptions in sync with the reviewed-mask retraining job."""
import csv
import fcntl
import io
import time
from pathlib import Path

from run_reviewed_retrack_training import DEFAULT_JOB, BASE, METRICS, read, write, atomic, now, evaluation


def refresh(job):
    request=read(job/'request.json');status=read(job/'status.json');fit_values={};fit_count=0
    for s in request['scenes']:
        state=status['scenes'][s['key']];case=job/'scenes'/s['key'];p=evaluation(s)
        result=read(p) if (case/'receipts/evaluate_5000.json').exists() and p.exists() else None
        root=BASE/'best_results'/s['scene'] if s['group']=='best_results' else BASE/'fit_release'/s['scene']
        if not root.exists():continue
        text=[f'# {s["group"]} / {s["scene"]}：新跟踪 mask 重训','',
              f'状态：{state["state"]}；当前阶段：{state.get("stage","pending")}。5000步，1/4分辨率，独立LaMa。','',
              '[全场景对比](../comparison.md) · [逐视角评估](results.json) · [实际训练配置](training_config.json)','',
              f'对象ID：{s["target_ids"]}；MASK_DILATION={s["mask_dilation"]}；日志：{case}/logs。','',
              '入口：run/ 为完整运行；model/、point_cloud.ply 为模型；renders/gt/masks 为测试评估；lama_input/lama_output 为补全输入输出；tracker/ 为已复核的跟踪结果。尚未生成的入口可能暂不可用。','']
        if result:
            text+=['| '+' | '.join(METRICS)+' |','|'+'---:|'*len(METRICS),
                   '| '+' | '.join(f'{result["metrics"][m]:.6f}' for m in METRICS)+' |','']
        else:text+=['本轮尚未评估，不显示旧指标。','']
        atomic(root/'README.md','\n'.join(text))
        if s['group']=='fit':
            if state['state']=='complete':fit_count+=1
            if result:fit_values[s['scene']]=dict(frame_count=result['frame_count'],**result['metrics'])
    out=BASE/'fit_release';payload=dict(updated_at=now(),complete=fit_count==11,evaluated=len(fit_values),
            completed=fit_count,total=11,scenes=fit_values,training_job=str(job))
    if len(fit_values)==11:
        payload['scene_macro']={m:sum(v[m] for v in fit_values.values())/11 for m in METRICS}
        payload['frame_weighted_fid_scene_mean']={m:payload['scene_macro'][m] if m=='FID' else sum(v[m]*v['frame_count'] for v in fit_values.values())/sum(v['frame_count'] for v in fit_values.values()) for m in METRICS}
    write(out/'metrics.json',payload)
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=['scene','frame_count',*METRICS]);w.writeheader()
    for scene,v in fit_values.items():w.writerow(dict(scene=scene,**v))
    atomic(out/'metrics.csv',stream.getvalue())
    write(out/'release.json',dict(updated_at=now(),complete=fit_count==11,state='complete' if fit_count==11 else 'retraining_from_reviewed_masks',
          live_status='../fit/status.json',live_comparison='../fit/comparison.md',job=str(job)))
    return all(s['state'] in ('complete','failed','blocked_memory_risk') for s in status['scenes'].values())


if __name__=='__main__':
    with (DEFAULT_JOB/'publication_refresh.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if refresh(DEFAULT_JOB):break
            time.sleep(30)
