#!/usr/bin/env python3
"""Rebuild best_results and fit from reviewed masks, with two durable GPU workers."""
import argparse
import copy
import csv
import fcntl
import io
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

import run_fit_test_poses as fit
import run_hard_reference_all as guard
from edgs_inpaint_io import load_config, record, verify_record
from run_muti_plan import PYTHON, changes, validate_evaluation
from run_geometry_ablation import REPO, METRICS, read, write, atomic, now, sha

BASE = REPO / 'output/paintmesh/Inpaint360GS'
SCRIPT = Path(__file__).resolve()
DEFAULT_JOB = REPO / 'logs/retrain_reviewed_best_fit_20261103'
TRACK_JOB = REPO / 'logs/retrack_best_fit_20261103'
POLICY = dict(idle_seconds=0, reserve_mib=1536, own_limit_mib=22500,
              sample_seconds=3, disk_free_gib=40)
REQUIRED = {'inpaint_3_3': 11000, 'inpaint_4_4': 12000,
            'inpaint_5_5': 16000, 'inpaint_7_8': 10000, 'evaluate_5000': 5000}


def codes():
    return {**fit.codes(), str(SCRIPT.relative_to(REPO)): sha(SCRIPT)}


def check_code(req):
    if req['code'] != codes():
        raise RuntimeError('Implementation changed after request snapshot')


def link(target, dest):
    target, dest = Path(target), Path(dest)
    if dest.is_symlink():
        if dest.resolve() == target.resolve(): return
        dest.unlink()
    elif dest.exists():
        raise ValueError('Refusing to replace unmanaged regular entry: ' + str(dest))
    dest.symlink_to(os.path.relpath(target, dest.parent))


def evaluation(s):
    return Path(s['inpaint_root']) / 'inpainted_3dgs/evaluation/inpaint360_r4/results.json'


def prepare(job):
    job.mkdir(parents=True, exist_ok=True)
    if (job/'request.json').exists():
        check_code(read(job/'request.json')); return
    original = read(BASE/'fit/request.json')
    fit_specs = {s['scene']: s for s in original['scenes']}
    cases = read(TRACK_JOB/'cases.json')
    specs = []
    # Freeze all metadata before any original output is replaced.
    for c in cases:
        key, group, scene = c['key'], c['group'], c['scene']
        case = job/'scenes'/key; case.mkdir(parents=True, exist_ok=True)
        f = fit_specs[scene]
        if group == 'fit':
            output = BASE/'fit/scenes'/scene/'inpaint'
            raw_config = read(BASE/'fit/scenes'/scene/'config.json')
            cfg = load_config(BASE/'fit/scenes'/scene/'config.json')
            previous = read(evaluation(dict(inpaint_root=str(output))))
        else:
            output = BASE/'best_results'/scene/'run'
            raw_config = read(BASE/'best_results'/scene/'training_config.json')
            cfg = load_config(BASE/'best_results'/scene/'training_config.json')
            previous = read(BASE/'best_results'/scene/'results.json')
        cfg['iterations'] = 5000
        cfg['init']['max_points'] = None
        write(case/'previous_config.json', raw_config)
        write(case/'config.json', cfg)
        write(case/'config_diff.json', changes(raw_config, cfg))
        write(case/'previous_evaluation.json', previous)
        spec = dict(key=key, group=group, scene=scene, config=record(case/'config.json'),
                    removal_root=str(Path(c['tracker']).parent), inpaint_root=str(output),
                    source_path=f['source_path'], target_ids=f['target_ids'],
                    mask_dilation=f['mask_dilation'], frames=f['frames'],
                    camera=record(Path(c['tracker'])/'virtual_cameras.json'),
                    source=copy.deepcopy(read(f['source']['path'])),
                    previous_evaluation=record(case/'previous_evaluation.json'),
                    old_output_target=str(output.resolve()),
                    tracking_publication=str(TRACK_JOB/key/'published.json'))
        specs.append(spec)
        write(case/'status.json', dict(state='pending', group=group, scene=scene))
    # Pair groups by scene; both workers claim disjoint queue entries with flock.
    specs.sort(key=lambda s: (list(fit_specs).index(s['scene']), s['group']))
    baselines = {}
    for method, b in original['baselines'].items():
        dst = job/'baseline_snapshot'/f'{method}.json'; dst.parent.mkdir(exist_ok=True)
        shutil.copy2(b['snapshot']['path'], dst); baselines[method] = record(dst)
    for group in ('best_results', 'fit'):
        dst = job/'previous_reports'/group; dst.mkdir(parents=True, exist_ok=True)
        for name in ('README.md', 'comparison.md', 'comparison.json', 'comparison.csv', 'status.json', 'TRAINING_PLAN.zh-CN.md'):
            src = BASE/group/name
            if src.exists(): shutil.copy2(src, dst/name)
    req = dict(created_at=now(), scenes=specs, code=codes(), baselines=baselines,
               iterations=5000, resolution=4, recursive_guide=False,
               overwrite_requested=True, tracking_job=str(TRACK_JOB),
               gpu_policy=POLICY, fit_uses_test_poses=True,
               training_uses_test_rgb=False, training_uses_test_masks=False)
    write(job/'request.json', req)
    for name in req['code']:
        dst=job/'code_snapshot'/name; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/name,dst)
    plan = '''# 新跟踪 mask 下重训 best_results 与 fit

本轮由用户要求覆盖两组发布结果，11场景×2组。每组保留原相机、MASK_DILATION、外观/几何监督和初始化模式，训练统一为5000步，初始化 max_points=null；独立 LaMa，RECURSIVE_GUIDE=false。详细配置差异位于任务日志 scenes/<group>__<scene>/config_diff.json；best_results 的 bag 原20000步，本轮按要求改5000步，不再称为历史最佳择优。

先逐帧检查新洞 mask；跟踪采用官方 SAM/DeAOT，必要处增加人工关键帧及遮挡空mask。使用删除后的虚拟渲染标注，未读取测试 RGB 或 GT mask。只有已审核发布的 tracker 才能进入训练。

每个场景从 Stage 1 开始重建工作区、LaMa RGB/depth/normal 输入与独立补全、匹配/深度初始化、局部高斯选择、5000步优化、模型发布、全部测试视角评估及 mesh。旧检查点和旧 LaMa 不参与本轮。分辨率1/4；不重新训练 seg 或改变 removal 几何。

best_results/<scene>/run 原为指向历史实验的链接，本轮改为该目录下独立的实际运行文件夹，并更新其模型/渲染/指标入口。历史实验原件不改。fit/scenes/<scene>/inpaint 原路径先归档到 fit/revisions/retrack_20261103，再在同一路径生成新结果。旧报告仅保存在本轮日志 previous_reports；新报告中未完成的场景明确写待完成。

GPU0/GPU1各一后台worker，共享22任务队列；每卡同一时间一个本轮GPU阶段。按阶段检查空闲显存，不终止其他任务。显存低于1536MiB或本任务占用达到22500MiB时，仅停止本任务子进程并标记风险。不能保证外部并发分配下绝不OOM，不降低分辨率或点数。

fit仍使用test*相机位姿插值，但不使用test RGB/GT mask训练。这与三个基线的训练相机协议不同。对比表使用三方法本机复现结果；所有425测试帧参与评估，显示7项指标、11场景宏平均及按帧数加权平均（FID为场景平均）。不按新指标重新挑选历史模型，不省略负结果。

任务：JOB_PATH。状态：status.json；每任务日志 scenes/<group>__<scene>/logs；后台日志 workers/gpu*/worker.log。
'''.replace('JOB_PATH',str(job))
    atomic(job/'TRAINING_PLAN.zh-CN.md', plan)
    for group in ('best_results','fit'):
        atomic(BASE/group/'TRAINING_PLAN.zh-CN.md', plan)
    report(job)


def report(job):
    with (job/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        req=read(job/'request.json'); states={}; metrics={}; counts={}
        baseline={k:read(v['path'])['per_scene'] for k,v in req['baselines'].items()}
        rows=[]
        for s in req['scenes']:
            key=s['key']; c=job/'scenes'/key; state=read(c/'status.json');states[key]=state
            counts[s['scene']]=s['frames']; p=evaluation(s)
            receipt=c/'receipts/evaluate_5000.json'
            if receipt.exists() and p.exists():
                value=read(p)
                verify_record(read(receipt)['artifacts'][0])
                if value.get('complete'):
                    metrics[key]=value['metrics']
        for group in ('best_results','fit'):
            method=f'PaintMesh {group} 5000 (new track)'; selected=[s for s in req['scenes'] if s['group']==group]
            lines=[f'# {group}：新跟踪结果重训5000步','',
                   '正在覆盖原发布结果；待完成项不沿用旧指标。fit使用test相机位姿，三个基线未统一为该轨迹；不把诊断对比当作同协议排名。','',
                   '[训练方案](TRAINING_PLAN.zh-CN.md) · [后台状态]('+str(job/'status.json')+')','',
                   '| 场景 | 方法 | '+' | '.join(METRICS)+' |', '|---|---|'+'---:|'*len(METRICS)]
            group_rows=[]; allvals={method:{},**{k:{} for k in baseline}}
            for s in selected:
                scene=s['scene']
                for m in allvals:
                    v=metrics.get(s['key']) if m==method else baseline[m][scene]['metrics']
                    if v:
                        allvals[m][scene]=v; group_rows.append(dict(scene=scene,method=m,**{k:v[k] for k in METRICS}))
                    lines.append('| '+scene+' | '+m+' | '+' | '.join(f'{v[k]:.6f}' if v else '待完成' for k in METRICS)+' |')
            complete=len(allvals[method])
            if complete==len(selected):
                for weighted in (False, True):
                    label='425帧加权（FID场景均值）' if weighted else '11场景宏平均'
                    for m,values in allvals.items():
                        avg={k:sum(v[k]*(counts[n] if weighted and k!='FID' else 1) for n,v in values.items())/sum(counts[n] if weighted and k!='FID' else 1 for n in values) for k in METRICS}
                        group_rows.append(dict(scene=label,method=m,**avg));lines.append('| '+label+' | '+m+' | '+' | '.join(f'{avg[k]:.6f}' for k in METRICS)+' |')
            lines+=['',f'本轮评估 {complete}/11；全流程完成 {sum(states[s["key"]]["state"]=="complete" for s in selected)}/11。',
                    '', '当前目录入口：best_results/<scene>/run 或 fit/scenes/<scene>/inpaint；renders/gt/masks 为同名测试评估产物。','']
            text='\n'.join(lines);root=BASE/group
            atomic(root/'comparison.md',text);atomic(root/'README.md',text)
            write(root/'comparison.json',dict(updated_at=now(),training_job=str(job),rows=group_rows,evaluated=complete,complete=complete==11))
            buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['scene','method',*METRICS]);w.writeheader();w.writerows(group_rows);atomic(root/'comparison.csv',buf.getvalue())
            write(root/'status.json',dict(updated_at=now(),evaluated=complete,total=11,scenes={s['scene']:states[s['key']] for s in selected},job=str(job)))
            rows+=group_rows
        write(job/'status.json',dict(updated_at=now(),evaluated=len(metrics),complete=sum(v['state']=='complete' for v in states.values()),total=len(states),scenes=states))
        write(job/'comparison.json',dict(rows=rows,updated_at=now()))


def running_decision(snapshot,own_pids=(),running=False):
    if running and (snapshot['free_mib']<POLICY['reserve_mib'] or sum(p['memory_mib'] for p in snapshot['processes'] if p['pid'] in own_pids)>=POLICY['own_limit_mib']):
        return 'memory_risk'
    return None


def redirect_entries(s):
    run=Path(s['inpaint_root']);model=run/'inpainted_3dgs';ev=model/'evaluation/inpaint360_r4'
    roots=[BASE/'best_results'/s['scene']] if s['group']=='best_results' else [BASE/'fit_release'/s['scene']]
    entries={'run':run,'model':model,'point_cloud.ply':model/'point_cloud/iteration_5000/point_cloud.ply',
             'evaluation':ev,'results.json':ev/'results.json','renders':ev/'renders','gt':ev/'gt','masks':ev/'masks',
             'training_config.json':run/'edgs_joint/config.resolved.json','lama_manifest.json':run/'manifests/lama_completion_manifest.json',
             'initialization_manifest.json':run/'manifests/edgs_init_manifest.json','joint_manifest.json':run/'manifests/edgs_joint_manifest.json',
             'initialization':run/'edgs_init','lama_input':run/'lama/input','lama_output':run/'lama/output',
             'tracker':Path(s['removal_root'])/'tracker','mesh':run/'inpainted_mesh',
             'mesh.ply':model/'mesh/ours_5000/tsdf_fusion_post.ply', 'train_renders':model/'train/ours_5000/renders'}
    for root in roots:
        if not root.exists():continue
        for name,target in entries.items():
            if (root/name).absolute()==target.absolute():continue
            link(target,root/name)


class Worker(guard.Worker):
    def __init__(self,root,gpu):
        guard.report=report;guard.check_code=check_code;guard.GPU_POLICY=POLICY;guard.gpu_decision=running_decision
        super().__init__(root,gpu)

    def wait_gpu(self,stage):
        while True:
            snap=guard.gpu_snapshot(self.uuid)
            if snap['free_mib']>=REQUIRED.get(stage,10000):return
            self.save(state='waiting_for_memory',stage=stage,gpu_snapshot=snap,child_pid=None)
            report(self.root);time.sleep(10)

    def reset(self,s):
        marker=self.case/'reset.json'
        if marker.exists():return
        run=Path(s['inpaint_root'])
        if s['group']=='fit':
            backup=BASE/'fit/revisions/retrack_20261103'/s['scene']/'inpaint'
            backup.parent.mkdir(parents=True,exist_ok=True)
            if run.exists():
                if backup.exists():raise ValueError('Archive collision: '+str(backup))
                run.rename(backup)
            elif not backup.exists():raise ValueError('Missing original fit outputs')
        else:
            if not run.is_symlink():raise ValueError('Expected original best run link')
            run.unlink();backup=Path(s['old_output_target'])
        run.mkdir(parents=True,exist_ok=True)
        write(marker,dict(reset_at=now(),previous=str(backup),new_run=str(run),fresh_initialization=True))
        redirect_entries(s)
        # Keep old selections/figures explicitly historical, not mixed with new metrics.
        if s['group']=='best_results':
            public=BASE/'best_results'/s['scene'];old=self.case/'previous_public';old.mkdir(exist_ok=True)
            for name in ('selection.json','comparison.jpg','figure_selection.json','pipeline.jpg','training_view.jpg','training_fit.json'):
                p=public/name
                if p.exists() and not p.is_symlink():shutil.move(str(p),str(old/name))
            atomic(public/'README.md',f'# {s["scene"]}: 新跟踪重训中\n\n结果生成在 run/；状态和详细阶段日志：{self.case}\n')

    pipeline=fit.Worker.pipeline

    def process(self,s):
        self.case=self.root/'scenes'/s['key'];self.state=dict(pid=os.getpid(),gpu=self.gpu,gpu_uuid=self.uuid,scene=s['scene'],group=s['group'])
        self.save(state='preflight',stage='verify_reviewed_tracking');check_code(self.req)
        verify_record(s['config']);verify_record(s['camera'])
        pub=read(s['tracking_publication']);tr=Path(s['removal_root'])/'tracker'
        if not pub['complete'] or not read(tr/'visual_review.json')['accepted']:raise ValueError('Tracker not visually accepted')
        fit.Worker.commit_masks(self,s)
        if not (self.case/'tracking_inputs.json').exists():
            write(self.case/'tracking_inputs.json',dict(session=record(tr/'tracking_session.json'),review=record(tr/'visual_review.json')))
        for r in read(self.case/'tracking_inputs.json').values():verify_record(r)
        self.reset(s);inp=Path(s['inpaint_root'])
        self.pipeline(s,1,2,[inp/'manifests/lama_input_manifest.json'],False)
        self.pipeline(s,3,3,[inp/'manifests/lama_completion_manifest.json'],True)
        self.pipeline(s,4,4,[inp/'manifests/edgs_init_manifest.json'],True)
        self.pipeline(s,5,5,[inp/'manifests/edgs_joint_manifest.json'],True)
        self.pipeline(s,6,6,[inp/'inpainted_3dgs/model_manifest.json'],False)
        self.execute('evaluate_5000',[PYTHON,REPO/'scripts/paintmesh/eval_inpaint360.py','run','--model-path',inp/'inpainted_3dgs',
              '--source-path',s['source_path'],'--resolution',4,'--iteration',5000,'--batch-size',4],REPO,[evaluation(s)],gpu=True)
        result=read(evaluation(s));validate_evaluation(result,read(s['previous_evaluation']['path']))
        self.save(state='evaluated',stage='mesh');report(self.root)
        self.pipeline(s,7,8,[inp/'inpaint_manifest.json'],True)
        note=tr/'DOWNSTREAM_REGEN_REQUIRED.json';old=read(note) if note.exists() else {}
        old['regenerated_runs']=dict(old.get('regenerated_runs',{}),**{str(inp):dict(completed_at=now(),job=str(self.root))});write(note,old)
        summary=dict(scene=s['scene'],method='new reviewed tracking; 5000 iterations; not historical best selection',
                     run_root=str(inp),metrics=result['metrics'],frame_count=result['frame_count'],job=str(self.case))
        public=BASE/'best_results'/s['scene'] if s['group']=='best_results' else BASE/'fit_release'/s['scene']
        if public.exists():write(public/'selection.json',summary)
        self.save(state='complete',stage='complete',child_pid=None,reason=None,finished_at=now());report(self.root)

    def run(self):
        with (self.wdir/'run.lock').open('a') as worker_lock, (BASE/f'.reviewed_retrain_gpu{self.gpu}.lock').open('a') as gpu_lock:
            fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN);signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
            for s in self.req['scenes']:
                case=self.root/'scenes'/s['key']
                with (case/'run.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if read(case/'status.json')['state'] in ('complete','failed','blocked_memory_risk'):continue
                    try:self.process(s)
                    except guard.MemoryRisk as e:
                        self.save(state='blocked_memory_risk',reason=str(e),child_pid=None);traceback.print_exc();report(self.root)
                    except Exception as e:
                        self.save(state='failed',reason=str(e),child_pid=None);traceback.print_exc();report(self.root)
            self.case=None;self.save(state='queue_finished',scene=None,stage=None,child_pid=None);report(self.root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,default=DEFAULT_JOB)
    p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--prepare-only',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();job=a.job.resolve()
    if a.worker is not None:Worker(job,a.worker).run();return
    if a.report:report(job);return
    job.mkdir(parents=True,exist_ok=True)
    with (job/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare(job)
        if a.prepare_only:return
        for s in read(job/'request.json')['scenes']:
            if not Path(s['tracking_publication']).exists():raise ValueError('Unpublished reviewed tracker: '+s['key'])
        workers={}
        for gpu in (0,1):
            wd=job/'workers'/f'gpu{gpu}';wd.mkdir(parents=True,exist_ok=True)
            state=read(wd/'status.json') if (wd/'status.json').exists() else {};pid=state.get('pid')
            proc=Path(f'/proc/{pid}/cmdline')
            if pid and proc.exists() and str(SCRIPT).encode() in proc.read_bytes():
                workers[str(gpu)]=dict(pid=pid,state='already_running');continue
            log=wd/'worker.log'
            with log.open('a') as out:
                child=subprocess.Popen([PYTHON,str(SCRIPT),'--job',str(job),'--worker',str(gpu)],cwd=REPO,
                    stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            workers[str(gpu)]=dict(pid=child.pid,log=str(log))
        write(job/'workers.json',workers);print(workers,flush=True)


if __name__=='__main__':main()
