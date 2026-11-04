#!/usr/bin/env python3
"""All-scene hard-reference experiment with opportunistic, yielding GPU workers."""
from __future__ import annotations

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
import sys
import time
import traceback

from run_geometry_ablation import REPO, METRICS, atomic, now, read, sha, write
from run_muti_plan import PYTHON, Runner, changes, validate_evaluation
from run_shared_reference_pilot import codes as parent_codes, GUIDE, LAMA
from edgs_inpaint_io import implementation, record, verify_record

BASE = REPO / 'output/paintmesh/Inpaint360GS'
SCRIPT = Path(__file__).resolve()
GPU_POLICY = dict(idle_seconds=90, start_free_mib=22000, reserve_mib=4096,
                  own_limit_mib=18000, sample_seconds=2, idle_util_max=5,
                  foreign_process_policy='yield_our_child_only', disk_free_gib=40)


def codes():
    c = parent_codes()
    for p in (SCRIPT, SCRIPT.with_name('run_geometry_ablation.py'),
              REPO/'submodules/Inpaint360GS/utils/image_utils.py',
              REPO/'submodules/Inpaint360GS/utils/loss_utils.py'):
        c[str(p.relative_to(REPO))] = sha(p)
    return c


def check_code(req):
    if req['code'] != codes():
        raise RuntimeError('Frozen experiment implementation changed; refusing mixed results')


def evaluation_path(case):
    return case/'inpainted_3dgs_5000/evaluation/inpaint360_r4/results.json'


def gpu_snapshot(uuid):
    def query(args):
        return subprocess.check_output(['nvidia-smi', *args, '--format=csv,noheader,nounits'],
                                       text=True, timeout=10).strip()
    free, used, util = map(int, query(['-i', uuid, '--query-gpu=memory.free,memory.used,utilization.gpu']).split(','))
    processes = []
    for line in query(['--query-compute-apps=gpu_uuid,pid,used_memory']).splitlines():
        fields = [v.strip() for v in line.split(',')]
        if fields[0] == uuid:
            processes.append(dict(pid=int(fields[1]), memory_mib=int(fields[2])))
    return dict(free_mib=free, used_mib=used, utilization=util, processes=processes)


def gpu_decision(snapshot, own_pids=(), running=False):
    foreign = [p for p in snapshot['processes'] if p['pid'] not in own_pids]
    if foreign:
        return 'foreign_gpu_process'
    if running:
        if snapshot['free_mib'] < GPU_POLICY['reserve_mib']:
            return 'memory_risk'
        if sum(p['memory_mib'] for p in snapshot['processes']) >= GPU_POLICY['own_limit_mib']:
            return 'memory_risk'
    elif snapshot['free_mib'] < GPU_POLICY['start_free_mib']:
        return 'insufficient_free_memory'
    elif snapshot['utilization'] > GPU_POLICY['idle_util_max']:
        return 'gpu_busy'
    return None


def process_group_pids(pgid):
    # Only our separately created process group is eligible for termination.
    out = subprocess.check_output(['ps', '-eo', 'pid=,pgid='], text=True)
    return {int(pid) for pid, group in (line.split() for line in out.splitlines()) if int(group) == pgid}


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'request.json').exists():
        check_code(read(root/'request.json'))
        return
    old = read(BASE/'muti_plan/request.json')
    specs = []
    for row in old['scenes']:
        scene = row['scene']; case = root/'scenes'/scene
        case.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(row['variants'][0]['config']); cfg['iterations'] = 5000
        write(case/'source.json', row)
        write(case/'config.json', cfg)
        write(case/'config_diff_from_best.json', changes(row['historical_config'], cfg))
        best = BASE/'best_results'/scene
        for name in ('results.json', 'selection.json', 'training_config.json'):
            dest = root/'best_snapshot'/scene/name
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(best/name, dest)
        reference = read(root/'best_snapshot'/scene/'results.json')
        validate_evaluation(reference, row['historical_evaluation'])
        spec = dict(scene=scene, frames=reference['frame_count'], source=record(case/'source.json'),
                    config=record(case/'config.json'), best=record(root/'best_snapshot'/scene/'results.json'),
                    initial_points=read(row['inputs']['initialization']['path'])['point_count'])
        if scene in ('truck', 'cube'):
            completed = BASE/'shared_reference_pilot_20261101/scenes'/scene
            joint = read(completed/'manifests/edgs_joint_manifest.json')
            if joint['request']['implementation'] != implementation():
                raise ValueError('Completed pilot training code changed: '+scene)
            old_cfg = copy.deepcopy(joint['request']['config']); old_cfg['iterations'] = 5000
            if old_cfg != cfg or read(completed/'source.json') != row:
                raise ValueError('Completed pilot configuration/input mismatch: '+scene)
            ev = read(completed/'guide/evidence.json')
            from shared_reference_completion import POLICY
            if ev['policy'] != POLICY or ev['producer'] != record(GUIDE):
                raise ValueError('Completed pilot guidance differs: '+scene)
            result = read(evaluation_path(completed)); validate_evaluation(result, reference)
            # Evaluation receipts predate the size_bytes field used by training
            # receipts; validate their actual model content using the SHA.
            if sha(result['inputs']['model']['path']) != result['inputs']['model']['sha256']:
                raise ValueError('Completed pilot evaluated PLY changed: '+scene)
            for name in ('inpainted_3dgs_5000', 'guide', 'lama', 'manifests', 'edgs_joint', 'training_source.json'):
                dest = case/name
                if not dest.exists():
                    dest.symlink_to(os.path.relpath(completed/name, case), target_is_directory=(completed/name).is_dir())
            reuse = dict(source=str(completed), original_training_budget=10000, evaluated_iteration=5000,
                         result=record(evaluation_path(completed)), verified_at=now(),
                         policy='Reuse genuine 5000 snapshot; not a newly trained run; config.json describes its training prefix')
            write(case/'reuse.json', reuse); spec['reuse'] = record(case/'reuse.json')
            write(case/'status.json', dict(state='complete_reused', finished_at=now(), stage='complete'))
        elif not (case/'status.json').exists():
            write(case/'status.json', dict(state='pending', updated_at=now()))
        specs.append(spec)
    code = codes()
    for name in code:
        dest = root/'code_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, dest)
    req = dict(created_at=now(), root=str(root), scenes=specs, code=code,
               gpu_policy=GPU_POLICY, iterations=5000, resolution=4, seed=0,
               source_request=record(BASE/'muti_plan/request.json'),
               comparison='Frozen best_results snapshot; seven metrics, same complete test set; no cherry-picked replacement',
               experiment='Fixed-reference hard RGB propagation + residual LaMa; existing per-scene inputs and supervision')
    write(root/'request.json', req)
    plan(root, specs)
    report(root)


def plan(root, specs):
    lines = ['# 全场景固定参考硬引导实验（5000 步）', '',
        '本轮按用户指定的 truck 硬引导方案扩展至全部 11 场景。输出、配置、日志和对比均在本目录，旧实验及 best_results 不被覆盖。', '',
        '## 实验边界', '',
        '- 固定 seed=0，分辨率 1/4，5000 步为唯一评估节点，所有 425 个真实 test 视角均评估。',
        '- 复用各场景已核验的分割、删除、跟踪 mask、虚拟相机、depth/normal、初始高斯所有字段及旧点更新名单；只替换训练 RGB 目标并固定训练预算。',
        '- 沿用 muti_plan 的 00_control 原本外观＋几何监督，不切换官方外观、不新增损失；只剪枝、不克隆分裂。cube 保持其旧点冻结配置，其余沿用局部更新名单。',
        '- 所有场景保持 RECURSIVE_GUIDE=false；不会重新初始化或新增点数上限。配置内历史初始化上限只记录历史，复用的 PLY 不截断。',
        '- truck、cube 复用已完成硬引导训练中的真实 5000 步快照，原运行总预算为 10000；reuse.json 明确来源，不冒充本轮新训练。其余 9 场景重训 5000 步。',
        '- 这轮推广的是相同硬引导机制，并非把不同场景的既有初始化和相机强制改成 truck。每场景完整参数见 scenes/<scene>/config.json，历史配置差异见 config_diff_from_best.json。', '',
        '## 硬引导实现', '',
        '在训练虚拟相机中均匀选最多 18 个候选参考，用几何可靠洞覆盖率选择一个固定参考（不使用 test GT）。固定该视角已有 LaMa 补全；由目标深度反投影到参考视角，要求相对深度误差≤2%、往返投影误差≤1.5px、视线夹角≤60°，排除>5%的深度突变，可信区域腐蚀1px。', '',
        '可信参考 RGB 成为其他视角的已知条件，LaMa 仅修复剩余洞；无可信重叠的视角沿用原独立补全。验证 mask 外像素逐字节不变、可信参考像素不变；保留深度/法线和全部初始化参数。guide/validation.json 和 target_comparison.png 记录覆盖及跨视角一致性。', '',
        '## 文献依据与限制', '',
        '- [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 联合优化高斯属性与密度；本轮固定原有剪枝策略，避免同时改变拓扑造成混杂。仅剪枝不是原版完整自适应密度控制。',
        '- [PGSR](https://arxiv.org/html/2406.06521v2) 强调几何与多视角一致性，并通过重投影过滤不可靠对应。本实验借用这一原则筛选 RGB 传播；具体阈值来自既有 pilot，不宣称为论文默认参数。',
        '- [Inpaint360GS](https://arxiv.org/html/2511.06457v1) 使用深度引导与多视角一致补全。这里采用固定参考的像素硬引导，与官方递归特征引导不同，属于 PaintMesh 新实验，不能承诺完全复现其论文数值。',
        '- 伪深度一致不等于真实几何正确；参考补错会传播错误。此前 cube 的硬引导5000结果低于独立对照，本轮仍按用户要求完整报告全部场景，不隐去负结果。', '',
        '## GPU 调度与显存保护', '',
        '- GPU 0、1 各一后台 worker，共享剩余场景队列；CPU 预处理先进行。GPU 阶段仅在无其他计算进程、利用率≤5%、空闲显存≥22000MiB，持续90秒后启动。',
        '- 不与正在使用显卡的其他任务挤显存，不暂停或终止其他任务；每2秒检查一次。发现外部进程时仅结束本 worker 创建的子进程组，回到等待状态。',
        '- 自身显存达到18000MiB或整卡空闲低于4096MiB时，结束自己的 GPU 阶段并将该场景记为 blocked_memory_risk，不自动重试冒险。NVML查询失败则本任务让出GPU。',
        '- 训练每100步原子保存检查点，因让卡停止时最多重算最近不足100步；LaMa/评估重试同阶段。代码和输入哈希不一致时拒绝续跑。',
        '- 这是保守的启动门槛与运行监测，不是硬件显存预留：外部进程可能在两次检查之间突然分配显存，不能数学保证绝不OOM。若CUDA报告OOM则停止该场景，不降低质量参数或自动重复。',
        '- 独立工作流可能不遵守本队列锁；因此等待空闲而非仅看当前显存余量。状态和风险原因写入 status.json、workers/gpu*/status.json、scenes/*/status.json。', '',
        '## 对比和文件', '',
        'comparison.md/csv/json 自动更新各场景 best_results、hard5000 及差值。best_snapshot 冻结本轮开始时的旧指标及配置；best_results 属于测试集事后择优且预算不同（bag为20000步），不能将差值完全归因于单一模块。保留全部负差值；不拿不同模型的单项最佳指标拼表。', '',
        '全部11场景完成后给出场景均值和425帧加权均值；FID始终为场景FID算术均值，不冒称所有图像合并计算的FID。尚未完成时不以已完成子集冒充全场景均值。', '',
        '`scenes/<scene>/` 包含 config/source、guide/、lama/、manifests/、edgs_joint/、inpainted_3dgs_5000/、logs/ 和 status.json。复用大文件为符号链接，来源保留；logs/gpu*.log 为总日志。', '',
        '| 场景 | 相机/帧数 | dilation | 初始化 | 局部旧点更新 | 初始高斯数 | test帧 | 本轮 |',
        '|---|---|---:|---|---|---:|---:|---|']
    for s in specs:
        row = read(s['source']['path']); cfg = read(s['config']['path']); c = row['configuration']
        lines.append(f"| {s['scene']} | {c['trajectory']} / {c['virtual_frames']} | {c['mask_dilation']} | depth={cfg['init']['use_depth']}, {cfg['surface_init']['mode']} | {cfg['local_background']['enabled']} | {s['initial_points']} | {s['frames']} | {'复用已完成5000快照' if 'reuse' in s else '新训练5000'} |")
    atomic(root/'TRAINING_PLAN.zh-CN.md', '\n'.join(lines)+'\n')
    atomic(root/'README.md', '# PaintMesh 全场景硬引导5000\n\n[训练方案](TRAINING_PLAN.zh-CN.md) · [与 best_results 对比](comparison.md) · [实时状态](status.json) · [冻结请求](request.json)\n\n后台自动等待 GPU 0/1 空闲；不会停止其他任务。九场景新训练，truck/cube复用已核验5000快照。每个场景的配置、阶段日志和结果在 scenes/<scene>/。\n')


def report(root):
    with (root/'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        req = read(root/'request.json'); rows = []; states = {}; done = 0
        lines = ['# Hard5000 与历史 best_results 对比', '', '固定全部 test 视角、同评估协议。差值为 hard − best；PSNR/SSIM 越高越好，LPIPS/FID 越低越好。历史 best 为事后择优且预算不同，详见训练方案。', '',
                 '| 场景 | 方法 | '+' | '.join(METRICS)+' |', '|---|---|'+'---:|'*len(METRICS)]
        pairs = []
        for s in req['scenes']:
            case = root/'scenes'/s['scene']; state = read(case/'status.json')
            states[s['scene']] = state
            best = read(s['best']['path']); hard = None
            if state['state'] in ('complete', 'complete_reused'):
                hard = read(evaluation_path(case)); validate_evaluation(hard, best); done += 1
                pairs.append((s['frames'], best['metrics'], hard['metrics']))
            for method, metrics in [('best_results', best['metrics']), ('hard 5000', hard['metrics'] if hard else None),
                                    ('差值', {k:hard['metrics'][k]-best['metrics'][k] for k in METRICS} if hard else None)]:
                if metrics is None:
                    if method == 'hard 5000':
                        lines.append(f"| {s['scene']} | {state['state']} | "+' | '.join('—' for _ in METRICS)+' |')
                    continue
                rows.append(dict(scene=s['scene'], method=method, **metrics))
                fmt = '+.6f' if method == '差值' else '.6f'
                lines.append(f"| {s['scene']} | {method} | "+' | '.join(format(metrics[k], fmt) for k in METRICS)+' |')
        if done == len(req['scenes']):
            for weighted in (False, True):
                label = '425帧加权（FID场景均值）' if weighted else '11场景均值'
                averages = []
                for idx in (1, 2):
                    averages.append({k:sum(p[idx][k]*(p[0] if weighted and k!='FID' else 1) for p in pairs)/sum(p[0] if weighted and k!='FID' else 1 for p in pairs) for k in METRICS})
                for method, m in [('best_results', averages[0]), ('hard 5000', averages[1]), ('差值', {k:averages[1][k]-averages[0][k] for k in METRICS})]:
                    rows.append(dict(scene=label, method=method, **m))
                    lines.append(f'| {label} | {method} | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
        lines += ['', f'已完成 {done}/{len(req["scenes"])}；包括 truck/cube 复用结果。未完成时不计算全场景均值。', '', '[实时状态](status.json) · [训练方案](TRAINING_PLAN.zh-CN.md)', '']
        atomic(root/'comparison.md', '\n'.join(lines))
        write(root/'comparison.json', dict(updated_at=now(), complete=done==len(req['scenes']), rows=rows))
        stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=('scene','method',*METRICS));writer.writeheader();writer.writerows(rows)
        atomic(root/'comparison.csv', stream.getvalue())
        workers = {p.parent.name:read(p) for p in (root/'workers').glob('*/status.json')}
        write(root/'status.json', dict(updated_at=now(), complete=done, total=len(req['scenes']), scenes=states, workers=workers))


class MemoryRisk(RuntimeError):
    pass


class YieldGPU(RuntimeError):
    pass


class Worker:
    def __init__(self, root, gpu):
        self.root = root; self.gpu = gpu; self.req = read(root/'request.json'); self.child = None
        self.uuid = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        self.wdir = root/'workers'/f'gpu{gpu}'; self.wdir.mkdir(parents=True,exist_ok=True)
        self.case = None; self.state = dict(pid=os.getpid(), gpu=gpu, gpu_uuid=self.uuid)
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        GPU=str(gpu), PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        self.env.pop('PYTHONPATH',None)

    def save(self, **kw):
        self.state.update(kw, updated_at=now())
        write(self.wdir/'status.json', self.state)
        if self.case:write(self.case/'status.json', self.state)

    def stop_child(self):
        if self.child and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
            try:self.child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid,signal.SIGKILL);self.child.wait()

    def stop(self, *_):
        self.stop_child();self.save(state='stopped', child_pid=None);raise SystemExit(0)

    def wait_gpu(self, stage):
        since = None
        while True:
            try:
                snap = gpu_snapshot(self.uuid); reason = gpu_decision(snap)
            except Exception as e:
                snap = {}; reason = 'gpu_query_failed: '+str(e)
            if reason:since = None
            elif since is None:since = time.monotonic()
            if since is not None and time.monotonic()-since >= GPU_POLICY['idle_seconds']:
                return
            self.save(state='waiting_for_safe_gpu', stage=stage, reason=reason or 'confirming_90s_idle', gpu_snapshot=snap, child_pid=None)
            report(self.root);time.sleep(10)

    def execute(self, stage, cmd, cwd, expected, gpu=False):
        check_code(self.req)
        cmd = list(map(str, cmd)); receipt = self.case/'receipts'/(stage+'.json')
        if receipt.exists():
            r = read(receipt)
            if r['command'] != cmd:raise ValueError('Stage command changed')
            for v in r['artifacts']:verify_record(v)
            return
        while True:
            if shutil.disk_usage(self.root).free < GPU_POLICY['disk_free_gib']*1024**3:
                raise RuntimeError('Insufficient free disk; need 40 GiB')
            if gpu:self.wait_gpu(stage)
            env = dict(self.env)
            if not gpu:env['CUDA_VISIBLE_DEVICES']=''
            if cmd[0] == LAMA:env['PYTHONPATH']=str(REPO/'submodules/Inpaint360GS/LaMa')
            logfile = self.case/'logs'/(stage+'.log');logfile.parent.mkdir(parents=True,exist_ok=True)
            self.save(state='running', stage=stage, reason=None, command=cmd, log=str(logfile))
            print(now(), self.case.name, stage, flush=True)
            try:
                with logfile.open('a') as log:
                    log.write('\n'+now()+' START '+stage+'\n');log.flush()
                    self.child = subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    self.save(child_pid=self.child.pid)
                    last = 0
                    while self.child.poll() is None:
                        if gpu:
                            try:
                                snap = gpu_snapshot(self.uuid)
                                reason = gpu_decision(snap, process_group_pids(self.child.pid), running=True)
                            except Exception as e:raise YieldGPU('GPU query failed: '+str(e))
                            if reason == 'memory_risk':raise MemoryRisk(str(snap))
                            if reason:raise YieldGPU(reason+': '+str(snap))
                        if time.monotonic()-last > 30:
                            self.save(gpu_snapshot=snap if gpu else None);report(self.root);last=time.monotonic()
                        time.sleep(GPU_POLICY['sample_seconds'])
                    code = self.child.wait()
                self.save(child_pid=None)
                if code:
                    with logfile.open('rb') as f:
                        f.seek(max(0,logfile.stat().st_size-16000));tail=f.read().decode(errors='replace')
                    if 'out of memory' in tail.lower():raise MemoryRisk('GPU allocation failed; no automatic retry: '+str(logfile))
                    raise RuntimeError(f'{stage} failed ({code}): {logfile}')
                write(receipt,dict(command=cmd,finished_at=now(),artifacts=[record(p) for p in expected]))
                return
            except YieldGPU as e:
                self.stop_child();self.save(state='yielded_to_other_task',reason=str(e),child_pid=None)
                print(now(),'Yield our GPU child:',e,flush=True);report(self.root);time.sleep(30)
            except BaseException:
                self.stop_child();self.save(child_pid=None);raise

    def process(self, spec):
        self.case = self.root/'scenes'/spec['scene']
        self.state = dict(pid=os.getpid(),gpu=self.gpu,gpu_uuid=self.uuid,scene=spec['scene'])
        self.save(state='preflight',stage='verify_inputs')
        check_code(self.req)
        for key in ('source','config','best'):verify_record(spec[key])
        row = read(spec['source']['path']);cfg=read(spec['config']['path'])
        from edgs_inpaint_io import validate_init, check_initialization_config, verify_targets, validate_selection, appearance_weights
        for value in row['inputs'].values():verify_record(value)
        initial=validate_init(row['inputs']['initialization']['path']);check_initialization_config(initial,cfg)
        verify_targets(row['inputs']['lama']['path'],row['inputs']['camera']['path'],**cfg['supervision'])
        if cfg['local_background']['enabled']:
            validate_selection(row['inputs']['local_selection']['path'],initial=initial,config=cfg['local_background'])
        for value in appearance_weights(cfg).values():verify_record(value)
        write(self.case/'preflight.json',dict(valid=True,checked_at=now(),initial_points=initial['point_count']))
        self.execute('prepare_guide',[PYTHON,GUIDE,'prepare','--root',self.case],REPO,[self.case/'guide/evidence.json'])
        evidence=read(self.case/'guide/evidence.json')
        expected=[self.case/'lama/output/color'/(s+'.png') for s,v in evidence['frames'].items() if v['mode']=='warp_and_lama_residual']
        if expected:
            lama=read(row['inputs']['lama']['path']);model=Path(lama['model']['config']['path']).parent
            self.execute('lama_shared_reference',[LAMA,'bin/predict_color.py','--input-dir',self.case/'guide/input',
                         '--output-dir',self.case/'lama/output/color','--model-path',model],REPO/'submodules/Inpaint360GS/LaMa',expected,gpu=True)
        self.execute('validate_and_bind',[PYTHON,GUIDE,'finalize','--root',self.case],REPO,[self.case/'training_source.json',self.case/'guide/validation.json'])
        row=read(self.case/'training_source.json')
        self.execute('train_5000',Runner.training_command(self,row,self.case,saves=False),REPO/'submodules/EDGS',[self.case/'manifests/edgs_joint_manifest.json'],gpu=True)
        self.execute('publish_5000',Runner.publish_command(self,row,self.case,iteration=5000),REPO/'submodules/Inpaint360GS',[self.case/'inpainted_3dgs_5000/model_manifest.json'])
        self.execute('evaluate_5000',[PYTHON,REPO/'scripts/paintmesh/eval_inpaint360.py','run','--model-path',self.case/'inpainted_3dgs_5000',
                     '--source-path',row['source'],'--resolution',4,'--iteration',5000,'--batch-size',4],REPO,[evaluation_path(self.case)],gpu=True)
        validate_evaluation(read(evaluation_path(self.case)),read(spec['best']['path']))
        self.save(state='complete',stage='complete',finished_at=now(),reason=None,child_pid=None);report(self.root)

    def run(self):
        # Global advisory lock prevents two copies of this worker owning a card.
        with (BASE/f'.hard_gpu{self.gpu}.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN);signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
            for spec in self.req['scenes']:
                case=self.root/'scenes'/spec['scene']
                with (case/'run.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    state=read(case/'status.json')['state']
                    if state in ('complete','complete_reused','blocked_memory_risk','failed'):continue
                    try:self.process(spec)
                    except MemoryRisk as e:
                        self.save(state='blocked_memory_risk',reason=str(e),child_pid=None);traceback.print_exc();report(self.root)
                    except Exception as e:
                        self.save(state='failed',reason=str(e),child_pid=None);traceback.print_exc();report(self.root)
            self.case=None;self.save(state='queue_finished',scene=None,stage=None,child_pid=None);report(self.root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=BASE/'hard')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--report',action='store_true')
    p.add_argument('--worker',type=int,choices=(0,1))
    a=p.parse_args();root=a.root.resolve()
    if a.worker is not None:Worker(root,a.worker).run();return
    if a.report:report(root);return
    root.mkdir(parents=True,exist_ok=True)
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(root)
        if a.prepare_only:return
        workers={}
        for gpu in (0,1):
            statepath=root/'workers'/f'gpu{gpu}'/'status.json'
            state=read(statepath) if statepath.exists() else {}
            pid=state.get('pid');proc=Path(f'/proc/{pid}/cmdline')
            if pid and proc.exists() and str(root).encode() in proc.read_bytes() and b'--worker' in proc.read_bytes():
                workers[str(gpu)]=dict(pid=pid,state='already_running');continue
            logfile=root/'logs'/f'gpu{gpu}.log';logfile.parent.mkdir(exist_ok=True)
            with logfile.open('a') as log:
                child=subprocess.Popen([PYTHON,str(SCRIPT),'--root',str(root),'--worker',str(gpu)],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                       env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            workers[str(gpu)]=dict(pid=child.pid,log=str(logfile))
        write(root/'workers.json',workers);print(workers,flush=True)


if __name__=='__main__':main()
