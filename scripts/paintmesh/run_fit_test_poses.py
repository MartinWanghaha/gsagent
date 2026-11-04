#!/usr/bin/env python3
"""Test-pose-conditioned PaintMesh: independent LaMa, 5000 iterations, idle GPUs."""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import io
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from run_geometry_ablation import REPO, METRICS, read, write, atomic, now, sha
from edgs_inpaint_io import implementation, record, verify_record
from run_muti_plan import PYTHON, changes, validate_evaluation
import run_hard_reference_all as guard
from run_inpaint360_official_camera import Runner as TrackingContract
from fit_test_pose_cameras import generate

BASE=REPO/'output/paintmesh/Inpaint360GS'
SCRIPT=Path(__file__).resolve()
METHODS={'Inpaint360GS':'Inpaint360GS','Gaussian Grouping':'gaussian-grouping','AuraFusion360':'AuraFusion360_official'}


def codes():
    code=implementation()
    files=[SCRIPT,SCRIPT.with_name('fit_test_pose_cameras.py'),SCRIPT.with_name('run_hard_reference_all.py'),
           SCRIPT.with_name('run_inpaint360_official_camera.py'),SCRIPT.with_name('run_geometry_ablation.py'),
           SCRIPT.with_name('run_muti_plan.py'),SCRIPT.with_name('run_inpaint.sh'),SCRIPT.with_name('render_virtual_views.py'),
           SCRIPT.with_name('render_virtual_worker.py'),SCRIPT.with_name('track_holes_headless.py'),
           SCRIPT.with_name('tracker_web.py'),SCRIPT.with_name('eval_inpaint360.py')]
    files+=list((REPO/'submodules/Inpaint360GS/tools').glob('*paintmesh*.py'))
    files+=[REPO/'submodules/Inpaint360GS/tools/prepare_inpaint_workspace.py',
            REPO/'submodules/Inpaint360GS/scene/colmap_loader.py']
    files+=list((REPO/'submodules/Inpaint360GS/LaMa/bin').glob('predict_*.py'))
    files+=[REPO/'submodules/Inpaint360GS/LaMa/saicinpainting/evaluation/refinement.py']
    return {**code,**{str(p.relative_to(REPO)):sha(p) for p in files}}


def check_code(req):
    if req['code']!=codes():raise ValueError('Fit experiment implementation changed; refusing mixed artifacts')


def eval_path(case):return case/'inpaint/inpainted_3dgs/evaluation/inpaint360_r4/results.json'


def link(source,dest):
    source=Path(source).resolve(strict=True)
    if dest.is_symlink():
        if dest.resolve()!=source:raise ValueError('Changed symlink: '+str(dest))
    elif dest.exists():raise ValueError('Unexpected existing output: '+str(dest))
    else:dest.symlink_to(os.path.relpath(source,dest.parent),target_is_directory=source.is_dir())


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'request.json').exists():check_code(read(root/'request.json'));return
    old=read(BASE/'muti_plan/request.json');specs=[]
    for row in old['scenes']:
        name=row['scene'];case=root/'scenes'/name;case.mkdir(parents=True,exist_ok=True)
        removal=case/'removal';removal.mkdir(exist_ok=True)
        source=Path(row['publish']['inputs']['removal_manifest']['path']).parent
        (removal/'work_model').mkdir(exist_ok=True)
        for p in (source/'work_model').iterdir():
            if p.name!='virtual':link(p,removal/'work_model'/p.name)
        for entry in ('removal_manifest.json','removed_3dgs','removed_mesh'):
            if (source/entry).exists():link(source/entry,removal/entry)
        # Independent config copy is safe to rewrite during preparation, before
        # any consumer exists. Native removal geometry remains immutable.
        shutil.copytree(source/'config',removal/'config',dirs_exist_ok=True)
        for p in (removal/'config').rglob('*.json'):
            cfg=read(p)
            if 'finetune_iteration' in cfg:cfg['finetune_iteration']=5000;write(p,cfg)
        cfg=copy.deepcopy(row['variants'][0]['config']);cfg['iterations']=5000
        cfg['init']['max_points']=None
        write(case/'config.json',cfg);write(case/'source.json',row)
        write(case/'config_diff_from_hard.json',changes(read(BASE/'hard/scenes'/name/'config.json'),cfg))
        write(case/'config_diff_from_historical.json',changes(row['historical_config'],cfg))
        info=generate(row['source'],row['inputs']['camera']['path'],removal/'tracker',row['configuration']['virtual_frames'])
        if {Path(a['name']).stem for a in info['anchors']}!=set(row['historical_evaluation']['inputs']['frames']):
            raise ValueError('test* anchors differ from registered evaluation frames: '+name)
        # Verify source records used by the removal stage; deeper SHA/PLY checks
        # also run in the producer/consumer contracts before GPU optimization.
        for key in ('removed_model_manifest','removal_manifest','edgs_config'):
            verify_record(row['publish']['inputs'][key])
        spec=dict(scene=name,frames=row['frame_count'],source=record(case/'source.json'),config=record(case/'config.json'),
                  camera=record(removal/'tracker/virtual_cameras.json'),camera_source=record(removal/'tracker/camera_source.json'),
                  removal_root=str(removal),inpaint_root=str(case/'inpaint'),target_ids=row['publish']['parameters']['target_ids'],
                  source_path=row['source'],mask_dilation=row['configuration']['mask_dilation'],
                  virtual_count=info['frame_count'],anchor_count=info['anchor_count'])
        specs.append(spec);write(case/'status.json',dict(state='pending',updated_at=now()))
    baseline={}
    for label,folder in METHODS.items():
        path=REPO.parent/folder/'output/evaluation_summary.json';value=read(path)
        if value['scene_count']!=11 or value['frame_count']!=425:raise ValueError('Incomplete baseline: '+label)
        for s in specs:
            row=value['per_scene'][s['scene']]
            if row['frame_count']!=s['frames'] or row['resolution_factor']!=4:raise ValueError('Baseline data mismatch')
            verify_record(row['source'])
            if not all(math.isfinite(row['metrics'][m]) for m in METRICS):raise ValueError('Nonfinite baseline')
        dest=root/'baseline_snapshot'/folder/'evaluation_summary.json';dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        baseline[label]=dict(original=record(path),snapshot=record(dest),source_policy=value['policy'])
    code=codes()
    for name in code:
        dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,dest)
    req=dict(created_at=now(),root=str(root),scenes=specs,code=code,baselines=baseline,iterations=5000,resolution=4,seed=0,
        completion='independent LaMa; recursive_guide=false; no hard reference propagation',
        gpu_policy=guard.GPU_POLICY,init_max_points=None,uses_test_poses=True,uses_test_rgb_for_training=False,uses_test_masks_for_training=False,
        protocol='test-pose-conditioned camera trajectory experiment; baseline rows do not share this training protocol')
    write(root/'request.json',req);plan(root,req);report(root)


def plan(root,req):
    lines=['# Fit：test* 相机位置插值实验','',
        '用户原定：独立 LaMa 补全、5000步训练，改变相机轨迹；2026-11-02追加要求：全部11场景取消 init.max_points=500000 限制，统一设置 null。新结果均写入本目录；不修改其他实验或方法。','',
        '## 相机方案','',
        '只读取各场景 sparse/0/images.bin 中文件名以 test 开头的 COLMAP 外参。由 C=−Rᵀt 恢复世界相机中心；相机朝向在 SO(3) 上用四元数 SLERP 插值，中心在线段上插值，绝不直接线性插值 world-to-camera 的 t。',
        '先用归一化中心距离＋旋转角做确定性的最近邻排序和开放路径2-opt，减少相邻图像跳变；不强行首尾闭合。每个 test* 相机中心/朝向都作为精确锚点保留，额外帧按平移和角度长度分配。相邻旋转≤15°、中心位移≤路径半径的20%；原帧数不足时增加最少过渡帧，最多180帧，仍不足则停止并报告。',
        '优先保留每场景原虚拟帧数（90或180），redbull因83个test锚点之间的跳变需增加至107帧。FoV和图像尺寸保持原样，不改变相机渲染内参。camera_source.json 保存原/新帧数、每个 test* 锚点到虚拟帧的映射与插值权重；trajectory.png 显示旧/新路径。','',
        '## 不变配置与必须重建的输入','',
        '继承已审核的 muti_plan/00_control 原本外观＋几何监督，5000步、seed=0、仅剪枝/不克隆分裂、各场景原初始化模式及局部旧点更新规则。除明确取消初始化点数上限外，与 hard/config.json 的训练参数相同；使用独立 LaMa，RECURSIVE_GUIDE=false，不启用硬引导。',
        '沿用既有正确对象ID和删除点云，重渲染原场景/删除后场景的新轨迹；洞 mask 由两者PGSR差异自动标注并用DeAOT跟踪，几何校正记录在 automatic_annotation.json。允许完全遮挡帧空mask，全轨迹不可见则失败。mask膨胀及形状保持各场景配置。',
        '相机变更后重新执行 LaMa RGB/depth/normal、匹配/深度初始化及局部可训练点选择。全部场景 init.max_points=null，同时取消对应的8倍候选池限制及表面采样的总点数截断；保留置信度、重投影、深度一致性、体素融合和采样步长。显存安全监控仍生效，不暗中降分辨率或减少点数。',
        '初始化配置改变后，旧初始化及其下游训练/评估归档到 revisions，再按新配置重新生成；已完成的相机、跟踪及LaMa保留。原本已为null的场景不重训。各次变更及归档来源见 request.json 的 revisions。','',
        '训练5000步后导出模型、评估全部test视角七项指标，再执行PGSR mesh及语义导出。评估完成数量与全流程完成数量分别记录；mesh失败不会抹掉已完成的指标。','',
        '## 评估协议与文献依据','',
        '**本实验使用 test 相机位姿设计训练虚拟视角，是测试位姿条件化实验，不属于完全未见相机的泛化评估。测试RGB与GT mask仅在训练结束后的评估阶段读取；不会作为LaMa输入、跟踪标注或RGB监督。**',
        '其他三方法列使用本机已登记的 AuraFusion360_official、gaussian-grouping、Inpaint360GS 完整复现结果（含已记录的修复），不是论文原表数值。由于它们没有统一重训为同一test位姿轨迹，比较用于诊断，不能将优势直接归因于算法本身。',
        '所有方法展示全部425测试帧；前六指标按帧数加权，同时提供场景宏平均；FID是各场景FID平均，不是混合425张图重新计算的pooled FID。未完成全场景前不以部分结果充当全场景平均。',
        '- [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 的高斯优化依赖训练视角覆盖；本轮控制监督和密度策略以诊断轨迹覆盖影响。',
        '- [PGSR](https://arxiv.org/html/2406.06521v2) 使用几何与多视角约束；保留已有几何监督，中心和朝向连续化服务于跟踪与跨视角匹配。',
        '- [Inpaint360GS](https://arxiv.org/html/2511.06457v1) 使用虚拟相机与深度引导修复；本轮test条件轨迹属于新增实验，不宣称为其官方相机配置。','',
        '## 后台执行和资源','',
        '两张卡各一worker共享场景队列，与hard共用GPU锁，优先让已运行hard结束。没有其他GPU计算进程、利用率≤5%、空闲≥22000MiB并持续90秒才运行GPU阶段。每2秒监测；遇到外部任务只终止自己的子进程并等待。自身显存≥18000MiB或可用<4096MiB时标记 blocked_memory_risk，不自动冒险重试。',
        '无法硬件预留显存，外部瞬时分配仍有竞态；OOM日志会单独标记并停止该场景。没有权限或意图停止其他任务。训练每100步保存，重启从最近检查点续训。重建/LaMa中断按阶段重复，已完成阶段校验收据后跳过。',
        'logs/gpu*.log：worker阶段日志；scenes/<scene>/logs/：每阶段输出；status.json：进度/阻塞原因；comparison.md/csv/json：四方法逐场景指标和均值；baseline_snapshot：三方法冻结结果来源。','',
        '| 场景 | test锚点 | 虚拟帧 | MASK_DILATION | 对象ID |','|---|---:|---:|---:|---|']
    for s in req['scenes']:lines.append(f"| {s['scene']} | {s['anchor_count']} | {s['virtual_count']} | {s['mask_dilation']} | {s['target_ids']} |")
    atomic(root/'TRAINING_PLAN.zh-CN.md','\n'.join(lines)+'\n')
    atomic(root/'README.md','# PaintMesh fit：测试位姿条件化轨迹\n\n独立LaMa，5000步，全部11场景 init.max_points=null（无初始化点数上限）；使用test相机位姿，不用test图像/GT mask训练。显存保护继续生效。\n\n[训练方案](TRAINING_PLAN.zh-CN.md) · [四方法对比](comparison.md) · [状态](status.json) · [请求与来源](request.json)\n\nscenes/<scene>/removal/tracker/trajectory.png 为相机图；inpaint/ 包含新补全、初始化、训练、评估和mesh；logs/ 为后台日志；revisions/ 保存配置变更前的记录和受影响结果。\n')


def report(root):
    with (root/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        req=read(root/'request.json');states={};rows=[];counts={};completed=0;finished=0
        baseline={m:read(r['snapshot']['path'])['per_scene'] for m,r in req['baselines'].items()}
        metrics={m:{} for m in (*METHODS,'PaintMesh fit 5000')}
        lines=['# Fit 与三个方法：全场景结果','',
            '**协议差异：PaintMesh fit 使用 test* 相机位姿规划虚拟训练相机，其他三方法未据此重训；test RGB/GT mask不用于fit训练。这里只比较本机复现结果，不代表同协议算法排名。**','',
            '| 场景 | 方法 | '+' | '.join(METRICS)+' |','|---|---|'+'---:|'*len(METRICS)]
        for s in req['scenes']:
            scene=s['scene'];case=root/'scenes'/scene;state=read(case/'status.json');states[scene]=state;counts[scene]=s['frames']
            if state['state']=='complete':finished+=1
            for m in METHODS:metrics[m][scene]={k:baseline[m][scene]['metrics'][k] for k in METRICS}
            p=eval_path(case)
            if p.exists() and read(p).get('complete'):
                value=read(p);validate_evaluation(value,read(s['source']['path'])['historical_evaluation'])
                metrics['PaintMesh fit 5000'][scene]=value['metrics'];completed+=1
            for m in metrics:
                v=metrics[m].get(scene)
                if v:rows.append(dict(scene=scene,method=m,**{k:v[k] for k in METRICS}))
                lines.append(f'| {scene} | {m} | '+' | '.join(f'{v[k]:.6f}' if v else '待完成' for k in METRICS)+' |')
        if completed==len(req['scenes']):
            for weighted in (False,True):
                label='425帧加权（FID场景均值）' if weighted else '11场景宏平均'
                for m,v in metrics.items():
                    avg={k:sum(v[s][k]*(counts[s] if weighted and k!='FID' else 1) for s in counts)/sum(counts[s] if weighted and k!='FID' else 1 for s in counts) for k in METRICS}
                    rows.append(dict(scene=label,method=m,**avg));lines.append(f'| {label} | {m} | '+' | '.join(f'{avg[k]:.6f}' for k in METRICS)+' |')
        lines+=['',f'评估完成 {completed}/11；含mesh的全流程完成 {finished}/11。未完成时不计算全场景均值。','',
                '[训练方案与来源](TRAINING_PLAN.zh-CN.md) · [状态](status.json)','']
        atomic(root/'comparison.md','\n'.join(lines));write(root/'comparison.json',dict(updated_at=now(),evaluated=completed,complete=completed==11,metrics=metrics,rows=rows))
        stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=('scene','method',*METRICS));w.writeheader();w.writerows(rows);atomic(root/'comparison.csv',stream.getvalue())
        write(root/'status.json',dict(updated_at=now(),evaluated=completed,complete=finished,total=11,scenes=states,
              workers={p.parent.name:read(p) for p in (root/'workers').glob('*/status.json')}))


class Worker(guard.Worker):
    """Reuse tested process-group/memory guard; no modifications to hard files.

    Guard callbacks are rebound only inside this separate fit process. The hard
    worker process and its frozen implementation remain untouched.
    """
    def __init__(self,root,gpu):
        guard.report=report;guard.check_code=check_code
        super().__init__(root,gpu)

    def commit_masks(self,s):
        root=Path(s['removal_root']);path=root/'tracker/tracking_session.json';session=read(path)
        if not session.get('complete'):
            TrackingContract.commit_masks(self,s);return
        # A committed session is an immutable input to LaMa and training.
        # Verify it on resume without changing completed_at or its file record.
        from virtual_render_io import verify_tracking_render
        verify_record(session['input_archive']);verify_record(session['input_cameras'])
        verify_tracking_render(session)
        records=session['outputs']['masks'];names=[r['file'] for r in records]
        if sorted(names)!=sorted(session['expected_masks']) or len(names)!=len(set(names)):
            raise ValueError('Committed tracking mask set changed')
        for r in records:
            verify_record(dict(r,path=str(root/'tracker/results/images/images_masks'/r['file'])))

    def pipeline(self,s,start,end,expected,gpu):
        previous=dict(self.env)
        for key in list(self.env):
            if key.startswith('EDGS_') or key in ('FINETUNE_ITERATION','START_STAGE','END_STAGE','PYTHONPATH'):
                self.env.pop(key,None)
        self.env.update(GPU=self.uuid if gpu else '',PAINTMESH_ENV='paintmesh',LAMA_ENV='lama',
            DATA_ROOT=str(REPO.parent/'data'),CONDA_BIN=shutil.which('conda') or '/home/anaconda3/condabin/conda',
            REMOVAL_ROOT=s['removal_root'],INPAINT_RUN_ROOT=s['inpaint_root'],INPAINT_RUN_NAME='fit_test_poses',
            END_STAGE=str(end),INPAINT_PIPELINE='edgs-pgsr',EDGS_INPAINT_CONFIG=s['config']['path'],
            RECURSIVE_GUIDE='false',MASK_DILATION=str(s['mask_dilation']),MASK_DILATION_SHAPE='ellipse',
            ALLOW_EMPTY_MASKS='true',LOCAL_GEOMETRY_REFINE='false',SUPPORT_DENSITY_MODE='legacy',
            RENDER_INPAINT_VIDEO='false',RENDER_INPAINT_TRAIN='false',RENDER_INPAINT_TEST='false',RENDER_EDGS_TEST='false',
            MAX_DEPTH='5.0',VOXEL_SIZE='0.002',NUM_CLUSTERS='1',USE_DEPTH_FILTER='false',WRITE_COLORED_MESH='true')
        try:
            self.execute(f'inpaint_{start}_{end}',['bash',REPO/'scripts/paintmesh/run_inpaint.sh','Inpaint360GS/inpaint360',s['scene'],4,','.join(map(str,s['target_ids'])),'none',start],REPO,expected,gpu=gpu)
        finally:self.env=previous

    def process(self,s):
        self.case=self.root/'scenes'/s['scene'];self.state=dict(pid=os.getpid(),gpu=self.gpu,gpu_uuid=self.uuid,scene=s['scene'])
        self.save(state='preflight',stage='verify_inputs');check_code(self.req)
        for k in ('source','config','camera','camera_source'):verify_record(s[k])
        row=read(s['source']['path']);rem=Path(s['removal_root']);inp=Path(s['inpaint_root'])
        for k in ('removed_model_manifest','removal_manifest','edgs_config'):verify_record(row['publish']['inputs'][k])
        self.execute('virtual_render',[PYTHON,REPO/'scripts/paintmesh/render_virtual_views.py','--backend','edgs-pgsr',
            '--model-path',rem/'work_model','--edgs-model-path',Path(row['inputs']['edgs_config']['path']).parent,
            '--source-path',s['source_path'],'--iteration',2000,'--resolution',4,'--images','images','--alpha-min','.01',
            '--camera-manifest',rem/'tracker/virtual_cameras.json','--tracker-archive',rem/'tracker/images.zip'],REPO,
            [rem/'work_model/virtual/virtual_render_manifest.json',rem/'tracker/images.zip'],gpu=True)
        TrackingContract.tracking_session(self,s)
        # Tracking publishes its mask folder atomically. If a process was
        # stopped after publication, finish review/commit rather than overwrite.
        masks=rem/'tracker/results/images/images_masks'
        if not masks.exists():
            self.execute('track_masks',[PYTHON,REPO/'scripts/paintmesh/track_holes_headless.py','--removal-root',rem,'--allow-empty-masks'],REPO,[rem/'tracker/automatic_annotation.json'],gpu=True)
        self.commit_masks(s)
        annotation=rem/'tracker/automatic_annotation.json'
        if annotation.exists():
            a=read(annotation)
            # Upstream generic tracker declares "no test data". Clarify that
            # this special experiment's render cameras are test-pose-derived.
            if not a.get('test_poses_used'):
                a.update(test_data_used=True,test_poses_used=True,test_rgb_used=False,test_gt_masks_used=False,
                         camera_provenance=s['camera_source']);write(annotation,a)
                receipt=self.case/'receipts/track_masks.json'
                if receipt.exists():
                    value=read(receipt);value['artifacts']=[record(annotation)];write(receipt,value)
        self.pipeline(s,1,2,[inp/'manifests/lama_input_manifest.json'],False)
        self.pipeline(s,3,3,[inp/'manifests/lama_completion_manifest.json'],True)
        self.pipeline(s,4,4,[inp/'manifests/edgs_init_manifest.json'],True)
        self.pipeline(s,5,5,[inp/'manifests/edgs_joint_manifest.json'],True)
        self.pipeline(s,6,6,[inp/'inpainted_3dgs/model_manifest.json'],False)
        self.execute('evaluate_5000',[PYTHON,REPO/'scripts/paintmesh/eval_inpaint360.py','run','--model-path',inp/'inpainted_3dgs',
            '--source-path',s['source_path'],'--resolution',4,'--iteration',5000,'--batch-size',4],REPO,[eval_path(self.case)],gpu=True)
        validate_evaluation(read(eval_path(self.case)),row['historical_evaluation']);report(self.root)
        self.pipeline(s,7,8,[inp/'inpaint_manifest.json'],True)
        self.save(state='complete',stage='complete',finished_at=now(),child_pid=None,reason=None);report(self.root)

    def run(self):
        self.save(state='waiting_for_gpu_worker_lock',reason='Respect an existing hard/fit worker on this card')
        with (self.wdir/'run.lock').open('a') as ownlock,(BASE/f'.hard_gpu{self.gpu}.lock').open('a') as gpu_lock:
            fcntl.flock(ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN);signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
            while True:
                try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:self.save(state='waiting_for_gpu_worker_lock');report(self.root);time.sleep(15)
            for s in self.req['scenes']:
                case=self.root/'scenes'/s['scene']
                with (case/'run.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if read(case/'status.json')['state'] in ('complete','blocked_memory_risk','failed'):continue
                    try:self.process(s)
                    except guard.MemoryRisk as e:
                        self.save(state='blocked_memory_risk',reason=str(e),child_pid=None);traceback.print_exc();report(self.root)
                    except Exception as e:
                        self.save(state='failed',reason=str(e),child_pid=None);traceback.print_exc();report(self.root)
            self.case=None;self.save(state='queue_finished',scene=None,stage=None,child_pid=None);report(self.root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=BASE/'fit')
    p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--prepare-only',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.worker is not None:Worker(root,a.worker).run();return
    if a.report:report(root);return
    root.mkdir(parents=True,exist_ok=True)
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare(root)
        if a.prepare_only:return
        workers={}
        for gpu in (0,1):
            p=root/'workers'/f'gpu{gpu}'/'status.json';old=read(p) if p.exists() else {};pid=old.get('pid')
            cmd=Path(f'/proc/{pid}/cmdline')
            if pid and cmd.exists() and str(SCRIPT).encode() in cmd.read_bytes():workers[str(gpu)]=dict(pid=pid,state='already_running');continue
            log=root/'logs'/f'gpu{gpu}.log';log.parent.mkdir(exist_ok=True)
            with log.open('a') as f:
                process=subprocess.Popen([PYTHON,str(SCRIPT),'--root',str(root),'--worker',str(gpu)],cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                    start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            workers[str(gpu)]=dict(pid=process.pid,log=str(log))
        write(root/'workers.json',workers);print(workers,flush=True)


if __name__=='__main__':main()
