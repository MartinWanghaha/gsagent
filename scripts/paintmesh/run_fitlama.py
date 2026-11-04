#!/usr/bin/env python3
"""Reviewed fit trajectory + official Inpaint360GS LaMa; durable two-GPU queue."""
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
from edgs_inpaint_io import record, verify_record
from run_geometry_ablation import REPO, METRICS, read, write, atomic, now, sha
from run_muti_plan import PYTHON, validate_evaluation

BASE = fit.BASE
SCRIPT = Path(__file__).resolve()
BINDER = SCRIPT.with_name('fitlama_targets.py')
OFFICIAL = REPO.parent/'Inpaint360GS'
OFFICIAL_LAMA = OFFICIAL/'LaMa'
OFFICIAL_PYTHON = Path('/home/member/.conda/envs/inpaint360gs-lama/bin/python')
NORMAL_PYTHON = Path('/home/member/.conda/envs/lama/bin/python')
EDGS = REPO/'submodules/EDGS'
INPAINT = REPO/'submodules/Inpaint360GS'
RETRAIN = REPO/'logs/retrain_reviewed_best_fit_20261103'
LABEL = 'PaintMesh fitlama 5000 (new track)'
CONTROL = 'PaintMesh fit 5000 (new track)'
POLICY = dict(idle_seconds=0, reserve_mib=1536, own_limit_mib=22500, sample_seconds=3, disk_free_gib=40)


def codes():
    return {**fit.codes(), str(SCRIPT.relative_to(REPO)):sha(SCRIPT), str(BINDER.relative_to(REPO)):sha(BINDER)}


def check_code(req):
    if req['code'] != codes(): raise ValueError('fitlama implementation changed after snapshot')
    for rec in req['official']['files'].values(): verify_record(rec)


def result_path(case):
    return case/'inpaint/inpainted_3dgs/evaluation/inpaint360_r4/results.json'


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'request.json').exists():
        check_code(read(root/'request.json')); return
    previous = read(RETRAIN/'request.json'); specs = []
    for old in previous['scenes']:
        if old['group'] != 'fit': continue
        scene = old['scene']; case = root/'scenes'/scene; case.mkdir(parents=True, exist_ok=True)
        verify_record(old['config']); verify_record(old['camera'])
        cfg = read(old['config']['path'])
        if cfg['iterations'] != 5000 or cfg['init']['max_points'] is not None:
            raise ValueError('Unexpected fit budget/point cap: '+scene)
        shutil.copy2(old['config']['path'], case/'config.json')
        write(case/'source.json', old['source'])
        spec = copy.deepcopy(old)
        spec.update(key=scene, group='fitlama', config=record(case/'config.json'),
            source=record(case/'source.json'), fit_scene=str(BASE/'fit/scenes'/scene),
            inpaint_root=str(case/'inpaint'), mask_dilation=10, mask_shape='square')
        tr = Path(old['removal_root'])/'tracker'
        if not read(tr/'visual_review.json')['accepted']: raise ValueError('Unreviewed tracker: '+scene)
        spec['tracking'] = record(tr/'tracking_session.json')
        spec['tracking_review'] = record(tr/'visual_review.json')
        fit.link(old['removal_root'], case/'removal')
        write(case/'spec.json', spec); specs.append(spec)
        write(case/'config_diff_from_fit.json', dict(training_config_changes=[],
            original_config=old['config'], camera_unchanged=True, tracker_unchanged=True,
            lama_changes=dict(old_dilation=old['mask_dilation'], old_shape='ellipse', dilation=10, shape='square',
                predictor='official RGB/depth scripts + official environment; PaintMesh normal extension retained',
                output_policy='raw official RGB/depth prediction; no PaintMesh known-pixel recomposition')))
        write(case/'status.json', dict(state='pending', updated_at=now()))
    baselines = {}
    for name, rec in previous['baselines'].items():
        verify_record(rec); dst = root/'baseline_snapshot'/f'{name}.json'; dst.parent.mkdir(exist_ok=True)
        shutil.copy2(rec['path'], dst); baselines[name] = record(dst)
    files = [OFFICIAL/'tools/prepare_lama_data.py', OFFICIAL_LAMA/'configs/prediction/default.yaml',
        OFFICIAL_LAMA/'bin/predict_color.py', OFFICIAL_LAMA/'bin/predict_depth.py']
    files += sorted((OFFICIAL_LAMA/'saicinpainting').rglob('*.py'))
    official = dict(python=str(OFFICIAL_PYTHON), model_root=str(OFFICIAL_LAMA/'big-lama'),
        files={str(p.relative_to(OFFICIAL)):record(p) for p in files},
        model={k:record(OFFICIAL_LAMA/'big-lama'/p) for k,p in [('config','config.yaml'),('checkpoint','models/best.ckpt')]})
    for exe, name in [(OFFICIAL_PYTHON,'official'), (NORMAL_PYTHON,'normal')]:
        freeze = subprocess.check_output([str(exe), '-m', 'pip', 'freeze'], text=True, env=dict(os.environ, PYTHONNOUSERSITE='1'))
        atomic(root/f'{name}_environment.txt', freeze)
    req = dict(created_at=now(), scenes=specs, code=codes(), official=official, baselines=baselines,
        source_fit_retraining_request=record(RETRAIN/'request.json'), resolution=4, iterations=5000,
        recursive_guide=False, init_max_points=None, gpu_policy=POLICY,
        uses_test_poses=True, uses_test_rgb_for_training=False, uses_test_masks_for_training=False)
    write(root/'request.json', req)
    for name in req['code']:
        dst = root/'code_snapshot'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(REPO/name, dst)
    for name, rec in official['files'].items():
        dst = root/'official_code_snapshot'/name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(rec['path'], dst)
    plan(root, req); report(root)


def plan(root, req):
    text = '''# fitlama：新 track 的 fit，仅更换 LaMa 配置与官方推理实现

## 固定项

11场景、1/4分辨率、5000步。相机直接复用 fit 的 test* COLMAP位姿插值（相机中心插值＋朝向SLERP），帧数、内参、已有删除模型与最新人工审核 SAM/DeAOT tracker 不变。逐场景 config.json 与新 track 的 fit 配置逐字节相同，包括初始化、局部背景策略、仅剪枝、损失及学习率。max_points=null，不增加点数上限。

新的 LaMa 结果会改变匹配点、深度初始化与局部名单；这是上游目标变化的下游影响。重新运行全部初始化及训练，不复用旧模型冒充新训练。

## 唯一实验模块：LaMa

- 全场景固定 min_area=50、8连通分量清理，保留合格分量，无合格分量时保留最大分量；21×21方形最大值滤波，即膨胀半径10。与官方 tools/prepare_lama_data.py 一致。新track中的全遮挡空mask保持空。
- RGB/depth直接运行本机官方 Inpaint360GS/LaMa/bin/predict_color.py、predict_depth.py，使用 inpaint360gs-lama 环境、官方 big-lama 权重。代码、环境、模型哈希保存在 request.json 和快照。
- RECURSIVE_GUIDE=false；refine=true；每尺度15次，lr=0.002，min_side=512，max_scales=3，px_budget=1800000，pad modulo=8。保持输入分辨率和官方内部预处理。
- 深度按官方 reference_depth 的 min/max 编码、补全、反归一化；RGB/depth原始官方输出直接作目标，不再额外把洞外值还原。输入仍是 fit 的 PGSR 删除后渲染，不能保证与官方不同轨迹/不同渲染器的输出相同。
- 官方没有法线补全。为保持 PaintMesh 几何监督不变，保留现有 predict_normal.py，使用相同新mask和同一权重。这是明确记录的 PaintMesh 扩展，不称为完整官方 Inpaint360GS 管线。
- fitlama_targets.py 检查完整帧集、尺寸、有限深度、法线方向与有效性、模型/相机/代码来源。临时副本仅用于复用洞外不变校验，实际训练与记录均指向未经修改的官方RGB/depth输出。

## 结果与对比

scenes/<scene>/inpaint/ 内保存 LaMa input/output、edgs_init、edgs_joint、inpainted_3dgs、inpainted_mesh。scenes/<scene>/removal 指向 fit 的已有 removal，仅作为输入。各阶段日志和收据在该scene的 logs/、receipts/；根目录 workers/gpu*/status.json、logs/gpu*.log 为后台状态。

comparison.md/csv/json 自动更新新 track 的 fit、fitlama、Inpaint360GS、Gaussian Grouping、AuraFusion360的7指标；未完成保持待完成。11场景共425测试帧，评估完整测试集；完成后给场景宏平均和前六项按帧数加权平均，FID为场景FID均值。三基线来自本机冻结复现结果，不冒充论文表值。fit使用测试位姿，但不使用测试RGB/GT mask训练；与三个基线的训练视角协议不同。

覆盖对象仅为 fitlama 本组输出和报告。原fit、best_results、fit-fix不改；本轮根目录原先不存在。重复启动按收据校验续跑，禁止混用参数不同的旧文件。

## 双卡调度

GPU0/1各一后台worker。CPU准备不占GPU；GPU阶段共享现有新track重训的每卡互斥锁。已有best_results/fit worker持锁时，fitlama等待，防止其CPU阶段结束后突然上卡撞车；锁释放后自动接续，无需再次手动运行。

每阶段检查可用显存：LaMa 11000MiB、初始化12000MiB、训练16000MiB、评估5000MiB、mesh10000MiB；运行期间余量低于1536MiB或自身达到22500MiB只停止本组子进程并标记风险，不停止其他任务。错误保留阶段日志，不通过降低分辨率/点数隐藏失败。
'''
    atomic(root/'TRAINING_PLAN.zh-CN.md', text)
    atomic(root/'README.md', '# PaintMesh fitlama\n\n[训练方案](TRAINING_PLAN.zh-CN.md) · [五组结果对比](comparison.md) · [状态](status.json) · [冻结配置与来源](request.json)\n\n新track fit → 官方LaMa RGB/depth → 原初始化/监督5000步 → 完整测试集评估与mesh。\n')


def report(root):
    with (root/'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); req=read(root/'request.json')
        values={k:{s:v['metrics'] for s,v in read(r['path'])['per_scene'].items()} for k,r in req['baselines'].items()}
        values.update({CONTROL:{}, LABEL:{}}); states={}; rows=[]
        for s in req['scenes']:
            case=root/'scenes'/s['scene']; states[s['scene']]=read(case/'status.json')
            for label, path, receipt in [
                (CONTROL, result_path(Path(s['fit_scene'])), RETRAIN/'scenes'/('fit__'+s['scene'])/'receipts/evaluate_5000.json'),
                (LABEL, result_path(case), case/'receipts/evaluate_5000.json')]:
                if receipt.exists() and path.exists():
                    r=read(path); validate_evaluation(r, read(s['source']['path'])['historical_evaluation'])
                    values[label][s['scene']]=r['metrics']
        lines=['# fitlama：新track fit及三个方法对比','',
            'fit/fitlama使用测试相机位姿设计轨迹，未使用test RGB/GT mask训练；三基线的训练轨迹未统一，属于诊断对比。全测试集评估。','',
            '| 场景 | 方法 | '+' | '.join(METRICS)+' |', '|---|---|'+'---:|'*len(METRICS)]
        for s in req['scenes']:
            for name, val in values.items():
                m=val.get(s['scene'])
                if m: rows.append(dict(scene=s['scene'],method=name,**{k:m[k] for k in METRICS}))
                lines.append('| '+s['scene']+' | '+name+' | '+' | '.join(f'{m[k]:.6f}' if m else '待完成' for k in METRICS)+' |')
        for name, val in values.items():
            if len(val)!=len(req['scenes']):continue
            for weighted in (False,True):
                label='425帧加权（FID场景均值）' if weighted else '11场景宏平均'
                avg={k:sum(val[s['scene']][k]*(s['frames'] if weighted and k!='FID' else 1) for s in req['scenes'])/sum(s['frames'] if weighted and k!='FID' else 1 for s in req['scenes']) for k in METRICS}
                rows.append(dict(scene=label,method=name,**avg)); lines.append('| '+label+' | '+name+' | '+' | '.join(f'{avg[k]:.6f}' for k in METRICS)+' |')
        complete=sum(v['state']=='complete' for v in states.values()); evaluated=len(values[LABEL])
        lines+=['',f'fitlama评估 {evaluated}/11；完整流程 {complete}/11。','', '[训练方案](TRAINING_PLAN.zh-CN.md) · [实时状态](status.json)','']
        atomic(root/'comparison.md','\n'.join(lines));write(root/'comparison.json',dict(updated_at=now(),rows=rows,evaluated=evaluated,complete=complete))
        buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['scene','method',*METRICS]);w.writeheader();w.writerows(rows);atomic(root/'comparison.csv',buf.getvalue())
        write(root/'status.json',dict(updated_at=now(),evaluated=evaluated,complete=complete,total=11,scenes=states,
            workers={p.parent.name:read(p) for p in (root/'workers').glob('*/status.json')}))


def gpu_decision(snap, own_pids=(), running=False):
    if running and (snap['free_mib']<POLICY['reserve_mib'] or sum(p['memory_mib'] for p in snap['processes'] if p['pid'] in own_pids)>=POLICY['own_limit_mib']):
        return 'memory_risk'
    return None


class Worker(guard.Worker):
    def __init__(self, root, gpu):
        guard.report=report;guard.check_code=check_code;guard.gpu_decision=gpu_decision;guard.GPU_POLICY=POLICY
        super().__init__(root,gpu)
        self.gpu_lock=(BASE/f'.reviewed_retrain_gpu{gpu}.lock').open('a')

    def wait_gpu(self, stage):
        need=16000 if stage=='train_5000' else 12000 if stage=='initialize' else 5000 if stage=='evaluate_5000' else 11000 if stage in ('official_color','official_depth','normal') else 10000
        while True:
            acquired=False
            try:
                fcntl.flock(self.gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                snap=guard.gpu_snapshot(self.uuid)
                if snap['free_mib']>=need:return
                reason=f'need_{need}_MiB_free'
            except BlockingIOError:
                snap=None;reason='existing_reviewed_retrain_worker_owns_card'
            finally:
                if acquired and ('snap' not in locals() or snap['free_mib']<need):fcntl.flock(self.gpu_lock,fcntl.LOCK_UN)
            self.save(state='waiting_for_gpu',stage=stage,reason=reason,gpu_snapshot=snap,child_pid=None)
            report(self.root);time.sleep(15)

    def execute(self,stage,cmd,cwd,expected,gpu=False):
        previous=dict(self.env)
        if Path(cwd)==INPAINT:self.env['PYTHONPATH']=str(INPAINT)
        elif Path(cmd[0])==OFFICIAL_PYTHON:self.env['PYTHONPATH']=str(OFFICIAL_LAMA)
        elif Path(cmd[0])==NORMAL_PYTHON:self.env['PYTHONPATH']=str(INPAINT/'LaMa')
        try:return super().execute(stage,cmd,cwd,expected,gpu=gpu)
        finally:
            self.env=previous
            if gpu:fcntl.flock(self.gpu_lock,fcntl.LOCK_UN)

    def prepare_inputs(self,s):
        inp=Path(s['inpaint_root']);rem=Path(s['removal_root']);tr=rem/'tracker';work=inp/'work_model'
        for k in ('config','camera','tracking','tracking_review'):verify_record(s[k])
        fit.Worker.commit_masks(self,s)
        self.execute('workspace',[PYTHON,INPAINT/'tools/prepare_inpaint_workspace.py',
            '--removal-workspace',rem/'work_model','--removal-manifest',rem/'removal_manifest.json',
            '--tracking-session',tr/'tracking_session.json','--camera-manifest',s['camera']['path'],
            '--tracking-masks',tr/'results/images/images_masks','--source-iteration',2000,'--output',work],
            INPAINT,[work/'workspace_manifest.json'])
        self.execute('lama_inputs',[PYTHON,INPAINT/'tools/prepare_paintmesh_lama_data.py','prepare',
            '--tracking-masks',work/'tracking_masks','--removed-rgb',work/'virtual/ours_object_removal/iteration_2000/renders',
            '--removed-depth',work/'virtual/ours_object_removal/iteration_2000/depth','--reference-depth',work/'virtual/ours_2000/depth',
            '--camera-manifest',s['camera']['path'],'--color-input',inp/'lama/input/color','--depth-input',inp/'lama/input/depth',
            '--manifest',inp/'manifests/lama_input_manifest.json','--frames',len(read(s['camera']['path'])['cameras']),
            '--min-area',50,'--dilation',10,'--dilation-shape','square','--allow-empty-masks'],INPAINT,[inp/'manifests/lama_input_manifest.json'])
        workspace=self.case/'official_lama';workspace.mkdir(exist_ok=True)
        for name in ('bin','configs','saicinpainting','big-lama'):fit.link(OFFICIAL_LAMA/name,workspace/name)
        data_name='360_'+s['scene']+'_virtual'
        for name in ('color','depth'):
            (workspace/'data'/name).mkdir(parents=True,exist_ok=True);(workspace/'output'/name).mkdir(parents=True,exist_ok=True)
            (inp/'lama/output'/name).mkdir(parents=True,exist_ok=True)
            fit.link(inp/'lama/input'/name,workspace/'data'/name/data_name)
            fit.link(inp/'lama/output'/name,workspace/'output'/name/data_name)
        return workspace,data_name

    def process(self,s):
        self.case=self.root/'scenes'/s['scene'];self.state=dict(pid=os.getpid(),gpu=self.gpu,gpu_uuid=self.uuid,scene=s['scene'])
        self.save(state='preflight',stage='verify_inputs');check_code(self.req)
        workspace,data_name=self.prepare_inputs(s)
        inp=Path(s['inpaint_root']);rem=Path(s['removal_root']);cfg=read(s['config']['path']);row=read(s['source']['path'])
        camera=s['camera']['path'];lp=inp/'manifests/lama_completion_manifest.json';ip=inp/'manifests/edgs_init_manifest.json';jp=inp/'manifests/edgs_joint_manifest.json'
        stems=read(inp/'manifests/lama_input_manifest.json')['parameters']['frame_names']
        for name,suffix in [('color','.png'),('depth','.npy')]:
            self.execute('official_'+name,[OFFICIAL_PYTHON,'bin/predict_'+name+'.py','--data_name',data_name],workspace,
                [inp/'lama/output'/name/(stem+suffix) for stem in stems],gpu=True)
        self.execute('normal',[NORMAL_PYTHON,INPAINT/'LaMa/bin/predict_normal.py','--input-dir',inp/'lama/input/normal',
            '--output-dir',inp/'lama/output/normal','--input-manifest',inp/'manifests/lama_input_manifest.json',
            '--model-path',OFFICIAL_LAMA/'big-lama'],INPAINT/'LaMa',[inp/'lama/output/normal/prediction.json'],gpu=True)
        self.execute('bind_targets',[PYTHON,BINDER,'--case',self.case],REPO,[lp,self.case/'official_lama_validation.json'])
        ic=rem/f'config/object_inpaint/Inpaint360GS/inpaint360/{s["scene"]}.json'
        self.execute('initialize',[PYTHON,EDGS/'tools/initialize_paintmesh_edgs.py','--removed-model',rem/'removed_3dgs/model_manifest.json',
            '--inpaint-config',ic,'--lama',lp,'--camera',camera,'--output-root',inp/'edgs_init','--manifest',ip,'--config',s['config']['path']],EDGS,[ip],gpu=True)
        local=[]
        if cfg['local_background']['enabled']:
            sp=inp/'manifests/local_selection_manifest.json';cmd=[PYTHON,EDGS/'tools/select_paintmesh_local_background.py','--initialization',ip,'--lama',lp,'--camera',camera,
                '--output-root',inp/'local_selection','--manifest',sp,'--config',s['config']['path']]
            for target in s['target_ids']:cmd+=['--target-ply',rem/f'work_model/point_cloud_object_removal/iteration_2000/point_cloud_{target}.ply']
            self.execute('local_selection',cmd,EDGS,[sp],gpu=True);local=['--local-selection',sp]
        ec=row['inputs']['edgs_config']['path']
        self.execute('train_5000',[PYTHON,EDGS/'tools/train_paintmesh_pgsr.py','--initialization',ip,'--lama',lp,'--camera',camera,
            '--edgs-config',ec,'--output-root',inp/'edgs_joint','--manifest',jp,'--config',s['config']['path'],*local],EDGS,[jp],gpu=True)
        model=inp/'inpainted_3dgs';ply=inp/'edgs_joint/point_cloud/iteration_5000/point_cloud.ply';mm=model/'model_manifest.json'
        cmd=[PYTHON,INPAINT/'tools/publish_inpainted_edgs_model.py','--inpainted-ply',ply,
            '--classifier',inp/'work_model/point_cloud/iteration_2000/classifier.pth','--edgs-config',ec,'--cfg-args',inp/'work_model/cfg_args',
            '--source-iteration',2000,'--output-iteration',5000,'--target-ids',','.join(map(str,s['target_ids'])),'--surrounding-ids','none',
            '--removed-model-manifest',rem/'removed_3dgs/model_manifest.json','--removal-manifest',rem/'removal_manifest.json',
            '--workspace-manifest',inp/'work_model/workspace_manifest.json','--tracking-session',rem/'tracker/tracking_session.json',
            '--lama-manifest',lp,'--edgs-joint-manifest',jp,'--inpaint-config',ic,'--output',model]
        self.execute('publish',cmd,INPAINT,[mm],gpu=False)
        self.execute('evaluate_5000',[PYTHON,REPO/'scripts/paintmesh/eval_inpaint360.py','run','--model-path',model,'--source-path',s['source_path'],
            '--resolution',4,'--iteration',5000,'--batch-size',4],REPO,[result_path(self.case)],gpu=True)
        validate_evaluation(read(result_path(self.case)),row['historical_evaluation']);report(self.root)
        mesh=model/'mesh/ours_5000';raw=mesh/'tsdf_fusion_post.ply'
        self.execute('mesh_render',[PYTHON,EDGS/'render.py','--model-path',model,'--source-path',s['source_path'],'--images','images','--resolution',4,'--iteration',5000,
            '--renderer','pgsr','--extract-mesh','--max-depth',5.,'--voxel-size',.002,'--num-clusters',1,'--skip-test'],EDGS,[raw,model/'train/ours_5000/render_manifest.json'],gpu=True)
        gp=model/'point_cloud/iteration_5000/point_cloud.ply'
        self.execute('mesh_publish',[PYTHON,INPAINT/'tools/publish_inpaint_mesh.py','--model-manifest',mm,'--gaussian-ply',gp,'--mesh',raw,
            '--train-render-manifest',model/'train/ours_5000/render_manifest.json','--output',mesh/'mesh_manifest.json','--iteration',5000,
            '--source-path',s['source_path'],'--images','images','--resolution',4,'--max-depth',5.,'--voxel-size',.002,'--num-clusters',1],INPAINT,[mesh/'mesh_manifest.json'],gpu=False)
        sem=inp/'inpainted_mesh'
        self.execute('mesh_semantics',[PYTHON,INPAINT/'tools/lift_gaussian_semantics_to_mesh.py','--gaussian-ply',gp,
            '--classifier',model/'point_cloud/iteration_5000/classifier.pth','--scene-info',Path(s['source_path'])/'associated_hqsam/scene.json','--mesh',raw,
            '--output-dir',sem,'--write-colored-ply','--overwrite'],INPAINT,[sem/'semantic_manifest.json'],gpu=False)
        fit.link(raw,sem/'geometry.ply')
        self.execute('finalize',[PYTHON,INPAINT/'tools/finalize_inpaint_result.py','--model-manifest',mm,'--mesh-manifest',mesh/'mesh_manifest.json',
            '--semantic-manifest',sem/'semantic_manifest.json','--removal-manifest',rem/'removal_manifest.json','--workspace-manifest',inp/'work_model/workspace_manifest.json',
            '--lama-manifest',lp,'--gaussian-ply',gp,'--mesh',raw,'--geometry',sem/'geometry.ply','--output',inp/'inpaint_manifest.json'],INPAINT,[inp/'inpaint_manifest.json'],gpu=False)
        self.save(state='complete',stage='complete',child_pid=None,reason=None,finished_at=now());report(self.root)


    def run(self):
        with (self.wdir/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN);signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
            for s in self.req['scenes']:
                case=self.root/'scenes'/s['scene']
                with (case/'run.lock').open('a') as sl:
                    try:fcntl.flock(sl,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if read(case/'status.json')['state'] in ('complete','failed','blocked_memory_risk'):continue
                    try:self.process(s)
                    except guard.MemoryRisk as e:
                        self.save(state='blocked_memory_risk',reason=str(e),child_pid=None);traceback.print_exc();report(self.root)
                    except Exception as e:
                        self.save(state='failed',reason=str(e),child_pid=None);traceback.print_exc();report(self.root)
            self.case=None;self.save(state='queue_finished',scene=None,stage=None,child_pid=None);report(self.root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=BASE/'fitlama')
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
            state=root/f'workers/gpu{gpu}/status.json';old=read(state) if state.exists() else {};pid=old.get('pid');proc=Path(f'/proc/{pid}/cmdline')
            if pid and proc.exists() and str(SCRIPT).encode() in proc.read_bytes():workers[str(gpu)]=dict(pid=pid,state='already_running');continue
            log=root/f'logs/gpu{gpu}.log';log.parent.mkdir(exist_ok=True)
            with log.open('a') as f:
                child=subprocess.Popen([PYTHON,str(SCRIPT),'--root',str(root),'--worker',str(gpu)],cwd=REPO,
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            workers[str(gpu)]=dict(pid=child.pid,log=str(log))
        write(root/'workers.json',workers);print(workers,flush=True)


if __name__=='__main__':main()
