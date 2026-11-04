#!/usr/bin/env python3
"""fit-fix: independent fit LaMa -> local Difix -> new initialization/training."""
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
import time
import traceback
from run_geometry_ablation import REPO,METRICS,read,write,atomic,now,sha
from edgs_inpaint_io import record,verify_record,verify_targets,load_config
from run_muti_plan import PYTHON,validate_evaluation
import run_fit_test_poses as fit
import run_hard_reference_all as guard

BASE=fit.BASE
SCRIPT=Path(__file__).resolve()
PREDICTOR=SCRIPT.with_name('difix_targets.py')
DIFIX_ROOT=REPO.parent/'Difix3D'
MODEL_ROOT=REPO.parent/'ckpt/difix'
DIFIX_PYTHON=Path('/home/member/.conda/envs/img2img-turbo/bin/python')
EDGS=REPO/'submodules/EDGS';INPAINT=REPO/'submodules/Inpaint360GS'
LABEL='PaintMesh fit-fix 5000';CONTROL='PaintMesh fit 5000'
POLICY=dict(idle_seconds=0,start_free_mib=12000,reserve_mib=1536,own_limit_mib=22500,sample_seconds=2,
    foreign_process_policy='do_not_wait_for_foreign_process_exit; admit_by_free_memory; never_signal_foreign',disk_free_gib=40)


def codes():
    return {**fit.codes(),str(SCRIPT.relative_to(REPO)):sha(SCRIPT),str(PREDICTOR.relative_to(REPO)):sha(PREDICTOR)}


def check_code(req):
    if req['code']!=codes():raise ValueError('fit-fix implementation changed; refusing mixed results')


def gpu_decision(snapshot,own_pids=(),running=False):
    if running:
        own=sum(p['memory_mib'] for p in snapshot['processes'] if p['pid'] in own_pids)
        if snapshot['free_mib']<POLICY['reserve_mib'] or own>=POLICY['own_limit_mib']:return 'memory_risk'
    return None


def result_path(case):return case/'inpaint/inpainted_3dgs/evaluation/inpaint360_r4/results.json'


def prepare(root,scope):
    root.mkdir(parents=True,exist_ok=True)
    if (root/'request.json').exists():
        req=read(root/'request.json');check_code(req)
        if req['difix']['scope']!=scope:raise ValueError('Existing fit-fix scope differs; use another experiment root')
        return
    parent=read(BASE/'fit/request.json');fit.check_code(parent);specs=[]
    for s in parent['scenes']:
        case=root/'scenes'/s['scene'];case.mkdir(parents=True,exist_ok=True)
        for k in ('config','source','camera','camera_source'):verify_record(s[k])
        cfg=load_config(s['config']['path'])
        if cfg['iterations']!=5000 or cfg['init']['max_points'] is not None:raise ValueError('Unexpected fit configuration')
        shutil.copy2(s['config']['path'],case/'config.json');shutil.copy2(s['source']['path'],case/'source.json')
        spec=copy.deepcopy(s);spec.update(config=record(case/'config.json'),source=record(case/'source.json'),
            fit_scene=str(BASE/'fit/scenes'/s['scene']),inpaint_root=str(case/'inpaint'))
        fit.link(s['removal_root'],case/'removal')
        specs.append(spec);write(case/'status.json',dict(state='pending',updated_at=now()))
    baselines={}
    for name,entry in parent['baselines'].items():
        verify_record(entry['snapshot']);dest=root/'baseline_snapshot'/fit.METHODS[name]/'evaluation_summary.json'
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(entry['snapshot']['path'],dest)
        baselines[name]=dict(original=entry['original'],snapshot=record(dest),source_policy=entry['source_policy'])
    models={str(p.relative_to(MODEL_ROOT)):record(p) for p in sorted(MODEL_ROOT.rglob('*'))
            if p.is_file() and '.cache' not in p.parts and (p.suffix in ('.json','.safetensors','.py') or p.name=='merges.txt')}
    env=subprocess.check_output([str(DIFIX_PYTHON),'-m','pip','freeze'],text=True,env=dict(os.environ,PYTHONNOUSERSITE='1'))
    atomic(root/'difix_environment.txt',env)
    req=dict(created_at=now(),root=str(root),source_fit_request=record(BASE/'fit/request.json'),scenes=specs,
        code=codes(),baselines=baselines,resolution=4,iterations=5000,seed=0,init_max_points=None,gpu_policy=POLICY,
        difix=dict(scope=scope,prompt='remove degradation',timestep=199,guidance_scale=0.,steps=1,seed=0,
                   dtype='float32',reference=False,resize='none; reflect pad bottom/right to multiple 8; crop exact original size'),
        difix_python=str(DIFIX_PYTHON),difix_environment=record(root/'difix_environment.txt'),model_root=str(MODEL_ROOT),
        model_files=models,pipeline=record(DIFIX_ROOT/'src/pipeline_difix.py'),
        difix_commit=subprocess.check_output(['git','-C',str(DIFIX_ROOT),'rev-parse','HEAD'],text=True).strip(),
        uses_test_poses=True,uses_test_rgb_for_training=False,uses_test_masks_for_training=False)
    for name in req['code']:
        dest=root/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
    write(root/'request.json',req);plan(root,req);report(root)


def plan(root,req):
    atomic(root/'TRAINING_PLAN.zh-CN.md',f'''# fit-fix：独立 LaMa 后加入 Difix 的全场景实验

本轮是 PaintMesh 新消融，不是 Difix3D 官方渐进训练流程的完整复现。

## 固定方案

继承 fit 的全部11场景、425测试帧，1/4分辨率、test* COLMAP相机中心线性插值＋朝向SLERP，帧数和内参逐场景相同。复用 fit 已完成的删除、跟踪、独立LaMa RGB/depth/normal；尚未生成的输入等待 fit 原队列完成，不拿旧轨迹补全替代。RECURSIVE_GUIDE=false。

LaMa RGB → Difix单步去伪影 → 新RGB训练目标 → 按原算法重新匹配/反投影初始化 → 原外观＋几何监督训练5000步 → 原生PGSR渲染评估与mesh。不是只给最终测试渲染加后处理；测试图不经过Difix，也不用于选择参考/调参。

- 本地权重：{MODEL_ROOT}，nvidia/difix 无参考图版本，模型参数冻结。
- 官方参数：prompt=remove degradation，num_inference_steps=1，timesteps=[199]，guidance_scale=0。
- FP32，batch=1，固定每帧seed=0+帧序号；保持原分辨率，只在右/下反射补齐到8倍数，输出裁回原尺寸，不缩放相机或mask。
- Difix作用范围：{req['difix']['scope']}。完整输出存入difix/raw；训练图存入lama/output/color。mask模式只替换原LaMa洞内，洞外逐像素保持；full模式使用整图输出。
- 深度、法线和有效区域逐字节复用fit。初始化算法/损失/学习率/背景更新/仅剪枝策略逐场景与fit相同，init.max_points=null。RGB变化会改变特征匹配、初始化点及局部名单，这是新增模块的下游影响，不能称为固定初始化的纯损失消融。
- 每个场景完整config.json与fit配置内容相同；cube等场景沿用自己的历史设置，未统一强制改成truck。
- Difix可能生成错误纹理或跨视角不一致，不能保证指标提升；保留全部场景、全部指标和负结果。

## 数据来源与评估

相机直接复用fit的camera_manifest，使用测试位姿属于test-pose-conditioned实验，未见视角泛化结论需另做独立轨迹验证。所有GT RGB/mask仅评估时读取。沿用本机冻结的三方法结果，不当作论文原表数值。

comparison.md/csv/json列出 Inpaint360GS、Gaussian Grouping、AuraFusion360、PaintMesh fit 5000、PaintMesh fit-fix 5000。fit列随其结果完成后登记；前六指标按425帧加权，并另给场景宏平均；FID是11场景FID平均，不冒称pooled FID。未齐全的列不计算全场景均值。

## 双卡后台执行

GPU0/1各一worker，共享场景队列。按本次要求不因存在其他任务就等待，也不按利用率排队；按实际空闲显存启动：Difix≥14000MiB，初始化≥10000MiB，训练≥18000MiB，评估≥5000MiB，mesh≥10000MiB。不向其他任务发送停止信号。每2秒监测，本任务使用≥22500MiB或整卡余量<1536MiB则只停止本任务并标记风险。不是硬件预留，无法保证其他进程突然分配时不发生OOM。失败场景保留日志并继续其余场景。

原fit调度器有自行让卡策略，可能自行等待。fit-fix仅在原LaMa输入完成后消费；缺少上游时释放GPU等待，避免阻塞其继续产出。

## 路径

- request.json：输入、模型SHA256、官方代码commit、环境、设置。
- scenes/<scene>/inpaint/lama_original：原fit LaMa的只读来源入口。
- scenes/<scene>/inpaint/difix/raw：Difix完整输出。
- scenes/<scene>/inpaint/difix/frames：逐帧输入输出SHA、变化、显存峰值。
- scenes/<scene>/inpaint/difix/preview.jpg：均匀抽样的LaMa/Difix/训练目标对比。
- scenes/<scene>/inpaint/lama/output/color：实际训练RGB。
- scenes/<scene>/inpaint/manifests：明确标注Difix派生来源的补全、初始化、训练清单。
- scenes/<scene>/inpaint/edgs_init、edgs_joint、inpainted_3dgs、inpainted_mesh：新初始化、训练、最终渲染/评估及mesh。
- scenes/<scene>/logs：阶段日志；logs/gpu*.log：后台worker日志；status.json：实时进度。

[官方Difix代码](https://github.com/nv-tlabs/Difix3D) · [论文](https://arxiv.org/abs/2503.01774)
''')
    atomic(root/'README.md','# PaintMesh fit-fix\n\n[训练方案](TRAINING_PLAN.zh-CN.md) · [五方法对比](comparison.md) · [实时状态](status.json) · [配置及来源](request.json)\n\n独立LaMa → Difix → 重新初始化 → 5000步训练；两张GPU后台队列。\n')


def report(root):
    with (root/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);req=read(root/'request.json');rows=[];states={};evaluated=0;finished=0
        values={k:{} for k in (*fit.METHODS,CONTROL,LABEL)}
        for name,r in req['baselines'].items():
            values[name]={n:v['metrics'] for n,v in read(r['snapshot']['path'])['per_scene'].items()}
        for s in req['scenes']:
            case=root/'scenes'/s['scene'];state=read(case/'status.json');states[s['scene']]=state
            finished+=state['state']=='complete';ref=read(s['source']['path'])['historical_evaluation']
            for label,path in [(CONTROL,result_path(Path(s['fit_scene']))),(LABEL,result_path(case))]:
                if path.exists():
                    result=read(path);validate_evaluation(result,ref);values[label][s['scene']]=result['metrics']
                    if label==LABEL:evaluated+=1
        lines=['# fit-fix：与原fit及三个方法对比','',
            'PaintMesh fit/fit-fix使用test位姿设计训练虚拟视角，三方法未统一重训同一轨迹；用于诊断，不能当作完全相同训练视角协议。所有方法评估完整测试集合，不对测试渲染应用Difix。','',
            '| 场景 | 方法 | '+' | '.join(METRICS)+' |','|---|---|'+'---:|'*len(METRICS)]
        for s in req['scenes']:
            for name,v in values.items():
                m=v.get(s['scene'])
                if m is not None:
                    rows.append(dict(scene=s['scene'],method=name,**{k:m[k] for k in METRICS}));line=' | '.join(f'{m[k]:.6f}' for k in METRICS)
                else:line=' | '.join('—' for _ in METRICS)
                lines.append(f'| {s["scene"]} | {name} | '+line+' |')
        for name,v in values.items():
            if len(v)!=len(req['scenes']):continue
            for weighted in (False,True):
                label='425帧加权（FID场景均值）' if weighted else '11场景宏平均'
                m={k:sum(v[s['scene']][k]*(s['frames'] if weighted and k!='FID' else 1) for s in req['scenes'])/sum(s['frames'] if weighted and k!='FID' else 1 for s in req['scenes']) for k in METRICS}
                rows.append(dict(scene=label,method=name,**m));lines.append(f'| {label} | {name} | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
        lines+=['',f'fit-fix已评估 {evaluated}/11；完整流程 {finished}/11。','', '[状态](status.json) · [方案](TRAINING_PLAN.zh-CN.md)','']
        atomic(root/'comparison.md','\n'.join(lines));write(root/'comparison.json',dict(updated_at=now(),rows=rows,evaluated=evaluated,complete=finished))
        buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['scene','method',*METRICS]);w.writeheader();w.writerows(rows);atomic(root/'comparison.csv',buf.getvalue())
        write(root/'status.json',dict(updated_at=now(),evaluated=evaluated,complete=finished,total=11,scenes=states,
              workers={p.parent.name:read(p) for p in (root/'workers').glob('*/status.json')}))


def ready(s):
    parent=Path(s['fit_scene']);manifest=parent/'inpaint/manifests/lama_completion_manifest.json'
    return manifest.exists() and (parent/'receipts/inpaint_3_3.json').exists()


class Worker(guard.Worker):
    def __init__(self,root,gpu):
        guard.report=report;guard.check_code=check_code;guard.gpu_decision=gpu_decision;guard.GPU_POLICY=POLICY
        super().__init__(root,gpu)

    def wait_gpu(self,stage):
        required=10000 if stage=='difix' else 12000 if stage=='train_5000' else 4000 if stage=='evaluate_5000' else 6000 if stage in ('initialize','local_selection') else 8000
        while True:
            try:snap=guard.gpu_snapshot(self.uuid);reason=None if snap['free_mib']>=required else f'need_{required}_MiB_free'
            except Exception as e:snap={};reason=str(e)
            if reason is None:return
            self.save(state='waiting_for_memory',stage=stage,reason=reason,gpu_snapshot=snap,child_pid=None);report(self.root);time.sleep(15)

    def execute(self,stage,cmd,cwd,expected,gpu=False):
        # Direct execution of Inpaint360GS/tools requires its repository root
        # on sys.path, as run_inpaint.sh provides for the original fit pipeline.
        previous=dict(self.env)
        if Path(cwd).resolve()==INPAINT.resolve():self.env['PYTHONPATH']=str(INPAINT)
        try:return super().execute(stage,cmd,cwd,expected,gpu=gpu)
        finally:self.env=previous

    def targets(self,s):
        case=self.case;inp=Path(s['inpaint_root']);parent=Path(s['fit_scene'])/'inpaint';inp.mkdir(exist_ok=True)
        for k in ('config','source','camera','camera_source'):verify_record(s[k])
        for src,dst in [(parent/'work_model',inp/'work_model'),(parent/'lama',inp/'lama_original')]:fit.link(src,dst)
        (inp/'lama/output').mkdir(parents=True,exist_ok=True)
        fit.link(parent/'lama/input',inp/'lama/input')
        for name in ('depth','normal'):fit.link(parent/'lama/output'/name,inp/'lama/output'/name)
        manifest=parent/'manifests/lama_completion_manifest.json'
        req_path=inp/'difix/request.json'
        if req_path.exists():
            r=read(req_path);verify_record(r['parent_lama']);verify_record(r['camera'])
            if r['parameters']!=self.req['difix'] or r['model_files']!=self.req['model_files']:raise ValueError('Difix inputs changed')
            return req_path
        lama,inputs,_=verify_targets(manifest,s['camera']['path'],use_depth=True,use_normal=True)
        cases=[]
        for stem,f in sorted(lama['frames'].items()):
            cases.append(dict(name=stem,input=record(f['outputs']['color']['path']),mask=record(inputs['frames'][stem]['outputs']['color_mask']['path']),
                raw=str(inp/'difix/raw'/f'{stem}.png'),target=str(inp/'lama/output/color'/f'{stem}.png'),metadata=str(inp/'difix/frames'/f'{stem}.json')))
        write(req_path,dict(parent_lama=record(manifest),camera=s['camera'],model_root=self.req['model_root'],model_files=self.req['model_files'],pipeline=self.req['pipeline'],
            parameters=self.req['difix'],cases=cases,receipt=str(inp/'difix/prediction_receipt.json'),completion_manifest=str(inp/'manifests/lama_completion_manifest.json')))
        return req_path

    def process(self,s):
        self.case=self.root/'scenes'/s['scene'];self.state=dict(pid=os.getpid(),gpu=self.gpu,gpu_uuid=self.uuid,scene=s['scene'])
        self.save(state='preflight',stage='verify_fit_inputs');report(self.root);check_code(self.req)
        req_path=self.targets(s);inp=Path(s['inpaint_root']);rem=Path(s['removal_root']);cfg=read(s['config']['path']);row=read(s['source']['path'])
        camera=s['camera']['path'];lp=inp/'manifests/lama_completion_manifest.json';ip=inp/'manifests/edgs_init_manifest.json';jp=inp/'manifests/edgs_joint_manifest.json'
        self.execute('difix',[DIFIX_PYTHON,PREDICTOR,'infer','--request',req_path],REPO,[inp/'difix/prediction_receipt.json'],gpu=True)
        self.execute('bind_targets',[PYTHON,PREDICTOR,'bind','--request',req_path],REPO,[lp],gpu=False)
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
            while True:
                pending=False;progress=False
                for s in self.req['scenes']:
                    case=self.root/'scenes'/s['scene']
                    with (case/'run.lock').open('a') as sl:
                        try:fcntl.flock(sl,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:pending=True;continue
                        state=read(case/'status.json')['state']
                        if state in ('complete','failed','blocked_memory_risk'):continue
                        if not ready(s):
                            pending=True;write(case/'status.json',dict(state='waiting_for_fit_lama',updated_at=now(),source=s['fit_scene']));continue
                        progress=True
                        try:self.process(s)
                        except guard.MemoryRisk as e:
                            self.save(state='blocked_memory_risk',child_pid=None,reason=str(e));traceback.print_exc();report(self.root)
                        except Exception as e:
                            self.save(state='failed',child_pid=None,reason=str(e));traceback.print_exc();report(self.root)
                self.case=None
                if not pending:break
                self.save(state='waiting_for_upstream_or_other_worker',scene=None,stage=None,child_pid=None);report(self.root);time.sleep(30)
            self.save(state='queue_finished',scene=None,stage=None,child_pid=None);report(self.root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=BASE/'fit-fix');p.add_argument('--scope',choices=('mask','full'),default='mask')
    p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--prepare-only',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.worker is not None:Worker(root,a.worker).run();return
    if a.report:report(root);return
    root.mkdir(parents=True,exist_ok=True)
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare(root,a.scope)
        if a.prepare_only:return
        workers={}
        for gpu in (0,1):
            state=root/f'workers/gpu{gpu}/status.json';old=read(state) if state.exists() else {};pid=old.get('pid');proc=Path(f'/proc/{pid}/cmdline')
            if pid and proc.exists() and str(SCRIPT).encode() in proc.read_bytes():workers[str(gpu)]=dict(pid=pid,state='already_running');continue
            log=root/f'logs/gpu{gpu}.log';log.parent.mkdir(exist_ok=True)
            with log.open('a') as f:
                child=subprocess.Popen([PYTHON,str(SCRIPT),'--root',str(root),'--worker',str(gpu)],cwd=REPO,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            workers[str(gpu)]=dict(pid=child.pid,log=str(log))
        write(root/'workers.json',workers);print(workers,flush=True)


if __name__=='__main__':main()
