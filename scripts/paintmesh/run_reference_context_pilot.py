#!/usr/bin/env python3
"""Two-GPU fixed-context pilot, screened before immutable matched 3DGS training."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from run_geometry_ablation import REPO,METRICS,read,write,atomic,sha,now
from run_muti_plan import Runner,PYTHON,validate_evaluation
from edgs_inpaint_io import implementation,record,verify_record
from context_completion_io import VARIANTS,PARAMETERS

OLD=REPO/'output/paintmesh/Inpaint360GS/muti_plan'
HARD=OLD.parent/'shared_reference_pilot_20261101'
LAMA='/home/member/.conda/envs/lama/bin/python'
IO=Path(__file__).with_name('context_completion_io.py')
PREDICTOR=Path(__file__).with_name('lama_context_predict.py')
GPUS={'cube':0,'truck':1}


def codes():
    result=implementation()
    for p in (Path(__file__),IO,PREDICTOR,Path(__file__).with_name('shared_reference_completion.py'),
        Path(__file__).with_name('run_geometry_ablation.py'),Path(__file__).with_name('run_muti_plan.py'),
        Path(__file__).with_name('run_inpaint360_full.py'),Path(__file__).with_name('eval_inpaint360.py'),
        REPO/'submodules/Inpaint360GS/LaMa/configs/prediction/default.yaml',
        REPO/'submodules/Inpaint360GS/tools/publish_inpainted_edgs_model.py'):
        result[str(p.relative_to(REPO))]=sha(p)
    # Model modules are dependencies of the frozen network as well as the ckpt.
    for rel in ('training/modules/ffc.py','training/trainers/__init__.py','training/trainers/default.py'):
        p=REPO/'submodules/Inpaint360GS/LaMa/saicinpainting'/rel;result[str(p.relative_to(REPO))]=sha(p)
    return result


def prepare(root):
    if (root/'request.json').exists():
        if read(root/'request.json')['code']!=codes():raise ValueError('Existing pilot code changed')
        return
    source=read(OLD/'request.json');rows=[]
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuids={int(line.split(',')[0]):line.split(',')[1].strip() for line in devices.strip().splitlines()}
    for scene,gpu in GPUS.items():
        row=next(x for x in source['scenes'] if x['scene']==scene)
        baseline=OLD/'scenes'/scene/'00_control';prior=HARD/'scenes'/scene
        joint=read(baseline/'manifests/edgs_joint_manifest.json')
        if joint['request']['implementation']!=implementation():raise ValueError('3D trainer differs from baseline')
        for step in (5000,10000):
            validate_evaluation(read(baseline/f'inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json'),row['historical_evaluation'])
        dest=root/'scenes'/scene;dest.mkdir(parents=True,exist_ok=True)
        write(dest/'source.json',row);shutil.copyfile(prior/'guide/evidence.json',dest/'prior_guide.json')
        for name,path in [('independent_control',baseline),('hard_warp_control',prior)]:
            if not (dest/name).exists():(dest/name).symlink_to(path,target_is_directory=True)
        rows.append(dict(scene=scene,gpu=gpu,gpu_uuid=uuids[gpu],source=record(dest/'source.json'),
            prior_guide=record(dest/'prior_guide.json'),baseline=str(baseline),hard_warp=str(prior),test_frames=row['frame_count']))
    req=dict(created_at=now(),root=str(root),code=codes(),scenes=rows,variants=VARIANTS,parameters=PARAMETERS,
        steps=[5000,10000],primary_step=5000,resolution=4,seed=0,
        gate=dict(masked_psnr_gain_min=.2,psnr_drop_max=.05,masked_lpips_relative_increase_max=.05,
            fid_relative_increase_max=.05,step10000_masked_psnr_drop_max=.1),
        screen_policy='6 fixed real training photos: first 2 diagnostic, last 4 validation; mean ROI delta >= -0.5 dB and worst >= -3 dB; visual review required; no test GT')
    write(root/'request.json',req)
    atomic(root/'TRAINING_PLAN.zh-CN.md', '''# 完整 mask 的固定参考补全：双卡验证

本轮只改变训练 RGB 目标。GPU 0=cube，GPU 1=truck；每个场景依次跑 A_context、B_context_opt10。

| 组 | 完整洞 mask | 参考 | 潜变量更新 |
|---|---|---|---:|
| 已有独立 LaMa | 是 | 无 | 原有行为 |
| 已有硬重投影 | 否 | 固定参考硬写入 | 原有行为 |
| A_context | 是 | 固定参考并排作为上下文 | 0 |
| B_context_opt10 | 是 | 与 A 完全相同 | 10 次实际 optimizer.step |

参考仍为 cube 00078 / truck 00000，参与引导的帧集合沿用上一轮，不叠加参考重选或逐帧递归。RECURSIVE_GUIDE=false。网络权重冻结；B 仅优化 local/global 两个特征张量，Adam lr=0.002，损失为参考重建 L1 + 当前侧已知区域 L1 + 0.01×相对初始特征的归一化平方偏移。当前洞内无重投影像素硬约束。显式支持单尺度，日志验证 10 次梯度更新及 A/B 零步输出一致。

## 开始 3D 训练前

每场景生成 6 个均匀选取的虚拟视角预览；另取 6 张真实 train 照片，按 1/4 缩放，在关联分割的目标 ID 膨胀 12px 之外选固定背景矩形，比较独立补全/A/B 与被隐藏的原像素。区域大小约 16%图宽×18%图高，位置按距离目标中心排序，不按补全效果选取。前 2 张为诊断、后 4 张为验证；参数事先固定，不依据 test 调参。

数值筛查要求：验证区域平均 ROI PSNR 比独立补全下降不超过 0.5 dB，最差下降不超过 3 dB。另须检查虚拟洞纹理连接/颜色污染、网络权重与已知像素不变、B 实际更新 10 次。严重失败组不进入训练。该自测只排除明显问题，不能证明从未观察的洞正确。

screen/review.json 记录本次助手对预览图的检查结论，不需要用户手工批准；等待时不占 GPU。所有测试照片只在最终评估使用。

## 3D 训练控制变量

复用 muti_plan/00_control 的初始化 PLY 全部字段、相机、mask、depth/normal 目标、背景更新名单、优化器、3D 损失及 seed。cube 保留其 hemisphere90、dilation20、RGB 初始化、旧点冻结；truck 保留 adaptive180、dilation10、depth 初始化、局部旧点更新。两者均仅剪枝，不重采样或截断初始点云。

每组训练 10000 步，保存真实 5000 步快照。两节点都评估全部 test 视角和七项指标；5000 是主要节点，10000 是稳定性复核。

推广复核要求：同一个变体在两个场景主要节点 masked_PSNR ≥ 对照+0.2 dB，全图 PSNR 下降≤0.05 dB，masked_LPIPS/FID 增加≤5%，10000 masked_PSNR 不下降超过0.1 dB，并通过可视检查。未通过不全场景重训；通过也先复核后再安排未参与调整的场景，不自动覆盖 best_results。

## 文件位置

- request.json：输入、实现哈希、固定参数和判断标准。
- scenes/<scene>/screen：训练背景自测、虚拟视角预览、数值结果和人工视觉检查记录。
- scenes/<scene>/<variant>/lama/output/color：新训练目标；manifests：完整来源记录。
- scenes/<scene>/<variant>/edgs_joint：点云、检查点、训练诊断。
- scenes/<scene>/<variant>/inpainted_3dgs_{5000,10000}/evaluation/inpaint360_r4：test 渲染和指标。
- scenes/<scene>/status.json、logs：实时任务状态和逐阶段日志。
- comparison.md/json、decision.json：所有方法/节点完整比较，不只展示有利结果。
''')
    atomic(root/'README.md','# 固定参考上下文补全试验\n\n[训练方案](TRAINING_PLAN.zh-CN.md) · [完整指标](comparison.md) · [配置](request.json)\n\nGPU 0：cube；GPU 1：truck。先检查二维补全，再进入 3D 训练。旧实验结果保留。\n')
    report(root)


def report(root):
    with (root/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);req=read(root/'request.json');rows=[];pending=[];checks={};rejected=[]
        lines=['# 固定参考上下文与历史对照','', '| 场景 | 步数 | 方法 | '+' | '.join(METRICS)+' |','|---|---:|---|'+'---:|'*len(METRICS)]
        for scene in req['scenes']:
            d=root/'scenes'/scene['scene'];review=read(d/'screen/review.json') if (d/'screen/review.json').exists() else None
            for step in req['steps']:
                baseline=read(Path(scene['baseline'])/f'inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json')
                entries=[('independent',Path(scene['baseline'])),('hard_warp',Path(scene['hard_warp']))]+[(v,d/v) for v in VARIANTS]
                for name,folder in entries:
                    p=folder/f'inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json'
                    if not p.exists():
                        if review is not None and name in VARIANTS and name not in review['approved_variants']:
                            rejected.append(scene['scene']+'/'+name)
                        else:pending.append(f"{scene['scene']}/{name}/{step}")
                        continue
                    result=read(p);validate_evaluation(result,baseline)
                    delta={k:result['metrics'][k]-baseline['metrics'][k] for k in METRICS}
                    rows.append(dict(scene=scene['scene'],step=step,variant=name,metrics=result['metrics'],delta=delta,result=record(p)))
                    lines.append(f"| {scene['scene']} | {step} | {name} | "+' | '.join(f"{result['metrics'][k]:.6f}" for k in METRICS)+' |')
        for v in VARIANTS:
            selected=[x for x in rows if x['variant']==v];g=req['gate'];valid=[]
            for s in req['scenes']:
                primary=next((x for x in selected if x['scene']==s['scene'] and x['step']==5000),None)
                final=next((x for x in selected if x['scene']==s['scene'] and x['step']==10000),None)
                if primary and final:
                    b=read(Path(s['baseline'])/'inpainted_3dgs_5000/evaluation/inpaint360_r4/results.json')['metrics'];d=primary['delta']
                    valid.append(d['masked_PSNR']>=g['masked_psnr_gain_min'] and d['PSNR']>=-g['psnr_drop_max']
                        and d['masked_LPIPS']<=b['masked_LPIPS']*g['masked_lpips_relative_increase_max']
                        and d['FID']<=b['FID']*g['fid_relative_increase_max']
                        and final['delta']['masked_PSNR']>=-g['step10000_masked_psnr_drop_max'])
            checks[v]=dict(evaluated_scenes=len(valid),eligible_for_review=len(valid)==2 and all(valid))
        decision=dict(updated_at=now(),pending=pending,screen_rejected=sorted(set(rejected)),checks=checks,
                      all_scene_started=False,requires_visual_review_before_expansion=True)
        write(root/'comparison.json',dict(rows=rows,decision=decision));write(root/'decision.json',decision)
        lines+=['','待完成：'+str(len(pending))+'；二维筛查未通过：'+(', '.join(sorted(set(rejected))) or '暂无记录')+'。',
                '','[判据与状态](decision.json) · [训练方案](TRAINING_PLAN.zh-CN.md)','']
        atomic(root/'comparison.md','\n'.join(lines))


class Worker:
    def __init__(self,root,scene):
        self.root=root;self.dest=root/'scenes'/scene;self.req=read(root/'request.json');self.scene=scene;self.child=None
        self.spec=next(s for s in self.req['scenes'] if s['scene']==scene)
        self.state=dict(scene=scene,pid=os.getpid(),gpu=self.spec['gpu'])
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES=self.spec['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        self.env.pop('PYTHONPATH',None)

    def save(self,**kw):self.state.update(kw,updated_at=now());write(self.dest/'status.json',self.state)

    def stop(self,*_):
        if self.child and self.child.poll() is None:os.killpg(self.child.pid,signal.SIGTERM)
        self.save(state='stopped');raise InterruptedError('Stopped, artifacts/checkpoints preserved')

    def execute(self,stage,cmd,cwd,expected,gpu=False):
        if codes()!=self.req['code']:raise ValueError('Immutable experiment source changed')
        cmd=list(map(str,cmd));receipt=self.dest/'receipts'/(stage+'.json')
        if receipt.exists():
            r=read(receipt)
            if r['command']!=cmd:raise ValueError('Completed command changed')
            for value in r['artifacts']:verify_record(value)
            return
        if shutil.disk_usage(self.root).free<30*1024**3:raise RuntimeError('Less than 30 GiB disk free')
        if gpu:
            while True:
                free=int(subprocess.check_output(['nvidia-smi','-i',self.spec['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=10000:break
                self.save(state='waiting_for_gpu_memory',stage=stage,free_mib=free);time.sleep(20)
        log=self.dest/'logs'/(stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        self.save(state='running',stage=stage,progress=None,command=cmd,log=str(log));print(now(),self.scene,stage,flush=True)
        env=dict(self.env)
        if cmd[0]==LAMA:env['PYTHONPATH']=str(REPO/'submodules/Inpaint360GS/LaMa')
        with log.open('a') as f:
            f.write('\nStarting '+now()+'\n');f.flush()
            self.child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            self.save(child_pid=self.child.pid)
            for line in self.child.stdout:
                f.write(line);f.flush()
                if line.startswith(('PGSR joint ','Context ','Screen:','Targets ready:')):
                    self.save(progress=line.rstrip());print(line.rstrip(),flush=True)
            code=self.child.wait()
        self.save(child_pid=None)
        if code:raise RuntimeError(f'{stage} failed ({code}): {log}')
        write(receipt,dict(command=cmd,complete=True,finished_at=now(),artifacts=[record(p) for p in expected]))

    def run(self):
        with (self.dest/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN);signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
            for k in ('source','prior_guide'):verify_record(self.spec[k])
            self.execute('prepare',[PYTHON,IO,'prepare','--root',self.dest],REPO,[self.dest/'screen/request.json',self.dest/'screen/input_manifest.json'])
            self.execute('screen_predict',[LAMA,PREDICTOR,'--request',self.dest/'screen/request.json'],REPO/'submodules/Inpaint360GS/LaMa',[self.dest/'screen/prediction_receipt.json'],gpu=True)
            self.execute('screen_measure',[PYTHON,IO,'screen','--root',self.dest],REPO,[self.dest/'screen/results.json'])
            review=self.dest/'screen/review.json'
            while not review.exists():
                self.save(state='awaiting_visual_review',stage='screen',child_pid=None);time.sleep(10)
            for variant in VARIANTS:
                if variant not in read(review)['approved_variants']:continue
                if not read(self.dest/'screen/results.json')['gates'][variant]['numeric_pass']:raise ValueError('Visual review cannot override numerical failure')
                folder=self.dest/variant;self.save(variant=variant)
                self.execute(variant+'_targets',[LAMA,PREDICTOR,'--request',folder/'prediction_request.json'],REPO/'submodules/Inpaint360GS/LaMa',[folder/'prediction_receipt.json'],gpu=True)
                self.execute(variant+'_bind',[PYTHON,IO,'bind','--root',self.dest,'--variant',variant],REPO,[folder/'training_source.json',folder/'completion_validation.json'])
                row=read(folder/'training_source.json')
                self.execute(variant+'_train',Runner.training_command(self,row,folder),REPO/'submodules/EDGS',
                    [folder/'manifests/edgs_joint_manifest.json',folder/'edgs_joint/snapshots/iteration_5000/manifest.json'],gpu=True)
                from run_inpaint360_full import export_snapshot
                export_snapshot(folder,5000)
                self.execute(variant+'_publish',Runner.publish_command(self,row,folder),REPO/'submodules/Inpaint360GS',[folder/'inpainted_3dgs_10000/model_manifest.json'])
                for step in (5000,10000):
                    model=folder/f'inpainted_3dgs_{step}';result=model/'evaluation/inpaint360_r4/results.json'
                    self.execute(variant+f'_eval_{step}',[PYTHON,REPO/'scripts/paintmesh/eval_inpaint360.py','run','--model-path',model,
                        '--source-path',row['source'],'--resolution',4,'--iteration',step,'--batch-size',4],REPO,[result],gpu=True)
                    validate_evaluation(read(result),row['historical_evaluation']);report(self.root)
            self.save(state='complete' if read(review)['approved_variants'] else 'screen_rejected',stage='complete',child_pid=None,finished_at=now());report(self.root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',choices=tuple(GPUS));p.add_argument('--prepare-only',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if a.report:report(root);return
    if a.worker:
        worker=Worker(root,a.worker)
        try:worker.run()
        except BaseException as e:
            if not isinstance(e,InterruptedError):worker.save(state='failed',error=str(e),child_pid=None)
            traceback.print_exc();sys.exit(1)
        return
    prepare(root)
    if a.prepare_only:return
    processes={}
    for scene in GPUS:
        d=root/'scenes'/scene;status=read(d/'status.json') if (d/'status.json').exists() else {}
        if status.get('state') in ('complete','screen_rejected'):continue
        proc=Path(f"/proc/{status.get('pid',0)}/cmdline")
        if proc.exists() and str(root).encode() in proc.read_bytes():raise RuntimeError('Worker already running: '+scene)
        with (d/'worker.log').open('a') as log:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--root',str(root),'--worker',scene],
                cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[scene]=dict(pid=child.pid,gpu=GPUS[scene]);write(root/'workers.json',processes)
    print(root,processes,flush=True)


if __name__=='__main__':main()
