#!/usr/bin/env python3
"""Matched, half-strength reference-completion ablation for the truck pilot."""
import argparse
import copy
import fcntl
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

from run_geometry_ablation import REPO, METRICS, read, write, atomic, now
from run_reference_context_pilot import Worker as BaseWorker, codes
from run_muti_plan import PYTHON, Runner, validate_evaluation
from shared_reference_completion import load_scene, consistency
from edgs_inpaint_io import (record, verify_record, verify_targets, identity, validate_init,
    validate_selection, seal, INIT_KIND, SELECTION_KIND)

ALPHA = .5
SCRIPT = Path(__file__).resolve()


def mix(independent, shared, original, hole, alpha):
    if not 0 <= alpha <= 1 or independent.shape != shared.shape or independent.shape != original.shape:
        raise ValueError('Invalid RGB mix')
    if hole.shape != original.shape[:2] or hole.dtype != np.bool_:
        raise ValueError('Invalid hole mask')
    out = original.copy()
    # Whole-hole interpolation, not confidence at individual pixels. This is an
    # explicit target-strength experiment and makes no claim of GT correctness.
    out[hole] = np.rint((1-alpha)*independent[hole].astype(np.float32)
                       +alpha*shared[hole].astype(np.float32)).clip(0,255).astype(np.uint8)
    return out


def prepare(parent):
    root = parent/'truck_soft_reference'
    if (root/'request.json').exists():
        verify_record(read(root/'request.json')['producer']); return root
    original = read(parent/'request.json')
    spec = copy.deepcopy(next(x for x in original['scenes'] if x['scene']=='truck'))
    dest = root/'scenes/truck'; dest.mkdir(parents=True,exist_ok=True)
    row = read(verify_record(spec['source']))
    write(dest/'source.json',row); write(dest/'config.json',row['variants'][0]['config'])
    spec.update(source=record(dest/'source.json'),config=record(dest/'config.json'))
    shared = Path(spec['hard_warp'])/'manifests/lama_completion_manifest.json'
    req=dict(created_at=now(),root=str(root),scenes=[spec],code=codes(),producer=record(SCRIPT),
        alpha=ALPHA,shared_lama=record(shared),parent_request=record(parent/'request.json'),
        steps=[5000,10000],primary_step=5000,gate=original['gate'],
        policy='RGB targets only: 0.5 independent + 0.5 hard-reference completion inside full original mask; known pixels exactly preserved. No new LaMa inference or test-based alpha selection.')
    write(root/'request.json',req)
    atomic(root/'TRAINING_PLAN.zh-CN.md','''# truck：参考补全强度消融

已有固定参考上下文组未带来收益，几何硬引导组的 masked_PSNR 改善但后期 FID 变差。本组检查降低参考目标强度是否减少这项取舍。

只改 RGB 目标：洞内 `T = 0.5 × 独立 LaMa + 0.5 × 上轮几何引导补全`，洞外严格使用原输入。alpha 固定为 0.5，不搜索 test 最优权重。不重新推理 LaMa，也不修改 depth/normal、初始点云、相机、mask、局部更新名单、剪枝、损失、seed 或训练设置。

这只是明确的目标强度消融，不把混合权重称为置信度；混合可能导致重影、模糊。先检查均匀抽取的六个视角，明显伪影时停止。数值图像检查包括洞外逐字节不变、混合公式正确、来源/相机一致；这些不能证明遮挡区域正确。该组沿用既有两组已验证输入，不冒用上下文模型的伪洞数值筛查结论。

GPU 1，训练 10000 步并保存真实 5000 步快照；两节点均评估所有 40 个 test 视角与七项指标。主要节点为 5000，10000 为稳定性复核。报告独立 LaMa、硬引导、固定上下文和半强度四组，不只展示有利指标。通过阈值沿用父实验；单场景通过不自动推广全部场景。

`request.json` 固定参数和来源；`scenes/truck/screen` 为输入预览和检查；`scenes/truck/status.json` 为进度；`comparison.md` 为评估。
''')
    atomic(root/'README.md','# truck 参考目标半强度试验\n\n[方案](TRAINING_PLAN.zh-CN.md) · [状态](scenes/truck/status.json) · [对比](comparison.md)\n')
    report(root)
    return root


def targets(root):
    request=read(root/'request.json');verify_record(request['producer'])
    dest=root/'scenes/truck';row=read(verify_record(request['scenes'][0]['source']))
    lama,inputs,cams,stems,data=load_scene(row)
    shared,_,_=verify_targets(verify_record(request['shared_lama']),row['inputs']['camera']['path'],use_depth=True,use_normal=True)
    if shared['input_artifact_id']!=lama['input_artifact_id'] or set(shared['frames'])!=set(stems):
        raise ValueError('Independent and reference target inputs differ')
    frames={};completed=[]
    for i,stem in enumerate(stems):
        for key in ('depth','normal'):
            if shared['frames'][stem]['outputs'][key]!=lama['frames'][stem]['outputs'][key]:
                raise ValueError('Geometry differs')
        ref=np.asarray(Image.open(verify_record(shared['frames'][stem]['outputs']['color'])).convert('RGB'))
        original=data[i]['original'];hole=data[i]['mask'];independent=data[i]['rgb']
        if ref.shape!=original.shape or not np.array_equal(ref[~hole],original[~hole]):
            raise ValueError('Reference changed known RGB')
        result=mix(independent,ref,original,hole,request['alpha'])
        p=dest/'lama/output/color'/f'{stem}.png';p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            if not np.array_equal(np.asarray(Image.open(p).convert('RGB')),result):raise ValueError('Existing target differs')
        else:Image.fromarray(result).save(p)
        frames[stem]=dict(independent=lama['frames'][stem]['outputs']['color'],shared=shared['frames'][stem]['outputs']['color'],output=record(p))
        completed.append(result)
    validation=dict(complete=True,alpha=request['alpha'],outside_mask_byte_exact=True,formula_verified=True,
        depth_and_normal_unchanged=True,frame_count=len(stems),
        consistency=consistency(cams,stems,data,completed),frames=frames)
    write(dest/'target_validation.json',validation)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    choices=np.linspace(0,len(stems)-1,6).astype(int);fig,axs=plt.subplots(6,3,figsize=(14,16))
    for j,i in enumerate(choices):
        stem=stems[i];ys,xs=np.where(data[i]['mask']);y0=max(0,ys.min()-24);y1=min(data[i]['mask'].shape[0],ys.max()+25)
        x0=max(0,xs.min()-24);x1=min(data[i]['mask'].shape[1],xs.max()+25)
        hard=np.asarray(Image.open(frames[stem]['shared']['path']).convert('RGB'))
        for col,(im,label) in enumerate([(data[i]['rgb'],'Independent'),(hard,'Hard reference'),(completed[i],'50% reference')]):
            axs[j,col].imshow(im[y0:y1,x0:x1]);axs[j,col].axis('off');axs[j,col].set_title(stem+': '+label)
    (dest/'screen').mkdir(exist_ok=True);fig.tight_layout();fig.savefig(dest/'screen/target_comparison.png',dpi=140);plt.close(fig)
    print('Targets ready:',len(stems),flush=True)


def bind(root):
    request=read(root/'request.json');verify_record(request['producer']);dest=root/'scenes/truck'
    if not read(dest/'screen/review.json')['approved']:raise ValueError('Visual screen rejected')
    row=read(verify_record(request['scenes'][0]['source']));validation=read(dest/'target_validation.json')
    original_lama,_,_=verify_targets(row['inputs']['lama']['path'],row['inputs']['camera']['path'],use_depth=True,use_normal=True)
    derived=copy.deepcopy(original_lama);derived['created_at']=now();derived['roots']['color_output']=str(dest/'lama/output/color')
    derived['parameters']['soft_reference_target']=dict(alpha=request['alpha'],baseline=row['inputs']['lama'],
        shared=request['shared_lama'],producer=request['producer'],validation=record(dest/'target_validation.json'),
        visual_review=record(dest/'screen/review.json'),request=record(root/'request.json'))
    for stem,frame in validation['frames'].items():
        p=verify_record(frame['output']);derived['frames'][stem]['outputs']['color']=dict(record(p),mtime_ns=p.stat().st_mtime_ns)
    derived['artifact_id']=identity(dict(kind=derived['kind'],schema_version=derived['schema_version'],
        input_artifact_id=derived['input_artifact_id'],model={k:v['sha256'] for k,v in derived['model'].items()},
        parameters=derived['parameters'],outputs={s:{k:v['sha256'] for k,v in f['outputs'].items()} for s,f in derived['frames'].items()}))
    lp=dest/'manifests/lama_completion_manifest.json';write(lp,derived)
    verify_targets(lp,row['inputs']['camera']['path'],use_depth=True,use_normal=True)
    old=validate_init(row['inputs']['initialization']['path'])
    ii=dict(old['inputs'],lama=record(lp),proposal_initialization=row['inputs']['initialization'])
    ir=copy.deepcopy(old['request']);ir.update(inputs=ii,variant='fixed_initialization_half_reference_target',
        output_root=str(dest/'edgs_init'),manifest=str(dest/'manifests/edgs_init_manifest.json'),
        reuse_policy='Reuse every original Gaussian field/support row, initialized using independent LaMa')
    initial=seal(INIT_KIND,request=ir,inputs=ii,outputs=old['outputs'],point_count=old['point_count'],parameters=old['parameters'],report=old['report'])
    validate_init(initial);ip=dest/'manifests/edgs_init_manifest.json';write(ip,initial)
    new=copy.deepcopy(row);new['inputs'].update(lama=record(lp),initialization=record(ip))
    if 'local_selection' in row['inputs']:
        original=validate_selection(row['inputs']['local_selection']['path'],initial=old)
        si=dict(original['inputs'],lama=record(lp),initialization=record(ip),baseline_selection=row['inputs']['local_selection'])
        sr=copy.deepcopy(original['request']);sr.update(inputs=si,output_root=str(dest/'local_selection'),policy='Identical eligibility; only training RGB targets changed')
        selection=seal(SELECTION_KIND,request=sr,inputs=si,outputs=original['outputs'],initialization_id=initial['artifact_id'])
        validate_selection(selection,initial=initial,config=row['variants'][0]['config']['local_background'])
        sp=dest/'manifests/local_selection_manifest.json';write(sp,selection);new['inputs']['local_selection']=record(sp)
    new['publish']['inputs']['lama_manifest']=record(lp);write(dest/'training_source.json',new)


def report(root):
    req=read(root/'request.json');spec=req['scenes'][0];dest=root/'scenes/truck';parent=Path(req['parent_request']['path']).parent
    sources={'independent':Path(spec['baseline']),'hard_reference':Path(spec['hard_warp']),
        'context_only':parent/'scenes/truck/A_context','half_reference':dest}
    lines=['# truck 参考目标强度完整对比','','5000 为主要节点；10000 为稳定性复核。','',
        '| 步数 | 组 | '+' | '.join(METRICS)+' |','|---:|---|'+'---:|'*len(METRICS)]
    rows=[];pending=[]
    for step in req['steps']:
        baseline=read(sources['independent']/f'inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json')
        for name,folder in sources.items():
            p=folder/f'inpainted_3dgs_{step}/evaluation/inpaint360_r4/results.json'
            if not p.exists():pending.append(f'{name}/{step}');continue
            result=read(p);validate_evaluation(result,baseline);m=result['metrics']
            rows.append(dict(step=step,variant=name,metrics=m,delta={k:m[k]-baseline['metrics'][k] for k in METRICS},result=record(p)))
            lines.append(f'| {step} | {name} | '+' | '.join(f'{m[k]:.6f}' for k in METRICS)+' |')
    checks={};variant=[r for r in rows if r['variant']=='half_reference']
    if len(variant)==2:
        main=next(r for r in variant if r['step']==5000);end=next(r for r in variant if r['step']==10000)
        b=next(r['metrics'] for r in rows if r['variant']=='independent' and r['step']==5000);d=main['delta'];g=req['gate']
        checks=dict(masked_psnr=d['masked_PSNR']>=g['masked_psnr_gain_min'],full_psnr=d['PSNR']>=-g['psnr_drop_max'],
            masked_lpips=d['masked_LPIPS']<=b['masked_LPIPS']*g['masked_lpips_relative_increase_max'],
            fid=d['FID']<=b['FID']*g['fid_relative_increase_max'],
            final_masked_psnr=end['delta']['masked_PSNR']>=-g['step10000_masked_psnr_drop_max'])
    write(root/'comparison.json',dict(updated_at=now(),rows=rows,pending=pending,checks=checks,all_scene_started=False))
    atomic(root/'comparison.md','\n'.join(lines)+'\n\n待完成：'+str(pending)+'。仅单场景验证，不自动推广。\n')


class Worker(BaseWorker):
    def execute(self,*args,**kwargs):
        verify_record(self.req['producer'])
        return super().execute(*args,**kwargs)

    def run(self):
        with (self.dest/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN);signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
            for key in ('source','config'):verify_record(self.spec[key])
            verify_record(self.req['shared_lama'])
            self.execute('targets',[PYTHON,SCRIPT,'targets','--root',self.root],REPO,[self.dest/'target_validation.json',self.dest/'screen/target_comparison.png'])
            review=self.dest/'screen/review.json'
            while not review.exists():self.save(state='awaiting_visual_review',stage='screen',child_pid=None);time.sleep(10)
            if not read(review)['approved']:
                self.save(state='screen_rejected',stage='complete',finished_at=now());return
            self.execute('bind',[PYTHON,SCRIPT,'bind','--root',self.root],REPO,[self.dest/'training_source.json'])
            row=read(self.dest/'training_source.json')
            self.execute('train',Runner.training_command(self,row,self.dest),REPO/'submodules/EDGS',
                [self.dest/'manifests/edgs_joint_manifest.json',self.dest/'edgs_joint/snapshots/iteration_5000/manifest.json'],gpu=True)
            from run_inpaint360_full import export_snapshot
            export_snapshot(self.dest,5000)
            self.execute('publish',Runner.publish_command(self,row,self.dest),REPO/'submodules/Inpaint360GS',[self.dest/'inpainted_3dgs_10000/model_manifest.json'])
            for step in self.req['steps']:
                model=self.dest/f'inpainted_3dgs_{step}';result=model/'evaluation/inpaint360_r4/results.json'
                self.execute(f'eval_{step}',[PYTHON,REPO/'scripts/paintmesh/eval_inpaint360.py','run',
                    '--model-path',model,'--source-path',row['source'],'--resolution',4,'--iteration',step,'--batch-size',4],REPO,[result],gpu=True)
                validate_evaluation(read(result),row['historical_evaluation']);report(self.root)
            self.save(state='complete',stage='complete',child_pid=None,finished_at=now());report(self.root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('launch','worker','targets','bind','report'))
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if a.action=='launch':
        root=prepare(root);status=root/'scenes/truck/status.json'
        if status.exists():raise ValueError('Already launched; inspect status before explicitly resuming worker')
        with (root/'scenes/truck/worker.log').open('a') as log:
            child=subprocess.Popen([PYTHON,SCRIPT,'worker','--root',root],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'workers.json',dict(truck=dict(pid=child.pid,gpu=1)));print(root,child.pid)
    elif a.action=='worker':
        w=Worker(root,'truck')
        try:w.run()
        except BaseException as e:
            if not isinstance(e,InterruptedError):w.save(state='failed',error=str(e),child_pid=None)
            traceback.print_exc();sys.exit(1)
    else:globals()[a.action](root)
