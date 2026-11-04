#!/usr/bin/env python3
"""Run isolated, recorded PaintMesh repairs and compare full held-out evaluations."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from run_geometry_ablation import read, write, now, sha, METRICS

REPO=Path(__file__).resolve().parents[2]
PY='/home/member/.conda/envs/paintmesh/bin/python'


def report(job):
    request=read(job/'request.json');plan=read(job/'experiments.json')
    failures={};running={}
    for queue in [job]+sorted(p.parent for p in job.glob('*/request.json')):
        failure_path=queue/'failures.json'
        if failure_path.exists():failures.update(read(failure_path))
        status_path=queue/'status.json'
        if status_path.exists():
            status=read(status_path)
            child=status.get('child_pid')
            if status.get('state')=='running' and child:
                command_path=Path('/proc')/str(child)/'cmdline'
                try:active=command_path.read_bytes().replace(b'\0',b' ').decode(errors='replace')
                except FileNotFoundError:active=''
                command=status.get('command',[])
                if len(command)>1 and str(command[1]) in active:
                    running[status['scene']+'/'+status['experiment']]=status
    rows=[]
    for case in plan:
        source=next(x for x in request['scenes'] if x['scene']==case['scene'])
        result=Path(case['output'])/'inpainted_3dgs/evaluation/inpaint360_r4/results.json'
        row=dict(scene=case['scene'],experiment=case['name'],changes=case['changes'],output=case['output'],baseline=source['metrics_5000'])
        if result.exists():
            data=read(result)
            baseline=read(Path(source['root'])/'inpainted_3dgs_5000/evaluation/inpaint360_r4/results.json')
            assert data['complete'] and data['protocol']==baseline['protocol']
            for k in ('frames','source_path','resolution','renderer','reference_images','official_masks','colmap_images','colmap_cameras'):
                assert data['inputs'][k]==baseline['inputs'][k],k
            row.update(state='complete',metrics=data['metrics'],delta={k:data['metrics'][k]-row['baseline'][k] for k in METRICS},results=str(result))
        else:
            key=case['scene']+'/'+case['name']
            if key in running:
                row.update(state='running',stage=running[key]['stage'],log=running[key]['log'])
            elif key in failures:
                row.update(state='failed',error=failures[key])
                detail=Path(case['output'])/'edgs_init/failure.json'
                if detail.exists():row['failure_detail']=read(detail).get('error')
            else:row['state']='pending'
        rows.append(row)
    write(job/'comparison.json',dict(updated_at=now(),scenes=rows))
    lines=['# PaintMesh 质量修复实验','',f'更新：{now()}。GPU 1；独立输出；原基线保留。',
           '', '相同真实测试相机、分辨率和官方 mask；训练及标注不读取测试 GT。表中变化均相对本批次原 5000 步结果。',
           '', '| 场景 | 实验 | PSNR 原→新 | ΔPSNR | masked PSNR 原→新 | SSIM | LPIPS | FID | 状态 |',
           '|---|---|---|---:|---|---:|---:|---:|---|']
    for r in rows:
        if 'metrics' in r:
            b,m=r['baseline'],r['metrics']
            lines.append(f"| {r['scene']} | {r['experiment']} | {b['PSNR']:.4f}→{m['PSNR']:.4f} | {r['delta']['PSNR']:+.4f} | {b['masked_PSNR']:.4f}→{m['masked_PSNR']:.4f} | {m['SSIM']:.4f} | {m['LPIPS']:.4f} | {m['FID']:.3f} | 完成 |")
        else:
            state={'failed':'失败，见下方原因','running':'运行中：'+r.get('stage',''),'pending':'尚未完成，见 status.json'}[r['state']]
            lines.append(f"| {r['scene']} | {r['experiment']} | — | — | — | — | — | — | {state} |")
    for r in rows:
        if r['state']=='failed':
            lines+=['',f"- {r['scene']} / {r['experiment']}：{r.get('failure_detail') or r['error']}。未计为完成。"]
    lines+=['','全部七项指标与差值见 comparison.json；实验改动见 experiments.json。模型优劣按完整测试集记录，不替换基线为单项最优拼接结果。']
    (job/'comparison.md').write_text('\n'.join(lines)+'\n')
    if (job/'parent_job.json').exists():report(Path(read(job/'parent_job.json')['job']))


class Runner:
    def __init__(self,job):
        self.job=job;self.req=read(job/'request.json');self.child=None
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(self.req['gpu']),GPU=str(self.req['gpu']),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
        self.env.pop('PYTHONPATH',None)

    def stop(self,*args):
        if self.child and self.child.poll() is None:os.killpg(self.child.pid,signal.SIGTERM)
        raise InterruptedError('Repair runner stopped; completed receipts are retained')

    def execute(self,case,stage,cmd,cwd,env=None):
        receipt=self.job/'receipts'/case['scene']/case['name']/(stage+'.json')
        if receipt.exists():
            old=read(receipt)
            if old['command']!=list(map(str,cmd)):raise ValueError('Completed command changed')
            return
        log=self.job/f"{case['scene']}_{case['name']}_{stage}.log"
        state=dict(state='running',scene=case['scene'],experiment=case['name'],stage=stage,time=now(),log=str(log),output=case['output'],command=list(map(str,cmd)))
        write(self.job/'status.json',state);print(state,flush=True)
        with log.open('a') as handle:
            self.child=subprocess.Popen(list(map(str,cmd)),cwd=cwd,env=env or self.env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            write(self.job/'status.json',dict(state,child_pid=self.child.pid));code=self.child.wait()
        if code:raise RuntimeError(f'{stage} exited {code}: {log}')
        write(receipt,dict(state,state='complete',finished_at=now()))

    def fixed_inputs(self,case):
        source=next(x for x in self.req['scenes'] if x['scene']==case['scene'])
        old=Path(case.get('input_root',source['root']));root=Path(case['output']);root.mkdir(parents=True,exist_ok=True)
        (root/'manifests').mkdir(exist_ok=True)
        cfg=copy.deepcopy(source['config']);cfg['iterations']=5000
        for section,changes in case['config_updates'].items():cfg[section].update(changes)
        config=root/'config.json'
        if config.exists():assert read(config)==cfg
        else:write(config,cfg)
        joint=read(old/'manifests/edgs_joint_manifest.json');inputs=copy.deepcopy(joint['inputs'])
        old_selection=read(Path(inputs['local_selection']['path']))
        select=[PY,REPO/'submodules/EDGS/tools/select_paintmesh_local_background.py']
        for k in ('initialization','lama','camera'):select+=['--'+k,inputs[k]['path']]
        for k,v in old_selection['inputs'].items():
            if k.startswith('target_'):select+=['--target-ply',v['path']]
        local=root/'manifests/edgs_local_selection_manifest.json'
        select+=['--config',config,'--output-root',root/'local_selection','--manifest',local]
        self.execute(case,'local_selection',select,REPO/'submodules/EDGS')
        train=[PY,REPO/'submodules/EDGS/tools/train_paintmesh_pgsr.py']
        for flag,key in [('initialization','initialization'),('lama','lama'),('camera','camera'),('edgs-config','edgs_config')]:train+=['--'+flag,inputs[key]['path']]
        train+=['--local-selection',local,'--config',config,'--output-root',root/'edgs_joint','--manifest',root/'manifests/edgs_joint_manifest.json']
        self.execute(case,'train_5000',train,REPO/'submodules/EDGS')
        self.publish_evaluate(case,old)

    def publish_evaluate(self,case,reference):
        root=Path(case['output']);published=read(reference/'inpainted_3dgs/model_manifest.json');params=published['parameters']
        ids=lambda v:','.join(map(str,v)) if v else 'none'
        publish=[PY,'-m','tools.publish_inpainted_edgs_model','--inpainted-ply',root/'edgs_joint/point_cloud/iteration_5000/point_cloud.ply',
                 '--source-iteration',str(params['source_iteration']),'--output-iteration','5000','--target-ids',ids(params['target_ids']),
                 '--surrounding-ids',ids(params['surrounding_ids'])]
        for flag,key in [('classifier','classifier'),('edgs-config','edgs_config'),('cfg-args','cfg_args'),('removed-model-manifest','removed_model_manifest'),
                         ('removal-manifest','removal_manifest'),('workspace-manifest','workspace_manifest'),('tracking-session','tracking_session'),
                         ('lama-manifest','lama_manifest'),('inpaint-config','inpaint_config')]:
            publish+=['--'+flag,published['inputs'][key]['path']]
        publish+=['--edgs-joint-manifest',root/'manifests/edgs_joint_manifest.json','--output',root/'inpainted_3dgs']
        self.execute(case,'publish',publish,REPO/'submodules/Inpaint360GS')
        self.evaluate(case)

    def initialization_only(self,case):
        """Change new-point initialization while preserving exact old-row eligibility."""
        from run_surface_initialization_ablation import make_variant, initialization_command, rebind_selection
        source=next(x for x in self.req['scenes'] if x['scene']==case['scene'])
        old=Path(source['root']);root=Path(case['output'])
        (root/'manifests').mkdir(parents=True,exist_ok=True)
        filtered=case.get('initialization_method')=='consistency_filter'
        cfg=copy.deepcopy(source['config']) if filtered else make_variant(source['config'])['config']
        cfg['iterations']=5000
        config=root/'config.yaml'
        if config.exists():assert read(config)==cfg
        else:write(config,cfg)
        joint=read(old/'manifests/edgs_joint_manifest.json');inputs=copy.deepcopy(joint['inputs'])
        initial=read(inputs['initialization']['path'])
        command=([PY,REPO/'scripts/paintmesh/prepare_consistent_initialization.py','--baseline',old,
            '--evidence-summary',case['evidence_summary'],'--scene',case['scene'],'--output-root',root,'--config',config]
            if filtered else initialization_command(PY,initial,root))
        self.execute(case,'initialization',command,REPO/'submodules/EDGS')
        initialization=root/'manifests/edgs_init_manifest.json'
        local,stats,new=rebind_selection(inputs,initialization,cfg,root/'local_selection')
        write(root/'local_selection_check.json',stats)
        train=[PY,REPO/'submodules/EDGS/tools/train_paintmesh_pgsr.py','--initialization',initialization]
        for flag,key in [('lama','lama'),('camera','camera'),('edgs-config','edgs_config')]:
            train+=['--'+flag,inputs[key]['path']]
        train+=['--local-selection',local['path'],'--config',config,'--output-root',root/'edgs_joint',
                '--manifest',root/'manifests/edgs_joint_manifest.json']
        self.execute(case,'train_5000',train,REPO/'submodules/EDGS')
        self.publish_evaluate(case,old)

    def evaluate(self,case):
        self.execute(case,'evaluate',[PY,REPO/'scripts/paintmesh/eval_inpaint360.py','run','--model-path',Path(case['output'])/'inpainted_3dgs',
            '--source-path',Path('/home/member/wyc/data/Inpaint360GS/inpaint360')/case['scene'],'--resolution','4','--iteration','5000','--batch-size','4'],REPO)
        report(self.job)

    def upstream(self,case):
        source=next(x for x in self.req['scenes'] if x['scene']==case['scene'])
        removal=Path(case['removal_root']);root=Path(case['output']);root.mkdir(parents=True,exist_ok=True)
        cfg=copy.deepcopy(source['config']);cfg['iterations']=5000;cfg['init']['max_points']=None
        config=root/'config.json'
        if config.exists():assert read(config)==cfg
        else:write(config,cfg)
        raw=subprocess.check_output(['bash','-c','source "$1"; env -0','bash',str(REPO/'scripts/paintmesh/configs/inpaint360_recommended.env')],cwd=REPO,env=self.env)
        env=dict(v.decode().split('=',1) for v in raw.split(b'\0') if b'=' in v)
        env.update(self.env)
        old_request=read(Path(self.req['baseline_job'])/'request.json')
        scene_request=next(s for s in old_request['scenes'] if s['name']==case['scene'])
        env.update(REMOVAL_ROOT=str(removal),INPAINT_RUN_ROOT=str(root),RUN_NAME=removal.name,INPAINT_RUN_NAME=case['name'],
            EDGS_INPAINT_CONFIG=str(config),EDGS_INPAINT_ITERATIONS='5000',MASK_DILATION=str(case['mask_dilation']),
            DATA_ROOT='/home/member/wyc/data',OUTPUT_ROOT=str(REPO/'output'),LAUNCH_REFINER='false',
            VIRTUAL_ADAPTIVE_CONFIG=scene_request.get('adaptive_config',str(REPO/'scripts/paintmesh/configs/inpaint360_adaptive180.json')))
        env.pop('PYTHONPATH',None)
        import shutil
        env['CONDA_BIN']=shutil.which('conda') or '/home/member/miniconda3/bin/conda'
        ids=','.join(map(str,source['target_ids']))
        remove=['bash','scripts/paintmesh/run_remove.sh','Inpaint360GS/inpaint360',case['scene'],'4',ids,'none']
        if case['scene']=='doppelherz':
            old=read(Path(case['source_removal'])/'removed_3dgs/model_manifest.json')
            publish=[PY,'-m','tools.publish_removed_edgs_model','--removed-ply',removal/'work_model/point_cloud_object_removal/iteration_2000/point_cloud.ply',
                     '--iteration','2000','--target-ids',ids,'--surrounding-ids','none','--removal-threshold','0.7','--output',removal/'removed_3dgs']
            for flag,key in [('classifier','classifier'),('edgs-config','edgs_config'),('cfg-args','cfg_args'),('bridge-manifest','bridge_manifest'),('semantic-manifest','semantic_manifest')]:publish+=['--'+flag,old['inputs'][key]['path']]
            self.execute(case,'publish_removed',publish,REPO/'submodules/Inpaint360GS',env)
            if case.get('reuse_source_cameras'):
                if not (removal/'removed_mesh/semantic_manifest.json').exists():
                    raise ValueError('Corrected removal mesh must be complete before camera retry')
                self.execute(case,'reuse_camera_render',[PY,REPO/'scripts/paintmesh/prepare_quality_camera_retry.py',
                    '--plan',self.job/'experiments.json','--scene',case['scene']],REPO,env)
            else:
                self.execute(case,'remove_mesh_virtual',remove+['3'],REPO,dict(env,END_STAGE='4'))
            self.execute(case,'track',[PY,REPO/'scripts/paintmesh/track_holes_headless.py','--removal-root',removal],REPO,env)
        # A dedicated review receipt is written only after inspecting the pages.
        review=removal/'tracker/quality_review.json'
        if not review.exists() or not read(review).get('approved_for_training'):
            write(self.job/'status.json',dict(state='awaiting_mask_review',scene=case['scene'],experiment=case['name'],review=str(review),time=now()))
            while not review.exists() or not read(review).get('approved_for_training'):time.sleep(10)
        # Headless tracking writes masks in an active session. Native Stage 5
        # validates their archive, camera, shape and timestamps before sealing it.
        # With complete existing masks it never needs to open the browser.
        if case['scene']=='doppelherz':
            self.execute(case,'commit_tracking',remove+['5'],REPO,dict(env,END_STAGE='5',LAUNCH_REFINER='true'))
        inpaint=['bash','scripts/paintmesh/run_inpaint.sh','Inpaint360GS/inpaint360',case['scene'],'4',ids,'none','1']
        self.execute(case,'inpaint_5000',inpaint,REPO,dict(env,END_STAGE='6'))
        self.evaluate(case)

    def run(self,wait_pid=0):
        with (self.job/'retrain.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN);signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)
            if wait_pid:
                from repair_low_psnr_cases import process_identity
                identity=process_identity(wait_pid)
                while identity and process_identity(wait_pid)==identity:
                    write(self.job/'status.json',dict(state='waiting_for_diagnosis',wait_pid=wait_pid,time=now()))
                    time.sleep(10)
            failures={};plan=read(self.job/'experiments.json');write(self.job/'executed_plan.json',plan)
            for case in plan:
                try:
                    if case['kind']=='fixed_inputs':self.fixed_inputs(case)
                    elif case['kind']=='upstream':self.upstream(case)
                    elif case['kind']=='initialization_only':self.initialization_only(case)
                    else:raise ValueError('Unknown experiment kind: '+case['kind'])
                except InterruptedError:raise
                except Exception as exc:
                    traceback.print_exc();failures[case['scene']+'/'+case['name']]=str(exc);write(self.job/'failures.json',failures)
                report(self.job)
            write(self.job/'status.json',dict(state='complete' if not failures else 'completed_with_failures',failures=failures,time=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True,type=Path);p.add_argument('--wait-pid',type=int,default=0);a=p.parse_args()
    Runner(a.job).run(a.wait_pid)
