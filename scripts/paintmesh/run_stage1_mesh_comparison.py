#!/usr/bin/env python3
"""Add actual first-stage meshes, recompute the common cohort, and publish all figures."""
import fcntl
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,METHODS,STATES,read,write,identity
from mesh_benchmark.layout import work_root
from evaluate_native_mesh_rgb import available_ram

PUBLIC=BASE/'mesh_comparison_whole'
ROOT=work_root(PUBLIC)
WORK=ROOT/'stage1_expansion'


def snapshot():
    folder=WORK/'before';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    write(folder/'protocol.json',read(ROOT/'protocol.json'))
    for state in STATES:
        shutil.copy2(PUBLIC/'reports'/(state+'.md'),folder/(state+'_report.md'))
        for scene in SCENES:
            src=ROOT/'common'/scene/(state+'_support')
            shutil.copytree(src,folder/'common'/scene/(state+'_support'),dirs_exist_ok=True)
            for method in METHODS:
                if method=='paintmesh_stage1':continue
                path=ROOT/state/method/scene/'metrics.json'
                if path.exists():
                    dest=folder/'metrics'/state/method/scene;dest.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,dest/'metrics.json')
    shutil.copy2(PUBLIC/'figures_all_methods/metadata/selected_views.json',folder/'selected_views.json')
    write(folder/'complete.json',dict(complete=True,updated=time.time(),meaning='Five-method results before first-stage column'))


def status(stage,tasks):
    write(WORK/'status.json',dict(complete=stage=='complete',stage=stage,pid=os.getpid(),updated=time.time(),tasks=tasks))
    lines=['# 第一阶段mesh加入与模糊核查进度','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}；阶段：{stage}。','',
      '| 任务 | 状态 |','|---|---|']+[f'| {k} | {v["status"]} |' for k,v in tasks.items()]
    (PUBLIC/'docs/STAGE1_MESH_STATUS.md').write_text('\n'.join(lines)+'\n')


def cpu_stage(stage,scenes):
    tasks={s:dict(status='pending') for s in scenes};active={}
    for scene in scenes:
        if stage=='prepare' and all((ROOT/'cache/paintmesh_stage1'/scene/state/'complete.json').exists() for state in STATES):
            tasks[scene]['status']='complete'
        elif stage=='support' and all((ROOT/'common'/scene/(state+'_support')/'complete.json').exists() and
                read(ROOT/'common'/scene/(state+'_support')/'complete.json')['methods']==METHODS for state in STATES):
            tasks[scene]['status']='complete'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    while True:
        for scene,(proc,log) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();tasks[scene].update(status='complete' if code==0 else 'failed',exit_code=code);del active[scene]
            print('STAGE1_CPU_END',stage,scene,code,flush=True)
        while len(active)<2 and available_ram()>(32 if stage=='prepare' else 16):
            pending=[s for s,r in tasks.items() if r['status']=='pending']
            if not pending:break
            scene=pending[0]
            command=[sys.executable,'-u',str(HERE/'evaluate_recomposition_meshes.py'),'--output',str(PUBLIC),
                '--worker',stage,'--scene',scene]
            if stage=='prepare':command+=['--method','paintmesh_stage1']
            log=open(WORK/(scene+'_'+stage+'.log'),'w')
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            active[scene]=(proc,log);tasks[scene].update(status='running',pid=proc.pid)
            print('STAGE1_CPU_START',stage,scene,proc.pid,flush=True)
        status(stage,tasks)
        if not active and not any(r['status']=='pending' for r in tasks.values()):break
        time.sleep(10)
    if any(r['status']!='complete' for r in tasks.values()):raise RuntimeError('First-stage CPU task failed; see per-scene logs')


def run():
    WORK.mkdir(exist_ok=True,parents=True)
    lock=open(WORK/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot();cpu_stage('prepare',SCENES)
    protocol=read(ROOT/'protocol.json')
    protocol['methods']=METHODS;protocol['cohorts']={s:METHODS for s in SCENES}
    stage1=[]
    for scene in SCENES:
        selection=read(PUBLIC/'recomposition'/scene/'composition/scene.json')['input_selection']
        source=Path(selection['original_mesh'])
        stage1.append(dict(scene=scene,mesh=identity(source),model=str(source.parents[2]),object_present=True))
        link=PUBLIC/'meshes'/scene/'stage1.ply'
        if not link.exists():link.symlink_to(source)
    protocol['paintmesh_stage1']=dict(enabled=True,source='Actual original pre-removal mesh from input_selection.original_mesh',
        sources=stage1,geometry_unchanged=True,training_performed=False,renderer='Same PaintMesh CPU triangle ray caster',
        background_mode='Exclude target mask; original object remains in the source mesh',recomposition_mode='Original pre-removal scene')
    write(ROOT/'protocol.json',protocol);write(WORK/'stage1_sources.json',stage1)
    for scene in SCENES:
        for state in STATES:
            support=ROOT/'common'/scene/(state+'_support')
            if support.exists() and read(support/'complete.json')['methods']!=METHODS:shutil.rmtree(support)
    cpu_stage('support',SCENES)
    status('native RGB scoring and all comparison figures',{})
    with (WORK/'native_evaluation.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(HERE/'evaluate_native_mesh_rgb.py'),'--output',str(PUBLIC)],stdout=log,stderr=subprocess.STDOUT,check=True)
    status('blur analysis and publication verification',{})
    with (WORK/'blur_audit.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(HERE/'audit_paintmesh_blur.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    report=PUBLIC/'docs/PAINTMESH_COLOR_BLUR.md'
    text=report.read_text().replace('记录实际体素为相机半径/384，带宽为6个体素；',
        '记录多数场景实际体素为相机半径/384，cone_red与doppelherz进一步粗化为半径/192，带宽均为6个体素；')
    report.write_text(text)
    from verify_native_mesh_delivery import verify
    verify(PUBLIC)
    # Previous color-fix experiments remain valid on their dated five-method
    # cohort. Point readers to the expanded current comparison explicitly.
    for name in ['COLOR_EXPORT_REPAIRS.md','GOF_COLOR_REPAIR.md','COME_COLOR_REPAIR.md']:
        p=PUBLIC/'docs'/name;text=p.read_text()
        note='**当前主比较已加入第一阶段mesh并按六方法共同支持域重评估。本文颜色修复前后表保留当时的五方法支持域；最新主表和模糊核查见 [第一阶段与模糊报告](PAINTMESH_COLOR_BLUR.md)。**\n\n'
        if '当前主比较已加入第一阶段mesh' not in text:p.write_text(note+text)
    p=PUBLIC/'README.md';text=p.read_text();banner='**第一阶段mesh已加入全部场景对比，两个报告和八列总图已更新。** [几何/颜色模糊原因与证据](docs/PAINTMESH_COLOR_BLUR.md)。\n\n'
    if '第一阶段mesh已加入全部场景对比' not in text:p.write_text(banner+text)
    p=PUBLIC/'docs/README.md';text=p.read_text()
    if 'PAINTMESH_COLOR_BLUR.md' not in text:p.write_text(text+'\n- [PaintMesh几何/颜色模糊与第一阶段对比](PAINTMESH_COLOR_BLUR.md)\n')
    status('complete',{s:dict(status='complete') for s in SCENES});print('STAGE1_MESH_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':run()
