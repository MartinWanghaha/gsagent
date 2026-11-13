#!/usr/bin/env python3
"""Resumeable native CoMe/GOF support repair, geometry cache and two-GPU render."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,METHODS,STATES,read,write,identity
from mesh_benchmark.layout import work_root,public_root,configure_sources
from repair_opacity_meshes import WORK,POLICY,publish_result
from recompose.io import fingerprint

PUBLIC=BASE/'mesh_comparison_whole';ROOT=work_root(PUBLIC)
NATIVE_PYTHON=Path('/home/member/wyc/mesh_gs_research/envs/meshgs_wrapping/bin/python')


def ram():
    return next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))


def gpu_free():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return [tuple(map(int,row.split(','))) for row in text.splitlines()]


def status(tasks,stage):
    write(WORK/'status.json',dict(pid=os.getpid(),updated=time.time(),stage=stage,tasks=tasks,policy=POLICY))
    lines=['# CoMe / GOF 原生网格修复','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}；阶段：{stage}。','',
        '固定源高斯有效支撑规则，保留原坐标、颜色和所有原始文件；仅剔除无支撑的长插值面。原生网格后端双卡渲染。没有使用参考点云、测试图片或指标选择删面。','',
        '| 任务 | 状态 | GPU |','|---|---|---|']
    lines += [f'| {key} | {row["status"]} | {row.get("gpu","-")} |' for key,row in tasks.items()]
    (PUBLIC/'docs/OPACITY_MESH_REPAIR_STATUS.md').write_text('\n'.join(lines)+'\n')


def invalidate_if_needed(method,scene):
    marker=read(BASE/method/scene/'mesh_complete.json');wanted=identity(marker.get('geometry_path',marker['path']))
    complete=ROOT/'cache'/method/scene/'recomposition/complete.json'
    valid=complete.exists() and read(complete)['request']['specs']==[dict(identity=wanted,transform=None)]
    if valid:return True
    folder=ROOT/'cache'/method/scene
    if folder.exists():shutil.rmtree(folder)
    for state in STATES:
        for suffix in ['_support','_native_support']:
            folder=ROOT/'common'/scene/(state+suffix)
            if folder.exists():shutil.rmtree(folder)
        for m in METHODS:
            for path in [ROOT/state/m/scene/'metrics.json',ROOT/'cache'/m/scene/state/'geometry_metrics.json']:
                if path.exists():path.unlink()
    return False


def render_ready(method,scene):
    p=ROOT/'native_renderers'/method/scene/'complete.json'
    if not p.exists():return False
    meta=read(p)
    return meta.get('complete') and not meta.get('pilot') and meta['request']['mesh']==fingerprint(read(BASE/method/scene/'mesh_complete.json')['path']) and meta['request']['adapter']==fingerprint(HERE/'render_native_colored_meshes.py')


def run():
    configure_sources(ROOT);WORK.mkdir(parents=True,exist_ok=True)
    own=open(WORK/'campaign.lock','a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks=[]
    for name in ['campaign.lock','native_rgb_campaign.lock']:
        lock=open(ROOT/name,'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    protocol=read(ROOT/'protocol.json');protocol['native_support_validation']=dict(enabled=True,methods=['come','gof'],policy=POLICY,
        original_files_preserved=True,geometry='Original vertex values and retained faces; unsupported long interpolation faces removed',
        baseline_label='Native extraction + source Gaussian support validation; not unmodified vanilla extraction')
    write(ROOT/'protocol.json',protocol)
    sys.path.insert(0,str(BASE/'_scripts'));from common import environment
    order=[('come','truck'),('come','cube'),('gof','cube')]
    order += [(m,s) for s in SCENES for m in ['come','gof'] if (m,s) not in order]
    tasks={f'{m}/{s}/{stage}':dict(status='pending') for m,s in order for stage in ['filter','prepare','render']}
    running={};failed=[]
    while True:
        for key,(process,log,gpu) in list(running.items()):
            code=process.poll()
            if code is None:continue
            log.close();tasks[key].update(status='complete' if code==0 else 'failed',exit_code=code);del running[key]
            print('SUPPORT_REPAIR_FINISHED',key,'exit',code,flush=True)
            method,scene,stage=key.split('/')
            if code:
                failed.append(key)
                for dependent in ['prepare','render']:
                    if tasks[f'{method}/{scene}/{dependent}']['status']=='pending':tasks[f'{method}/{scene}/{dependent}']['status']='blocked'
                continue
            if stage=='filter':
                result=read(WORK/method/scene/'complete.json');publish_result(result)
                if invalidate_if_needed(method,scene):tasks[f'{method}/{scene}/prepare']['status']='complete'
                if render_ready(method,scene):tasks[f'{method}/{scene}/render']['status']='complete'
        cpu_count=sum(key.endswith(('/filter','/prepare')) for key in running)
        occupied={value[2] for value in running.values() if value[2] is not None}
        gpu_choices=[gpu for gpu,free in gpu_free() if gpu not in occupied and free>=12000]
        for method,scene in order:
            # Finish a published scene's cache promptly while the other CPU
            # slot continues validating the remaining scenes.
            for stage in ['prepare','filter','render']:
                key=f'{method}/{scene}/{stage}';row=tasks[key]
                if row['status']!='pending':continue
                if stage!='filter' and tasks[f'{method}/{scene}/filter']['status']!='complete':continue
                if ram()<35:continue
                gpu=None
                if stage=='render':
                    if not gpu_choices:continue
                    gpu=gpu_choices.pop(0)
                    cmd=[str(NATIVE_PYTHON),str(HERE/'render_native_colored_meshes.py'),'--output',str(PUBLIC),'--method',method,'--scene',scene]
                    env=environment(gpu)
                else:
                    if cpu_count>=2:continue
                    if stage=='filter':cmd=[sys.executable,str(HERE/'repair_opacity_meshes.py'),'--method',method,'--scene',scene]
                    else:cmd=[sys.executable,str(HERE/'evaluate_recomposition_meshes.py'),'--output',str(PUBLIC),'--worker','prepare','--method',method,'--scene',scene]
                    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';cpu_count+=1
                env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',TORCH_CUDA_ARCH_LIST='8.6')
                log=(ROOT/'logs'/('support_repair__'+key.replace('/','__')+'.log')).open('w')
                p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                row.update(status='running',pid=p.pid,gpu=gpu);running[key]=(p,log,gpu)
                print('SUPPORT_REPAIR_START',key,'gpu',gpu,'pid',p.pid,flush=True)
        status(tasks,'native support validation / cache / render')
        if not running and not any(v['status']=='pending' for v in tasks.values()):break
        time.sleep(10)
    if failed:raise RuntimeError('Support repair failed: '+', '.join(failed))
    from mesh_benchmark.overlap import prepare_support
    for scene in SCENES:
        prepare_support(ROOT,scene,METHODS,protocol['config'])
    # CoMe truck's old cap warning is superseded by source-support repair;
    # source scene pose/generation review flags remain untouched.
    flags=PUBLIC/'docs/MESH_QUALITY_FLAGS.json'
    if flags.exists():
        values=read(flags);values.pop('come/truck',None);write(flags,values)
    status(tasks,'geometry complete; native RGB scoring')
    for lock in locks:lock.close()
    subprocess.run([sys.executable,str(HERE/'evaluate_native_mesh_rgb.py'),'--output',str(PUBLIC)],check=True)
    status(tasks,'complete')
    print('NATIVE_SUPPORT_REPAIR_AND_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':run()
