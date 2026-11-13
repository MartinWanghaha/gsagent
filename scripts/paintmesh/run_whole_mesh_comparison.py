#!/usr/bin/env python3
"""Disconnect-safe whole-background extraction -> two reports -> all figures."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
DEFAULT_OUTPUT=REPO/'output/3dgs_with_mesh/mesh_comparison_whole'

def live(pid):
    try:return bool(Path(f'/proc/{pid}/cmdline').read_bytes())
    except FileNotFoundError:return False

def resources():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if result.returncode:return []
    return sorted([(int(a),int(b)) for a,b in [line.split(',') for line in result.stdout.splitlines()]],key=lambda x:-x[1])

def status(root,stage,jobs,**extra):
    from mesh_benchmark.core import write
    data=dict(pid=os.getpid(),stage=stage,updated=time.time(),scenes=jobs,**extra)
    write(root/'whole_status.json',data)
    lines=['# 全背景 mesh 任务进度','',f'阶段：{stage}。','', '| 场景 | 状态 | 阶段 | 说明 |','|---|---|---|---|']
    for scene,row in jobs.items():lines.append('| '+' | '.join([scene,row.get('status','pending'),row.get('stage','-'),row.get('message','')])+' |')
    (root/'WHOLE_STATUS.md').write_text('\n'.join(lines)+'\n')

def supervise(root,command,logname,jobs,stage,env=None):
    with open(root/'logs'/logname,'a') as log:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        while process.poll() is None:
            status(root,stage,jobs,worker_pid=process.pid);time.sleep(10)
    if process.returncode:raise RuntimeError(f'{stage} exited {process.returncode}; see logs/{logname}')

def plan(root):
    from mesh_benchmark.core import SCENES,read,write
    from mesh_benchmark.whole import extraction_parameters
    file=root/'whole_scene_plan.json'
    if file.exists():return read(file)
    parameters={}
    for scene in SCENES:
        pilot=root/'pilot_parameters.json'
        parameters[scene]=read(pilot) if scene=='bag' and pilot.exists() else extraction_parameters(scene)
    result=dict(created=time.time(),method='PaintMesh inpaint Gaussian checkpoint -> scene-adaptive whole TSDF',
       source_training='Existing checkpoints unchanged; train-only camera fusion',scenes=parameters,
       extraction=dict(depth='max(2*camera_radius,1.2*99.5th percentile of reliable training-track depths)',
                       voxel='max(0.002,camera_radius/384)',sdf_trunc='6*voxel',alpha_min=.5,
                       components='Keep every component with >=50 faces; no largest-component-only filtering'),
       evaluation='Existing conditional surface/pixel v2 protocol; F1 0.5/1/2% radius, primary 1%',
       gaussians_retrained=False,mvsam_geometry_and_poses='Reuse existing generated assets and calibrated poses')
    write(file,result)
    docs=['# 全场景背景 mesh 方案','',
       '输出包含两个模式：我们完整背景 mesh；完整背景 + 现有 MV-SAM3D 物体 mesh。源模型、原实验目录保持独立。','',
       '现有高斯不再训练，使用各场景既有5000步 inpaint 检查点。只使用共同训练相机融合，不将留出图像引入本轮网格提取。','',
       '融合距离由可靠训练 SfM 观测深度的99.5%分位×1.2和相机半径确定。体素为半径/384（下限0.002），截断带宽6体素；alpha≥0.5。保留所有至少50个面的连通块。','',
       '这是覆盖有限可观测场景的自适应 TSDF，不是无限背景或空间收缩无界网格。仍可能存在缺乏可靠几何的区域，实际覆盖率会保存到 coverage_validation.json。','',
       'bag 首轮半径分辨率768触发内存余量保护；本轮统一采用384，保留原始深度缓存，调整依据资源需求而不是测试指标。','',
       '| 场景 | 训练视角 | 深度上限（COLMAP单位） | 体素大小 |','|---|---:|---:|---:|']
    for scene,p in parameters.items():docs.append(f'| {scene} | {p["train_views"]} | {p["max_depth"]:.6f} | {p["voxel_size"]:.6f} |')
    docs += ['','## 评估与结果','',
       '- `meshes/<scene>/background.ply`：我们的完整背景网格。',
       '- `meshes/<scene>/world/mesh_scene.gltf`：背景模式。',
       '- `recomposition/<scene>/composition/world/mesh_scene.gltf`：加入现有生成物体的组合模式；物体位姿不变。',
       '- `background/report.md` / `recomposition/report.md`：两份指标报告。',
       '- `figures_all_methods/`：总图、各场景详细图、PDF；`subplots/`按方法/模式/场景整理子图。',
       '- 对照方法仅复用经过相机、mask、mesh 身份检查的不可变深度缓存及既有外观渲染；共同覆盖区域因我们的完整背景扩大，需要重新计算所有方法分数。',
       '- 我们的两种模式重新训练同规格的固定几何颜色场，仅用于mesh渲染评估；不更新高斯或mesh几何。',
       '- 几何参照依然是稀疏COLMAP点云，F1/CD为有效可观测表面范围内的一致性指标，非独立扫描精度。',
       '- 既有 PaintMesh 高斯和生成证据可能用过当前留出相机，仍为已有结果审计。','',
       '任务链：逐场景提取 → 验证覆盖 → 发布两种模式 → 两份评估 → 全方法图片 → 独立子图。']
    (root/'WHOLE_SCENE_PLAN.zh-CN.md').write_text('\n'.join(docs)+'\n')
    (root/'README.md').write_text('# 完整背景 mesh 对比\n\n[执行方案](WHOLE_SCENE_PLAN.zh-CN.md) · [任务进度](WHOLE_STATUS.md)\n\n完成后自动生成 background/report.md、recomposition/report.md 和 figures_all_methods/README.md。\n')
    return result

def reduce_fusion_memory(root,scene):
    """Fail without mutating quality settings; never trade fidelity for memory."""
    raise RuntimeError(f'{scene}: automatic TSDF coarsening is disabled. '
                       'Keep voxel size, truncation, images and cameras unchanged; '
                       'wait for resources or use a spatially tiled extraction.')

def run(root):
    from mesh_benchmark.core import SCENES,read,write
    from mesh_benchmark.whole import available_ram
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    lock=open(root/'whole_campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'whole_campaign.pid').write_text(str(os.getpid())+'\n')
    configuration=plan(root);jobs={s:dict(status='pending',stage='extraction') for s in SCENES}
    for scene in SCENES:
        marker=root/'models'/scene/'whole_mesh_complete.json'
        pilot=root/'pilot.pid'
        if scene=='bag' and pilot.exists() and not marker.exists():
            pid=int(pilot.read_text())
            while live(pid):
                jobs[scene]=dict(status='running',stage='existing pilot',pid=pid)
                status(root,'whole background extraction',jobs);time.sleep(10)
        try:
            if not marker.exists():
                jobs[scene]=dict(status='waiting',stage='resource check')
                while True:
                    candidates=[g for g,free in resources() if free>=12288]
                    cached=(root/'models'/scene/'geometry_frames/complete.json').exists()
                    if available_ram()>=40 and (candidates or cached):break
                    status(root,'waiting for resources',jobs);time.sleep(10)
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='' if cached else str(candidates[0])
                jobs[scene]=dict(status='running',stage='whole background extraction',gpu=None if cached else candidates[0])
                supervise(root,[sys.executable,str(Path(__file__).resolve()),'--output',str(root),'--worker','extract','--scene',scene],
                          scene+'_extraction.log',jobs,'extract '+scene,env)
            validation=root/'meshes'/scene/'coverage_validation.json'
            published=root/'recomposition'/scene/'composition/scene.json'
            if not validation.exists() or not published.exists():
                jobs[scene]=dict(status='running',stage='coverage validation / publication')
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
                supervise(root,[sys.executable,str(Path(__file__).resolve()),'--output',str(root),'--worker','finalize','--scene',scene],
                          scene+'_validation.log',jobs,'validate and publish '+scene,env)
            check=read(validation)
            jobs[scene]=dict(status='complete' if check['passed'] else 'needs_review',stage='whole mesh ready',
                message=f'background coverage {check["whole_background_coverage"]:.1%}; previous {check["previous_background_coverage"]:.1%}')
            status(root,'whole background extraction',jobs)
            if scene=='bag' and not check['passed']:
                raise RuntimeError('Bag whole-background coverage did not improve enough; inspect pilot before proceeding')
        except Exception as error:
            jobs[scene]=dict(status='failed',stage='extraction/validation',message=str(error))
            status(root,'scene failure',jobs)
            if scene=='bag':raise
    ready=[s for s in SCENES if (root/'recomposition'/s/'composition/scene.json').exists()]
    if not ready:raise RuntimeError('No whole-scene meshes available for evaluation')
    status(root,'preparing verified baseline reuse',jobs)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    supervise(root,[sys.executable,str(Path(__file__).resolve()),'--output',str(root),'--worker','reuse'],
              'baseline_reuse.log',jobs,'baseline reuse',env)
    eval_command=[sys.executable,str(HERE/'evaluate_recomposition_meshes.py'),'--output',str(root),'--scenes',','.join(ready),'--refresh-completed-meshes']
    supervise(root,eval_command,'evaluation.log',jobs,'two mesh metric reports',env)
    supervise(root,[sys.executable,str(HERE/'plot_mesh_method_comparison.py'),'--root',str(root),'--scenes',','.join(ready)],
              'figures.log',jobs,'all scenes and methods figures',env)
    supervise(root,[sys.executable,str(HERE/'organize_mesh_subplots.py'),'--figures',str(root/'figures_all_methods')],
              'subplots.log',jobs,'individual method subplots',env)
    status(root,'complete' if len(ready)==len(SCENES) else 'partial; inspect failed scenes',jobs)
    print('WHOLE_COMPARISON_FINISHED',len(ready),'scenes',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--worker',choices=['extract','finalize','reuse']);p.add_argument('--scene')
    from mesh_benchmark.layout import work_root
    args=p.parse_args();root=work_root(args.output)
    os.environ['PAINTMESH_RECOMPOSE_ROOT']=str(root/'recomposition')
    os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
    sys.path.insert(0,str(HERE))
    if args.worker:
        from mesh_benchmark.whole import extract,validate_whole,publish_whole,reuse_baseline_caches
        from mesh_benchmark.core import read
        if args.worker=='extract':extract(root,args.scene,read(root/'whole_scene_plan.json')['scenes'][args.scene])
        elif args.worker=='finalize':
            validation=validate_whole(root,args.scene);publish_whole(root,args.scene)
            print('WHOLE_VALIDATION',args.scene,validation,flush=True)
        else:reuse_baseline_caches(root)
    else:run(root)

if __name__=='__main__':main()
