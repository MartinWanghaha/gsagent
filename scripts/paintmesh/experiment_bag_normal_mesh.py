#!/usr/bin/env python3
"""Whole-scene normal-guided mesh experiment with immutable production inputs."""
from pathlib import Path
import argparse
import fcntl
import gc
import os
import shutil
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,read,write,identity,digest
from mesh_benchmark.layout import work_root,configure_sources
from mesh_benchmark.fine_tsdf import VOXEL,TRUNCATION,BLOCK,partitions,retain_owned_faces
from mesh_benchmark.normal_surface_allocation import allocate
from mesh_benchmark.full_surface_fusion import save_part

PUBLIC=BASE/'mesh_comparison_whole';ROOT=work_root(PUBLIC)
DEFAULT=PUBLIC/'experiments/bag_normal_mesh_20261112'


def host_available():
    return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))


def disk_available(out):return shutil.disk_usage(out).free


def update(out,stage,**details):
    record=dict(complete=stage=='complete',updated=time.time(),pid=os.getpid(),stage=stage,**details)
    write(out/'status.json',record)
    lines=['# bag 全场景深度与法线 mesh 实验','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}；阶段：{stage}。','',
        '固定最终 inpaint 5000 步检查点，85 个训练视角及13个留出视角；0.002体素、0.008带宽、原生1/4图像。完整范围提取和真实三角网格整图渲染；不拼接旧网格或截图。','',
        'A：普通完整TSDF；N：深度与法线融合；R：N加实际顶点几何精修。生成物体和位姿不改变。正式结果和项目训练流程不修改。','',
        '阶段详情：`'+str(details)+'`。','', '[实验方案](PLAN.md) · [完整状态](status.json) · [运行日志](campaign.log)']
    (out/'STATUS.md').write_text('\n'.join(lines)+'\n')


def wait_worker_resources(out,active_blocks=0):
    import torch
    needed=max(12*1024**3,active_blocks*BLOCK**3*100+5*1024**3)
    while True:
        free,total=torch.cuda.mem_get_info()
        if free>=needed and host_available()>=24*1024**3 and disk_available(out)>=150*1024**3:return
        write(out/'extraction_wait.json',dict(updated=time.time(),free_vram=int(free),needed_vram=int(needed),
            host_free=host_available(),disk_free=disk_available(out)))
        print('NORMAL_EXPERIMENT_WAIT_RESOURCES',free,needed,flush=True);time.sleep(30)


def extract(out):
    import torch,open3d as o3d
    from mesh_benchmark.normal_surface_fusion import integrate_normal_partition
    from mesh_benchmark.normal_mesh_refinement import refine_vertices_anchored
    out=Path(out);torch.set_num_threads(4)
    lock=open(out/'extraction.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(out/'plan.json');evidence=read(out/'frames/complete.json')
    coords=np.load(out/'allocation/blocks.npy')
    allocation=read(out/'allocation/complete.json')
    request=dict(plan=digest(plan),evidence=identity(out/'frames/complete.json'),allocation=allocation['request_id'],
        normal_implementation=identity(HERE/'mesh_benchmark/normal_surface_fusion.py'),
        refinement_implementation=identity(HERE/'mesh_benchmark/normal_mesh_refinement.py'),
        voxel=VOXEL,sdf_trunc=TRUNCATION,full_scene=True,roi=None,
        refinement=dict(iterations=3,max_step=.001,max_displacement=.008,teacher='All85TrainingViews',ground_truth_used=False))
    frames=[]
    for row in evidence['frames']:
        if row.get('identity') and identity(row['path'])!=row['identity']:raise ValueError('Normal evidence changed')
        with np.load(row['path']) as data:frames.append({k:data[k].copy() for k in data.files})
    specs={k:[] for k in ['N','R']};faces={k:0 for k in specs};vertices={k:0 for k in specs}
    for key in specs:(out/key/'parts').mkdir(parents=True,exist_ok=True)
    for i,(core,active) in enumerate(partitions(coords)):
        sign=digest(dict(request=request,core=core.tolist(),active=active.tolist()))
        records={k:out/k/'parts'/f'{i:05d}.json' for k in specs}
        if all(p.exists() for p in records.values()):
            for key,p in records.items():
                row=read(p)
                if row['signature']!=sign:raise ValueError('Partition request changed')
                if row['spec']:
                    if identity(row['spec']['path'])!=row['spec_identity']:raise ValueError('Saved triangle partition changed')
                    specs[key].append(row['spec'])
                faces[key]+=row['faces'];vertices[key]+=row['vertices']
            continue
        wait_worker_resources(out,len(active))
        runtime=dict(stage='normal fusion and vertex refinement',partition=i,total_blocks=len(coords),core_blocks=len(core),
            active_blocks=len(active),faces=faces.copy(),pid=os.getpid(),updated=time.time())
        write(out/'extraction_runtime.json',runtime)
        print('NORMAL_PART_START',i,len(core),len(active),flush=True)
        v,f,c=integrate_normal_partition(active,frames,device='CUDA:0',chunk_blocks=64)
        o3d.core.cuda.release_cache();torch.cuda.empty_cache()
        # Ownership is based on the UNREFINED triangle. Reassigning moved
        # centroids could create seams/duplicates between partitions.
        owned=retain_owned_faces(v,f,core)
        moved,refinement=refine_vertices_anchored(v,frames,band=.008,iterations=3,device='CUDA:0',
            max_step=.001,max_displacement=.008,faces=f)
        if moved.shape!=v.shape or not np.isfinite(moved).all():raise ValueError('Refinement returned invalid vertices')
        if len(v) and np.max(np.linalg.norm(moved-v,axis=1))>.00801:raise ValueError('Refinement exceeded displacement bound')
        count=int(owned.sum())
        for key,position in [('N',v),('R',moved)]:
            spec=None;nv=nf=0
            if count:spec,nv,nf=save_part(out/key/'parts',i,position,f[owned],c)
            if spec:specs[key].append(spec)
            faces[key]+=nf;vertices[key]+=nv
            row=dict(complete=True,signature=sign,spec=spec,spec_identity=identity(spec['path']) if spec else None,
                vertices=nv,faces=nf,core_blocks=len(core),halo_blocks=len(active),
                ownership='Original, unrefined centroid block',refinement=refinement if key=='R' else None)
            write(records[key],row)
        if faces['N']!=faces['R']:raise ValueError('N and R must retain identical topology')
        print('NORMAL_PART_COMPLETE',i,'owned_faces',count,'totals',faces,flush=True)
        del v,f,c,moved;gc.collect();o3d.core.cuda.release_cache();torch.cuda.empty_cache()
    for key in specs:
        write(out/key/'complete.json',dict(complete=True,request=request,method=key,mesh_specs=specs[key],
            faces=faces[key],vertices=vertices[key],full_scene=True,roi=None,all_input_views=len(frames),
            geometry_simplified=False,images_downsampled=False,colors_optimized=False,
            topology_preserved_between_N_and_R=True,source_kind=plan['source_kind']))
    print('NORMAL_FULL_EXTRACTION_COMPLETE',faces,flush=True)


def available_gpu(minimum=16*1024):
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    choices=[tuple(map(int,line.split(','))) for line in result.stdout.splitlines()]
    choices=[(gpu,free) for gpu,free in choices if free>=minimum]
    return max(choices,key=lambda p:p[1])[0] if choices else None


def wait_checks(out,names):
    for name in names:
        receipt=out/'checks'/(name+'.json')
        while not receipt.exists():update(out,'waiting for independent geometry checks',check=name);time.sleep(30)
        record=read(receipt)
        if not record.get('complete') or record.get('implementation')!=identity(HERE/record['module']):
            raise ValueError(f'Experiment checks are missing or stale: {name}')


def export_assets(out):
    import copy
    from recompose.mesh_export import export_mesh_scene
    from recompose.io import fingerprint
    graph=read(PUBLIC/'recomposition/bag/composition/scene.json')
    if graph['objects']!=read(out/'plan.json')['fixed_generated_objects']:
        raise ValueError('Generated object geometry or placement changed during the experiment')
    for method in ['A','N','R']:
        refs=[fingerprint(s['path']) for s in read(out/method/'complete.json')['mesh_specs']]
        scene=copy.deepcopy(graph);scene['background'].update(mesh=refs[0],patches=refs[1:])
        export_mesh_scene(scene,out/method/'composed_world')
        scene['objects']=[];export_mesh_scene(scene,out/method/'background_world')


def spawn(out,script,args,logname):
    gpu=None
    while gpu is None or host_available()<32*1024**3 or disk_available(out)<150*1024**3:
        gpu=available_gpu()
        if gpu is not None and host_available()>=32*1024**3 and disk_available(out)>=150*1024**3:break
        update(out,'waiting for resources',next_script=script);time.sleep(30)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    log=open(out/logname,'a')
    p=subprocess.Popen([sys.executable,'-u',str(HERE/script),'--output',str(out),*args],env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    print('NORMAL_EXPERIMENT_WORKER_START',script,gpu,p.pid,flush=True)
    return p,log,gpu


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);configure_sources(ROOT)
    lock=open(out/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_folder=ROOT/'fine_whole/background/bag';source=read(source_folder/'frames/complete.json')
    baseline=read(source_folder/'complete.json');graph=read(PUBLIC/'recomposition/bag/composition/scene.json')
    files=[Path(__file__),HERE/'mesh_benchmark/normal_surface_allocation.py',HERE/'mesh_benchmark/normal_surface_fusion.py',
        HERE/'mesh_benchmark/normal_mesh_refinement.py',
        HERE/'export_normal_fusion_frames.py',HERE/'evaluate_normal_mesh_experiment.py']
    plan=dict(scene='bag',source_kind='background',source_checkpoint=source['request']['checkpoint'],
        source_config=source['request']['config'],source_frame_receipt=identity(source_folder/'frames/complete.json'),
        input_frames=source['frames'],implementation=[identity(p) for p in files],
        voxel=VOXEL,sdf_trunc=TRUNCATION,full_scene=True,roi=None,
        methods=dict(A='Unchanged complete RGB-D TSDF',N='Depth and learned-normal surface fusion',R='N plus actual vertex refinement'),
        refinement=dict(iterations=3,max_step=.001,max_displacement=.008),
        fixed_generated_objects=graph['objects'],production_pipeline_modified=False,
        input_views=85,evaluation_views=13,images_downsampled=False,mesh_decimation=False,
        near_mesh_splicing=False,texture_fitting=False)
    if (out/'plan.json').exists() and read(out/'plan.json')!=plan:raise ValueError('Normal experiment sources changed')
    write(out/'plan.json',plan)
    snapshot=out/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for file in files:
        target=snapshot/file.name
        if not target.exists():shutil.copy2(file,target)
    (out/'PLAN.md').write_text('# bag 最终 inpaint 3DGS 的全场景深度与法线导出实验\n\n'
        '使用用户选定 fitlama 的5000步最终检查点。3DGS、相机、已有生成物体与位姿均不改变，主项目流程和正式结果保持原样。\n\n'
        '| 组别 | 几何导出 |\n|---|---|\n'
        '| A | 相同检查点、相同85帧已经完成的完整细TSDF，复用全部原生分区重新渲染 |\n'
        '| N | 全场景深度与学习法线约束的有向表面融合；分配完整像素表面覆盖 |\n'
        '| R | N后实际顶点的鲁棒点到面与局部切向约束精修；三轮同步更新，每步≤0.001，总位移≤0.008 |\n\n'
        '- 统一0.002体素、0.008带宽，全部85个训练视角，原生1/4图像。所有13个留出视角渲染整张RGB/深度/几何法线。\n'
        '- 学习法线先单位化、保持相机坐标约定；用alpha和深度法线一致性软置信度，不把法线幅值再当独立置信度。输入深度、颜色与A缓存逐字节保持一致。\n'
        '- 连续表面激活使用源像素相邻三角形完整覆盖，长三角形只为体素分配做保守细分，不修改输出网格密度，不省略像素或视角。\n'
        '- R使用固定初始邻居的局部切向约束，解析每点总位移安全界限制翻面和塌缩，避免分区全局回退导致共享顶点不一致；按N未移动三角形唯一归属裁分区，N/R保留相同三角拓扑与颜色。禁止全局平滑、减面、缩图、旧网格或截图拼接。\n'
        '- 背景报告排除物体mask；组合报告将相同MV-SAM3D对象以相同姿态和原生mesh颜色组合。两组均报告自身覆盖率和共同有效域PSNR/SSIM/LPIPS、COLMAP观测点CD/F1。\n'
        '- 额外渲染冻结GS在13留出相机的几何，只用来测mesh与teacher的深度/法线一致性；不读取留出RGB来拟合mesh。\n'
        '- 本次不做纹理拟合，保留原生融合颜色；N的观测权重也会影响融合颜色，RGB差异不能全部归因于几何。R与N颜色和拓扑保持一致，单独检查实际坐标精修。\n'
        '- 判断同时依据CD/F1、对冻结GS的深度/法线一致性、覆盖和图像指标，不以面更少或更平滑宣告成功。对双面零厚度薄片，当前局部符号融合仍有相反观测抵消的限制，不保证无观测区域也完整闭合。\n'
        '- 独立后台调度：不停止其他任务，空闲显存不足16GiB/主机内存不足32GiB/磁盘不足150GiB则等待，保持质量参数不变。\n')
    write(out/'A/complete.json',dict(complete=True,method='A',source_kind='background',full_scene=True,
        mesh_specs=baseline['mesh_specs'],faces=baseline['faces'],
        source=identity(source_folder/'complete.json'),reused_identical_complete_extraction=True))
    exporter=None
    if not (out/'frames/complete.json').exists():
        update(out,'rendering learned normal evidence')
        exporter=spawn(out,'export_normal_fusion_frames.py',['--source-kind','background'],'frames.log')
    update(out,'complete-scene surface allocation',normal_evidence_worker=exporter[0].pid if exporter else None)
    allocate(out/'allocation',source['frames'],source_folder/'blocks.npy',
        status_callback=lambda row:update(out,'complete-scene surface allocation',progress=row,
            normal_evidence_worker=exporter[0].pid if exporter else None))
    if exporter:
        p,log,gpu=exporter
        while p.poll() is None:update(out,'waiting for learned normal evidence',worker_pid=p.pid,gpu=gpu);time.sleep(30)
        log.close()
        if p.returncode:raise RuntimeError('Normal evidence failed; see frames.log')
    if not (out/'frames/complete.json').exists():raise ValueError('Complete normal evidence is missing')
    wait_checks(out,['normal_surface_fusion','normal_mesh_refinement'])
    if not all((out/key/'complete.json').exists() for key in ['N','R']):
        p,log,gpu=spawn(out,Path(__file__).name,['--extract'],'extraction.log')
        while p.poll() is None:update(out,'complete-scene normal fusion and vertex refinement',worker_pid=p.pid,gpu=gpu);time.sleep(30)
        log.close()
        if p.returncode:raise RuntimeError('Normal mesh extraction failed; exact partitions are resumable, see extraction.log')
    if not (out/'teacher_views/complete.json').exists():
        p,log,gpu=spawn(out,'export_normal_fusion_frames.py',['--source-kind','background','--teacher-views-only'],'teacher_views.log')
        while p.poll() is None:update(out,'rendering frozen GS teacher geometry',worker_pid=p.pid,gpu=gpu);time.sleep(30)
        log.close()
        if p.returncode:raise RuntimeError('Teacher geometry export failed')
    update(out,'complete-scene native mesh rendering and evaluation')
    wait_checks(out,['normal_mesh_evaluation'])
    export_assets(out)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with (out/'evaluation.log').open('a') as log:
        subprocess.run([sys.executable,'-u',str(HERE/'evaluate_normal_mesh_experiment.py'),'--output',str(out)],env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    update(out,'complete')
    print('BAG_FULL_NORMAL_MESH_EXPERIMENT_COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--extract',action='store_true');args=parser.parse_args()
    if args.extract:extract(args.output)
    else:run(args.output)
