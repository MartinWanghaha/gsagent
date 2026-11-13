#!/usr/bin/env python3
"""Isolated whole-image TSDF vertex refinement; production inputs stay read-only.

The existing complete native mesh is reused exactly, including the original
near mesh and every overlap-filtered far partition. Both GPUs share a locked
work queue. Geometry/teacher implementations have separate receipts so a
restarted worker cannot silently mix different objectives.
"""
from pathlib import Path
import argparse
import fcntl
import gc
import os
import signal
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,read,write,identity,digest
from mesh_benchmark.layout import work_root

PUBLIC=BASE/'mesh_comparison_whole'
DEFAULT=PUBLIC/'experiments/bag_tsdf_refine_20261112'
CHILDREN=[]


def stopped(out):
    if (Path(out)/'STOP_REQUESTED.json').exists():
        raise KeyboardInterrupt('Explicit experiment stop request')


def state(out,label,**details):
    write(Path(out)/'status.json',dict(complete=label=='complete',stage=label,
        updated=time.time(),pid=os.getpid(),**details))
    (Path(out)/'STATUS.md').write_text('\n'.join([
        '# bag 原生 TSDF + PGSR 几何顶点微调','',
        time.strftime('%Y-%m-%d %H:%M:%S %Z')+'：'+label,'',str(details),'',
        '85 个完整训练视角，13 个完整评估视角；全部原三角面、颜色、原生分辨率。',
        '冻结 3DGS；分块边界固定，防止独立微调产生接缝。',
        '优化使用固定表面对应的点面深度项与实际三角形法线项；不是可微全场景光栅化。',
        '', '[方案](PLAN.md) · [对比图](figures/README.md) · [日志](campaign.log)',
    ])+'\n',encoding='utf-8')


def input_record(spec):
    path=Path(spec['path']);record=dict(spec=dict(spec),identity=identity(path))
    if path.name.endswith('.mesh.json'):
        meta=read(path)
        for name,stamp in [('vertices','vertex_identity'),('indices','index_identity')]:
            actual=identity(meta[name]['path'])
            if actual!=meta[stamp]:raise ValueError('Original native triangle buffer changed')
            record[name]=actual
    return record


def prepare(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    scene_path=PUBLIC/'recomposition/bag/composition/scene.json'
    scene=read(scene_path)
    if scene['input_selection']['group']!='fitlama':raise ValueError('Unexpected bag source group')
    specs=[dict(path=scene['background']['mesh']['path'])]
    specs += [dict(path=r['path']) for r in scene['background'].get('patches',[])]
    # The extension list contains the exact published overlap-filtered indices.
    extension=read(work_root(PUBLIC)/'fine_whole/extensions/background/bag/complete.json')
    if [Path(s['path']).resolve() for s in specs]!=[Path(s['path']).resolve() for s in extension['mesh_specs']]:
        raise ValueError('Published scene and complete extension disagree')
    protected=[scene_path,PUBLIC/'reports/background.md',PUBLIC/'reports/recomposition.md',
        Path(scene['background']['gaussians']['path'])]
    request=dict(scene='bag',source_scene=identity(scene_path),
        source_checkpoint=identity(scene['background']['gaussians']['path']),
        inputs=[input_record(s) for s in specs],train_views=85,held_out_views=13,
        native_resolution=True,image_downsampling=False,faces_removed=False,
        colors_changed=False,gaussian_training=False,
        teacher_loss='Fixed visible surface point-plane correspondences + actual triangle normals',
        differentiable_rasterization=False,partition_boundary_vertices_frozen=True,
        iterations=8,max_displacement=.008,normal_weight=.1)
    jobs=[dict(index=i,source=r) for i,r in enumerate(request['inputs'])]
    plan=dict(complete=True,request=request,request_id=digest(request),jobs=jobs,
        protected=[identity(p) for p in protected],created=time.time())
    marker=out/'plan.json'
    if marker.exists():
        old=read(marker)
        if old['request']!=request:raise ValueError('Existing experiment input/config differs')
        return old
    write(marker,plan)
    (out/'PLAN.md').write_text('''# bag 全图 TSDF 顶点微调实验

来源为 fitlama/bag 的最终 5000 步 3DGS。复用正式完整 TSDF 初值，包含原始近景 mesh 和全部去重远景分块，不重新融合、不降低体素/图像分辨率、不减面。

仅优化 mesh 顶点：全部 85 个训练视角的 PGSR plane_z 与单位相机法线建立固定的表面点面对应；后续以深度二次项和实际三角形法线约束微调顶点。它是几何对应求解，**不是可微全场景 raster 深度损失的实现**。原 GS、相机、三角面连接和颜色固定，分块边界顶点固定以防裂缝；保守位移/翻面约束保护原拓扑。顶点微调不能填补初始拓扑孔洞。

两张卡通过逐任务文件锁共同处理所有网格分块。原图尺寸、所有融合视角、全部三角面均保留。测试 RGB 不进入优化；13 个留出相机的 GS 几何只用于诊断。测试协议沿用现有目录，注意该协议不是对严格未见测试轨迹的额外证明。

A 为正式原完整 TSDF；B 为相同初值的顶点微调。纯背景排除目标 mask；组合模式加入相同 MV-SAM3D 物体和位姿。使用真实 mesh 原生渲染，输出全部 13 视角的前后和四方法 RGB/几何对比；在共同有效像素上计算 PSNR/SSIM/LPIPS，同时报告各自 coverage。几何 CD/Precision/Recall/F1 使用现有已观测稀疏 COLMAP 参考和固定距离阈值，不能称扫描真值。

正式资产与报告只读，所有实验产物放在本目录。只有实际数据完成后才填写结果；性能/质量是否改善均待验证。
''',encoding='utf-8')
    return plan


def load_input(record):
    if input_record(record['spec'])!=record:raise ValueError('Mesh input changed')
    path=Path(record['spec']['path'])
    if path.name.endswith('.mesh.json'):
        meta=read(path);packed=np.load(meta['vertices']['path'],mmap_mode='r')
        vertices=np.asarray(packed['position'],dtype='float32',order='C')
        colors=np.asarray(packed['color'],dtype='uint8',order='C')
        faces=np.asarray(np.load(meta['indices']['path'],mmap_mode='r'),dtype='int64',order='C')
        return vertices,faces,colors,Path(meta['indices']['path'])
    from recompose.io import ply,xyz
    data=ply(path);v=data['vertex'].data
    # Preserve the PLY's stored coordinate precision. The native renderer
    # performs its own Float32 conversion identically for both A and B.
    vertices=xyz(v)
    faces=np.asarray(data['face'].data['vertex_indices'],dtype='int64',order='C')
    colors=np.full((len(vertices),4),255,dtype='uint8')
    colors[:,:3]=np.column_stack([v[n] for n in ['red','green','blue']])
    return vertices,faces,colors,None


def commit_geometry(folder,vertices,faces,colors,source_indices,original_spec,receipt):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if receipt['moved_vertices']==0:return dict(original_spec)
    if source_indices is None and vertices.dtype==np.float64:
        from plyfile import PlyData,PlyElement
        # Store actual double positions; do not silently quantize the original
        # PLY. Omit obsolete pre-refinement shading normals. Native evaluation
        # uses triangle cross-product normals, not those shading attributes.
        data=np.empty(len(vertices),dtype=[('x','<f8'),('y','<f8'),('z','<f8'),
            ('red','u1'),('green','u1'),('blue','u1')])
        for i,name in enumerate(['x','y','z']):data[name]=vertices[:,i]
        for i,name in enumerate(['red','green','blue']):data[name]=colors[:,i]
        triangle=np.empty(len(faces),dtype=[('vertex_indices','<u4',(3,))])
        triangle['vertex_indices']=faces
        path=folder/'refined.ply';temporary=folder/'refined.tmp.ply'
        PlyData([PlyElement.describe(data,'vertex'),PlyElement.describe(triangle,'face')],
            text=False,byte_order='<',comments=['Frozen PGSR mesh vertex refinement; original double coordinate precision']).write(str(temporary))
        temporary.replace(path)
        return dict(path=str(path))
    packed=np.empty(len(vertices),dtype=[('position','<f4',(3,)),('color','u1',(4,))])
    packed['position']=vertices;packed['color']=colors
    vp=folder/'vertices.npy';tmp=folder/'vertices.tmp.npy';np.save(tmp,packed);tmp.replace(vp)
    if source_indices is None:
        source_indices=folder/'indices.npy';temp=folder/'indices.tmp.npy'
        np.save(temp,np.asarray(faces,dtype='<u4'));temp.replace(source_indices)
    meta=dict(format='paintmesh-native-triangles-v1',vertices=dict(path=str(vp)),
        indices=dict(path=str(source_indices)),vertex_identity=identity(vp),index_identity=identity(source_indices),
        vertex_count=len(vertices),face_count=len(faces),bounds=[vertices.min(0).tolist(),vertices.max(0).tolist()],
        vertex_stride=16,vertex_color='Unchanged original native RGB8, opaque RGBA',
        geometry_simplified=False,faces_unchanged=True)
    path=folder/'refined.mesh.json';write(path,meta);return dict(path=str(path))


def worker(out,rank,single_job=None):
    import torch
    from mesh_benchmark.tsdf_vertex_refine import prepare_teacher,refine_mesh
    out=Path(out).resolve();plan=read(out/'plan.json');evidence=read(out/'frames/complete.json')
    torch.set_num_threads(4)
    free,total=torch.cuda.mem_get_info()
    if free<12*1024**3:raise RuntimeError('Refinement needs 12 GiB free VRAM before start')
    implementation=identity(HERE/'mesh_benchmark/tsdf_vertex_refine.py')
    frames=[]
    for row in evidence['frames']:
        if identity(row['path'])!=row['identity']:raise ValueError('Teacher input changed')
        with np.load(row['path']) as a:
            frames.append({k:a[k].copy() for k in ['depth','normal','alpha','confidence','rgb','K','w2c']})
    teacher=prepare_teacher(frames,device='cuda:0');del frames;gc.collect()
    for job in plan['jobs']:
        index=job['index']
        if single_job is not None and index!=single_job:continue
        stopped(out);folder=out/'refined'/f'{index:04d}';folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            request=dict(plan=plan['request_id'],teacher=evidence['request_id'],source=job['source'],implementation=implementation)
            marker=folder/'complete.json'
            if marker.exists():
                old=read(marker)
                if old['request']!=request or input_record(old['mesh_spec'])!=old['result_identity']:
                    raise ValueError('Refinement output/request changed')
                continue
            started=time.time();write(out/'workers'/f'{rank}.json',dict(stage='refining',job=index,pid=os.getpid(),updated=started))
            vertices,faces,colors,source_indices=load_input(job['source'])
            original=np.array(vertices,copy=True)
            print('TSDF_REFINE_JOB_START',rank,index,'vertices',len(vertices),'faces',len(faces),flush=True)
            parameters=plan['request']
            refined,detail=refine_mesh(vertices,faces,teacher,iterations=parameters['iterations'],
                max_displacement=parameters['max_displacement'],normal_weight=parameters['normal_weight'])
            if refined.shape!=vertices.shape or not np.isfinite(refined).all():raise ValueError('Invalid refined vertex array')
            if not np.array_equal(vertices,original):raise ValueError('Implementation mutated original vertex positions')
            actual_moved=int(np.any(refined!=original,axis=1).sum())
            detail['moved_vertices']=actual_moved
            spec=commit_geometry(folder,refined,faces,colors,source_indices,job['source']['spec'],detail)
            result=dict(complete=True,index=index,worker=rank,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                request=request,mesh_spec=spec,result_identity=input_record(spec),vertices=len(vertices),faces=len(faces),
                detail=detail,elapsed=time.time()-started,finished=time.time(),
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
            write(marker,result);print('TSDF_REFINE_JOB_COMPLETE',rank,index,'moved',actual_moved,'seconds',result['elapsed'],flush=True)
            del vertices,faces,colors,original,refined;gc.collect();torch.cuda.empty_cache()
    write(out/'workers'/f'{rank}.json',dict(stage='complete',pid=os.getpid(),updated=time.time()))


def progress(out):
    rows=[read(p) for p in (Path(out)/'refined').glob('*/complete.json')]
    return dict(completed=len(rows),total=len(read(Path(out)/'plan.json')['jobs']),
        vertices=sum(r['vertices'] for r in rows),faces=sum(r['faces'] for r in rows),
        moved_vertices=sum(r['detail']['moved_vertices'] for r in rows),
        total_worker_seconds=sum(r['elapsed'] for r in rows))


def finalize(out):
    out=Path(out);plan=read(out/'plan.json');rows=[]
    for job in plan['jobs']:
        row=read(out/'refined'/f'{job["index"]:04d}'/'complete.json')
        if row['request']['plan']!=plan['request_id'] or input_record(row['mesh_spec'])!=row['result_identity']:
            raise ValueError('Incomplete or changed refined mesh')
        rows.append(row)
    document=dict(complete=True,scene='bag',source_scene=plan['request']['source_scene'],
        A=dict(complete=True,mesh_specs=[r['source']['spec'] for r in plan['jobs']]),
        B=dict(complete=True,mesh_specs=[r['mesh_spec'] for r in rows]),
        vertices=sum(r['vertices'] for r in rows),faces=sum(r['faces'] for r in rows),
        topology_unchanged=True,colors_unchanged=True,geometry_simplified=False,
        boundary_vertices_frozen=True,differentiable_rasterization=False,
        statistics=progress(out))
    write(out/'mesh_manifest.json',document)
    return document


def launch(out,label,command,gpu=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=(Path(out)/(label+'.log')).open('a')
    p=subprocess.Popen(command,env=env,cwd=HERE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    CHILDREN.append((p,log));write(Path(out)/'launches'/(label+'.json'),dict(pid=p.pid,gpu=gpu,command=command,started=time.time()))
    return p


def wait_all(out,processes,label):
    while any(p.poll() is None for p in processes):
        stopped(out)
        failed=[(p.pid,p.returncode) for p in processes if p.poll() not in [None,0]]
        if failed:raise RuntimeError(f'{label} worker failed: {failed}')
        state(out,label,progress=progress(out) if (Path(out)/'refined').exists() else {})
        time.sleep(15)
    if any(p.returncode for p in processes):raise RuntimeError(label+' failed')


def run(out,workers_per_gpu=3):
    out=Path(out).resolve();prepare(out)
    if workers_per_gpu<1 or workers_per_gpu>3:
        raise ValueError('Use between one and three bounded-memory workers per GPU')
    with (out/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while not (out/'frames/complete.json').exists():
                stopped(out);state(out,'waiting for exact PGSR depth/normal export');time.sleep(15)
            while not (out/'checks/tsdf_vertex_refine.json').exists():
                stopped(out);state(out,'waiting for geometric correctness checks');time.sleep(15)
            check=read(out/'checks/tsdf_vertex_refine.json')
            if not check.get('complete') or check['implementation']!=identity(HERE/'mesh_benchmark/tsdf_vertex_refine.py'):
                raise ValueError('Missing/current implementation validation')
            write(out/'execution.json',dict(driver=identity(Path(__file__)),
                workers_per_gpu=workers_per_gpu,physical_gpus=[0,1],
                geometry_objective_unchanged=True,started=time.time()))
            processes=[]
            for gpu in [0,1]:
                for slot in range(workers_per_gpu):
                    name=f'{gpu}_{slot}'
                    processes.append(launch(out,'refine_gpu_'+name,[sys.executable,'-u',str(Path(__file__).resolve()),
                        '--output',str(out),'--worker',name],gpu))
            wait_all(out,processes,'dual GPU full-scene mesh vertex refinement')
            finalize(out)
            state(out,'rendering and scoring full-image mesh comparisons')
            evaluation=launch(out,'evaluation',[sys.executable,'-u',str(HERE/'evaluate_tsdf_refine_experiment.py'),
                '--output',str(out),'--device','cuda:0'],0)
            wait_all(out,[evaluation],'rendering and scoring full-image mesh comparisons')
            protection=[]
            for original in read(out/'plan.json')['protected']:
                actual=identity(original['path'])
                changed=actual!=original
                # An independent, already-running all-scene publication can
                # legitimately update shared reports while this pilot runs.
                # Source checkpoint/scene changes invalidate the experiment;
                # report changes are recorded, never overwritten or rolled back.
                report=Path(original['path']).parent==PUBLIC/'reports'
                if changed and not report:raise ValueError('Protected geometry source changed')
                protection.append(dict(before=original,after=actual,changed=changed,
                    experiment_wrote_file=False,concurrent_publication_possible=report))
            write(out/'production_inputs_check.json',dict(complete=True,checks=protection))
            state(out,'complete',progress=progress(out),figures='figures/README.md',reports='reports/')
        except BaseException as error:
            state(out,'stopped' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),progress=progress(out) if (out/'refined').exists() else {})
            for p,_ in CHILDREN:
                if p.poll() is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            raise
        finally:
            for _,log in CHILDREN:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--single-job',type=int)
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--workers-per-gpu',type=int,default=3)
    args=parser.parse_args()
    if args.prepare:prepare(args.output)
    elif args.worker is not None:worker(args.output,args.worker,args.single_job)
    elif args.finalize:finalize(args.output)
    else:run(args.output,args.workers_per_gpu)
