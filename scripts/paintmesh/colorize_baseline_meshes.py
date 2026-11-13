#!/usr/bin/env python3
"""Recover native Wrapping colors and evaluate official CoMe colors on fixed meshes."""
import argparse
import ast
import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,read,write,identity,digest
from mesh_benchmark.color_export import read_mesh,positions,exact_color_indices,attach_colors
from recompose.io import fingerprint

RESEARCH=Path('/home/member/wyc/mesh_gs_research')
ROOT=BASE/'mesh_comparison_whole'
RUNTIME=ROOT/'_internal/workspace'


def atomic_link(target,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name('.'+path.name+'.new')
    if temporary.is_symlink() or temporary.exists():temporary.unlink()
    temporary.symlink_to(os.path.relpath(target,path.parent));os.replace(temporary,path)


def publish(method,scene,source,colored,detail):
    out=BASE/method/scene
    mesh=read_mesh(colored);vertices=mesh['vertex'].data
    total=np.zeros(3);squares=np.zeros(3);white=0
    for start in range(0,len(vertices),262144):
        rgb=np.column_stack([vertices[n][start:start+262144] for n in ['red','green','blue']]).astype('float64')
        total+=rgb.sum(0);squares+=(rgb*rgb).sum(0);white+=int((rgb==255).all(1).sum())
    mean=total/len(vertices);std=np.sqrt(np.maximum(0,squares/len(vertices)-mean*mean))
    result=dict(complete=True,method=method,scene=scene,source_geometry=fingerprint(source),
                colored_mesh=fingerprint(colored),geometry_unchanged=True,training_performed=False,
                color_statistics=dict(mean_rgb=mean.tolist(),std_rgb=std.tolist(),exact_white_fraction=white/len(vertices)),
                updated=time.time(),**detail)
    write(out/'mesh_color_complete.json',result)
    atomic_link(colored,out/'mesh_colored.ply');atomic_link(colored,out/'mesh.ply')
    atomic_link(colored,ROOT/'meshes'/scene/(method+'_colored.ply'))
    marker=read(out/'mesh_complete.json');marker.update(path=str(colored),sha256=result['colored_mesh']['sha256'],
              geometry_path=str(source),geometry_sha256=result['source_geometry']['sha256'],
              colored_path=str(colored),has_vertex_colors=True,
              color_receipt=str(out/'mesh_color_complete.json'),geometry_unchanged_by_colorization=True)
    write(out/'mesh_complete.json',marker)
    path=out/'result.json'
    if path.exists():
        item=read(path);item.update(mesh=str(colored),geometry_mesh=str(source),colored_mesh=str(colored));write(path,item)
    return result


def recover_wrapping(scene):
    out=BASE/'gaussian_wrapping'/scene;marker=read(out/'mesh_complete.json');source=Path(marker.get('geometry_path',marker['path']))
    original=source.with_name(source.name.replace('_post.ply','.ply'))
    raw=read_mesh(original);post=read_mesh(source)
    colors=np.column_stack([raw['vertex'].data[n] for n in ['red','green','blue']])
    indices=exact_color_indices(positions(raw['vertex'].data),positions(post['vertex'].data),colors)
    colored=source.with_name(source.stem+'_colored.ply')
    verified=attach_colors(source,colored,colors[indices])
    return publish('gaussian_wrapping',scene,source,colored,dict(
        color_method='Exact unchanged vertex correspondence to official pre-postprocess colored mesh',
        original_colored_mesh=fingerprint(original),interpolated_colors=False,**verified))


def come(scene,chunk,probe_only=False,query_file=None,inspect_colors=None,normalize_colors=False):
    scripts=RESEARCH/'scripts/inpaint360_baselines';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('baseline_worker_setup',scripts/'worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.setup('come','mesh')
    import torch
    torch.cuda.set_per_process_memory_fraction((10*1024**3)/torch.cuda.get_device_properties(0).total_memory)
    from arguments import ModelParams,PipelineParams
    from scene import Scene,GaussianModel
    from diff_gaussian_rasterization import ExtendedSettings
    import extract_mesh_tets as official_meshing
    evaluate_alpha=official_meshing.evaluate_alpha
    official_meshing.tqdm=lambda iterable,**kwargs:iterable
    out=BASE/'come'/scene;model=out/'model';marker=read(out/'mesh_complete.json');source=Path(marker.get('geometry_path',marker['path']))
    split=read(BASE/'_data'/scene/'split.json')
    parser=argparse.ArgumentParser();mp=ModelParams(parser);pp=PipelineParams(parser)
    args=parser.parse_args(['-s',str(BASE/'_data'/scene),'-m',str(model),'-r','4','--eval','--data_device','cpu'])
    saved=ast.parse((model/'cfg_args').read_text(),mode='eval').body
    for kw in saved.keywords:
        if hasattr(args,kw.arg):setattr(args,kw.arg,ast.literal_eval(kw.value))
    args.init_type='sfm'
    # Native color integration reads camera geometry and the trained Gaussians,
    # never original_image. Keep the native Camera class and exact measured
    # dimensions, avoiding decoding every full-resolution training JPEG.
    from utils import camera_utils
    from scene.cameras import Camera
    sizes={Path(row['name']).stem:(row['width'],row['height']) for row in split['images']}
    native_load_camera=camera_utils.loadCam
    camera_checks=[]
    def projection_only_camera(arguments,index,info,resolution_scale):
        if resolution_scale!=1.0 or arguments.resolution!=4:raise ValueError('Unexpected camera resolution')
        width,height=sizes[info.image_name]
        camera=Camera(colmap_id=info.uid,R=info.R,T=info.T,FoVx=info.FovX,FoVy=info.FovY,
             image=torch.zeros((3,height,width)),gt_alpha_mask=None,image_name=info.image_name,
             uid=index,resolution=(width,height),data_device=arguments.data_device)
        if index==0:
            reference=native_load_camera(arguments,index,info,resolution_scale)
            for name in ['world_view_transform','full_proj_transform','camera_center']:
                if not torch.equal(getattr(camera,name),getattr(reference,name)):
                    raise ValueError(f'Projection-only camera differs from upstream: {name}')
            if (camera.image_width,camera.image_height)!=(reference.image_width,reference.image_height):
                raise ValueError('Projection-only camera has different raster dimensions')
            camera_checks.append(dict(view=info.image_name,projection_and_view_matrices_exact=True,raster_dimensions_exact=True))
        return camera
    camera_utils.loadCam=projection_only_camera
    with torch.no_grad():
        gaussian=GaussianModel(args.sh_degree)
        scene_model=Scene(mp.extract(args),gaussian,load_iteration=30000,shuffle=False,skip_test=True)
        cameras=scene_model.getTrainCameras()
        assert sorted(c.image_name for c in cameras)==sorted(Path(n).stem for n in split['train'])
        assert not any(c.image_name.lower().startswith('test') for c in cameras)
        # Upstream launches identifyPixelRanges with zero CUDA blocks when no
        # query survives its frustum/alpha culling (rasterizer_impl.cu:959).
        # Point queries are independent: one disposable in-frustum query makes
        # that launch nonempty without changing any real query or Gaussian.
        native_integrate=official_meshing.integrate
        dummy={id(c):(torch.tensor([0.,0.,1.,1.],device='cuda')@torch.linalg.inv(c.world_view_transform))[:3][None] for c in cameras}
        def integrate_nonempty(points,alpha,view,*args,**kwargs):
            padded_alpha=torch.cat([alpha,alpha.new_ones(1)])
            result=native_integrate(torch.cat([points,dummy[id(view)]]),padded_alpha,view,*args,**kwargs)
            alpha.copy_(padded_alpha[:-1]);result['color_integrated']=result['color_integrated'][:-1]
            return result
        official_meshing.integrate=integrate_nonempty
        gaussian.compute_3D_filter(cameras=cameras.copy())
        pipe=pp.extract(args);pipe.convert_SBs_python=gaussian.use_SBs
        settings=ExtendedSettings.from_json(str(model/'config.json'))
        settings.meshing_settings.return_color=True
        if inspect_colors:
            from audit_come_mesh_colors import inspect_native
            return inspect_native(scene,inspect_colors,cameras,gaussian,pipe,settings,official_meshing,model)
        if normalize_colors:
            from repair_come_mesh_colors import normalize_native
            return normalize_native(scene,source,chunk,cameras,gaussian,pipe,settings,official_meshing,model,camera_checks)
        mesh=read_mesh(source);vertices=mesh['vertex'].data
        if query_file:
            # Read-only field audit: compare native fast meshing alpha with the
            # exact per-view alpha union; never alter geometry or color caches.
            from diff_gaussian_rasterization import GlobalSortOrder
            groups={};fresh_colors={};background=torch.zeros(3,device='cuda')
            data=np.load(query_file)
            for key in data.files:
                points=torch.from_numpy(np.asarray(data[key],dtype='float32')).cuda()
                fast_settings=ExtendedSettings.from_json(str(model/'config.json'))
                fast_settings.meshing_settings.return_color=False
                fast_settings.meshing_settings.alpha_early_stop=True
                fast_settings.sort_settings.sort_order=GlobalSortOrder.MIN_Z_BOUNDING
                fast,_=evaluate_alpha(points,cameras,gaussian,pipe,background,.003,splat_args=fast_settings)
                exact_settings=ExtendedSettings.from_json(str(model/'config.json'))
                exact_settings.meshing_settings.return_color=True
                exact,exact_colors=evaluate_alpha(points,cameras,gaussian,pipe,background,.003,splat_args=exact_settings)
                fresh_colors[key]=np.rint(np.clip(exact_colors.cpu().numpy(),0,1)*255).astype('uint8')
                a=fast.cpu().numpy();b=exact.cpu().numpy()
                groups[key]=dict(points=len(points),fast_quantiles=np.quantile(a,[0,.1,.5,.9,1]).tolist(),
                    exact_quantiles=np.quantile(b,[0,.1,.5,.9,1]).tolist(),
                    median_fast_isovalue_error=float(np.median(np.abs(a-.5))),
                    median_exact_isovalue_error=float(np.median(np.abs(b-.5))),
                    fraction_exact_alpha_above_09=float(np.mean(b>.9)),
                    maximum_fast_exact_difference=float(np.max(np.abs(a-b))))
                print('COME_SURFACE_FIELD_AUDIT',scene,key,groups[key],flush=True)
            result=dict(scene=scene,groups=groups,gaussians_unchanged=True,mesh_unchanged=True)
            write(Path(query_file).with_suffix('.result.json'),result)
            np.savez(Path(query_file).with_suffix('.colors.npz'),**fresh_colors)
            return result
        cache=out/'colorization';cache.mkdir(exist_ok=True)
        request=dict(geometry=identity(source),checkpoint=identity(model/'point_cloud/iteration_30000/point_cloud.ply'),
            split=identity(BASE/'_data'/scene/'split.json'),config=identity(model/'config.json'),
            official_evaluate_alpha=fingerprint(RESEARCH/'baselines/CoMe/extract_mesh_tets.py'),
            kernel_size=.003,active_sh_degree=gaussian.active_sh_degree,
            compatibility='Native camera geometry without unused RGB decoding; independent disposable query avoids zero-block CUDA launches')
        signature=digest(request);progress=cache/'progress.json';color_path=cache/'vertex_colors.npy'
        old=read(progress) if progress.exists() else None
        if old and old['signature']!=signature:
            previous={k:v for k,v in old['request'].items() if k!='compatibility'}
            current={k:v for k,v in request.items() if k!='compatibility'}
            if previous!=current:raise ValueError('Colorization inputs changed')
        start=old['completed_vertices'] if old else 0
        colors=np.lib.format.open_memmap(color_path,mode='r+' if old else 'w+',dtype='uint8',shape=(len(vertices),3))
        background=torch.zeros(3,device='cuda');history=old.get('chunks',[]) if old else []
        if not old:
            began=time.time();probe=torch.from_numpy(positions(vertices[:8192]).astype('float32')).cuda()
            _,sample=evaluate_alpha(probe,cameras,gaussian,pipe,background,.003,splat_args=settings)
            if not torch.isfinite(sample).all():raise ValueError('Native CoMe color probe is nonfinite')
            consistency=None
            if scene=='cone_red':
                halves=[]
                for part in probe.split(4096):
                    _,value=evaluate_alpha(part,cameras,gaussian,pipe,background,.003,splat_args=settings);halves.append(value)
                consistency=float((sample-torch.cat(halves)).abs().max())
                if consistency>1e-5:raise ValueError(f'Point chunking changed official colors: {consistency}')
                del halves,value
            write(cache/'native_probe.json',dict(vertices=len(probe),train_views=len(cameras),seconds=time.time()-began,
                peak_mib=torch.cuda.max_memory_allocated()/1024**2,finite=True,chunk_consistency_max_error=consistency,
                physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
            print('NATIVE_COLOR_PROBE',scene,read(cache/'native_probe.json'),flush=True)
            del probe,sample
        if probe_only:return read(cache/'native_probe.json')
        while start<len(vertices):
            stop=min(start+chunk,len(vertices));began=time.time()
            points=torch.from_numpy(positions(vertices[start:stop]).astype('float32')).cuda()
            try:
                _,rgb=evaluate_alpha(points,cameras,gaussian,pipe,background,.003,splat_args=settings)
                values=rgb.detach().cpu().numpy()
                if values.shape!=(stop-start,3) or not np.isfinite(values).all():raise ValueError('Invalid native vertex colors')
                colors[start:stop]=np.rint(np.clip(values,0,1)*255).astype('uint8');colors.flush()
            except torch.cuda.OutOfMemoryError:
                del points;torch.cuda.empty_cache()
                if chunk<=16384:raise
                chunk=max(16384,chunk//2);print('COLOR_CHUNK_RETRY',scene,chunk,flush=True);continue
            del points,rgb
            history.append(dict(start=start,stop=stop,elapsed=time.time()-began,peak_mib=torch.cuda.max_memory_allocated()/1024**2))
            start=stop
            write(progress,dict(signature=signature,request=request,completed_vertices=start,total_vertices=len(vertices),chunks=history))
            print('NATIVE_COLORS',scene,start,'/',len(vertices),'chunk seconds',history[-1]['elapsed'],flush=True)
        colored=source.with_name(source.stem+'_colored.ply')
        verified=attach_colors(source,colored,colors)
        return publish('come',scene,source,colored,dict(color_method='Official CoMe evaluate_alpha with return_color=True on the existing mesh vertices',
            official_color_source=request['official_evaluate_alpha'],request=request,train_views=[c.image_name for c in cameras],
            held_out_rgb_used=False,camera_checks=camera_checks,vertex_chunk_size=chunk,
            cuda_peak_mib=torch.cuda.max_memory_allocated()/1024**2,**verified))


def update(status):
    status.update(updated=time.time(),pid=os.getpid());write(RUNTIME/'mesh_colorization.json',status)
    lines=['# 原生彩色 mesh 输出','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}。','',
        'CoMe 使用官方顶点颜色积分，不重新提取几何；Wrapping 恢复后处理遗漏的原始顶点色。所有顶点坐标、三角形及其余原属性逐项核对，几何不变。',
        '', '| 场景 | CoMe | Gaussian Wrapping |','|---|---|---|']
    for scene in SCENES:
        cells=[]
        for method in ['come','gaussian_wrapping']:
            row=status['jobs'].get(method+'/'+scene,{})
            cells.append(f'[彩色 PLY](../meshes/{scene}/{method}_colored.ply)' if row.get('status')=='complete' else row.get('status','pending'))
        lines.append('| '+' | '.join([scene,*cells])+' |')
    lines+=['','各方法原输出目录的 `mesh_colored.ply` 和 `mesh.ply` 指向彩色结果，`mesh_color_complete.json` 记录颜色来源与几何保持不变的核验。原始几何文件保留用于核对，当前 CD/F1 无需因颜色改变而重算。']
    (ROOT/'docs/COLORED_MESHES.md').write_text('\n'.join(lines)+'\n')


def run():
    lock=open(RUNTIME/'mesh_colorization.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(started=time.time(),jobs={});running={};handles=[]
    for method in ['come','gaussian_wrapping']:
        for scene in SCENES:
            marker=BASE/method/scene/'mesh_color_complete.json'
            status['jobs'][method+'/'+scene]=dict(status='complete' if marker.exists() and read(marker).get('complete') else 'pending')
    for scene in SCENES:
        key='gaussian_wrapping/'+scene
        if status['jobs'][key]['status']=='complete':continue
        print('RESTORE_WRAPPING_COLORS',scene,flush=True)
        try:recover_wrapping(scene);status['jobs'][key]=dict(status='complete')
        except Exception as error:
            import traceback;traceback.print_exc();status['jobs'][key]=dict(status='failed',error=str(error))
        update(status)
    sys.path.insert(0,str(RESEARCH/'scripts/inpaint360_baselines'))
    from common import environment
    while True:
        for gpu,job in list(running.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();key=job['key'];marker=BASE/'come'/key.split('/')[1]/'mesh_color_complete.json'
            status['jobs'][key].update(status='complete' if code==0 and marker.exists() else 'failed',exit_code=code)
            print('COLOR_JOB_FINISHED',key,'exit',code,flush=True);del running[gpu]
        query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        for line in query.stdout.splitlines():
            gpu,free=map(int,line.split(','))
            if gpu in running or free<18000:continue
            pending=[k for k,v in status['jobs'].items() if k.startswith('come/') and v['status']=='pending']
            if not pending:break
            key=pending[0];scene=key.split('/')[1]
            log=open(RUNTIME/'logs'/f'colorize_come__{scene}.log','a')
            command=[str(RESEARCH/'envs/come_probe/bin/python'),'-u',str(Path(__file__).resolve()),'--worker','--scene',scene]
            p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=environment(gpu))
            running[gpu]=dict(key=key,process=p,log=log);status['jobs'][key]=dict(status='running',gpu=gpu,pid=p.pid)
            print('COLOR_JOB_START',key,'gpu',gpu,'pid',p.pid,flush=True)
        update(status)
        if not running and not any(v['status']=='pending' for v in status['jobs'].values()):break
        time.sleep(15)
    status['complete']=all(v['status']=='complete' for v in status['jobs'].values());update(status)
    print('COLORED_MESH_CAMPAIGN_FINISHED',status['complete'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--scene',choices=SCENES);p.add_argument('--chunk',type=int,default=4194304)
    p.add_argument('--probe-only',action='store_true')
    p.add_argument('--query-points',type=Path)
    p.add_argument('--inspect-colors',type=Path)
    p.add_argument('--normalize-colors',action='store_true')
    p.add_argument('--resume-after-pid',type=int)
    a=p.parse_args()
    if a.resume_after_pid:
        while True:
            try:cmd=Path(f'/proc/{a.resume_after_pid}/cmdline').read_bytes()
            except FileNotFoundError:break
            if b'colorize_baseline_meshes.py' not in cmd:break
            time.sleep(15)
    if a.worker:come(a.scene,a.chunk,a.probe_only,a.query_points,a.inspect_colors,a.normalize_colors)
    else:run()
