#!/usr/bin/env python3
"""Two-GPU native triangle RGB evaluation after color-only mesh publication."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,METHODS,STATES,read,write,digest,identity,cameras
from mesh_benchmark.layout import work_root,public_root,configure_sources
from recompose.io import fingerprint

RESEARCH=Path('/home/member/wyc/mesh_gs_research')
NATIVE_PYTHON=RESEARCH/'envs/meshgs_wrapping/bin/python'


def verified_file_fingerprint(root,path):
    """Reuse a full SHA256 only while inode, size, mtime AND ctime stay exact.

    A single validation/publication should not reread the same multi-GB PLY
    repeatedly. Changed bytes, replaced files and restored mtimes invalidate
    the cache. This changes no image, triangle, metric or evaluation view.
    """
    path=Path(path).resolve(strict=True)
    folder=Path(root)/'verified_file_hashes';folder.mkdir(exist_ok=True,parents=True)
    key=digest(str(path));cache=folder/(key+'.json')
    def stamp():
        s=path.stat()
        return dict(path=str(path),device=s.st_dev,inode=s.st_ino,size_bytes=s.st_size,
                    mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
    with (folder/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=stamp()
        # Linux may report equal ctimes for multiple writes within one clock
        # tick. Never cache a hash captured inside that fresh-write window.
        # Stable, days-old model artifacts can still reuse their full digest.
        stat_aged=time.time_ns()-max(before['mtime_ns'],before['ctime_ns'])>=1_000_000_000
        if cache.exists():
            previous=read(cache)
            previous_aged=previous.get('stat_aged',previous.get('verified_at',0)-max(previous['stat']['mtime_ns'],previous['stat']['ctime_ns'])/1e9>=1.)
            if stat_aged and previous_aged and previous['stat']==before:return previous['fingerprint']
        result=fingerprint(path)
        if before!=stamp():raise ValueError(f'Source file changed while computing SHA256: {path}')
        write(cache,dict(stat=before,fingerprint=result,full_sha256_verified=True,stat_aged=stat_aged,verified_at=time.time()))
        return result


def publish_render_link(source,destination):
    """Relative links must be based on the physical publication directory."""
    source=Path(source).resolve(strict=True);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name('.'+destination.name+'.native')
    if temporary.is_symlink() or temporary.exists():temporary.unlink()
    temporary.symlink_to(os.path.relpath(source,destination.parent.resolve()))
    os.replace(temporary,destination)
    if destination.resolve(strict=True)!=source:raise ValueError('Published render link differs from source')


def status(root,stage,tasks):
    record=dict(pid=os.getpid(),updated=time.time(),stage=stage,tasks=tasks)
    write(root/'native_rgb_status.json',record)
    lines=['# 原生彩色 mesh 渲染评估','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}；阶段：{stage}。','',
        '真实三角网格和原生顶点色。CoMe / Wrapping 使用各自 NVDiffRast MeshRenderer；2DGS / GOF 使用其 TNT 网格脚本的 pyrender 后端，增加 RGB 输出。PaintMesh 使用已有的真实组合网格三角形渲染。无统一颜色场拟合，无高斯图片替代。','',
        '| 任务 | 状态 | GPU |','|---|---|---|']
    for key,row in tasks.items():lines.append(f'| {key} | {row["status"]} | {row.get("gpu","-")} |')
    (public_root(root)/'docs/NATIVE_RGB_STATUS.md').write_text('\n'.join(lines)+'\n')


def gpus():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return [tuple(map(int,line.split(','))) for line in p.stdout.splitlines()]


def available_ram():
    return next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))


def schedule(root,tasks,stage):
    sys.path.insert(0,str(RESEARCH/'scripts/inpaint360_baselines'))
    from common import environment
    running={}
    while True:
        for gpu,(key,p,log) in list(running.items()):
            code=p.poll()
            if code is None:continue
            log.close();tasks[key].update(status='complete' if code==0 else 'failed',exit_code=code);del running[gpu]
            print('NATIVE_TASK_END',key,'exit',code,flush=True)
        for gpu,free in gpus():
            # Native color meshes and VGG scoring have bounded working sets;
            # retain ample host RAM without requiring 35 GiB to be idle.
            minimum_host_ram=16 if stage=='score' else 24
            if gpu in running or free<12000 or available_ram()<minimum_host_ram:continue
            pending=[key for key,row in tasks.items() if row['status']=='pending']
            if not pending:break
            key=pending[0];method,scene,*state=key.split('/')
            if stage=='render':
                command=[str(NATIVE_PYTHON),str(HERE/'render_native_colored_meshes.py'),'--output',str(root),'--method',method,'--scene',scene]
                env=environment(gpu)
            else:
                command=[sys.executable,str(Path(__file__).resolve()),'--output',str(root),'--score',state[0],'--method',method,'--scene',scene]
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',TORCH_CUDA_ARCH_LIST='8.6')
            log=open(root/'logs'/('native_'+stage+'__'+key.replace('/','__')+'.log'),'w')
            p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            tasks[key].update(status='running',gpu=gpu,pid=p.pid);running[gpu]=(key,p,log)
            print('NATIVE_TASK_START',stage,key,'gpu',gpu,'pid',p.pid,flush=True)
        status(root,stage,tasks)
        if not running and not any(row['status']=='pending' for row in tasks.values()):break
        time.sleep(10)
    if any(row['status']!='complete' for row in tasks.values()):raise RuntimeError(f'Native mesh {stage} tasks failed; see logs/status')


def rendering(root,method,scene,state):
    if method in ['paintmesh','paintmesh_stage1']:
        cache=root/'cache'/method/scene/state;meta=read(cache/'complete.json')
        return dict(rgb=cache/'native_renders',depth=cache/'depth',
             request=dict(backend='PaintMesh CPU triangle ray casting with native vertex colors',geometry=meta['request'],gaussian_renderer_used=False))
    folder=root/'native_renderers'/method/scene;meta=read(folder/'complete.json')
    if meta.get('pilot'):raise ValueError('A one-view renderer pilot cannot enter evaluation')
    return dict(rgb=folder/'renders',depth=folder/'depth',request=meta['request'])


def selected_scenes(scenes=None):
    """Limit work to complete scenes while keeping the global comparison cohort."""
    selected=list(SCENES if scenes is None else scenes)
    if not selected or len(set(selected))!=len(selected) or any(s not in SCENES for s in selected):
        raise ValueError(f'Expected distinct scenes from {SCENES}, got {selected}')
    return selected


def publisher_scenes():
    """Honor a running pre-upgrade publisher's explicit --scenes argument.

    The bag renderer can finish without restarting its expensive in-memory
    visibility pass. Only our direct parent publisher may provide this scope;
    unrelated manual evaluations continue to default to all scenes.
    """
    try:command=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode().strip('\0').split('\0')
    except FileNotFoundError:return None
    if not any(Path(arg).name=='publish_fine_whole_meshes.py' for arg in command):return None
    for i,arg in enumerate(command):
        if arg=='--scenes' and i+1<len(command):return selected_scenes(command[i+1].split(','))
        if arg.startswith('--scenes='):return selected_scenes(arg.split('=',1)[1].split(','))
    return None


def supports(root,scenes=None):
    for scene in selected_scenes(scenes):
        cs=[c for c in cameras(scene) if c['part']=='held_out']
        for state in STATES:
            sources={m:rendering(root,m,scene,state) for m in METHODS}
            out=root/'common'/scene/(state+'_native_support');out.mkdir(exist_ok=True)
            request=dict(methods=METHODS,state=state,sources={m:s['request'] for m,s in sources.items()},
                geometry_reference='Frozen CPU virtual-scan CD/F1; native RGB masks are independent raster coverage',
                masks=[identity(root/'common'/scene/'held_out'/(c['stem']+'_foreground.png')) for c in cs])
            coverage=[]
            for c in cs:
                fg=np.asarray(Image.open(root/'common'/scene/'held_out'/(c['stem']+'_foreground.png')))>127
                valid=~fg if state=='background' else np.ones(fg.shape,dtype=bool)
                eligible=valid.copy();own={}
                for method,s in sources.items():
                    depth=np.load(s['depth']/(c['stem']+'.npy'),mmap_mode='r')
                    own[method]=np.isfinite(depth)&(depth>0)&eligible;valid&=own[method]
                    rgb=s['rgb']/(c['stem']+'.png')
                    if not rgb.exists():raise FileNotFoundError(rgb)
                    dest=root/state/method/scene/'renders'
                    publish_render_link(rgb,dest/rgb.name)
                Image.fromarray(valid.astype('uint8')*255).save(out/(c['stem']+'.png'))
                coverage.append(dict(name=c['name'],eligible_pixels=int(eligible.sum()),common_pixels=int(valid.sum()),
                    common_fraction=float(valid.sum()/max(eligible.sum(),1)),
                    own_fraction={m:float(v.sum()/max(eligible.sum(),1)) for m,v in own.items()}))
            write(out/'complete.json',dict(complete=True,request=request,request_id=digest(request),coverage=coverage))


def score(root,method,scene,state):
    import torch,lpips
    from mesh_benchmark.score import image_metrics
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(3*1024**3/torch.cuda.get_device_properties(0).total_memory)
    device='cuda:0';criterion=lpips.LPIPS(net='vgg',version='0.1',spatial=True).to(device).eval()
    for p in criterion.parameters():p.requires_grad_(False)
    source=rendering(root,method,scene,state);common=root/'common'/scene
    support=read(common/(state+'_native_support')/'complete.json')
    geometry=read(root/'cache'/method/scene/state/'geometry_metrics.json')
    dest=root/state/method/scene;dest.mkdir(parents=True,exist_ok=True);(dest/'diagnostics').mkdir(exist_ok=True)
    rows=[];own_rows=[]
    for c in [c for c in cameras(scene) if c['part']=='held_out']:
        stem=c['stem'];prediction=np.asarray(Image.open(source['rgb']/(stem+'.png')).convert('RGB'))
        target=np.asarray(Image.open(common/'held_out'/(stem+'_gt.png')).convert('RGB'))
        fg=np.asarray(Image.open(common/'held_out'/(stem+'_foreground.png')))>127
        eligible=~fg if state=='background' else np.ones(fg.shape,dtype=bool)
        depth=np.load(source['depth']/(stem+'.npy'),mmap_mode='r');own=np.isfinite(depth)&(depth>0)&eligible
        allowed=np.asarray(Image.open(common/(state+'_native_support')/(stem+'.png')))>127
        if np.any(allowed&~own):raise ValueError('Native common mask contains a raster miss')
        result=image_metrics(prediction,target,allowed,criterion,device)
        own_result=image_metrics(prediction,target,own,criterion,device)
        row=dict(name=c['name'],**result,evaluation_pixels=int(allowed.sum()),own_pixels=int(own.sum()),
            common_coverage=float(allowed.sum()/max(eligible.sum(),1)),mesh_coverage=float(own.sum()/max(eligible.sum(),1)))
        rows.append(row);own_rows.append(dict(name=c['name'],**own_result,pixels=int(own.sum())))
        Image.fromarray(allowed.astype('uint8')*255).save(dest/'diagnostics'/(stem+'_evaluation_mask.png'))
        Image.fromarray(own.astype('uint8')*255).save(dest/'diagnostics'/(stem+'_coverage.png'))
        error=np.clip(np.mean(np.abs(prediction.astype('float32')-target),2)*3,0,255).astype('uint8');error[~allowed]=0
        Image.fromarray(error).save(dest/'diagnostics'/(stem+'_error.png'))
        print('NATIVE_RGB_SCORE',method,scene,state,row,flush=True)
    def mean(data,keys):
        return {k:float(np.mean([r[k] for r in data if r.get(k) is not None])) if any(r.get(k) is not None for r in data) else None for k in keys}
    info=read(common/'complete.json');graph=read(public_root(root)/'recomposition'/scene/'composition/scene.json')
    result=dict(complete=True,scoring_protocol_version=2,native_mesh_rendering_version=1,
        request=dict(renderer=source['request'],support_id=support['request_id'],geometry_support_id=geometry['support_id']),
        method=method,scene=scene,state=state,geometry=geometry,
        mean=mean(rows,['PSNR','SSIM','LPIPS','mesh_coverage','common_coverage']),per_view=rows,
        own_surface_mean=mean(own_rows,['PSNR','SSIM','LPIPS']),own_surface_per_view=own_rows,native_color_mean=None,
        train_views=info['train_count'],evaluation_views=len(rows),scored_views=sum(r['evaluation_pixels']>0 for r in rows),
        quality_status=graph['status'] if method=='paintmesh' and state=='recomposition' else 'complete',
        gaussian_renderer_used=False,appearance_optimized=False,appearance_history=[],
        notes=['Native colored triangle meshes; no VM-SH field or Gaussian RGB substitution',
               'RGB intersection uses each native rasterizer positive depth; geometry remains the frozen CPU virtual scan',
               'Original-state held-out views; background excludes target mask; sparse SfM is not independent scan GT'])
    flags=public_root(root)/'docs/MESH_QUALITY_FLAGS.json'
    finding=read(flags).get(method+'/'+scene) if flags.exists() else None
    if finding:
        result['quality_status']=finding['status'];result['quality_findings']=finding
    if method in ['come','gof']:
        mesh_marker=read(BASE/method/scene/'mesh_complete.json')
        if mesh_marker.get('support_validation_receipt'):
            result['geometry_postprocessing']=read(mesh_marker['support_validation_receipt'])
        if mesh_marker.get('color_range_correction_receipt'):
            result['color_range_correction']=read(mesh_marker['color_range_correction_receipt'])
        if mesh_marker.get('color_normalization_receipt'):
            result['color_normalization']=read(mesh_marker['color_normalization_receipt'])
    write(dest/'metrics.json',result)


def reuse_unchanged_score(root,method,scene,state):
    """Reuse only identical RGB sources and pixel masks after a color-only fix.

    A source mesh identity in the cohort changes its support request ID even
    when all raster depths/masks are identical. That does not change scores
    for another method. Compare every published mask and geometry record,
    then refresh the support provenance without repeating LPIPS.
    """
    path=root/state/method/scene/'metrics.json'
    if not path.exists():return False
    old=read(path);source=rendering(root,method,scene,state)
    if not old.get('complete') or old.get('native_mesh_rendering_version')!=1:return False
    if old['request']['renderer']!=source['request']:return False
    geometry=read(root/'cache'/method/scene/state/'geometry_metrics.json')
    if old['geometry']!=geometry:return False
    common=root/'common'/scene;support=read(common/(state+'_native_support')/'complete.json')
    for c in [c for c in cameras(scene) if c['part']=='held_out']:
        name=c['stem']+'.png';saved=path.parent/'diagnostics'/(c['stem']+'_evaluation_mask.png')
        if not saved.exists():return False
        current=np.asarray(Image.open(common/(state+'_native_support')/name))>127
        if not np.array_equal(np.asarray(Image.open(saved))>127,current):return False
        coverage=path.parent/'diagnostics'/(c['stem']+'_coverage.png')
        depth=np.load(source['depth']/(c['stem']+'.npy'),mmap_mode='r')
        eligible=~(np.asarray(Image.open(common/'held_out'/(c['stem']+'_foreground.png')))>127) if state=='background' else np.ones(current.shape,dtype=bool)
        own=np.isfinite(depth)&(depth>0)&eligible
        if not coverage.exists() or not np.array_equal(np.asarray(Image.open(coverage))>127,own):return False
    old['request']['support_id']=support['request_id']
    old['request']['geometry_support_id']=geometry['support_id']
    old['score_reuse_reason']='Identical native RGB source, every shared/own pixel mask and frozen geometry; support provenance refreshed'
    write(path,old)
    return True


def run(output,scenes=None):
    scenes=selected_scenes(scenes)
    root=work_root(output);configure_sources(root);locks=[]
    for name in ['native_rgb_campaign.lock','campaign.lock']:
        lock=open(root/name,'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    for method in ['come','gaussian_wrapping']:
        for scene in scenes:
            if not read(BASE/method/scene/'mesh_color_complete.json')['complete']:raise ValueError('Native colors not ready')
    protocol=read(root/'protocol.json');protocol['native_mesh_rgb']=dict(enabled=True,version=1,
        appearance='Original vertex colors; no appearance fitting',geometry='Unchanged conditional surface CD/F1',
        backend=dict(come='CoMe ScalableMeshRenderer / NVDiffRast',gaussian_wrapping='Wrapping ScalableMeshRenderer / NVDiffRast',
                     **{'2dgs':'2DGS TNT pyrender backend + RGB attachment','gof':'GOF TNT pyrender backend + RGB attachment',
                        'paintmesh_stage1':'PaintMesh triangle ray caster, actual original stage-1 mesh',
                        'paintmesh':'PaintMesh triangle ray caster'}))
    protocol.setdefault('previous_appearance',protocol.get('appearance'))
    protocol['appearance']='Original vertex colors and each project mesh backend; no appearance fitting or Gaussian RGB substitution'
    protocol['appearance_optimization_steps']=0
    protocol['config_appearance_fields_are_legacy']=True
    for script in ['evaluate_native_mesh_rgb.py','render_native_colored_meshes.py','colorize_baseline_meshes.py']:
        name=str(HERE/script)
        if name not in protocol['scripts']:protocol['scripts'].append(name)
    write(root/'protocol.json',protocol)
    tasks={m+'/'+s:dict(status='pending') for m in METHODS if m not in ['paintmesh','paintmesh_stage1'] for s in scenes}
    for key,row in tasks.items():
        m,s=key.split('/');p=root/'native_renderers'/m/s/'complete.json'
        if p.exists():
            old=read(p)
            if not old.get('pilot') and old.get('complete') and old['request']['mesh']==verified_file_fingerprint(root,read(BASE/m/s/'mesh_complete.json')['path']) and old['request']['adapter']==fingerprint(HERE/'render_native_colored_meshes.py'):
                row['status']='complete'
            print('NATIVE_MESH_IDENTITY_CHECK',m,s,row['status'],flush=True)
    schedule(root,tasks,'render');supports(root,scenes)
    tasks={m+'/'+s+'/'+state:dict(status='pending') for s in scenes for m in METHODS for state in STATES}
    for key,row in tasks.items():
        method,scene,state=key.split('/')
        if reuse_unchanged_score(root,method,scene,state):row.update(status='complete',reused_exact_inputs=True)
    schedule(root,tasks,'score')
    from mesh_benchmark.report import update
    update(root)
    for script,args in [('plot_mesh_method_comparison.py',['--root',str(root)]),('organize_mesh_subplots.py',['--figures',str(public_root(root)/'figures_all_methods')])]:
        with (root/'logs'/('native_'+script+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(HERE/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    status(root,'complete',tasks)
    print('NATIVE_MESH_RGB_EVALUATION_COMPLETE',scenes,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(BASE/'mesh_comparison_whole'))
    p.add_argument('--score',choices=STATES);p.add_argument('--method',choices=METHODS);p.add_argument('--scene',choices=SCENES)
    p.add_argument('--scenes',help='Comma-separated complete scenes; reports and overviews still include all scenes')
    p.add_argument('--resume-after-pid',type=int)
    a=p.parse_args();root=work_root(a.output);configure_sources(root)
    if a.resume_after_pid:
        while True:
            try:command=Path(f'/proc/{a.resume_after_pid}/cmdline').read_bytes()
            except FileNotFoundError:break
            if b'evaluate_native_mesh_rgb.py' not in command:break
            time.sleep(10)
    if a.score:score(root,a.method,a.scene,a.score)
    else:run(a.output,selected_scenes(a.scenes.split(',')) if a.scenes else publisher_scenes())
