#!/usr/bin/env python3
"""Whole-scene A/B/C extraction and native mesh rendering, isolated from production."""
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
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,read,write,identity,digest,MeshScene,rays,surface_metrics
from mesh_benchmark.layout import work_root,configure_sources
from mesh_benchmark.fine_tsdf import VOXEL,TRUNCATION,BLOCK,partitions,retain_owned_faces,integrate_partition
from mesh_benchmark.surface_fusion_probe import integrate_sampled
from mesh_benchmark.full_surface_fusion import allocate,save_part

PUBLIC=BASE/'mesh_comparison_whole';ROOT=work_root(PUBLIC)
DEFAULT=PUBLIC/'experiments/bag_full_surface_20261112'
LABELS=dict(A='Original complete fine TSDF',B='Full surface allocation + nearest depth',C='Full surface allocation + inverse depth')


def resource_ready(folder):
    return shutil.disk_usage(folder).free>=150*1024**3 and next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))>=32*1024**2


def status(out,stage,**extra):
    write(out/'status.json',dict(updated=time.time(),stage=stage,pid=os.getpid(),**extra))
    (out/'STATUS.md').write_text('# bag 全场景实验进度\n\n'+f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}；阶段：{stage}。\n\n'+
        'A 为相同输入下已完成的全场景细TSDF；B/C 在完整空间重新提取，不拼接旧网格、不使用局部替换图。所有既定原尺寸 RGB-D 和留出视角参与。\n\n'+
        f'详细状态见 [JSON](status.json)。当前任务：`{extra}`。\n')


def extract(out,key):
    import torch,open3d as o3d
    torch.set_num_threads(4);out=Path(out);plan=read(out/'plan.json');folder=out/key;folder.mkdir(exist_ok=True)
    lock=open(folder/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    allocation=read(out/'allocation/complete.json');coords=np.load(out/'allocation/blocks.npy')
    request=dict(allocation=allocation['request_id'],method=key,source=plan['source'],implementation=plan['implementation'],
        voxel=VOXEL,sdf_trunc=TRUNCATION,weight_threshold=0,all_frames=True,full_scene=True)
    if (folder/'complete.json').exists():
        if read(folder/'complete.json')['request']!=request:raise ValueError('Extraction request changed')
        return
    frames=[]
    for r in plan['frames']:
        if identity(r['path'])!=r['identity']:raise ValueError('RGB-D input changed')
        with np.load(r['path']) as a:frames.append({k:a[k].copy() for k in ['depth','rgb','K','w2c']})
    specs=[];total=0;vertices_total=0
    for i,(core,active) in enumerate(partitions(coords)):
        marker=folder/'parts'/f'{i:05d}.json';sign=digest(dict(request=request,core=core.tolist(),active=active.tolist()))
        if marker.exists():
            a=read(marker)
            if a['signature']!=sign:raise ValueError('Partition changed')
            if a['faces']:specs.append(a['spec'])
            total+=a['faces'];vertices_total+=a['vertices'];continue
        while True:
            free,_=torch.cuda.mem_get_info()
            if resource_ready(out) and free>=12*1024**3:break
            write(folder/'runtime.json',dict(stage='waiting for resources',partition=i,free_vram=int(free),updated=time.time()))
            time.sleep(30)
        write(folder/'runtime.json',dict(stage='full extraction',partition=i,core_blocks=len(core),active_blocks=len(active),
            finished_core_blocks=i*6000,total_blocks=len(coords),faces=total,pid=os.getpid(),updated=time.time()))
        print('FULL_PART_START',key,i,len(core),len(active),flush=True)
        if key=='B':v,f,c=integrate_partition(active,frames,'CUDA:0')
        else:v,f,c=integrate_sampled(active,frames,'CUDA:0','inverse',chunk_blocks=256)
        o3d.core.cuda.release_cache();torch.cuda.empty_cache()
        keep=retain_owned_faces(v,f,core);f=f[keep]
        spec=None;nv=nf=0
        if len(f):spec,nv,nf=save_part(folder/'parts',i,v,f,c);specs.append(spec)
        total+=nf;vertices_total+=nv
        write(marker,dict(complete=True,signature=sign,spec=spec,vertices=nv,faces=nf,core_blocks=len(core)))
        print('FULL_PART_COMPLETE',key,i,'faces',nf,'total_faces',total,flush=True)
        del v,f,c;gc.collect()
    write(folder/'complete.json',dict(complete=True,request=request,mesh_specs=specs,faces=total,vertices=vertices_total,
        full_scene=True,roi=None,images_downsampled=False,geometry_simplified=False))
    print('FULL_EXTRACTION_COMPLETE',key,total,flush=True)


def render(out,key):
    from mesh_benchmark.stream_cache import outside_frustum
    from recompose.io import fingerprint
    from recompose.mesh_export import export_mesh_scene
    folder=out/key;meta=read(folder/'complete.json');specs=meta['mesh_specs']
    cs=[c for c in read(ROOT/'common/bag/complete.json')['cameras'] if c['part']=='held_out']
    request=dict(meshes=[identity(s['path']) for s in specs],cameras=cs,backend='PaintMesh native triangle ray casting',full_scene=True)
    marker=folder/'renders/complete.json'
    if marker.exists() and read(marker)['request_id']==digest(request):return
    views=[dict(camera=c,rr=rays(c),depth=np.zeros(c['width']*c['height'],dtype='float32'),rgb=np.zeros((c['width']*c['height'],3),dtype='float32')) for c in cs]
    for i,spec in enumerate(specs):
        model=MeshScene([spec]);box=model.counts[0]['bounds']
        for v in views:
            if outside_frustum(box,v['camera']):continue
            z,color=model.cast(v['rr'],color=True);front=(z>0)&((v['depth']==0)|(z<v['depth']))
            v['depth'][front]=z[front];v['rgb'][front]=color[front]
        del model;gc.collect()
        if i%10==0:print('FULL_RENDER',key,i+1,len(specs),flush=True)
    (folder/'renders').mkdir(exist_ok=True)
    for v in views:
        c=v['camera'];np.save(folder/'renders'/(c['stem']+'_depth.npy'),v['depth'].reshape(c['height'],c['width']))
        Image.fromarray(np.rint(np.clip(v['rgb'],0,1)*255).astype('uint8').reshape(c['height'],c['width'],3)).save(folder/'renders'/(c['stem']+'.png'))
    write(marker,dict(complete=True,request_id=digest(request),request=request,views=len(cs),all_triangles_rendered=True))
    refs=[fingerprint(s['path']) for s in specs]
    export_mesh_scene(dict(background=dict(mesh=refs[0],patches=refs[1:]),objects=[],status='experimental'),folder/'world')


def evaluate(out):
    import torch,lpips
    from mesh_benchmark.score import image_metrics
    from mesh_benchmark.overlap import observations,select_anchors,prediction_points
    from plot_mesh_method_comparison import font
    torch.set_num_threads(4);configure_sources(ROOT)
    cs=[c for c in read(ROOT/'common/bag/complete.json')['cameras'] if c['part']=='held_out']
    depth={k:[np.load(out/k/'renders'/(c['stem']+'_depth.npy'),mmap_mode='r') for c in cs] for k in LABELS}
    criterion=lpips.LPIPS(net='vgg',version='0.1',spatial=True).to('cpu').eval()
    for parameter in criterion.parameters():parameter.requires_grad_(False)
    means={};all_results={}
    for state in ['background','original_scene']:
        masks=[];eligible=[]
        for i,c in enumerate(cs):
            fg=np.asarray(Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_foreground.png')))>127
            allowed=~fg if state=='background' else np.ones(fg.shape,dtype=bool);eligible.append(allowed)
            masks.append(allowed&np.logical_and.reduce([depth[k][i]>0 for k in LABELS]))
        with np.load(ROOT/'common/bag/reference_points.npz') as data:reference=data['background' if state=='background' else 'recomposition']
        obs=observations('bag',cs,reference)
        chosen=select_anchors(obs[:,0],obs[:,1],obs[:,2],masks,np.ones(len(reference),dtype=bool),42,max_points=10**9)
        for key in LABELS:
            rows=[]
            for i,c in enumerate(cs):
                gt=np.asarray(Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_gt.png')).convert('RGB'))
                rgb=np.asarray(Image.open(out/key/'renders'/(c['stem']+'.png')).convert('RGB'))
                score=image_metrics(rgb,gt,masks[i],criterion,'cpu')
                own=(depth[key][i]>0)&eligible[i]
                rows.append(dict(camera=c['name'],**score,shared_pixels=int(masks[i].sum()),
                    coverage=float(own.sum()/max(eligible[i].sum(),1))))
            pred=prediction_points(cs,depth[key],chosen,obs)
            geometry=surface_metrics(pred,reference[obs[chosen,0]],read(ROOT/'common/bag/complete.json')['extent'])
            mean={name:float(np.mean([r[name] for r in rows if r['shared_pixels']])) for name in ['PSNR','SSIM','LPIPS','coverage']}
            all_results[state+'/'+key]=dict(mean=mean,geometry=geometry,per_view=rows)
            print('FULL_METRICS',state,key,mean,'CD',geometry['CD_L1'],'F1',geometry['F1'],flush=True)
    (out/'figures').mkdir(exist_ok=True)
    for c in cs:
        width=c['width'];height=c['height'];sheet=Image.new('RGB',(4*width,height+75),'white');draw=ImageDraw.Draw(sheet)
        draw.text((10,5),f'bag | {c["name"]} | complete extraction, all frames and original image pixels',font=font(20),fill='black')
        gt=Image.open(ROOT/'common/bag/held_out'/(c['stem']+'_gt.png')).convert('RGB')
        for j,key in enumerate(['GT','A','B','C']):
            picture=gt if key=='GT' else Image.open(out/key/'renders'/(c['stem']+'.png')).convert('RGB')
            if picture.size!=(width,height):raise ValueError('Unexpected image resizing')
            draw.text((j*width+10,35),key if key=='GT' else key+': '+LABELS[key],font=font(17),fill='black')
            sheet.paste(picture,(j*width,70))
        sheet.save(out/'figures'/(c['stem']+'.png'))
    write(out/'results.json',dict(complete=True,updated=time.time(),full_scene=True,roi=None,results=all_results,
        source='Same 30000-step stage1 checkpoint; original object retained; no generated object or inpaint substitution',
        geometry_reference='Observed COLMAP tracks, not independent scan GT',all_eval_views=len(cs)))
    lines=['# bag 完整场景 A/B/C 提取结果','',
        '三组都是完整三角网格渲染，未拼接原场景截图。A 复用完全相同输入下已提取的完整细TSDF；B/C 使用全部85帧在完整空间重新提取。全部13个留出视角，不裁剪、不减面、不缩图。','',
        '| 模式 | 组别 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 自身覆盖率 ↑ | CD-L1 ↓ | F1@1% ↑ |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name,r in all_results.items():
        m=r['mean'];g=r['geometry'];lines.append('| '+' | '.join([*name.split('/'),*[f'{x:.6f}' for x in [m['PSNR'],m['SSIM'],m['LPIPS'],m['coverage'],g['CD_L1'],g['F1']]]])+' |')
    lines+=['','original_scene 是带原物体的 stage1，不能冒充MV-SAM3D组合结果。此实验不覆盖正式两份报告。','',
        *[f'- [{c["name"]} 整图对比](figures/{c["stem"]}.png)' for c in cs]]
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);configure_sources(ROOT)
    lock=open(out/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_folder=ROOT/'fine_whole/stage1/bag';source=read(source_folder/'frames/complete.json')
    files=[Path(__file__),HERE/'mesh_benchmark/full_surface_fusion.py',HERE/'mesh_benchmark/surface_fusion_probe.py',HERE/'mesh_benchmark/fine_tsdf.py']
    plan=dict(scene='bag',source=identity(source_folder/'frames/complete.json'),frames=source['frames'],
        implementation=[identity(p) for p in files],voxel=VOXEL,sdf_trunc=TRUNCATION,roi=None,
        full_scene=True,train_views=len(source['frames']),methods=LABELS,
        production_pipeline_modified=False,near_mesh_splicing=False,mesh_decimation=False,image_downsampling=False)
    marker=out/'plan.json'
    if marker.exists() and read(marker)!=plan:raise ValueError('Full experiment inputs changed')
    write(marker,plan)
    (out/'PLAN.md').write_text('# bag 全场景重新提取和渲染\n\n'
        '这是独立全场景实验，不是局部ROI替换展示。先验证带原物体的 stage1 完整场景，主项目流程和正式对比结果不变。\n\n'
        '- A：相同85帧、0.002体素、0.008带宽下已经提取的完整原生TSDF。复用其全部原始分区，重新渲染，避免重复生成同一份十亿面资产。\n'
        '- B：连续表面覆盖分配，在完整空间重新积分和提取，最近邻深度。\n'
        '- C：相同完整分配，加边缘感知逆深度积分，完整重新提取。无原网格拼接。\n'
        '- 保持原生1/4分辨率、全部85个训练RGB-D视角、全部13个留出视角；各组使用相同单位融合权重和提取权重门限。\n'
        '- 每组使用真实三角网格渲染器输出整张RGB/深度。背景评分排除目标mask；original_scene评分保留原物体。此处不是inpaint/composed的新训练。\n'
        '- 先检查完整有效域覆盖率，再在共同有效像素计算PSNR/SSIM/LPIPS，在真实观测SfM点处计算CD/F1；不按误差选图或选点。\n'
        '- 保留150GiB磁盘和32GiB主机内存余量，提取前检查空闲显存；资源不足等待，不粗化体素，不中止其他任务。\n')
    snapshot=out/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for p in files:
        dest=snapshot/p.name
        if not dest.exists():shutil.copy2(p,dest)
    baseline=read(source_folder/'complete.json');(out/'A').mkdir(exist_ok=True)
    write(out/'A/complete.json',dict(complete=True,mesh_specs=baseline['mesh_specs'],faces=baseline['faces'],
        request=baseline['request'],reused_identical_full_scene_extraction=True,source=identity(source_folder/'complete.json')))
    status(out,'complete-scene surface allocation',frames=len(source['frames']))
    while not resource_ready(out):status(out,'waiting for allocation resources');time.sleep(30)
    allocate(out/'allocation',source['frames'],source_folder/'blocks.npy')
    active={};failures={};finished={key:(out/key/'complete.json').exists() for key in ['B','C']}
    while not all(finished.values()):
        for gpu,(key,p,log) in list(active.items()):
            code=p.poll()
            if code is None:continue
            log.close();del active[gpu];finished[key]=(out/key/'complete.json').exists()
            if code and not finished[key]:failures[key]=failures.get(key,0)+1
            print('FULL_WORKER_END',key,code,flush=True)
        if any(v>=3 for v in failures.values()):raise RuntimeError(f'Full extraction requires inspection: {failures}')
        query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        for line in query.stdout.splitlines():
            gpu,free=map(int,line.split(','))
            pending=[k for k in finished if not finished[k] and k not in [r[0] for r in active.values()]]
            if not pending or not resource_ready(out):break
            if gpu in active or free<16384:continue
            key=pending[0];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            log=open(out/(key+'.log'),'a');p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(out),'--extract',key],
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(key,p,log);print('FULL_WORKER_START',key,gpu,p.pid,flush=True)
        status(out,'complete-scene extraction',finished=finished,active={str(g):dict(method=r[0],pid=r[1].pid) for g,r in active.items()},failures=failures)
        time.sleep(30)
    for key in LABELS:
        status(out,'complete-scene native mesh rendering',method=key);render(out,key)
    status(out,'complete-scene scoring and figures');evaluate(out)
    status(out,'complete',complete=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--extract',choices=['B','C']);a=p.parse_args()
    if a.extract:extract(a.output,a.extract)
    else:run(a.output)
