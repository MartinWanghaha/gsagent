#!/usr/bin/env python3
"""Validate CoMe color normalization, regenerate all scenes and comparison files."""
import argparse
import fcntl
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,STATES,read,write,identity
from mesh_benchmark.layout import work_root
from recompose.io import fingerprint
from repair_come_mesh_colors import PUBLIC,WORK,POLICY,publish
from evaluate_native_mesh_rgb import gpus,available_ram

NATIVE=Path('/home/member/wyc/mesh_gs_research/envs/come_probe/bin/python')
RENDER=Path('/home/member/wyc/mesh_gs_research/envs/meshgs_wrapping/bin/python')


def status(stage,tasks):
    value=dict(complete=stage=='complete',stage=stage,tasks=tasks,pid=os.getpid(),updated=time.time())
    write(WORK/'status.json',value)
    lines=['# CoMe 黑斑修复进度','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}；阶段：{stage}。','',
      '使用官方积分器测量同视角真实颜色权重。先验证cube/car完整mesh，再全场景生成、原生mesh渲染、评分并覆盖报告与图片。','',
      '| 任务 | 状态 | GPU |','|---|---|---|']
    lines += [f'| {key} | {r["status"]} | {r.get("gpu","—")} |' for key,r in tasks.items()]
    (PUBLIC/'docs/COME_COLOR_REPAIR_STATUS.md').write_text('\n'.join(lines)+'\n')


def alive(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.exists() and bool(p.read_bytes())


def schedule(scenes,stage,adopt=None):
    sys.path.insert(0,str(BASE/'_scripts'))
    from common import environment
    tasks={s:dict(status='pending') for s in scenes};active={}
    if stage=='normalize':
        for scene in scenes:
            receipt=WORK/scene/'complete.json'
            if not receipt.exists():continue
            old=read(receipt);mesh=Path(old['path']);marker=read(BASE/'come'/scene/'mesh_complete.json')
            if old.get('complete') and mesh.exists() and old['candidate_mesh']['size_bytes']==mesh.stat().st_size and \
               old['request']['geometry']==identity(marker.get('geometry_path',marker['path'])) and \
               old['request']['checkpoint']==identity(BASE/'come'/scene/'model/point_cloud/iteration_30000/point_cloud.ply') and \
               old['request']['policy']==POLICY:
                tasks[scene]['status']='complete'
    for scene,pid in (adopt or {}).items():
        if scene not in tasks:continue
        if tasks[scene]['status']=='complete':continue
        proc=Path(f'/proc/{pid}')
        env={p.split(b'=',1)[0]:p.split(b'=',1)[1] for p in (proc/'environ').read_bytes().split(b'\0') if b'=' in p}
        gpu=int(env[b'CUDA_VISIBLE_DEVICES'].split(b',')[0])
        active[gpu]=(scene,None,None,pid)
        tasks[scene].update(status='running',gpu=gpu,pid=pid)
    while True:
        for gpu,(scene,proc,log,pid) in list(active.items()):
            done=not alive(pid) if proc is None else proc.poll() is not None
            if not done:continue
            if log:log.close()
            if stage=='normalize':success=(WORK/scene/'complete.json').exists()
            elif stage=='pilot_render':success=(WORK/'pilot'/scene/'render_complete.json').exists()
            else:success=(WORK/'pilot'/scene/'metrics.json').exists()
            if proc is not None:success=success and proc.returncode==0
            tasks[scene].update(status='complete' if success else 'failed');del active[gpu]
            print('COME_REPAIR_TASK_END',stage,scene,success,flush=True)
        for gpu,free in gpus():
            # Actual native color workers peak at about 3 GiB host RSS.
            # Retain 16 GiB available host RAM plus 14 GiB device headroom.
            if gpu in active or free<14000 or available_ram()<16:continue
            pending=[s for s,r in tasks.items() if r['status']=='pending']
            if not pending:break
            scene=pending[0]
            if stage=='normalize':
                command=[str(NATIVE),'-u',str(HERE/'colorize_baseline_meshes.py'),'--worker','--scene',scene,'--normalize-colors','--chunk','2097152']
            elif stage=='pilot_render':command=[str(RENDER),'-u',str(HERE/'render_come_color_candidate.py'),'--scene',scene]
            else:command=[sys.executable,'-u',str(HERE/'render_come_color_candidate.py'),'--scene',scene,'--score']
            log=open(WORK/(scene+'_'+stage+'.log'),'w')
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=environment(gpu))
            active[gpu]=(scene,proc,log,proc.pid);tasks[scene].update(status='running',gpu=gpu,pid=proc.pid)
            print('COME_REPAIR_TASK_START',stage,scene,gpu,proc.pid,flush=True)
        status(stage,tasks)
        if not active and not any(r['status']=='pending' for r in tasks.values()):break
        time.sleep(10)
    if any(r['status']!='complete' for r in tasks.values()):raise RuntimeError('CoMe repair stage failed; see scene logs')
    return tasks


def snapshot():
    root=work_root(PUBLIC);before=WORK/'before'
    if (before/'complete.json').exists():return
    for scene in SCENES:
        dest=before/scene;dest.mkdir(parents=True,exist_ok=True);native=root/'native_renderers/come'/scene
        shutil.copytree(native/'renders',dest/'renders',dirs_exist_ok=True)
        write(dest/'native_complete.json',read(native/'complete.json'))
        write(dest/'depth_hashes.json',{p.name:fingerprint(p)['sha256'] for p in (native/'depth').glob('*.npy')})
        for state in STATES:
            shutil.copy2(root/state/'come'/scene/'metrics.json',dest/(state+'_metrics.json'))
            shutil.copytree(root/'common'/scene/(state+'_native_support'),dest/(state+'_native_support'),dirs_exist_ok=True)
    write(before/'complete.json',dict(complete=True,updated=time.time()))


def publish_report():
    root=work_root(PUBLIC);figures=PUBLIC/'figures_all_methods/come_color_repair';figures.mkdir(exist_ok=True)
    chosen=read(PUBLIC/'figures_all_methods/metadata/selected_views.json')['scenes'];records=[]
    for scene in SCENES:
        result=read(WORK/scene/'complete.json');comparison=[]
        for state in STATES:
            old=read(WORK/'before'/scene/(state+'_metrics.json'));new=read(root/state/'come'/scene/'metrics.json')
            assert old['geometry']==new['geometry'],'Color-only repair changed geometric evaluation'
            for view in old['per_view']:
                name=Path(view['name']).stem+'.png'
                np.testing.assert_array_equal(np.asarray(Image.open(WORK/'before'/scene/(state+'_native_support')/name)),
                    np.asarray(Image.open(root/'common'/scene/(state+'_native_support')/name)))
            comparison.append(dict(state=state,before=old['mean'],after=new['mean']))
        for name,sha in read(WORK/'before'/scene/'depth_hashes.json').items():
            assert fingerprint(root/'native_renderers/come'/scene/'depth'/name)['sha256']==sha,'Mesh depth changed'
        records.append(dict(scene=scene,vertices=result['vertices'],changed_vertices=result['changed_vertices'],
            zero_weight_vertices=result['zero_weight_vertices'],native_probe=result['native_probe'],comparison=comparison))
        c=chosen[scene]['camera'];name=c['stem']+'.png'
        paths=[WORK/'before'/scene/'renders'/name,root/'native_renderers/come'/scene/'renders'/name,
               root/'common'/scene/'held_out'/(c['stem']+'_gt.png')]
        width=650;height=round(width*c['height']/c['width']);canvas=Image.new('RGB',(width*3,height+40),'white');draw=ImageDraw.Draw(canvas)
        for i,(p,label) in enumerate(zip(paths,['Before: premultiplied native RGB','After: native white-field normalization','Reference image'])):
            canvas.paste(Image.open(p).convert('RGB').resize((width,height)),(i*width,36));draw.text((i*width+10,10),scene+' | '+label,fill='black')
        canvas.save(figures/(scene+'.jpg'),quality=95)
    write(PUBLIC/'docs/COME_COLOR_REPAIR_RESULTS.json',dict(complete=True,updated=time.time(),policy=POLICY,records=records,
        geometry_exact=True,depth_files_exact=True,evaluation_masks_exact=True,native_mesh_backend='CoMe ScalableMeshRenderer / NVDiffRast'))
    lines=['# CoMe 原生 mesh 黑斑修复','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}。','',
      '源码输出的是颜色累积 C=sum(w_i*RGB_i)，随后直接保存为不透明mesh的顶点色；颜色权重 W=sum(w_i) 小于1时会发暗。几何不透明度与颜色合成权重不同，不能用固定乘2或几何alpha代替。细节见 [逐视角与白场审计](COME_BLACK_PATCH_AUDIT.md)。','',
      '本次保留官方 evaluate_alpha 的最低不透明度选视角规则，记录每个顶点的获选视角；在同一个视角临时以恒白色查询高斯，精确测量原颜色权重W，再输出clip(C/W,0,1)。W为0时保留原RGB并记录未知，不补造颜色。无GT着色、无图像修补、无重训、无顶点移动。','',
      '每批重放官方RGB并与原PLY字节核对；白场前后几何alpha不变，cube另验证分批与整批查询一致。所有非RGB属性/面记录完全一致，全部原生深度、共同评分mask及CD/F1验证不变。','',
      '该结果标记为官方原生提取加颜色归一化修正，不冒充未修改的官方导出。已沿用各方法原生mesh渲染器更新全部比较。','',
      '| 场景 | 模式 | PSNR前 | 后 | SSIM前 | 后 | LPIPS前 | 后 | 零权重顶点 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        for c in r['comparison']:
            a,b=c['before'],c['after'];lines.append(f'| {r["scene"]} | {c["state"]} | {a["PSNR"]:.4f} | {b["PSNR"]:.4f} | {a["SSIM"]:.4f} | {b["SSIM"]:.4f} | {a["LPIPS"]:.4f} | {b["LPIPS"]:.4f} | {r["zero_weight_vertices"]} |')
    lines+=['','[11场景修复前后图](../figures_all_methods/come_color_repair/) · [完整记录](COME_COLOR_REPAIR_RESULTS.json)','',
      '颜色归一化消除由权重不足导致的系统偏暗；原视角选色、SH高光和重建几何仍可能造成纹理不连续。COLMAP稀疏点云仅作为观测表面一致性参照，不能视为独立扫描真值。', '',
      '[纯背景报告](../reports/background.md) · [组合mesh报告](../reports/recomposition.md) · [全方法对比图](../figures_all_methods/README.md)']
    (PUBLIC/'docs/COME_COLOR_REPAIR.md').write_text('\n'.join(lines)+'\n')
    for p in [PUBLIC/'README.md',PUBLIC/'docs/README.md',PUBLIC/'docs/QUALITY_NOTES.md']:
        text=p.read_text();link='docs/COME_COLOR_REPAIR.md' if p.parent==PUBLIC else 'COME_COLOR_REPAIR.md'
        text=text.replace('CoMe 全场景颜色权重归一化修复已启动，验证和最终更新将自动执行：','CoMe 全场景颜色权重归一化修复及最终更新已完成：')
        if 'CoMe 全场景颜色权重归一化已完成' not in text:p.write_text(text+f'\nCoMe 全场景颜色权重归一化已完成，两个报告与全部子图已更新：[源码原因和同像素修复结果]({link})。\n')
    p=PUBLIC/'docs/OPACITY_MESH_REPAIR.md';text=p.read_text()
    if 'CoMe 当前另已完成原生颜色权重归一化' not in text:
        p.write_text('**CoMe 当前另已完成原生颜色权重归一化，以下几何修复表保留当时颜色，最新结果见 [CoMe颜色修复](COME_COLOR_REPAIR.md)。**\n\n'+text)
    p=PUBLIC/'docs/COLORED_MESHES.md';text=p.read_text()
    if 'COME_COLOR_REPAIR.md' not in text:p.write_text(text+'\n当前 CoMe 彩色文件进一步修正原生RGB预乘权重，几何保持一致：[颜色修复](COME_COLOR_REPAIR.md)。GOF另有 [范围截断修复](GOF_COLOR_REPAIR.md)。\n')
    delivery=root/'colored_mesh_delivery_check.json'
    if delivery.exists():
        value=read(delivery)
        value.update(native_color_export_corrections=dict(come='Same-view white-field unpremultiplication',gof='Saturating RGB range conversion'),
            vertex_coordinates_and_faces_unchanged_by_color_corrections=True)
        for entry in value.get('meshes',[]):
            if entry['method']=='come':
                receipt=read(BASE/'come'/entry['scene']/'mesh_color_complete.json')
                entry.update(colored_path=receipt['colored_mesh']['path'],geometry_unchanged_by_colorization=True,
                             source_native_colors_corrected=True,color_receipt=str(BASE/'come'/entry['scene']/'mesh_color_complete.json'))
        write(delivery,value)
    # The earlier audit describes its own read-only stage, not the current
    # delivery; explicitly connect that historical stage to the final repair.
    p=PUBLIC/'docs/COME_BLACK_PATCH_AUDIT.md';text=p.read_text()
    if '完整mesh修复已完成' not in text:p.write_text('**完整mesh修复已完成，最新结果见 [CoMe颜色修复](COME_COLOR_REPAIR.md)。以下保留修复前只读诊断。**\n\n'+text)
    refresh_previews()
    return records


def refresh_previews():
    """Replace early diagnostic previews with current images; archive evidence."""
    figures=PUBLIC/'figures_all_methods';dest=figures/'native_color_preview'
    history=WORK/'before_preview';history.mkdir(exist_ok=True)
    for p in dest.glob('*.jpg'):
        old=history/p.name
        if not old.exists():shutil.copy2(p,old)
    for method,name in [('gof','gof_color_patch_audit.jpg'),('come','come_black_patch_audit.jpg')]:
        width,height=480,335;canvas=Image.new('RGB',(3*width,4*height),'white');draw=ImageDraw.Draw(canvas)
        for i,scene in enumerate(SCENES):
            source=figures/'subplots'/method/'recomposition'/scene/'rgb.png';im=Image.open(source).convert('RGB')
            im.thumbnail((width-14,height-35));x=(i%3)*width;y=(i//3)*height
            canvas.paste(im,(x+(width-im.width)//2,y+28));draw.text((x+8,y+6),scene+' | '+method+' | color corrected',fill='black')
        canvas.save(dest/name,quality=95)
    for scene,method,name in [('car','come','car_come.jpg'),('bag','gaussian_wrapping','bag_wrapping.jpg'),
                              ('truck','gaussian_wrapping','truck_wrapping.jpg')]:
        Image.open(figures/'subplots'/method/'recomposition'/scene/'rgb.png').convert('RGB').save(dest/name,quality=95)
    methods=['2dgs','come','gof','gaussian_wrapping','paintmesh'];scenes=['cube','car','truck']
    width=400;height=310;canvas=Image.new('RGB',(width*len(methods),height*len(scenes)),'white');draw=ImageDraw.Draw(canvas)
    for row,scene in enumerate(scenes):
        for col,method in enumerate(methods):
            im=Image.open(figures/'subplots'/method/'recomposition'/scene/'rgb.png').convert('RGB');im.thumbnail((width-10,height-35))
            canvas.paste(im,(col*width+(width-im.width)//2,row*height+28));draw.text((col*width+6,row*height+6),scene+' | '+method,fill='black')
    canvas.save(dest/'all_methods_problem_scene_audit.jpg',quality=95)
    (dest/'README.md').write_text('# 当前原生mesh颜色预览\n\n本层JPG已更新到GOF范围截断及CoMe颜色权重归一化后的结果。原先诊断图片存于 `_internal/workspace/come_color_repair/before_preview`；`index_precision_fix` 是此前整数索引修复的历史证据。\n\n[颜色修复总览](../../docs/COLOR_EXPORT_REPAIRS.md) · [全场景主图](../README.md)\n')


def joint_report():
    root=work_root(PUBLIC)
    lines=['# GOF / CoMe 原生mesh颜色导出修复完成','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}。','',
      '| 方法 | 源码原因 | 修正 |','|---|---|---|',
      '| GOF | 浮点积分颜色超出1，直接uint8转换导致回绕为彩色斑点 | 只将超范围通道饱和至0/255，范围内原字节保留 |',
      '| CoMe | 颜色合成权重不足，累计RGB直接当不透明mesh顶点色 | 用同视角原生恒白场测量权重W，再输出C/W；零权重保留并记录 |','',
      '两项修正均不使用GT，不重训3DGS，不改变mesh顶点坐标或三角面，不改变评估mask/阈值。CD/F1与修正前完全一致。各方法仍使用原来的mesh三角形渲染后端。当前对比显式注明颜色校正与已有几何支撑校验，原始导出保留。','',
      '## 当前主报告均值（11场景等权）','',
      '| 模式 | 方法 | PSNR | SSIM | LPIPS | CD-L1 | F1@1% |','|---|---|---:|---:|---:|---:|---:|']
    for state in STATES:
        for method in ['2dgs','come','gof','gaussian_wrapping','paintmesh']:
            rows=[read(root/state/method/s/'metrics.json') for s in SCENES]
            mean=lambda key:float(np.mean([r['mean'][key] for r in rows]))
            # Geometry schema is preserved by the benchmark; use its actual
            # field names rather than invent a Chamfer variant.
            g=rows[0]['geometry'];cd_key=next((k for k in ['CD_L1','CD-L1','CD','chamfer_l1'] if k in g),None)
            cd=f'{np.mean([r["geometry"][cd_key] for r in rows]):.6f}' if cd_key else '见主报告'
            f1=float(np.mean([r['geometry']['F1'] for r in rows]))
            lines.append(f'| {state} | {method} | {mean("PSNR"):.6f} | {mean("SSIM"):.6f} | {mean("LPIPS"):.6f} | {cd} | {f1:.6f} |')
    lines+=['','[纯背景逐场景报告](../reports/background.md) · [组合mesh逐场景报告](../reports/recomposition.md) · [全部对比图及各方法子图](../figures_all_methods/README.md)','',
      '[GOF原因、修复数据与11场景前后图](GOF_COLOR_REPAIR.md) · [CoMe原因、修复数据与11场景前后图](COME_COLOR_REPAIR.md)','',
      '稀疏COLMAP点云是观测表面参照，不是独立扫描真值；F1解释为共同观测范围内的表面一致性。纯背景目标mask内不评分；无mesh像素不评分，覆盖率同时报告。原有PaintMesh needs_review物体/位姿状态保留。','',
      '消除导出错误不等于解决所有重建误差。CoMe的单视角选色与SH视角依赖仍会留下纹理/高光不连续；这种残余误差保留在图片和指标里。']
    (PUBLIC/'docs/COLOR_EXPORT_REPAIRS.md').write_text('\n'.join(lines)+'\n')


def run(adopt):
    WORK.mkdir(exist_ok=True,parents=True);lock=open(WORK/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    schedule(['cube','car'],'normalize',adopt)
    schedule(['cube','car'],'pilot_render');schedule(['cube','car'],'pilot_score')
    pilot=[]
    for scene in ['cube','car']:
        p=read(WORK/'pilot'/scene/'metrics.json');pilot.append(p)
        for state,row in p['states'].items():
            assert row['after']['PSNR']>row['before']['PSNR'],(scene,state,'No full-mesh PSNR improvement')
            assert row['after']['SSIM']>=row['before']['SSIM'],(scene,state,'SSIM regressed')
            assert row['after']['LPIPS']<=row['before']['LPIPS'],(scene,state,'LPIPS regressed')
    write(WORK/'pilot_validation.json',dict(complete=True,passed=True,policy=POLICY,scenes=pilot,
          note='Fixed correction derived from constant-field invariance; no parameter tuning from held-out scores'))
    schedule([s for s in SCENES if s not in ['cube','car']],'normalize',adopt)
    # GOF owns the current evaluation/publication pass. Let it finish before
    # publishing another method's new primary meshes or score provenance.
    gof=PUBLIC/'_internal/workspace/gof_color_repair/status.json'
    while gof.exists() and not read(gof).get('complete'):
        status('waiting for GOF evaluation publication',{})
        pid=read(gof).get('pid')
        if pid and not alive(pid):raise RuntimeError('GOF publication did not complete; inspect its campaign log')
        time.sleep(10)
    snapshot()
    for scene in SCENES:publish(scene)
    root=work_root(PUBLIC);protocol=read(root/'protocol.json')
    protocol['native_color_normalization']=dict(enabled=True,method='come',scenes=SCENES,policy=POLICY)
    for name in ['repair_gof_mesh_colors.py','run_gof_color_repair.py','repair_come_mesh_colors.py',
                 'run_come_color_repair.py','render_come_color_candidate.py','audit_come_mesh_colors.py']:
        path=str(HERE/name)
        if path not in protocol['scripts']:protocol['scripts'].append(path)
    write(root/'protocol.json',protocol);status('native rerender and metric refresh',{})
    with (WORK/'evaluate.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(HERE/'evaluate_native_mesh_rgb.py'),'--output',str(PUBLIC)],stdout=log,stderr=subprocess.STDOUT,check=True)
    publish_report()
    from verify_native_mesh_delivery import verify
    verify(PUBLIC)
    joint_report()
    status('complete',{s:dict(status='complete') for s in SCENES});print('COME_COLOR_REPAIR_DELIVERY_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--adopt',action='append',default=[]);a=p.parse_args()
    run({x.split(':')[0]:int(x.split(':')[1]) for x in a.adopt})
