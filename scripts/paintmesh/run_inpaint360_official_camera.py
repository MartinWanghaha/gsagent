#!/usr/bin/env python3
"""Rerun PaintMesh with native Inpaint360GS circle cameras and square dilation.

Create once, then run --job PATH (resumable, serial GPU execution). Geometry and
supervision come from the currently registered PaintMesh results. Native camera
generation uses training poses only; test images are used only by evaluation.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from run_geometry_ablation import read, write, atomic, sha, now, METRICS

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = REPO.parent / 'Inpaint360GS'
DATA = REPO.parent / 'data/Inpaint360GS/inpaint360'
REPORT = REPO / 'logs/paintmesh_full_r4_gpu1_20261025_105925'
PYTHON = Path('/home/member/.conda/envs/paintmesh/bin/python')


def record(path):
    p = Path(path).resolve(strict=True); s = p.stat()
    return dict(path=str(p), size_bytes=s.st_size, mtime_ns=s.st_mtime_ns, sha256=sha(p))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def export_cameras(scene):
    import numpy as np
    from PIL import Image
    from virtual_render_io import camera_contract
    # These are the official source files, not the PaintMesh path planner.
    sys.path.insert(0, str(OFFICIAL))
    colmap = module(OFFICIAL / 'scene/colmap_loader.py', 'official_circle_colmap')
    pose = module(OFFICIAL / 'utils/pose_utils.py', 'official_circle_pose')
    work = Path(scene['official_workspace']); name = scene['name']
    config = work / 'config/object_inpaint/inpaint360' / f'{name}.json'
    native = read(config); radius = float(native['circle_radius'])
    sparse = work / 'data/inpaint360' / name / 'sparse/0'
    ex = colmap.read_extrinsics_binary(str(sparse / 'images.bin'))
    intr = colmap.read_intrinsics_binary(str(sparse / 'cameras.bin'))
    original = sorted((c for c in ex.values() if 'test' not in c.name), key=lambda c: Path(c.name).stem)
    train = [c for i, c in enumerate(original) if i % 8 != 0]
    views = [SimpleNamespace(R=colmap.qvec2rotmat(c.qvec).T, T=np.array(c.tvec)) for c in train]
    k = intr[train[0].camera_id]
    if k.model not in ('SIMPLE_PINHOLE', 'PINHOLE'): raise ValueError(k.model)
    fx = k.params[0]; fy = k.params[1] if k.model == 'PINHOLE' else fx
    template = work / 'output/inpaint360' / name / 'virtual/ours_2000/renders/00000.png'
    w, h = Image.open(template).size
    poses = pose.generate_ellipse_path(views, n_frames=30, is_circle=True, circle_radius=radius)
    cameras = [SimpleNamespace(image_name=f'{i:05d}', R=p[:3,:3].T, T=p[:3,3],
        FoVx=float(2*np.arctan(k.width/(2*fx))), FoVy=float(2*np.arctan(k.height/(2*fy))),
        image_width=w, image_height=h, znear=.01, zfar=100., trans=np.zeros(3), scale=1.)
        for i, p in enumerate(poses)]
    root = Path(scene['removal_root'])
    path = camera_contract().write_virtual_camera_manifest(root/'tracker/virtual_cameras.json',
        cameras, iteration=2000, circle_radius=radius)
    write(root/'tracker/official_camera_source.json', dict(complete=True, no_test_data_used=True,
        generator=record(OFFICIAL/'utils/pose_utils.py'), config=record(config),
        extrinsics=record(sparse/'images.bin'), intrinsics=record(sparse/'cameras.bin'),
        native_template=record(template), train_images=[c.name for c in train],
        circle_radius=radius, frame_count=30, image_size=[w,h], output=record(path),
        policy='Native inpaint cameras: circle radius from saved official config (4 decimal places), official eval training split, native template intrinsics and image dimensions.'))
    print(f'{name}: native circle30 radius={radius} size={w}x{h}', flush=True)


def create():
    import yaml
    stamp = time.strftime('%Y%m%d_%H%M%S')
    job = REPO / 'logs' / ('paintmesh_circle30_d10_gpu1_' + stamp)
    job.mkdir(); backup = job/'previous_report'; backup.mkdir()
    for name in ('comparison.md','comparison.json','report_corrections.json','request.json','status.json'):
        shutil.copy2(REPORT/name, backup/name)
    old = read(REPORT/'request.json'); corrections = read(REPORT/'report_corrections.json')['records']
    name = 'official_circle30_d10_' + stamp
    scenes = []
    for previous in old['scenes']:
        scene = previous['name']; source = Path(previous['inpaint_root'])
        for c in corrections:
            if c['method'] == 'PaintMesh 5000' and c['scene'] == scene:
                if sha(c['source']) != c['sha256']: raise ValueError('changed registered source')
                source = next(p for p in Path(c['source']).parents if (p/'manifests/edgs_joint_manifest.json').is_file())
        removal_source = next(p for p in source.parents if (p/'removal_manifest.json').is_file())
        config = copy.deepcopy(read(source/'manifests/edgs_joint_manifest.json')['request']['config'])
        source_joint = read(source/'manifests/edgs_joint_manifest.json')
        source_views = read(source_joint['inputs']['camera']['path'])['frame_count']
        previous_min_views = config['local_background']['min_views']
        # Preserve the old support fraction when replacing 180 views by 30;
        # still require at least two independent observations.
        config['local_background']['min_views'] = max(2, math.ceil(previous_min_views * 30 / source_views))
        config['iterations'] = 10000
        config['init']['max_points'] = None  # User explicitly removed the old 500,000 cap.
        assert config['init']['use_depth'] and not config['init']['use_normal']
        assert config['freeze_background'] and config['local_background']['enabled']
        assert config['pruning']['enabled'] and not config['adaptive_density']['enabled']
        assert config['surface_init']['mode'] == 'none' and config['appearance']['mode'] == 'masked'
        cfg = job/'configs'/f'{scene}.yaml'; cfg.parent.mkdir(exist_ok=True)
        cfg.write_text(yaml.safe_dump(config, sort_keys=False))
        root = REPO/'output/paintmesh/Inpaint360GS/inpaint360'/scene/'removal'/name
        root.mkdir(); (root/'work_model').mkdir()
        for entry in (removal_source/'work_model').iterdir():
            if entry.name != 'virtual': (root/'work_model'/entry.name).symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
        for entry in ('removal_manifest.json','removed_3dgs','removed_mesh'):
            target = removal_source/entry
            if target.exists(): (root/entry).symlink_to(target.resolve(), target_is_directory=target.is_dir())
        shutil.copytree(removal_source/'config', root/'config')
        work = OFFICIAL/'output/paper_r4_all_scenes/scenes'/scene/'workspace'
        for c in corrections:
            if c['method'] == 'Inpaint360GS' and c['scene'] == scene:
                work = next(p for p in Path(c['source']).parents if p.name == 'workspace')
        native_config = read(work/'config/object_inpaint/inpaint360'/f'{scene}.json')
        for p in (root/'config').rglob('*.json'):
            value = read(p)
            if 'circle_radius' in value: value['circle_radius'] = native_config['circle_radius']
            if 'finetune_iteration' in value: value['finetune_iteration'] = 10000
            write(p,value)
        item = dict(name=scene, target_ids=previous['target_ids'], mask_dilation=10,
            dilation_shape='square', circle_radius=native_config['circle_radius'],
            official_workspace=str(work.resolve()), source_inpaint=str(source), source_removal=str(removal_source),
            source_joint=record(source/'manifests/edgs_joint_manifest.json'), config=str(cfg),
            local_min_views_conversion=dict(previous=previous_min_views, previous_camera_count=source_views,
                current=config['local_background']['min_views'], current_camera_count=30),
            removal_root=str(root), inpaint_root=str(root/'inpaint'/name))
        scenes.append(item)
    request = dict(created_at=now(), gpu=1, run_name=name, report_job=str(REPORT), resolution=4,
        iterations=10000, save_iterations=[5000,10000], mask_dilation=10, dilation_shape='square',
        virtual_path='official circle', virtual_count=30, renderer='edgs-pgsr', recursive_guide=False,
        min_free_vram_mib=17500, scenes=scenes,
        inherited_policy='Per-scene loss/local update settings from currently reported PaintMesh 5000; min_views preserves support fraction (minimum 2). No initialization point cap. No posthoc best-result selection.',
        changes=['native circle30 camera calibration/poses','square mask dilation radius 10',
                 'fresh geometry-guided masks/LaMa/init; empty masks allowed for full occlusion',
                 '10000-step training with 5000/10000 evaluation for every scene'])
    write(job/'request.json',request)
    write(job/'status.json',dict(state='prepared', scenes={s['name']:dict(state='queued') for s in scenes}))
    print(job)
    return job


def publish_report(job):
    req = read(job/'request.json'); state = read(job/'status.json'); report = Path(req['report_job'])
    old = read(job/'previous_report/comparison.json')
    methods = {k: v for k,v in old['metrics'].items() if not k.startswith('PaintMesh')}
    corrections = [r for r in read(job/'previous_report/report_corrections.json')['records'] if not r['method'].startswith('PaintMesh')]
    for step in (5000,10000):
        label=f'PaintMesh {step}'; methods[label]={}
        for s in req['scenes']:
            model = Path(s['inpaint_root'])/('inpainted_3dgs_5000' if step==5000 else 'inpainted_3dgs')
            path = model/'evaluation/inpaint360_r4/results.json'
            if not path.exists(): continue
            result=read(path)
            if not result.get('complete') or result['inputs']['iteration'] != step: continue
            values=result['metrics']
            if not all(math.isfinite(values[m]) for m in METRICS): raise ValueError(f'Invalid evaluation: {path}')
            methods[label][s['name']]=values
            corrections.append(dict(method=label,scene=s['name'],source=str(path),sha256=sha(path),
                format='paintmesh',iteration=step,label='circle30 / d10方形',
                reason='官方圆轨迹与方形膨胀配置重跑；直接替换本次结果，不按 PSNR 择优'))
    counts=old['test_frame_counts']; common=set.intersection(*(set(v) for v in methods.values()))
    def mean(rows, names, weighted):
        if not names:return None
        return {m:sum(rows[s][m]*(counts[s] if weighted and m!='FID' else 1) for s in names)/
            sum(counts[s] if weighted and m!='FID' else 1 for s in names) for m in METRICS}
    agg={m:dict(available_scenes=sorted(rows),all_available_scene_macro=mean(rows,set(rows),False),
        common_scene_macro=mean(rows,common,False),common_official_code_average=mean(rows,common,True)) for m,rows in methods.items()}
    summary=dict(updated_at=now(),status=state,active_rerun=str(job),request=req,metrics=methods,
        original_metrics=old['metrics'],test_frame_counts=counts,common_scenes=sorted(common),aggregates=agg,
        report_corrections=corrections,baseline_sources=old.get('baseline_sources',{}))
    write(report/'comparison.json',summary)
    write(report/'report_corrections.json',dict(updated_at=now(),records=corrections,active_rerun=str(job)))
    lines=['# PaintMesh 全场景 1/4 分辨率实验','',f"更新：{now()}；状态：{state['state']}；当前：{state.get('scene','—')} / {state.get('phase','—')}",'',
        '本轮统一采用 Inpaint360GS 官方 circle 30 帧相机配置（逐场景原生半径、位姿、内参与图像尺寸），MASK_DILATION=10，21×21 方形核。PGSR 渲染、原始外观+几何监督、depth=true、normal=false、surface_init=none、局部旧点更新/远处冻结、仅剪枝、RECURSIVE_GUIDE=false；初始化总点数无上限。',
        '训练监督与局部更新参数继承上一轮逐场景配置（含已登记修复）；局部更新支持帧数按视角数量换算且至少为 2（原 18/180 改为 3/30，保持 10%）。统一训练 10000 步并保存 5000 快照。逐场景配置与差异源保存在 request.json 和 configs/。本轮结果全部替换，未完成场景标记待完成，不与旧轨迹结果混合平均。',
        '训练/相机/跟踪不使用测试 GT。删除目标沿用已修正的对象选择；重新渲染、跟踪、LaMa、初始化、训练、评估、提取 mesh。完全遮挡帧允许空 mask。',
        '其他三种方法列保留当前复现结果，包括已登记的目标/跟踪修复；AuraFusion360/redbull 含初始化修正版，不能视为未经修改的官方代码。','',
        f"[本轮配置与状态]({os.path.relpath(job,report)}/request.json) · [运行状态]({os.path.relpath(job,report)}/status.json) · [原报告备份]({os.path.relpath(job,report)}/previous_report/comparison.md)",'',
        '| 场景 | ID | MASK_DILATION | circle 半径 | 状态 |','|---|---|---:|---:|---|']
    for s in req['scenes']:
        st=state['scenes'].get(s['name'],{})
        lines.append(f"| {s['name']} | {s['target_ids']} | 10（方形） | {s['circle_radius']:.4f} | {st.get('state','queued')} / {st.get('phase','—')} |")
    lines+=['','## 每场景指标','','| 场景 | 方法 | '+' | '.join(METRICS)+' |','|---|---|'+'---:|'*len(METRICS)]
    for s in req['scenes']:
        for method,rows in methods.items():
            v=rows.get(s['name'])
            lines.append(f"| {s['name']} | {method} | "+' | '.join(f'{v[m]:.6f}' if v else '待完成' for m in METRICS)+' |')
    lines+=['','## 相同场景均值','',f"共同完成场景（{len(common)}/11）：{', '.join(sorted(common)) or '尚无'}。",
        '前六项按测试帧数加权，FID 按场景平均；不做 pooled FID。场景宏平均另存 comparison.json。','',
        '| 方法 | '+' | '.join(METRICS)+' |','|---|'+'---:|'*len(METRICS)]
    for method,values in agg.items():
        row=values['common_official_code_average']
        lines.append(f'| {method} | '+' | '.join(f'{row[m]:.6f}' if row else '待完成' for m in METRICS)+' |')
    lines+=['','此前 qualitative_* 对比图属于旧 adaptive180 实验，本轮不引用其作为 circle30 结果。']
    atomic(report/'comparison.md','\n'.join(lines)+'\n')


class Runner:
    def __init__(self,job):
        self.job=job;self.req=read(job/'request.json');self.status=read(job/'status.json');self.child=None
        raw=subprocess.check_output(['bash','-c','source "$1"; env -0','bash',str(REPO/'scripts/paintmesh/configs/inpaint360_recommended.env')])
        self.env=dict(x.decode().split('=',1) for x in raw.split(b'\0') if b'=' in x)
        self.env.update(GPU='1',CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',MASK_DILATION='10',
            MASK_DILATION_SHAPE='square',ALLOW_EMPTY_MASKS='true',VIRTUAL_CAMERA_PATH='circle',
            VIRTUAL_CAMERA_COUNT='30',RECURSIVE_GUIDE='false',EDGS_JOINT_SAVE_ITERATIONS='5000')
        self.env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', DATA_ROOT=str(DATA.parent.parent),
                        CONDA_BIN=shutil.which('conda') or '/home/member/miniconda3/bin/conda')
        self.env.pop('PYTHONPATH',None)
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)

    def stop(self,*args):
        if self.child and self.child.poll() is None:os.killpg(self.child.pid,signal.SIGTERM)
        self.status['state']='interrupted';self.save();raise InterruptedError('runner stopped')

    def save(self):
        self.status.update(updated_at=now(),pid=os.getpid(),gpu=1)
        write(self.job/'status.json',self.status);publish_report(self.job)

    def command(self,s,phase,args,extra=None):
        self.status.update(scene=s['name'],phase=phase)
        self.status['scenes'][s['name']].update(state='running',phase=phase);self.save()
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
            if free>=self.req['min_free_vram_mib']:break
            self.status['phase']=f'waiting for GPU1 memory before {phase} ({free} MiB free)';self.save();time.sleep(30)
        self.status['phase']=phase
        env=dict(self.env);env.update(REMOVAL_ROOT=s['removal_root'],INPAINT_RUN_ROOT=s['inpaint_root'],
            INPAINT_RUN_NAME=self.req['run_name'],EDGS_INPAINT_CONFIG=s['config'])
        if extra:env.update(extra)
        log=self.job/f"{s['name']}_{phase}.log"
        print(f'[{now()}] {s["name"]} {phase}: {shlex.join(map(str,args))}',flush=True)
        with log.open('ab',buffering=0) as out:
            self.child=subprocess.Popen(list(map(str,args)),cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            self.status.update(child_pid=self.child.pid,current_log=str(log));self.save()
            code=self.child.wait()
        self.child=None;self.status.pop('child_pid',None)
        if code:raise RuntimeError(f'{phase} exited {code}; see {log}')

    def tracking_session(self,s):
        from virtual_render_io import read_camera_manifest,validate_render
        root=Path(s['removal_root']);path=root/'tracker/tracking_session.json'
        if path.exists():return
        camera=root/'tracker/virtual_cameras.json'; cams=read_camera_manifest(camera)
        render=root/'work_model/virtual/ours_object_removal/iteration_2000/render_manifest.json'
        rendered=validate_render(render.parent)
        write(path,dict(schema_version=2,kind='paintmesh-tracking-session',complete=False,status='in_progress',
            started_at=now(),started_at_ns=time.time_ns(),input_archive=record(root/'tracker/images.zip'),
            input_cameras=dict(record(camera),artifact_id=cams['artifact_id']),
            expected_masks=[c['image_name']+'.png' for c in cams['cameras']],
            input_virtual_render=dict(path=str(render.resolve()),sha256=sha(render),artifact_id=rendered['artifact_id'],backend=rendered['backend'])))

    def commit_masks(self,s):
        import numpy as np
        from PIL import Image
        root=Path(s['removal_root']);p=root/'tracker/tracking_session.json';session=read(p)
        if session['input_archive']!=record(root/'tracker/images.zip'):raise ValueError('archive changed')
        camera=root/'tracker/virtual_cameras.json';cams=read(camera)
        if session['input_cameras']!=dict(record(camera),artifact_id=cams['artifact_id']):raise ValueError('cameras changed')
        masks=root/'tracker/results/images/images_masks';records=[];total=0
        for c in cams['cameras']:
            name=c['image_name']+'.png'; path=masks/name;r=record(path)
            if r['mtime_ns']<session['started_at_ns']:raise ValueError('stale masks')
            with Image.open(path) as im:
                a=np.array(im)
                if im.size!=(c['image_width'],c['image_height']) or a.ndim!=2 or (a!=0).all():raise ValueError('invalid mask')
                total+=int((a!=0).sum())
            records.append(dict(file=name,**{k:r[k] for k in ('size_bytes','mtime_ns','sha256')}))
        if not total:raise ValueError('no visible holes')
        session.update(complete=True,status='complete',completed_at=now(),outputs=dict(masks=records))
        write(p,session)

    def run(self):
        from run_inpaint360_full import export_snapshot
        self.status.update(state='running',started_at=self.status.get('started_at',now()));self.save()
        for s in self.req['scenes']:
            entry=self.status['scenes'][s['name']]
            if entry['state']=='complete':continue
            root=Path(s['removal_root']); inp=Path(s['inpaint_root'])
            try:
                if shutil.disk_usage(REPO).free<30*1024**3:raise RuntimeError('less than 30 GiB free disk')
                pair=root/'work_model/virtual/virtual_render_manifest.json'
                validated=['--validate-only'] if pair.is_file() and read(pair).get('complete') else []
                self.command(s,'virtual_render',[PYTHON,'scripts/paintmesh/render_virtual_views.py',
                    '--backend','edgs-pgsr','--model-path',root/'work_model',
                    '--edgs-model-path',REPO/'output/paintmesh/Inpaint360GS/inpaint360'/s['name']/'edgs',
                    '--source-path',DATA/s['name'],'--iteration','2000','--resolution','4','--images','images',
                    '--alpha-min','.01','--camera-manifest',root/'tracker/virtual_cameras.json',
                    '--tracker-archive',root/'tracker/images.zip']+validated)
                self.tracking_session(s)
                if not (root/'tracker/results/images/images_masks').exists():
                    self.command(s,'tracking',[PYTHON,'scripts/paintmesh/track_holes_headless.py',
                        '--removal-root',root,'--allow-empty-masks'])
                self.commit_masks(s)
                cmd=['bash','scripts/paintmesh/run_inpaint.sh','Inpaint360GS/inpaint360',s['name'],'4',
                    ','.join(map(str,s['target_ids'])),'none']
                self.command(s,'inpaint_train',cmd+['1'],dict(END_STAGE='5'))
                model=export_snapshot(inp,5000)
                self.command(s,'eval_5000',[PYTHON,'scripts/paintmesh/eval_inpaint360.py','run',
                    '--model-path',model,'--source-path',DATA/s['name'],'--resolution','4','--iteration','5000','--batch-size','4'])
                self.save()
                self.command(s,'publish_10000',cmd+['6'],dict(END_STAGE='6'))
                self.command(s,'eval_10000',[PYTHON,'scripts/paintmesh/eval_inpaint360.py','run',
                    '--model-path',inp/'inpainted_3dgs','--source-path',DATA/s['name'],
                    '--resolution','4','--iteration','10000','--batch-size','4'])
                self.save()
                self.command(s,'mesh',cmd+['7'],dict(END_STAGE='8'))
                entry.update(state='complete',completed_at=now());entry.pop('error',None)
            except InterruptedError:raise
            except Exception as exc:
                entry.update(state='failed',error=str(exc),failed_at=now());print(traceback.format_exc(),flush=True)
            self.save()
        failed=[k for k,v in self.status['scenes'].items() if v['state']!='complete']
        self.status.update(state='complete' if not failed else 'finished_with_errors',phase='finished',
            failed_scenes=failed,finished_at=now());self.save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--create',action='store_true');p.add_argument('--job',type=Path)
    p.add_argument('--export-cameras',action='store_true');p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    if a.create: create();return
    job=a.job.resolve(strict=True)
    if a.export_cameras:
        for s in read(job/'request.json')['scenes']:export_cameras(s)
        return
    if a.report_only:publish_report(job);return
    with (job/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Runner(job).run()


if __name__=='__main__':main()
