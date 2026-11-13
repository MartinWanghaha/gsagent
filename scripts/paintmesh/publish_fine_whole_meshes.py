#!/usr/bin/env python3
"""Publish complete fine meshes and refresh native-mesh reports in place."""
from pathlib import Path
import argparse
import copy
import fcntl
import gc
import os
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,METHODS,STATES,read,write,identity,digest
from mesh_benchmark.layout import work_root,configure_sources
from recompose.io import fingerprint
from recompose.mesh_export import export_mesh_scene

PUBLIC=BASE/'mesh_comparison_whole';ROOT=work_root(PUBLIC);WORK=ROOT/'fine_whole'


def mesh_refs(receipt):
    return [fingerprint(s['path']) for s in receipt['mesh_specs']]


def publish_scene(scene):
    done=WORK/'publication'/scene/'complete.json'
    fine={k:read(WORK/k/scene/'complete.json') for k in ['stage1','background']}
    graph=read(PUBLIC/'recomposition'/scene/'composition/scene.json')
    from mesh_benchmark.extend_fine import extend
    original_stage1=Path(graph['input_selection']['original_mesh'])
    gaussian=Path(graph['background']['gaussians']['path']);iteration=int(gaussian.parent.name.split('_')[-1])
    original_background=gaussian.parents[2]/'mesh'/f'ours_{iteration}'/'tsdf_fusion_post.ply'
    fine['stage1']=extend(original_stage1,fine['stage1'],WORK/'extensions/stage1'/scene)
    fine['background']=extend(original_background,fine['background'],WORK/'extensions/background'/scene)
    request={k:digest(r['request']) for k,r in fine.items()}
    if done.exists():
        prior=read(done)
        if prior['request']!=request:raise ValueError('Published fine inputs changed')
        return
    old=PUBLIC/'recomposition'/scene/'composition/scene.json';document=read(old)
    archive=WORK/'before'/scene;archive.mkdir(parents=True,exist_ok=True)
    if not (archive/'snapshot_complete.json').exists():
        shutil.copy2(old,archive/'scene.json')
        for state in STATES:
            for suffix in ['_support','_native_support']:
                source=ROOT/'common'/scene/(state+suffix)
                if source.exists():shutil.copytree(source,archive/'common'/(state+suffix),dirs_exist_ok=True)
            for method in METHODS:
                source=ROOT/state/method/scene/'metrics.json'
                if source.exists():
                    dest=archive/'metrics'/state/method;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest/'metrics.json')
        for name in ['all_methods.png','background.png','recomposition.png']:
            shutil.copy2(PUBLIC/'figures_all_methods/scenes'/scene/name,archive/name)
        write(archive/'snapshot_complete.json',dict(complete=True,updated=time.time()))
    refs={k:mesh_refs(r) for k,r in fine.items()}
    first=copy.deepcopy(document);first['objects']=[]
    first['background']['mesh']=refs['stage1'][0];first['background']['patches']=refs['stage1'][1:]
    first['background']['gaussians']=fingerprint(read(WORK/'stage1'/scene/'frames/complete.json')['request']['checkpoint']['path'])
    first.update(status='complete',source_mesh_kind='Full-scene first-stage checkpoint; fixed .002/.008 TSDF, all components')
    stage1_world=PUBLIC/'meshes'/scene/'stage1_world'
    export_mesh_scene(first,stage1_world)
    write(stage1_world/'scene.json',first)
    document['stage1_whole']=dict(complete=True,mesh_specs=fine['stage1']['mesh_specs'],
        source_checkpoint=read(WORK/'stage1'/scene/'frames/complete.json')['request']['checkpoint'],
        extraction=str(WORK/'extensions/stage1'/scene/'complete.json'),world=str(stage1_world/'mesh_scene.gltf'),
        original_truncated_mesh=document['input_selection']['original_mesh'],
        voxel_size=.002,sdf_trunc=.008,all_connected_components_retained=True,
        original_mesh_preserved_exact=True,original_mesh=identity(original_stage1),
        added_faces=fine['stage1']['added_faces'],overlap_faces_removed=fine['stage1']['redundant_new_faces_removed'])
    document['background']['mesh']=refs['background'][0];document['background']['patches']=refs['background'][1:]
    document['input_selection']['background_mesh']=refs['background'][0]['path']
    document['fine_whole_extraction']=dict(complete=True,voxel_size=.002,sdf_trunc=.008,
        extraction=str(WORK/'extensions/background'/scene/'complete.json'),source_checkpoint=read(WORK/'background'/scene/'frames/complete.json')['request']['checkpoint'],
        all_connected_components_retained=True,coarse_background_substitution=False,mesh_decimation=False,
        original_mesh_preserved_exact=True,original_mesh=identity(original_background),
        added_faces=fine['background']['added_faces'],overlap_faces_removed=fine['background']['redundant_new_faces_removed'])
    export_mesh_scene(document,old.parent/'world')
    pure=copy.deepcopy(document);pure['objects']=[];export_mesh_scene(pure,PUBLIC/'meshes'/scene/'world')
    write(old,document)
    write(PUBLIC/'meshes'/scene/'fine_mesh_manifest.json',dict(complete=True,stage1=document['stage1_whole'],
        background=document['fine_whole_extraction'],background_meshes=refs['background'],stage1_meshes=refs['stage1'],
        note='The glTF contains every original triangle partition. No coarse-background splice or mesh decimation.'))
    # The historical stage1.ply link is kept explicitly as a local artifact;
    # the complete asset is stage1_world/mesh_scene.gltf and all source parts.
    legacy=PUBLIC/'meshes'/scene/'stage1.ply'
    if legacy.is_symlink():legacy.rename(legacy.with_name('stage1_original_truncated.ply'))
    coarse=PUBLIC/'meshes'/scene/'background.ply'
    if coarse.exists() and not (archive/'background_coarse.ply').exists():coarse.rename(archive/'background_coarse.ply')
    (PUBLIC/'meshes'/scene/'README.md').write_text(f'# {scene} 完整细网格\n\n'
        '- [第一阶段：完整背景](stage1_world/mesh_scene.gltf)\n'
        '- [补全后的完整背景](world/mesh_scene.gltf)\n'
        f'- [背景＋生成物体](../../recomposition/{scene}/composition/world/mesh_scene.gltf)\n'
        '- [各原生三角网格分区与检查点来源](fine_mesh_manifest.json)\n\n'
        'glTF直接引用全部原生三角网格分区，保留完整几何和颜色。stage1_original_truncated.ply是历史截断产物；粗背景网格已移入内部归档，不再作为当前结果。\n')
    info_path=ROOT/'common'/scene/'complete.json';info=read(info_path);old_id=info['request_id']
    info['request']['selection']=identity(old);info['request_id']=digest(info['request']);write(info_path,info)
    for method in METHODS:
        for state in STATES:
            cache=ROOT/'cache'/method/scene/state
            if method in ['paintmesh','paintmesh_stage1']:
                if cache.exists():
                    backup=archive/'cache'/method/state;backup.parent.mkdir(parents=True,exist_ok=True)
                    if not backup.exists():cache.rename(backup)
            else:
                marker=cache/'complete.json';meta=read(marker)
                assert meta['request']['common']==old_id
                meta['request']['common']=info['request_id'];write(marker,meta)
                geo=cache/'geometry_metrics.json'
                if geo.exists():geo.unlink()
            metric=ROOT/state/method/scene/'metrics.json'
            if metric.exists():metric.unlink()
    for state in STATES:
        for suffix in ['_support','_native_support']:
            folder=ROOT/'common'/scene/(state+suffix)
            if folder.exists():shutil.rmtree(folder)
    protocol=read(ROOT/'protocol.json')
    protocol.setdefault('fine_whole',dict(enabled=True,voxel=.002,sdf_trunc=.008,scenes={},
        all_original_cameras=True,all_depth_pixels=True,no_coarsening=True,no_decimation=True,
        stage1='Own first-stage 30000-step checkpoint; full-scene extraction, all components'))
    protocol['fine_whole']['scenes'][scene]=dict(complete=True,stage1=str(WORK/'stage1'/scene/'complete.json'),
        background=str(WORK/'background'/scene/'complete.json'))
    protocol['paintmesh_stage1'].update(source='Full-scene fine TSDF from actual first-stage checkpoint',geometry_unchanged=False,
        training_performed=False,all_connected_components_retained=True)
    protocol['paintmesh_stage1'].setdefault('fine_sources',{})[scene]=document['stage1_whole']
    write(ROOT/'protocol.json',protocol)
    write(done,dict(complete=True,request=request,updated=time.time(),common_id=info['request_id']))


def prepare_one(method,scene,cfg):
    configure_sources(ROOT)
    from mesh_benchmark.cache import prepare
    prepare(ROOT,method,scene,cfg)


def evaluate_scene(scene):
    from mesh_benchmark.overlap import prepare_support
    configure_sources(ROOT);cfg=read(ROOT/'protocol.json')['config']
    available=next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available>=40:
        import multiprocessing
        with multiprocessing.get_context('spawn').Pool(2) as pool:
            pool.starmap(prepare_one,[(m,scene,cfg) for m in ['paintmesh_stage1','paintmesh']])
    else:
        for method in ['paintmesh_stage1','paintmesh']:prepare_one(method,scene,cfg)
    prepare_support(ROOT,scene,METHODS,cfg)


def delivered_scenes(including=()):
    return [s for s in SCENES if s in including or (
        (WORK/'publication'/s/'evaluated.json').exists() and read(WORK/'publication'/s/'evaluated.json').get('complete'))]


def scene_delivery_receipt(scene):
    """Reject stale figures or scores before acknowledging scene completion."""
    from plot_mesh_method_comparison import input_signature,choose_view
    figure=PUBLIC/'figures_all_methods/scenes'/scene
    manifest=read(figure/'figure_manifest.json');camera,_,_=choose_view(ROOT,scene)
    signature=input_signature(ROOT,scene,camera)
    if manifest['signature']!=signature:raise ValueError(f'{scene}: figure inputs have changed since drawing')
    if any(r['method'] and r['status']!='complete' for r in manifest['records']):
        raise ValueError(f'{scene}: comparison contains unfinished methods')
    metrics={}
    for state in STATES:
        support=read(ROOT/'common'/scene/(state+'_native_support')/'complete.json')
        for method in METHODS:
            path=ROOT/state/method/scene/'metrics.json';metric=read(path)
            geometry=read(ROOT/'cache'/method/scene/state/'geometry_metrics.json')
            if not metric.get('complete') or metric['request']['support_id']!=support['request_id'] or metric['geometry']!=geometry:
                raise ValueError(f'{scene}/{method}/{state}: missing or stale scores')
            metrics[state+'/'+method]=identity(path)
    assets=[PUBLIC/'meshes'/scene/'stage1_world/mesh_scene.gltf',PUBLIC/'meshes'/scene/'world/mesh_scene.gltf',
            PUBLIC/'recomposition'/scene/'composition/world/mesh_scene.gltf']
    pictures=[figure/(name+'.png') for name in ['all_methods','background','recomposition']]
    pictures += [PUBLIC/'figures_all_methods/geometry_latest'/(scene+'_all_methods.png')]
    return dict(complete=True,scene=scene,updated=time.time(),camera=camera['name'],figure_signature=signature,
        models=[identity(p) for p in assets],metrics=metrics,figures=[identity(p) for p in pictures],
        reports=[identity(PUBLIC/'reports'/(s+'.md')) for s in STATES],
        native_triangles=True,additional_image_downsampling=False,mesh_decimation=False,
        source_stage1_mesh_preserved=True)


def finish_scenes(scenes):
    subprocess.run([sys.executable,'-u',str(HERE/'evaluate_native_mesh_rgb.py'),'--output',str(PUBLIC),
                    '--scenes',','.join(scenes)],check=True)
    from publish_latest_mesh_geometry import figures
    figures(PUBLIC,scenes)
    from verify_native_mesh_delivery import verify
    verify(PUBLIC)
    from compare_fine_mesh_effects import compare
    compare(scenes)
    receipts={scene:scene_delivery_receipt(scene) for scene in scenes}
    delivered=delivered_scenes(scenes)
    doc=['# 完整背景细网格更新','',
        '第一阶段和背景各自完整保留原有局部细网格的全部三角形和颜色；各自检查点提取完整范围的细TSDF扩展，固定体素0.002、带宽0.008。新增网格仅去除被原有表面覆盖的重叠面，原网格文件不改变。全部既定融合相机和原生有效深度像素参与，不做减面或粗背景拼接。','',
        '每场景 `meshes/<scene>/stage1_world/mesh_scene.gltf` 为有完整可观测背景的第一阶段网格；`world/mesh_scene.gltf` 为补全背景。组合场景保留原有MV-SAM3D物体位姿。源高斯没有重训。','',
        '原先截断mesh为 `stage1_original_truncated.ply`，仅作历史核对；原生场景由多个完整三角网格分区组成，glTF统一加载全部分区，不含粗背景替代。','',
        '几何参考仍为COLMAP观测点，主表为六方法共同表面/像素，纯背景排除目标mask。增加覆盖后共同域改变，所有方法重新评分；不将数值变化全归为质量提升。','',
        f'逐场景完成渲染、指标、对比图与核验后立即发布。目前完成 {len(delivered)}/11 场景；其余场景的旧图保留到各自更新完成。','',
        '| 场景 | 第一阶段完整mesh | 背景mesh | 最新对比图 |','|---|---|---|---|']
    for scene in delivered:doc.append(f'| {scene} | [glTF](../meshes/{scene}/stage1_world/mesh_scene.gltf) | [glTF](../meshes/{scene}/world/mesh_scene.gltf) | [PNG](../figures_all_methods/scenes/{scene}/all_methods.png) |')
    doc+=['','[纯背景指标](../reports/background.md) · [组合指标](../reports/recomposition.md) · [八列图片](../figures_all_methods/README.md) · [同像素新旧核对](FINE_WHOLE_EFFECTS.md)']
    (PUBLIC/'docs/FINE_WHOLE_RESULTS.md').write_text('\n'.join(doc)+'\n')
    completed=list(read(ROOT/'protocol.json')['fine_whole']['scenes'])
    p=PUBLIC/'meshes/README.md';entries=[]
    for scene in SCENES:
        if scene in completed:entries.append(f'- [{scene}：第一阶段、背景和组合细网格]({scene}/README.md)')
        else:entries.append(f'- [{scene}：上一轮背景mesh（细网格待完成）]({scene}/background.ply)')
    p.write_text('# 完整可观测背景与第一阶段细网格\n\n'+'\n'.join(entries)+'\n\n已更新场景的glTF保留全部原生三角网格分区，不进行减面或粗背景拼接。\n')
    write(PUBLIC/'docs/QUALITY_PRESERVATION.json',dict(complete=len(delivered)==len(SCENES),updated=time.time(),
        updated_scenes=delivered,asset_ready_scenes=completed,source_gaussians_retrained=False,source_stage1_meshes_modified=False,new_meshes_extracted=True,
        voxel_size=.002,sdf_trunc=.008,additional_image_downsampling=False,mesh_decimation=False,
        held_out_views_dropped=False,held_out_view_count=sum(read(ROOT/'common'/s/'complete.json')['held_out_count'] for s in SCENES),
        geometry_thresholds_unchanged=True,all_connected_components_retained=True,automatic_tsdf_coarsening_disabled=True,
        historical_whole_mesh_quality_reductions_reversed=len(delivered)==len(SCENES)))
    receipt=PUBLIC/'docs/STAGE1_PUBLICATION_CHECK.json'
    if receipt.exists():
        current=read(receipt);current.update(updated=time.time(),fine_whole_updated_scenes=completed,
            first_stage_asset='meshes/<scene>/stage1_world/mesh_scene.gltf for updated scenes; original truncated PLY for others',
            current_local_link_scan_pending=True)
        write(receipt,current)
    for name in ['PAINTMESH_COLOR_BLUR.md','PAINTMESH_BLUR_AUDIT.json']:
        path=PUBLIC/'docs'/name
        if name.endswith('.md'):
            text=path.read_text();note='**本文记录细体素重建前的历史粗网格诊断；当前完整背景细网格结果见 [最新更新](FINE_WHOLE_RESULTS.md)。**\n\n'
            if note not in text:path.write_text(note+text)
    for name in ['README.md','figures_all_methods/README.md']:
        import re
        p=PUBLIC/name;text=p.read_text()
        text=re.sub(r'^[^\n]*的新模型已导出：[^\n]*指标和图片仍在重算。\n\n','',text,flags=re.MULTILINE)
        text=re.sub(r'^\*\*完整背景细网格正在重新提取。.*?\n\n','',text,flags=re.MULTILINE)
        text=re.sub(r'^\*\*完整细网格更新：.*?\n\n','',text,flags=re.MULTILINE)
        note=f'**完整细网格更新：{len(delivered)}/11场景（{", ".join(delivered)}）；每完成一个场景立即更新对比图和指标。** [资产、参数与评估说明]('+('docs/' if name=='README.md' else '../docs/')+'FINE_WHOLE_RESULTS.md)。\n\n'
        p.write_text(note+text)
    p=PUBLIC/'figures_all_methods/paintmesh_blur/README.md'
    note='**此目录保留细体素重建前的模糊定位证据；最新完整细网格图片见 [当前八列对比](../README.md)。**\n\n'
    if note not in p.read_text():p.write_text(note+p.read_text())
    # Completion is the last action: models, scores, images, reports, and their
    # independent audit must all exist before the supervisor advances.
    for scene in scenes:
        receipt=receipts[scene]
        path=PUBLIC/'figures_all_methods/scenes'/scene/'delivery.json';write(path,receipt)
        write(WORK/'publication'/scene/'evaluated.json',dict(complete=True,updated=time.time(),
            delivery=str(path),figure_signature=receipt['figure_signature']))
    write(WORK/'publication_complete.json',dict(complete=len(delivered)==len(SCENES),updated=time.time(),scenes=delivered))
    print('FINE_WHOLE_PUBLICATION_COMPLETE',scenes,flush=True)


def run(scenes):
    from evaluate_native_mesh_rgb import selected_scenes
    scenes=selected_scenes(scenes)
    configure_sources(ROOT);WORK.mkdir(parents=True,exist_ok=True)
    lock=open(WORK/'publication.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    doc=PUBLIC/'docs/FINE_WHOLE_RESULTS.md'
    if not doc.exists():doc.write_text('# 完整背景细网格更新\n\n固定体素0.002、TSDF带宽0.008，全部连通块保留。逐场景发布和重评估进行中，详见 [进度](FINE_WHOLE_STATUS.md)。\n')
    for scene in scenes:
        marker=WORK/'publication'/scene/'evaluated.json'
        if marker.exists() and read(marker).get('complete'):
            # The final all-scene pass verifies deliveries without re-rendering
            # or rewriting already completed scene scores and images.
            receipt=scene_delivery_receipt(scene)
            write(PUBLIC/'figures_all_methods/scenes'/scene/'delivery.json',receipt)
            continue
        publish_scene(scene);evaluate_scene(scene);finish_scenes([scene])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scenes',default=','.join(SCENES))
    run(parser.parse_args().scenes.split(','))
