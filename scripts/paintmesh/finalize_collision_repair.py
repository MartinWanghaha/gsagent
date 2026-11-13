#!/usr/bin/env python3
"""Bind exact post-repair collision proofs to published assets and refresh reports."""
from pathlib import Path
import argparse
import copy
import hashlib
import os
import shutil
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,read,write,identity

PUBLIC=BASE/'mesh_comparison_whole'


def link_file(source,destination):
    source=Path(source).resolve(strict=True);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source==destination.resolve():return
    temporary=destination.with_name('.'+destination.name+'.new')
    temporary.unlink(missing_ok=True)
    # Immutable detection buffers may be large. Reuse the same physical bytes;
    # neither model reduction nor another collision computation is performed.
    os.link(source,temporary)
    os.replace(temporary,destination)


def bind_collision_proof(public,scene):
    root=public/'collision_refinement';destination=public/'collision_audit/scenes'/scene
    update=read(root/'final_updates'/scene/'publication_updates.json')
    graph_path=public/'recomposition'/scene/'composition/scene.json';graph=read(graph_path)
    proofs=[Path(r['full_mesh_collision']['path']) for r in update['validation']['objects'].values()]
    folders={p.parents[2] for p in proofs}
    if len(folders)!=1:raise ValueError('Object proofs must belong to the same final scene check')
    source=folders.pop();prepared=read(source/'candidates.json');audit=read(source/'audit.json')
    inputs=prepared['request'];objects={o['id']:o for o in prepared['objects']}
    expected=[graph['background']['mesh']['path']]+[p['path'] for p in graph['background'].get('patches',[])]
    if expected!=[r['source']['path'] for r in inputs['environment_sources']]:
        raise ValueError('Published environment differs from exact final collision audit')
    for record in inputs['environment_sources']:
        if identity(record['source']['path'])!=record['source']:raise ValueError('Environment source changed')
        for key in ['vertices','indices']:
            if key in record and identity(record[key]['path'])!=record[key]:raise ValueError('Environment buffer changed')
    for node in graph['objects']:
        oid=node['id'];row=objects[oid]
        if identity(node['mesh']['path'])!=row['source'] or node['object_to_world']!=row['pose']:
            raise ValueError('Published object geometry/pose differs from final collision proof')
        with np.load(node['mesh']['path']) as a:
            v=a['vertices'];f=a['faces'];T=np.asarray(node['object_to_world']);world=v@T[:3,:3].T+T[:3,3]
        with np.load(source/'objects'/oid/'mesh_world.npz') as a:
            if not np.array_equal(world,a['vertices']) or not np.array_equal(f,a['faces']):
                raise ValueError('Collision world triangle buffers are stale')
    # Bind the already-completed exact computation to the now-published graph.
    # Keep its original provenance as a separate field; never pretend to rerun it.
    destination.mkdir(parents=True,exist_ok=True)
    link_file(source/'background_roi.npz',destination/'background_roi.npz')
    final_request=copy.deepcopy(inputs);final_request['scene_graph']=identity(graph_path)
    for obj in prepared['objects']:
        oid=obj['id'];node=next(n for n in graph['objects'] if n['id']==oid)
        for name in ['mesh_world.npz','intersections.npz','inside_environment_vertices.npz']:
            path=source/'objects'/oid/name
            if path.exists():link_file(path,destination/'objects'/oid/name)
            else:(destination/'objects'/oid/name).unlink(missing_ok=True)
        obj['mesh_world']=str(destination/'objects'/oid/'mesh_world.npz')
        obj['pose_report']=identity(node['pose_report']['path'])
    final_request['objects']=prepared['objects']
    prepared.update(request=final_request,roi=identity(destination/'background_roi.npz'),
        publication_complete=graph['complete'],publication_status=graph['status'],
        exact_computation_source=identity(source/'audit.json'),published_geometry_and_pose_verified=True)
    write(destination/'candidates.json',prepared)
    audit.update(publication_complete=graph['complete'],publication_status=graph['status'],
        exact_computation_source=identity(source/'audit.json'),published_geometry_and_pose_verified=True,
        exact_computation_reused=True,updated=time.time())
    audit['request']=dict(candidates=identity(destination/'candidates.json'),
        narrow_phase=audit['request']['narrow_phase'])
    for row in audit['objects']:
        oid=row['id'];old_path=source/'objects'/oid/'audit.json';data=read(old_path)
        data['published_geometry_and_pose_verified']=True
        data['exact_computation_source']=identity(old_path)
        write(destination/'objects'/oid/'audit.json',data)
    write(destination/'audit.json',audit)
    return audit


def numeric_geometry(value):
    if isinstance(value,dict):return {k:numeric_geometry(v) for k,v in value.items() if k not in ['support_id','request_id','updated']}
    if isinstance(value,list):return [numeric_geometry(v) for v in value]
    return value


def finalize(public=PUBLIC,verified_figures_resume=False):
    public=Path(public);root=public/'collision_refinement';before=read(root/'before_numeric_and_asset_checks.json')
    rows=[];integrity=[]
    for scene in SCENES:
        audit=bind_collision_proof(public,scene)
        update=read(root/'final_updates'/scene/'publication_updates.json')
        # Source image metrics can change through a common support-domain change;
        # display this fact rather than claiming a matched-domain improvement.
        metric=read(public/'_internal/workspace/recomposition/paintmesh'/scene/'metrics.json')
        bg=read(public/'_internal/workspace/background/paintmesh'/scene/'metrics.json')
        original=before['scenes'][scene]
        if original['background']['mean']!=bg['mean'] or original['background']['per_view']!=bg['per_view']:
            raise AssertionError('Background numerical scores changed')
        if numeric_geometry(original['background']['geometry'])!=numeric_geometry(bg['geometry']):
            raise AssertionError('Background geometry metrics changed')
        for ref in original['gaussian_sources']:
            path=Path(ref['path']);stat=path.stat()
            if stat.st_size!=ref['size'] or stat.st_mtime_ns!=ref['mtime_ns']:
                raise AssertionError('Original Gaussian source changed')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:
                raise AssertionError('Original Gaussian content changed')
        integrity.append(dict(scene=scene,background_numeric_scores_unchanged=True,
            original_gaussian_sha256_unchanged=True,geometry_and_pose_equal_exact_collision_proof=True))
        for obj in audit['objects']:
            val=update['validation']['objects'][obj['id']]
            report=read(update['object_updates'][obj['id']]['report_path'])
            oldn=val['crossings_before'];newn=obj['intersection_counts']['transverse']
            rows.append(dict(scene=scene,id=obj['id'],crossings_before=oldn,crossings_after=newn,
                crossing_reduction_fraction=1-newn/max(oldn,1),
                no_transverse_intersections=newn==0,closed_object=obj['closed_object'],
                significant_inside_vertices=obj['penetration'].get('significant_inside_vertices'),
                maximum_background_vertex_internal_distance=obj['penetration'].get('maximum_depth'),
                maximum_background_vertex_internal_distance_fraction=obj['penetration'].get('maximum_depth_fraction'),
                accepted=val['accepted'],unresolved=val['unresolved'],reasons=val['reasons'],
                generated_mesh_geometry_changed='mesh_path' in update['object_updates'][obj['id']],
                contact_tradeoff_adopted=val.get('contact_tradeoff_adopted',False),
                observation_guard_policy=val.get('observation_guard_policy',report.get('observation_guard_policy','score increase<=.05, median IoU drop<=.05')),
                PSNR_before=original['recomposition']['mean']['PSNR'],PSNR_after=metric['mean']['PSNR'],
                LPIPS_before=original['recomposition']['mean']['LPIPS'],LPIPS_after=metric['mean']['LPIPS']))
    summary=dict(complete=True,updated=time.time(),scenes=len(SCENES),objects=len(rows),rows=rows,
        objects_without_transverse_intersections=sum(r['no_transverse_intersections'] for r in rows),
        total_crossing_pairs_before=sum(r['crossings_before'] for r in rows),
        total_crossing_pairs_after=sum(r['crossings_after'] for r in rows),
        original_gaussians_unchanged=True,background_meshes_unchanged=True,
        benchmark_test_rgb_used_for_optimization=False,
        caveats=['Intersection pair counts depend on tessellation, not a penetration-volume measure.',
            'Open objects have no certified inside/outside depth.',
            'Some source objects already contain self-intersections; repairs reject any increase, not certify zero source defects.',
            'Fruits/garden_toys explicitly adopt a contact/observation tradeoff; not all appearance metrics improve.',
            'Recomposition metrics were rescored on current common positive-depth support for all methods; before/after domains may differ.'])
    write(root/'summary.json',summary)
    integrity_result=dict(complete=True,checks=integrity,updated=time.time(),post_repair=True,
        original_sources_frozen=True,current_published_registration_verified=True)
    write(root/'final_integrity_check.json',integrity_result)
    write(public/'collision_audit/integrity_check.json',integrity_result)
    write(public/'collision_audit/status.json',dict(complete=True,stage='complete',scenes=SCENES,updated=time.time(),post_repair=True))
    import visualize_composed_collisions as vis
    for scene in SCENES:vis.draw_scene(public,public/'collision_audit',scene,.002)
    vis.aggregate(public/'collision_audit')
    from finalize_collision_refinement_publication import finalize as finalize_publication
    finalize_publication(public,SCENES,verified_figures_resume=verified_figures_resume)
    lines=['# 全场景组合 mesh 接触修正结果','',
        '[更新后的全部对比图](../figures_all_methods/README.md) · [当前穿模定位总图](../collision_audit/overview.png) · [结构化前后数据](../collision_refinement/summary.json)','',
        '固定原背景 mesh 和原物体 3DGS，修正生成 mesh 的初始注册、必要的尺寸标定与局部接触形状。所有三角面连接、颜色、UV 和原纹理图保留；没有减面、重新训练高斯或删除背景来掩盖碰撞。新纹理 GLB 按新几何与原纹理重新导出，不引用旧形状。','',
        '先在原照片观测约束下做小幅刚体修正，再对有必要的生成物体底部使用真实环境射线高度场与连续单调接触映射。最终重新纳入完整环境的全部原网格分区，用 FCL 候选与双精度交线核验；检查实际写盘资产的自交数不增加。','',
        '| 场景 | 物体 ID | 修正前相交对 | 修正后相交对 | 检查结论 |',
        '|---|---:|---:|---:|---|']
    for r in rows:
        verdict='未检出横向相交' if r['no_transverse_intersections'] else '仍有残余，needs_review'
        if r['contact_tradeoff_adopted']:verdict+='；采用接触/观测取舍'
        lines.append(f'| {r["scene"]} | {r["id"]} | {r["crossings_before"]:,} | {r["crossings_after"]:,} | {verdict} |')
    lines += ['',f'共 {len(rows)} 个物体，{summary["objects_without_transverse_intersections"]} 个未检出横向相交。相交对数依赖三角化，不是独立碰撞事件数或穿透体积。开放物体不能据此获得全局封闭体证明；保留原生成模型已存在的缺陷记录。', '',
        'fruits、garden_toys 的强接触改善会增加部分原照片观测误差：选取保留至少 90% 初始 median IoU、综合观测误差增加不超过 .08 的候选，另行标记；其余采用 .05/.05 观测预算。不能声称全部场景 PSNR/SSIM/LPIPS 同时提升。优化未读取指标用的留出 RGB。','',
        '组合指标在新的共同可见表面/像素支持域上对全部方法重算；因此前后数字不必然对应完全相同的评分像素。纯背景指标和图像保持原样。CD/F1 参考仍是 COLMAP 已观测稀疏点，非独立扫描真值。','',
        '原高斯 SHA256、背景指标数值、正式 mesh 注册与最终相交证据的一致性已逐场景核验。详见 [完整性检查](../collision_refinement/final_integrity_check.json)。']
    cone_report_path=root/'final_updates/cone_yellow/8/report.json'
    if cone_report_path.exists():
        cone=read(cone_report_path)
        if cone.get('radial_geometry_scale') is not None:
            lines += ['', 'cone_yellow 追加校准：从现有原采集 mask/depth 证据重新标定位姿和径向宽度；'
                f'径向几何乘 {cone["radial_geometry_scale"]:.6f}、高度几何乘 {cone["height_geometry_scale"]:.6f}，'
                '正仿射形变烘焙到生成 mesh 顶点，场景节点的统一尺度保持原值。'
                f'最终接触上移 {cone["contact_lift_scene_units"]:.6f} 场景单位；'
                f'11 视角 median IoU 从 {cone["observation_before"]["median_iou"]:.6f} '
                f'变为 {cone["observation_after"]["median_iou"]:.6f}。'
                '这些视角用于场景内注册标定，不能将其解释为严格未见视角的位姿泛化结果；未使用指标 RGB 优化。']
    doc='\n'.join(lines)+'\n';(public/'docs/COLLISION_REFINEMENT_RESULTS.zh-CN.md').write_text(doc)
    (root/'README.md').write_text(doc.replace('../figures_all_methods/','../figures_all_methods/'))
    (public/'collision_audit/CONCLUSION.zh-CN.md').write_text('# 当前组合 mesh 碰撞检查\n\n本目录已覆盖为接触修正后的当前结果。\n\n[修正前后结论与统计](../docs/COLLISION_REFINEMENT_RESULTS.zh-CN.md) · [逐场景当前定位图](README.md)\n')
    write(public/'collision_audit/summary.json',summary)
    for name in ['README.md','figures_all_methods/README.md']:
        path=public/name;text=path.read_text();link='docs/COLLISION_REFINEMENT_RESULTS.zh-CN.md' if name=='README.md' else '../docs/COLLISION_REFINEMENT_RESULTS.zh-CN.md'
        note=f'\n[完整环境碰撞修正与残余问题]({link})\n'
        if link not in text:path.write_text(text+note)
    path=public/'docs/FINE_WHOLE_RESULTS.md'
    if path.exists():
        text=path.read_text().replace('组合场景保留原有MV-SAM3D物体位姿。','生成 mesh 已按完整环境支撑面修正初始注册及必要的底部接触形状，原始删除物体 3DGS 保持不动；见 COLLISION_REFINEMENT_RESULTS.zh-CN.md。')
        path.write_text(text)
    write(root/'complete.json',dict(complete=True,updated=time.time(),scenes=SCENES,
        figures_and_reports_overwritten=True,final_collision_proofs_verified=True,
        original_gaussians_unchanged=True,unresolved_objects=[r for r in rows if r['unresolved']]))
    print('ALL_COLLISION_REPAIR_DELIVERED',summary['objects_without_transverse_intersections'],len(rows),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=PUBLIC)
    p.add_argument('--verified-figures-resume',action='store_true')
    args=p.parse_args();finalize(args.output,args.verified_figures_resume)
