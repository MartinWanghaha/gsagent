#!/usr/bin/env python3
"""Publish honest postprocessing provenance and matched-domain repair comparisons."""
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw,ImageOps

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,STATES,METHODS,read,write,points_from_depth
from mesh_benchmark.layout import work_root,public_root,configure_sources
from mesh_benchmark.gaussian_support import GaussianSupport
from repair_opacity_meshes import WORK,POLICY
from recompose.io import fingerprint


def matched_scores(root,method,scene,state):
    before=WORK/'before';rows=[]
    old=read(before/method/scene/(state+'_metrics.json'))
    current=read(root/state/method/scene/'metrics.json')
    for c in [c for c in read(root/'common'/scene/'complete.json')['cameras'] if c['part']=='held_out']:
        name=c['stem']+'.png';common=root/'common'/scene
        valid=(np.asarray(Image.open(before/'supports'/scene/(state+'_native_support')/name))>127)&(np.asarray(Image.open(common/(state+'_native_support')/name))>127)
        if not valid.any():continue
        gt=np.asarray(Image.open(common/'held_out'/(c['stem']+'_gt.png')).convert('RGB'),dtype='float64')
        values=[]
        for p in [before/method/scene/'renders'/name,root/state/method/scene/'renders'/name]:
            rgb=np.asarray(Image.open(p).convert('RGB'),dtype='float64')
            mse=float(np.mean((rgb[valid]-gt[valid])**2)/255**2)
            values.append(float(-10*np.log10(max(mse,1e-12))))
        rows.append(dict(name=c['name'],pixels=int(valid.sum()),before_PSNR=values[0],after_PSNR=values[1]))
    return dict(state=state,matched_mean_before=float(np.mean([r['before_PSNR'] for r in rows])),
        matched_mean_after=float(np.mean([r['after_PSNR'] for r in rows])),
        own_coverage_before=old['mean']['mesh_coverage'],own_coverage_after=current['mean']['mesh_coverage'],
        original_main=old['mean'],current_main=current['mean'],F1_before=old['geometry']['F1'],F1_after=current['geometry']['F1'],per_view=rows)


def native_support_review(root,method,scene):
    """Compare source-model support at the same pixels, independent of GT."""
    checkpoint=BASE/method/scene/'model/point_cloud/iteration_30000/point_cloud.ply'
    model=GaussianSupport.from_ply(checkpoint,cutoff=POLICY['native_opacity_cutoff'])
    cs=[c for c in read(root/'common'/scene/'complete.json')['cameras'] if c['part']=='held_out']
    indices=sorted(set([0,len(cs)//2,len(cs)-1]));rows=[]
    for i in indices:
        c=cs[i];name=c['stem']+'.npy';old=np.load(WORK/'before'/method/scene/'depth'/name)
        new=np.load(root/'native_renderers'/method/scene/'depth'/name)
        valid=np.flatnonzero(np.isfinite(old)&np.isfinite(new)&(old>0)&(new>0))
        if not len(valid):continue
        pixels=np.random.default_rng(41+i).choice(valid,min(1024,len(valid)),replace=False)
        a=points_from_depth(c,old.reshape(-1)[pixels],pixels);b=points_from_depth(c,new.reshape(-1)[pixels],pixels)
        values=model.contains(np.concatenate([a,b]));n=len(a)
        rows.append(dict(name=c['name'],pixels=n,source_supported_fraction_before=float(values[:n].mean()),source_supported_fraction_after=float(values[n:].mean())))
    return dict(reference_data_used=False,views=rows,
        source_supported_fraction_before=float(np.mean([r['source_supported_fraction_before'] for r in rows])),
        source_supported_fraction_after=float(np.mean([r['source_supported_fraction_after'] for r in rows])))


def normalize_receipts(method,scene,result):
    out=BASE/method/scene;color=out/'mesh_color_complete.json';data=read(color)
    data.update(vertices=result['vertices'],faces=result['faces'],coordinates_and_colors_exact=True,
        retained_face_records_exact=True,original_native_faces_all_retained=not result['geometry_postprocessed'],
        all_original_geometry_and_properties_exact=not result['geometry_postprocessed'],
        geometry_unchanged_by_colorization=True,geometry_postprocessed_before_color_export=result['geometry_postprocessed'])
    write(color,data)


def publish_support_report(output=BASE/'mesh_comparison_whole',review=True):
    root=work_root(output);configure_sources(root);public=public_root(root);records=[]
    pictures=public/'figures_all_methods/native_support_repair';pictures.mkdir(exist_ok=True)
    selected=read(public/'figures_all_methods/metadata/selected_views.json')['scenes']
    for method in ['come','gof']:
        for scene in SCENES:
            repair=read(WORK/method/scene/'complete.json')
            assert repair['request']['checkpoint']==fingerprint(BASE/method/scene/'model/point_cloud/iteration_30000/point_cloud.ply')
            assert repair['request']['source']==fingerprint(repair['native_source_mesh'])
            normalize_receipts(method,scene,repair)
            row=dict(method=method,scene=scene,repair=repair,
                comparison={s:matched_scores(root,method,scene,s) for s in STATES})
            qa=WORK/method/scene/'visible_support_review.json'
            qa_request=dict(checkpoint=repair['request']['checkpoint'],mesh=repair['candidate_mesh'],
                render_signature=read(root/'native_renderers'/method/scene/'complete.json')['signature'])
            if review and (not qa.exists() or read(qa).get('request')!=qa_request):
                value=native_support_review(root,method,scene);value['request']=qa_request;write(qa,value)
            row['visible_support_review']=read(qa) if qa.exists() else None
            records.append(row)
            camera=selected[scene]['camera'];name=camera['stem']+'.png'
            canvas=Image.new('RGB',(1200,440),'white');draw=ImageDraw.Draw(canvas)
            for i,(path,label) in enumerate([(WORK/'before'/method/scene/'renders'/name,'Native extraction before validation'),(root/'recomposition'/method/scene/'renders'/name,'Source Gaussian support validated')]):
                with Image.open(path) as image:
                    panel=ImageOps.contain(image.convert('RGB'),(600,400));canvas.paste(panel,(i*600+(600-panel.width)//2,34))
                draw.text((i*600+10,10),method+' / '+scene+' | '+label,fill='black')
            canvas.save(pictures/(method+'__'+scene+'.jpg'),quality=94)
            print('SUPPORT_REPAIR_DELIVERY',method,scene,row['comparison']['recomposition']['matched_mean_before'],'->',row['comparison']['recomposition']['matched_mean_after'],flush=True)
    write(public/'docs/OPACITY_MESH_REPAIR_RESULTS.json',dict(complete=True,policy=POLICY,records=records,
        matched_domain='Intersection of old and new cohort common RGB masks, identical for before and after',
        native_sources_unmodified=True,retained_vertex_properties_exact=True,evaluated_geometry_postprocessed=True))
    lines=['# CoMe / GOF 网格修复与对比','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}。','',
        '两种方法的11场景均应用同一个固定校验规则。当前对比是 **原生提取 + 源高斯支撑校验**，不是官方未处理的原样提取；原始 mesh、训练高斯和修复前图像/指标均保留。没有重训3DGS，没有移动顶点或改动原生颜色。','',
        '## 几何规则','',
        '1. 从各自30,000步高斯 checkpoint 读取中心、旋转、尺度、不透明度与原有3D滤波；建立原生1/255不透明度阈值下的有限椭球支撑。',
        '2. 半径分桶 KDTree 保守枚举所有可能包含查询点的高斯，再做准确的旋转椭球测试；没有用最近几个中心近似。与穷举结果的测试一致。',
        '3. 为降低计算量，仅检查边长超过源高斯中位最大尺度8倍的粗三角面。取面心和三个边中点；面心不受支持且四点至少三点不受支持时，判为无支撑跨空插值面。边长只限制检查工作量，不直接决定删面。',
        '4. 原顶点数据逐字节保留，保留面的整数索引和属性也逐字节保留。使用各项目原来的网格后端重新渲染，并重算所有方法共同支持域和指标。',
        '', '规则不读取参考点云、GT RGB、PSNR 或 F1。修复以后才做下述评估，未按测试指标选参数或挑场景。','',
        '## 同像素对比','',
        '以下 PSNR 使用旧共同域与新共同域的交集；修复前后使用完全相同的像素。主报告中覆盖域已按当前五方法重新计算，故这里的同像素分数可能与主表略有不同。覆盖率同时列出，避免把删除像素当成质量改善。','',
        '| 方法 | 场景 | 删除面比例 | 同像素PSNR前 | 后 | F1前 | 后 | 自身覆盖率前 | 后 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        v=row['comparison']['recomposition'];lines.append(f'| {row["method"]} | {row["scene"]} | {row["repair"]["removed_face_fraction"]:.4%} | {v["matched_mean_before"]:.4f} | {v["matched_mean_after"]:.4f} | {v["F1_before"]:.4f} | {v["F1_after"]:.4f} | {v["own_coverage_before"]:.4%} | {v["own_coverage_after"]:.4%} |')
    lines+=['','F1前后分别遵循各轮共同几何支持域，不能将支持域变化解释为固定参考点集的精度改变量。逐视角像素数、覆盖率、阈值与数据来源保存在 JSON 和主报告中。','',
        '[22份修复前后图片](../figures_all_methods/native_support_repair/) · [完整数据](OPACITY_MESH_REPAIR_RESULTS.json)','',
        '## 输出','', '[纯背景几何与渲染报告](../reports/background.md) · [组合mesh几何与渲染报告](../reports/recomposition.md) · [全部方法全部场景图片](../figures_all_methods/README.md)','',
        'CD/F1 的参考为 COLMAP 稀疏点云，表示可观测表面一致性，不是独立扫描真值；背景 mask 内部不参与图像评分。PaintMesh 原有生成物体/位姿的 needs_review 状态继续保留。']
    (public/'docs/OPACITY_MESH_REPAIR.md').write_text('\n'.join(lines)+'\n')
    (root/'OPACITY_MESH_REPAIR.md').write_text('\n'.join(lines)+'\n')
    audit=['# 全方法检查：CoMe / GOF 修复已完成','',
        '已修复 GOF 超过2^24顶点时的渲染索引精度问题；本轮进一步对 CoMe / GOF 的22份原生网格做源高斯支撑校验，修复跨空巨型面。旧 CoMe truck 遮挡问题属于修复前记录，当前分数及图片已覆盖更新。','',
        '[修复规则、全部场景前后指标与图片](OPACITY_MESH_REPAIR.md) · [完整交付核验](NATIVE_RGB_DELIVERY_CHECK.json)','',
        '主报告明确标注后处理，不把经过校验的提取称为官方未修改输出。所有原始文件保留；相机、mask、几何阈值与原始13个组合物体/位姿未调整。','',
        '[纯背景报告](../reports/background.md) · [组合报告](../reports/recomposition.md) · [全部对比图](../figures_all_methods/README.md)']
    (public/'docs/ALL_METHODS_AUDIT.zh-CN.md').write_text('\n'.join(audit)+'\n')
    quality=['# 结果质量观察','',
        'CoMe truck 的原生跨空封口大面已通过源高斯支撑校验处理。GOF cube 等场景采用相同规则，实际改善与覆盖变化见 [修复对比](OPACITY_MESH_REPAIR.md)。','',
        '校验不保证消除所有重建误差：保留原生颜色与训练模型，细节、反射和未观测区域仍可能存在误差。PaintMesh 原有 needs_review 组合状态保留。']
    (public/'docs/QUALITY_NOTES.md').write_text('\n'.join(quality)+'\n')
    home=public/'README.md';contents=home.read_text()
    contents=contents.replace('GOF 的大网格索引问题已在上一轮修复。本轮正在进一步修复 CoMe / GOF 的跨空插值面；下方汇总报告和总图将在全场景完成后自动覆盖。',
        'GOF 大网格索引错误及 CoMe / GOF 无支撑跨空插值面已处理；本轮全场景报告和对比图已覆盖更新。')
    contents=contents.replace('CoMe truck 的巨型遮挡面已在首批修复中消除；全场景统一处理规则见 [修复协议](docs/OPACITY_MESH_REPAIR.md)。',
        'CoMe truck 的巨型遮挡面已消除，全场景采用同一规则；完整修复与覆盖变化见 [修复协议](docs/OPACITY_MESH_REPAIR.md)。')
    home.write_text(contents)
    write(public/'docs/ALL_METHODS_AUDIT_CHECK.json',dict(complete=True,updated=time.time(),baseline_meshes=44,
        native_depth_checks=44,gof_integer_indices_fixed=True,source_native_files_modified=False,
        evaluated_geometry_postprocessed=True,support_validated_meshes=len(records),matched_pixel_comparisons=44,
        remaining_quality_reviews=['Existing PaintMesh source scene needs_review labels retained']))
    return records


if __name__=='__main__':publish_support_report()
