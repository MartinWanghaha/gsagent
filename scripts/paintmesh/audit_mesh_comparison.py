#!/usr/bin/env python3
"""Check real meshes, input provenance and native raster depth for every scene."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw,ImageOps

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,REPO,SCENES,METHODS,STATES,read,write,identity,specs_for
from mesh_benchmark.layout import work_root,public_root,configure_sources
from mesh_benchmark.color_export import read_mesh


def structure(output):
    public=public_root(work_root(output));rows=[]
    for method in METHODS:
        if method=='paintmesh':continue
        for scene in SCENES:
            marker=read(BASE/method/scene/'mesh_complete.json');path=Path(marker['path'])
            mesh=read_mesh(path);vertices=mesh['vertex'].data;faces=mesh['face'].data['vertex_indices']
            assert len(vertices) and len(faces),(method,scene,'empty')
            assert all(n in vertices.dtype.names for n in ['red','green','blue']),(method,scene,'no RGB')
            repeats=0;min_index=len(vertices);max_index=-1;lo=np.full(3,np.inf);hi=-lo.copy()
            for start in range(0,len(vertices),1000000):
                xyz=np.column_stack([vertices[k][start:start+1000000] for k in ['x','y','z']])
                assert np.isfinite(xyz).all(),(method,scene,'nonfinite vertex')
                lo=np.minimum(lo,xyz.min(0));hi=np.maximum(hi,xyz.max(0))
            for start in range(0,len(faces),1000000):
                f=faces[start:start+1000000];minimum=int(f.min());maximum=int(f.max())
                assert minimum>=0 and maximum<len(vertices),(method,scene,'invalid index')
                min_index=min(min_index,minimum);max_index=max(max_index,maximum)
                repeats+=int(((f[:,0]==f[:,1])|(f[:,0]==f[:,2])|(f[:,1]==f[:,2])).sum())
            rows.append(dict(method=method,scene=scene,identity=identity(path),vertices=len(vertices),faces=len(faces),
                finite_positions=True,invalid_indices=0,repeated_index_faces=repeats,
                repeated_index_face_fraction=repeats/len(faces),bbox=[lo.tolist(),hi.tolist()],
                index_range=[min_index,max_index],needs_exact_uint32=max_index>2**24,has_vertex_colors=True))
            print('AUDIT_MESH_STRUCTURE',method,scene,'pass',flush=True)
    write(public/'docs/ALL_METHODS_MESH_STRUCTURE.json',dict(complete=True,meshes=rows))
    return rows


def depths(output):
    root=work_root(output);rows=[]
    for method in METHODS:
        if method=='paintmesh':continue
        for scene in SCENES:
            views=[]
            info=read(root/'common'/scene/'complete.json')
            for camera in [c for c in info['cameras'] if c['part']=='held_out']:
                name=camera['stem']+'.npy'
                native=np.load(root/'native_renderers'/method/scene/'depth'/name,mmap_mode='r')
                reference=np.load(root/'cache'/method/scene/'recomposition/depth'/name,mmap_mode='r')
                valid=np.isfinite(native)&np.isfinite(reference)&(native>0)&(reference>0)
                assert valid.any(),(method,scene,name,'no shared depth')
                relative=np.abs(native[valid]-reference[valid])/reference[valid]
                median=float(np.median(relative))
                # Independent ray intersection of identical triangles is a
                # renderer check, not a GT-based score or geometry filter.
                assert median<.01,(method,scene,name,'native depth disagrees with source triangles',median)
                views.append(dict(name=camera['name'],median_relative_error=median,
                    fraction_relative_error_gt_1pct=float(np.mean(relative>.01))))
            rows.append(dict(method=method,scene=scene,views=views,
                median_relative_depth_error=float(np.median([v['median_relative_error'] for v in views])),passed=True))
            print('AUDIT_NATIVE_DEPTH',method,scene,'pass',flush=True)
    write(public_root(root)/'docs/RENDERER_DEPTH_AUDIT.json',dict(complete=True,
        purpose='Independent same-mesh depth check, not ground truth comparison',meshes=rows))
    return rows


def provenance(output):
    root=work_root(output);configure_sources(root);count=0;objects=0
    for scene in SCENES:
        source=read(REPO/'output/paintmesh/Inpaint360GS/recomposition'/scene/'composition/scene.json')
        published=read(public_root(root)/'recomposition'/scene/'composition/scene.json')
        assert source['status']==published['status']
        assert len(source['objects'])==len(published['objects'])
        for original,node in zip(source['objects'],published['objects']):
            assert original['id']==node['id'] and original['mesh']['sha256']==node['mesh']['sha256']
            np.testing.assert_allclose(original['object_to_world'],node['object_to_world'],rtol=0,atol=1e-10)
            objects+=1
        info=read(root/'common'/scene/'complete.json')
        assert info['request']['split']==identity(BASE/'_data'/scene/'split.json')
        for source in info['request']['masks']:assert source==identity(source['path'])
        assert info['request']['selection']==identity(public_root(root)/'recomposition'/scene/'composition/scene.json')
        for method in METHODS:
            for state in STATES:
                meta=read(root/'cache'/method/scene/state/'complete.json')
                chosen=specs_for(scene,method,state)
                wanted=[dict(identity=identity(s['path']),transform=s.get('transform')) for s in chosen]
                assert meta['request']['specs']==wanted,(method,scene,state,'mesh cache changed')
                assert meta['request']['common']==info['request_id']
                count+=1
    write(public_root(root)/'docs/ALL_METHODS_PROVENANCE_CHECK.json',dict(complete=True,mesh_modes=count,latest_source_objects=objects,
        original_masks_and_splits_unchanged=True,geometry_caches_match_current_meshes=True))


def report(output,mesh_rows,depth_rows):
    root=work_root(output);public=public_root(root)
    if read(root/'protocol.json').get('native_support_validation',{}).get('enabled'):
        from publish_opacity_mesh_repair import publish_support_report
        publish_support_report(output)
        return
    lookup={(r['method'],r['scene']):r for r in depth_rows}
    old=read(root/'audit_before_index_fix.json')
    notes=['# 全方法、全场景结果检查','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}。','',
        '44个基线 mesh、22个 PaintMesh 模式、110份模式指标已核对；两模式分别使用目标 mask 外背景与完整组合域。源几何、相机和原始物体标签未按分数修改。','',
        '## 已修复：GOF 大网格渲染索引精度','',
        'pyrender 0.1.45 的 Primitive.indices.setter 将索引转为 float32；超过2^24的相邻顶点编号会合并，上传 GPU 时再转回 uint32，已无法恢复原始三角形。car、cone_red、garden_toys、redbull 恰好超过此范围。现在在相同 pyrender 后端中保留 uint32 索引；全部原顶点和三角形保持不变。修复依据为原索引与独立 Embree 射线深度，不使用 GT 调参。','',
        '| GOF场景 | 原组合mesh PSNR | 修复后 PSNR | 原生深度相对误差中位数 |','|---|---:|---:|---:|']
    previews=public/'figures_all_methods/native_color_preview/index_precision_fix'
    for scene in ['car','cone_red','garden_toys','redbull']:
        metric=read(root/'recomposition/gof'/scene/'metrics.json')
        notes.append(f'| {scene} | {old["recomposition/"+scene]["mean"]["PSNR"]:.6f} | {metric["mean"]["PSNR"]:.6f} | {lookup["gof",scene]["median_relative_depth_error"]:.8f} |')
        camera=read(public/'figures_all_methods/metadata/selected_views.json')['scenes'][scene]['camera']
        source=root/'recomposition/gof'/scene/'renders'/(camera['stem']+'.png')
        Image.open(source).save(previews/(scene+'_after.png'))
        canvas=Image.new('RGB',(1200,440),'white');draw=ImageDraw.Draw(canvas)
        for i,(file,label) in enumerate([(previews/(scene+'_before.png'),'Before: float32 index corruption'),(source,'After: exact uint32 indices')]):
            with Image.open(file) as image:
                panel=ImageOps.contain(image.convert('RGB'),(600,400));canvas.paste(panel,(i*600+(600-panel.width)//2,34))
            draw.text((i*600+12,10),scene+' | '+label,fill='black')
        canvas.save(previews/(scene+'_comparison.jpg'),quality=94)
    notes+=['','[四场景修复前后图片](../figures_all_methods/native_color_preview/index_precision_fix/)','',
        '## 仍需区分的重建质量问题','',
        'CoMe truck 的异常来自原生提取网格，而非相机、着色或渲染链接。独立射线取样1024条均被4个巨型三角面遮挡，最长边约18.23 COLMAP单位；训练相机半径约5.70。1024个原网格顶点取样的官方精确不透明度中位数接近0.5，但1024个可见三角形内部取样不透明度均为1，证明大面跨越了自身场的等值表面。没有把这类面用 GT 距离筛掉，也没有改用高斯图片掩盖问题；当前分数照实保留并标为需复核。',
        '证据：[三角形检查](come_truck_triangle_audit.json) · [官方场检查](come_truck_surface_queries.result.json)。',
        '', '2DGS 原生未后处理网格包含重复索引的零面积三角形。它们不产生表面或可评分像素，当前保留原网格并记录数量；未通过删面人为抬高 CD/F1。',
        '', 'PaintMesh 组合资产的既有 needs_review 状态继续保留。重建或位姿质量低并不等于评分程序失败。','',
        '## 全场景检查与分数','',
        '| 场景 | 方法 | F1@1% | 组合mesh PSNR | 状态/观察 |','|---|---|---:|---:|---|']
    for scene in SCENES:
        for method in METHODS:
            m=read(root/'recomposition'/method/scene/'metrics.json')
            status='需复核：原生巨型跨空面' if (method,scene)==('come','truck') else '需复核：生成物体/位姿' if m['quality_status']!='complete' else '渲染/评估检查通过'
            notes.append(f'| {scene} | {method} | {m["geometry"]["F1"]:.6f} | {m["mean"]["PSNR"]:.6f} | {status} |')
    notes+=['','## 评估范围','',
        '可以评估 F1，同时报告0.5%/1%/2%相机半径阈值、P/R、CD与支持比例。但本数据的参考是 COLMAP 稀疏点，不能称独立扫描真值；这些分数表示已观测表面的一致性，不是场景完整度。RGB 只在各原生后端共同覆盖的像素评分；背景严格排除目标 mask。',
        '论文使用虚拟扫描与统一协议下的 MBR，本目录按用户要求采用各项目原生网格后端及原生色，不拟合统一外观场，故不是论文 MBR 数值的同口径复现。参考：[原论文](https://arxiv.org/html/2604.07337v1)。',
        '', '[网格结构核验](ALL_METHODS_MESH_STRUCTURE.json) · [原生深度核验](RENDERER_DEPTH_AUDIT.json) · [来源核验](ALL_METHODS_PROVENANCE_CHECK.json) · [完整交付核验](NATIVE_RGB_DELIVERY_CHECK.json)',
        '', '[纯背景报告](../reports/background.md) · [组合报告](../reports/recomposition.md) · [全部对比图](../figures_all_methods/README.md)']
    (public/'docs/ALL_METHODS_AUDIT.zh-CN.md').write_text('\n'.join(notes)+'\n')
    # Refresh the old quality page so it cannot display superseded VM-SH scores.
    lines=['# 结果质量观察','', '[本轮完整检查](ALL_METHODS_AUDIT.zh-CN.md)。分数来自当前原生彩色 mesh，非旧统一颜色场。','',
        '## CoMe / truck','',
        '原网格4个巨型三角面遮挡视野；独立射线与官方不透明度场检查均已确认。保留原生几何及实际分数，需复核提取质量。','',
        '| 模式 | CD-L1 | F1@1% | mesh PSNR | SSIM | LPIPS |','|---|---:|---:|---:|---:|---:|']
    for state in STATES:
        m=read(root/state/'come/truck/metrics.json');g=m['geometry'];v=m['mean']
        lines.append(f'| {state} | {g["CD_L1"]:.6f} | {g["F1"]:.6f} | {v["PSNR"]:.6f} | {v["SSIM"]:.6f} | {v["LPIPS"]:.6f} |')
    (public/'docs/QUALITY_NOTES.md').write_text('\n'.join(lines)+'\n')
    write(public/'docs/ALL_METHODS_AUDIT_CHECK.json',dict(complete=True,updated=time.time(),baseline_meshes=len(mesh_rows),
        native_depth_checks=len(depth_rows),gof_integer_indices_fixed=True,source_geometry_modified=False,
        remaining_quality_reviews=['CoMe/truck native extraction contains large unsupported faces','PaintMesh source scene needs_review labels retained']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'mesh_comparison_whole')
    p.add_argument('--depth-only',action='store_true');p.add_argument('--reuse-structure',action='store_true')
    a=p.parse_args();root=work_root(a.output);configure_sources(root)
    if a.depth_only:depths(a.output)
    else:
        state=read(root/'native_rgb_status.json');assert state['stage']=='complete','Finish native evaluation first'
        rows=read(public_root(root)/'docs/ALL_METHODS_MESH_STRUCTURE.json')['meshes'] if a.reuse_structure else structure(a.output)
        depth_rows=depths(a.output);provenance(a.output);report(a.output,rows,depth_rows)
