#!/usr/bin/env python3
"""Localize PaintMesh detail loss without changing any model or benchmark image.

Gaussian images are diagnostic evidence only. The benchmark continues to use
actual triangle renders. Every held-out image and every source triangle is used;
triangle processing is chunked, never decimated.
"""
from pathlib import Path
import argparse
import hashlib
import fcntl
import sys
import time
import gc
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,read,write,identity,digest,masked_ssim_map
from mesh_benchmark.layout import work_root,configure_sources
from recompose.io import ply

PUBLIC=BASE/'mesh_comparison_whole'
ROOT=work_root(PUBLIC)
DEST=PUBLIC/'figures_all_methods/paintmesh_blur'
FONTS='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def decoded(path,shape):
    image=np.asarray(Image.open(path).convert('RGB'))
    if image.shape!=shape:raise ValueError(f'Image resize is forbidden: {path}: {image.shape} != {shape}')
    return image


def image_evidence(pred,target,valid):
    """Full native pixels; derivatives evaluated only with a valid 5x5 stencil."""
    valid=np.asarray(valid,dtype=bool)
    if not valid.any():raise ValueError('Empty diagnostic domain')
    p=pred.astype('float64')/255.;t=target.astype('float64')/255.
    mse=float(np.mean((p[valid]-t[valid])**2))
    inner=cv2.erode(valid.astype('uint8'),np.ones((5,5),'uint8')).astype(bool)
    def energy(image):
        gray=image@np.array([.2126,.7152,.0722])
        dx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)/8
        dy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)/8
        return float(np.mean(dx[inner]**2+dy[inner]**2)) if inner.any() else 0.
    pe,te=energy(p),energy(t)
    return dict(pixels=int(valid.sum()),PSNR=float(-10*np.log10(max(mse,1e-12))),
        SSIM=float(masked_ssim_map(p,t,valid)[valid].mean()),
        gradient_energy=pe,GT_gradient_energy=te,gradient_energy_ratio=pe/te if te>0 else None,
        note='Gradient ratio is a detail/noise diagnostic, not a geometry or perceptual accuracy metric.')


def triangle_footprint(path,camera,depth,allowed):
    """Visit ALL original faces and measure the visible triangle pixel footprint.

    Visibility uses this mesh's own depth only, never RGB/GT error. Same 1%
    relative depth agreement rule for both meshes. No triangle is altered.
    """
    data=ply(path);v=data['vertex'].data;faces=data['face'].data['vertex_indices']
    K=np.asarray(camera['K']);T=np.asarray(camera['w2c']);h,w=depth.shape
    # Transform each original vertex once. This avoids repeated random reads
    # from a multi-GB interleaved PLY, without dropping a single face/vertex.
    camera_vertices=np.column_stack([v[n] for n in ['x','y','z']]).astype('float64')@T[:3,:3].T+T[:3,3]
    projected=camera_vertices@K.T
    pixel_vertices=projected[:,:2]/np.maximum(projected[:,2:],1e-12)
    del projected
    output={k:[] for k in ['world_edge','pixel_edge','pixel_area']};visited=0
    for start in range(0,len(faces),100000):
        ids=faces[start:start+100000]
        q=camera_vertices[ids];center=q.mean(1);positive=(q[:,:,2]>.000001).all(1)
        uvw=center@K.T;uv=uvw[:,:2]/np.maximum(uvw[:,2:],1e-12)
        xx=np.floor(uv[:,0]).astype('int64');yy=np.floor(uv[:,1]).astype('int64')
        valid=positive&(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        ix=np.where(valid)[0]
        if len(ix):
            z=depth[yy[ix],xx[ix]]
            valid[ix]&=(z>0)&allowed[yy[ix],xx[ix]]&(np.abs(center[ix,2]-z)<=.01*np.maximum(z,1e-12))
        q=q[valid];pix=pixel_vertices[ids[valid]]
        a=pix[:,1]-pix[:,0];b=pix[:,2]-pix[:,0]
        world=np.linalg.norm(q-np.roll(q,1,axis=1),axis=2).max(1)
        screen=np.linalg.norm(pix-np.roll(pix,1,axis=1),axis=2).max(1)
        area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])*.5
        for k,values in zip(output,[world,screen,area]):output[k].append(values)
        visited+=len(ids)
    stats={}
    for k,values in output.items():
        values=np.concatenate(values)
        if not len(values):raise ValueError(f'No visible triangles: {path}')
        stats[k]=dict(mean=float(values.mean()),p10=float(np.quantile(values,.1)),
                      median=float(np.median(values)),p90=float(np.quantile(values,.9)),p95=float(np.quantile(values,.95)))
    assert visited==len(faces)
    return dict(source=identity(path),vertices=len(v),faces=len(faces),faces_visited=visited,
                visible_triangles=len(values),camera=camera['name'],all_faces_processed=True,
                visibility='Centroid inside shared background pixels and within 1% of this mesh native depth',**stats)


def outline(image,foreground):
    image=image.copy();edge=foreground&~cv2.erode(foreground.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
    image[edge]=[255,40,150];return image


def pipeline_figure(scene,camera,roi,panels,foreground):
    # Full images are pasted at exactly their original size; only ROI is enlarged.
    w,h=camera['width'],camera['height'];zoom_h=max(240,int((roi[3]-roi[1])*w/max(1,roi[2]-roi[0])))
    top=100;row=64+h+28+zoom_h;canvas=Image.new('RGB',(w*len(panels),top+row+42),'white')
    d=ImageDraw.Draw(canvas);f=ImageFont.truetype(FONTS,22);small=ImageFont.truetype(FONTS,17)
    d.text((12,10),f'{scene} | {camera["name"]} | native {w} x {h} | no image downsampling / no mesh simplification',fill='black',font=f)
    d.text((12,45),'Gaussian columns are diagnostics only. Mesh evaluation figures always use triangle meshes. Pink: original target.',fill='black',font=small)
    for i,(label,pixels) in enumerate(panels):
        x=i*w;d.text((x+10,top+6),label,fill='black',font=f)
        image=Image.fromarray(outline(pixels,foreground) if i==0 else pixels)
        canvas.paste(image,(x,top+50));d.rectangle((x+roi[0],top+50+roi[1],x+roi[2],top+50+roi[3]),outline=(255,40,150),width=2)
        d.text((x+10,top+56+h),'Same ROI; enlarged for inspection',fill='black',font=small)
        crop=Image.fromarray(pixels).crop(roi).resize((w,zoom_h),Image.Resampling.LANCZOS)
        canvas.paste(crop,(x,top+85+h))
    path=DEST/(scene+'_pipeline.png');canvas.save(path)
    return path


def evidence_request(scene,selection):
    """Bind reusable diagnostics to every actual input, including mask contents.

    The evaluator rewrites identical coverage PNGs when it publishes scores.
    Their decoded bits, rather than write timestamps, determine this evidence.
    All mesh faces/views are still processed whenever any input changes.
    """
    graph=PUBLIC/'recomposition'/scene/'composition/scene.json';document=read(graph)
    original=Path(document['input_selection']['original_mesh'])
    stage1=original.parents[2];final=Path(document['background']['gaussians']['path']).parents[2]
    common=ROOT/'common'/scene;info=read(common/'complete.json')
    paths=[Path(__file__),HERE/'mesh_benchmark/core.py',HERE/'recompose/io.py',graph,original,
        Path(document['background']['mesh']['path']),Path(document['background']['gaussians']['path']),
        stage1/'point_cloud/iteration_30000/point_cloud.ply',
        ROOT/'models'/scene/'whole_mesh_complete.json',common/'complete.json',
        stage1/'train/ours_30000/render_manifest.json',final/'train/ours_5000/render_manifest.json']
    for node in document['objects']:
        asset=Path(node['mesh']['path']);paths.extend([asset,asset.with_name('generation.json')])
    masks={}
    for c in info['cameras']:
        if c['part']!='held_out':continue
        stem=c['stem']
        paths.extend([Path(document['input_selection']['source'])/'images'/c['name'],
            common/'held_out'/(stem+'_gt.png'),common/'held_out'/(stem+'_foreground.png')])
        for model,iteration in [(stage1,30000),(final,5000)]:
            paths.extend([model/f'train/ours_{iteration}/renders'/(stem+'.png'),
                          model/f'train/ours_{iteration}/gt'/(stem+'.png')])
        for method,state in [('paintmesh_stage1','background'),('paintmesh','background'),('paintmesh','recomposition')]:
            paths.append(ROOT/'cache'/method/scene/state/'native_renders'/(stem+'.png'))
        mask=np.asarray(Image.open(common/'background_native_support'/(stem+'.png')))>127
        masks[stem]=dict(shape=list(mask.shape),sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    for method in ['paintmesh_stage1','paintmesh']:
        paths.append(ROOT/'cache'/method/scene/'background/depth'/(selection['camera']['stem']+'.npy'))
    files=[]
    for path in paths:
        resolved=path.resolve(strict=True);stat=resolved.stat()
        files.append(dict(**identity(resolved),inode=stat.st_ino,device=stat.st_dev,ctime_ns=stat.st_ctime_ns))
    return dict(files=files,masks=masks,camera=selection['camera'],roi=selection['roi'])


def audit_scene(scene,selection):
    request=evidence_request(scene,selection);request_id=digest(request)
    cached=DEST/'metadata'/(scene+'.json')
    if cached.exists():
        old=read(cached);picture=Path(old['figure'])
        if old.get('complete') and old.get('evidence_request_id')==request_id and picture.exists() and old.get('figure_identity')==identity(picture):
            assert all(g['all_faces_processed'] and g['faces_visited']==g['faces'] for g in old['geometry_footprint'].values())
            print('PAINTMESH_BLUR_REUSE_EXACT_INPUTS',scene,flush=True);return old
    document=read(PUBLIC/'recomposition'/scene/'composition/scene.json')
    original=Path(document['input_selection']['original_mesh']);whole=Path(document['background']['mesh']['path'])
    stage1model=original.parents[2];finalmodel=Path(document['background']['gaussians']['path']).parents[2]
    extraction=read(ROOT/'models'/scene/'whole_mesh_complete.json')
    info=read(ROOT/'common'/scene/'complete.json');camera=selection['camera'];perview=[];panels=None
    immutable=[original,whole,stage1model/'point_cloud/iteration_30000/point_cloud.ply',Path(document['background']['gaussians']['path'])]
    before=[identity(p) for p in immutable]
    for c in info['cameras']:
        if c['part']!='held_out':continue
        stem=c['stem'];common=ROOT/'common'/scene
        gt=np.asarray(Image.open(common/'held_out'/(stem+'_gt.png')).convert('RGB'))
        foreground=np.asarray(Image.open(common/'held_out'/(stem+'_foreground.png')))>127
        valid=np.asarray(Image.open(common/'background_native_support'/(stem+'.png')))>127
        paths=dict(stage1_gaussian=stage1model/'train/ours_30000/renders'/(stem+'.png'),
           stage1_mesh=ROOT/'cache/paintmesh_stage1'/scene/'background/native_renders'/(stem+'.png'),
           inpaint_gaussian=finalmodel/'train/ours_5000/renders'/(stem+'.png'),
           whole_background_mesh=ROOT/'cache/paintmesh'/scene/'background/native_renders'/(stem+'.png'),
           composed_mesh=ROOT/'cache/paintmesh'/scene/'recomposition/native_renders'/(stem+'.png'))
        images={k:decoded(p,gt.shape) for k,p in paths.items()}
        evidence={k:image_evidence(a,gt,valid) for k,a in images.items()}
        # The RGB saved with Gaussian renders is checked against our frozen GT.
        alignment={}
        # EDGS saves PIL's default BICUBIC-resized GT; this frozen benchmark
        # uses BILINEAR. Verify the actual source photo and both conventions
        # rather than mistaking a known interpolation difference for a new pose.
        source_photo=Image.open(Path(document['input_selection']['source'])/'images'/c['name']).convert('RGB')
        native_gt=np.asarray(source_photo.resize((c['width'],c['height']),Image.Resampling.BICUBIC))
        frozen_gt=np.asarray(source_photo.resize((c['width'],c['height']),Image.Resampling.BILINEAR))
        np.testing.assert_array_equal(gt,frozen_gt)
        for name,model,iteration in [('stage1_gaussian',stage1model,30000),('inpaint_gaussian',finalmodel,5000)]:
            manifest=read(model/f'train/ours_{iteration}/render_manifest.json')
            assert manifest['views'][c['name']]==stem+'.png'
            gtfile=model/f'train/ours_{iteration}/gt'/(stem+'.png')
            expected=decoded(gtfile,gt.shape)
            alignment[name]=dict(manifest=identity(model/f'train/ours_{iteration}/render_manifest.json'),
                 saved_GT_vs_source_bicubic_max_difference=int(np.abs(expected.astype('int16')-native_gt.astype('int16')).max()),
                 frozen_GT_vs_saved_GT_max_difference=int(np.abs(expected.astype('int16')-gt.astype('int16')).max()),
                 saved_GT_filter='PIL BICUBIC',frozen_benchmark_GT_filter='PIL BILINEAR (unchanged)')
            if alignment[name]['saved_GT_vs_source_bicubic_max_difference']>1:raise ValueError(f'Diagnostic camera/GT mismatch: {scene}/{stem}/{name}')
        perview.append(dict(name=c['name'],shape=list(gt.shape),domain='Six-method common background support; foreground excluded',
            metrics=evidence,sources={k:identity(v) for k,v in paths.items()},alignment=alignment))
        print('BLUR_RGB_VIEW',scene,len(perview),c['name'],flush=True)
        if c['name']==camera['name']:
            panels=[('Original RGB',gt),('Stage 1: 3DGS (diagnostic)',images['stage1_gaussian']),('Stage 1: actual mesh',images['stage1_mesh']),
                ('Inpaint: 3DGS (diagnostic)',images['inpaint_gaussian']),('Whole background: mesh',images['whole_background_mesh']),('Composed: actual mesh',images['composed_mesh'])]
            picture=pipeline_figure(scene,camera,selection['roi'],panels,foreground)
    if panels is None:raise ValueError('Fixed figure camera was not scored')
    means={k:{m:float(np.mean([r['metrics'][k][m] for r in perview])) for m in ['PSNR','SSIM','gradient_energy_ratio']} for k in perview[0]['metrics']}
    stem=camera['stem'];allowed=np.asarray(Image.open(ROOT/'common'/scene/'background_native_support'/(stem+'.png')))>127
    geo={}
    for label,method,path in [('stage1','paintmesh_stage1',original),('whole','paintmesh',whole)]:
        depth=np.load(ROOT/'cache'/method/scene/'background/depth'/(stem+'.npy'))
        print('BLUR_ALL_TRIANGLES',scene,label,str(path),flush=True)
        geo[label]=triangle_footprint(path,camera,depth,allowed)
    assert before==[identity(p) for p in immutable],'Source model changed during read-only audit'
    objects=[]
    for node in document['objects']:
        asset_path=Path(node['mesh']['path'])
        with np.load(asset_path) as asset:
            texture_names=[k for k in asset.files if k.startswith('texture_') and not k.startswith('texture_factor_')]
            objects.append(dict(id=node['id'],source=identity(asset_path),vertices=len(asset['vertices']),faces=len(asset['faces']),
                textures={k:list(asset[k].shape) for k in texture_names},
                material_ids=np.unique(asset['triangle_material']).tolist() if 'triangle_material' in asset else [],
                generation=read(asset_path.with_name('generation.json'))))
    result=dict(scene=scene,complete=True,updated=time.time(),figure=str(picture),native_images_only=True,
        immutable_sources=before,source_models_unchanged=True,held_out_views=len(perview),per_view=perview,means=means,
        whole_extraction=extraction['parameters'],geometry_footprint=geo,
        generated_objects=objects,
        primary_F1_threshold=.01*info['extent'],
        primary_F1_threshold_in_whole_voxels=.01*info['extent']/extraction['parameters']['voxel_size'],
        stage1_parameters=dict(voxel_size_default=.002,sdf_trunc_default=.008,max_depth_default=5.,
            confidence='Source-code defaults only: original stage-1 mesh has no per-artifact extraction-parameter receipt. Do not treat defaults as verified historical settings.'),
        no_quality_reduction=dict(mesh_decimation=False,image_downsampling=False,views_dropped=False,
            geometry_sampling_reduced=False,triangle_footprint_all_faces=True))
    assert request==evidence_request(scene,selection),'Diagnostic input changed during full-resolution audit'
    result.update(evidence_request_id=request_id,evidence_request=request,figure_identity=identity(picture))
    write(cached,result)
    print('PAINTMESH_BLUR_SCENE',scene,means,flush=True)
    gc.collect();return result


def run(evidence_only=False):
    configure_sources(ROOT);DEST.mkdir(exist_ok=True,parents=True)
    lock=open(DEST/'.audit.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    selections=read(PUBLIC/'figures_all_methods/metadata/selected_views.json')['scenes']
    records=[audit_scene(s,selections[s]) for s in SCENES]
    if evidence_only:
        print('PAINTMESH_BLUR_EVIDENCE_READY',len(records),flush=True);return
    from publish_latest_mesh_geometry import figures
    figures(PUBLIC)
    index=PUBLIC/'README.md';contents=index.read_text()
    contents=contents.replace('11 个场景，四个基线与 PaintMesh 纯背景/组合两种模式。110 份几何和原生彩色 mesh RGB 指标、132 组含 RGB 的独立子图均已更新。',
        '11 个场景，四个基线、真实 PaintMesh 第一阶段mesh，以及 PaintMesh 纯背景/组合两种模式。132 份几何和原生彩色 mesh RGB 指标、154 组含 RGB 的独立子图均已更新。')
    contents=contents.replace('CoMe 暗斑已完成只读溯源，包含三场景原生白场/权重实验；尚未覆盖颜色或主表指标：',
        'CoMe 暗斑历史溯源报告（后续颜色归一化与主表更新已经完成）：')
    index.write_text(contents)
    index=PUBLIC/'results/README.md';contents=index.read_text()
    contents=contents.replace('`gaussian_wrapping`、`paintmesh`','`gaussian_wrapping`、`paintmesh_stage1`、`paintmesh`')
    index.write_text(contents)
    index=PUBLIC/'meshes/README.md';contents=index.read_text()
    if 'stage1.ply' not in contents:index.write_text(contents+'\n各场景 `stage1.ply` 指向实际第一阶段原始mesh（含原物体），用于新增对照列；没有重新提取或简化。\n')
    write(PUBLIC/'docs/PAINTMESH_BLUR_AUDIT.json',dict(complete=True,updated=time.time(),scenes=records,
        evidence='Read-only native-image comparison and all-face triangle footprint audit',Gaussian_images_in_benchmark=False))
    write(PUBLIC/'docs/QUALITY_PRESERVATION.json',dict(complete=True,updated=time.time(),
        original_meshes_and_gaussians_unchanged=all(r['source_models_unchanged'] for r in records),
        additional_image_downsampling=False,mesh_decimation=False,held_out_views_dropped=False,
        held_out_view_count=sum(r['held_out_views'] for r in records),geometry_thresholds_unchanged=True,
        automatic_tsdf_coarsening_disabled=True,
        historical_whole_mesh_quality_reductions_reversed=False,
        explanation='This update diagnoses existing coarse extraction; it does not falsely claim to have restored its missing detail.'))
    historical=PUBLIC/'docs/FUSION_MEMORY_ADJUSTMENTS.md'
    historical_note='**历史提取记录：本次核查没有继续放大体素。自动降分辨率重试已禁用；现有 mesh 尚未重新提取，不能把新增列/重评估声称为几何修复。**\n\n'
    if '历史提取记录：本次核查' not in historical.read_text():historical.write_text(historical_note+historical.read_text())
    old_plan=PUBLIC/'docs/WHOLE_SCENE_PLAN.zh-CN.md'
    plan_note='**本文为历史全背景提取计划。当前评估已使用各项目原生网格渲染器与顶点色，不再拟合统一颜色场；已加入实际第一阶段 mesh。历史体素增大已在模糊核查中明确列出，自动降质重试现已禁用。最新结果见两份主报告。**\n\n'
    if '本文为历史全背景提取计划' not in old_plan.read_text():old_plan.write_text(plan_note+old_plan.read_text())
    lines=['# PaintMesh 几何/颜色模糊核查与第一阶段 mesh 对照','',
        '本次加入的是每个场景实际保存的第一阶段 EDGS 30000 步网格，来自组合场景的 `input_selection.original_mesh`。不是重新生成的高斯图，也不是语义标签着色网格。两个主报告、所有场景图和独立子图已加入这一列。','',
        f'本次未更改、重训、简化任何源网格或 3DGS；全部{sum(r["held_out_views"] for r in records)}个留出视角按既有原生分辨率计算。图像直接使用全部像素，几何细节诊断逐面遍历原网格，仅分块控制内存。已禁用内存不足时自动增大体素的逻辑，后续资源不足应停止/等待或分块提取，不能暗中降低质量。','',
        '## 主要原因','',
        '1. **全背景提取本身使用了较粗的空间离散。** `whole_mesh_complete.json` 记录实际体素为相机半径/384，带宽为6个体素；既有记录明确记载首轮768因内存改为384。这会降低窄缝、薄边和曲面细节的可表达分辨率。当前总面数仍可很大，因为它覆盖了更大背景；总面数不能代表目标附近细节密度。','',
        '2. **较宽 TSDF 带宽会把相互不完全一致的深度融合。** 多视角 PGSR plane depth 不是精确表面；若深度误差、遮挡或掠射面未充分过滤，融合可能造成边缘变钝/几何偏移。体素和带宽的实际值是已核实事实；它们各自造成的误差比例仍需固定其他条件的重提取消融，本报告不虚构因果百分比。','',
        '3. **颜色只保存在顶点，并经过多视角融合。** 当前全背景使用 Open3D TSDF `RGB8` 融合，mesh 渲染再用三角形三个顶点颜色做重心插值；没有背景高分辨率 UV 纹理。深度/曝光/视角差异会使融合颜色平均化，较大的三角形也难保留木纹等高频纹理。生成物体的纹理情况见下表，不能把没有纹理图的顶点色资产称为高分辨率纹理 mesh。','',
        '4. **训练与补全也可能带来洞内误差，但不能把所有模糊都归给 LaMa。** 下方同视角 3DGS→mesh 对照把外观在提取前后的差别单独列出。该诊断只在 mask 外共同有效像素比较；洞内没有真实“去除物体后”的 RGB 真值，不把原物体照片当补全真值。第一阶段3DGS与最终3DGS也不是同一训练状态，不能将两者差别全部归为网格提取。','',
        '5. **几何图只是可视化。** 主对比的几何图采用原始 mesh 深度加深度梯度光照，GT 只投影稀疏点；这不是独立扫描真值，也不是精确法线渲染。下面另外量化实际三角形投影尺寸。三角形尺寸只表示离散密度，不等价于准确性；CD/F1 请看主报告。','',
        '此外，**F1 很高不代表网格细节清晰**：主阈值是相机半径1%，通常约等于当前3.84个体素（再粗化的场景约1.92个）。很多细节损失仍落在这个容许误差内，F1可能接近饱和；同时看 CD、0.5%阈值 F1、原尺寸放大图与投影三角形尺寸。','',
        '## 实际提取参数与局部三角形尺寸','',
        '下表体素/带宽来自实际全背景提取回执。第一阶段代码默认体素0.002、带宽0.008、深度上限5；原产物没有完整提取参数回执，因此不把代码默认值冒充历史实测。边长按固定展示视角、共同有效背景内可见面的最长边统计；逐面遍历，不降面。','',
        '| 场景 | 全背景体素 | TSDF带宽 | 第一阶段面数 | 全背景面数 | 第一阶段投影边长中位数(px) | 全背景投影边长中位数(px) |','|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        g=r['geometry_footprint'];p=r['whole_extraction']
        lines.append(f'| {r["scene"]} | {p["voxel_size"]:.8f} | {p["sdf_trunc"]:.8f} | {g["stage1"]["faces"]:,} | {g["whole"]["faces"]:,} | {g["stage1"]["pixel_edge"]["median"]:.4f} | {g["whole"]["pixel_edge"]["median"]:.4f} |')
    lines+=['','## 同视角原生图像证据','',
        '以下 PSNR 是所有留出视角的逐视角均值，只在同一六方法共同背景域计算。3DGS 两列仅用于诊断，不进入任何 mesh 方法主表。所有列继续对同一冻结 GT 评分；EDGS 自存 GT 使用 BICUBIC、当前评估 GT 使用 BILINEAR，这一既有差异已对原始照片核验并记录，没有为本次诊断改 GT。梯度能量比仅辅助判断细节/噪声，数值更高不保证重建更准。','',
        '| 场景 | 视角数 | 第一阶段3DGS | 第一阶段mesh | inpaint 3DGS | 全背景mesh | mesh相对inpaint 3DGS变化(dB) |','|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        m=r['means'];a=m['stage1_gaussian']['PSNR'];b=m['stage1_mesh']['PSNR'];c=m['inpaint_gaussian']['PSNR'];d=m['whole_background_mesh']['PSNR']
        lines.append(f'| {r["scene"]} | {r["held_out_views"]} | {a:.4f} | {b:.4f} | {c:.4f} | {d:.4f} | {d-c:+.4f} |')
    lines+=['','## 组合模式的生成物体','',
        '生成物体与全背景 mesh 是两个独立来源。当前 MV-SAM3D 官方 `run_weighted_inference` 调用明确为 `with_mesh_postprocess=False, with_texture_baking=False, use_vertex_color=True`；因此不经过官方可选的95%减面分支，但也没有启用纹理烘焙。下表从实际 `asset.npz` 读取面数与纹理图数量；材质号全为-1时渲染使用生成的顶点色。物体的圆滑几何来自生成结果及其姿态/形状变换，不能归给背景 TSDF 的体素；没有凭空补出真实观测中的全部高频形状与颜色。','',
        '已核对的生成回执仍为原始照片输入、官方物体裁剪到518，不是先把整张照片缩到768。`observation_max_size=768` 是位姿证据配置，不能据此认定当前生成输入也缩到了768；应看每个对象实际 `generation.json/input_checks`。本轮不修改生成参数或物体网格。','',
        '| 场景 | 物体ID | 原生顶点数 | 原生面数 | 实际纹理图数 | 生成输入策略 |','|---|---|---:|---:|---:|---|']
    for r in records:
        for obj in r['generated_objects']:
            lines.append(f'| {r["scene"]} | {obj["id"]} | {obj["vertices"]:,} | {obj["faces"]:,} | {len(obj["textures"])} | {obj["generation"].get("generation_input_policy","未记录")} |')
    lines+=['','## 详细图与当前结果','',
        '- [纯背景报告](../reports/background.md)：目标 mask 内不评分。',
        '- [组合报告](../reports/recomposition.md)：背景与生成物体一起评分。',
        '- [全场景八列 RGB 总图](../figures_all_methods/overview/all_scenes_rgb.png)。',
        '- [全场景八列几何总图](../figures_all_methods/overview/all_scenes_geometry.png)。',
        '- [机器可读核查结果](PAINTMESH_BLUR_AUDIT.json)：源产物身份、所有视角、逐面统计、图像对齐检查。','']
    lines += [f'- [{r["scene"]}：原图 → 第一阶段3DGS/mesh → inpaint 3DGS → 背景/组合mesh](../figures_all_methods/paintmesh_blur/{r["scene"]}_pipeline.png)' for r in records]
    lines+=['','## 比较范围说明','',
        '加入第一阶段后，主表按六方法共同覆盖的表面/像素重算。第一阶段原网格带原物体且远处可能因原深度范围缺失，不能称为“去除物体后的背景”。背景模式对它也排除同一目标 mask。共同域变化可能使所有方法分数改变，这不是重建质量自动改善；自身覆盖率和自身域指标仍保留。','',
        '几何参照为 COLMAP 稀疏点云，CD/F1 是已观测支持域内的一致性指标，不能称为独立扫描精度。F1 主阈值为相机半径1%，同时给出0.5%与2%。历史存在生成物体位姿待核查的场景继续标为 needs_review。','',
        '与 [From Blobs to Spokes 原文§5](https://arxiv.org/html/2604.07337v1#S5) 的关系：沿用“几何 + 真三角网格渲染”两类检查；原文几何使用 DTU/Tanks and Temples 扫描数据，并强调统一协议及网格完整度。这里按要求使用稀疏 COLMAP 观测与有效覆盖域，不惩罚空洞，因此不是原文完整 MBR/扫描精度协议，不能直接横比其论文表中数值。','',
        '后续若要恢复更清晰 mesh，优先固定高斯/相机/深度不变，以空间分块 TSDF 保持细体素，再独立验证颜色融合或原生纹理烘焙；不能靠减少测试视角、简化网格或缩图制造改善。本轮只定位问题、加入真实第一阶段列与重评估，未把建议冒充已完成优化。']
    (PUBLIC/'docs/PAINTMESH_COLOR_BLUR.md').write_text('\n'.join(lines)+'\n')
    (DEST/'README.md').write_text('# PaintMesh 原生细节诊断\n\nGaussian 图片仅用于定位提取前后差异，不属于 mesh 比较结果。各图保留原始评估分辨率，并使用同一 ROI。\n\n'+
        '\n'.join(f'- [{r["scene"]}]({r["scene"]}_pipeline.png)' for r in records)+'\n')
    print('PAINTMESH_BLUR_AUDIT_COMPLETE',len(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence-only',action='store_true')
    run(parser.parse_args().evidence_only)
