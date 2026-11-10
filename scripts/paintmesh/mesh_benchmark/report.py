"""Two separate reports; incomplete runs never enter means as zeros."""
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .core import SCENES,METHODS,STATES,RECOMPOSE,read,write

LABEL={'2dgs':'2DGS','come':'CoMe','gof':'GOF','gaussian_wrapping':'Gaussian Wrapping','paintmesh':'PaintMesh'}

def previews(root,state,metrics):
    """A deterministic first shared camera; never select a favorable test image."""
    dest=root/state/'figures';dest.mkdir(exist_ok=True)
    produced=[]
    for scene in SCENES:
        complete=[m for m in METHODS if (m,scene) in metrics]
        if not complete:continue
        com=root/'common'/scene/'complete.json'
        info=read(com)
        camera=next((c for c in info['cameras'] if c['part']=='held_out' and
             any(r['name']==c['name'] and r.get('PSNR') is not None for m in complete for r in metrics[m,scene]['per_view'])),None)
        if camera is None:continue
        name=camera['stem'];signature='|'.join(f'{m}:{(root/state/m/scene/"metrics.json").stat().st_mtime_ns}' for m in complete)
        marker=dest/(scene+'.signature');imagefile=dest/(scene+'.jpg')
        produced.append((scene,imagefile.name))
        if marker.exists() and marker.read_text()==signature and imagefile.exists():continue
        panels=[('GT + shared target outline',root/'common'/scene/'held_out'/(name+'_gt.png'),None)]
        for method in METHODS:
            metric=metrics.get((method,scene))
            panels.append((LABEL[method],root/state/method/scene/'renders'/(name+'.png'),metric))
        width=480;height=round(camera['height']/camera['width']*width)
        canvas=Image.new('RGB',(width*3,(height+65)*2+36),'white');draw=ImageDraw.Draw(canvas)
        draw.text((12,10),f'{scene} | {state} | {camera["name"]} | fixed geometry + shared learned appearance',fill='black')
        for i,(title,path,metric) in enumerate(panels):
            x=i%3*width;y=i//3*(height+65)+36
            draw.text((x+8,y+5),title,fill='black')
            if path.exists():
                picture=Image.open(path).convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
                evaluation=root/'common'/scene/(state+'_support')/(name+'.png')
                if evaluation.exists():
                    valid=np.asarray(Image.open(evaluation).resize((width,height),Image.Resampling.NEAREST))>127
                    pixels=np.array(picture);pixels[~valid]=(pixels[~valid]*.2+160*.8).astype('uint8');picture=Image.fromarray(pixels)
                if i==0:
                    import cv2
                    mask=np.asarray(Image.open(root/'common'/scene/'held_out'/(name+'_foreground.png')).resize((width,height),Image.Resampling.NEAREST))>127
                    edge=mask&~cv2.erode(mask.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
                    pixels=np.array(picture);pixels[edge]=[255,50,180];picture=Image.fromarray(pixels)
                canvas.paste(picture,(x,y+30))
                if metric:
                    view=next(r for r in metric['per_view'] if r['name']==camera['name'])
                    if view.get('PSNR') is not None:
                        draw.text((x+8,y+height+36),f'PSNR {view["PSNR"]:.2f} | SSIM {view["SSIM"]:.3f} | LPIPS {view["LPIPS"]:.3f} | own {view["mesh_coverage"]:.1%}',fill='black')
            else:draw.text((x+20,y+65),'Pending mesh or evaluation',fill='gray')
        canvas.save(imagefile,quality=92);marker.write_text(signature)
    return produced

def update(output):
    from .layout import work_root,publish_reports
    root=work_root(output)
    if not (root/'protocol.json').exists():return
    protocol=read(root/'protocol.json');cohorts=protocol.get('cohorts',{});rows=[]
    fmt=lambda v:f'{v:.6f}' if v is not None else '—'
    for state in STATES:
        report=root/state;report.mkdir(parents=True,exist_ok=True)
        title='mask 外纯背景 mesh' if state=='background' else '背景 + MV-SAM3D 组合 mesh'
        text=[f'# {title}：仅有效表面/像素评估（v2）','',
              f'更新时间：{datetime.now().isoformat(timespec="seconds")}。'+('本目录为重新提取的全背景网格评估，原评估目录保持独立。' if (root/'whole_scene_plan.json').exists() else '已覆盖旧版全画面/空洞惩罚评估。'),'',
              '**主表只比较同场景所有已完成方法共同覆盖的表面和像素。** 另列各自覆盖域的结果。空洞和未覆盖像素不进入分数；纯背景目标 mask 内部也不进入分数。覆盖率只作范围说明，不作为误差惩罚。','',
              '几何参照为 COLMAP 稀疏点云，不是独立扫描真值。只在参考点真实 COLMAP 观测处采样可见 mesh 表面；不按误差或 F1 阈值筛点。CD 为双向平均最近距离的半和；F1 主阈值为共同训练相机半径的1%，另报告0.5%/2%。这些是覆盖范围内的条件指标，不是全场景完整度指标。','',
              'PSNR 按有效像素 MSE 计算；SSIM 的局部统计按有效像素重新归一化；LPIPS 两幅输入无效区同时置零 RGB，距离图仅在有效区平均。固定原三角网格，统一 VM-SH 颜色场；不使用高斯渲染。已有颜色场/渲染缓存复用，未重训几何。','',
              '数据使用原始有物体图像的共同留出相机；无物体 test* 不作为组合 RGB 真值。已有 PaintMesh 几何/证据可能使用过这些相机，因此是已有结果审计，不能声称严格同训练划分论文复现。','',
              '## 共同覆盖域结果','',
              '| 场景 | 方法 | CD-L1 ↓ | F1@1% ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 共同像素比例 | 自身覆盖率 | 参考点数 | 状态 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        metrics={};geometries={}
        for scene in SCENES:
            for method in METHODS:
                gp=root/'cache'/method/scene/state/'geometry_metrics.json';mp=report/method/scene/'metrics.json'
                g=read(gp) if gp.exists() else None;m=read(mp) if mp.exists() else None
                if g and g.get('scoring_protocol_version')!=2:g=None
                if m and m.get('scoring_protocol_version')!=2:m=None
                if g:geometries[method,scene]=g
                if m:metrics[method,scene]=m
                mm=m['mean'] if m else {}
                status='完成' if m else ('像素重评估中' if g else '网格缓存/共同支持域待完成')
                if method not in cohorts.get(scene,[]):status='源 mesh 未完成，本轮不纳入交集'
                if method=='paintmesh':
                    d=RECOMPOSE/scene/'composition/scene.json'
                    if d.exists() and read(d)['status']!='complete':status+='；组合需复核'
                text.append('| '+' | '.join([scene,LABEL[method],fmt(g.get('CD_L1') if g else None),fmt(g.get('F1') if g else None),
                     fmt(mm.get('PSNR')),fmt(mm.get('SSIM')),fmt(mm.get('LPIPS')),fmt(mm.get('common_coverage')),fmt(mm.get('mesh_coverage')),
                     str(g['reference_points']) if g else '—',status])+' |')
                rows.append(dict(scoring_protocol_version=2,state=state,method=method,scene=scene,status=status,geometry=g,render=mm))
        shared=[s for s in SCENES if all((m,s) in metrics and metrics[m,s]['mean'].get('PSNR') is not None and metrics[m,s]['geometry'].get('CD_L1') is not None for m in METHODS)]
        text+=['','## 五方法共同场景均值','','只对五方法均完成且有有效支持域的场景做宏平均；缺失不记0。几何跨场景平均使用除以共同相机半径的 CD。','',
               '共同场景：'+(', '.join(shared) if shared else '暂无，重评估运行中。'),'',
               '| 方法 | 场景数 | 归一化 CD-L1 ↓ | F1@1% ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for method in METHODS:
            selected=[metrics[method,s] for s in shared];values=[]
            for section,key in [('geometry','CD_L1_normalized'),('geometry','F1'),('mean','PSNR'),('mean','SSIM'),('mean','LPIPS')]:
                values.append(fmt(float(np.mean([m[section][key] for m in selected]))) if selected else '—')
            text.append('| '+' | '.join([LABEL[method],str(len(shared)),*values])+' |')
        text+=['','## 各自覆盖域结果','','此表同样排除空洞和背景目标 mask；各方法参与评分的区域不同，与主表分开，不作为同面积排名。','',
               '| 场景 | 方法 | CD-L1 ↓ | F1@1% ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 自身覆盖率 |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
        for (method,scene),m in metrics.items():
            g=m['geometry'].get('own_surface') or {};a=m['own_surface_mean']
            text.append('| '+' | '.join([scene,LABEL[method],fmt(g.get('CD_L1')),fmt(g.get('F1')),fmt(a.get('PSNR')),fmt(a.get('SSIM')),fmt(a.get('LPIPS')),fmt(m['mean']['mesh_coverage'])])+' |')
        text+=['','## F1 阈值与几何支持范围','',
               '| 场景 | 方法 | F1@0.5% | P@1% | R@1% | F1@1% | F1@2% | τ@1%（COLMAP单位） | 参考点参与比例 |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for (method,scene),g in geometries.items():
            fs=g.get('F_scores',{})
            text.append('| '+' | '.join([scene,LABEL[method],fmt(fs.get('0.005',{}).get('F1')),fmt(g.get('precision')),fmt(g.get('recall')),fmt(g.get('F1')),
                    fmt(fs.get('0.02',{}).get('F1')),fmt(fs.get('0.01',{}).get('threshold_colmap_units')),fmt(g.get('reference_support_fraction'))])+' |')
        text+=['','## 原生颜色/纹理附加结果','','不训练统一颜色场的原生 mesh 渲染；只对共同有效像素评分。无颜色或纹理的导出不填分数。','',
               '| 场景 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |','|---|---|---:|---:|---:|']
        for (method,scene),m in metrics.items():
            n=m.get('native_color_mean')
            if n:text.append('| '+' | '.join([scene,LABEL[method],*[fmt(n.get(k)) for k in ['PSNR','SSIM','LPIPS']]])+' |')
        figures=previews(root,state,metrics)
        if figures:
            text+=['','## 对比图','','使用首个共同可评分视角；灰色区域不参与当前主表评分。','']
            text += [f'- [{s}](figures/{p})' for s,p in figures]
        text+=['','## 文件与协议','',
               '- `metrics.json`：本轮 v2 逐视角分数、共同域/自身域/原生颜色结果、覆盖率与 F1 阈值。',
               '- `../common/<scene>/<state>_support/`：共同有效 mask、参考几何样本、方法集合及覆盖统计。',
               '- `diagnostics/*_evaluation_mask.png`：实际参与当前分数的像素。',
               '- [完整协议](../PROTOCOL.zh-CN.md)。颜色场是 MILo 思路的256网格适配，不是原论文完整 MBR 协议；本轮按要求忽略空洞。']
        path=report/'report.md';contents='\n'.join(text)+'\n'
        from .layout import configuration,document_links,public_root
        if configuration(root):
            destination=public_root(root)/'results'/state/'report.md'
            contents=document_links(contents,path,destination);path=destination
            contents=contents.replace('`metrics.json`','`results/'+state+'/<method>/<scene>/metrics.json`')
            contents=contents.replace('`../common/<scene>/<state>_support/`','`_internal/common/<scene>/<state>_support/`')
            contents=contents.replace('`diagnostics/*_evaluation_mask.png`','`results/'+state+'/<method>/<scene>/diagnostics/*_evaluation_mask.png`')
        tmp=path.with_suffix('.tmp');tmp.write_text(contents);tmp.replace(path)
    write(root/'results.json',rows)
    (root/'README.md').write_text('# Mesh 重新评估 v2\n\n- [纯背景报告](background/report.md)\n- [组合 mesh 报告](recomposition/report.md)\n- [协议](PROTOCOL.zh-CN.md)\n\n仅计算有效表面/像素；旧评估已覆盖。主表为共同覆盖域，另列自身覆盖域。status.json 和 campaign.log 显示进度。\n')
    if (root/'whole_scene_plan.json').exists():
        (root/'README.md').write_text('# 全背景 mesh 对比\n\n- [纯背景报告](background/report.md)\n- [组合 mesh 报告](recomposition/report.md)\n- [提取方案](WHOLE_SCENE_PLAN.zh-CN.md)\n- [实际融合参数及内存调整](FUSION_MEMORY_ADJUSTMENTS.md)\n- [整体进度](WHOLE_STATUS.md)\n- [全方法图片](figures_all_methods/README.md)\n- [各方法独立子图](figures_all_methods/subplots/README.md)\n\n重新从现有 PaintMesh 高斯提取完整可观测背景。采用有效表面/像素协议；主表为共同覆盖域，另列各自覆盖域。原评估目录保持独立。\n')

    publish_reports(root)
