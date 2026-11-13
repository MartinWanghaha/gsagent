#!/usr/bin/env python3
"""Export interoperable mesh scenes and honest visual/metric delivery reports."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
from recompose.io import read_json,write_json,fingerprint
from recompose.evidence import SELECTION
from recompose.mesh_export import export_mesh_scene
from recompose.geometry import RayMesh,camera_rays
from recompose.texture import sample_colors
from recompose.render import render_composition
from recompose.report import write_status


def review_scene(scene_path,gpu):
    scene_path=Path(scene_path);scene=read_json(scene_path);out=scene_path.parent/'review';out.mkdir(exist_ok=True)
    refs=[scene_path,Path(__file__),Path(__file__).parent/'recompose/render.py']+[Path(n['evidence'])/'observations.json' for n in scene['objects']]
    signature=hashlib.sha256(''.join(fingerprint(p)['sha256'] for p in refs).encode()).hexdigest()
    receipt=out/'review.json'
    if receipt.exists() and read_json(receipt).get('signature')==signature and (out/'comparison.jpg').exists():return out/'comparison.jpg'
    node=scene['objects'][0];meta=read_json(Path(node['evidence'])/'observations.json')
    camera=next(v for v in meta['views'] if v['validation']);obs=Path(node['evidence'])/'observations'/Path(camera['name']).with_suffix('.npz')
    o=np.load(obs);rgb=o['rgb'];h,w=rgb.shape[:2]
    background=copy.deepcopy(scene);background['objects']=[];write_json(out/'background_scene.json',background)
    render_composition(out/'background_scene.json',obs,out/'background.png',f'cuda:{gpu}')
    render_composition(scene_path,obs,out/'composed.png',f'cuda:{gpu}')
    # Rasterize the actual generated mesh without blending it with the original object.
    # The completed background depth handles occlusion within the local object ROI.
    mesh_image=np.array(Image.open(out/'background.png'));yy,xx=np.mgrid[:h,:w]
    rays=camera_rays(o['K'],o['w2c'],np.c_[xx.ravel(),yy.ravel()])
    z_buffer=np.where(o['background_depth']>0,o['background_depth'],np.inf).ravel().copy()
    for n in scene['objects']:
        a=np.load(n['mesh']['path']);mesh=RayMesh(a['vertices'],a['faces'])
        depth,face,_=mesh.cast(rays,n['object_to_world']);visible=np.isfinite(depth)&(depth<=z_buffer+.003*meta['diameter'])
        colors=sample_colors(a,face[visible],np.full((visible.sum(),3),1/3))
        mesh_image.reshape(-1,3)[visible]=np.rint(np.clip(colors,0,1)*255).astype('uint8');z_buffer[visible]=depth[visible]
    Image.fromarray(mesh_image).save(out/'composed_mesh.png')
    images=[rgb,np.array(Image.open(out/'background.png')),mesh_image,np.array(Image.open(out/'composed.png'))]
    labels=['Original training-view RGB (held out)','Selected completed background','Generated mesh at estimated pose','Original deleted 3DGS + inpaint background']
    panel_w=480;panel_h=round(h*panel_w/w);sheet=Image.new('RGB',(4*panel_w,panel_h+65),'white');draw=ImageDraw.Draw(sheet)
    heading=f'{scene["input_selection"]["scene"]} | source={scene["input_selection"]["group"]} | {scene["status"]} | {camera["original"]}'
    draw.text((8,6),heading,fill='black' if scene['complete'] else 'red')
    for i,(image,label) in enumerate(zip(images,labels)):
        draw.text((i*panel_w+8,31),label,fill='black');sheet.paste(Image.fromarray(image).resize((panel_w,panel_h),Image.Resampling.LANCZOS),(i*panel_w,65))
    sheet.save(out/'comparison.jpg',quality=94)
    write_json(receipt,dict(signature=signature,scene=fingerprint(scene_path),camera=camera['original'],
                           status=scene['status'],description='Actual source RGB, background GS, mesh and joint GS render; no image-generation postprocessing.'))
    return out/'comparison.jpg'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]/'output/paintmesh/Inpaint360GS/recomposition'))
    p.add_argument('--gpu',type=int,default=1);p.add_argument('--wait-pids',default='');p.add_argument('--scenes',default='all')
    p.add_argument('--summary-only',action='store_true',help='Refresh the report from existing artifacts without mesh export or GPU rendering')
    args=p.parse_args();root=Path(args.root)
    while True:
        active=[]
        for pid in filter(None,args.wait_pids.split(',')):
            try:cmd=Path(f'/proc/{int(pid)}/cmdline').read_bytes()
            except FileNotFoundError:continue
            if b'run_recompose.py' in cmd or b'resume_recompose_outputs.py' in cmd:active.append(pid)
        if not active:break
        print('Waiting for scene writers:',','.join(active),flush=True);time.sleep(20)
    scenes=list(SELECTION) if args.scenes=='all' else args.scenes.split(',')
    rows=[];previews=[];errors=[]
    for name in scenes:
        folder=root/name;path=folder/'composition/scene.json'
        current=read_json(folder/'status.json') if (folder/'status.json').exists() else {}
        if current.get('state') not in ('complete','needs_review'):
            rows.append(f'| {name} | {SELECTION[name]} | {current.get("state","pending")} / {current.get("stage","pending")} | - | - | 本轮尚未完成，旧输出不计入本轮结果 |')
            continue
        if not path.exists():
            status=read_json(folder/'status.json') if (folder/'status.json').exists() else dict(state='pending')
            poses=list(folder.glob('objects/*/placement/pose.json'));metrics=[]
            for report in poses:
                pose=read_json(report);views=pose['validation'] or pose['train']
                metrics.append(f"{report.parent.parent.name}: {np.median([v['iou'] for v in views]):.3f}")
            rows.append(f'| {name} | {SELECTION[name]} | {status["state"]} / {status.get("stage","candidate export pending")} | {"; ".join(metrics) or "-"} | - | 组合导出排队中；见 STATUS.md |')
            continue
        try:
            scene=read_json(path);print(f'Exporting mesh scene and review: {name}',flush=True)
            preview=path.parent/'review/comparison.jpg'
            if not args.summary_only:
                export_mesh_scene(scene,path.parent/'world')
                # Respect our active inference/appearance jobs when allocating render memory.
                from recompose.pipeline import gpu_slot
                with gpu_slot(args.gpu):
                    try:preview=review_scene(path,args.gpu)
                    finally:
                        import gc,torch
                        gc.collect()
                        with torch.cuda.device(args.gpu):torch.cuda.empty_cache()
            if preview.exists():previews.append(preview)
            metrics=[];reasons=[]
            for node in scene['objects']:
                pose=read_json(node['pose_report']['path']);iou=np.median([v['iou'] for v in pose['validation']])
                metrics.append(f"{node['id']}: {iou:.3f}");reasons+=pose['reasons']
            preview_link=f'[图]({preview.relative_to(root)})' if preview.exists() else '渲染对照待生成'
            rows.append(f"| {name} | {SELECTION[name]} | {scene['status']} | {'; '.join(metrics)} | [{path.parent.name}]({path.parent.relative_to(root)}/scene.json) | {preview_link} {'; '.join(sorted(set(reasons)))} |")
        except Exception as e:
            errors.append(dict(scene=name,error=str(e)));print(name,repr(e),flush=True)
            rows.append(f'| {name} | {SELECTION[name]} | export/review error | - | - | {e} |')
    if previews:
        pages=[Image.open(p).convert('RGB') for p in previews];width=1600;scaled=[im.resize((width,round(im.height*width/im.width)),Image.Resampling.LANCZOS) for im in pages]
        sheet=Image.new('RGB',(width,40+sum(im.height for im in scaled)),'white');y=40
        ImageDraw.Draw(sheet).text((10,12),f'Available rendered scenes: {len(previews)} / 11. Pending scenes are not shown; see README.md / STATUS.md.',fill='black')
        for im in scaled:sheet.paste(im,(0,y));y+=im.height
        sheet.save(root/'all_scenes_comparison.jpg',quality=93)
    text=['# 组合式可移动 3DGS + mesh：实际交付记录','',
          '背景按指定的 fitlama / best_results / fit 选择。所有新产物位于本目录，原始重建结果保持原样。',
          '', '**3DGS = inpaint 背景 + 实际删除的原始物体高斯，不重新训练。mesh = inpaint 背景 + MV-SAM3D 物体 mesh。**',
          '', '当前生成输入：原始分辨率 RGB → 官方 1.2 倍物体裁剪 → 518；相机与 pointmap 同步配准。选图、锚点和留出验证保持原有尺度。详见 NATIVE_INPUT_PLAN.md。',
          '', '**complete** 表示 mesh 通过当前轮廓/支撑/碰撞阈值；**needs_review** 表示 mesh 位姿仍待检查。原始高斯的精确恢复独立于 mesh 验收，状态写在 components.gaussians 中。',
          '', '| 场景 | 来源 | 状态 | 物体 ID：留出轮廓 IoU | 场景图 | 对照图 / 问题 |','|---|---|---|---|---|---|',*rows,'',
          '## 打开文件','',
          '- `<scene>/composition/world/composed_gaussians.ply`：inpaint 背景 + 原始删除高斯。各项原始渲染属性直接复制，可在标准 3DGS 查看器中打开。',
          '- 同目录 `mesh_scene.gltf` + `mesh_buffers/`：全分辨率背景 mesh 和独立物体节点，可在 Blender 等 glTF 工具中导入；保留独立变换。此版本使用顶点颜色；`<id>_textured.glb` 保留生成器原生材质，不表示一定有 UV 纹理。本批未启用 UV 纹理烘焙。',
          '- `scene.json`：mesh 使用 object_to_world；原始世界坐标高斯使用 Δ = object_to_world × inverse(original_pose)。初始 Δ=I，移动时两种表示同步。',
          '- `world/gaussian_ranges.json`：合并 PLY 中背景和各物体的实例行范围。',
          '- `<scene>/original_gaussians/manifest.json`：精确集合差、源模型、实际删除点数及每物体点数；training=false。',
          '- `<scene>/original_gaussians/<id>/original_indices.npy`：每个恢复高斯在去除前模型中的原始行号。',
          '- `all_scenes_comparison.jpg`：原图、补全背景、生成 mesh、组合 3DGS 的真实渲染对照。',
          '- `car/movement_demo/`：实际平移、重新验证与导出的示例（若生成成功）。',
          '', '## 当前限制','',
          'fruits、garden_toys 等候选的补全背景高度与原物体观测不一致。doppelherz 的原语义 ID 18 同时覆盖盒子与旁边白色瓶状物，新增 target_isolation 根据实际删除点剔除保留的同 ID 物体，重建状态以表格及 STATUS.md 为准。redbull 的细杆结构与留出语义 mask 的一致性不足，真实形状误差和 mask 误差仍需区分。问题保留在候选报告中，不靠放宽阈值消除。',
          '', '不再从生成 mesh 采样高斯，也不再做 SH、透明度、尺度或颜色训练。原始物体高斯与生成 mesh 是同一物体的两个独立表示，通过共同刚体运动关联；不是逐顶点绑定。移动不包含动态重打光或阴影更新。',
          '', '实现和命令说明：[OBJECT_RECOMPOSITION.zh-CN.md](../../../../scripts/paintmesh/OBJECT_RECOMPOSITION.zh-CN.md)。',
          '',f'报告更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}。']
    audit=root/'original_gaussian_audit.json'
    if audit.exists():
        check=read_json(audit)
        text.insert(6,f"实际数据逐属性核对：{sum(v['passed'] for v in check['scenes'])}/11 场景通过；核对背景及恢复物体的 XYZ、SH、opacity、scale、rotation 与原始记录完全相同。")
    if (root/'SCENE_CHECKS.zh-CN.md').exists():
        text.extend(['','重新检测：[全部场景完整性与摆放](SCENE_CHECKS.zh-CN.md)；[逐图观察](checks_all_scenes/VISUAL_REVIEW.zh-CN.md)。'])
    (root/'README.md').write_text('\n'.join(text)+'\n');write_json(root/'delivery_errors.json',errors)
    write_status(root)
    return bool(errors)


if __name__=='__main__':raise SystemExit(main())
