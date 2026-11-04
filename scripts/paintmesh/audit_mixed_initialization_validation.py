#!/usr/bin/env python3
"""Audit a finished screening pilot without relaxing its registered policy."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts/paintmesh'),str(REPO/'submodules/EDGS')]
import numpy as np
from PIL import Image
from plyfile import PlyData,PlyElement
from verify_mixed_initialization import read,write,warp_observations,interpolate
from source.paintmesh_edgs_init import camera,project,unproject
from source.paintmesh_local_data import make_views
from local_geometry_io import record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);args=p.parse_args()
    root=args.root.resolve(strict=True);req=read(root/'request.json');summary=read(root/'summary.json')
    initial=read(req['baseline']['path']);lama=read(initial['inputs']['lama']['path'])
    inputs=read(lama['input_manifest']['path']);support=dict(np.load(req['support']['path']))
    evidence=dict(np.load(root/'point_evidence.npz'));n=req['candidate_count'];policy=req['policy']
    cameras=[camera(v) for v in make_views(initial['inputs']['camera']['path'],device='cpu')]
    gates={};rgb_masks=[]
    for role in ('fit','check'):
        available=evidence[role+'_depth_available'][:n];agree=evidence[role+'_depth_agree'][:n]
        conditions=dict(depth_views=agree>=policy['min_depth_views'],
            depth_agreement=agree/np.maximum(available,1)>=policy['min_depth_agreement'],
            pseudo_front=evidence[role+'_depth_front'][:n]/np.maximum(available,1)<=policy['max_pseudo_front_fraction'],
            known_front=evidence[role+'_depth_known_front'][:n]<=policy['max_known_front_views'],
            rgb_edges=evidence[role+'_rgb_edges'][:n]>=policy['min_rgb_edges'],
            rgb_views=evidence[role+'_rgb_views'][:n].sum(1)>=policy['min_rgb_views'],
            sigma=evidence[role+'_rgb_sigma_spacing'][:n]<=policy['max_sigma_spacing'])
        gates[role]={key:int(value.sum()) for key,value in conditions.items()}
        rgb_masks.append(conditions['rgb_edges']&conditions['rgb_views']&conditions['sigma'])
    core=rgb_masks[0]&rgb_masks[1];indices=np.flatnonzero(core);xyz=support['xyz'][indices]
    control_core=np.ones(len(evidence['xyz'])-n,dtype=bool)
    for role in ('fit','check'):
        control_core &= ((evidence[role+'_rgb_edges'][n:]>=policy['min_rgb_edges'])
            &(evidence[role+'_rgb_views'][n:].sum(1)>=policy['min_rgb_views'])
            &(evidence[role+'_rgb_sigma_spacing'][n:]<=policy['max_sigma_spacing']))
    vertex=PlyData.read(initial['outputs']['ply']['path'],mmap=True)['vertex'].data
    new=vertex[np.load(initial['outputs']['editable_mask']['path'])]
    np.testing.assert_array_equal(np.column_stack([new[k] for k in 'xyz']),support['xyz'])
    PlyData([PlyElement.describe(new[indices].copy(),'vertex')]).write(str(root/'rgb_consistent_core_gaussians.ply'))
    np.save(root/'rgb_consistent_core_indices.npy',indices,allow_pickle=False)
    dtype=[(key,'f4') for key in 'xyz']+[(key,'u1') for key in ('red','green','blue')]
    points=np.zeros(len(indices),dtype=dtype)
    for k,key in enumerate('xyz'):points[key]=xyz[:,k]
    for k,key in enumerate(('red','green','blue')):points[key]=np.rint(support['rgb'][indices,k]*255).clip(0,255).astype('u1')
    PlyData([PlyElement.describe(points,'vertex')]).write(str(root/'rgb_consistent_core_points.ply'))
    original_mask_votes=np.zeros(len(indices),int);visible=np.zeros(len(indices),int)
    coverage=[];plot_data=[]
    from scipy.ndimage import distance_transform_edt
    for frame in range(len(cameras)):
        stem=f'{frame:05d}';uv,z,valid=project(xyz,cameras[frame]);safe=np.where(valid[:,None],uv,0.)
        with Image.open(inputs['frames'][stem]['inputs']['mask']['path']) as im:
            original_mask=np.asarray(im.convert('L'))>0
        original_mask_votes += valid&(interpolate(original_mask.astype(float),safe,0)>0)
        visible += valid
        if frame not in (0,30,60,90,120,150):continue
        with Image.open(inputs['frames'][stem]['outputs']['color_mask']['path']) as im:
            mask=np.asarray(im.convert('L'))>0
        image=np.asarray(Image.open(lama['frames'][stem]['outputs']['color']['path']).convert('RGB'))
        pixels=np.rint(uv[valid]).astype(int)
        raster=np.zeros_like(mask)
        if len(pixels):raster[pixels[:,1],pixels[:,0]]=True
        radius=float(np.median(req['spacing_world']*np.linalg.norm(cameras[frame]['w2c'][2,:3])
            *cameras[frame]['k'][0,0]/np.maximum(z[valid],1e-8))) if valid.any() else 0.
        dist=distance_transform_edt(~raster) if raster.any() else np.full(mask.shape,np.inf)
        coverage.append(dict(frame=stem,projected_points=int(valid.sum()),
            approximate_spacing_footprint_pixels=radius,
            completed_mask_coverage=float((dist[mask]<=radius).mean()),
            original_mask_coverage=float((dist[original_mask]<=radius).mean())))
        plot_data.append((stem,image,original_mask,uv[valid]))
    positive=[]
    for pair in req['pairs']:
        a,b=pair['a'],pair['b'];stems=[f'{k:05d}' for k in (a,b)]
        masks=[np.asarray(Image.open(inputs['frames'][s]['outputs']['color_mask']['path']).convert('L'))>0 for s in stems]
        depth=[np.load(inputs['frames'][s]['inputs']['removed_depth']['path']) for s in stems]
        alpha=np.load(inputs['frames'][stems[0]]['inputs']['removed_alpha']['path'])
        yy,xx=np.where(~masks[0]&(depth[0]>0)&(alpha>.95))
        selected=np.random.default_rng(0).choice(len(xx),min(10000,len(xx)),replace=False)
        uv=np.column_stack((xx[selected],yy[selected])).astype(float)
        points=unproject(uv,depth[0][yy[selected],xx[selected]],cameras[a]);buv,bz,bv=project(points,cameras[b])
        safe_b=np.where(bv[:,None],buv,0.)
        data=np.load(root/'match_cache'/f"{pair['role']}_{a:05d}_{b:05d}.npz")
        matched,confidence,cycle,valid=warp_observations(data['forward'],data['forward_confidence'],
            data['reverse'],data['reverse_confidence'],uv,masks[0].shape,masks[1].shape)
        valid &= bv&(interpolate(masks[1].astype(float),safe_b,0)==0)
        tolerance=policy['depth_tolerance_spacing']*req['spacing_world']*np.linalg.norm(cameras[b]['w2c'][2,:3])
        valid &= np.abs(bz-interpolate(depth[1],safe_b))<=tolerance
        confident=valid&(confidence>=policy['roma_confidence'])&(cycle<=policy['cycle_pixels'])
        error=np.linalg.norm(matched-buv,axis=1);accepted=confident&(error<=policy['reprojection_pixels'])
        positive.append(dict(**pair,visible_depth_consistent=int(valid.sum()),
            confident_cycle_consistent=int(confident.sum()),reprojection_passed=int(accepted.sum()),
            reprojection_pass_among_confident=float(accepted.sum()/max(1,confident.sum())),
            median_reprojection_error=float(np.median(error[confident])) if confident.any() else None))
    write(root/'unedited_background_control.json',positive)
    stats={}
    for role in ('fit','check'):
        edges=evidence[role+'_rgb_edges'][:n][core]
        stats[role]=dict(rgb_views=evidence[role+'_rgb_views'][:n][core].sum(1).tolist(),
            median_reprojection_error_pixels=float(np.median(evidence[role+'_rgb_error_sum'][:n][core]/np.maximum(edges,1))) if len(indices) else None,
            max_conditional_sigma_spacing=float(evidence[role+'_rgb_sigma_spacing'][:n][core].max()) if len(indices) else None,
            median_depth_agreement=float(np.median(evidence[role+'_depth_agree'][:n][core]/np.maximum(evidence[role+'_depth_available'][:n][core],1))) if len(indices) else None)
    confidence_total=sum(p['confident_cycle_consistent'] for p in positive)
    control=dict(confident_correspondences=confidence_total,passed=sum(p['reprojection_passed'] for p in positive),
        pass_fraction=sum(p['reprojection_passed'] for p in positive)/max(1,confidence_total))
    inside_any=original_mask_votes>0
    audit=dict(registered_joint_policy_unchanged=True,registered_joint_selected=summary['populations']['mixed']['strict_selected'],
        rgb_only_core_count=len(indices),rgb_only_core_fraction=len(indices)/n,
        rgb_only_core_is_joint_high_confidence=False,statistics=stats,
        core_inside_original_mask_in_any_view=int(inside_any.sum()),
        core_outside_original_mask_in_all_views=int((~inside_any).sum()),
        conclusion='No sufficiently supported point cloud in the actual repair interior in this pilot.',
        perturbed_rgb_core_count=int(control_core.sum()),perturbed_count=len(control_core),
        original_mask_projection_fraction=(original_mask_votes/np.maximum(visible,1)).tolist(),
        mask_coverage=coverage,gates=gates,unedited_background_control=control,script=record(__file__),
        caveat='RGB-only core is exploratory modality-specific evidence; fails registered joint policy. No claim of accurate hidden geometry.')
    write(root/'audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,(stem,image,mask,uv) in zip(axes.flat,plot_data):
        yy,xx=np.where(mask);y0=max(0,int(yy.min())-50);y1=min(image.shape[0],int(yy.max())+51)
        x0=max(0,int(xx.min())-50);x1=min(image.shape[1],int(xx.max())+51)
        ax.imshow(image);ax.contour(mask,levels=[.5],colors=['#ff4444'],linewidths=.8)
        ax.scatter(uv[:,0],uv[:,1],s=15,facecolors='none',edgecolors='#00ff88',linewidths=.8)
        ax.set(xlim=(x0,x1),ylim=(y1,y0),title=f'{stem}: RGB-consistent core, original mask in red');ax.axis('off')
    fig.savefig(root/'rgb_core_projection.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(12,5),constrained_layout=True)
    center=np.median(support['xyz'],axis=0)
    cam_positions=np.array([c['c2w'][:3,3] for c in cameras]);_,_,basis=np.linalg.svd(cam_positions-cam_positions.mean(0),full_matrices=False)
    coords=(support['xyz']-center)@basis.T
    for index,(title,rows) in enumerate((('Original mixed',np.arange(n)[::5]),('RGB-only consistent core',indices))):
        ax=fig.add_subplot(1,2,index+1,projection='3d')
        ax.scatter(*coords[rows].T,c=support['rgb'][rows],s=1 if index==0 else 15)
        ax.set(title=f'{title}: {n if index==0 else len(indices)}',xlabel='Camera-PCA X',ylabel='Camera-PCA Y',zlabel='Camera-PCA Z')
        lo,hi=np.quantile(coords,[.005,.995],axis=0);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2])
    fig.savefig(root/'rgb_core_geometry.png',dpi=160);plt.close(fig)
    report=root/'report.zh-CN.md'
    lines=['','## 固定联合门槛的失败原因与独立诊断','',
        f"严格联合门槛仍为 {audit['registered_joint_selected']} 点；阈值没有修改。两组RGB证据单独均通过的候选为 {len(indices)} 点，占 {len(indices)/n:.4%}。它们没有通过联合门槛，不能作为RGB与depth共同支持的高置信度点。",'',
        f"RGB-only core复查组重投影误差中位数为 {stats['check']['median_reprojection_error_pixels']} 像素，depth一致率中位数为 {stats['check']['median_depth_agreement']}。扰动对照的RGB-only core通过数为 {int(control_core.sum())}/{len(control_core)}；这是敏感性对照，不是准确率。",'',
        f"**修复区内部验证未通过：这 {len(indices)} 个RGB-only点中，{int((~inside_any).sum())} 个在全部180帧投影中均位于原始目标mask之外；落入任一原始目标mask的只有 {int(inside_any.sum())} 个。它们来自膨胀mask覆盖的周边背景，不能作为洞内补全几何的成功证据。** 红色轮廓是未膨胀目标mask，绿色点是RGB-only core。",'',
        f"未编辑背景对照：有高置信度、双向闭环支持的 {control['confident_correspondences']} 次匹配中，{control['pass_fraction']:.2%} 通过同样的2像素重投影门槛。该对照支持坐标/匹配流程可工作，但不是GT精度证明。",'',
        '联合检查要求75%的补全深度观测同意，是保守的诊断门槛；不能由其失败直接推断候选几何错误，尤其输入depth本就不准。后续融合应显式建模异常值和相关偏差，而非将此门槛原样用于最终初始化。','',
        '![RGB-only core与原始mask](rgb_core_projection.png)','','![RGB-only core空间分布](rgb_core_geometry.png)','',
        'RGB-only候选文件：rgb_consistent_core_gaussians.ply / rgb_consistent_core_points.ply。这是探索性几何核心，不替代空的联合高置信度点云。详细门槛淘汰计数、投影覆盖与对照在audit.json。', '']
    original_report=report.read_text().split('\n## 固定联合门槛的失败原因与独立诊断')[0].rstrip()+'\n'
    report.write_text(original_report+'\n'.join(lines))
    print({k:v for k,v in audit.items() if k not in ('mask_coverage','gates','original_mask_projection_fraction','script')})


if __name__=='__main__':main()
