#!/usr/bin/env python3
"""Interpolate COLMAP test* poses, without opening any test RGB or masks."""
from __future__ import annotations
import argparse
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from run_geometry_ablation import REPO, read, write
from edgs_inpaint_io import record
from virtual_render_io import camera_contract


def continuous_order(centers, rotations):
    """Deterministic open nearest-pose path; 2-opt removes unnecessary jumps."""
    n=len(centers)
    radius=max(float(np.linalg.norm(centers-np.median(centers,axis=0),axis=1).max()),1e-8)
    distances=np.linalg.norm(centers[:,None]-centers[None,:],axis=-1)/radius
    quats=Rotation.from_matrix(rotations).as_quat()
    angles=2*np.arccos(np.clip(np.abs(quats@quats.T),0,1))
    costs=distances+angles
    order=[0];remaining=set(range(1,n))
    while remaining:
        nxt=min(remaining,key=lambda j:(costs[order[-1],j],j));order.append(nxt);remaining.remove(nxt)
    for _ in range(n):
        improved=False
        for i in range(1,n-1):
            for j in range(i+1,n):
                before=costs[order[i-1],order[i]]
                after=costs[order[i-1],order[j]]
                if j+1<n:
                    before+=costs[order[j],order[j+1]];after+=costs[order[i],order[j+1]]
                if after+1e-10<before:
                    order[i:j+1]=reversed(order[i:j+1]);improved=True
        if not improved:break
    return np.array(order)


def interpolate(centers, rotations, count):
    """Open acquisition-order path: linear world centers, SO(3) orientations.

    Every anchor is retained. Remaining samples are allocated by combined
    translational/angular arc length; never interpolate w2c translations.
    """
    centers=np.asarray(centers,dtype=np.float64);rotations=np.asarray(rotations,dtype=np.float64)
    n=len(centers)
    if n<2 or count<n:raise ValueError('Need at least two anchors and at least one frame per anchor')
    radius=max(float(np.linalg.norm(centers-np.median(centers,axis=0),axis=1).max()),1e-8)
    lengths=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    angles=(Rotation.from_matrix(rotations[:-1]).inv()*Rotation.from_matrix(rotations[1:])).magnitude()
    # Hard continuity gates bound each adjacent rendering, including large
    # acquisition jumps. If fixed view budget cannot satisfy them, stop.
    segments=np.maximum(1,np.maximum(np.ceil(angles/np.deg2rad(15)),np.ceil(lengths/(.2*radius)))).astype(int)
    if segments.sum()+1>count:
        raise ValueError(f'{count} frames cannot preserve anchors and continuity; need {segments.sum()+1}')
    weights=lengths/radius+angles
    if weights.sum()==0:weights=np.ones_like(weights)
    extra=(count-1-segments.sum())*weights/weights.sum();whole=np.floor(extra).astype(int)
    segments+=whole
    for i in np.argsort(-(extra-whole),kind='stable')[:count-1-segments.sum()]:segments[i]+=1
    out_c=[];out_r=[];mapping=[];anchor_indices=[]
    for i,steps in enumerate(segments):
        anchor_indices.append(len(out_c))
        t=np.arange(steps)/steps
        out_c.extend((1-t[:,None])*centers[i]+t[:,None]*centers[i+1])
        out_r.extend(Slerp([0.,1.],Rotation.from_matrix(rotations[i:i+2]))(t).as_matrix())
        mapping.extend(dict(left=i,right=i+1,fraction=float(v)) for v in t)
    anchor_indices.append(len(out_c));out_c.append(centers[-1]);out_r.append(rotations[-1])
    mapping.append(dict(left=n-1,right=n-1,fraction=0.))
    return np.array(out_c),np.array(out_r),mapping,anchor_indices,radius


def minimum_frame_count(centers, rotations):
    radius=max(float(np.linalg.norm(centers-np.median(centers,axis=0),axis=1).max()),1e-8)
    lengths=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    angles=(Rotation.from_matrix(rotations[:-1]).inv()*Rotation.from_matrix(rotations[1:])).magnitude()
    return int(np.maximum(1,np.maximum(np.ceil(angles/np.deg2rad(15)),np.ceil(lengths/(.2*radius)))).sum()+1)


def generate(source_path, template_path, output_root, count):
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    loader=REPO/'submodules/Inpaint360GS/scene/colmap_loader.py'
    spec=importlib.util.spec_from_file_location('fit_colmap_pose',loader)
    colmap=importlib.util.module_from_spec(spec);sys.modules[spec.name]=colmap;spec.loader.exec_module(colmap)
    posefile=Path(source_path)/'sparse/0/images.bin'
    extrinsics=colmap.read_extrinsics_binary(posefile)
    selected=sorted((c for c in extrinsics.values() if Path(c.name).name.lower().startswith('test')),key=lambda c:c.name)
    names=[c.name for c in selected]
    # COLMAP: x_cam=R_world_to_camera*x_world+t; stored GS R is its transpose.
    w2c=np.array([colmap.qvec2rotmat(c.qvec) for c in selected])
    centers=np.array([-r.T@np.array(c.tvec) for r,c in zip(w2c,selected)])
    c2w=w2c.transpose(0,2,1)
    order=continuous_order(centers,c2w)
    centers=centers[order];c2w=c2w[order];names=[names[i] for i in order]
    requested_count=count
    count=max(count,minimum_frame_count(centers,c2w))
    if count>max(180,requested_count):raise ValueError('More than 180 views needed for continuous anchor coverage; inspect trajectory')
    cs,rs,mapping,indices,radius=interpolate(centers,c2w,count)
    original=read(template_path);template=original['cameras'][0]
    views=[]
    for i,(c,r) in enumerate(zip(cs,rs)):
        view=copy.deepcopy(template);view.update(image_name=f'{i:05d}',R=r,T=-r.T@c,trans=np.zeros(3),scale=1.)
        views.append(SimpleNamespace(**view))
    output=root/'virtual_cameras.json'
    camera_contract().write_virtual_camera_manifest(output,views,iteration=original['iteration'],circle_radius=original['circle_radius'])
    # Legacy interchange schema1 stores actual poses, not a trajectory label.
    # The separate provenance explicitly declares test-conditioned interpolation.
    adjacent_angles=(Rotation.from_matrix(rs[:-1]).inv()*Rotation.from_matrix(rs[1:])).magnitude()*180/np.pi
    errors=np.linalg.norm(cs[indices]-centers,axis=1)
    if errors.max()>1e-8:raise ValueError('Anchor center was not retained exactly')
    provenance=dict(complete=True,algorithm='test_pose_linear_centers_so3_slerp_v1',
        test_poses_used=True,test_rgb_used=False,test_gt_masks_used=False,
        protocol='test-pose-conditioned synthesis; not held-out-pose generalization',
        order='Deterministic nearest pose + open-path 2-opt using normalized center distance and SO(3) angle; no closing edge',
        input_poses=record(posefile),template=record(template_path),producer=record(Path(__file__)),
        anchors=[dict(name=n,virtual_index=int(i),center=c.tolist()) for n,i,c in zip(names,indices,centers)],
        frame_count=count,previous_frame_count=requested_count,anchor_count=len(names),frames=mapping,output=record(output),
        max_anchor_center_error=float(errors.max()),max_adjacent_rotation_deg=float(adjacent_angles.max()),
        max_adjacent_translation_over_radius=float(np.linalg.norm(np.diff(cs,axis=0),axis=1).max()/radius),
        rendering_intrinsics='Unchanged per-scene old virtual FoV/width/height; test intrinsics not substituted',
        legacy_circle_radius_field='Retained solely for schema compatibility; not used to generate poses')
    write(root/'camera_source.json',provenance)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=np.array([-np.linalg.inv(np.array(c['R']).T)@np.array(c['T']) for c in original['cameras']])
    fig=plt.figure(figsize=(14,6));ax=fig.add_subplot(121,projection='3d')
    ax.plot(*old.T,color='gray',alpha=.6,label='Previous virtual path')
    ax.plot(*cs.T,color='teal',label='Interpolated test-pose path')
    ax.scatter(*centers.T,color='orange',s=12,label='test* camera anchors')
    ax.set(xlabel='World X',ylabel='World Y',zlabel='World Z',title=f'{Path(source_path).name}: {len(names)} anchors / {count} views');ax.legend()
    ax=fig.add_subplot(122);ax.plot(adjacent_angles);ax.axhline(15,linestyle='--',color='gray')
    ax.set(xlabel='Adjacent view index',ylabel='Rotation change (degrees)',title='Open-path continuity')
    fig.tight_layout();fig.savefig(root/'trajectory.png',dpi=150);plt.close(fig)
    print(f'{Path(source_path).name}: {len(names)} test anchors -> {count} virtual views; max rotation {adjacent_angles.max():.3f} deg',flush=True)
    return provenance


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--template',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--count',type=int,required=True)
    a=p.parse_args();generate(a.source,a.template,a.output_root,a.count)
