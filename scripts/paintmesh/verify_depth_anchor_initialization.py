#!/usr/bin/env python3
"""Isolated bag pilot: RGB-only tracks -> depth correction -> held-out checks.

No production initializer, original output, old Gaussian, or training is changed.
Confidence means conditional consistency of synthetic observations, NOT accuracy
against the unknown object-free surface. Camera partitions and thresholds are
registered before processing; check images never fit the depth correction.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/paintmesh'), str(REPO / 'submodules/EDGS')]
import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree
from verify_mixed_initialization import (read, write, interpolate, warp_observations,
                                        point_information, sigma_max)

POLICY = dict(confidence=.7, cycle_pixels=3., reprojection_pixels=2.,
    min_angle_degrees=5., max_sigma_spacing=1., anchor_pixel_step=4,
    anchor_sigma_policy='finite conditional sigma, inverse-variance weighting; not final-confidence gating',
    collar_pixels=16, grid_pixels=32, smooth_weight=.3, ridge_weight=.001,
    irls_steps=8, huber_standardized=1.5, candidate_pixel_step=3,
    cross_depth_spacing=2., min_cross_depth_views=2, min_cross_depth_fraction=.6,
    min_rgb_edges=2, min_rgb_views=3, min_rgb_agreement=.75,
    max_known_front_views=1, seed=0)
ROOT = REPO / 'output/paintmesh/Inpaint360GS/inpaint360/bag/removal/table1_adaptive180_sweep/inpaint'


def project(xyz, cam):
    pc = xyz @ cam['w2c'][:3, :3].T + cam['w2c'][:3, 3]
    z = pc[:, 2]
    uv = (pc @ cam['k'].T)[:, :2] / np.maximum(z[:, None], 1e-12)
    valid = np.isfinite(uv).all(1) & (z > 0) & (uv >= 0).all(1)
    valid &= (uv <= [cam['w']-1, cam['h']-1]).all(1)
    return uv, z, valid


def unproject(uv, depth, cam):
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(cam['k']).T
    return (rays * depth[:, None]) @ cam['c2w'][:3, :3].T + cam['c2w'][:3, 3]


def triangulate_tracks(cameras, observations):
    """Batch multi-view DLT + pixel-only Gauss Newton; no depth input."""
    if not len(observations[0]):
        return np.empty((0, 3)), np.empty((0, len(cameras))), np.empty((0, 3, 3))
    rows = []
    for cam, uv in zip(cameras, observations):
        p = cam['k'] @ cam['w2c'][:3]
        rows.extend((uv[:, 0, None]*p[2]-p[0], uv[:, 1, None]*p[2]-p[1]))
    matrix = np.stack(rows, axis=1)
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    h = vh[:, -1]
    den = np.where(np.abs(h[:, 3]) > 1e-12, h[:, 3], np.nan)
    xyz = h[:, :3] / den[:, None]
    for _ in range(4):
        lhs = np.zeros((len(xyz), 3, 3)); rhs = np.zeros((len(xyz), 3))
        for cam, uv in zip(cameras, observations):
            r = cam['w2c'][:3, :3]
            pc = xyz @ r.T + cam['w2c'][:3, 3]; z = np.maximum(pc[:, 2], 1e-8)
            for axis in range(2):
                residual = cam['k'][axis, axis]*pc[:, axis]/z + cam['k'][axis, 2]-uv[:, axis]
                jac = cam['k'][axis, axis]*(r[axis]/z[:, None]-pc[:, axis, None]*r[2]/z[:, None]**2)
                weight = 1/np.maximum(1, np.abs(residual))
                lhs += np.einsum('ni,nj,n->nij', jac, jac, weight)
                rhs += jac*(residual*weight)[:, None]
        xyz -= np.linalg.solve(lhs + np.eye(3)[None]*1e-8, rhs)
    errors = np.stack([np.linalg.norm(project(xyz, c)[0]-uv, axis=1)
                       for c, uv in zip(cameras, observations)], axis=1)
    information = sum(point_information(xyz, c) for c in cameras)
    covariance = np.linalg.pinv(information, rcond=1e-12)
    return xyz, errors, covariance


def grid_basis(uv, origin, shape, step):
    """Bilinear correction grid; only correction, not scene depth, is smooth."""
    q = (np.asarray(uv)-origin)/step
    q = np.clip(q, 0, np.array(shape[::-1])-1-1e-9)
    xy = np.floor(q).astype(int); t = q-xy
    cols = np.stack([xy[:, 1]*shape[1]+xy[:, 0], xy[:, 1]*shape[1]+xy[:, 0]+1,
        (xy[:, 1]+1)*shape[1]+xy[:, 0], (xy[:, 1]+1)*shape[1]+xy[:, 0]+1], axis=1)
    vals = np.stack([(1-t[:, 0])*(1-t[:, 1]), t[:, 0]*(1-t[:, 1]),
                     (1-t[:, 0])*t[:, 1], t[:, 0]*t[:, 1]], axis=1)
    return sparse.csr_matrix((vals.ravel(), (np.repeat(np.arange(len(uv)), 4), cols.ravel())),
                             shape=(len(uv), int(np.prod(shape))))


def editable_anchor_rows(mask, uv):
    """Only constrain depth that is actually editable at all bilinear corners."""
    uv=np.asarray(uv);xy=np.floor(uv).astype(int);h,w=mask.shape
    valid=np.isfinite(uv).all(1)&(xy>=0).all(1)&(xy[:,0]<w-1)&(xy[:,1]<h-1)
    safe=np.where(valid[:,None],xy,0)
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        valid &= mask[safe[:,1]+dy,safe[:,0]+dx]
    return valid


def fit_correction(uv, residual, uncertainty, origin, shape, step,
                   boundary_uv, smooth_weight=.3, ridge_weight=.001, edge_valid=None):
    """IRLS in world-spacing-normalized depth units, with a zero correction collar."""
    a = grid_basis(uv, origin, shape, step)
    boundary = grid_basis(boundary_uv, origin, shape, step)
    ids = np.arange(np.prod(shape)).reshape(shape)
    left = np.r_[ids[:, :-1].ravel(), ids[:-1].ravel()]
    right = np.r_[ids[:, 1:].ravel(), ids[1:].ravel()]
    if edge_valid is not None:
        left, right = left[edge_valid], right[edge_valid]
    reg = sparse.csr_matrix((np.tile([1., -1.], len(left)),
        (np.repeat(np.arange(len(left)), 2), np.column_stack((left, right)).ravel())),
        shape=(len(left), len(ids.ravel())))
    fixed = sparse.vstack((boundary*4, reg*np.sqrt(smooth_weight),
                           sparse.eye(len(ids.ravel()))*np.sqrt(ridge_weight)), format='csr')
    sigma = np.maximum(uncertainty, .25)
    solution = np.zeros(a.shape[1]); weights = np.ones(len(uv))
    for _ in range(POLICY['irls_steps']):
        standard = (a @ solution-residual)/sigma
        weights = np.minimum(1, POLICY['huber_standardized']/np.maximum(np.abs(standard), 1e-12))
        w = np.sqrt(weights)/sigma
        system = sparse.vstack((a.multiply(w[:, None]), fixed), format='csr')
        target = np.r_[residual*w, np.zeros(fixed.shape[0])]
        solution = lsqr(system, target, atol=1e-9, btol=1e-9, iter_lim=1000)[0]
    return solution, weights


def accepted_rgb(evidence):
    return ((evidence['edges'] >= POLICY['min_rgb_edges'])
        & (evidence['views'].sum(1) >= POLICY['min_rgb_views'])
        & (evidence['sigma_spacing'] <= POLICY['max_sigma_spacing'])
        & (evidence['edges']/np.maximum(evidence['confident'], 1) >= POLICY['min_rgb_agreement']))


def ply_points(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    data = np.zeros(len(xyz), dtype=[(k, 'f4') for k in 'xyz']+
                    [(k, 'u1') for k in ('red', 'green', 'blue')])
    for i, k in enumerate('xyz'): data[k] = xyz[:, i]
    for i, k in enumerate(('red', 'green', 'blue')): data[k] = np.rint(rgb[:, i]*255).clip(0, 255).astype('u1')
    PlyData([PlyElement.describe(data, 'vertex')]).write(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, default=ROOT/'mixed_confidence_pilot_20261022_175604')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--reuse-closure-root', type=Path)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import torch
    from PIL import Image
    from source.paintmesh_edgs_init import camera, RomaMatcher
    from source.paintmesh_local_data import make_views
    from edgs_inpaint_io import record, verify_record
    started = time.time(); out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'request.json').exists(): raise ValueError('Use a fresh output directory')
    old = read(args.cache_root/'request.json'); initial = read(old['baseline']['path'])
    for k in ('camera', 'lama'): verify_record(initial['inputs'][k])
    verify_record(initial['outputs']['support'])
    lama = read(initial['inputs']['lama']['path']); inputs = read(lama['input_manifest']['path'])
    cameras = [camera(v) for v in make_views(initial['inputs']['camera']['path'], device='cpu')]
    support = dict(np.load(initial['outputs']['support']['path'])); gap = float(support['spacing'])
    pairs = old['pairs']; fit_frames = sorted({p[k] for p in pairs if p['role']=='fit' for k in ('a','b')})
    check_frames = sorted({p[k] for p in pairs if p['role']=='check' for k in ('a','b')})
    assert not set(fit_frames)&set(check_frames)
    groups = {}
    for p in pairs:
        if p['role']=='fit': groups.setdefault(p['a'], []).append(p['b'])
    (out/'script_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write(out/'request.json', dict(created_at=datetime.now().astimezone().isoformat(), policy=POLICY,
        script=record(__file__), reused_helpers=record(REPO/'scripts/paintmesh/verify_mixed_initialization.py'),
        initial=record(old['baseline']['path']), lama=initial['inputs']['lama'], camera=initial['inputs']['camera'],
        support=initial['outputs']['support'], cache_request=record(args.cache_root/'request.json'),
        cached_matches=[record(args.cache_root/'match_cache'/f"{p['role']}_{p['a']:05d}_{p['b']:05d}.npz") for p in pairs],
        fit_frames=fit_frames, check_frames=check_frames, fit_groups=groups, gpu=args.gpu,
        reused_closures=([record(p) for p in sorted((args.reuse_closure_root/'match_cache').glob('fit_closure_*.npz'))]
                         if args.reuse_closure_root else []),
        scope='Initialization-only pilot; correction evaluated on held-out synthetic views; no GT/test photos.',
        confidence='Conditional evidence, not calibrated true-surface probability. Fixed thresholds; no check-driven tuning.'))
    write(out/'status.json', dict(state='running', phase='independent RGB anchors'))
    data = {}
    def frame(i):
        if i not in data:
            s = f'{i:05d}'; f = inputs['frames'][s]
            data[i] = dict(depth=np.load(lama['frames'][s]['outputs']['depth']['path']),
                mask=np.asarray(Image.open(f['outputs']['color_mask']['path']).convert('L'))>0,
                original=np.asarray(Image.open(f['inputs']['mask']['path']).convert('L'))>0,
                rgb=np.asarray(Image.open(lama['frames'][s]['outputs']['color']['path']).convert('RGB'))/255.,
                observed=np.load(f['inputs']['removed_depth']['path']), alpha=np.load(f['inputs']['removed_alpha']['path']))
        return data[i]
    def load_pair(p):
        return dict(np.load(args.cache_root/'match_cache'/f"{p['role']}_{p['a']:05d}_{p['b']:05d}.npz"))
    def observations(cache, uv, a, b):
        return warp_observations(cache['forward'], cache['forward_confidence'], cache['reverse'],
                                 cache['reverse_confidence'], uv, frame(a)['mask'].shape, frame(b)['mask'].shape)
    matcher = None; tracks = []; track_stats = []
    (out/'match_cache').mkdir()
    for a, bs in groups.items():
        b, c = bs; fa = frame(a); region = binary_dilation(fa['mask'], iterations=POLICY['collar_pixels'])
        yy, xx = np.where(region); use = (xx % POLICY['anchor_pixel_step']==0)&(yy % POLICY['anchor_pixel_step']==0)
        uv = np.column_stack((xx[use], yy[use])).astype(float)
        caches = [load_pair(dict(role='fit', a=a, b=k)) for k in (b,c)]
        matches = [observations(cache, uv, a, k) for cache, k in zip(caches, (b,c))]
        valid = np.ones(len(uv), bool)
        for _, conf, cycle, ok in matches:
            valid &= ok&(conf>=POLICY['confidence'])&(cycle<=POLICY['cycle_pixels'])
        obs = [uv[valid], matches[0][0][valid], matches[1][0][valid]]
        # Independent third edge closes the triangle; all these cameras are fit-only.
        closure_path=(args.reuse_closure_root/'match_cache'/f'fit_closure_{b:05d}_{c:05d}.npz'
                      if args.reuse_closure_root else None)
        if closure_path and closure_path.is_file():
            previous=read(args.reuse_closure_root/'request.json')
            if previous['lama']!=initial['inputs']['lama'] or previous['camera']!=initial['inputs']['camera']:
                raise ValueError('closure cache input identity mismatch')
            closure=dict(np.load(closure_path))
        else:
            if matcher is None:
                for rec in initial['request']['matcher_weights'].values(): verify_record(rec)
                matcher = RomaMatcher(initial['request']['config']['matcher'], initial['request']['matcher_weights'])
            images = [Image.fromarray(np.rint(frame(k)['rgb']*255).astype('u1')) for k in (b,c)]
            with torch.no_grad():
                f, fc = matcher.model.match(*images, device='cuda')
                r, rc = matcher.model.match(*images[::-1], device='cuda')
            closure = dict(zip(('forward','forward_confidence','reverse','reverse_confidence'),
                              [x.cpu().numpy() for x in (f,fc,r,rc)]))
        np.savez_compressed(out/'match_cache'/f'fit_closure_{b:05d}_{c:05d}.npz', **closure)
        matched, conf, cycle, ok = observations(closure, obs[1], b, c)
        closed = ok&(conf>=POLICY['confidence'])&(cycle<=POLICY['cycle_pixels'])
        closed &= np.linalg.norm(matched-obs[2], axis=1)<=POLICY['reprojection_pixels']
        obs = [x[closed] for x in obs]; cams = [cameras[k] for k in (a,b,c)]
        xyz, errors, cov = triangulate_tracks(cams, obs)
        good = np.isfinite(xyz).all(1)&(errors.max(1)<=POLICY['reprojection_pixels'])
        for cam in cams: good &= project(xyz, cam)[2]
        sigma = sigma_max(sum(point_information(xyz, cam) for cam in cams))/gap
        reprojection_passed=int(good.sum())
        # A noisy but valid RGB track is a weak anchor, not a rejected candidate.
        # Final high-confidence output still requires <= one spacing on both splits.
        good &= np.isfinite(sigma)
        direction = [xyz-cam['c2w'][:3,3] for cam in cams]
        angles = [np.degrees(np.arccos(np.clip(np.einsum('ni,ni->n',direction[i],direction[j]) /
            (np.linalg.norm(direction[i],axis=1)*np.linalg.norm(direction[j],axis=1)), -1, 1)))
            for i,j in ((0,1),(0,2),(1,2))]
        good &= np.max(angles, axis=0)>=POLICY['min_angle_degrees']
        track = dict(xyz=xyz[good], covariance=cov[good], sigma_spacing=sigma[good],
                     frames=np.tile([a,b,c], (int(good.sum()),1)), uv=np.stack(obs,axis=1)[good])
        tracks.append(track)
        interior = interpolate(fa['original'].astype(float), track['uv'][:,0], 0)>0
        stat=dict(a=a,b=b,c=c,candidates=len(uv),two_edges=int(valid.sum()),
            closed=int(closed.sum()),reprojection_passed=reprojection_passed,
            anchors=int(good.sum()),source_original_hole=int(interior.sum()),
            median_sigma_spacing=float(np.median(sigma[good])) if good.any() else None,
            anchors_sigma_le_one=int((good&(sigma<=POLICY['max_sigma_spacing'])).sum()))
        track_stats.append(stat); write(out/'track_statistics.json', track_stats)
        print('ANCHORS',stat,flush=True)
    del matcher; torch.cuda.empty_cache()
    anchors = {k:np.concatenate([t[k] for t in tracks]) for k in tracks[0]}
    np.savez_compressed(out/'rgb_anchors.npz', **anchors)
    write(out/'status.json',dict(state='running',phase='depth correction',anchors=len(anchors['xyz'])))
    correction = {}; field_stats = []; (out/'corrected_depth').mkdir(); fields = {}
    for i in fit_frames:
        f = frame(i); mask = f['mask']; yy,xx = np.where(binary_dilation(mask,iterations=POLICY['collar_pixels']))
        origin = np.array([xx.min(),yy.min()],float); step = POLICY['grid_pixels']
        shape = (int(np.ceil((yy.max()-origin[1])/step))+1, int(np.ceil((xx.max()-origin[0])/step))+1)
        rows,slots = np.where(anchors['frames']==i); uv = anchors['uv'][rows,slots]
        _,z,_ = project(anchors['xyz'][rows],cameras[i]); sampled = interpolate(f['depth'],uv)
        scale = np.linalg.norm(cameras[i]['w2c'][2,:3]); unit = gap*scale
        available_anchors=len(uv)
        valid = (sampled>0)&(z>0)&editable_anchor_rows(mask,uv)
        rows,uv,z,sampled = rows[valid],uv[valid],z[valid],sampled[valid]
        vector = cameras[i]['w2c'][2,:3]
        sigma = np.sqrt(np.einsum('i,nij,j->n',vector,anchors['covariance'][rows],vector))/unit
        residual = (z-sampled)/unit
        # Boundary observations are from unedited retained geometry, not GT.
        border = binary_dilation(mask,iterations=POLICY['collar_pixels'])&~mask&(f['alpha']>.95)&(f['observed']>0)
        by,bx = np.where(border); take = (bx%8==0)&(by%8==0)
        boundary_uv = np.column_stack((bx[take],by[take])).astype(float)
        gy,gx = np.mgrid[:shape[0],:shape[1]]; nodes = np.column_stack((gx.ravel(),gy.ravel()))*step+origin
        nd = interpolate(f['depth'],nodes).reshape(shape)
        edge_valid = np.r_[(np.abs(nd[:,1:]-nd[:,:-1])<=.05*np.maximum(nd[:,1:],nd[:,:-1])).ravel(),
                          (np.abs(nd[1:]-nd[:-1])<=.05*np.maximum(nd[1:],nd[:-1])).ravel()]
        coeff,weights = fit_correction(uv,residual,sigma,origin,shape,step,boundary_uv,
            POLICY['smooth_weight'],POLICY['ridge_weight'],edge_valid)
        hy,hx = np.where(mask&(f['depth']>0)); huv = np.column_stack((hx,hy)).astype(float)
        delta = (grid_basis(huv,origin,shape,step)@coeff)*unit
        corrected = f['depth'].copy(); corrected[hy,hx] += delta
        corrected[(corrected<=0)|~np.isfinite(corrected)] = 0
        correction[i] = corrected
        np.save(out/'corrected_depth'/f'{i:05d}.npy',corrected)
        np.savez_compressed(out/'corrected_depth'/f'{i:05d}_field.npz',origin=origin,shape=shape,coefficients=coeff,
                            anchor_uv=uv,anchor_residual_spacing=residual,anchor_sigma_spacing=sigma,robust_weights=weights)
        after = (interpolate(corrected,uv)-z)/unit
        field_stats.append(dict(frame=i,anchors=len(uv),available_rgb_anchors=available_anchors,
            boundary_samples=len(boundary_uv),
            median_abs_anchor_before_spacing=float(np.median(abs(residual))) if len(uv) else None,
            median_abs_anchor_after_spacing=float(np.median(abs(after))) if len(uv) else None,
            correction_p95_spacing=float(np.quantile(abs(delta/unit),.95)),
            changed_pixels=int((np.abs(delta/unit)>1e-5).sum()),
            invalid_after=int((corrected[mask]<=0).sum())))
        fields[i]=(origin,shape,coeff)
    write(out/'field_statistics.json',field_stats)
    batches=[]
    for i in groups:
        f=frame(i); y,x=np.where(f['mask']&(f['depth']>0)&(correction[i]>0))
        keep=(x%POLICY['candidate_pixel_step']==0)&(y%POLICY['candidate_pixel_step']==0); y,x=y[keep],x[keep]
        uv=np.column_stack((x,y)).astype(float)
        batches.append(dict(before=unproject(uv,f['depth'][y,x],cameras[i]),
            after=unproject(uv,correction[i][y,x],cameras[i]),rgb=f['rgb'][y,x],uv=uv,
            frame=np.full(len(uv),i),source_original=f['original'][y,x]))
    candidates={k:np.concatenate([b[k] for b in batches]) for k in batches[0]}
    np.savez_compressed(out/'corrected_candidates.npz',**candidates)
    n=len(candidates['after']); xyz=np.concatenate((candidates['before'],candidates['after'],anchors['xyz']))
    print(f'CORRECTION complete: {len(anchors["xyz"])} anchors; {n} candidate pixels',flush=True)
    write(out/'status.json',dict(state='running',phase='held-out evaluation',candidates=n))
    evidence={role:dict(edges=np.zeros(len(xyz),int),confident=np.zeros(len(xyz),int),
        views=np.zeros((len(xyz),len(cameras)),bool),information=np.zeros((len(xyz),3,3)),
        error_sum=np.zeros(len(xyz)),errors=[]) for role in ('fit','check')}
    for p in pairs:
        a,b,role=p['a'],p['b'],p['role']; auv,_,av=project(xyz,cameras[a]); buv,_,bv=project(xyz,cameras[b])
        safe=np.where(av[:,None],auv,0.); matched,conf,cycle,ok=observations(load_pair(p),safe,a,b)
        error=np.linalg.norm(matched-buv,axis=1)
        eligible=av&bv&ok&(conf>=POLICY['confidence'])&(cycle<=POLICY['cycle_pixels'])
        # No raw completed-depth agreement gate: depth is what is being corrected.
        directions=[xyz-cameras[k]['c2w'][:3,3] for k in (a,b)]
        angle=np.degrees(np.arccos(np.clip(np.einsum('ni,ni->n',*directions)/
            (np.linalg.norm(directions[0],axis=1)*np.linalg.norm(directions[1],axis=1)),-1,1)))
        eligible &= angle>=POLICY['min_angle_degrees']
        accepted=eligible&(error<=POLICY['reprojection_pixels']); e=evidence[role]
        e['confident']+=eligible; e['edges']+=accepted; e['error_sum']+=np.where(accepted,error,0.)
        e['errors'].append(np.where(eligible,error,np.nan))
        for k in (a,b):
            fresh=accepted&~e['views'][:,k]; e['information'][fresh]+=point_information(xyz[fresh],cameras[k])
            e['views'][accepted,k]=True
    for role,e in evidence.items():
        e['sigma_spacing']=sigma_max(e.pop('information'))/gap; e['errors']=np.stack(e['errors'],axis=1)
        np.savez_compressed(out/f'{role}_evidence.npz',**e)
    fit_rgb=accepted_rgb(evidence['fit']); check_rgb=accepted_rgb(evidence['check'])
    # Fit-only corrected-depth agreement; all known-background free-space checks
    # are diagnostic and use rendered retained geometry, not ground truth.
    depth_count=np.zeros(2*n,int); depth_agree=np.zeros(2*n,int); known_front=np.zeros(2*n,int)
    sources=np.tile(candidates['frame'],2)
    for i in range(len(cameras)):
        f=frame(i); uv,z,inside=project(xyz[:2*n],cameras[i]); safe=np.where(inside[:,None],uv,0.)
        tol=POLICY['cross_depth_spacing']*gap*np.linalg.norm(cameras[i]['w2c'][2,:3])
        hole=interpolate(f['mask'].astype(float),safe,0)>0
        d=interpolate(f['observed'],safe)
        trusted=inside&~hole&(d>0)&(interpolate(f['alpha'],safe)>.95)
        known_front+=trusted&(z<d-tol)
        if i in correction:
            # Compare each version to its own fit-view depth surfaces.
            d=np.r_[interpolate(f['depth'],safe[:n]),interpolate(correction[i],safe[n:])]
            available=inside&hole&(d>0)&(sources!=i)
            depth_count+=available; depth_agree+=available&(np.abs(z-d)<=tol)
        # Keep memory bounded: only images needed for plots/fit remain cached.
        if i not in fit_frames and i not in groups: data.pop(i,None)
    geometry=(depth_agree>=POLICY['min_cross_depth_views'])&(depth_agree/np.maximum(depth_count,1)>=POLICY['min_cross_depth_fraction'])
    valid_bg=known_front<=POLICY['max_known_front_views']
    selected_before=fit_rgb[:n]&check_rgb[:n]&geometry[:n]&valid_bg[:n]
    selected=fit_rgb[n:2*n]&check_rgb[n:2*n]&geometry[n:]&valid_bg[n:]
    # Merge only after consistency selection. Unresolved layers are not averaged.
    def dedup(rows):
        cells=np.floor(candidates['after'][rows]/gap).astype(np.int64)
        _,first=np.unique(cells,axis=0,return_index=True);return rows[np.sort(first)]
    selected_rows=dedup(np.flatnonzero(selected))
    raw_rows=dedup(np.arange(n))
    np.savez_compressed(out/'selection.npz',selected=selected,selected_rows=selected_rows,
        selected_before=selected_before,fit_rgb=fit_rgb,check_rgb=check_rgb,
        depth_count=depth_count,depth_agree=depth_agree,known_front=known_front)
    for name,points,colors in [('depth_before.ply',candidates['before'],candidates['rgb']),
        ('corrected_candidates.ply',candidates['after'][raw_rows],candidates['rgb'][raw_rows]),
        ('high_confidence_points.ply',candidates['after'][selected_rows],candidates['rgb'][selected_rows])]:
        ply_points(out/name,points,colors)
    anchor_colors=[]
    for j in range(len(anchors['xyz'])):
        i=int(anchors['frames'][j,0]);uv=anchors['uv'][j,0:1]
        anchor_colors.append([interpolate(frame(i)['rgb'][...,c],uv)[0] for c in range(3)])
    anchor_colors=np.asarray(anchor_colors).reshape(-1,3)
    ply_points(out/'rgb_anchors.ply',anchors['xyz'],anchor_colors)
    ply_points(out/'rgb_anchors_check_confirmed.ply',anchors['xyz'][check_rgb[2*n:]],anchor_colors[check_rgb[2*n:]])
    # Standard 3DGS PLY for the new subset only, same shape/color rules as baseline.
    from plyfile import PlyData,PlyElement
    template=PlyData.read(initial['outputs']['ply']['path'],mmap=True)['vertex'].data
    gs=np.zeros(len(selected_rows),dtype=template.dtype)
    for axis,key in enumerate('xyz'):gs[key]=candidates['after'][selected_rows,axis]
    for axis in range(3):
        gs[f'f_dc_{axis}']=(candidates['rgb'][selected_rows,axis]-.5)/.28209479177387814
        gs[f'scale_{axis}']=np.log(gap)
    gs['rot_0']=1.;opacity=initial['request']['config']['init']['opacity'];gs['opacity']=np.log(opacity/(1-opacity))
    PlyData([PlyElement.describe(gs,'vertex')]).write(str(out/'high_confidence_gaussians.ply'))
    coverage=[];votes=np.zeros(len(selected_rows),int);anchor_votes=np.zeros(len(anchors['xyz']),int)
    for i in range(len(cameras)):
        f=frame(i);uv,z,valid=project(candidates['after'][selected_rows],cameras[i]);safe=np.where(valid[:,None],uv,0.)
        votes+=valid&(interpolate(f['original'].astype(float),safe,0)>0)
        auv,_,av=project(anchors['xyz'],cameras[i]);anchor_votes+=av&(interpolate(f['original'].astype(float),np.where(av[:,None],auv,0.),0)>0)
        if i in groups:
            raster=np.zeros_like(f['mask']);pixels=np.rint(uv[valid]).astype(int)
            if len(pixels):raster[pixels[:,1],pixels[:,0]]=True
            dist=distance_transform_edt(~raster) if raster.any() else np.full(raster.shape,np.inf)
            radius=float(np.median(gap*np.linalg.norm(cameras[i]['w2c'][2,:3])*cameras[i]['k'][0,0]/z[valid])) if valid.any() else 0.
            coverage.append(dict(frame=i,original_mask_coverage=float((dist[f['original']]<=radius).mean()),
                dilated_mask_coverage=float((dist[f['mask']]<=radius).mean()),approximate_footprint_pixels=radius))
        if i not in fit_frames and i not in groups:data.pop(i,None)
    np.save(out/'anchor_original_mask_votes.npy',anchor_votes)
    e=evidence['check'];before=e['errors'][:n];after=e['errors'][n:2*n]
    common=np.isfinite(before)&np.isfinite(after)
    def population(sl):
        return dict(fit_rgb_pass=int(fit_rgb[sl].sum()),check_rgb_pass=int(check_rgb[sl].sum()),
            both_rgb_pass=int((fit_rgb[sl]&check_rgb[sl]).sum()))
    def paired(subset):
        ok=common&subset[:,None]
        return dict(observations=int(ok.sum()),before_median=float(np.median(before[ok])) if ok.any() else None,
            after_median=float(np.median(after[ok])) if ok.any() else None,
            fraction_improved=float((after[ok]<before[ok]).mean()) if ok.any() else None)
    summary=dict(complete=True,scope='Sparse RGB anchors actually corrected dense depth; initialization only.',
        camera_fit_count=len(fit_frames),camera_check_count=len(check_frames),depth_corrected_views=len(correction),
        depth_views_with_nonzero_correction=sum(f['changed_pixels']>0 for f in field_stats),
        anchors=len(anchors['xyz']),anchors_inside_original_mask_any_view=int((anchor_votes>0).sum()),
        editable_anchor_observations=sum(f['anchors'] for f in field_stats),
        anchors_check_confirmed=int(check_rgb[2*n:].sum()),
        anchors_check_confirmed_inside_original=int((check_rgb[2*n:]&(anchor_votes>0)).sum()),
        candidate_pixels=n,corrected_candidate_voxels=len(raw_rows),
        before=population(slice(0,n)),after=population(slice(n,2*n)),
        before_high_confidence_pixels=int(selected_before.sum()),after_high_confidence_pixels=int(selected.sum()),
        high_confidence_voxels=len(selected_rows),high_confidence_inside_original_any_view=int((votes>0).sum()),
        high_confidence_source_original_pixels=int((selected&candidates['source_original']).sum()),
        paired_held_out_pixels_all=paired(np.ones(n,bool)),paired_held_out_source_original=paired(candidates['source_original']),
        mask_coverage=coverage,policy=POLICY,elapsed_seconds=time.time()-started,
        ground_truth_used=False,background_modified=False,confidence_is_calibrated_probability=False,
        limitations=['All RGB/depth observations are synthetic; agreement does not prove unknown true geometry.',
            'Only six reference depth views are sampled; sparse tracks connect fit-view corrections, not a full dense global solver.',
            'Held-out match eligibility can change with location; paired statistics use the same eligible observation pairs.',
            'Final supported subset is selected using check results; its pass rate is not another independent test.',
            'Known-depth checks are against retained rendered geometry; original-mask projections are not visibility-proven coverage.',
            'No surface GT, Gaussian training, or final image-quality improvement is claimed.'])
    write(out/'summary.json',summary)
    make_plots(out,candidates,selected_rows,anchors,anchor_colors,groups,cameras,frame,correction,gap)
    write_report(out,summary,field_stats)
    write(out/'status.json',dict(state='complete',summary=str(out/'summary.json')))
    print('COMPLETE',out,flush=True);print(summary,flush=True)


def make_plots(out,candidates,selected,anchors,anchor_colors,groups,cameras,frame,correction,gap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pos=np.array([c['c2w'][:3,3] for c in cameras]);_,_,basis=np.linalg.svd(pos-pos.mean(0),full_matrices=False)
    center=np.median(candidates['before'],axis=0)
    before=(candidates['before']-center)@basis.T;after=(candidates['after']-center)@basis.T
    lo,hi=np.quantile(np.concatenate((before,after)),[.005,.995],axis=0)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for col,(title,coords,rows) in enumerate((('Raw depth',before,np.arange(len(before))),
        ('Anchor-corrected depth',after,np.arange(len(after))),('Passed fit + held-out checks',after,selected))):
        subset=rows[::max(1,len(rows)//20000)]
        for ax,(a,b) in zip(axes[:,col],((0,1),(0,2))):
            ax.scatter(coords[subset,a],coords[subset,b],c=candidates['rgb'][subset],s=1,alpha=.5)
            ax.set(title=f'{title}: {len(rows):,}',xlabel=f'Camera-PCA {a}',ylabel=f'Camera-PCA {b}',
                xlim=(lo[a],hi[a]),ylim=(lo[b],hi[b]));ax.set_aspect('equal',adjustable='box')
    fig.savefig(out/'geometry_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,i in zip(axes.flat,groups):
        f=frame(i);uv,_,valid=project(candidates['after'][selected],cameras[i]);au,_,av=project(anchors['xyz'],cameras[i])
        ax.imshow(f['rgb']);ax.contour(f['original'],levels=[.5],colors=['red'],linewidths=.8)
        ax.scatter(au[av,0],au[av,1],s=5,c='#ffbb00',label='RGB anchors')
        ax.scatter(uv[valid,0],uv[valid,1],s=8,c='#00ff88',label='Confirmed points')
        y,x=np.where(f['mask']);ax.set(xlim=(x.min()-15,x.max()+15),ylim=(y.max()+15,y.min()-15),title=f'Frame {i:05d}')
        ax.axis('off')
    axes.flat[0].legend(loc='upper left',fontsize=8)
    fig.savefig(out/'mask_projection.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,i in zip(axes.flat,groups):
        f=frame(i);unit=gap*np.linalg.norm(cameras[i]['w2c'][2,:3]);d=(correction[i]-f['depth'])/unit
        y,x=np.where(f['mask']);crop=np.s_[y.min():y.max()+1,x.min():x.max()+1]
        im=ax.imshow(np.where(f['mask'],d,np.nan)[crop],cmap='coolwarm',vmin=-4,vmax=4)
        ax.set_title(f'{i:05d}: correction / world spacing');ax.axis('off')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7)
    fig.savefig(out/'depth_correction.png',dpi=160);plt.close(fig)


def write_report(out,s,fields):
    passed=s['high_confidence_inside_original_any_view']
    verdict=('存在通过本轮多视图检查的洞内候选；仍需检查覆盖与真实几何。' if passed else
             '本轮未得到通过全部检查、且投影进入原始修复区的点云；不能宣称高置信度洞内初始化成功。')
    lines=['# bag：RGB 锚点引导深度校正的初步验证','',verdict,'',
        '本次实际重新三角化 RGB 三视图轨迹、求解连续深度修正场、生成新点云。没有使用平面拟合，没有训练或覆盖旧结果。',
        'fit 与 check 相机不重叠；check 不参与锚点生成、修正场求解或参数选择。阈值在 request.json 中预先记录。',
        '所有视图来自同一场景的合成补全，置信度仅表示条件几何一致性，不是与未知真实表面比较得到的准确率。','',
        '| 项目 | 数量 |','|---|---:|',f"| RGB-only 三视图锚点 | {s['anchors']} |",
        f"| 锚点在留出视图中通过 RGB 检查 | {s['anchors_check_confirmed']} |",
        f"| 上述通过锚点进入原始 mask | {s['anchors_check_confirmed_inside_original']} |",
        f"| 有效作用于补全 mask 的锚点观测 | {s['editable_anchor_observations']} |",
        f"| 生成深度图 / 有非零修正 | {s['depth_corrected_views']} / {s['depth_views_with_nonzero_correction']} |",
        f"| 校正前后相同采样像素 | {s['candidate_pixels']} |",
        f"| 校正前 fit/check RGB 均通过 | {s['before']['both_rgb_pass']} |",
        f"| 校正后 fit/check RGB 均通过 | {s['after']['both_rgb_pass']} |",
        f"| 校正前全部门槛通过像素 | {s['before_high_confidence_pixels']} |",
        f"| 校正后全部门槛通过像素 | {s['after_high_confidence_pixels']} |",
        f"| 校正后通过点体素去重 | {s['high_confidence_voxels']} |",
        f"| 通过点进入任一原始 mask | {passed} |",'',
        '## 留出观测的前后对照','',
        '| 区域 | 同一观测对数量 | 原始重投影误差中位数 px | 校正后 px | 误差降低比例 |',
        '|---|---:|---:|---:|---:|']
    for label,key in [('全部采样','paired_held_out_pixels_all'),('源视图原始洞内','paired_held_out_source_original')]:
        q=s[key];lines.append(f"| {label} | {q['observations']} | {q['before_median']} | {q['after_median']} | {q['fraction_improved']} |")
    lines+=['','误差只比较校正前后都存在高置信匹配的同一观测对，不能将丢掉困难点后的误差下降当成提升。',
        '最终通过点使用 check 做了筛选，因此它的通过率不能再称为独立测试结果。','',
        '## 几何与覆盖','', '![前后点云](geometry_comparison.png)','',
        '![原始 mask 与点投影：黄色锚点，绿色通过点](mask_projection.png)','',
        '![深度修正量](depth_correction.png)','',
        '| 帧 | 原始 mask 投影覆盖 | 膨胀 mask 投影覆盖 |','|---|---:|---:|']
    for c in s['mask_coverage']:lines.append(f"| {c['frame']:05d} | {c['original_mask_coverage']:.3%} | {c['dilated_mask_coverage']:.3%} |")
    lines+=['','覆盖使用一个世界点间距的近似投影足迹，仅代表投影覆盖，不是真实可见表面覆盖。','',
        '## 文件','',
        '- corrected_candidates.ply：校正并体素去重的全部候选，**未全部通过验证**。',
        '- high_confidence_points.ply / high_confidence_gaussians.ply：通过全部本轮检查的子集；可能为空。',
        '- rgb_anchors.ply / rgb_anchors_check_confirmed.ply：拟合锚点与通过留出检查的锚点。',
        '- corrected_depth/：实际校正深度与控制网格、锚点残差、鲁棒权重。',
        '- corrected_candidates.npz、fit_evidence.npz、check_evidence.npz、selection.npz：逐点可复查证据。',
        '- track_statistics.json、field_statistics.json、summary.json：匹配、优化与验证统计。','',
        '本轮采用稀疏三维轨迹共同约束各 fit 深度图，然后进行投影一致性检查与筛选；未实现完整稠密联合曲面优化。',
        '没有读取测试 RGB、无物体 GT 或用 GT 选择参数。没有修改生产训练/初始化代码。','']
    (out/'report.zh-CN.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
