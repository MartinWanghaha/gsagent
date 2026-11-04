#!/usr/bin/env python3
"""CPU-only consistency diagnostic of synthetic training targets, never test GT.

This is not a geometric accuracy score. Depth disagreement can include real
occlusion and render errors; signed residuals and outside-mask controls are kept.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'submodules/EDGS'))
from source.paintmesh_local_data import make_views
from source.paintmesh_edgs_init import camera, project, unproject, sample
from run_geometry_ablation import read, write, now, sha


def run(job, scenes):
    request = read(job / 'request.json')
    result = dict(created_at=now(), interpretation='Synthetic target consistency, not true-surface accuracy; no test data read.',
                  policy=dict(source_stride=10, offsets=[-20, -5, 5, 20], samples=512,
                              relative_depth_tolerance=.02, ring_width_pixels=20, seed=0), scenes={})
    for scene in scenes:
        root = Path(next(s['root'] for s in request['scenes'] if s['scene'] == scene))
        init = read(root / 'manifests/edgs_init_manifest.json')
        lama = read(init['inputs']['lama']['path']); inputs = read(lama['input_manifest']['path'])
        cams = [camera(v) for v in make_views(init['inputs']['camera']['path'], device='cpu')]
        frames = sorted(lama['frames']); cache = {}
        def frame(i):
            if i not in cache:
                f = lama['frames'][frames[i]]['outputs']; src = inputs['frames'][frames[i]]
                cache[i] = dict(depth=np.load(f['depth']['path']),
                    rgb=np.array(Image.open(f['color']['path']).convert('RGB')) / 255.,
                    mask=np.array(Image.open(src['outputs']['color_mask']['path']).convert('L')) > 0)
            return cache[i]
        rng = np.random.default_rng(0); measurements = []; self_error = []
        for i in range(0, len(cams), 10):
            a = frame(i)
            ring = binary_dilation(a['mask'], iterations=20) & ~a['mask']
            for region, valid in [('hole', a['mask']), ('outside_ring', ring)]:
                y, x = np.where(valid & (a['depth'] > 0))
                rows = rng.choice(len(x), min(512, len(x)), replace=False)
                uv = np.column_stack((x[rows], y[rows])); xyz = unproject(uv, a['depth'][y[rows], x[rows]], cams[i])
                self_error.append(float(np.max(np.abs(project(xyz, cams[i])[0] - uv))))
                for offset in (-20, -5, 5, 20):
                    j = (i + offset) % len(cams); b = frame(j)
                    q, z, in_frame = project(xyz, cams[j]); d = sample(b['depth'], q)
                    dst_mask = sample(b['mask'].astype(float), q, 0) > 0
                    valid_pair = in_frame & (d > 0) & (dst_mask if region == 'hole' else ~dst_mask)
                    if not valid_pair.any(): continue
                    signed = (z - d) / np.maximum(d, 1e-8)
                    colors = np.stack([sample(b['rgb'][..., k], q) for k in range(3)], 1)
                    rgb = np.abs(colors - a['rgb'][y[rows], x[rows]]).mean(1)
                    good = valid_pair & (np.abs(signed) < .02)
                    measurements.append(dict(source=i, target=j, offset=offset, region=region,
                        count=int(valid_pair.sum()), depth_abs_relative=list(np.abs(signed[valid_pair]).astype(float)),
                        depth_signed_relative=list(signed[valid_pair].astype(float)),
                        rgb_mae_depth_consistent=list(rgb[good].astype(float))))
        summary = {}
        for region in ('hole', 'outside_ring'):
            selected = [v for v in measurements if v['region'] == region]
            absolute = np.array([x for m in selected for x in m['depth_abs_relative']])
            signed = np.array([x for m in selected for x in m['depth_signed_relative']])
            colors = np.array([x for m in selected for x in m['rgb_mae_depth_consistent']])
            summary[region] = dict(observations=len(absolute), median_depth_relative=float(np.median(absolute)),
                p90_depth_relative=float(np.percentile(absolute, 90)), fraction_depth_within_2pct=float(np.mean(absolute < .02)),
                fraction_in_front_over_2pct=float(np.mean(signed < -.02)),
                fraction_behind_over_2pct=float(np.mean(signed > .02)),
                depth_consistent_rgb_samples=len(colors), mean_rgb_mae_255=float(np.mean(colors) * 255))
        assert max(self_error) < 1e-6, 'Projection convention inconsistency'
        result['scenes'][scene] = dict(summary=summary, self_projection_max_pixels=max(self_error),
            inputs={k:init['inputs'][k] for k in ('camera', 'lama')},
            support=init['outputs']['support'])
        # Save the evidence separately so the short report is easy to inspect.
        write(job / 'pseudo_view_consistency' / (scene + '.json'), dict(measurements=measurements, **result['scenes'][scene]))
        write(job / 'pseudo_view_consistency.json', result)
        print(scene, summary, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True, type=Path)
    p.add_argument('--scenes', nargs='+', default=['garden_toys', 'fruits', 'redbull', 'toys'])
    args = p.parse_args(); run(args.job, args.scenes)
