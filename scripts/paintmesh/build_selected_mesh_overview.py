#!/usr/bin/env python3
"""Publish alternative qualitative views without changing mesh assets or scores.

Each scene uses a different existing held-out camera than the original overview.
Selection explicitly favors PaintMesh hole coverage and a local normal-stability
proxy. All methods use the same camera, crop, depth scale and original renders.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt, label

import plot_mesh_method_comparison as plots
from mesh_benchmark.core import SCENES, METHODS, read, write, digest, identity
from mesh_benchmark.layout import work_root, configure_sources
from organize_mesh_subplots import export_method, indexes, FILES


RULE = (
    'Qualitative selection favoring PaintMesh: exclude original overview camera; '
    'retain target area >=60% of original maximum when possible; otherwise use '
    'targets >=60% of the largest alternate target. Minimize worst target hole fraction '
    'across background/composed; within 0.002 of best, minimize severe local '
    'normal changes then largest hole. All methods share the selected camera. '
    'No RGB score or baseline quality is used for selection.'
)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def roi_for(camera, mask):
    yy, xx = np.where(mask)
    x0, y0, x1, y1 = xx.min(), yy.min(), xx.max() + 1, yy.max() + 1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = max((y1 - y0) * 1.6, (x1 - x0) * .8, 50.)
    w = 2 * h
    if w > camera['width']:
        w = camera['width']; h = w / 2
    if h > camera['height']:
        h = camera['height']; w = h * 2
    x0 = max(0, min(camera['width'] - w, cx - w / 2))
    y0 = max(0, min(camera['height'] - h, cy - h / 2))
    return [int(round(v)) for v in [x0, y0, x0 + w, y0 + h]]


def normal_instability(depth, camera, region):
    """Severe normal changes, excluding invalid samples and sharp depth edges.

    This is a presentation-selection proxy, never a geometry-accuracy metric.
    """
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    q = np.stack([xx + .5, yy + .5, np.ones_like(xx)], -1)
    points = (q @ np.linalg.inv(camera['K']).T) * depth[..., None]
    n = np.cross(np.gradient(points, axis=1), np.gradient(points, axis=0))
    lengths = np.linalg.norm(n, axis=-1)
    n /= np.maximum(lengths[..., None], 1e-12)
    valid = binary_erosion(region & np.isfinite(depth) & (depth > 0), iterations=2)
    valid &= lengths > 1e-12
    bad = total = 0
    for axis in [0, 1]:
        a = [slice(None), slice(None)]; b = a.copy()
        a[axis] = slice(None, -1); b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        pair = valid[a] & valid[b]
        pair &= np.abs(depth[a] - depth[b]) < .05 * np.minimum(depth[a], depth[b])
        # Face-normal sign ambiguity is removed; real depth boundaries excluded.
        dot = np.abs((n[a] * n[b]).sum(-1))
        bad += int((pair & (dot < .5)).sum())
        total += int(pair.sum())
    return bad / max(total, 1)


def coverage_stats(depth, camera, target):
    valid = np.isfinite(depth) & (depth > 0)
    hole = target & ~valid
    components, count = label(hole)
    largest = int(np.bincount(components.ravel())[1:].max()) if count else 0
    area = int(target.sum())
    return dict(target_pixels=area, hole_pixels=int(hole.sum()),
                target_coverage=float(valid[target].mean()),
                largest_hole_pixels=largest,
                largest_hole_fraction=largest / area,
                global_coverage=float(valid.mean()),
                severe_normal_change_fraction=normal_instability(depth, camera, target))


def choose_all(root, original, minimum_area_ratio=.6, coverage_tolerance=.002):
    selections, audits = {}, {}
    for scene in SCENES:
        info = read(root / 'common' / scene / 'complete.json')
        cameras = [c for c in info['cameras'] if c['part'] == 'held_out']
        old_camera = original['scenes'][scene]['camera']['name']
        rows = []
        for camera in cameras:
            mask_path = root / 'common' / scene / 'held_out' / (camera['stem'] + '_foreground.png')
            target = np.asarray(Image.open(mask_path)) > 127
            if not target.any():
                continue
            stats = {}
            for state in ['background', 'recomposition']:
                path = root / 'cache/paintmesh' / scene / state / 'depth' / (camera['stem'] + '.npy')
                depth = np.load(path)
                assert depth.shape == target.shape
                stats[state] = coverage_stats(depth, camera, target)
                for method in METHODS:
                    record = plots.sources(root, scene, method, method, state, camera)
                    assert record['rgb'].exists(), record['rgb']
                    assert record['depth'].exists(), record['depth']
                    assert plots.view_metric(record, camera) is not None, (scene, method, state, camera['name'])
                mask = plots.scoring_mask(root, scene, state, camera)
                assert mask is not None and mask.sum() >= 100
            rows.append(dict(camera=camera, target_fraction=float(target.mean()),
                original_camera=camera['name'] == old_camera,
                worst_hole_fraction=max(1 - s['target_coverage'] for s in stats.values()),
                worst_largest_hole_fraction=max(s['largest_hole_fraction'] for s in stats.values()),
                background_normal_instability=stats['background']['severe_normal_change_fraction'],
                minimum_global_coverage=min(s['global_coverage'] for s in stats.values()),
                stats=stats, roi=roi_for(camera, target)))
        assert len(rows) >= 2, scene
        threshold = minimum_area_ratio * max(r['target_fraction'] for r in rows)
        alternate = [r for r in rows if not r['original_camera']]
        eligible = [r for r in alternate if r['target_fraction'] >= threshold]
        fallback = not eligible
        effective_threshold = threshold
        if fallback:
            max_alternate = max(r['target_fraction'] for r in alternate)
            effective_threshold = minimum_area_ratio * max_alternate
            eligible = [r for r in alternate if r['target_fraction'] >= effective_threshold]
        best_holes = min(r['worst_hole_fraction'] for r in eligible)
        near_best = [r for r in eligible if r['worst_hole_fraction'] <= best_holes + coverage_tolerance]
        # At nearly identical pixel coverage, prefer stable local geometry,
        # then smaller connected holes; one isolated pixel is not decisive.
        selected = min(near_best, key=lambda r: (
            r['background_normal_instability'], r['worst_largest_hole_fraction'],
            r['worst_hole_fraction'], -r['minimum_global_coverage'],
            -r['target_fraction'], r['camera']['name']))
        for row in rows:
            row['eligible'] = row in eligible
            row['selected'] = row is selected
        _, scale = plots.reference_depth(root, scene, selected['camera'], 'recomposition')
        selections[scene] = dict(camera=selected['camera'], roi=selected['roi'],
            target_fraction=selected['target_fraction'], depth_range=scale.tolist(),
            original_camera=old_camera, selection_stats=selected['stats'],
            target_area_threshold=effective_threshold,
            original_target_area_threshold=threshold, target_area_fallback=fallback)
        audits[scene] = rows
        print('SELECTED', scene, old_camera, '->', selected['camera']['name'],
              'BG coverage', selected['stats']['background']['target_coverage'],
              'composed coverage', selected['stats']['recomposition']['target_coverage'],
              'area_fallback', fallback, flush=True)
    return selections, audits


def write_selection(out, selections, audits, original):
    metadata = out / 'metadata'; metadata.mkdir(parents=True, exist_ok=True)
    write(metadata / 'selected_views.json', dict(rule=RULE, scenes=selections,
        view_selection_is_method_favorable=True, quantitative_benchmark_unchanged=True))
    write(metadata / 'candidate_views.json', dict(rule=RULE, scenes=audits))
    with (metadata / 'candidate_views.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['scene', 'camera', 'original', 'eligible', 'selected', 'target_fraction',
            'bg_target_coverage', 'composed_target_coverage', 'bg_largest_hole_pixels',
            'composed_largest_hole_pixels', 'bg_severe_normal_changes'])
        for scene, rows in audits.items():
            for r in rows:
                writer.writerow([scene, r['camera']['name'], r['original_camera'], r['eligible'], r['selected'],
                    r['target_fraction'], r['stats']['background']['target_coverage'],
                    r['stats']['recomposition']['target_coverage'],
                    r['stats']['background']['largest_hole_pixels'],
                    r['stats']['recomposition']['largest_hole_pixels'], r['background_normal_instability']])


def original_snapshot(public):
    paths = sorted((public / 'figures_all_methods/overview').glob('*'))
    paths += [public / 'figures_all_methods/metadata/selected_views.json',
              public / 'figures_all_methods/metadata/manifest.json',
              public / 'figures_all_methods/subplots/manifest.json']
    paths += sorted((public / 'results').glob('*/*/*/metrics.json'))
    paths += sorted((public / 'reports').glob('*.md'))
    return {str(p): hash_file(p) for p in paths if p.is_file()}


def annotate_overview(canvas):
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 48, canvas.width, 83), fill=plots.BG)
    plots.text(draw, (16, 55),
        'Qualitative views selected for PaintMesh coverage / surface stability | Same camera for every method | P/S/L: selected view; CD/F1: scene',
        size=21)
    return canvas


def publish(public, out, select_only=False):
    root = work_root(public); configure_sources(root)
    original = read(public / 'figures_all_methods/metadata/selected_views.json')
    if out == public / 'figures_all_methods/overview':
        raise ValueError('An alternative output directory is required')
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'metadata/complete.json', dict(complete=False,
        stage='selecting and generating', updated=time.time()))
    frozen = original_snapshot(public)
    write(out / 'metadata/original_outputs_snapshot.json', frozen)
    selections, audits = choose_all(root, original)
    write_selection(out, selections, audits, original)
    if select_only:
        return

    # Reuse the established layout/render-derived geometry drawing functions.
    # These overrides are confined to this process, not repository source edits.
    plots.choose_view = lambda root, scene: (
        selections[scene]['camera'], selections[scene]['roi'], selections[scene]['target_fraction'])
    plots.figure_bucket = lambda folder, kind: Path(folder) if kind == 'overview' else Path(folder) / kind
    original_draw = plots.draw_scene
    original_overview = plots.overview

    def draw_selected(*args, **kwargs):
        canvas = original_draw(*args, **kwargs)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 48, canvas.width, 86), fill=plots.BG)
        camera = args[3]
        plots.text(draw, (18, 58),
            f'Camera: {camera["name"]} | Selected for PaintMesh target coverage/geometry; same camera for every method', size=21)
        return canvas

    plots.draw_scene = draw_selected
    plots.overview = lambda *args, **kwargs: annotate_overview(original_overview(*args, **kwargs))
    plots.build(root, out, SCENES)
    write_selection(out, selections, audits, original)

    records = []
    subplots = out / 'subplots'; subplots.mkdir(exist_ok=True)
    for scene in SCENES:
        info = read(out / 'scenes' / scene / 'figure_manifest.json')
        for mode in ['background', 'recomposition']:
            for method in ['gt', *METHODS]:
                record, _ = export_method(root, out, subplots, scene, mode, method, info)
                records.append(record)
        print('SELECTED_SUBPLOTS', scene, flush=True)
    write(subplots / 'manifest.json', dict(complete=True, directory_count=len(records), records=records,
        scenes=SCENES, methods=['gt', *METHODS], modes=['background', 'recomposition'],
        selection_rule=RULE, original_figures_unchanged=True))
    indexes(subplots, records, [])
    index = subplots / 'README.md'
    index.write_text(index.read_text().replace('原总图', '本目录总图'))

    # Verify every exported raw PNG exactly matches its actual renderer source,
    # every camera differs from the old overview, and all shared masks align.
    for record in records:
        scene, mode, method = record['scene'], record['mode'], record['method']
        assert record['original_image'] == selections[scene]['camera']['name']
        assert record['original_image'] != original['scenes'][scene]['camera']['name']
        folder = subplots / method / mode / scene
        assert set(FILES) <= set(record['generated_files']) if method != 'gt' else True
        assert hash_file(folder / 'rgb.png') == hash_file(record['source_rgb'])
        assert hash_file(folder / 'geometry.png') == hash_file(record['source_geometry'])
        with Image.open(folder / 'rgb.png') as im:
            assert im.size == (record['camera']['width'], record['camera']['height'])
    for filename, old_hash in frozen.items():
        assert hash_file(filename) == old_hash, f'Original result changed: {filename}'

    lines = ['# 新视角 Mesh 对比图', '',
        '本目录是按 PaintMesh 补全区域覆盖与局部表面稳定性挑选的定性展示，选图明确偏向我们方法；不代表随机视角或全场景平均质量提升。', '',
        '全部 11 个场景均更换了原 overview 的相机。每个场景的所有方法保持同一真实相机、同一完整画面和裁剪、同一深度色标。复用各方法已有的真实网格渲染，不修改 mesh、像素内容、渲染器、正式报告或指标。', '',
        '## 总览', '',
        '| 内容 | PNG | JPG | PDF |', '|---|---|---|---|']
    for key, name in [('rgb', 'RGB'), ('geometry', '几何'), ('background', '纯背景同域'), ('recomposition', '组合同域')]:
        stem = 'all_scenes_' + key
        lines.append(f'| {name} | [PNG]({stem}.png) | [JPG]({stem}.jpg) | [PDF]({stem}.pdf) |')
    lines += ['', '[逐场景八列详细图，多页 PDF](all_methods_all_scenes.pdf) · [纯背景详细 PDF](background_all_scenes.pdf) · [组合详细 PDF](recomposition_all_scenes.pdf)', '',
        '[全部独立子图](subplots/README.md)：`subplots/<方法>/<background或recomposition>/<场景>/`，含原尺寸 RGB、几何、放大裁剪、同域图、目标/覆盖/评分 mask、带指标的 panel 与来源 JSON。', '',
        '## 选图记录', '',
        '排除原 overview 相机；优先保留目标面积至少为原最大值 60% 的视角。若没有符合面积要求的新视角，改为至少达到其他视角最大目标面积的 60%，并明确标记。先最小化 background/composed 目标区域的最差破洞比例；在最优值 0.2 个百分点以内，先比较排除深度边缘后的剧烈法线变化比例，再比较最大连通破洞。前者仅是选图代理量，不是几何准确度指标。', '',
        '[全部候选 CSV](metadata/candidate_views.csv) · [候选与完整统计 JSON](metadata/candidate_views.json) · [选中相机、ROI、依据](metadata/selected_views.json)', '',
        '| 场景 | 原相机 | 新相机 | BG 目标覆盖率 | composed 目标覆盖率 | 面积条件 | 详细图 |',
        '|---|---|---|---:|---:|---|---|']
    for scene, s in selections.items():
        b = s['selection_stats']['background']['target_coverage']
        c = s['selection_stats']['recomposition']['target_coverage']
        flag = '按其他视角最大面积的60%筛选；不足原最大值60%' if s['target_area_fallback'] else '满足60%'
        lines.append(f'| {scene} | {s["original_camera"]} | {s["camera"]["name"]} | {b:.3%} | {c:.3%} | {flag} | [八列图](scenes/{scene}/all_methods.png) |')
    lines += ['', '覆盖率在目标 mask 内用各模式自身有限正深度计算，未使用背景评估中已经排除目标的评分 mask。洞少不能证明表面位置正确，也不能消除模型本身的几何缺陷。', '',
        '图中 PSNR/SSIM/LPIPS 为对应相机原有真实评分；CD/F1 是场景汇总，选角不会改变它。纯背景与组合的评分域不同。GT 几何列为实际 COLMAP 稀疏点投影，不是稠密扫描真值。', '',
        '[原总图](../overview/) · [原图索引](../README.md) · [纯背景指标](../../reports/background.md) · [组合指标](../../reports/recomposition.md)', '']
    (out / 'README.md').write_text('\n'.join(lines))
    write(out / 'metadata/complete.json', dict(complete=True, updated=time.time(),
        scenes=len(SCENES), alternative_cameras=len(selections), candidate_cameras=sum(map(len, audits.values())),
        subplot_directories=len(records), original_outputs_sha256_unchanged=True,
        raw_rgb_and_geometry_copies_verified=True, benchmark_metrics_unchanged=True,
        selection_rule=RULE))
    print('ALTERNATIVE_OVERVIEW_COMPLETE', out, len(records), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=plots.BASE / 'mesh_comparison_whole')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    public = args.root.resolve()
    out = (args.output or public / 'figures_all_methods/overview_selected_views').resolve()
    publish(public, out, args.select_only)


if __name__ == '__main__':
    main()
