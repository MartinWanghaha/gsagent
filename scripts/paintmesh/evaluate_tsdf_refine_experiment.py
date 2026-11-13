#!/usr/bin/env python3
"""Full-frame A/B mesh evaluation without changing any production outputs.

A is the exact published full-background TSDF triangle set. B preserves its
faces/colors and changes positions only. Four baselines retain their own native
mesh RGB/depth backends. All methods are rescored on one common visible support.
"""
from pathlib import Path
import argparse
import gc
import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mesh_benchmark.core import BASE, MeshScene, read, write, identity, digest, rays, surface_metrics
from mesh_benchmark.layout import work_root, configure_sources
from mesh_benchmark.stream_cache import outside_frustum
from evaluate_normal_mesh_experiment import (source_specs, cast_native, atomic_arrays,
    output_files, compose_arrays, render_specs)

PUBLIC = BASE / 'mesh_comparison_whole'
ROOT = work_root(PUBLIC)
DEFAULT = PUBLIC / 'experiments/bag_tsdf_refine_20261112'
BASELINES = ['2dgs', 'come', 'gof', 'gaussian_wrapping']
METHODS = ['A', 'B', *BASELINES]
LABELS = dict(A='PaintMesh TSDF (before)', B='PaintMesh depth/normal refinement',
    **dict(zip(BASELINES, ['2DGS', 'CoMe', 'GOF', 'Gaussian Wrapping'])))
STATES = ['background', 'composed']


def same_cameras(actual, expected):
    names = {c['name']: c for c in actual}
    if len(names) != len(actual) or set(names) != {c['name'] for c in expected}:
        raise ValueError('Native mesh-renderer camera set differs from the experiment')
    for camera in expected:
        other = names[camera['name']]
        for key in ['stem', 'width', 'height']:
            if other[key] != camera[key]:
                raise ValueError('Native camera name/image dimensions changed')
        for key in ['K', 'w2c']:
            if not np.array_equal(other[key], camera[key]):
                raise ValueError('Native camera calibration changed')


def covered_support(depths, foreground, background):
    foreground = np.asarray(foreground, dtype=bool)
    eligible = ~foreground if background else np.ones(foreground.shape, dtype=bool)
    shared = eligible.copy()
    if not depths:
        raise ValueError('At least one method depth image is required')
    for depth in depths:
        if np.shape(depth) != foreground.shape:
            raise ValueError('Depth and evaluation mask resolutions differ')
        shared &= np.isfinite(depth) & (depth > 0)
    return eligible, shared


def evaluation_cameras():
    configure_sources(ROOT)
    info = read(ROOT / 'common/bag/complete.json')
    cameras = [c for c in info['cameras'] if c['part'] == 'held_out']
    if not info.get('complete') or len(cameras) != 13 or info['held_out_count'] != 13:
        raise ValueError('All 13 original held-out bag views are required')
    same_cameras(cameras, cameras)
    return cameras


def status(output, stage, **detail):
    write(Path(output) / 'evaluation_status.json', dict(complete=False,
        updated=time.time(), pid=os.getpid(), stage=stage, **detail))


def possible_ray_indices(rr, bounds, nearest=None):
    """Conservative slab test, including parallel rays and nearest-hit pruning.

    Native rays are not unit length: their parameter is camera z. The same
    parameter therefore compares directly against the accumulated mesh depth.
    This rejects no possible frontmost triangle intersection.
    """
    rr = np.asarray(rr)
    lo, hi = np.asarray(bounds, dtype='float64')
    margin = 1e-5 * (1 + np.max(np.abs([lo, hi])))
    lo = lo - margin
    hi = hi + margin
    origin, direction = rr[:, :3], rr[:, 3:]
    parallel = direction == 0
    invalid = np.any(parallel & ((origin < lo) | (origin > hi)), axis=1)
    first = np.full(direction.shape, -np.inf, dtype='float64')
    last = np.full(direction.shape, np.inf, dtype='float64')
    np.divide(lo - origin, direction, out=first, where=~parallel)
    np.divide(hi - origin, direction, out=last, where=~parallel)
    near = np.max(np.minimum(first, last), axis=1)
    far = np.min(np.maximum(first, last), axis=1)
    keep = ~invalid & (far >= np.maximum(near, 0)) & (far > 0)
    if nearest is not None:
        nearest = np.asarray(nearest)
        if nearest.shape != (len(rr),):
            raise ValueError('Nearest depth shape differs from rays')
        keep &= (nearest <= 0) | ~np.isfinite(nearest) | (near <= nearest)
    return np.flatnonzero(keep)


def projected_indices(bounds, camera):
    """Conservative pixel rectangle of a positive-z AABB, or the full canvas."""
    import itertools
    w, h = camera['width'], camera['height']
    lo, hi = np.asarray(bounds, dtype='float64')
    margin = 1e-5 * (1 + np.max(np.abs([lo, hi])))
    corners = np.asarray(list(itertools.product(*zip(lo-margin, hi+margin))))
    T = np.asarray(camera['w2c'])
    pc = corners @ T[:3, :3].T + T[:3, 3]
    if not np.all(pc[:, 2] > 0):
        return np.arange(w*h)
    q = pc @ np.asarray(camera['K']).T
    uv = q[:, :2] / q[:, 2:3]
    xmin = max(0, int(np.floor(uv[:, 0].min())) - 2)
    xmax = min(w, int(np.ceil(uv[:, 0].max())) + 2)
    ymin = max(0, int(np.floor(uv[:, 1].min())) - 2)
    ymax = min(h, int(np.ceil(uv[:, 1].max())) + 2)
    if xmin >= xmax or ymin >= ymax:
        return np.empty(0, dtype='int64')
    return (np.arange(ymin, ymax)[:, None] * w + np.arange(xmin, xmax)).reshape(-1)


def render_streamed(folder, specs, cameras, checkpoint_parts=32):
    """Render every original triangle; AABB pruning only saves impossible rays."""
    if not specs or checkpoint_parts < 1:
        raise ValueError('At least one mesh and a positive checkpoint interval are required')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request = dict(inputs=source_specs(specs), cameras=cameras,
        implementation=identity(Path(__file__)),
        rendering_helper=identity(HERE / 'evaluate_normal_mesh_experiment.py'),
        backend='PaintMesh Embree original-triangle RGB/geometric normals with conservative AABB ray pruning',
        full_scene=True, additional_image_downsampling=False, mesh_decimation=False)
    request_id = digest(request)
    marker = folder / 'complete.json'
    if marker.exists():
        old = read(marker)
        if old.get('complete') and old.get('request_id') == request_id:
            try:
                current = output_files(folder, cameras)
            except FileNotFoundError:
                current = None
            if current is not None and old.get('outputs') == current:
                return old
    marker.unlink(missing_ok=True)
    saved = folder / 'checkpoint.json'
    start, previous = 0, None
    if saved.exists():
        old = read(saved)
        if old['request_id'] != request_id:
            raise ValueError('Render checkpoint inputs changed')
        if identity(old['arrays']['path']) != old['arrays']:
            raise ValueError('Render checkpoint arrays changed')
        previous = Path(old['arrays']['path'])
        start = old['completed_parts']
        with np.load(previous) as data:
            arrays = {k: data[k].copy() for k in data.files}
    else:
        arrays = {}
        for i, c in enumerate(cameras):
            n = c['width'] * c['height']
            arrays[f'depth_{i}'] = np.zeros(n, dtype='float32')
            arrays[f'rgb_{i}'] = np.zeros((n, 3), dtype='float32')
            arrays[f'normal_{i}'] = np.zeros((n, 3), dtype='float32')
    rr = [rays(c) for c in cameras]
    started = time.time()
    cast_rays = 0
    for j, spec in enumerate(specs):
        if j < start:
            continue
        mesh = MeshScene([spec])
        if not all(r['has_native_color'] for r in mesh.counts):
            raise ValueError('Mesh is missing native colors')
        box = mesh.counts[0]['bounds']
        for i, camera in enumerate(cameras):
            if outside_frustum(box, camera):
                continue
            candidates = projected_indices(box, camera)
            depth = arrays[f'depth_{i}']
            selected = candidates[possible_ray_indices(rr[i][candidates], box, depth[candidates])]
            if not len(selected):
                continue
            cast_rays += len(selected)
            z, rgb, normal = cast_native(mesh, rr[i][selected])
            current = depth[selected]
            hit = (z > 0) & ((current <= 0) | (z < current))
            front = selected[hit]
            depth[front] = z[hit]
            arrays[f'rgb_{i}'][front] = rgb[hit]
            arrays[f'normal_{i}'][front] = normal[hit]
        del mesh
        gc.collect()
        print('TSDF_REFINE_RENDER', j+1, len(specs), 'cast_rays', cast_rays,
            'elapsed', round(time.time()-started, 2), flush=True)
        if (j+1) % checkpoint_parts == 0 or j+1 == len(specs):
            current = folder / f'checkpoint_{j+1:06d}.npz'
            atomic_arrays(current, arrays)
            write(saved, dict(request_id=request_id, completed_parts=j+1, arrays=identity(current)))
            if previous is not None and previous != current:
                previous.unlink(missing_ok=True)
            previous = current
            write(folder / 'status.json', dict(updated=time.time(), stage='rendering',
                completed_parts=j+1, total_parts=len(specs), views=len(cameras),
                all_source_faces_preserved=True, cast_rays_this_process=cast_rays))
    for i, camera in enumerate(cameras):
        shape = camera['height'], camera['width']
        np.save(folder / (camera['stem']+'_depth.npy'), arrays[f'depth_{i}'].reshape(shape))
        np.save(folder / (camera['stem']+'_normal.npy'), arrays[f'normal_{i}'].reshape(*shape, 3))
        rgb = np.rint(np.clip(arrays[f'rgb_{i}'], 0, 1)*255).astype('uint8').reshape(*shape, 3)
        Image.fromarray(rgb).save(folder / (camera['stem']+'.png'))
    result = dict(complete=True, request=request, request_id=request_id, views=len(cameras),
        all_triangles_rendered=True, full_scene=True, roi=None,
        gaussian_rgb_substitution=False, outputs=output_files(folder, cameras))
    write(marker, result)
    if previous is not None:
        previous.unlink(missing_ok=True)
    saved.unlink(missing_ok=True)
    return result


def validate_manifest(output):
    path = Path(output) / 'mesh_manifest.json'
    manifest = read(path)
    if manifest.get('scene') != 'bag':
        raise ValueError('This experiment is scoped to bag')
    for name in ['A', 'B']:
        method = manifest[name]
        if not method.get('complete') or not method.get('mesh_specs'):
            raise ValueError('Incomplete full-scene mesh: '+name)
        source_specs(method['mesh_specs'])
    a, b = manifest['A']['mesh_specs'], manifest['B']['mesh_specs']
    if len(a) != len(b):
        raise ValueError('A/B must preserve all source partitions')
    for aa, bb in zip(a, b):
        if aa.get('transform') != bb.get('transform'):
            raise ValueError('A/B partition transforms differ')
        if Path(aa['path']).name.endswith('.mesh.json') and Path(bb['path']).name.endswith('.mesh.json'):
            ma, mb = read(aa['path']), read(bb['path'])
            for key in ['vertex_count', 'face_count', 'index_identity']:
                if ma[key] != mb[key]:
                    raise ValueError('A/B topology changed: '+key)
    return manifest


def combine_render_folders(background, objects, destination, cameras, scene_identity):
    background, objects, destination = map(Path, [background, objects, destination])
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {name: read(folder / 'complete.json') for name, folder in [('background', background), ('objects', objects)]}
    for name, folder in [('background', background), ('objects', objects)]:
        if not metadata[name].get('complete') or metadata[name]['outputs'] != output_files(folder, cameras):
            raise ValueError('Incomplete or changed actual triangle rendering: '+name)
    request = dict(background=metadata['background']['request_id'], objects=metadata['objects']['request_id'],
        scene=scene_identity, implementation=identity(Path(__file__)),
        composition='Nearest positive finite true triangle depth; fixed generated object and pose')
    marker = destination / 'complete.json'
    if marker.exists():
        old = read(marker)
        if old.get('complete') and old.get('request') == request:
            try:
                current = output_files(destination, cameras)
            except FileNotFoundError:
                current = None
            if current is not None and old.get('outputs') == current:
                return old
    marker.unlink(missing_ok=True)
    for camera in cameras:
        stem = camera['stem']
        def load(folder):
            return (np.load(folder / (stem+'_depth.npy')),
                np.asarray(Image.open(folder / (stem+'.png')).convert('RGB')),
                np.load(folder / (stem+'_normal.npy')))
        depth, rgb, normal = compose_arrays(*load(background), *load(objects))
        np.save(destination / (stem+'_depth.npy'), depth)
        np.save(destination / (stem+'_normal.npy'), normal)
        Image.fromarray(rgb).save(destination / (stem+'.png'))
    result = dict(complete=True, request=request, request_id=digest(request), views=len(cameras),
        frozen_object_geometry=True, frozen_object_pose=True, all_triangles_rendered=True,
        gaussian_rgb_substitution=False, outputs=output_files(destination, cameras))
    write(marker, result)
    return result


def render_experiment(output, cameras, manifest):
    output = Path(output)
    graph_path = PUBLIC / 'recomposition/bag/composition/scene.json'
    graph = read(graph_path)
    if not graph.get('complete') or not graph.get('objects'):
        raise ValueError('Fixed composed object source is incomplete')
    object_specs = [dict(path=item['mesh']['path'], transform=item['object_to_world']) for item in graph['objects']]
    status(output, 'rendering all refined triangle partitions', evaluation_views=13)
    render_streamed(output / 'renders/B/background', manifest['B']['mesh_specs'], cameras)
    status(output, 'rendering fixed generated object', evaluation_views=13)
    render_specs(output / 'renders/objects', object_specs, cameras)
    combine_render_folders(output / 'renders/B/background', output / 'renders/objects',
        output / 'renders/B/composed', cameras, identity(graph_path))


def native_sources(output, cameras, manifest):
    from evaluate_native_mesh_rgb import rendering, verified_file_fingerprint
    output = Path(output)
    sources = {state: {} for state in STATES}
    stamps = []
    common = read(ROOT / 'common/bag/complete.json')
    for method in BASELINES:
        folder = ROOT / 'native_renderers' / method / 'bag'
        marker = folder / 'complete.json'
        meta = read(marker)
        if not meta.get('complete') or meta.get('pilot') or meta.get('views') != 13:
            raise ValueError('Native baseline rendering is incomplete: '+method)
        if not meta.get('all_source_triangles_preserved') or meta['request'].get('gaussian_renderer_used'):
            raise ValueError('Real complete native mesh renders are required')
        same_cameras(meta['request']['cameras'], cameras)
        fingerprint = meta['request']['mesh']
        if verified_file_fingerprint(output, fingerprint['path']) != fingerprint:
            raise ValueError('Native baseline mesh differs from its renderer source: '+method)
        stamps.append(identity(marker))
        for state in STATES:
            sources[state][method] = dict(rgb=folder / 'renders', depth=folder / 'depth',
                backend=meta['request']['backend'], renderer=identity(marker), depth_suffix='.npy')
    for state in STATES:
        original_state = 'recomposition' if state == 'composed' else 'background'
        folder = ROOT / 'cache/paintmesh/bag' / original_state
        marker = folder / 'complete.json'
        meta = read(marker)
        if not meta.get('complete') or meta['request'].get('common') != common['request_id']:
            raise ValueError('Published A renderer differs from experiment cameras')
        expected = [dict(identity=identity(s['path']), transform=s.get('transform')) for s in manifest['A']['mesh_specs']]
        # The composed A cache must also use the exact currently frozen object pose.
        if state == 'composed':
            graph = read(PUBLIC / 'recomposition/bag/composition/scene.json')
            expected += [dict(identity=identity(o['mesh']['path']), transform=o['object_to_world']) for o in graph['objects']]
        if meta['request']['specs'] != expected:
            raise ValueError('A manifest/object placement is not the exact published triangle set')
        for item in meta['request']['specs']:
            if identity(item['identity']['path']) != item['identity']:
                raise ValueError('Published A geometry changed')
        native = rendering(ROOT, 'paintmesh', 'bag', original_state)
        if native['rgb'] != folder / 'native_renders':
            raise ValueError('Learned color-field RGB is forbidden')
        sources[state]['A'] = dict(rgb=native['rgb'], depth=native['depth'],
            backend=native['request']['backend'], renderer=identity(marker), depth_suffix='.npy')
        stamps.append(identity(marker))
        own = output / 'renders/B' / state
        own_meta = read(own / 'complete.json')
        if not own_meta.get('complete') or own_meta['outputs'] != output_files(own, cameras):
            raise ValueError('Refined full renders are incomplete or changed')
        sources[state]['B'] = dict(rgb=own, depth=own, backend='PaintMesh actual triangle Embree with native vertex RGB',
            renderer=identity(own / 'complete.json'), depth_suffix='_depth.npy')
        stamps.append(identity(own / 'complete.json'))
    for state in STATES:
        for method, source in sources[state].items():
            source['files'] = []
            for camera in cameras:
                rgb = source['rgb'] / (camera['stem']+'.png')
                depth = source['depth'] / (camera['stem']+source['depth_suffix'])
                if Image.open(rgb).size != (camera['width'], camera['height']):
                    raise ValueError('Native RGB resolution mismatch: '+str(rgb))
                if np.load(depth, mmap_mode='r').shape != (camera['height'], camera['width']):
                    raise ValueError('Native depth resolution mismatch: '+str(depth))
                source['files'].extend([identity(rgb), identity(depth)])
            stamps.extend(source['files'])
    return sources, stamps


def average(rows, keys):
    return {key: float(np.mean([r[key] for r in rows if r.get(key) is not None]))
        if any(r.get(key) is not None for r in rows) else None for key in keys}


def depth_normals(depth, camera):
    """Common geometric visualization from actual triangle-rendered z depth."""
    depth = np.asarray(depth, dtype='float64')
    h, w = depth.shape
    uv = np.stack(np.meshgrid(np.arange(w)+.5, np.arange(h)+.5), axis=-1)
    k = np.asarray(camera['K'])
    ray = np.dstack([(uv[..., 0]-k[0, 2])/k[0, 0], (uv[..., 1]-k[1, 2])/k[1, 1], np.ones((h, w))])
    points = ray * depth[..., None]
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1)
    valid = np.isfinite(depth) & (depth > 0)
    inner = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    inner &= np.isfinite(norm) & (norm > 1e-12)
    n = np.divide(n, norm[..., None], out=np.zeros_like(n), where=norm[..., None] > 1e-12)
    n = np.where((np.sum(n*ray[1:-1, 1:-1], axis=-1) > 0)[..., None], -n, n)
    image = np.zeros((h, w, 3), dtype='uint8')
    colors = np.rint(np.clip((n+1)*.5, 0, 1)*255).astype('uint8')
    image[1:-1, 1:-1][inner] = colors[inner]
    return image


def figures(output, cameras, sources, results=None):
    from plot_mesh_method_comparison import font
    output = Path(output)
    inventory = []
    for state in STATES:
        rgb_folder = output / 'figures' / state / 'rgb'
        geometry_folder = output / 'figures' / state / 'geometry'
        ab_folder = output / 'figures' / state / 'before_after'
        for folder in [rgb_folder, geometry_folder, ab_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        overview_rows = []
        for i, camera in enumerate(cameras):
            w, h = camera['width'], camera['height']
            row_height = h+77
            rgb_canvas = Image.new('RGB', (4*w, 2*row_height+55), (247, 248, 250))
            geo_canvas = rgb_canvas.copy()
            rd, gd = ImageDraw.Draw(rgb_canvas), ImageDraw.Draw(geo_canvas)
            title = 'bag | '+camera['name']+' | '+state+' | all source triangles, original-size panels'
            rd.text((10, 8), title, font=font(19), fill='black')
            gd.text((10, 8), title+' | depth-derived geometric normals', font=font(18), fill='black')
            overview = Image.new('RGB', (7*w, h+70), (247, 248, 250))
            od = ImageDraw.Draw(overview)
            od.text((8, 3), camera['name']+' | '+state, font=font(18), fill='black')
            for j, method in enumerate(['GT', *METHODS]):
                x, y = (j % 4)*w, 55+(j//4)*row_height
                image_path = ROOT / 'common/bag/held_out' / (camera['stem']+'_gt.png') if method == 'GT' else sources[state][method]['rgb'] / (camera['stem']+'.png')
                image = Image.open(image_path).convert('RGB')
                if image.size != (w, h):
                    raise ValueError('Figure images must retain the original native resolution')
                label = 'GT photo' if method == 'GT' else LABELS[method]
                rd.text((x+7, y+2), label, font=font(17), fill='black')
                gd.text((x+7, y+2), label, font=font(17), fill='black')
                rgb_canvas.paste(image, (x, y+25))
                overview.paste(image, (j*w, 70))
                od.text((j*w+7, 36), label, font=font(17), fill='black')
                subplot = output / 'subplots' / method / state
                subplot.mkdir(parents=True, exist_ok=True)
                image.save(subplot / (camera['stem']+'.png'))
                if method == 'GT':
                    geo_canvas.paste(image, (x, y+25))
                    continue
                source = sources[state][method]
                depth = np.load(source['depth'] / (camera['stem']+source['depth_suffix']))
                normal_image = Image.fromarray(depth_normals(depth, camera))
                normal_image.save(subplot / (camera['stem']+'_geometry.png'))
                geo_canvas.paste(normal_image, (x, y+25))
                own = np.isfinite(depth) & (depth > 0)
                text = f'coverage over full canvas {own.mean():.1%}'
                if results is not None:
                    metric = results[state+'/'+method]['per_view'][i]
                    text = 'no common covered pixels' if metric['PSNR'] is None else f'P {metric["PSNR"]:.2f} | S {metric["SSIM"]:.3f} | L {metric["LPIPS"]:.3f}'
                    text += f' | eligible coverage {metric["mesh_coverage"]:.1%}'
                rd.text((x+7, y+h+33), text, font=font(14), fill='black')
                gd.text((x+7, y+h+33), 'Same z-depth normal visualization; black = unavailable', font=font(14), fill='black')
            rgb_path, geo_path = rgb_folder / (camera['stem']+'.png'), geometry_folder / (camera['stem']+'.png')
            rgb_canvas.save(rgb_path)
            geo_canvas.save(geo_path)
            # Two full-size before/after panels, with no crops or resizing.
            pair = Image.new('RGB', (2*w, h+60), (247, 248, 250))
            pd = ImageDraw.Draw(pair)
            for j, method in enumerate(['A', 'B']):
                pd.text((j*w+7, 5), LABELS[method]+' | '+camera['name'], font=font(17), fill='black')
                pair.paste(Image.open(sources[state][method]['rgb'] / (camera['stem']+'.png')).convert('RGB'), (j*w, 60))
            pair.save(ab_folder / (camera['stem']+'.png'))
            overview_rows.append(overview)
            inventory.append(dict(camera=camera['name'], state=state, rgb=str(rgb_path), geometry=str(geo_path), native_panel_size=[w, h]))
        width = overview_rows[0].width
        overview = Image.new('RGB', (width, sum(row.height for row in overview_rows)), 'white')
        offset = 0
        for row in overview_rows:
            overview.paste(row, (0, offset))
            offset += row.height
        overview.save(output / 'figures' / (state+'_all_views.jpg'), quality=95)
    write(output / 'figures/manifest.json', dict(images_complete=True, metrics_complete=results is not None,
        updated=time.time(), views=13, states=STATES, rgb_figures=26, geometry_figures=26,
        before_after_figures=26, records=inventory, images_resized=False,
        geometry_visualization='Identical camera-space normals derived from actual mesh z-depth for every method; not learned shading normals'))
    lines = ['# bag TSDF 顶点微调：完整视角对比', '',
        '每个子图保留原生尺寸。列为 GT、PaintMesh TSDF 初值、深度/几何法线微调、2DGS、CoMe、GOF、Gaussian Wrapping。', '',
        '几何图采用每个方法真实 mesh 深度计算的同一套相机空间法线可视化；黑色表示无有效表面或无有效邻域，不是任意学习的着色法线。', '',
        '[纯背景全部视角总图](background_all_views.jpg) · [组合全部视角总图](composed_all_views.jpg)', '',
        '| 视角 | 纯背景 RGB | 纯背景几何 | 组合 RGB | 组合几何 |', '|---|---|---|---|---|']
    for camera in cameras:
        stem = camera['stem']
        lines.append(f'| {camera["name"]} | [PNG](background/rgb/{stem}.png) | [PNG](background/geometry/{stem}.png) | [PNG](composed/rgb/{stem}.png) | [PNG](composed/geometry/{stem}.png) |')
    lines += ['', '指标'+('已完成。' if results is not None else '仍在计算；图片不代表最终验收。'), '正式项目输出未修改。']
    (output / 'figures/README.md').write_text('\n'.join(lines)+'\n')


def score(output, cameras, sources, device='cpu'):
    import torch
    import lpips
    from mesh_benchmark.score import image_metrics
    from mesh_benchmark.overlap import observations, select_anchors, prediction_points
    output = Path(output)
    status(output, 'scoring common visible surfaces', evaluation_views=13, metric_device=device)
    torch.set_num_threads(4)
    criterion = lpips.LPIPS(net='vgg', version='0.1', spatial=True).to(device).eval()
    for parameter in criterion.parameters():
        parameter.requires_grad_(False)
    info = read(ROOT / 'common/bag/complete.json')
    results = {}
    for state in STATES:
        depths = {method: [np.load(source['depth'] / (camera['stem']+source['depth_suffix']), mmap_mode='r')
            for camera in cameras] for method, source in sources[state].items()}
        eligible, shared = [], []
        folder = output / 'support' / state
        folder.mkdir(parents=True, exist_ok=True)
        for i, camera in enumerate(cameras):
            foreground = np.asarray(Image.open(ROOT / 'common/bag/held_out' / (camera['stem']+'_foreground.png'))) > 127
            allowed, mask = covered_support([depths[m][i] for m in METHODS], foreground, state == 'background')
            eligible.append(allowed)
            shared.append(mask)
            Image.fromarray(mask.astype('uint8')*255).save(folder / (camera['stem']+'.png'))
        with np.load(ROOT / 'common/bag/reference_points.npz') as data:
            reference = data['background' if state == 'background' else 'recomposition']
        obs = observations('bag', cameras, reference)
        selected = select_anchors(obs[:, 0], obs[:, 1], obs[:, 2], shared,
            np.ones(len(reference), dtype=bool), 42, max_points=max(1, len(reference))) if len(obs) else np.empty(0, dtype='int64')
        np.savez(folder / 'geometry_reference.npz', points=reference[obs[selected, 0]], observations=obs[selected])
        for method in METHODS:
            rows = []
            for i, camera in enumerate(cameras):
                target = np.asarray(Image.open(ROOT / 'common/bag/held_out' / (camera['stem']+'_gt.png')).convert('RGB'))
                rgb = np.asarray(Image.open(sources[state][method]['rgb'] / (camera['stem']+'.png')).convert('RGB'))
                own = eligible[i] & np.isfinite(depths[method][i]) & (depths[method][i] > 0)
                if rgb.shape != target.shape or rgb.shape[:2] != own.shape:
                    raise ValueError('No evaluation image resizing is permitted')
                rows.append(dict(name=camera['name'], **image_metrics(rgb, target, shared[i], criterion, device),
                    eligible_pixels=int(eligible[i].sum()), evaluation_pixels=int(shared[i].sum()), own_pixels=int(own.sum()),
                    mesh_coverage=float(own.sum()/max(1, eligible[i].sum())),
                    common_coverage=float(shared[i].sum()/max(1, eligible[i].sum()))))
            geometry = surface_metrics(prediction_points(cameras, depths[method], selected, obs),
                reference[obs[selected, 0]], info['extent']) if len(selected) else None
            result = dict(complete=True, method=method, label=LABELS[method], state=state, evaluation_views=13,
                scored_views=sum(r['evaluation_pixels'] > 0 for r in rows), per_view=rows,
                mean=average(rows, ['PSNR', 'SSIM', 'LPIPS', 'mesh_coverage', 'common_coverage']), geometry=geometry,
                geometry_reference_observations=len(selected), geometry_eligible_reference_points=len(reference),
                renderer=sources[state][method]['backend'], gaussian_rgb_substitution=False,
                appearance_field_fitted=False,
                reference_warning='Visible-surface consistency against actual sparse COLMAP observations; not independent scan GT or unobserved-surface completeness')
            results[state+'/'+method] = result
            write(output / 'metrics' / state / (method+'.json'), result)
            print('TSDF_REFINE_METRICS', state, method, result['mean'], 'reference_points', len(selected), flush=True)
            gc.collect()
        write(folder / 'complete.json', dict(complete=True, methods=METHODS, evaluation_views=13,
            reference_observations=len(selected), selection='Shared positive finite triangle depth; background excludes target mask; actual track observations; no error-based filtering'))
    return results


def reports(output, results):
    output = Path(output)
    folder = output / 'reports'
    folder.mkdir(exist_ok=True)
    for state in STATES:
        lines = ['# bag：'+('纯背景（目标 mask 内不评分）' if state == 'background' else '完整背景与相同生成物体组合'), '',
            '同一 13 个原始留出相机与原生像素尺寸。四个基线复用各自项目的原生 mesh 渲染；A/B 使用 PaintMesh 的真实三角形后端，颜色与连接不变。组合模式使用相同物体和位姿。', '',
            '所有指标在六个方法共同正深度覆盖域重新计算，同时给出自身覆盖率。纯背景排除相同目标 mask；不能以丢失表面换取更高指标。', '',
            '几何参考是 COLMAP 实际特征轨迹的稀疏点，并非独立扫描真值。CD-L1 是双向平均最近距离半和；F1 使用固定共同相机半径的 0.5%、1%、2% 距离阈值。', '',
            '| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 自身覆盖率 ↑ | 共同覆盖率 | CD-L1 ↓ | F1@0.5% ↑ | F1@1% ↑ | F1@2% ↑ | 参考点数 | 评分视角/全部 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for method in METHODS:
            row = results[state+'/'+method]
            a, g = row['mean'], row['geometry']
            values = [a[key] for key in ['PSNR', 'SSIM', 'LPIPS', 'mesh_coverage', 'common_coverage']]
            values += [g['CD_L1'] if g else None, *[g['F_scores'][key]['F1'] if g else None for key in ['0.005', '0.01', '0.02']]]
            lines.append('| '+' | '.join([LABELS[method], *['—' if v is None else f'{v:.6f}' for v in values],
                str(row['geometry_reference_observations']), f'{row["scored_views"]}/13'])+' |')
        lines += ['', '[全部视角 RGB 与几何对比](../figures/README.md) · [逐视角与几何数据](../results.json)', '',
            '没有共同有效像素的视角仍保留并标记无评分域；自身覆盖率计入全部 13 个视角。正式结果未覆盖。']
        (folder / (state+'.md')).write_text('\n'.join(lines)+'\n')


def run(output, render_only=False, score_only=False, device='cpu'):
    output = Path(output).absolute()
    if not output.resolve().is_relative_to((PUBLIC / 'experiments').resolve()):
        raise ValueError('Write only an isolated directory under mesh_comparison_whole/experiments')
    output.mkdir(parents=True, exist_ok=True)
    configure_sources(ROOT)
    (output / 'evaluation_complete.json').unlink(missing_ok=True)
    (output / 'results.json').unlink(missing_ok=True)
    cameras = evaluation_cameras()
    manifest = validate_manifest(output)
    frozen = dict(manifest=identity(output / 'mesh_manifest.json'),
        methods={m: source_specs(manifest[m]['mesh_specs']) for m in ['A', 'B']},
        scene=identity(PUBLIC / 'recomposition/bag/composition/scene.json'))
    if not score_only:
        render_experiment(output, cameras, manifest)
    sources, stamps = native_sources(output, cameras, manifest)
    request = dict(implementation=identity(Path(__file__)), inputs=frozen,
        methods=METHODS, cameras=cameras, renders=stamps, full_scene=True,
        additional_downsampling=False, production_pipeline_modified=False,
        appearance='Unchanged native vertex RGB and same generated-object texture')
    status(output, 'publishing complete-view comparisons', evaluation_views=13)
    figures(output, cameras, sources)
    write(output / 'sources.json', request)
    (output / 'README.md').write_text('# bag TSDF 深度/几何法线顶点微调\n\n[完整视角对比图](figures/README.md) · [实时评估状态](evaluation_status.json) · [输入来源](sources.json)\n\n图片已生成，指标尚未完成。所有原始三角形和原生像素参与，正式结果未覆盖。\n')
    if render_only:
        status(output, 'all RGB/geometry comparison images ready; metrics not requested',
            images_complete=True, metrics_complete=False)
        return
    results = score(output, cameras, sources, device)
    for stamp in stamps:
        if identity(stamp['path']) != stamp:
            raise ValueError('A native renderer input changed during evaluation: '+stamp['path'])
    for method in ['A', 'B']:
        if source_specs(manifest[method]['mesh_specs']) != frozen['methods'][method]:
            raise ValueError('A/B mesh changed during evaluation')
    if identity(output / 'mesh_manifest.json') != frozen['manifest'] or identity(PUBLIC / 'recomposition/bag/composition/scene.json') != frozen['scene']:
        raise ValueError('Manifest or composed object source changed during evaluation')
    figures(output, cameras, sources, results)
    reports(output, results)
    result = dict(complete=True, updated=time.time(), request=request, methods=METHODS,
        results=results, evaluation_views=13, states=STATES, full_scene=True,
        rgb_comparison_figures=26, geometry_comparison_figures=26,
        production_pipeline_modified=False, mesh_decimation=False,
        additional_image_downsampling=False, uv_baking_performed=False)
    write(output / 'results.json', result)
    write(output / 'evaluation_complete.json', dict(complete=True, request_id=digest(request),
        updated=time.time(), evaluation_views=13, rgb_figures=26, geometry_figures=26))
    write(output / 'evaluation_status.json', dict(complete=True, stage='complete',
        updated=time.time(), evaluation_views=13, rgb_figures=26, geometry_figures=26))
    (output / 'README.md').write_text('# bag TSDF 深度/几何法线顶点微调\n\n[完整视角 RGB 与几何对比](figures/README.md) · [纯背景报告](reports/background.md) · [组合报告](reports/composed.md) · [逐视角数据](results.json) · [输入来源](sources.json)\n\nA 为正式完整 TSDF，B 仅微调顶点；四个基线保留各自原生 mesh 后端。颜色、物体与位姿固定，无新纹理烘焙，无 GS RGB 替代。正式项目流程和报告未修改。\n\n几何参考为实际观测的 COLMAP 稀疏点，非独立扫描真值。指标只比较共同可见表面，同时报告覆盖率。\n')
    print('BAG_TSDF_REFINE_EVALUATION_COMPLETE', output, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--device', default='cpu', help='LPIPS only, e.g. cpu or cuda:0')
    args = parser.parse_args()
    if args.render_only and args.score_only:
        parser.error('Choose at most one partial stage')
    run(args.output, args.render_only, args.score_only, args.device)
