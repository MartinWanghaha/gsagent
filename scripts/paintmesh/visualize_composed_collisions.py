#!/usr/bin/env python3
"""Visualize completed composition collision audits without modifying scenes.

The original full-frame RGB/depth caches supply the overview camera. This is
an annotation tool, not a renderer or collision detector. Side-view occupancy
maps bin every supplied vertex/intersection into display pixels; they neither
decimate the mesh nor change the audit's triangle tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO = Path(__file__).resolve().parents[2]
PUBLIC = REPO / 'output/3dgs_with_mesh/mesh_comparison_whole'
SCENES = ['bag', 'car', 'cone_red', 'cone_yellow', 'cube', 'doppelherz',
          'fruits', 'garden_toys', 'redbull', 'toys', 'truck']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
RED = (238, 39, 65)
ORANGE = (255, 131, 27)
YELLOW = (249, 202, 36)
INK = (30, 36, 44)
BG = (248, 249, 251)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def stamp(path):
    p = Path(path).resolve(strict=True)
    s = p.stat()
    return dict(path=str(p), size_bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def font(size=17):
    return ImageFont.truetype(FONT, size)


def save_image(image, path):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.tmp')
    image.save(tmp, format='PNG')
    tmp.replace(path)


def audit_folder(output, scene):
    candidates = [Path(output) / 'scenes' / scene, Path(output) / scene]
    return next((p for p in candidates if (p / 'audit.json').exists()), candidates[0])


def camera_and_sources(public, scene):
    manifest_path = public / 'figures_all_methods/scenes' / scene / 'figure_manifest.json'
    meta = read(manifest_path)
    camera = meta['selection']['camera']
    records = [r for r in meta['records'] if r.get('method') == 'paintmesh'
               and r.get('state') in ['composed', 'recomposition']]
    if len(records) != 1:
        raise ValueError('Exactly one published composed mesh rendering is required')
    rgb_path, depth_path = Path(records[0]['rgb_source']), Path(records[0]['depth_source'])
    rgb = Image.open(rgb_path).convert('RGB')
    depth = np.load(depth_path)
    expected = (camera['width'], camera['height'])
    if rgb.size != expected or depth.shape != expected[::-1]:
        raise ValueError('RGB/depth cache differs from the selected native camera')
    return camera, rgb, depth, dict(figure_manifest=stamp(manifest_path),
                                   rgb=stamp(rgb_path), depth=stamp(depth_path))


def project(points, camera):
    points = np.asarray(points, dtype='float64')
    t, k = np.asarray(camera['w2c']), np.asarray(camera['K'])
    q = points @ t[:3, :3].T + t[:3, 3]
    z = q[:, 2]
    homogeneous = q @ k.T
    uv = np.divide(homogeneous[:, :2], z[:, None],
                   out=np.full((len(points), 2), np.nan), where=z[:, None] > 0)
    return uv, z


def native_visibility(points, camera, depth, tolerance):
    """Approximate display visibility only; does not classify any collisions."""
    uv, z = project(points, camera)
    finite = np.isfinite(uv).all(1) & np.isfinite(z) & (z > 0)
    ij = np.zeros((len(points), 2), dtype='int64')
    ij[finite] = np.floor(uv[finite]).astype('int64')
    w, h = camera['width'], camera['height']
    inside = finite & (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    visible = np.zeros(len(points), dtype=bool)
    ids = np.flatnonzero(inside)
    zz = depth[ij[ids, 1], ij[ids, 0]]
    visible[ids] = np.isfinite(zz) & (zz > 0) & (z[ids] <= zz + tolerance)
    return uv, inside, visible


def load_intersections(folder, object_id):
    path = folder / 'objects' / str(object_id) / 'intersections.npz'
    with np.load(path, allow_pickle=False) as d:
        values = {k: d[k] for k in d.files}
    count = len(values['classification'])
    for name, tail in [('positions', (3,)), ('segments', (2, 3)), ('face_pairs', (2,))]:
        if count == 0 and values[name].size == 0:
            values[name] = values[name].reshape((0, *tail))
        if values[name].shape != (count, *tail):
            raise ValueError(f'Unexpected {name} collision buffer shape: {path}')
    values['classification'] = values['classification'].astype(str)
    return values, path


def classification_counts(rows):
    if not len(rows):
        return {}
    names, counts = np.unique(rows, return_counts=True)
    return dict(zip(names.tolist(), counts.astype(int).tolist()))


def contact_mask(classification):
    return np.isin(classification, ['boundary_touch', 'coplanar_overlap', 'point_touch'])


def mark_points(layer, uv, keep, color):
    """Native-pixel occupancy uses every supplied point, merging coincident marks."""
    if not keep.any():
        return
    uv = np.floor(uv[keep]).astype('int64')
    w, h = layer.size
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[valid]
    if not len(uv):
        return
    arr = np.zeros((h, w, 4), dtype='uint8')
    arr[uv[:, 1], uv[:, 0]] = [*color, 235]
    # A one-pixel center plus a faint 3x3 halo aids visibility at native size.
    from PIL import ImageFilter
    marks = Image.fromarray(arr, 'RGBA')
    alpha = marks.getchannel('A').filter(ImageFilter.MaxFilter(3))
    halo = Image.new('RGBA', (w, h), (*color, 0))
    halo.putalpha(alpha.point(lambda x: int(x * .4)))
    layer.alpha_composite(halo)
    layer.alpha_composite(marks)


def collision_overlay(public, folder, audit, visibility_fraction=.002):
    camera, rgb, depth, sources = camera_and_sources(public, audit['scene'])
    w, h = rgb.size
    layer = Image.new('RGBA', (w, h))
    display = []
    source_rows = []
    for obj in audit['objects']:
        data, path = load_intersections(folder, obj['id'])
        c, positions = data['classification'], data['positions']
        diameter = float(obj.get('diameter', 1.))
        uv, inside, visible = native_visibility(positions, camera, depth, diameter * visibility_fraction)
        crossing, contact = c == 'transverse', contact_mask(c)
        mark_points(layer, uv, inside & contact, YELLOW)
        mark_points(layer, uv, inside & crossing & ~visible, ORANGE)
        mark_points(layer, uv, inside & crossing & visible, RED)
        # Sample each exact intersection segment at fixed fractional positions
        # for a display-pixel line layer. Counts remain the detector's full counts.
        segments = data['segments']
        valid = np.isfinite(segments).all((1, 2)) & (crossing | contact)
        for start in range(0, len(segments), 65536):
            end = min(start + 65536, len(segments))
            keep = valid[start:end]
            if not keep.any():
                continue
            ss = segments[start:end][keep]
            cc = c[start:end][keep]
            for f in np.linspace(0, 1, 9):
                pp = ss[:, 0] * (1-f) + ss[:, 1] * f
                puv, pinside, pvisible = native_visibility(pp, camera, depth, diameter * visibility_fraction)
                cross, touch = cc == 'transverse', contact_mask(cc)
                mark_points(layer, puv, pinside & touch, YELLOW)
                mark_points(layer, puv, pinside & cross & ~pvisible, ORANGE)
                mark_points(layer, puv, pinside & cross & pvisible, RED)
        display.append(dict(id=str(obj['id']), classifications=classification_counts(c),
                            transverse_positions_in_canvas=int((crossing & inside).sum()),
                            approximately_visible_transverse_positions=int((crossing & visible).sum()),
                            visibility_tolerance_scene_units=diameter*visibility_fraction))
        source_rows.append(stamp(path))
    annotated = Image.alpha_composite(rgb.convert('RGBA'), layer).convert('RGB')
    canvas = Image.new('RGB', (2*w, h+143), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f'{audit["scene"]} | {camera["name"]} | original complete mesh RGB', font=font(17), fill=INK)
    draw.text((w+8, 5), 'Exact mesh intersection locations (annotation)', font=font(17), fill=INK)
    canvas.paste(rgb, (0, 35))
    canvas.paste(annotated, (w, 35))
    lines = [
        'Red: transverse intersection at visible depth; orange: hidden/uncertain visibility; yellow: contact/coplanar.',
        'Visibility uses cached mesh z-depth with a small display tolerance; it is not the collision classification.',
        'Every intersection contributes to native-pixel marks. Side views reveal intersections hidden by the environment.',
        f'Original panels: {w} x {h}; no RGB resize, relighting, mesh simplification or pose change.',
    ]
    for i, line in enumerate(lines):
        draw.text((8, h+44+i*22), line, font=font(14), fill=INK)
    save_image(canvas, folder / 'collision_overlay.png')
    return dict(camera=camera, native_panel_size=[w, h], sources=sources,
                intersection_sources=source_rows, objects=display,
                display_visibility_only=True, rgb_resized=False)


def side_basis(public, scene, objects, camera):
    graph = read(public / 'recomposition' / scene / 'composition/scene.json')
    up = None
    confidence = None
    for node in graph.get('objects', []):
        anchors = Path(node['evidence']) / 'anchors.npz'
        if anchors.exists():
            with np.load(anchors) as d:
                candidate = np.asarray(d['up'], dtype='float64')
            if candidate.shape == (3,) and np.isfinite(candidate).all() and np.linalg.norm(candidate) > 0:
                up = candidate / np.linalg.norm(candidate)
                info = read(Path(node['evidence']) / 'observations.json')
                confidence = info.get('up_confidence')
                break
    if up is None:
        up = -np.asarray(camera['w2c'])[1, :3]
        up /= np.linalg.norm(up)
    horizontal = np.asarray(camera['w2c'])[0, :3]
    horizontal = horizontal - up * (horizontal @ up)
    if np.linalg.norm(horizontal) < 1e-8:
        horizontal = np.cross(up, [1., 0., 0.])
    if np.linalg.norm(horizontal) < 1e-8:
        horizontal = np.cross(up, [0., 1., 0.])
    horizontal /= np.linalg.norm(horizontal)
    second = np.cross(up, horizontal)
    return up, horizontal, second, confidence


def occupancy(points, horizontal, up, origin, bounds, size=(1250, 1000)):
    """Bin all actual vertices/positions, with no random or stride sampling."""
    width, height = size
    result = np.zeros((height, width), dtype='uint32')
    xmin, xmax, ymin, ymax = bounds
    for start in range(0, len(points), 262144):
        pp = np.asarray(points[start:start+262144], dtype='float64') - origin
        xx, yy = pp @ horizontal, pp @ up
        keep = np.isfinite(xx) & np.isfinite(yy) & (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
        ix = np.minimum(((xx[keep]-xmin)/max(xmax-xmin, 1e-12)*width).astype('int64'), width-1)
        iy = np.minimum(((yy[keep]-ymin)/max(ymax-ymin, 1e-12)*height).astype('int64'), height-1)
        result += np.bincount(iy*width+ix, minlength=height*width).reshape(height, width).astype('uint32')
    return result


def count_overlay(count, rgb, base_alpha, logarithmic=False):
    out = np.zeros((*count.shape, 4), dtype='float32')
    out[..., :3] = np.asarray(rgb)/255
    if logarithmic and np.any(count):
        alpha = np.log1p(count) / max(float(np.log1p(count).max()), 1e-12)
        out[..., 3] = np.where(count > 0, .2 + base_alpha*alpha, 0)
    else:
        out[..., 3] = (count > 0)*base_alpha
    return out


def side_views(public, folder, audit, camera):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    roi = folder / 'background_roi.npz'
    with np.load(roi, allow_pickle=False) as d:
        background = d['vertices']
    objects = []
    crossings, contacts = [], []
    sources = [stamp(roi)]
    for obj in audit['objects']:
        mesh_path = folder / 'objects' / str(obj['id']) / 'mesh_world.npz'
        with np.load(mesh_path, allow_pickle=False) as d:
            objects.append(d['vertices'])
        sources.append(stamp(mesh_path))
        values, _ = load_intersections(folder, obj['id'])
        finite = np.isfinite(values['positions']).all(1)
        crossings.append(values['positions'][finite & (values['classification'] == 'transverse')])
        contacts.append(values['positions'][finite & contact_mask(values['classification'])])
    object_vertices = np.concatenate(objects) if objects else np.empty((0, 3))
    if not len(object_vertices):
        raise ValueError('A composed collision plot requires object vertices')
    crossing = np.concatenate(crossings) if crossings else np.empty((0, 3))
    contact = np.concatenate(contacts) if contacts else np.empty((0, 3))
    up, h0, h1, confidence = side_basis(public, audit['scene'], objects, camera)
    origin = (object_vertices.min(0)+object_vertices.max(0))/2
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=160)
    for axis, horizontal, title in zip(axes, [h0, h1], ['Side projection A', 'Side projection B (orthogonal)']):
        ranges = []
        for pp in [background, object_vertices]:
            if not len(pp):
                continue
            qq = pp-origin
            ranges.append((float((qq@horizontal).min()), float((qq@horizontal).max()),
                           float((qq@up).min()), float((qq@up).max())))
        xmin = min(x[0] for x in ranges)
        xmax = max(x[1] for x in ranges)
        ymin = min(x[2] for x in ranges)
        ymax = max(x[3] for x in ranges)
        margin = .035*max(xmax-xmin, ymax-ymin, 1e-6)
        bounds = [xmin-margin, xmax+margin, ymin-margin, ymax+margin]
        for points, color, alpha, logarithmic in [
                (background, (110, 118, 126), .55, True),
                (object_vertices, (21, 138, 188), .75, True),
                (contact, YELLOW, .9, False), (crossing, RED, 1., False)]:
            count = occupancy(points, horizontal, up, origin, bounds)
            axis.imshow(count_overlay(count, color, alpha, logarithmic), extent=bounds,
                        origin='lower', interpolation='nearest', aspect='equal')
        axis.set_title(title)
        axis.set_xlabel('Horizontal position (scene units, centered)')
        axis.set_ylabel('Estimated support-up position (scene units, centered)')
        axis.set_xlim(bounds[:2]);axis.set_ylim(bounds[2:])
        axis.grid(alpha=.18)
    figure.suptitle(f'{audit["scene"]} | all supplied geometry vertices and exact intersection locations', fontsize=15)
    legend = [Patch(color=np.array(c)/255, label=t) for c, t in [
        ((110,118,126), 'Environment ROI vertices'), ((21,138,188), 'Object vertices'),
        (RED, 'Transverse intersections'), (YELLOW, 'Contacts / coplanar overlaps')]]
    figure.legend(handles=legend, loc='lower center', ncol=4, bbox_to_anchor=(.5, .052))
    figure.text(.5, .014, 'Diagnostic orthographic occupancy projections; all vertices binned, not a filled mesh render. '
                f'Support-up confidence: {confidence if confidence is not None else "unavailable"}.', ha='center', fontsize=9)
    figure.subplots_adjust(top=.89, bottom=.17, left=.06, right=.98, wspace=.22)
    target = folder / 'side_view.png'
    temp = target.with_name('.'+target.name+'.tmp')
    figure.savefig(temp, format='png', dpi=160)
    plt.close(figure);temp.replace(target)
    return dict(sources=sources, environment_vertices=len(background), object_vertices=len(object_vertices),
                transverse_positions=len(crossing), contact_positions=len(contact),
                all_vertices_binned=True, display_sampling='No vertex sampling; native data aggregated into 1250x1000 display bins',
                support_up=up.tolist(), support_up_confidence=confidence,
                horizontal_axes=[h0.tolist(), h1.tolist()], origin=origin.tolist())


def penetration_fields(obj):
    value = obj.get('penetration', obj.get('penetration_diagnostics', {}))
    return value if isinstance(value, dict) else {}


def object_row(obj):
    counts = obj.get('intersection_counts', {})
    if not isinstance(counts, dict):
        counts = {}
    depth = penetration_fields(obj)
    applicable = depth.get('applicable', obj.get('closed_object', False))
    maximum = depth.get('maximum_depth') if applicable else None
    return dict(id=str(obj['id']), status=obj.get('collision_status', 'unknown'),
                transverse=int(counts.get('transverse', 0)),
                contacts=sum(int(counts.get(k, 0)) for k in ['boundary_touch', 'coplanar_overlap', 'point_touch']),
                closed_object=obj.get('closed_object'), maximum_depth=maximum,
                significant_inside_vertices=depth.get('significant_inside_vertices') if applicable else None,
                penetration_applicable=bool(applicable), normalized_by_object_diagonal=depth.get('normalized_by_object_diagonal'))


def scene_report(folder, audit, visual):
    rows = [object_row(o) for o in audit['objects']]
    lines = [f'# {audit["scene"]}：组合 mesh 碰撞核查', '',
        '[原生全图与相交标记](collision_overlay.png) · [两个正交侧投影](side_view.png) · [完整检测数据](audit.json)', '',
        '| 物体 | 检测状态 | 横向穿越三角形对 | 接触/共面三角形对 | 闭合物体 | 背景在物体内部的显著顶点数 | 最大背景顶点内部距离（场景单位） |',
        '|---|---|---:|---:|---|---:|---:|']
    for row in rows:
        depth = 'N/A' if row['maximum_depth'] is None else f'{row["maximum_depth"]:.8g}'
        lines.append(f'| {row["id"]} | {row["status"]} | {row["transverse"]} | {row["contacts"]} | '
                     f'{row["closed_object"]} | {row["significant_inside_vertices"] if row["significant_inside_vertices"] is not None else "N/A"} | {depth} |')
    lines += ['',
        '红色表示 transverse 三角形相交位置；橙色表示同类相交在当前视角被遮挡或可见性不确定；黄色表示边界接触、点接触或共面重叠。接触/共面不能直接解释为严重穿透。', '',
        '全图底图直接使用现有真实组合 mesh 渲染，左右子图均保留相同相机的原生像素尺寸。可见性只用于标注颜色，不影响碰撞判定。交点密集时按显示像素合并；侧图使用全部实际顶点/交点的正交占据投影，不是删面后的新模型，也不是完整填充表面渲染。', '',
        '内部距离仅在闭合物体上有适用性；它统计背景顶点到物体表面的内部距离，并非连续三角面上的精确全局穿透深度。环境 mesh 可以开放，不能把全场景假设为闭合体。所有长度为场景坐标单位，不能直接称厘米或米。非闭合物体的 N/A 不表示无穿透。', '',
        f'原发布状态：{"complete" if audit.get("publication_complete") else "未通过全部位姿检查/候选结果"}。本工具未修改 mesh、位姿、正式指标或主对比图。']
    (folder / 'README.md').write_text('\n'.join(lines)+'\n')
    return rows


def draw_scene(public, output, scene, visibility_fraction):
    folder = audit_folder(output, scene)
    audit = read(folder / 'audit.json')
    if not audit.get('complete'):
        raise ValueError('Collision detection is not complete: '+scene)
    overlay = collision_overlay(public, folder, audit, visibility_fraction)
    side = side_views(public, folder, audit, overlay['camera'])
    rows = scene_report(folder, audit, dict(overlay=overlay, side=side))
    result = dict(complete=True, scene=scene, updated=time.time(), detector=stamp(folder/'audit.json'),
                  implementation=stamp(Path(__file__)), objects=rows, overlay=overlay, side=side,
                  original_assets_modified=False)
    write(folder / 'visualization.json', result)
    print('COLLISION_VISUALIZED', scene, flush=True)
    return result


def aggregate(output):
    completed = []
    for scene in SCENES:
        folder = audit_folder(output, scene)
        marker = folder / 'visualization.json'
        if marker.exists():
            visual = read(marker)
            if visual.get('complete') and visual.get('detector') == stamp(folder/'audit.json'):
                completed.append((scene, folder, visual))
    if not completed:
        return
    width, panel_h = 1200, 615
    rows = (len(SCENES)+1)//2
    overview = Image.new('RGB', (width*2, rows*panel_h+65), BG)
    draw = ImageDraw.Draw(overview)
    draw.text((12, 8), 'All scenes | exact composed-mesh collision audit | original and annotated RGB', font=font(27), fill=INK)
    draw.text((12, 42), 'Overview thumbnails only; full native-size figures and diagnostic side views are in each scene folder.', font=font(17), fill=INK)
    by_scene = {s: (f, v) for s, f, v in completed}
    inventory = []
    for i, scene in enumerate(SCENES):
        x, y = (i%2)*width, 65+(i//2)*panel_h
        draw.text((x+10, y+5), scene, font=font(23), fill=INK)
        if scene not in by_scene:
            draw.text((x+30, y+90), 'Audit / visualization pending', font=font(23), fill=(120, 120, 120))
            continue
        folder, visual = by_scene[scene]
        with Image.open(folder/'collision_overlay.png') as image:
            thumb = ImageOps.contain(image.convert('RGB'), (width-16, panel_h-45), Image.Resampling.LANCZOS)
        overview.paste(thumb, (x+8, y+37))
        inventory.append(dict(scene=scene, folder=str(folder), objects=visual['objects']))
    save_image(overview, Path(output)/'overview.png')
    overview.save(Path(output)/'overview.jpg', quality=95, subsampling=0)
    lines = ['# 全场景组合 mesh 碰撞核查', '',
        '[全场景定位总图](overview.png)', '',
        '每个已完成场景使用现有 overview 的同一相机和真实组合 mesh RGB。红/橙为横向相交，黄为接触/共面。相交数量是三角形对数量，不能当作穿透体积或独立碰撞个数。', '',
        '| 场景 | 物体 | 状态 | 横向相交对 | 接触/共面对 | 最大背景顶点内部距离（场景单位） | 图与数据 |',
        '|---|---|---|---:|---:|---:|---|']
    for scene in SCENES:
        if scene not in by_scene:
            lines.append(f'| {scene} | — | 待完成 | — | — | — | — |')
            continue
        folder, visual = by_scene[scene]
        relative = folder.relative_to(output)
        for r in visual['objects']:
            maximum = 'N/A' if r['maximum_depth'] is None else f'{r["maximum_depth"]:.8g}'
            lines.append(f'| {scene} | {r["id"]} | {r["status"]} | {r["transverse"]} | {r["contacts"]} | {maximum} | [详情]({relative}/README.md) |')
    lines += ['',
        '开放物体的内部距离不适用，N/A 不代表没有碰撞。三角形相交需结合接触类型、内部距离及侧视图判断；未通过原位姿检查的候选模型不能表述为已经验证的组合结果。', '',
        '显示图允许把全部顶点/交点聚合到图像像素；检测是否完整、容差和输入来源以各场景 audit.json 为准。本可视化不更改模型、相机、位姿和正式评估结果。']
    (Path(output)/'README.md').write_text('\n'.join(lines)+'\n')
    write(Path(output)/'visualization_manifest.json', dict(complete=len(completed)==len(SCENES),
          completed_scenes=[s for s, _, _ in completed], expected_scenes=SCENES,
          updated=time.time(), scenes=inventory, original_assets_modified=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, default=PUBLIC)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    parser.add_argument('--partial', action='store_true',
                        help='Render completed scenes and list pending scenes; never claim an incomplete audit is complete')
    parser.add_argument('--visibility-tolerance-fraction', type=float, default=.002)
    args = parser.parse_args()
    output = args.output or args.public/'collision_audit'
    output.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        folder = audit_folder(output, scene)
        if not (folder/'audit.json').exists() or not read(folder/'audit.json').get('complete'):
            print('COLLISION_VISUALIZATION_PENDING', scene, flush=True)
            continue
        draw_scene(args.public, output, scene, args.visibility_tolerance_fraction)
    aggregate(output)


if __name__ == '__main__':
    main()
