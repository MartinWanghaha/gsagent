#!/usr/bin/env python3
"""Audit and correct GOF native mesh RGB quantization; never alter geometry."""
import argparse
import ast
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mesh_benchmark.core import BASE, SCENES, read, write, identity, digest
from mesh_benchmark.color_export import read_mesh, positions, attach_colors
from recompose.io import fingerprint

RESEARCH = Path('/home/member/wyc/mesh_gs_research')
PUBLIC = BASE / 'mesh_comparison_whole'
WORK = PUBLIC / '_internal/workspace/gof_color_repair'


def quantize_rgb(values):
    """Saturate finite native linear RGB before unsigned byte conversion."""
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite native GOF RGB')
    # Retain upstream truncation within [0,1]; change only range handling.
    return (np.clip(values, 0., 1.) * 255).astype(np.uint8)


def restore_overflow(values, original):
    """Correct only out-of-range channels, preserving every other stored byte."""
    quantize_rgb(values)  # Reject NaN/Inf before any publication.
    return np.where(values > 1, 255, np.where(values < 0, 0, original)).astype('uint8')


def load_native(scene):
    scripts = RESEARCH / 'scripts/inpaint360_baselines'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('gof_color_worker_setup', scripts / 'worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.setup('gof', 'mesh')
    import torch
    torch.cuda.set_per_process_memory_fraction(10 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    from arguments import ModelParams, PipelineParams
    from scene import Scene, GaussianModel
    from utils import camera_utils
    from scene.cameras import Camera
    import extract_mesh as official
    official.tqdm = lambda iterable, **kwargs: iterable
    out = BASE / 'gof' / scene
    model = out / 'model'
    split = read(BASE / '_data' / scene / 'split.json')
    parser = argparse.ArgumentParser()
    mp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(['-s', str(BASE / '_data' / scene), '-m', str(model), '-r', '4', '--eval', '--data_device', 'cpu'])
    saved = ast.parse((model / 'cfg_args').read_text(), mode='eval').body
    for kw in saved.keywords:
        if hasattr(args, kw.arg):
            setattr(args, kw.arg, ast.literal_eval(kw.value))
    sizes = {Path(r['name']).stem: (r['width'], r['height']) for r in split['images']}
    original = camera_utils.loadCam
    checks = []

    def load_camera(arguments, index, info, resolution_scale):
        if resolution_scale != 1.0 or arguments.resolution != 4:
            raise ValueError('Unexpected raster resolution')
        width, height = sizes[info.image_name]
        camera = Camera(colmap_id=info.uid, R=info.R, T=info.T, FoVx=info.FovX, FoVy=info.FovY,
                        image=torch.zeros((3, height, width)), gt_alpha_mask=None,
                        image_name=info.image_name, uid=index, data_device=arguments.data_device)
        if not checks:
            reference = original(arguments, index, info, resolution_scale)
            for name in ['world_view_transform', 'full_proj_transform', 'camera_center']:
                if not torch.equal(getattr(camera, name), getattr(reference, name)):
                    raise ValueError('Camera geometry differs: ' + name)
            assert (width, height) == (reference.image_width, reference.image_height)
            checks.append(dict(view=info.image_name, native_camera_matrices_and_dimensions_exact=True))
        return camera

    camera_utils.loadCam = load_camera
    with torch.no_grad():
        gaussian = GaussianModel(args.sh_degree)
        native_scene = Scene(mp.extract(args), gaussian, load_iteration=30000, shuffle=False)
        cameras = native_scene.getTrainCameras()
        assert sorted(c.image_name for c in cameras) == sorted(Path(n).stem for n in split['train'])
        assert not any(c.image_name.lower().startswith('test') for c in cameras)
        # extract_mesh.py uses filter_3D stored in the checkpoint. Recomputing
        # it here would change the color field and confound the quantization fix.
    pipe = pp.extract(args)
    background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
    info = dict(official_source=fingerprint(RESEARCH / 'baselines/GOF/extract_mesh.py'),
                checkpoint=identity(model / 'point_cloud/iteration_30000/point_cloud.ply'),
                split=identity(BASE / '_data' / scene / 'split.json'),
                kernel_size=args.kernel_size, train_views=len(cameras), camera_checks=checks,
                uses_reference_rgb=False, gaussians_unchanged=True, training_performed=False,
                filter_3D='Unchanged checkpoint values, exactly as native extract_mesh.py')

    def query(xyz):
        with torch.no_grad():
            points = torch.from_numpy(np.asarray(xyz, dtype='float32')).cuda()
            _, color = official.evaluage_alpha(points, cameras, gaussian, pipe, background,
                                              args.kernel_size, return_color=True)
            return color.cpu().numpy()

    return query, info


def audit(values, old):
    corrected = restore_overflow(values, old)
    wrapped = (values * 255).astype(np.uint8)
    overflow = np.any((values < 0) | (values > 1), axis=1)
    return dict(vertices=len(values), min_rgb=values.min(0).tolist(), max_rgb=values.max(0).tolist(),
                overflow_vertices=int(overflow.sum()), overflow_fraction=float(overflow.mean()),
                old_matches_unclipped_cast_fraction=float(np.mean(np.all(old == wrapped, axis=1))),
                old_matches_clipped_cast_fraction=float(np.mean(np.all(old == corrected, axis=1))),
                mean_byte_error_unclipped=float(np.mean(np.abs(old.astype('float32') - wrapped))),
                mean_byte_error_clipped=float(np.mean(np.abs(old.astype('float32') - corrected))),
                changed_vertices=int(np.any(old != corrected, axis=1).sum()))


def run(scene, probe, chunk):
    folder = WORK / scene
    folder.mkdir(parents=True, exist_ok=True)
    lock = open(folder / 'worker.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = BASE / 'gof' / scene
    marker = read(out / 'mesh_complete.json')
    source = Path(marker.get('geometry_path', marker['path']))
    mesh = read_mesh(source)
    vertices = mesh['vertex'].data
    request = dict(source=identity(source), checkpoint=identity(out / 'model/point_cloud/iteration_30000/point_cloud.ply'),
                   policy='Saturate native float channels outside [0,1]; retain all in-range stored bytes', version=1)
    signature = digest(request)
    if not probe and (folder / 'complete.json').exists():
        old = read(folder / 'complete.json')
        if old['signature'] == signature and old['colored_mesh'] == fingerprint(old['colored_mesh']['path']):
            print('GOF_COLORS_ALREADY_COMPLETE', scene, flush=True)
            return old
    query, native = load_native(scene)
    if probe:
        ids = np.sort(np.random.default_rng(0).choice(len(vertices), min(probe, len(vertices)), replace=False))
        values = query(positions(vertices[ids]))
        old = np.column_stack([vertices[n][ids] for n in ['red', 'green', 'blue']])
        report = dict(scene=scene, source=identity(source), native=native, audit=audit(values, old), updated=time.time())
        # Queries are independent: verify the result is not a chunking artifact.
        halves = np.concatenate([query(p) for p in np.array_split(positions(vertices[ids]), 2)])
        report['maximum_float_chunk_difference'] = float(np.max(np.abs(values - halves)))
        write(folder / 'probe.json', report)
        np.savez(folder / 'probe.npz', vertex_ids=ids, xyz=positions(vertices[ids]), native_float_rgb=values,
                 original_rgb=old, clipped_rgb=restore_overflow(values, old))
        print('GOF_COLOR_PROBE', scene, report['audit'], 'chunk_error', report['maximum_float_chunk_difference'], flush=True)
        return report
    # Match the actual native export, including its saved filter_3D, before
    # changing colors. A mismatch means this is not an isolated byte-range fix.
    ids = np.sort(np.random.default_rng(0).choice(len(vertices), min(16384, len(vertices)), replace=False))
    sampled = query(positions(vertices[ids]))
    stored = np.column_stack([vertices[n][ids] for n in ['red', 'green', 'blue']])
    reproduction = audit(sampled, stored)
    if reproduction['mean_byte_error_unclipped'] > .05:
        raise ValueError('Native float RGB does not reproduce original export: ' + str(reproduction))
    progress = folder / 'progress.json'
    old = read(progress) if progress.exists() else None
    if old and old['signature'] != signature:
        raise ValueError('Native color repair inputs changed')
    start = old['completed_vertices'] if old else 0
    colors = np.lib.format.open_memmap(folder / 'vertex_colors.npy', mode='r+' if old else 'w+',
                                     dtype='uint8', shape=(len(vertices), 3))
    history = old.get('chunks', []) if old else []
    began = time.time()
    while start < len(vertices):
        stop = min(start + chunk, len(vertices))
        try:
            values = query(positions(vertices[start:stop]))
        except Exception as error:
            import torch
            if not isinstance(error, torch.cuda.OutOfMemoryError) or chunk <= 65536:
                raise
            chunk //= 2
            torch.cuda.empty_cache()
            print('GOF_COLOR_CHUNK_RETRY', scene, chunk, flush=True)
            continue
        previous = np.column_stack([vertices[n][start:stop] for n in ['red', 'green', 'blue']])
        data = audit(values, previous)
        if data['mean_byte_error_unclipped'] > .05:
            raise ValueError('Original RGB reproduction failed in chunk: ' + str(data))
        colors[start:stop] = restore_overflow(values, previous)
        colors.flush()
        data.update(start=start, stop=stop)
        history.append(data)
        write(progress, dict(signature=signature, request=request, completed_vertices=stop,
                             vertices=len(vertices), chunks=history, updated=time.time(), pid=os.getpid()))
        print('GOF_COLOR_CHUNK', scene, stop, '/', len(vertices), 'overflow', data['overflow_fraction'],
              'seconds', time.time() - began, flush=True)
        start = stop
    destination = source.with_name(source.stem + '_rgb_clamped.ply')
    verification = attach_colors(source, destination, colors)
    from colorize_baseline_meshes import publish
    detail = dict(color_method=request['policy'], native=native, signature=signature,
                  original_float_reproduction=reproduction, source_native_colors_requantized=True,
                  in_range_color_bytes_exact=True,
                  overflow_vertices=sum(c['overflow_vertices'] for c in history),
                  changed_vertices=sum(c['changed_vertices'] for c in history),
                  elapsed=time.time() - began, **verification)
    # The helper preserves geometry_path, so all frozen CD/F1 caches still
    # identify the exact same vertex coordinates and integer triangles.
    result = publish('gof', scene, source, destination, detail)
    result.update(signature=signature, request=request)
    write(folder / 'complete.json', result)
    mesh_marker = read(out / 'mesh_complete.json')
    mesh_marker.update(color_range_correction=True, color_range_correction_receipt=str(folder / 'complete.json'))
    write(out / 'mesh_complete.json', mesh_marker)
    print('GOF_COLOR_REPAIR_COMPLETE', scene, 'changed', detail['changed_vertices'], flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scene', choices=SCENES, required=True)
    p.add_argument('--probe', type=int, default=8192)
    p.add_argument('--chunk', type=int, default=2097152)
    a = p.parse_args()
    run(a.scene, a.probe, a.chunk)
