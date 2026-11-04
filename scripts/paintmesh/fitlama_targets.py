#!/usr/bin/env python3
"""Bind unmodified official LaMa RGB/depth outputs to PaintMesh contracts.

The normal extension retains PaintMesh's strict validation. A temporary copy
composites known pixels solely for reusing that validator; the training target
and its records are always the raw official prediction, never that copy.
"""
import argparse
import copy
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

from run_geometry_ablation import REPO, read, write, now
from edgs_inpaint_io import record, verify_record, verify_targets, identity

sys.path.insert(0, str(REPO/'submodules/Inpaint360GS'))
from tools.prepare_paintmesh_lama_data import validate_lama_outputs, _artifact
from tools.publish_inpainted_edgs_model import _validate_lama_chain


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).copy()


def bind(case):
    case = case.resolve(); run = case/'inpaint'
    inputs_path = run/'manifests/lama_input_manifest.json'
    inputs = read(inputs_path); spec = read(case/'spec.json')
    request = read(case.parent.parent/'request.json')
    official = request['official']
    for rec in official['model'].values(): verify_record(rec)
    for rec in official['files'].values(): verify_record(rec)
    receipts = {}
    for stage in ('official_color', 'official_depth', 'normal'):
        path = case/'receipts'/f'{stage}.json'
        receipt = read(path)
        for rec in receipt['artifacts']: verify_record(rec)
        receipts[stage] = record(path)
    stems = inputs['parameters']['frame_names']; stats = {}
    output = run/'lama/output'; source = run/'lama/input'
    # Fail on missing/extra frames; debug PNGs in depth/vis are not frame arrays.
    for name, suffix in (('color', '.png'), ('depth', '.npy')):
        if {p.stem for p in (output/name).glob('*'+suffix)} != set(stems):
            raise ValueError(f'Unexpected official {name} frame set')
    with tempfile.TemporaryDirectory(prefix='.validation_', dir=run/'lama') as tmp:
        tmp = Path(tmp)
        for name in ('color', 'depth'): (tmp/name).mkdir()
        (tmp/'normal').symlink_to(output/'normal', target_is_directory=True)
        raw_records = {}
        for stem in stems:
            for rec in inputs['frames'][stem]['outputs'].values(): verify_record(rec)
            a = rgb(source/'color'/f'{stem}.png'); b = rgb(output/'color'/f'{stem}.png')
            m = np.asarray(Image.open(source/'color'/f'{stem}_mask.png').convert('L')) > 0
            d = np.load(source/'depth'/f'{stem}.npy'); e = np.load(output/'depth'/f'{stem}.npy')
            if a.shape != b.shape or d.shape != m.shape or e.shape != m.shape:
                raise ValueError('Official output shape mismatch: '+stem)
            if not np.issubdtype(e.dtype, np.floating) or not np.isfinite(e).all() or e.min() < 0 or e.max() <= e.min():
                raise ValueError('Invalid official depth: '+stem)
            bc = b.copy(); bc[~m] = a[~m]
            ec = e.copy(); ec[~m] = d[~m]
            Image.fromarray(bc).save(tmp/'color'/f'{stem}.png')
            np.save(tmp/'depth'/f'{stem}.npy', ec)
            raw_records[stem] = dict(color=_artifact(output/'color'/f'{stem}.png'), depth=_artifact(output/'depth'/f'{stem}.npy'))
            stats[stem] = dict(mask_fraction=float(m.mean()), depth_range=[float(e.min()), float(e.max())],
                known_rgb_max_255=int(np.abs(b.astype(int)-a)[~m].max()) if (~m).any() else 0,
                known_depth_max=float(np.abs(e-d)[~m].max()) if (~m).any() else 0.)
        checked = validate_lama_outputs(source/'color', source/'depth', tmp/'color', tmp/'depth',
            Path(official['model_root']), inputs_path, tmp/'validation.json', frames=len(stems), recursive_guide=False)
        value = copy.deepcopy(checked)
        value['created_at'] = now()
        value['roots'] = {f'{k}_output': str(output/k) for k in ('color', 'depth', 'normal')}
        value['parameters']['outside_mask_policy'] = 'official_full_prediction_rgb_depth; normal_preserve_input_exactly'
        value['parameters']['official_lama'] = dict(model=official['model'], code=official['files'],
            python=official['python'], receipts=receipts, refine=True, recursive_guide=False,
            refiner=dict(n_iters=15, lr=.002, min_side=512, max_scales=3, px_budget=1800000, modulo=8),
            mask_dilation=10, mask_shape='square', min_area=50,
            normal_extension='unchanged PaintMesh predictor; official Inpaint360GS has no normal completion')
        for stem in stems:
            value['frames'][stem]['outputs'].update(raw_records[stem])
            value['frames'][stem]['completed_depth_range'] = stats[stem]['depth_range']
        value['artifact_id'] = identity(dict(kind=value['kind'], schema_version=value['schema_version'],
            input_artifact_id=value['input_artifact_id'], model={k:v['sha256'] for k,v in value['model'].items()},
            parameters=value['parameters'], outputs={s:{k:r['sha256'] for k,r in f['outputs'].items()} for s,f in value['frames'].items()}))
        lp = run/'manifests/lama_completion_manifest.json'
        if lp.exists() and read(lp)['artifact_id'] != value['artifact_id']:
            raise ValueError('Existing fitlama targets differ')
        candidate = tmp/'raw_official_completion.json'
        write(candidate, value)
        verify_targets(candidate, spec['camera']['path'], use_depth=True, use_normal=True)
        _validate_lama_chain(candidate, value)
        write(lp, value)
    verify_targets(lp, spec['camera']['path'], use_depth=True, use_normal=True)
    write(case/'official_lama_validation.json', dict(complete=True, frames=len(stems), per_frame=stats,
        raw_rgb_depth_used=True, raw_outputs_modified=False, checked_at=now()))
    print('Official LaMa outputs validated and bound:', lp, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--case', type=Path, required=True)
    bind(p.parse_args().case)
