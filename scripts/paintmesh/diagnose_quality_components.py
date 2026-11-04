#!/usr/bin/env python3
"""Render Gaussian ownership subsets for diagnosis; never creates training targets."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def run(job, names):
    import torch
    sys.path.insert(0, str(REPO / 'submodules/EDGS'))
    from source.vendor import bootstrap_gaussian_splatting
    from source.renderers import build_renderer
    from render import load_training_config, _save_png
    bootstrap_gaussian_splatting()
    from scene import GaussianModel
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
    from scene.dataset_readers import readColmapCameras
    from utils.camera_utils import loadCam
    renderer = build_renderer({'backend': 'pgsr', 'clamp_rgb': True})
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    background = torch.zeros(3, device='cuda')
    summary = []
    for entry in read(job / 'request.json')['scenes']:
        name = entry['scene']
        if name not in names:
            continue
        root = Path(entry['root'])
        ev = root / 'inpainted_3dgs_5000/evaluation/inpaint360_r4'
        result = read(ev / 'results.json')
        ordered = sorted(result['per_view'], key=lambda k: result['per_view'][k]['PSNR'])
        frames = [ordered[0], ordered[len(ordered)//2]]
        source = Path(result['inputs']['source_path'])
        cfg = load_training_config(root / 'inpainted_3dgs_5000', source_path=source, resolution=4, images='images')
        assert not cfg.gs.dataset.white_background
        joint = read(root / 'manifests/edgs_joint_manifest.json')
        initial = read(joint['inputs']['initialization']['path'])
        ownership = np.load(initial['outputs']['editable_mask']['path'])
        selection = read(joint['inputs']['local_selection']['path'])
        trainable = np.load(selection['outputs']['trainable_mask']['path'])
        retained = np.load(root / 'edgs_joint/snapshots/iteration_5000/retained_rows.npy')
        ext = read_extrinsics_binary(source / 'sparse/0/images.bin')
        selected = {k:v for k,v in ext.items() if Path(v.name).stem in frames}
        infos = readColmapCameras(selected, read_intrinsics_binary(source / 'sparse/0/cameras.bin'), None, str(source/'images'), '', [])
        views = {Path(i.image_name).stem: loadCam(SimpleNamespace(resolution=4, data_device='cpu', train_test_exp=False), n, i, 1.0, False, True) for n,i in enumerate(infos)}
        out = job / 'component_diagnosis' / name
        out.mkdir(parents=True, exist_ok=True)
        record = dict(scene=name, frames=frames, selection='baseline worst and median full-image PSNR',
                      note='Subset renders use independent alpha compositing; they are not additive RGB layers.', components={})
        for stage, ply, editable, allowed in [('initial', initial['outputs']['ply']['path'], ownership, trainable),
                ('final', result['inputs']['model']['path'], ownership[retained], trainable[retained])]:
            model = GaussianModel(int(cfg.gs.sh_degree))
            model.load_ply(ply, False)
            assert len(model.get_xyz) == len(editable)
            with torch.no_grad():
                for group, mask in [('old', ~editable), ('new', editable), ('frozen_old', ~editable & ~allowed),
                                    ('trainable_old', ~editable & allowed), ('full', np.ones(len(editable), dtype=bool))]:
                    index = torch.as_tensor(mask, device='cuda')
                    subset = SimpleNamespace(active_sh_degree=model.active_sh_degree, max_sh_degree=model.max_sh_degree,
                        **{key:getattr(model,key)[index] for key in ('get_xyz','get_scaling','get_rotation','get_features','get_opacity')})
                    record['components'][stage+'_'+group] = int(mask.sum())
                    for frame in frames:
                        package = renderer.render(views[frame], subset, pipe, background)
                        _save_png(package['render'], out / f'{frame}_{stage}_{group}.png')
                        if stage=='final' and group=='full':
                            check=np.asarray(Image.open(out/f'{frame}_{stage}_{group}.png'), dtype=int)
                            reference=np.asarray(Image.open(ev/'renders'/f'{frame}.png'), dtype=int)
                            assert np.max(np.abs(check-reference)) <= 1, 'Full render differs from published evaluation'
                        del package
                    del subset
            del model
            torch.cuda.empty_cache()
        labels = ['GT', 'Initial old', 'Initial new', 'Final frozen old', 'Final trainable old', 'Final new', 'Final full']
        sheet = Image.new('RGB', (len(labels)*300, 2*2*244), 'white')
        draw = ImageDraw.Draw(sheet)
        for row,frame in enumerate(frames):
            mask=np.asarray(Image.open(ev/'masks'/f'{frame}.png'))>127
            yy,xx=np.where(mask)
            box=(max(0,int(xx.min())-25),max(0,int(yy.min())-25),min(mask.shape[1],int(xx.max())+26),min(mask.shape[0],int(yy.max())+26))
            files=[ev/'gt'/f'{frame}.png']+[out/f'{frame}_{stage}_{group}.png' for stage,group in [('initial','old'),('initial','new'),('final','frozen_old'),('final','trainable_old'),('final','new'),('final','full')]]
            for col,(label,path) in enumerate(zip(labels,files)):
                im=Image.open(path).convert('RGB')
                for zoom in (0,1):
                    tile=im.crop(box) if zoom else im.copy()
                    tile.thumbnail((296,212))
                    x,y=col*300,(row*2+zoom)*244
                    sheet.paste(tile,(x+(300-tile.width)//2,y+28))
                    draw.text((x+4,y+5), f'{label} | {frame}' if not zoom else label+' | ROI', fill='black')
        sheet.save(out/'ownership_comparison.jpg', quality=95)
        (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        summary.append(record)
        print(name, record['components'], out/'ownership_comparison.jpg', flush=True)
    (job/'component_diagnosis.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--scenes',nargs='+',default=['garden_toys','fruits'])
    a=p.parse_args();run(a.job,a.scenes)
