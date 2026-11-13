#!/usr/bin/env python3
"""Render a CoMe color-only candidate in its native triangle mesh backend."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,STATES,cameras,read,write,digest
from mesh_benchmark.color_export import read_mesh,positions
from mesh_benchmark.layout import work_root
from recompose.io import fingerprint
from repair_come_mesh_colors import PUBLIC,WORK


def render(scene):
    import torch
    from render_native_colored_meshes import NVDMesh
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
    result=read(WORK/scene/'complete.json');source=Path(result['path'])
    request=dict(mesh=fingerprint(source),adapter=fingerprint(HERE/'render_come_color_candidate.py'),
                 native_adapter=fingerprint(HERE/'render_native_colored_meshes.py'))
    dest=WORK/'pilot'/scene;dest.mkdir(parents=True,exist_ok=True);marker=dest/'render_complete.json'
    if marker.exists() and read(marker)['request']==request:return
    mesh=read_mesh(source);v=mesh['vertex'].data
    renderer=NVDMesh('come',positions(v),np.asarray(mesh['face'].data['vertex_indices']),
                     np.column_stack([v[k] for k in ['red','green','blue']]))
    root=work_root(PUBLIC);old=root/'native_renderers/come'/scene
    (dest/'renders').mkdir(exist_ok=True)
    for c in [c for c in cameras(scene) if c['part']=='held_out']:
        rgb,depth=renderer.render(c)
        original=np.load(old/'depth'/(c['stem']+'.npy'))
        np.testing.assert_array_equal(depth,original)
        Image.fromarray(rgb).save(dest/'renders'/(c['stem']+'.png'))
        print('COME_NORMALIZED_MESH_RENDER',scene,c['stem'],flush=True)
    write(marker,dict(complete=True,request=request,depths_exact=True,gaussian_rgb_substituted=False))


def score(scene):
    dest=WORK/'pilot'/scene
    cached=dest/'metrics.json'
    if cached.exists() and read(cached).get('render')==read(dest/'render_complete.json'):
        print('COME_PILOT_SCORE_ALREADY_COMPLETE',scene,flush=True);return
    import torch,lpips
    from mesh_benchmark.score import image_metrics
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(3*1024**3/torch.cuda.get_device_properties(0).total_memory)
    criterion=lpips.LPIPS(net='vgg',version='0.1',spatial=True).to('cuda').eval()
    for p in criterion.parameters():p.requires_grad_(False)
    root=work_root(PUBLIC);states={}
    for state in STATES:
        before=read(root/state/'come'/scene/'metrics.json');rows=[]
        for c in [c for c in cameras(scene) if c['part']=='held_out']:
            name=c['stem']+'.png'
            old=np.array(Image.open(root/'native_renderers/come'/scene/'renders'/name).convert('RGB'))
            new=np.array(Image.open(dest/'renders'/name).convert('RGB'))
            target=np.array(Image.open(root/'common'/scene/'held_out'/(c['stem']+'_gt.png')).convert('RGB'))
            mask=np.array(Image.open(root/'common'/scene/(state+'_native_support')/name))>127
            a=image_metrics(old,target,mask,criterion,'cuda');b=image_metrics(new,target,mask,criterion,'cuda')
            rows.append(dict(name=c['name'],pixels=int(mask.sum()),before=a,after=b))
        mean=lambda k:{metric:float(np.mean([r[k][metric] for r in rows])) for metric in ['PSNR','SSIM','LPIPS']}
        states[state]=dict(before=mean('before'),after=mean('after'),per_view=rows,
                           geometry_metrics=before['geometry'],same_depths=True,same_mask=True)
    result=dict(complete=True,scene=scene,states=states,reference_rgb_used_to_repair=False,
                native_mesh_rendering=True,render=read(dest/'render_complete.json'))
    write(dest/'metrics.json',result);print('COME_NORMALIZATION_PILOT',scene,states['recomposition']['before'],'->',states['recomposition']['after'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES,required=True);p.add_argument('--score',action='store_true');a=p.parse_args()
    if a.score:score(a.scene)
    else:render(a.scene)
