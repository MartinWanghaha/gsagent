#!/usr/bin/env python3
"""Validate and render synchronous movement of original GS and generated mesh."""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageDraw
from recompose.io import read_json,write_json
from recompose.scene import move_object
from recompose.render import render_composition
from recompose.pipeline import gpu_slot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',default='car');p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--distance',type=float,default=.05);p.add_argument('--wait-seconds',type=int,default=3600)
    p.add_argument('--gpu-max-used-mib',type=int,default=5120)
    args=p.parse_args();root=Path(__file__).resolve().parents[2]/'output/paintmesh/Inpaint360GS/recomposition'/args.scene
    source=root/'composition/scene.json';start=time.monotonic()
    while True:
        scene=read_json(source) if source.exists() else {}
        status=read_json(root/'status.json') if (root/'status.json').exists() else {}
        if scene.get('schema_version')==2 and scene.get('complete') and status.get('state')=='complete':break
        if time.monotonic()-start>args.wait_seconds:raise TimeoutError('Scene is not ready for validated movement')
        time.sleep(15)
    node=scene['objects'][0];evidence=Path(node['evidence']);meta=read_json(evidence/'observations.json')
    camera=evidence/'observations/00000.npz';o=np.load(camera);up=np.load(evidence/'anchors.npz')['up']
    direction=o['w2c'][0,:3].copy();direction-=direction@up*up;direction/=np.linalg.norm(direction)
    T=np.asarray(node['object_to_world']);T[:3,3]+=args.distance*meta['diameter']*direction
    out=root/'movement_demo';out.mkdir(exist_ok=True)
    write_json(out/'requested_pose.json',dict(object_to_world=T,translation_in_object_diameters=args.distance,direction=direction))
    move_object(source,node['id'],T,out/'moved.json')
    with gpu_slot(args.gpu,timeout=None,maximum_used_mib=args.gpu_max_used_mib):
        render_composition(source,camera,out/'before.png',f'cuda:{args.gpu}')
        render_composition(out/'moved.json',camera,out/'after.png',f'cuda:{args.gpu}')
    before=Image.open(out/'before.png').convert('RGB');after=Image.open(out/'after.png').convert('RGB')
    page=Image.new('RGB',(before.width*2,before.height+35),'white');draw=ImageDraw.Draw(page)
    draw.text((8,9),'Original deleted Gaussians + inpaint background',fill='black')
    draw.text((before.width+8,9),'Same rigid movement applied to original GS and generated mesh',fill='black')
    page.paste(before,(0,35));page.paste(after,(before.width,35));page.save(out/'movement_comparison.jpg',quality=94)
    print('Validated movement complete:',out,flush=True)


if __name__=='__main__':main()
