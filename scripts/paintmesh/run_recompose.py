#!/usr/bin/env python3
"""PaintMesh movable object composition (does not alter seg/remove/inpaint outputs)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import sys
import subprocess
from recompose.io import read_json, write_json
from recompose.evidence import SELECTION, resolve_scene
from recompose.pipeline import generator_probe, run_scene
from recompose.scene import move_object, export_scene

REPO=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(Path(__file__).with_name('configs')/'recompose.json'))
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('audit')
    run=sub.add_parser('run');run.add_argument('--scenes',default='all');run.add_argument('--gpus',default='0,1')
    run.add_argument('--until',choices=['gaussians','prepare','generate','align','compose'],default='compose')
    run.add_argument('--single-worker',action='store_true',help=argparse.SUPPRESS)
    exp=sub.add_parser('export');exp.add_argument('--scene',required=True);exp.add_argument('--output',required=True)
    move=sub.add_parser('move');move.add_argument('--scene',required=True);move.add_argument('--object',required=True)
    move.add_argument('--pose',required=True,help='JSON with object_to_world matrix');move.add_argument('--output',required=True)
    render=sub.add_parser('render');render.add_argument('--scene',required=True);render.add_argument('--camera',required=True)
    render.add_argument('--output',required=True);render.add_argument('--gpu',type=int,default=0)
    args=p.parse_args();cfg=read_json(args.config)
    if args.command=='audit':
        report=dict(generator=generator_probe(cfg['generator']),scenes={})
        for scene in SELECTION:
            try:report['scenes'][scene]=resolve_scene(REPO/cfg['inputs_root'],scene)
            except Exception as e:report['scenes'][scene]=dict(error=str(e))
        dest=REPO/cfg['output_root']/'preflight.json';write_json(dest,report)
        for scene,v in report['scenes'].items():print(scene,v.get('group'),v.get('target_ids'),v.get('error','inputs resolved'))
        print('Generator:',report['generator']);print('Report:',dest)
        return 0 if report['generator']['ready'] and not any('error' in s for s in report['scenes'].values()) else 2
    if args.command=='export':export_scene(args.scene,args.output);return 0
    if args.command=='move':move_object(args.scene,args.object,read_json(args.pose)['object_to_world'],args.output);return 0
    if args.command=='render':
        from recompose.render import render_composition
        render_composition(args.scene,args.camera,args.output,f'cuda:{args.gpu}');return 0
    scenes=list(SELECTION) if args.scenes=='all' else args.scenes.split(',')
    if len(set(scenes))!=len(scenes) or any(s not in SELECTION for s in scenes):raise ValueError('Unknown or repeated scenes')
    gpus=[int(x) for x in args.gpus.split(',')]
    if len(set(gpus))!=len(gpus) or not gpus:raise ValueError('GPU slots must be unique')
    if args.single_worker:
        if len(scenes)!=1 or len(gpus)!=1:raise ValueError('An isolated worker owns one scene and GPU')
        result=run_scene(REPO,Path(args.config).resolve(),scenes[0],gpus[0],args.until)
        print(result,flush=True)
        return 0 if result['state'] in ['complete',f'{args.until}_complete'] else 2
    # Probe lazily only when a generator stage actually needs execution. A
    # cached asset can still be aligned/exported without importing SAM3D.
    tasks=queue.Queue();results=[]
    for s in scenes:tasks.put(s)
    def worker(gpu):
        output=[]
        while True:
            try:s=tasks.get_nowait()
            except queue.Empty:return output
            print(f'GPU {gpu}: {s} -> {args.until}',flush=True)
            try:
                # Isolate CUDA allocators, seeds and extension device state between GPUs.
                command=[sys.executable,str(Path(__file__).resolve()),'--config',str(Path(args.config).resolve()),
                         'run','--scenes',s,'--gpus',str(gpu),'--until',args.until,'--single-worker']
                process=subprocess.run(command)
                status=REPO/cfg['output_root']/s/'status.json'
                r=dict(read_json(status),scene=s) if status.exists() else dict(scene=s,state='failed',exit_code=process.returncode)
                if process.returncode and r['state'] not in ['needs_review','failed']:
                    r=dict(scene=s,state='failed',last_stage=r.get('stage'),exit_code=process.returncode)
            except Exception as e:r=dict(scene=s,state='failed',error=str(e))
            print(r,flush=True);output.append(r);tasks.task_done()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for batch in pool.map(worker,gpus):results.extend(batch)
    write_json(REPO/cfg['output_root']/'batch_status.json',results)
    return 0 if all(r['state'] in ['complete',f'{args.until}_complete'] for r in results) else 2


if __name__=='__main__':raise SystemExit(main())
