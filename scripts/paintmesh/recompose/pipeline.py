"""Resumable composition stages with content-addressed receipts, no training edits."""
from __future__ import annotations
from pathlib import Path
import os
import hashlib
import json
import subprocess
import time
import fcntl
from contextlib import contextmanager
from .io import read_json, write_json, fingerprint
from .evidence import resolve_scene, prepare, SELECTION
from .optimize import optimize_pose
from .original_gaussians import extract,export_initial,sources
from .scene import publish
from .provenance import Fingerprinter

# Large source meshes otherwise get rescanned repeatedly during one worker's
# prepare, provenance checks, and receipts. Any identity/metadata change rehashes.
fingerprint=Fingerprinter()

# A long-running worker must fingerprint the implementation it imported, even
# if the source files are edited while an earlier GPU stage is still running.
_LOADED_CODE={p.name:p.read_bytes() for p in Path(__file__).parent.glob('*.py')}


def code_digest(stage_name=None):
    h=hashlib.sha256()
    # Generator/environment changes do not invalidate expensive, unchanged geometry.
    modules={
        'prepare':['evidence.py','geometry.py','io.py'],
        'generate':['mvsam_worker.py','geometry.py'],
        'align':['optimize.py','geometry.py','texture.py','io.py','collision.py'],
        'original_gaussians':['original_gaussians.py','gaussian.py','io.py'],
    }.get(stage_name)
    folder=Path(__file__).parent
    paths=[folder/name for name in modules] if modules else list(folder.glob('*.py'))
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes() if stage_name=='generate' else _LOADED_CODE.get(p.name,p.read_bytes()))
    return h.hexdigest()


def verify_observations(path):
    m=read_json(path)
    for key in ['original_mesh','background_mesh']:
        ref=m[key]
        if fingerprint(ref['path'])['sha256']!=ref['sha256']:raise ValueError(f'Geometry changed since prepare: {ref["path"]}')
    for v in m['views']:
        for key in ['image','mask','prepared_image','prepared_mask','observation']:
            ref=v[key]
            if fingerprint(ref['path'])['sha256']!=ref['sha256']:
                raise ValueError(f'Observation changed since prepare: {ref["path"]}; regenerate evidence before continuing')


def stage(root,name,inputs,outputs,fn):
    """Input bytes, config and implementation identify a stage; verify output bytes on resume."""
    receipt=Path(root)/'receipts'/f'{name}.json'
    signature=hashlib.sha256(json.dumps(dict(inputs=inputs,code=code_digest(name)),sort_keys=True,default=str).encode()).hexdigest()
    if receipt.exists():
        old=read_json(receipt)
        if old.get('signature')==signature and all(Path(x['path']).is_file() and fingerprint(x['path'])['sha256']==x['sha256'] for x in old.get('outputs',[])):
            return
    fn()
    write_json(receipt,dict(signature=signature,outputs=[fingerprint(p) for p in outputs],completed_at=time.time()))


def wait_gpu(gpu,minimum_mib=21000,timeout=0,on_wait=None,maximum_used_mib=None):
    start=time.monotonic();last_report=-60.
    while True:
        field='memory.used' if maximum_used_mib is not None else 'memory.free'
        r=subprocess.run(['nvidia-smi','-i',str(gpu),f'--query-gpu={field}','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if r.returncode==0:
            try:
                value=int(r.stdout.strip())
                if (value<maximum_used_mib if maximum_used_mib is not None else value>=minimum_mib):return
            except ValueError:pass
        elapsed=time.monotonic()-start
        if elapsed-last_report>=60:
            detail=dict(gpu=gpu,wait_seconds=int(elapsed))
            if maximum_used_mib is not None:
                detail.update(maximum_used_mib=maximum_used_mib,used_mib=r.stdout.strip() if r.returncode==0 else None)
            else:detail.update(minimum_mib=minimum_mib,free_mib=r.stdout.strip() if r.returncode==0 else None)
            print(f'Waiting for GPU memory: {detail}',flush=True)
            if on_wait:on_wait(detail)
            last_report=elapsed
        if timeout is not None and (timeout<=0 or elapsed>=timeout):
            condition=f'used memory must be below {maximum_used_mib} MiB' if maximum_used_mib is not None else f'free memory must be at least {minimum_mib} MiB'
            raise RuntimeError(f'GPU {gpu} unavailable: {condition}')
        time.sleep(10)


@contextmanager
def gpu_slot(gpu,timeout=3600,minimum_mib=21000,on_wait=None,maximum_used_mib=None):
    """Serialize our GPU stages across independent resume/queue processes."""
    directory=Path('/home/member/.cache/paintmesh-recompose-gpu');directory.mkdir(parents=True,exist_ok=True)
    with (directory/f'{gpu}.lock').open('a') as lock:
        print(f'Waiting for composition GPU slot {gpu}',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            wait_gpu(gpu,minimum_mib=minimum_mib,timeout=timeout,on_wait=on_wait,maximum_used_mib=maximum_used_mib)
            yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def generator_probe(cfg):
    repo=Path(cfg['repository']);python=Path(cfg['python']);errors=[]
    if not (repo/'run_inference_weighted.py').exists():errors.append(f'Missing MV-SAM3D checkout: {repo}')
    if not python.is_file():errors.append(f'Missing MV-SAM3D Python environment: {python}')
    if not (repo/'checkpoints'/cfg['model_tag']/'pipeline.yaml').exists():errors.append('Missing SAM3D pipeline.yaml/checkpoint files')
    if not errors:
        cmd=[str(python),str(Path(__file__).with_name('mvsam_worker.py')),'--repo',str(repo),'--evidence','.', '--output','.', '--model-tag',cfg['model_tag'],'--probe']
        r=subprocess.run(cmd,env=generator_environment(cfg),capture_output=True,text=True,timeout=600)
        if r.returncode:errors.append(r.stderr[-4000:])
    return dict(ready=not errors,errors=errors,required_revision=cfg['revision'])


def generator_environment(cfg,gpu=None):
    """Use the independent interpreter and its CUDA libraries when called from paintmesh."""
    prefix=Path(cfg['python']).parent.parent
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_HOME=str(prefix),CONDA_PREFIX=str(prefix),
               PATH=str(prefix/'bin')+os.pathsep+env.get('PATH',''),
               LD_LIBRARY_PATH=os.pathsep.join([str(prefix/'lib'),str(prefix/'targets/x86_64-linux/lib'),env.get('LD_LIBRARY_PATH','')]))
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    return env


def generation_with_retry(command,env,log_path,gpu,maximum_used_mib,on_start=None,on_wait=None,on_oom=None):
    """Release a failed CUDA process and requeue only explicit CUDA OOMs.

    Other users can allocate VRAM after a successful launch gate. A retry uses
    the same inputs, weights and seed; a failed attempt never gets a receipt.
    """
    log_path=Path(log_path);attempt=0
    while True:
        attempt+=1
        with gpu_slot(gpu,timeout=None,maximum_used_mib=maximum_used_mib,on_wait=on_wait):
            if on_start:on_start(attempt)
            with log_path.open('ab') as log:
                log.write(f'\n=== Generation attempt {attempt}, {time.strftime("%F %T")} ===\n'.encode());log.flush()
                offset=log.tell()
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode==0:return
        with log_path.open('rb') as log:
            log.seek(0,2);end=log.tell();log.seek(max(offset,end-65536));tail=log.read().decode(errors='replace')
        if not any(s in tail for s in ['CUDA out of memory','torch.cuda.OutOfMemoryError']):
            raise subprocess.CalledProcessError(result.returncode,command)
        detail=dict(gpu=gpu,attempt=attempt,reason='CUDA OOM; waiting for used VRAM below launch threshold')
        print(f'Retrying generation after VRAM contention: {detail}',flush=True)
        if on_oom:on_oom(detail)
        time.sleep(10)


def run_scene(repo,config,scene,gpu=0,until='compose'):
    repo=Path(repo);cfg=read_json(config);inputs=resolve_scene(repo/cfg['inputs_root'],scene)
    root=(repo/cfg['output_root']/scene).resolve();root.mkdir(parents=True,exist_ok=True)
    # Never let a configurable output path overwrite selected upstream results.
    for p in [Path(inputs['run']),Path(inputs['original_mesh']).parent]:
        if root==p or p in root.parents:raise ValueError('Composition output must be separate from upstream artifacts')
    lock=(root/'run.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError(f'Another process owns {root}')
    try:
        write_json(root/'selection.json',inputs);objects=[];pending=[]
        write_json(root/'status.json',dict(state='running',stage='extract_original_gaussians',gpu=gpu))
        original=root/'original_gaussians'
        refs=sources(inputs)
        stage(root,'original_gaussians',dict(sources={k:fingerprint(v) for k,v in refs.items()},targets=inputs['target_ids']),
              [original/'manifest.json']+[original/str(t)/name for t in inputs['target_ids'] for name in ['gaussians_world.ply','original_indices.npy']],
              lambda:extract(inputs,original))
        restored=read_json(original/'manifest.json')
        export_initial(inputs,restored,root/'composition/world')
        if until=='gaussians':
            result=dict(scene=scene,state='gaussians_complete',training=False,object_counts=restored['object_counts'])
            write_json(root/'status.json',result);return result
        if cfg.get('isolate_removed_targets',{}).get(scene,False):
            from .isolation import isolate_selection
            write_json(root/'status.json',dict(state='running',stage='target_isolation',gpu=gpu))
            inputs=isolate_selection(repo,inputs,root/'target_isolation',cfg['evidence']['observation_max_size'])
        write_json(root/'selection.json',inputs)
        for target in inputs['target_ids']:
            obj=root/'objects'/str(target);evidence=obj/'evidence';generated=obj/'generated';placement=obj/'placement'
            obj.mkdir(parents=True,exist_ok=True)
            write_json(root/'status.json',dict(state='running',stage='prepare',object=target,gpu=gpu))
            # Reused evidence is checked against both source pixels and prepared modalities.
            if (evidence/'observations.json').exists():
                try:verify_observations(evidence/'observations.json')
                except (ValueError,KeyError,FileNotFoundError):
                    receipt=obj/'receipts/prepare.json'
                    if receipt.exists():receipt.unlink()
            geometry_inputs={k:fingerprint(inputs[k]) for k in ['original_mesh','background_mesh','original_gaussians','gaussian_labels','gaussian_confidence']}
            stage(obj,'prepare',dict(selection=inputs,config=cfg['evidence'],manifest=fingerprint(inputs['manifest']),geometry=geometry_inputs),
                  [evidence/'observations.json',evidence/'anchors.npz',evidence/'pointmaps.npz',evidence/'original_proxy.ply',evidence/'background_proxy.ply'],
                  lambda:prepare(repo,inputs,evidence,cfg['evidence'],target))
            verify_observations(evidence/'observations.json')
            if until=='prepare':continue
            def generate():
                generated.mkdir(parents=True,exist_ok=True)
                command=[cfg['generator']['python'],str(Path(__file__).with_name('mvsam_worker.py')),
                         '--repo',cfg['generator']['repository'],'--evidence',str(evidence),'--output',str(generated),
                         '--model-tag',cfg['generator']['model_tag'],'--seed',str(cfg['generator']['seed'])]
                waiting=lambda detail:write_json(root/'status.json',dict(state='running',stage='waiting_gpu',object=target,**detail))
                env=generator_environment(cfg['generator'],gpu)
                env['PAINTMESH_RECOMPOSE_GPU_MAX_USED_MIB']=str(cfg['execution']['gpu_max_used_mib'])
                generation_with_retry(command,env,obj/'generation.log',gpu,cfg['execution']['gpu_max_used_mib'],
                    on_start=lambda attempt:write_json(root/'status.json',dict(state='running',stage='generate',object=target,gpu=gpu,attempt=attempt)),
                    on_wait=waiting,
                    on_oom=lambda detail:write_json(root/'status.json',dict(state='running',stage='retrying_gpu_oom',object=target,**detail)))
            write_json(root/'status.json',dict(state='running',stage='generate',object=target,gpu=gpu))
            stage(obj,'generate',dict(evidence=fingerprint(evidence/'observations.json'),pointmaps=fingerprint(evidence/'pointmaps.npz'),config=cfg['generator']),
                  [generated/'asset.npz',generated/'generated.glb',generated/'initial_pose.json',generated/'generation.json',generated/'input_validation.json'],generate)
            if until=='generate':continue
            write_json(root/'status.json',dict(state='running',stage='align',object=target,gpu=gpu))
            stage(obj,'align',dict(asset=fingerprint(generated/'asset.npz'),initial=fingerprint(generated/'initial_pose.json'),evidence=fingerprint(evidence/'observations.json'),config=cfg['pose']),
                  [placement/'pose.json',placement/'candidates.json'],
                  lambda:optimize_pose(evidence,generated/'asset.npz',placement,cfg['pose'],read_json(generated/'initial_pose.json')['object_to_world']))
            pose=read_json(placement/'pose.json')
            if pose['status']!='accepted':
                pending.append(dict(object=target,reasons=pose['reasons'],pose=str(placement/'pose.json')))
                # Finish independent objects even when one placement needs review.
                # Publication still requires every requested object to pass.
            if until=='align':continue
            objects.append(dict(id=target,asset=str(generated/'asset.npz'),
                                gaussians=restored['objects'][str(target)],
                                gaussian_provenance=str(original/'manifest.json'),
                                gaussian_indices=str(original/str(target)/'original_indices.npy'),
                                pose=str(placement/'pose.json'),evidence=str(evidence)))
        if pending:
            reasons=[f'ID {item["object"]}: {reason}' for item in pending for reason in item['reasons']]
            result=dict(scene=scene,state='needs_review',pending=pending,reasons=reasons)
            if until=='compose':
                publish(inputs,objects,root/'composition',review=True)
                result['candidate_scene']=str(root/'composition/scene.json')
            write_json(root/'status.json',result)
            return result
        if until=='compose':publish(inputs,objects,root/'composition')
        state='complete' if until=='compose' else f'{until}_complete'
        write_json(root/'status.json',dict(state=state,objects=inputs['target_ids'],finished_at=time.time()))
        return dict(scene=scene,state=state,root=str(root))
    except Exception as e:
        write_json(root/'status.json',dict(state='failed',error=str(e),updated_at=time.time()));raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
