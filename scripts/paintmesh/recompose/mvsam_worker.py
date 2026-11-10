#!/usr/bin/env python3
"""Run inside the dedicated, pinned official MV-SAM3D environment only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PIN='abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd'


def wait_before_model_allocation(output,maximum_used_mib=5120):
    """Imports and native input verification can outlast the parent's VRAM check."""
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')[0]
    if not physical:raise ValueError('An isolated physical GPU is required for inference')
    while True:
        r=subprocess.run(['nvidia-smi','-i',physical,'--query-gpu=memory.used',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        try:used=int(r.stdout.strip()) if r.returncode==0 else None
        except ValueError:used=None
        ready=used is not None and used<maximum_used_mib
        record=dict(stage='loading_model' if ready else 'waiting_before_model_load',
                    physical_gpu=physical,used_mib=used,maximum_used_mib=maximum_used_mib,updated_at=time.time())
        Path(output).write_text(json.dumps(record,indent=2))
        print('Pre-allocation GPU check:',record,flush=True)
        if ready:return
        time.sleep(10)


def validate_generation_inputs(evidence):
    """Reject resized RGB or a pointmap expressed on a different camera grid."""
    import numpy as np
    from PIL import Image
    evidence=Path(evidence);meta=json.loads((evidence/'observations.json').read_text())
    if meta.get('generation_input_policy')!='native_rgb_official_crop':
        raise ValueError('Regenerate evidence: native-resolution generation inputs are required')
    views=[v for v in meta['views'] if not v['validation']]
    with np.load(evidence/'pointmaps.npz') as data:
        names=[Path(str(name)).stem for name in data['image_files']]
        if names!=[Path(v['name']).stem for v in views]:
            raise ValueError('Pointmaps must match every generation view in the declared order')
        maps=data['pointmaps_sam3d'];Ks=data['intrinsics'];Ts=data['extrinsics']
        if len(maps)!=len(views):raise ValueError('Incomplete calibrated pointmaps')
        records=[]
        for i,v in enumerate(views):
            g=v['generation'];w,h=g['width'],g['height'];K=np.asarray(g['K'])
            with Image.open(v['image']['path']) as original, Image.open(evidence/'images'/v['name']) as prepared, Image.open(evidence/'object'/v['name']) as rgba:
                if original.size!=(w,h) or prepared.size!=(w,h) or rgba.size!=(w,h):
                    raise ValueError(f'Native source/input dimensions differ: {v["name"]}')
                if not np.array_equal(np.asarray(original.convert('RGB')),np.asarray(prepared.convert('RGB'))):
                    raise ValueError(f'Generation RGB differs from decoded original: {v["name"]}')
                if not np.array_equal(np.asarray(prepared.convert('RGB')),np.asarray(rgba.convert('RGB'))):
                    raise ValueError(f'RGB and masked RGB differ: {v["name"]}')
            if maps[i].shape!=(3,h,w):raise ValueError('Native pointmap/RGB dimensions differ')
            if not np.allclose(K,Ks[i]) or not np.allclose(Ts[i],np.asarray(v['w2c'])[:3]):
                raise ValueError('Pointmap camera calibration differs from the declared input')
            points=maps[i].reshape(3,-1);indices=np.flatnonzero(np.isfinite(points).all(0)&(points[2]>0))
            if not len(indices):raise ValueError('Native pointmap has no valid positive camera depths')
            indices=indices[::max(1,len(indices)//1024)]
            pixels=K@points[:,indices];uv=(pixels[:2]/pixels[2]).T
            error=float(np.abs(uv-np.c_[indices%w,indices//w]).max())
            if error>0.01:raise ValueError(f'Native pointmap ray mismatch: {error} pixels')
            records.append(dict(view=v['name'],source=v['original'],width=w,height=h,
                                rgb_matches_original=True,max_reprojection_error_px=error))
    return meta,views,records


def external_pointmap_inference(official,preprocessing_receipt=None):
    """Use calibrated pointmaps without loading an unused monocular depth model.

    DINO weights are already restored strictly from the SAM3D checkpoint. Avoid
    downloading a second initialization that would immediately be overwritten.
    These changes affect loading only; all used generation weights stay intact.
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from inference import check_hydra_safety, WHITELIST_FILTERS, BLACKLIST_FILTERS
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
    original_loader=InferencePipeline.instantiate_and_load_from_pretrained
    def load(self,config,*args,**kwargs):
        values=OmegaConf.to_container(config,resolve=True)
        def update(value):
            if isinstance(value,dict):
                if value.get('_target_')=='sam3d_objects.model.backbone.dit.embedder.dino.Dino':
                    value.setdefault('backbone_kwargs',{})['pretrained']=False
                for child in value.values():update(child)
            elif isinstance(value,list):
                for child in value:update(child)
        update(values)
        return original_loader(self,OmegaConf.create(values),*args,**kwargs)
    InferencePipeline.instantiate_and_load_from_pretrained=load
    class CalibratedInference(official.Inference):
        def __init__(self,config_file,compile=False):
            config=OmegaConf.load(config_file)
            config.rendering_engine='pytorch3d';config.compile_model=compile
            config.workspace_dir=os.path.dirname(config_file);config.depth_model=None
            check_hydra_safety(config,WHITELIST_FILTERS,BLACKLIST_FILTERS)
            if preprocessing_receipt is not None:
                wait_before_model_allocation(Path(preprocessing_receipt).with_name('runtime.json'),
                    int(os.environ.get('PAINTMESH_RECOMPOSE_GPU_MAX_USED_MIB','5120')))
            self._pipeline=instantiate(config)
            if preprocessing_receipt is not None:
                original_preprocess=self._pipeline.preprocess_image
                records=[]
                def preprocess(image,preprocessor,pointmap=None):
                    import numpy as np
                    height,width=np.asarray(image).shape[:2]
                    item=original_preprocess(image,preprocessor,pointmap=pointmap)
                    if tuple(item['image'].shape[-2:])!=(518,518):
                        raise ValueError('Official object crop did not produce the required 518x518 input')
                    record=dict(stage='ss' if preprocessor is self._pipeline.ss_preprocessor else 'slat',
                                source_width=width,source_height=height,
                                object_tensor_shape=list(item['image'].shape),
                                scene_tensor_shape=list(item['rgb_image'].shape),
                                pointmap_shape=list(pointmap.shape) if pointmap is not None else None)
                    records.append(record)
                    Path(preprocessing_receipt).write_text(json.dumps(dict(views=records),indent=2))
                    print('Official native preprocessing:',record,flush=True)
                    return item
                self._pipeline.preprocess_image=preprocess
    official.Inference=CalibratedInference


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--evidence',required=True)
    p.add_argument('--output',required=True);p.add_argument('--model-tag',default='hf');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--probe',action='store_true');p.add_argument('--probe-env',action='store_true',help='Check imports without loading or requiring model weights');args=p.parse_args()
    repo=Path(args.repo).resolve();evidence=Path(args.evidence).resolve();out=Path(args.output).resolve()
    sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if sha!=PIN:raise RuntimeError(f'MV-SAM3D revision mismatch: expected {PIN}, found {sha}')
    if subprocess.check_output(['git','-C',str(repo),'diff','--name-only','HEAD'],text=True).strip():
        raise RuntimeError('MV-SAM3D tracked code differs from the audited revision')
    config=repo/'checkpoints'/args.model_tag/'pipeline.yaml'
    if not config.exists() and not args.probe_env:raise FileNotFoundError(f'Missing official model config/weights: {config}')
    os.chdir(repo);sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'notebook'))
    import numpy as np
    import torch
    import trimesh
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from recompose.geometry import sam3d_pose_to_world
    import run_inference_weighted as official
    if args.probe or args.probe_env:
        print(json.dumps(dict(status='environment_ready' if args.probe_env else 'ready',revision=sha,cuda=torch.cuda.is_available(),config=str(config),model_config_present=config.is_file(),weights_loaded=False)));return
    meta,views,input_checks=validate_generation_inputs(evidence)
    out.mkdir(parents=True,exist_ok=True)
    external_pointmap_inference(official,out/'official_preprocessing.json')
    (out/'input_validation.json').write_text(json.dumps(dict(policy='native_rgb_official_crop',views=input_checks),indent=2))
    # The official directory name contains only input basename and a timestamp.
    # Parallel objects all have basename "evidence"; isolate outputs in this process.
    def private_output(**kwargs):
        directory=out/'official_output';directory.mkdir(parents=True,exist_ok=True);return directory
    official.get_output_dir=private_output
    result=official.run_weighted_inference(input_path=evidence,mask_prompt='object',image_names=[Path(v['name']).stem for v in views],
        seed=args.seed,stage1_steps=50,stage2_steps=25,decode_formats=['gaussian','mesh'],model_tag=args.model_tag,
        stage1_weighting=True,stage2_weighting=True,stage2_weight_source='mixed',
        da3_output_path=str(evidence/'pointmaps.npz'),run_pose_optimization=False)
    if not result or not result.get('glb_path'):raise RuntimeError('MV-SAM3D produced no mesh/pose')
    glb=Path(result['glb_path']);shutil.copy2(glb,out/'generated.glb')
    scene=trimesh.load(str(glb),force='scene',process=False)
    meshes=[];texture_arrays={};triangle_uvs=[];triangle_materials=[];texture_id=0
    for node in scene.graph.nodes_geometry:
        matrix,name=scene.graph[node];m=scene.geometry[name].copy();m.apply_transform(matrix)
        try:colors=m.visual.to_color().vertex_colors[:,:3]/255. if hasattr(m.visual,'to_color') else m.visual.vertex_colors[:,:3]/255.
        except (AttributeError,ValueError):colors=np.full((len(m.vertices),3),.5)
        face_material=np.full(len(m.faces),-1,dtype='int32');face_uv=np.zeros((len(m.faces),3,2))
        if getattr(m.visual,'kind',None)=='texture':
            face_uv=np.asarray(m.visual.uv)[m.faces]
            material=m.visual.material;materials=getattr(material,'materials',[material])
            assignments=getattr(m.visual,'face_materials',None)
            assignments=np.zeros(len(m.faces),dtype=int) if assignments is None else np.asarray(assignments)
            for mi,mat in enumerate(materials):
                image=getattr(mat,'baseColorTexture',None)
                if image is None:image=getattr(mat,'image',None)
                factor=getattr(mat,'baseColorFactor',None)
                factor=np.ones(4) if factor is None else np.asarray(factor,dtype=float)
                if factor.max()>1:factor/=255.
                if image is None:
                    # Preserve material colors even for meshes with no texture bitmap.
                    tex=np.full((1,1,3),255,dtype='uint8')
                else:tex=np.asarray(image.convert('RGB'))
                texture_arrays[f'texture_{texture_id}']=tex
                texture_arrays[f'texture_factor_{texture_id}']=factor[:3]
                face_material[assignments==mi]=texture_id;texture_id+=1
        triangle_uvs.append(face_uv);triangle_materials.append(face_material)
        meshes.append((np.asarray(m.vertices),np.asarray(m.faces),np.asarray(colors)))
    if not meshes:raise RuntimeError('Generated GLB has no triangle mesh')
    vertices=[];faces=[];colors=[];offset=0
    for v,f,c in meshes:vertices.append(v);faces.append(f+offset);colors.append(c);offset+=len(v)
    pose=result['pose'];shape,T=sam3d_pose_to_world(pose['scale'],pose['rotation'],pose['translation'],np.asarray(views[0]['w2c']))
    # Bake only the model's shape anisotropy once. Subsequent placement stays a true Sim(3).
    vertices=np.concatenate(vertices)@shape.T
    np.savez_compressed(out/'asset.npz',vertices=vertices,faces=np.concatenate(faces),colors=np.concatenate(colors),
        triangle_uv=np.concatenate(triangle_uvs),triangle_material=np.concatenate(triangle_materials),
        baked_shape_transform=shape,**texture_arrays)
    (out/'initial_pose.json').write_text(json.dumps(dict(object_to_world=T.tolist(),baked_shape_transform=shape.tolist(),source='official GLB -> PyTorch3D row rotation -> CV -> existing COLMAP world'),indent=2))
    raw=Path(result['output_dir'])/'result.ply'
    if raw.exists():shutil.copy2(raw,out/'generated_native_gaussians.ply')
    # Native GS is preserved for inspection, not assumed to share the GLB coordinate convention.
    (out/'generation.json').write_text(json.dumps(dict(status='complete',complete=True,revision=sha,model_config=str(config),
        seed=args.seed,views=[v['original'] for v in views],pointmap='pre-removal PGSR mesh, native calibrated camera rays',
        generation_input_policy='native_rgb_official_crop',input_checks=input_checks,
        parameters=dict(stage1_steps=50,stage2_steps=25,stage1_weighting=True,stage2_weighting=True,
                        stage2_weight_source='mixed',stage1_entropy_alpha=30.,stage2_entropy_alpha=30.,
                        stage2_visibility_alpha=30.,stage2_visibility_weight_ratio=.5),
        output_mesh=str(out/'generated.glb'),native_gaussians_used_for_composition=False,
        loading=dict(depth_model='disabled: calibrated pointmaps supplied for every view',
                     dino='strict checkpoint restore; no redundant pretrained initialization')),indent=2))


if __name__=='__main__':main()
