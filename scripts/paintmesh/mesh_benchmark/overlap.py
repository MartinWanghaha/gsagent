"""Coverage-only support selection; never select by reconstruction error.

    Geometry uses COLMAP's actual feature-track observations to restrict both
    reference and prediction to known visible surfaces. At each chosen reference
    observation, mesh depth is sampled at that image pixel. Each evaluated point
    therefore has one counterpart per method; unsupported/unobserved regions are
    excluded without introducing a nearest-distance acceptance test.
"""
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
from .core import (REPO,RECOMPOSE,STATES,read,write,identity,digest,cameras,
                   points_from_depth,surface_metrics)

VERSION=2

def select_anchors(ref_ids,cam_ids,pixels,valid_masks,eligible,seed,max_points=100000):
    """Choose one real observation per reference point, independent of mesh error."""
    keep=eligible[ref_ids].copy()
    for i,mask in enumerate(valid_masks):
        ii=np.flatnonzero(cam_ids==i)
        keep[ii]&=mask.reshape(-1)[pixels[ii]]
    candidates=np.flatnonzero(keep)
    if not len(candidates):return candidates
    rng=np.random.default_rng(seed);candidates=rng.permutation(candidates)
    _,first=np.unique(ref_ids[candidates],return_index=True)
    chosen=candidates[first]
    if len(chosen)>max_points:chosen=chosen[rng.choice(len(chosen),max_points,replace=False)]
    return chosen

def observations(scene,cs,reference):
    """Actual feature-track observations, excluding removed-state test* cameras."""
    selection=read(RECOMPOSE/scene/'composition/scene.json')['input_selection']
    file=REPO/'submodules/Inpaint360GS/tools/read_write_model.py'
    spec=importlib.util.spec_from_file_location('overlap_colmap',file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    intr,images,points=mod.read_model(str(Path(selection['source'])/'sparse/0'),ext='.bin')
    views={c['name']:(i,c) for i,c in enumerate(cs)}
    lookup={p.tobytes():i for i,p in enumerate(reference.astype('float32'))}
    result=[]
    for point in points.values():
        index=lookup.get(np.asarray(point.xyz,dtype='float32').tobytes())
        if index is None:continue
        for imid,idx in zip(point.image_ids,point.point2D_idxs):
            im=images[int(imid)]
            if im.name not in views:continue
            i,c=views[im.name];ic=intr[im.camera_id];uv=im.xys[int(idx)]
            u=int(np.floor(uv[0]*c['width']/ic.width));v=int(np.floor(uv[1]*c['height']/ic.height))
            if 0<=u<c['width'] and 0<=v<c['height']:result.append((index,i,v*c['width']+u))
    if not result:return np.empty((0,3),dtype='int64')
    return np.asarray(result,dtype='int64')

def prediction_points(cs,depths,selected,obs):
    result=np.empty((len(selected),3),dtype='float32')
    camera_ids=obs[selected,1];pixels=obs[selected,2]
    for i,c in enumerate(cs):
        keep=np.flatnonzero(camera_ids==i)
        if not len(keep):continue
        pp=pixels[keep];d=depths[i].reshape(-1)[pp]
        if not np.all(np.isfinite(d)&(d>0)):raise ValueError('Geometry sample outside positive depth support')
        result[keep]=points_from_depth(c,d,pp)
    return result

def prepare_support(output,scene,methods,config):
    root=Path(output);common=root/'common'/scene
    info=read(common/'complete.json');cs=[c for c in cameras(scene) if c['part']=='held_out']
    with np.load(common/'reference_points.npz') as data:
        reference=data['recomposition'];bg=data['background']
    bg_keys={p.tobytes() for p in bg.astype('float32')}
    bg_eligible=np.array([p.tobytes() in bg_keys for p in reference],dtype=bool)
    obs=observations(scene,cs,reference)
    depths={m:{} for m in methods}
    for method in methods:
        for state in STATES:
            depths[method][state]=[np.load(root/'cache'/method/scene/state/'depth'/(c['stem']+'.npy'),mmap_mode='r') for c in cs]
    for state in STATES:
        out=common/(state+'_support');out.mkdir(parents=True,exist_ok=True)
        request=dict(version=VERSION,methods=methods,
                     inputs=[dict(method=m,request=read(root/'cache'/m/scene/state/'complete.json')['request']) for m in methods],
                     threshold_fractions=config['f1_threshold_fractions'],primary=.01,
                     geometry_support='actual COLMAP held-out feature observations with shared positive depth; no distance-based filtering',
                     image_support='intersection of positive finite mesh depths across cohort, excluding target mask for background')
        request_id=digest(request)
        if (out/'complete.json').exists():
            old=read(out/'complete.json')
            if old['request_id']!=request_id:raise ValueError(f'Shared support changed: {out}; overwrite evaluation first')
            continue
        own={m:[] for m in methods};shared=[];coverage=[]
        for i,c in enumerate(cs):
            fg=np.asarray(Image.open(common/'held_out'/(c['stem']+'_foreground.png')))>127
            eligible=~fg if state=='background' else np.ones(fg.shape,dtype=bool)
            all_valid=eligible.copy()
            for method in methods:
                d=depths[method][state][i];valid=np.isfinite(d)&(d>0)&eligible
                own[method].append(valid);all_valid&=valid
            shared.append(all_valid)
            Image.fromarray(all_valid.astype('uint8')*255).save(out/(c['stem']+'.png'))
            coverage.append(dict(name=c['name'],eligible_pixels=int(eligible.sum()),common_pixels=int(all_valid.sum()),
                    common_fraction=float(all_valid.sum()/max(eligible.sum(),1)),
                    own_pixels={m:int(own[m][-1].sum()) for m in methods},
                    own_fraction={m:float(own[m][-1].sum()/max(eligible.sum(),1)) for m in methods}))
        eligible=bg_eligible if state=='background' else np.ones(len(reference),dtype=bool)
        chosen=select_anchors(obs[:,0],obs[:,1],obs[:,2],shared,eligible,config['seed']) if len(obs) else np.array([],dtype=int)
        gt=reference[obs[chosen,0]]
        np.savez(out/'geometry_reference.npz',points=gt,reference_indices=obs[chosen,0],camera_indices=obs[chosen,1],pixels=obs[chosen,2])
        for method in methods:
            selected_own=select_anchors(obs[:,0],obs[:,1],obs[:,2],own[method],eligible,config['seed']) if len(obs) else np.array([],dtype=int)
            own_ref=reference[obs[selected_own,0]]
            if len(chosen):
                pred=prediction_points(cs,depths[method][state],chosen,obs)
                geometry=surface_metrics(pred,gt,info['extent'],config['f1_threshold_fractions'])
                np.save(root/'cache'/method/scene/state/'visible_surface_points.npy',pred)
            else:geometry=dict(CD_L1=None,CD_L1_normalized=None,F1=None,precision=None,recall=None,reference_points=0,surface_samples=0,F_scores={})
            if len(selected_own):
                own_points=prediction_points(cs,depths[method][state],selected_own,obs)
                own_geometry=surface_metrics(own_points,own_ref,info['extent'],config['f1_threshold_fractions'])
            else:own_geometry=None
            geometry.update(scoring_protocol_version=VERSION,support_id=request_id,cohort=methods,
                    scope='common covered surface only; actual reference feature-track observations',
                    reference_kind=info['reference']['kind'],reference_eligible_count=int(eligible.sum()),
                    reference_support_fraction=float(len(chosen)/max(eligible.sum(),1)),
                    own_surface=own_geometry,
                    warning='Conditional visible-surface scores against sparse SfM, not dense-GT completeness; holes are excluded by request.')
            dest=root/'cache'/method/scene/state
            write(dest/'geometry_metrics.json',geometry)
            # Keep the immutable depth-cache request separate from scoring version.
            meta=read(dest/'complete.json');meta['geometry']=geometry;write(dest/'complete.json',meta)
            print('SURFACE_GEOMETRY',method,scene,state,'points',len(chosen),'CD',geometry['CD_L1'],'F1',geometry['F1'],flush=True)
        write(out/'complete.json',dict(complete=True,scoring_protocol_version=VERSION,request_id=request_id,
                request=request,methods=methods,coverage=coverage,geometry_points=len(chosen),
                reference_eligible_count=int(eligible.sum())))
