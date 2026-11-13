"""One bounded-memory CPU process caches fixed mesh intersections for both reports."""
import gc
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .core import (BASE,RECOMPOSE,cameras,image_mask,rays,points_from_depth,
                   MeshScene,specs_for,identity,digest,read,write,reference_cloud,chamfer)

def common(output,scene):
    root=Path(output)/'common'/scene;root.mkdir(parents=True,exist_ok=True)
    document=read(RECOMPOSE/scene/'composition/scene.json');selection=document['input_selection']
    cs=cameras(scene)
    request=dict(split=identity(BASE/'_data'/scene/'split.json'),
                 selection=identity(RECOMPOSE/scene/'composition/scene.json'),
                 masks=[identity(Path(selection['mask_root'])/(c['stem']+'.png')) for c in cs])
    ident=digest(request);marker=root/'complete.json'
    if marker.exists():
        old=read(marker)
        if old['request_id']!=ident:raise ValueError(f'Upstream inputs changed: {root}; use a new evaluation output')
        return cs,document,old
    for c in cs:
        if c['part']!='held_out':continue
        image,mask=image_mask(c,selection);dest=root/'held_out';dest.mkdir(exist_ok=True)
        Image.fromarray(image).save(dest/(c['stem']+'_gt.png'))
        Image.fromarray(mask.astype('uint8')*255).save(dest/(c['stem']+'_foreground.png'))
    original,bg,box,provenance=reference_cloud(scene,selection,cs)
    np.savez(root/'reference_points.npz',recomposition=original,background=bg,box=box)
    centers=np.array([np.linalg.inv(c['w2c'])[:3,3] for c in cs if c['part']=='train'])
    center=centers.mean(0);extent=float(np.linalg.norm(centers-center,axis=1).max()*1.1)
    info=dict(complete=True,request_id=ident,request=request,reference=provenance,
              center=center.tolist(),extent=extent,train_count=sum(c['part']=='train' for c in cs),
              held_out_count=sum(c['part']=='held_out' for c in cs),
              cameras=[{k:v for k,v in c.items() if k!='image'} for c in cs])
    write(marker,info);return cs,document,info

def prepare(output,method,scene,config):
    specs=specs_for(scene,method,'background')
    if specs and len(specs)>4:
        from .stream_cache import prepare_streamed
        return prepare_streamed(output,method,scene,config)
    output=Path(output);cs,document,info=common(output,scene)
    selection=document['input_selection'];ref=np.load(output/'common'/scene/'reference_points.npz')
    base=output/'cache'/method/scene
    specs=specs_for(scene,method,'background')
    if specs is None:raise FileNotFoundError(f'Mesh is not complete: {method}/{scene}')
    requests={}
    for state in ['background','recomposition']:
        chosen=specs_for(scene,method,state)
        requests[state]=dict(specs=[dict(identity=identity(s['path']),transform=s.get('transform')) for s in chosen],
                             common=info['request_id'],seed=config['seed'],pool_pixels=config['pool_pixels'],
                             protocol_version=config['protocol_version'])
    if all((base/state/'complete.json').exists() and read(base/state/'complete.json')['request']==requests[state] for state in requests):
        print('REUSE_GEOMETRY',method,scene,flush=True);return
    for state in requests:
        p=base/state/'complete.json'
        if p.exists() and read(p)['request']!=requests[state]:raise ValueError(f'Stale geometry cache {p}')
    print('BUILD_BVH',method,scene,time.strftime('%F %T'),flush=True)
    mesh=MeshScene(specs)
    extra=[] if method!='paintmesh' else specs_for(scene,method,'recomposition')[len(specs):]
    obj=MeshScene(extra) if extra else None
    pools={s:{k:[] for k in ['positions','directions','rgb','valid']} for s in requests}
    coverage={s:[] for s in requests};geometries={}
    native=all(x['has_native_color'] for x in mesh.counts)
    for i,c in enumerate(cs):
        image,fg=image_mask(c,selection)
        rng=np.random.default_rng(config['seed']+i)
        pixels=rng.integers(c['width']*c['height'],size=config['pool_pixels']) if c['part']=='train' else None
        rr=rays(c,pixels);z,rgb=mesh.cast(rr,color=native and c['part']=='held_out')
        rz=z.copy();rrgb=None if rgb is None else rgb.copy()
        if obj is not None:
            oz,orgb=obj.cast(rr,color=c['part']=='held_out')
            front=(oz>0)&((rz==0)|(oz<rz));rz[front]=oz[front]
            if rrgb is not None:rrgb[front]=orgb[front]
        for state,depth,color in [('background',z,rgb),('recomposition',rz,rrgb)]:
            dest=base/state;dest.mkdir(parents=True,exist_ok=True)
            allowed=~fg.reshape(-1) if state=='background' else np.ones(fg.size,dtype=bool)
            if pixels is not None:
                allowed=allowed[pixels];valid=(depth>0)&allowed
                points=rr[:,:3]+rr[:,3:]*depth[:,None]
                direction=rr[:,3:]/np.linalg.norm(rr[:,3:],axis=1,keepdims=True)
                # Training on original RGB inside the erased object would paint a ghost.
                # Drop excluded pixels entirely; keep mesh misses as constant error.
                pools[state]['positions'].append(points[allowed].astype('float32'))
                pools[state]['directions'].append(direction[allowed].astype('float32'))
                pools[state]['rgb'].append(image.reshape(-1,3)[pixels][allowed])
                pools[state]['valid'].append((depth[allowed]>0))
            else:
                dd=depth.reshape(c['height'],c['width']);(dest/'depth').mkdir(exist_ok=True)
                np.save(dest/'depth'/(c['stem']+'.npy'),dd)
                if color is not None:
                    (dest/'native_renders').mkdir(exist_ok=True)
                    Image.fromarray(np.rint(np.clip(color,0,1).reshape(image.shape)*255).astype('uint8')).save(dest/'native_renders'/(c['stem']+'.png'))
                coverage[state].append(dict(name=c['name'],coverage=float(((depth>0)&allowed).sum()/allowed.sum()),pixels=int(allowed.sum())))
        if i%20==0:print('RAYCAST',method,scene,i+1,'/',len(cs),flush=True)
    for state in requests:
        dest=base/state;pool={k:np.concatenate(v) for k,v in pools[state].items()}
        pool.update(center=np.asarray(info['center'],dtype='float32'),extent=np.array(info['extent'],dtype='float32'))
        np.savez(dest/'train_pool.npz',**pool)
        points=pool['positions'][pool['valid']]
        box=ref['box'];points=points[((points>=box[0])&(points<=box[1])).all(1)]
        if not len(points):raise ValueError(f'No visible surface points inside shared ROI: {method}/{scene}/{state}')
        rng=np.random.default_rng(config['seed']);points=points[rng.choice(len(points),min(100000,len(points)),replace=False)]
        np.save(dest/'visible_surface_points.npy',points)
        reference=ref[state]
        geometry=chamfer(points,reference)
        # Exact point-to-triangle distance avoids the sparse reference density bias
        # in the reference -> mesh direction. It still cannot measure unknown surfaces.
        distance=mesh.distance(reference)
        if state=='recomposition' and obj is not None:distance=np.minimum(distance,obj.distance(reference))
        geometry.update(reference_to_mesh_mean=float(distance.mean()),reference_to_mesh_rmse=float(np.sqrt(np.mean(distance**2))),
                        reference_to_mesh_median=float(np.median(distance)),reference_to_mesh_p95=float(np.quantile(distance,.95)),
                        reference_points=len(reference),visible_surface_samples=len(points),
                        CD_L1_normalized=geometry['CD_L1']/info['extent'],
                        reference_to_mesh_mean_normalized=float(distance.mean()/info['extent']),
                        reference_kind=info['reference']['kind'],
                        CD_sampling='Virtual scan: equal pixel samples per original-state training camera; shared ROI; background excludes target pixels',
                        warning='Sparse SfM reference is incomplete and reused during reconstruction: NOT independent dense GT accuracy.')
        geometries[state]=geometry
        write(dest/'geometry_metrics.json',geometry)
        write(dest/'complete.json',dict(complete=True,request=requests[state],mesh_counts=mesh.counts+(obj.counts if state=='recomposition' and obj else []),
              coverage=coverage[state],pool_rays=len(pool['valid']),has_native_color=native,scene_status=document['status'],geometry=geometry))
        print('GEOMETRY',method,scene,state,geometry['CD_L1'],geometry['reference_to_mesh_mean'],flush=True)
        del pool,points
    del mesh,obj,pools;gc.collect()

