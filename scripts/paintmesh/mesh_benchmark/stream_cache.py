"""Prepare exact mesh visibility one partition at a time, over every camera.

Near-depth composition is associative. All original triangles/rays participate;
this computes the same union as one enormous BVH with bounded host memory.
"""
import gc
from pathlib import Path
import numpy as np
from PIL import Image
from .core import (read,write,identity,rays,image_mask,MeshScene,specs_for,digest,
                   points_from_depth,chamfer)
from .cache import common


def outside_frustum(bounds,camera):
    import itertools
    lo,hi=np.asarray(bounds,dtype='float64')
    corners=np.array(list(itertools.product(*zip(lo,hi))))
    W=np.asarray(camera['w2c']);q=corners@W[:3,:3].T+W[:3,3];p=q@np.asarray(camera['K']).T
    planes=np.c_[q[:,2],p[:,0],camera['width']*q[:,2]-p[:,0],p[:,1],camera['height']*q[:,2]-p[:,1]]
    return bool(np.any(np.all(planes < -1e-6,axis=0)))


def aabb_distance(points,bounds):
    lo,hi=np.asarray(bounds,dtype='float64')
    return np.linalg.norm(np.maximum(np.maximum(lo-points,points-hi),0),axis=1)


def prepare_streamed(output,method,scene,config):
    output=Path(output);cs,document,info=common(output,scene)
    base=output/'cache'/method/scene;selection=document['input_selection']
    specs=specs_for(scene,method,'background')
    requests={state:dict(specs=[dict(identity=identity(s['path']),transform=s.get('transform')) for s in specs_for(scene,method,state)],
        common=info['request_id'],seed=config['seed'],pool_pixels=config['pool_pixels'],protocol_version=config['protocol_version'])
        for state in ['background','recomposition']}
    if all((base/s/'complete.json').exists() and read(base/s/'complete.json')['request']==r for s,r in requests.items()):return
    ref=np.load(output/'common'/scene/'reference_points.npz');views=[];counts=[]
    distances={s:np.full(len(ref[s]),np.inf,dtype='float32') for s in requests}
    for i,c in enumerate(cs):
        rgb,fg=image_mask(c,selection)
        pixels=np.random.default_rng(config['seed']+i).integers(c['width']*c['height'],size=config['pool_pixels']) if c['part']=='train' else None
        rr=rays(c,pixels)
        views.append(dict(camera=c,rr=rr,pixels=pixels,image=rgb,fg=fg,depth=np.zeros(len(rr),dtype='float32'),
            color=np.zeros((len(rr),3),dtype='float32') if pixels is None else None))
    for j,spec in enumerate(specs):
        mesh=MeshScene([spec]);counts.extend(mesh.counts)
        if not all(x['has_native_color'] for x in mesh.counts):raise ValueError('Fine native mesh lacks color')
        bounds=mesh.counts[0]['bounds']
        for v in views:
            # Conservative half-space culling only: a skipped AABB cannot
            # intersect any image ray. No projected-size or distance heuristic.
            if outside_frustum(bounds,v['camera']):continue
            z,color=mesh.cast(v['rr'],color=v['pixels'] is None)
            front=(z>0)&((v['depth']==0)|(z<v['depth']));v['depth'][front]=z[front]
            if color is not None:v['color'][front]=color[front]
        for state in requests:
            numerical_margin=1e-5*(1+float(np.max(np.abs(ref[state]))))
            improve=aabb_distance(ref[state],bounds)<distances[state]+numerical_margin
            if improve.any():distances[state][improve]=np.minimum(distances[state][improve],mesh.distance(ref[state][improve]))
        del mesh;gc.collect()
        print('FINE_STREAM_RENDER',method,scene,j+1,len(specs),flush=True)
    extra=[] if method!='paintmesh' else specs_for(scene,method,'recomposition')[len(specs):]
    obj=MeshScene(extra) if extra else None
    if obj is not None:distances['recomposition']=np.minimum(distances['recomposition'],obj.distance(ref['recomposition']))
    pools={s:{k:[] for k in ['positions','directions','rgb','valid']} for s in requests};coverage={s:[] for s in requests}
    for v in views:
        c=v['camera'];fg=v['fg'];pixels=v['pixels'];rr=v['rr'];z=v['depth'];rgb=v['color']
        rz=z.copy();rrgb=None if rgb is None else rgb.copy()
        if obj is not None:
            oz,orgb=obj.cast(rr,color=pixels is None);front=(oz>0)&((rz==0)|(oz<rz));rz[front]=oz[front]
            if rrgb is not None:rrgb[front]=orgb[front]
        for state,depth,color in [('background',z,rgb),('recomposition',rz,rrgb)]:
            dest=base/state;dest.mkdir(parents=True,exist_ok=True)
            allowed=~fg.reshape(-1) if state=='background' else np.ones(fg.size,dtype=bool)
            if pixels is not None:
                allowed=allowed[pixels];points=rr[:,:3]+rr[:,3:]*depth[:,None]
                directions=rr[:,3:]/np.linalg.norm(rr[:,3:],axis=1,keepdims=True)
                pools[state]['positions'].append(points[allowed].astype('float32'))
                pools[state]['directions'].append(directions[allowed].astype('float32'))
                pools[state]['rgb'].append(v['image'].reshape(-1,3)[pixels][allowed]);pools[state]['valid'].append(depth[allowed]>0)
            else:
                (dest/'depth').mkdir(exist_ok=True);(dest/'native_renders').mkdir(exist_ok=True)
                np.save(dest/'depth'/(c['stem']+'.npy'),depth.reshape(c['height'],c['width']))
                image=np.rint(np.clip(color,0,1).reshape(v['image'].shape)*255).astype('uint8')
                Image.fromarray(image).save(dest/'native_renders'/(c['stem']+'.png'))
                coverage[state].append(dict(name=c['name'],coverage=float(((depth>0)&allowed).sum()/allowed.sum()),pixels=int(allowed.sum())))
    for state in requests:
        dest=base/state;pool={k:np.concatenate(v) for k,v in pools[state].items()}
        pool.update(center=np.asarray(info['center'],dtype='float32'),extent=np.array(info['extent'],dtype='float32'))
        np.savez(dest/'train_pool.npz',**pool)
        points=pool['positions'][pool['valid']];box=ref['box'];points=points[((points>=box[0])&(points<=box[1])).all(1)]
        if not len(points):raise ValueError('No fine surface inside shared ROI')
        rng=np.random.default_rng(config['seed']);points=points[rng.choice(len(points),min(100000,len(points)),replace=False)]
        np.save(dest/'visible_surface_points.npy',points);geometry=chamfer(points,ref[state]);distance=distances[state]
        geometry.update(reference_to_mesh_mean=float(distance.mean()),reference_to_mesh_rmse=float(np.sqrt(np.mean(distance**2))),
            reference_to_mesh_median=float(np.median(distance)),reference_to_mesh_p95=float(np.quantile(distance,.95)),
            reference_points=len(ref[state]),visible_surface_samples=len(points),CD_L1_normalized=geometry['CD_L1']/info['extent'],
            reference_to_mesh_mean_normalized=float(distance.mean()/info['extent']),reference_kind=info['reference']['kind'],
            CD_sampling='Unchanged virtual scan, complete mesh partitions and original rays',
            warning='Sparse SfM, not independent scan GT')
        write(dest/'geometry_metrics.json',geometry)
        write(dest/'complete.json',dict(complete=True,request=requests[state],mesh_counts=counts+(obj.counts if obj and state=='recomposition' else []),
            coverage=coverage[state],pool_rays=len(pool['valid']),has_native_color=True,scene_status=document['status'],geometry=geometry,
            streamed_all_faces=True,geometry_simplified=False))
    del obj,views,pools;gc.collect()
