"""Full-pixel oriented samples for an isolated FSSR mesh export.

Each valid PGSR pixel becomes one sample. World positions are unprojected
camera z-depth; normals are rotated to world coordinates without averaging
opposite sides of a thin surface. FSSR's ``value`` is a POSITIVE sampling scale
(the square root of one pixel's tangent-plane area), not an SDF zero value.

All declared training views contribute to visibility-aware *soft* confidence.
Occlusion is not a contradiction, and observations with no other-view support
are retained. Alpha weights evidence; it is not a probability of accuracy.
This module does not merge or decimate samples. FSSR performs the subsequent
local implicit reconstruction from these overlapping oriented observations.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import fcntl
import gc
import json
import os
import re
import tempfile
import time
import numpy as np

from .core import read,write,identity,digest


FORMAT='paintmesh-full-pixel-fssr-surfels-v1'
DTYPE=np.dtype([(n,'<f4') for n in ['x','y','z','nx','ny','nz']]+
    [(n,'u1') for n in ['red','green','blue']]+
    [(n,'<f4') for n in ['confidence','value','scale']])
DEFAULTS=dict(chunk_points=65536,depth_relative_band=.01,normal_cosine_band=.25,
    prior_observations=1.,consistency_floor=.05,minimum_normal_length=1e-8,
    minimum_plane_ray_dot=1e-8,pixel_center_offset=0.,explicit_deduplication=False,
    point_subsampling=False,maximum_points=None)


def root(output):return Path(output)/'surfels'


def load_frame(record):
    path=Path(record.get('path',record.get('source',{}).get('path','')))
    stamp=record.get('identity',record.get('source',{}).get('identity'))
    if stamp and identity(path)!=stamp:raise ValueError('Training geometry frame changed: '+str(path))
    with np.load(path) as arrays:
        required=['depth','normal','rgb','K','w2c','alpha','confidence']
        missing=[name for name in required if name not in arrays]
        if missing:raise ValueError('Missing surfel frame fields: '+str(missing))
        return {name:arrays[name].copy() for name in required}


def checked_frame(frame):
    d=np.asarray(frame['depth'],dtype='float64')
    if d.ndim!=2:raise ValueError('HW camera z-depth is required')
    h,w=d.shape;n=np.asarray(frame['normal'],dtype='float64')
    if n.shape==(3,h,w) and n.shape!=(h,w,3):n=n.transpose(1,2,0)
    if n.shape!=(h,w,3):raise ValueError('HWC or CHW camera-space normal is required')
    alpha=np.asarray(frame['alpha'],dtype='float64').squeeze()
    confidence=np.asarray(frame['confidence'],dtype='float64').squeeze()
    if alpha.shape!=d.shape or confidence.shape!=d.shape:raise ValueError('Scalar evidence shape mismatch')
    rgb=np.asarray(frame['rgb'])
    if rgb.shape!=(h,w,3) or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>255:
        raise ValueError('Native RGB8 values are required')
    K=np.asarray(frame['K'],dtype='float64');T=np.asarray(frame['w2c'],dtype='float64')
    if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():
        raise ValueError('Finite camera calibration required')
    if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1],atol=1e-9):
        raise ValueError('Pinhole intrinsics with positive focal length required')
    if abs(K[0,1])+abs(K[1,0])>1e-9:raise ValueError('Skewed intrinsics are unsupported')
    R=T[:3,:3]
    if not np.allclose(R@R.T,np.eye(3),atol=2e-5) or not np.isclose(np.linalg.det(R),1,atol=2e-5):
        raise ValueError('Rigid world-to-camera rotation required')
    length=np.linalg.norm(n,axis=-1)
    valid=np.isfinite(d)&(d>0)&np.isfinite(n).all(-1)&(length>DEFAULTS['minimum_normal_length'])
    valid&=np.isfinite(alpha)&(alpha>0)&np.isfinite(confidence)&(confidence>=0)
    safe=np.where(np.isfinite(n),n,0)
    unit=np.divide(safe,length[...,None],out=np.zeros_like(safe),where=length[...,None]>DEFAULTS['minimum_normal_length'])
    return dict(depth=d,normal=unit,rgb=np.rint(rgb).astype('uint8'),K=K,w2c=T,
        alpha=np.clip(alpha,0,1),confidence=confidence,valid=valid)


def frame_surfels(frame):
    """Return original-pixel world surfels and validity accounting.

    For a unit camera normal n and ray r with r_z=1, the one-pixel tangent
    surface area is z²/(fx*fy*abs(n dot r)). Its square root is FSSR's scale.
    This uses the observing pixel footprint, never nearest-point spacing.
    """
    f=checked_frame(frame);d=f['depth'];h,w=d.shape;K=f['K'];T=f['w2c']
    yy,xx=np.mgrid[:h,:w]
    rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],-1)
    dot=(f['normal']*rays).sum(-1)
    valid=f['valid']&(np.abs(dot)>DEFAULTS['minimum_plane_ray_dot'])
    y,x=np.nonzero(valid);z=d[y,x];r=rays[y,x];n=f['normal'][y,x]
    n=np.where(((n*r).sum(-1)>0)[:,None],-n,n)
    world=(r*z[:,None]-T[:3,3])@T[:3,:3]
    nw=n@T[:3,:3];nw/=np.linalg.norm(nw,axis=1,keepdims=True)
    scale=np.sqrt(z*z/(K[0,0]*K[1,1]*np.abs((n*r).sum(-1))))
    incidence=np.abs((n*r).sum(-1))/np.linalg.norm(r,axis=1)
    # The upstream cache confidence ALREADY includes alpha. Clip its other
    # reliability factors to [0,1], retaining alpha exactly once.
    base=np.clip(f['confidence'][y,x],0,f['alpha'][y,x])*np.clip(incidence,0,1)
    result=np.empty(len(z),dtype=DTYPE)
    for j,key in enumerate(['x','y','z']):result[key]=world[:,j]
    for j,key in enumerate(['nx','ny','nz']):result[key]=nw[:,j]
    for j,key in enumerate(['red','green','blue']):result[key]=f['rgb'][y,x,j]
    result['confidence']=base;result['value']=scale;result['scale']=scale
    if len(result) and (not np.isfinite(world).all() or not np.isfinite(scale).all() or not (scale>0).all()):
        raise ValueError('Invalid world samples or positive footprint scale')
    for key in ['x','y','z','nx','ny','nz','confidence','value','scale']:
        if not np.isfinite(result[key]).all():raise ValueError('Float32 surfel export overflow: '+key)
    if len(result) and not (result['value']>0).all():raise ValueError('Positive footprint underflow in Float32 export')
    exported_world=np.column_stack([result[key] for key in ['x','y','z']])
    accounting=dict(image_pixels=int(h*w),finite_positive_depth_pixels=int((np.isfinite(d)&(d>0)).sum()),
        depth_normal_alpha_valid_pixels=int(f['valid'].sum()),valid_pixels=int(valid.sum()),points=len(result),
        undefined_tangent_footprint_pixels=int((f['valid']&~valid).sum()),
        pixel_center_offset=0.,pixel_convention='PGSR camera_rays integer pixel centers; z component equals one',
        value_semantics='Positive sqrt(pixel tangent-plane area), in COLMAP scene units; FSSR scale',
        base_confidence_semantics='min(upstream confidence,alpha) * incidence; upstream confidence already includes alpha once; not a calibrated accuracy probability',
        world_bounds=[exported_world.min(0).tolist(),exported_world.max(0).tolist()] if len(result) else None,
        minimum_positive_footprint=float(result['value'].min()) if len(result) else None,
        maximum_footprint=float(result['value'].max()) if len(result) else None)
    return result,accounting


def prepare_targets(frames,device='cpu'):
    import torch
    target=[]
    for original in frames:
        f=checked_frame(original);T=f['w2c'];R=T[:3,:3]
        normal=f['normal']
        # Orient each target's input normal toward its own camera, retaining
        # separate observations for the opposite side of thin surfaces.
        h,w=f['depth'].shape;yy,xx=np.mgrid[:h,:w];K=f['K']
        ray=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],-1)
        normal=np.where(((normal*ray).sum(-1)>0)[...,None],-normal,normal)
        tensor=lambda a:torch.as_tensor(np.ascontiguousarray(a),device=device)
        target.append(dict(depth=tensor(f['depth'].astype('float32')),normal=tensor(normal.astype('float32')),
            alpha=tensor(f['alpha'].astype('float32')),valid=tensor(f['valid']),
            R=tensor(R.astype('float32')),t=tensor(T[:3,3].astype('float32')),
            center=tensor((-T[:3,3]@R).astype('float32')),K=K))
    return target


def consistency_weights(surfels,source_index,targets,device='cpu',chunk_points=65536):
    """Soft scores from all other training views, with explicit visibility.

    A sample behind an observed surface is occluded and contributes no vote.
    A sample in front of an observed surface receives a low Cauchy score; it
    remains present. Views on the opposite side of the sample's oriented
    surface do not compare or average their normals. No other-view evidence
    leaves weight one, so far background is never deleted for lacking 2 views.
    """
    import torch
    if chunk_points<1:raise ValueError('Positive processing batch required')
    count=len(surfels);result=np.ones(count,dtype='float32')
    stats=dict(points=count,views_considered=len(targets),self_view_excluded=source_index is not None,
        eligible_observations=0,occluded_observations=0,opposite_side_observations=0,
        points_without_other_view_evidence=0,free_space_conflicts=0,normal_averaging=False,
        hard_multiview_point_rejection=False,all_training_views_considered=True)
    counters=torch.zeros(4,dtype=torch.int64,device=device)
    with torch.no_grad():
        for start in range(0,count,chunk_points):
            stop=min(start+chunk_points,count);part=surfels[start:stop]
            xyz=torch.as_tensor(np.column_stack([part[n] for n in ['x','y','z']]),device=device)
            normal=torch.as_tensor(np.column_stack([part[n] for n in ['nx','ny','nz']]),device=device)
            scale=torch.as_tensor(np.asarray(part['value']).copy(),device=device)
            votes=torch.zeros(len(part),device=device);agreement=torch.zeros_like(votes)
            evidence=torch.zeros(len(part),device=device,dtype=torch.bool)
            for index,frame in enumerate(targets):
                if index==source_index:continue
                q=xyz@frame['R'].T+frame['t'];z=q[:,2];K=frame['K'];h,w=frame['depth'].shape
                safe=torch.where(z.abs()>1e-12,z,torch.ones_like(z))
                u=q[:,0]*float(K[0,0])/safe+float(K[0,2]);v=q[:,1]*float(K[1,1])/safe+float(K[1,2])
                inside=(z>0)&torch.isfinite(q).all(-1)&(u>=-.5)&(u<w-.5)&(v>=-.5)&(v<h-.5)
                x=torch.floor(u+.5).long().clamp(0,w-1);y=torch.floor(v+.5).long().clamp(0,h-1)
                valid=inside&frame['valid'][y,x]
                same_side=(normal*(frame['center']-xyz)).sum(-1)>0
                opposite_count=(valid&~same_side).sum()
                valid&=same_side
                depth=frame['depth'][y,x]
                band=torch.maximum(depth.abs()*DEFAULTS['depth_relative_band'],2*scale).clamp_min(1e-8)
                occluded=valid&(z>depth+band)
                occluded_count=occluded.sum();valid&=~occluded
                conflict_count=(valid&(z<depth-band)).sum()
                nw=frame['normal'][y,x]@frame['R']
                cosine=(normal*nw).sum(-1).clamp(-1,1)
                depth_score=1/(1+((z-depth)/band)**2)
                normal_score=1/(1+((1-cosine)/DEFAULTS['normal_cosine_band'])**2)
                weight=torch.where(valid,frame['alpha'][y,x],torch.zeros_like(z))
                votes+=weight;agreement+=weight*depth_score*normal_score;evidence|=valid
                counters+=torch.stack([valid.sum(),occluded_count,opposite_count,conflict_count])
            reliability=(DEFAULTS['prior_observations']+agreement)/(DEFAULTS['prior_observations']+votes)
            reliability=DEFAULTS['consistency_floor']+(1-DEFAULTS['consistency_floor'])*reliability
            result[start:stop]=reliability.clamp(0,1).cpu().numpy()
            stats['points_without_other_view_evidence']+=int((~evidence).sum())
    for key,value in zip(['eligible_observations','occluded_observations','opposite_side_observations','free_space_conflicts'],counters.cpu().tolist()):
        stats[key]=int(value)
    stats.update(minimum_weight=float(result.min(initial=1)),mean_weight=float(result.mean()) if count else None,
        maximum_weight=float(result.max(initial=0)))
    return result,stats


def write_ply(path,surfels):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    header=['ply','format binary_little_endian 1.0','comment '+FORMAT,
        'comment value is POSITIVE sampling scale, not zero signed distance',f'element vertex {len(surfels)}']
    for key in DTYPE.names:header.append('property '+('uchar' if DTYPE[key].kind=='u' else 'float')+' '+key)
    header+=['end_header','']
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write('\n'.join(header).encode('ascii'));np.asarray(surfels,dtype=DTYPE).tofile(handle)
            handle.flush();os.fsync(handle.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def prepare(output,frames_manifest):
    output=Path(output);folder=root(output);folder.mkdir(parents=True,exist_ok=True)
    source=read(frames_manifest) if isinstance(frames_manifest,(str,Path)) else frames_manifest
    if not source.get('complete') or not source.get('frames'):raise ValueError('Complete training-frame manifest required')
    records=[];names=set()
    for index,row in enumerate(source['frames']):
        name=row['name']
        if name in names:raise ValueError('Duplicate training frame name')
        names.add(name);stamp=identity(row['path'])
        if row.get('identity') and row['identity']!=stamp:raise ValueError('Source frame fingerprint changed')
        records.append(dict(index=index,name=name,source=dict(path=str(Path(row['path']).resolve()),identity=stamp),
            path=str((folder/'frames'/f'{index:03d}.ply').resolve())))
    declared=source.get('request',{}).get('train_views',len(records))
    if declared!=len(records):raise ValueError('All declared training frames are required')
    request=dict(format=FORMAT,implementation=identity(Path(__file__)),frames=records,
        source_manifest=digest(source),parameters=DEFAULTS,training_views=len(records),
        all_source_frames=True,no_subsampling=True,no_point_cap=True,
        position_optimization=False,gaussian_training=False,explicit_deduplication=False,
        merging='No explicit deduplication: overlapping samples retain their own orientation and are locally fused by FSSR')
    marker=folder/'manifest.json'
    if marker.exists():
        old=read(marker)
        if old['request']!=request:raise ValueError('Surfel export inputs or implementation changed')
        return old
    result=dict(complete=True,format=FORMAT,request=request,request_id=digest(request),records=records,
        frames=len(records),all_source_frames=True,created=time.time())
    write(marker,result);return result


def manifest(output):
    result=read(root(output)/'manifest.json')
    if not result.get('complete') or result['request']['implementation']!=identity(Path(__file__)):
        raise ValueError('Surfel plan is incomplete or implementation changed')
    return result


def completed(output,row,request_id):
    path=root(output)/'completed'/f'{row["index"]:03d}.json'
    if not path.exists():return None
    record=read(path)
    if not record.get('complete') or record['request_id']!=request_id or record['source']!=row['source']:
        raise ValueError('Completed surfel frame request differs')
    if identity(record['path'])!=record['identity']:raise ValueError('Completed surfel PLY changed')
    if identity(row['source']['path'])!=row['source']['identity']:raise ValueError('Original frame changed after export')
    return record


@contextmanager
def claim(output,row,request_id):
    path=root(output)/'claims'/f'{row["index"]:03d}.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:yield False;return
        try:yield completed(output,row,request_id) is None
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def run_worker(output,worker_id,device='cuda:0'):
    import torch
    worker_id=str(worker_id)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',worker_id):raise ValueError('Unsafe worker ID')
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
    output=Path(output);plan=manifest(output);torch.set_num_threads(4)
    frames=[load_frame(row['source']) for row in plan['records']]
    targets=prepare_targets(frames,device=device);done=0
    status=root(output)/'workers'/(worker_id+'.json')
    for row in plan['records']:
        with claim(output,row,plan['request_id']) as acquired:
            if not acquired:continue
            started=time.time();write(status,dict(stage='running',pid=os.getpid(),worker=worker_id,
                frame=row['index'],name=row['name'],device=device,updated=started))
            try:
                surfels,accounting=frame_surfels(frames[row['index']])
                weight,consistency=consistency_weights(surfels,row['index'],targets,device=device,
                    chunk_points=DEFAULTS['chunk_points'])
                surfels['confidence']*=weight
                write_ply(row['path'],surfels)
                record=dict(complete=True,request_id=plan['request_id'],index=row['index'],name=row['name'],
                    source=row['source'],path=row['path'],identity=identity(row['path']),points=len(surfels),
                    accounting=accounting,multiview=consistency,worker=worker_id,pid=os.getpid(),device=device,
                    no_subsampling=True,no_explicit_deduplication=True,colors_preserved=True,
                    opposite_orientations_averaged=False,value_is_positive_scale=True,
                    elapsed=time.time()-started,updated=time.time())
                write(root(output)/'completed'/f'{row["index"]:03d}.json',record);done+=1
                print('SURFEL_FRAME_COMPLETE',worker_id,row['index'],row['name'],'points',len(surfels),flush=True)
                del surfels,weight;gc.collect()
            except Exception as error:
                write(root(output)/'failures'/f'{row["index"]:03d}_{worker_id}.json',dict(
                    index=row['index'],worker=worker_id,error=repr(error),updated=time.time(),resumable=True))
                raise
    write(status,dict(stage='complete',pid=os.getpid(),worker=worker_id,completed_by_worker=done,updated=time.time()))
    del frames,targets;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return done


def progress(output):
    plan=manifest(output);records=[]
    for row in plan['records']:
        record=completed(output,row,plan['request_id'])
        if record:records.append(record)
    return dict(complete=len(records)==plan['frames'],completed_frames=len(records),total_frames=plan['frames'],
        points=sum(row['points'] for row in records))


def finalize(output):
    plan=manifest(output);records=[]
    for row in plan['records']:
        result=completed(output,row,plan['request_id'])
        if result is None:raise ValueError('Surfel frame is incomplete: '+row['name'])
        records.append(result)
    nonempty=[r for r in records if r['points']]
    if not nonempty:raise ValueError('No valid oriented surface samples')
    lower=np.min([r['accounting']['world_bounds'][0] for r in nonempty],axis=0)
    upper=np.max([r['accounting']['world_bounds'][1] for r in nonempty],axis=0)
    smallest=min(r['accounting']['minimum_positive_footprint'] for r in nonempty)
    largest=max(r['accounting']['maximum_footprint'] for r in nonempty)
    extent=float(np.max(upper-lower));ratio=extent/smallest
    preflight=dict(complete=True,supported=bool(ratio<=2**20),
        world_bounds=[lower.tolist(),upper.tolist()],maximum_axis_extent=extent,
        minimum_positive_footprint=smallest,maximum_footprint=largest,
        extent_to_minimum_footprint_ratio=ratio,fssr_octree_maximum_levels=20,
        ratio_limit=2**20,scale_clipping=False,
        interpretation='Conservative risk screen only, not proof of the actual FSSR octree level: root power-of-two rounding and support expansion can increase its extent')
    write(root(output)/'preflight.json',preflight)
    if not preflight['supported']:
        raise ValueError('Unsupported scene extent / minimum footprint exceeds FSSR 20-level risk bound; scales and points have NOT been clipped')
    result=dict(complete=True,format=FORMAT,request_id=plan['request_id'],records=records,
        points=sum(row['points'] for row in records),frames=len(records),all_source_frames=True,
        no_subsampling=True,no_point_cap=True,no_explicit_deduplication=True,
        value_is_positive_pixel_footprint_scale=True,geometry_training=False,
        world_bounds=[lower.tolist(),upper.tolist()],minimum_positive_footprint=smallest,
        maximum_footprint=largest,extent_to_minimum_footprint_ratio=ratio,preflight=preflight,
        merging='Overlapping samples remain separate; FSSR locally fuses their weighted implicit surface contributions',
        updated=time.time())
    write(root(output)/'complete.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--worker-id')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(args.output,read(args.output/'frames/complete.json'))
    elif args.finalize:finalize(args.output)
    elif args.worker_id is not None:run_worker(args.output,args.worker_id,args.device)
    else:parser.error('Choose --prepare, --finalize or --worker-id')
