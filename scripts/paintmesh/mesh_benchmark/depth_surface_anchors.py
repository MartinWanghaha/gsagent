"""Error-checked PGSR surface anchors for an isolated tetrahedral mesh export.

The source image is never resized. Every source pixel participates in validity
and approximation tests. Recursive image tiles compress only geometrically
redundant observations: an accepted tile's inverse-depth interpolation must
explain *all* its pixels within the declared axial, pixel-footprint error and
normal cone. Anchors always retain exact original-pixel positions and colors.

The approximation test is not a statement about true geometry or held-out
reprojection accuracy. It is a bound on representing the frozen depth map.
The cross-view merge is similarly conservative and retains a source observation
instead of averaging distinct surface layers or opposite-facing normals.
"""
from itertools import product
import math
import numpy as np


PARAMETERS=dict(max_leaf_pixels=16,depth_error_pixels=.5,normal_cone_degrees=5.,
    depth_jump_relative=.05,pixel_center_offset=0.,minimum_normal_length=1e-8,
    minimum_plane_ray_dot=1e-8,merge_distance_pixels=.5,
    merge_point_plane_pixels=.25,merge_normal_degrees=5.,merge_scale_ratio=2.,
    maximum_points=None,point_subsampling=False)
ARRAY_FIELDS=('positions','normals','scale','confidence','colors','pixels',
    'frame_indices','anchor_spacing')


def empty_anchors():
    return dict(positions=np.empty((0,3),dtype='float32'),
        normals=np.empty((0,3),dtype='float32'),scale=np.empty(0,dtype='float32'),
        confidence=np.empty(0,dtype='float32'),colors=np.empty((0,3),dtype='uint8'),
        pixels=np.empty((0,2),dtype='int32'),frame_indices=np.empty(0,dtype='int32'),
        anchor_spacing=np.empty(0,dtype='float32'))


def checked_frame(frame):
    depth=np.asarray(frame['depth'],dtype='float64')
    if depth.ndim!=2:raise ValueError('HW camera z-depth required')
    h,w=depth.shape
    if not h or not w:raise ValueError('Nonempty native depth image required')
    normal=np.asarray(frame['normal'],dtype='float64')
    if normal.shape==(3,h,w) and normal.shape!=(h,w,3):normal=normal.transpose(1,2,0)
    if normal.shape!=(h,w,3):raise ValueError('Camera-space HWC normals required')
    K=np.asarray(frame['K'],dtype='float64');T=np.asarray(frame['w2c'],dtype='float64')
    if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():
        raise ValueError('Finite pinhole K and rigid world-to-camera transform required')
    if K[0,0]<=0 or K[1,1]<=0 or abs(K[0,1])+abs(K[1,0])>1e-9 or not np.allclose(K[2],[0,0,1]):
        raise ValueError('Positive focal lengths and unskewed pinhole intrinsics required')
    R=T[:3,:3]
    if not np.allclose(R@R.T,np.eye(3),atol=2e-5) or not np.isclose(np.linalg.det(R),1,atol=2e-5):
        raise ValueError('World-to-camera rotation must be rigid')
    alpha=np.asarray(frame.get('alpha',np.ones_like(depth)),dtype='float64')
    confidence=np.asarray(frame.get('confidence',alpha),dtype='float64')
    if alpha.shape==(1,h,w):alpha=alpha[0]
    if confidence.shape==(1,h,w):confidence=confidence[0]
    if alpha.shape!=depth.shape or confidence.shape!=depth.shape:raise ValueError('Scalar confidence/alpha shape mismatch')
    rgb=np.asarray(frame['rgb'])
    if rgb.shape!=(h,w,3) or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>255:
        raise ValueError('Native RGB8 image shape/range required')
    length=np.linalg.norm(normal,axis=-1)
    valid=np.isfinite(depth)&(depth>0)&np.isfinite(normal).all(-1)&(length>PARAMETERS['minimum_normal_length'])
    valid&=np.isfinite(alpha)&(alpha>0)&np.isfinite(confidence)&(confidence>=0)
    unit=np.divide(np.where(np.isfinite(normal),normal,0),length[...,None],
        out=np.zeros_like(normal),where=length[...,None]>PARAMETERS['minimum_normal_length'])
    yy,xx=np.mgrid[:h,:w]
    rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],-1)
    unit=np.where(((unit*rays).sum(-1)>0)[...,None],-unit,unit)
    dot=np.abs((unit*rays).sum(-1))
    footprint_defined=dot>PARAMETERS['minimum_plane_ray_dot']
    valid&=footprint_defined
    footprint=np.zeros_like(depth)
    footprint[valid]=depth[valid]/np.sqrt(K[0,0]*K[1,1]*dot[valid])
    return dict(depth=depth,normal=unit,rays=rays,valid=valid,footprint=footprint,
        rgb=np.rint(rgb).astype('uint8'),K=K,w2c=T,
        confidence=np.minimum(np.maximum(confidence,0),np.clip(alpha,0,1)),
        footprint_undefined=int((np.isfinite(depth)&(depth>0)&~footprint_defined).sum()))


def _tile_error(data,y0,y1,x0,x1):
    """All-pixel axial error expressed in local source pixel-depth units."""
    valid=data['valid'][y0:y1,x0:x1]
    if not valid.all():return None
    d=data['depth'][y0:y1,x0:x1];nh,nw=d.shape
    # Discontinuous depth layers are never represented by a shared tile,
    # including 2-pixel tiles whose bilinear fit would otherwise be exact.
    for axis in [0,1]:
        if d.shape[axis]>1:
            a=np.take(d,range(d.shape[axis]-1),axis=axis)
            b=np.take(d,range(1,d.shape[axis]),axis=axis)
            if np.any(np.abs(a-b)>PARAMETERS['depth_jump_relative']*np.minimum(a,b)):
                return None
    tx=np.linspace(0,1,nw) if nw>1 else np.zeros(1)
    ty=np.linspace(0,1,nh) if nh>1 else np.zeros(1)
    a=tx[None,:];b=ty[:,None]
    rho=((1-a)*(1-b)/d[0,0]+a*(1-b)/d[0,-1]+
         (1-a)*b/d[-1,0]+a*b/d[-1,-1])
    predicted=1/rho
    # z/f is the axial depth error corresponding to one source-pixel angular
    # scale. No claim is made that moving along the same ray changes its uv.
    focal=math.sqrt(data['K'][0,0]*data['K'][1,1])
    pixel_error=np.abs(predicted-d)*focal/d
    normals=data['normal'][y0:y1,x0:x1]
    ref=normals[(nh-1)//2,(nw-1)//2]
    cosine=np.clip((normals*ref).sum(-1),-1,1)
    error=float(pixel_error.max(initial=0));angle=float(np.degrees(np.arccos(cosine.min(initial=1))))
    if error>PARAMETERS['depth_error_pixels']+1e-9 or angle>PARAMETERS['normal_cone_degrees']+1e-8:
        return None
    return error,angle


def build_frame_anchors(frame):
    """Select exact source pixels after checking every tile pixel, without caps."""
    data=checked_frame(frame);h,w=data['depth'].shape
    selected=np.zeros((h,w),dtype=bool)
    spacing=np.zeros((h,w),dtype='float64')
    leaves=0;singletons=0;tested_pixels=0;max_error=0.;max_angle=0.;covered=np.zeros((h,w),dtype='uint8')
    block=PARAMETERS['max_leaf_pixels']
    stack=[(y,min(y+block,h),x,min(x+block,w)) for y in range(0,h,block) for x in range(0,w,block)]
    while stack:
        y0,y1,x0,x1=stack.pop();valid=data['valid'][y0:y1,x0:x1]
        tested_pixels+=int(valid.size)
        if not valid.any():covered[y0:y1,x0:x1]+=1;continue
        error=_tile_error(data,y0,y1,x0,x1)
        if error is None and (y1-y0>1 or x1-x0>1):
            ys=[y0,(y0+y1)//2,y1] if y1-y0>1 else [y0,y1]
            xs=[x0,(x0+x1)//2,x1] if x1-x0>1 else [x0,x1]
            for a,b in zip(ys[:-1],ys[1:]):
                for c,d in zip(xs[:-1],xs[1:]):stack.append((a,b,c,d))
            continue
        # A singleton's exact observed surface needs no interpolation. Every
        # remaining valid pixel is retained even near invalid-depth boundaries.
        if error is None:error=(0.,0.)
        covered[y0:y1,x0:x1]+=1;leaves+=1
        singletons+=int(y1-y0==1 and x1-x0==1)
        max_error=max(max_error,error[0]);max_angle=max(max_angle,error[1])
        points={(y0,x0),(y0,x1-1),(y1-1,x0),(y1-1,x1-1),((y0+y1-1)//2,(x0+x1-1)//2)}
        for y,x in points:
            if not data['valid'][y,x]:continue
            selected[y,x]=True
            spacing[y,x]=max(spacing[y,x],max(y1-y0,x1-x0)*data['footprint'][y,x])
    if not np.all(covered==1):raise AssertionError('Native source-pixel coverage of leaf checks is incomplete')
    y,x=np.nonzero(selected);T=data['w2c'];R=T[:3,:3]
    p=(data['rays'][y,x]*data['depth'][y,x,None]-T[:3,3])@R
    n=data['normal'][y,x]@R
    n/=np.linalg.norm(n,axis=1,keepdims=True)
    result=dict(positions=p.astype('float32'),normals=n.astype('float32'),
        scale=data['footprint'][y,x].astype('float32'),confidence=data['confidence'][y,x].astype('float32'),
        colors=data['rgb'][y,x].copy(),pixels=np.stack([x,y],axis=1).astype('int32'),
        frame_indices=np.full(len(p),int(frame.get('frame_index',0)),dtype='int32'),
        anchor_spacing=spacing[y,x].astype('float32'))
    if len(p) and (not all(np.isfinite(result[k]).all() for k in ['positions','normals','scale','confidence','anchor_spacing']) or not (result['scale']>0).all()):
        raise ValueError('Anchor Float32 export overflow or nonpositive footprint')
    receipt=dict(complete=True,image_shape=[h,w],source_pixels_considered=int(h*w),
        valid_source_pixels=int(data['valid'].sum()),all_native_pixels_checked=True,
        image_downsampling=False,no_point_cap=True,parameters=PARAMETERS,
        recursive_pixel_tests=tested_pixels,accepted_tiles=leaves,singleton_tiles=singletons,
        anchors=len(p),maximum_accepted_axial_depth_error_pixels=max_error,
        maximum_accepted_normal_cone_degrees=max_angle,
        undefined_tangent_footprint_pixels=data['footprint_undefined'],
        depth_error_interpretation='abs(z_bilinear_inverse_depth-z_observed)*sqrt(fx*fy)/z_observed; bound against frozen source observation, not true geometry or held-out reprojection',
        scale_interpretation='sqrt(source pixel local tangent-plane area); distinct from anchor_spacing',
        pixel_center_offset=0.,positions_and_colors='Exact observed source pixels; no geometry/color averaging',
        world_bounds=[p.min(0).tolist(),p.max(0).tolist()] if len(p) else None,
        test_data_used=False,gaussians_updated=False)
    return result,receipt


def _checked_anchors(arrays):
    if not all(name in arrays for name in ARRAY_FIELDS):raise ValueError('Incomplete anchor schema')
    n=len(arrays['positions'])
    if any(len(arrays[name])!=n for name in ARRAY_FIELDS):raise ValueError('Anchor array lengths disagree')
    if arrays['positions'].shape!=(n,3) or arrays['normals'].shape!=(n,3):raise ValueError('Expected N,3 world positions/normals')
    for name in ['positions','normals','scale','confidence','anchor_spacing']:
        if not np.isfinite(arrays[name]).all():raise ValueError('Nonfinite anchor '+name)
    if n and (np.any(arrays['scale']<=0) or np.any(np.linalg.norm(arrays['normals'],axis=1)<.99)):
        raise ValueError('Positive footprint and unit normals required')
    return n


def merge_anchors(frame_outputs):
    """Conservatively merge duplicate observations, retaining best native sample.

    A deterministic multi-scale hash only considers similar-scale neighbors.
    Different layers farther than the explicit point-to-plane tolerance and
    opposite-facing surfaces cannot merge. The representative is never moved,
    and colors/normals are never averaged. This is not a topology guarantee.
    """
    arrays=[a[0] if isinstance(a,tuple) else a for a in frame_outputs]
    counts=[_checked_anchors(a) for a in arrays]
    if not sum(counts):return empty_anchors(),dict(complete=True,input_anchors=0,output_anchors=0,merged_observations=0,no_point_cap=True)
    joined={key:np.concatenate([a[key] for a in arrays],axis=0) for key in ARRAY_FIELDS}
    count=sum(counts);positions=joined['positions'].astype('float64');normals=joined['normals'].astype('float64')
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    scales=joined['scale'].astype('float64');confidence=joined['confidence']
    # Best confidence first; then finer observations and source provenance.
    # Spatial tie-breakers make the result invariant to frame-list order.
    order=np.lexsort((positions[:,2],positions[:,1],positions[:,0],joined['pixels'][:,0],joined['pixels'][:,1],
        joined['frame_indices'],scales,-confidence.astype('float64')))
    minimum=float(scales.min());levels=np.floor(np.log2(scales/minimum)).astype('int32')
    buckets={};kept=[];multiplicity=[];candidate_pairs=0;merged_dist=0.;merged_plane=0.;merged_angle=0.
    cosine=math.cos(math.radians(PARAMETERS['merge_normal_degrees']))
    offsets=list(product([-1,0,1],repeat=3))
    # A level contains footprints [base,2*base). A cell of width
    # 2*merge_radius*base is at least the permitted merge distance, so an
    # exact 3x3x3 neighbor search cannot skip a compatible representative.
    for index in order:
        p=positions[index];n=normals[index];s=scales[index];level=int(levels[index]);matches=[]
        for other_level in [level-1,level,level+1]:
            if other_level<0:continue
            cell=minimum*(2.**other_level)*(2*PARAMETERS['merge_distance_pixels'])
            center=np.floor(p/cell).astype('int64')
            for offset in offsets:
                key=(other_level,int(center[0])+offset[0],int(center[1])+offset[1],int(center[2])+offset[2])
                for slot in buckets.get(key,()):
                    representative=kept[slot];rscale=scales[representative]
                    if max(s,rscale)>PARAMETERS['merge_scale_ratio']*min(s,rscale):continue
                    candidate_pairs+=1;delta=p-positions[representative];small=min(s,rscale)
                    distance=float(np.linalg.norm(delta))
                    if distance>PARAMETERS['merge_distance_pixels']*small:continue
                    dot=float(np.clip(n@normals[representative],-1,1))
                    if dot<cosine:continue
                    plane=max(abs(float(delta@n)),abs(float(delta@normals[representative])))
                    if plane>PARAMETERS['merge_point_plane_pixels']*small:continue
                    matches.append((distance,slot,plane,math.degrees(math.acos(dot))))
        if matches:
            distance,slot,plane,angle=min(matches)
            multiplicity[slot]+=1
            merged_dist=max(merged_dist,distance/min(s,scales[kept[slot]]))
            merged_plane=max(merged_plane,plane/min(s,scales[kept[slot]]))
            merged_angle=max(merged_angle,angle)
        else:
            slot=len(kept);kept.append(int(index));multiplicity.append(1)
            cell=minimum*(2.**level)*(2*PARAMETERS['merge_distance_pixels']);coord=np.floor(p/cell).astype('int64')
            key=(level,int(coord[0]),int(coord[1]),int(coord[2]));buckets.setdefault(key,[]).append(slot)
    ids=np.asarray(kept,dtype='int64');result={key:joined[key][ids].copy() for key in ARRAY_FIELDS}
    result['observation_counts']=np.asarray(multiplicity,dtype='int32')
    receipt=dict(complete=True,input_frames=len(arrays),input_anchors=count,output_anchors=len(ids),
        merged_observations=count-len(ids),candidate_pairs_checked=candidate_pairs,
        maximum_merged_distance_source_pixels=merged_dist,maximum_merged_point_plane_source_pixels=merged_plane,
        maximum_merged_normal_angle_degrees=merged_angle,parameters=PARAMETERS,
        deterministic_best_source_observation=True,world_positions_averaged=False,colors_averaged=False,
        opposite_orientations_merged=False,no_point_cap=True,all_frame_anchors_considered=True,
        scale_levels=len(np.unique(levels)),
        scale_policy='Only compatible neighboring scale levels are candidates; distant-scale observations are retained, never discarded',
        layer_limit='Surfaces separated by more than the explicit point-to-plane tolerance cannot merge; subpixel-separated same-facing layers remain observationally ambiguous',
        unknown_topology_preserved=True,test_data_used=False,gaussians_updated=False)
    return result,receipt
