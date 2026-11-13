"""Local PGSR surface charts for an isolated sparse tetrahedron experiment.

This is a supported, chart-oriented point-to-plane residual, NOT a global SDF
or opacity field. Unknown samples stay NaN. Opposing surface charts and charts
separated in their normal direction cannot share extracted tetrahedra. Edge
root queries keep a fixed chart; a nearest-anchor orientation switch can never
create a root inside an already accepted edge. Thin unobserved layers and
multiple roots can still require additional sampling; no watertightness claim.
"""
import numpy as np

NORMAL_COS=float(np.cos(np.deg2rad(45.)))
BAND_SCALE=4.
LAYER_SCALE=2.
EDGES=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],dtype='int64')


def _array(anchors,*names):
    for name in names:
        if name in anchors:return np.asarray(anchors[name])
    raise KeyError('Anchor field missing: '+str(names))


def anchor_arrays(anchors):
    p=_array(anchors,'points','p','positions').astype('float32')
    n=_array(anchors,'normals','n').astype('float32')
    s=_array(anchors,'scale','scales').astype('float32').reshape(-1)
    if p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape or len(s)!=len(p) or not len(p):raise ValueError('Nonempty anchor points, normals and scales required')
    length=np.linalg.norm(n,axis=1)
    if not np.isfinite(p).all() or not np.isfinite(n).all() or not np.isfinite(s).all() or (s<=0).any() or (length<1e-8).any():raise ValueError('Finite anchors and positive scales required')
    return p,n/length[:,None],s


def make_pivots(anchors,delta_factor=2.,include_centers=True):
    p,n,s=anchor_arrays(anchors)
    if delta_factor<=0:raise ValueError('Positive offset factor required')
    offsets=np.array([-1.,0.,1.] if include_centers else [-1.,1.],dtype='float32')
    points=(p[:,None,:]+offsets[None,:,None]*(s*delta_factor)[:,None,None]*n[:,None,:]).reshape(-1,3)
    ids=np.repeat(np.arange(len(p),dtype='int64'),len(offsets))
    unique,first,inverse=np.unique(points,axis=0,return_index=True,return_inverse=True)
    original_ids=ids;ids=ids[first]
    compatible=(n[original_ids]*n[ids[inverse]]).sum(1)>=NORMAL_COS
    separation=np.abs(((p[original_ids]-p[ids[inverse]])*n[ids[inverse]]).sum(1))
    compatible&=separation<=LAYER_SCALE*np.minimum(s[original_ids],s[ids[inverse]])+1e-8
    ambiguous=np.zeros(len(unique),dtype=bool);np.logical_or.at(ambiguous,inverse,~compatible)
    return dict(points=unique,reference_points=p[ids],reference_normals=n[ids],reference_scales=s[ids],
        anchor_indices=ids,delta=s[ids]*delta_factor,reference_ambiguous=ambiguous,
        diagnostics=dict(anchors=len(p),pivots=len(points),delta_factor=delta_factor,centers_included=include_centers,
            unique_pivots=len(unique),duplicate_coordinates_removed=len(points)-len(unique),
            ambiguous_chart_coordinates=int(ambiguous.sum()),ambiguous_coordinates_require_unknown=True,
            global_fixed_voxel_grid=False,normal_offsets_not_gaussian_centers=True))


def prepare_frames(frames,device='cpu'):
    import torch
    data=[]
    for source in frames:
        d=np.asarray(source['depth'],dtype='float32');h,w=d.shape
        n=np.asarray(source['normal'],dtype='float32')
        if n.shape==(3,h,w) and n.shape!=(h,w,3):n=n.transpose(1,2,0)
        if n.shape!=(h,w,3):raise ValueError('HWC camera normals required')
        K=np.asarray(source['K'],dtype='float64');T=np.asarray(source['w2c'],dtype='float64')
        if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all() or min(K[0,0],K[1,1])<=0:raise ValueError('Finite calibrated pinhole camera required')
        rgb=np.asarray(source['rgb'],dtype='float32')
        if rgb.shape!=(h,w,3) or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>255:raise ValueError('Native RGB8 source required')
        alpha=np.asarray(source.get('alpha',np.ones_like(d)),dtype='float32').squeeze()
        confidence=np.asarray(source.get('confidence',alpha),dtype='float32').squeeze()
        if alpha.shape!=d.shape or confidence.shape!=d.shape:raise ValueError('Scalar alpha/confidence image required')
        length=np.linalg.norm(n,axis=-1);valid=np.isfinite(d)&(d>0)&np.isfinite(n).all(-1)&(length>1e-8)&np.isfinite(confidence)&(confidence>0)&np.isfinite(alpha)&(alpha>0)
        n=np.divide(np.where(np.isfinite(n),n,0),length[...,None],out=np.zeros_like(n),where=length[...,None]>1e-8)
        yy,xx=np.mgrid[:h,:w];rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],-1)
        n=np.where((n*rays).sum(-1,keepdims=True)>0,-n,n)
        continuity=np.ones((max(h-1,0),max(w-1,0)),dtype=bool)
        for dy,dx in [(0,0),(0,1),(1,0),(1,1)]:
            v=valid[dy:h-1+dy,dx:w-1+dx];dd=d[dy:h-1+dy,dx:w-1+dx];nn=n[dy:h-1+dy,dx:w-1+dx]
            continuity&=v&(np.abs(dd-d[:-1,:-1])<=.05*np.maximum(np.abs(d[:-1,:-1]),1e-8))&((nn*n[:-1,:-1]).sum(-1)>=NORMAL_COS)
        tensor=lambda a:torch.as_tensor(np.ascontiguousarray(a),device=device)
        data.append(dict(depth=tensor(np.where(valid,d,0)),normal=tensor(n),rgb=tensor(rgb),valid=tensor(valid),
            confidence=tensor(np.where(valid,np.clip(confidence,0,1),0)),continuity=tensor(continuity),
            K=K,R=tensor(T[:3,:3].astype('float32')),t=tensor(T[:3,3].astype('float32')),
            center=tensor((-T[:3,3]@T[:3,:3]).astype('float32'))))
    return data


def _sample(points,frame):
    import torch
    q=points@frame['R'].T+frame['t'];z=q[:,2];K=frame['K'];h,w=frame['depth'].shape
    safe=torch.where(z.abs()>1e-12,z,torch.ones_like(z))
    u=q[:,0]*float(K[0,0])/safe+float(K[0,2]);v=q[:,1]*float(K[1,1])/safe+float(K[1,2])
    inside=(z>0)&torch.isfinite(q).all(-1)&(u>=0)&(u<w)&(v>=0)&(v<h)
    x=torch.floor(u).long().clamp(0,w-1);y=torch.floor(v).long().clamp(0,h-1)
    valid=inside&frame['valid'][y,x]
    d=frame['depth'][y,x];n=frame['normal'][y,x]
    if h>1 and w>1:
        cx=x.clamp(0,w-2);cy=y.clamp(0,h-2)
        interpolate=valid&(x<w-1)&(y<h-1)&frame['continuity'][cy,cx]
        a=(u-cx).clamp(0,1);b=(v-cy).clamp(0,1)
        weight=torch.stack([(1-a)*(1-b),a*(1-b),(1-a)*b,a*b],-1)
        depths=torch.stack([frame['depth'][cy,cx],frame['depth'][cy,cx+1],frame['depth'][cy+1,cx],frame['depth'][cy+1,cx+1]],-1)
        d=torch.where(interpolate,1/(weight/depths.clamp_min(1e-12)).sum(-1).clamp_min(1e-12),d)
        normals=torch.stack([frame['normal'][cy,cx],frame['normal'][cy,cx+1],frame['normal'][cy+1,cx],frame['normal'][cy+1,cx+1]],-2)
        n=torch.where(interpolate[:,None],(weight[...,None]*normals).sum(-2),n)
    n=n/torch.linalg.norm(n,dim=-1,keepdim=True).clamp_min(1e-12)
    ray=torch.stack([(u-float(K[0,2]))/float(K[0,0]),(v-float(K[1,2]))/float(K[1,1]),torch.ones_like(u)],-1)
    surface=(ray*d[:,None]-frame['t'])@frame['R']
    normal=n@frame['R']
    incidence=(-(n*ray).sum(-1)/torch.linalg.norm(ray,dim=-1).clamp_min(1e-12)).clamp(0,1)
    return surface,normal,frame['confidence'][y,x]*incidence,frame['rgb'][y,x],valid,z,d


def query_field(points,anchors,prepared,device='cpu',reference_normals=None,reference_points=None,
                reference_scales=None,chunk_points=32768):
    """Evaluate a locally oriented residual; preserve NaN for unsupported space.

    Explicit references MUST be reused during edge root finding. If omitted,
    nearest-anchor metadata is assigned once for initial candidate queries.
    Only compatible camera-facing observations within the reference surface
    layer and a four-footprint normal band participate; behind-surface points
    and unsupported query regions contribute no occupancy assumption.
    """
    import torch
    from scipy.spatial import cKDTree
    points=np.asarray(points,dtype='float32')
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or chunk_points<1:raise ValueError('Finite N,3 queries and positive chunk size required')
    ap,an,scale=anchor_arrays(anchors);count=len(points)
    missing=any(x is None for x in [reference_normals,reference_points,reference_scales])
    if missing:
        ids=cKDTree(ap).query(points,workers=4)[1] if count else np.empty(0,dtype='int64')
        if reference_normals is None:reference_normals=an[ids]
        if reference_points is None:reference_points=ap[ids]
        if reference_scales is None:reference_scales=scale[ids]
    nr=np.asarray(reference_normals,dtype='float32');pr=np.asarray(reference_points,dtype='float32');sr=np.asarray(reference_scales,dtype='float32').reshape(-1)
    if nr.shape!=points.shape or pr.shape!=points.shape or len(sr)!=count or not np.isfinite(nr).all() or not np.isfinite(pr).all() or not np.isfinite(sr).all() or (sr<=0).any():raise ValueError('One finite fixed chart and positive scale per query required')
    length=np.linalg.norm(nr,axis=1)
    if (length<1e-8).any():raise ValueError('Nonzero reference normals required')
    nr=nr/length[:,None]
    field=np.full(count,np.nan,dtype='float32');support=np.zeros(count,dtype=bool)
    normals=nr.copy();colors=np.zeros((count,3),dtype='uint8');weight_out=np.zeros(count,dtype='float32')
    counters=dict(occluded_observations=0,opposite_side_observations=0,incompatible_surface_layers=0,accepted_observations=0)
    with torch.no_grad():
        for start in range(0,count,chunk_points):
            end=min(start+chunk_points,count)
            tensor=lambda a:torch.as_tensor(np.ascontiguousarray(a),device=device)
            q=tensor(points[start:end]);ref=tensor(pr[start:end]);nref=tensor(nr[start:end]);sref=tensor(sr[start:end])
            total=torch.zeros(len(q),device=device);acc=torch.zeros_like(total);rgb=torch.zeros_like(q);nsum=torch.zeros_like(q)
            for frame in prepared:
                surface,n,w,color,valid,z,depth=_sample(q,frame)
                same_side=((frame['center']-ref)*nref).sum(-1)>0
                counters['opposite_side_observations']+=int((valid&~same_side).sum())
                cosine=(n*nref).sum(-1)
                compatible=cosine>=NORMAL_COS
                layer=((surface-ref)*nref).sum(-1).abs()<=LAYER_SCALE*sref
                signed=((q-surface)*n).sum(-1)
                # Incidence-aware normal-band visibility. Behind a measured
                # surface is unknown; do not let another depth layer fill it.
                ndot=((n*(q-frame['center'])).sum(-1).abs()/z.abs().clamp_min(1e-8)).clamp_min(1e-6)
                occluded=valid&(z-depth>BAND_SCALE*sref/ndot)
                counters['occluded_observations']+=int(occluded.sum())
                counters['incompatible_surface_layers']+=int((valid&same_side&~layer).sum())
                accepted=valid&same_side&compatible&layer&~occluded&torch.isfinite(signed)&(signed.abs()<=BAND_SCALE*sref)
                ww=torch.where(accepted,w*cosine.square(),torch.zeros_like(w))
                # Each source's normal gives its local tangent residual, while
                # the fixed reference determines a coherent sign chart.
                acc+=torch.where(accepted,signed,torch.zeros_like(signed))*ww;total+=ww
                rgb+=color*ww[:,None];nsum+=n*ww[:,None]
                counters['accepted_observations']+=int(accepted.sum())
            good=total>1e-8;denom=total.clamp_min(1e-8)
            values=(acc/denom).cpu().numpy();valid_np=good.cpu().numpy()
            field[start:end][valid_np]=values[valid_np];support[start:end]=valid_np
            weight_out[start:end]=total.cpu().numpy()
            color_np=(rgb/denom[:,None]).clamp(0,255).round().to(torch.uint8).cpu().numpy();colors[start:end]=color_np
            normal_np=(nsum/torch.linalg.norm(nsum,dim=-1,keepdim=True).clamp_min(1e-8)).cpu().numpy()
            normals[start:end][valid_np]=normal_np[valid_np]
    return dict(field=field,support=support,weights=weight_out,normals=normals,colors=colors,
        reference_points=pr,reference_normals=nr,reference_scales=sr,
        diagnostics=dict(queries=count,supported_queries=int(support.sum()),all_training_views_considered=len(prepared),
            field_kind='Supported fixed-chart local point-to-plane residual; NOT global signed distance or occupancy',
            unknown_values_are_nan=True,normal_band_footprints=BAND_SCALE,layer_band_footprints=LAYER_SCALE,
            chart_assigned_by_nearest_anchor=missing,**counters))


def _empty_mesh(diagnostics):
    return dict(vertices=np.empty((0,3),dtype='float32'),faces=np.empty((0,3),dtype='int64'),
        crossing_edges=np.empty((0,2),dtype='int64'),edge_reference_normals=np.empty((0,3),dtype='float32'),
        edge_reference_points=np.empty((0,3),dtype='float32'),edge_reference_scales=np.empty(0,dtype='float32'),
        diagnostics=diagnostics)


def marching_tetrahedra(vertices,cells,field_result,anchors=None,chunk_tets=200000):
    """Chart-safe NumPy marching tetrahedra with unique crossing-edge vertices.

    All four corners must have observations and mutually compatible reference
    charts. Normal separation between reference surfaces is limited by source
    footprint, so adjacent thin layers cannot produce a connecting fake sheet.
    This conservative test can leave open gaps, which coverage scoring retains.
    """
    vertices=np.asarray(vertices,dtype='float32');cells=np.asarray(cells,dtype='int64')
    if vertices.ndim!=2 or vertices.shape[1]!=3 or cells.ndim!=2 or cells.shape[1]!=4:raise ValueError('N,3 vertices and M,4 tetrahedra required')
    if len(cells) and (cells.min()<0 or cells.max()>=len(vertices)):raise ValueError('Tetrahedron index out of bounds')
    field=np.asarray(field_result['field']);known=np.asarray(field_result['support'],dtype=bool)&np.isfinite(field)
    nr=np.asarray(field_result['reference_normals']);pr=np.asarray(field_result['reference_points']);sr=np.asarray(field_result['reference_scales'])
    kept=[];stats=dict(input_tetrahedra=len(cells),unknown_tetrahedra=0,incompatible_normal_tetrahedra=0,incompatible_layer_tetrahedra=0,
        crossing_tetrahedra=0,not_global_sdf=True,global_closed_surface_guaranteed=False)
    for start in range(0,len(cells),chunk_tets):
        tt=cells[start:start+chunk_tets];valid=known[tt].all(1);stats['unknown_tetrahedra']+=int((~valid).sum())
        normal_ok=np.ones(len(tt),dtype=bool);layer_ok=normal_ok.copy()
        for a,b in EDGES:
            ia,ib=tt[:,a],tt[:,b];normal_ok&=(nr[ia]*nr[ib]).sum(1)>=NORMAL_COS
            displacement=pr[ib]-pr[ia]
            distance=np.maximum(np.abs((displacement*nr[ia]).sum(1)),np.abs((displacement*nr[ib]).sum(1)))
            layer_ok&=distance<=LAYER_SCALE*np.minimum(sr[ia],sr[ib])+1e-8
        stats['incompatible_normal_tetrahedra']+=int((valid&~normal_ok).sum())
        stats['incompatible_layer_tetrahedra']+=int((valid&normal_ok&~layer_ok).sum())
        signs=field[tt]>0;crossing=signs.any(1)&~signs.all(1)
        choose=valid&normal_ok&layer_ok&crossing
        kept.append(tt[choose]);stats['crossing_tetrahedra']+=int(choose.sum())
    tet=np.concatenate(kept) if kept else np.empty((0,4),dtype='int64')
    if not len(tet):return _empty_mesh(stats)
    all_edges=np.sort(tet[:,EDGES].reshape(-1,2),axis=1)
    unique,inverse=np.unique(all_edges,axis=0,return_inverse=True)
    crosses=(field[unique[:,0]]>0)!=(field[unique[:,1]]>0)
    indices=np.full(len(unique),-1,dtype='int64');indices[crosses]=np.arange(crosses.sum())
    local=indices[inverse].reshape(-1,6);edges=unique[crosses]
    a,b=edges[:,0],edges[:,1];fa,fb=field[a],field[b]
    fraction=fa/(fa-fb);positions=vertices[a]+fraction[:,None]*(vertices[b]-vertices[a])
    en=nr[a]+nr[b];en/=np.linalg.norm(en,axis=1,keepdims=True)
    er=pr[a]+fraction[:,None]*(pr[b]-pr[a]);es=np.minimum(sr[a],sr[b])
    # The six local edge IDs fix connectivity for the 16 tetrahedral sign
    # patterns. Final winding is oriented by the compatible chart normals.
    table=[[],[1,0,2],[4,0,3],[1,4,2,1,3,4],[3,1,5],[2,3,0,2,5,3],
        [1,4,0,1,5,4],[4,2,5],[4,5,2],[4,1,0,4,5,1],[3,2,0,3,5,2],
        [1,3,5],[4,1,2,4,3,1],[3,0,4],[2,0,1],[]]
    cases=((field[tet]>0)*np.array([1,2,4,8])).sum(1);faces=[]
    for case in range(1,15):
        selected=local[cases==case]
        if len(selected):faces.append(selected[:,table[case]].reshape(-1,3))
    faces=np.concatenate(faces)
    # Different crossing edges may meet an exact zero corner. Drop only truly
    # zero-area triangles, not small valid faces or components.
    cross=np.cross(positions[faces[:,1]]-positions[faces[:,0]],positions[faces[:,2]]-positions[faces[:,0]])
    good=np.linalg.norm(cross,axis=1)>0;stats['zero_area_faces_removed']=int((~good).sum());faces=faces[good];cross=cross[good]
    flip=(cross*en[faces].sum(1)).sum(1)<0;faces[flip]=faces[flip][:,[0,2,1]]
    stats.update(output_vertices=len(positions),output_faces=len(faces),unique_crossing_edges=len(edges))
    return dict(vertices=positions.astype('float32'),faces=faces,crossing_edges=edges,
        edge_reference_normals=en.astype('float32'),edge_reference_points=er.astype('float32'),
        edge_reference_scales=es.astype('float32'),diagnostics=stats)


def refine_crossing_edges(vertices,mesh,anchors,prepared,device='cpu',iterations=6,chunk_points=32768):
    """Fixed-chart bracketed bisection with quarter-point multi-root screening.

    Returned valid masks let callers remove faces touching unsupported roots.
    A finite set of samples cannot prove absence of sub-footprint multiple roots;
    the receipt explicitly records this limitation rather than claiming safety.
    """
    vertices=np.asarray(vertices,dtype='float32');edges=np.asarray(mesh['crossing_edges'],dtype='int64')
    left=vertices[edges[:,0]].copy();right=vertices[edges[:,1]].copy();count=len(edges)
    refs=dict(reference_normals=mesh['edge_reference_normals'],reference_points=mesh['edge_reference_points'],reference_scales=mesh['edge_reference_scales'])
    def evaluate(q):return query_field(q,anchors,prepared,device=device,chunk_points=chunk_points,**refs)
    first=evaluate(left);last=evaluate(right);fl=first['field'];fr=last['field']
    valid=first['support']&last['support']&((fl>0)!=(fr>0))
    sampled=[fl]
    for fraction in [.25,.5,.75]:
        row=evaluate(left+fraction*(right-left));valid&=row['support'];sampled.append(row['field'])
    sampled.append(fr);signs=np.stack(sampled,axis=1)>0
    changes=(signs[:,1:]!=signs[:,:-1]).sum(1);multiple=changes!=1;valid&=~multiple
    for step in range(iterations):
        mid=(left+right)*.5;row=evaluate(mid);valid&=row['support']
        same=(row['field']>0)==(fl>0)
        update=valid&same;left[update]=mid[update];fl[update]=row['field'][update]
        update=valid&~same;right[update]=mid[update];fr[update]=row['field'][update]
    positions=(left+right)*.5;row=evaluate(positions);valid&=row['support']
    return dict(vertices=positions,valid=valid,support=valid.copy(),normals=row['normals'],colors=row['colors'],
        diagnostics=dict(edges=count,valid_edges=int(valid.sum()),sampled_multiple_or_missing_root_edges=int(multiple.sum()),
            iterations=iterations,fixed_chart_during_refinement=True,unknown_edges_rejected=True,
            sub_sample_multiple_roots_excluded_by_proof=False,root_screen_samples=5))


def refine_edges(left,right,anchors,prepared,device='cpu',reference_points=None,
                 reference_normals=None,reference_scales=None,iterations=6,chunk_points=32768):
    """Shard-friendly wrapper; one fixed reference chart per left/right edge."""
    left=np.asarray(left,dtype='float32');right=np.asarray(right,dtype='float32')
    if left.shape!=right.shape or left.ndim!=2 or left.shape[1]!=3:raise ValueError('Matching N,3 bracket endpoints required')
    count=len(left)
    if any(value is None for value in [reference_points,reference_normals,reference_scales]):
        raise ValueError('Fixed reference points/normals/scales are mandatory for root refinement')
    mesh=dict(crossing_edges=np.column_stack([np.arange(count),np.arange(count)+count]),
        edge_reference_points=np.asarray(reference_points),edge_reference_normals=np.asarray(reference_normals),
        edge_reference_scales=np.asarray(reference_scales).reshape(-1))
    return refine_crossing_edges(np.concatenate([left,right]),mesh,anchors,prepared,device=device,
        iterations=iterations,chunk_points=chunk_points)
