"""Mass-conserving pixel surfels. Quality is continuous evidence, never a point veto.

The finite stencil/kNN model families are implementation choices, not user
quality thresholds. Models remain anchored at the original completed depth.
"""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.special import logsumexp

from support_geometry import DensityError, backproject, project

ALGORITHM = "mass_adaptive"


def load_config(path):
    import yaml
    cfg = yaml.safe_load((Path(__file__).parent / "configs/support_mass.yaml").read_text())
    given = yaml.safe_load(Path(path).read_text())
    if not isinstance(given, dict) or set(given)-set(cfg):
        raise ValueError("mass_adaptive accepts only algorithm/seed/batch_size/memory_budget_mb/debug; no quality thresholds")
    cfg.update(given)
    if cfg["algorithm"] != ALGORITHM or not isinstance(cfg["debug"], bool):
        raise ValueError("invalid mass algorithm/debug configuration")
    for key in ("seed", "batch_size", "memory_budget_mb"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key=="seed" else 1):
            raise ValueError(f"invalid mass {key}")
    return cfg


def noise_scale(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise DensityError("no finite values for noise estimation", {"reason":"geometry_unresolved"})
    # Round-off floor scales with the data, never a fixed world distance.
    floor = max(np.finfo(float).tiny, np.finfo(np.float64).eps * float(np.max(np.abs(finite))))
    return max(float(np.median(np.abs(finite-np.median(finite)))), floor)


def log_ratio(a,b):
    # Cast before flooring: float32 arrays cannot represent float64 tiny.
    tiny=np.finfo(float).tiny
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    # Invalid observations are masked by callers, but must not generate NaN
    # through 0 * NaN before that mask is applied.
    a=np.where(np.isfinite(a)&(a>0),a,tiny)
    b=np.where(np.isfinite(b)&(b>0),b,tiny)
    return np.log(a)-np.log(b)


def soft_weight(residual,scale):
    """Stable Cauchy weight even with zero measured noise (machine floor)."""
    return np.square(scale/np.hypot(residual,scale))


def soft_score(residual,scale):
    return 2*(np.log(np.hypot(residual,scale))-np.log(scale))


def surfel_points(uv, center_depth, slope, offset, cam):
    inv = 1/center_depth + np.sum(slope*offset, axis=-1)
    z = np.divide(1., inv, out=np.full_like(inv,np.nan), where=inv>0)
    pixel = uv+offset
    ray = np.column_stack(((pixel[:,0]-cam["w"]/2)/cam["fx"],
                           (pixel[:,1]-cam["h"]/2)/cam["fy"], np.ones(len(uv))))
    pc = ray*z[:,None]
    return pc@cam["c2w"][:3,:3].T+cam["center"]


def surfel_jacobian(uv, depth, slope, offset, cam):
    inv = 1/depth + (slope*offset).sum(-1)
    z = np.divide(1.,inv,out=np.full_like(inv,np.nan),where=inv>0)
    ray = np.column_stack(((uv[:,0]+offset[:,0]-cam["w"]/2)/cam["fx"],
                           (uv[:,1]+offset[:,1]-cam["h"]/2)/cam["fy"],np.ones(len(uv))))
    dx = -ray*(z*z*slope[:,0])[:,None]; dx[:,0] += z/cam["fx"]
    dy = -ray*(z*z*slope[:,1])[:,None]; dy[:,1] += z/cam["fy"]
    linear=cam["c2w"][:3,:3].T
    return np.linalg.norm(np.cross(dx@linear,dy@linear),axis=1)


def fit_surfels(depth, valid, batch_size, *, query_mask=None, with_models=False):
    """PRESS-selected inverse-depth slopes over full/one-sided multiscale stencils.

    All hypotheses pass through the center. No mixing of foreground/background
    depths. Predictive scores select a hypothesis; no score rejects a pixel.
    """
    h,w=depth.shape
    ids=np.flatnonzero(valid if query_mask is None else valid & query_mask)
    uv=np.column_stack((ids%w,ids//w)).astype(float)
    inv=np.divide(1.,depth,out=np.zeros_like(depth,dtype=float),where=valid)
    differences=[]
    for a,b in ((inv[:,1:],inv[:,:-1]),(inv[1:],inv[:-1])):
        differences.append((a-b)[(a>0)&(b>0)])
    sigma=noise_scale(np.concatenate(differences)) if any(v.size for v in differences) else noise_scale(inv[valid])
    slopes=np.zeros((len(ids),2)); uncertainty=np.zeros(len(ids)); choice=np.zeros(len(ids),np.int32)
    # Versioned model family, not configurable rejection radii.
    models=[(r,side) for r in (1,2,4) for side in ("full","left","right","up","down")]
    model_slopes=np.zeros((len(ids),len(models)+1,2)) if with_models else None
    model_scores=np.full((len(ids),len(models)+1),np.inf) if with_models else None
    for start in range(0,len(ids),batch_size):
        end=min(start+batch_size,len(ids)); centers=uv[start:end].astype(int)
        y0=inv.ravel()[ids[start:end]]
        predictions=[np.zeros((end-start,2))]; scores=[]
        for radius,side in models:
            delta=np.array([(x,y) for y in (-radius,0,radius) for x in (-radius,0,radius)
                if (x or y) and (side=="full" or (side=="left" and x<=0) or
                (side=="right" and x>=0) or (side=="up" and y<=0) or (side=="down" and y>=0))],float)
            pixels=centers[:,None,:]+delta[None].astype(int)
            in_image=(pixels[:,:,0]>=0)&(pixels[:,:,0]<w)&(pixels[:,:,1]>=0)&(pixels[:,:,1]<h)
            px=np.clip(pixels[:,:,0],0,w-1); py=np.clip(pixels[:,:,1],0,h-1)
            ok=in_image & valid[py,px]
            target=inv[py,px]-y0[:,None]
            # Smooth residual reliability prevents jumps from dominating a fit.
            weight=ok*soft_weight(target,sigma)
            gram=np.einsum('nk,ki,kj->nij',weight,delta,delta)
            rhs=np.einsum('nk,nk,ki->ni',weight,target,delta)
            inverse=np.linalg.pinv(gram,hermitian=True)
            beta=np.einsum('nij,nj->ni',inverse,rhs)
            leverage=np.einsum('ki,nij,kj->nk',delta,inverse,delta)*weight
            residual=target-beta@delta.T
            press=residual/np.maximum(1-leverage,np.sqrt(np.finfo(float).eps))
            score=np.sum(ok*soft_score(press,sigma),axis=1)/np.maximum(ok.sum(1),1)
            # Prediction variance prevents saturated tiny stencils from winning.
            score+=np.trace(inverse,axis1=1,axis2=2)/(radius**2)
            score[ok.sum(1)<2]=np.inf  # two slopes cannot be identified
            positive=(y0 > np.abs(beta).sum(1)*.5)
            score[~positive]=np.inf  # plane would cross infinity inside its pixel
            predictions.append(beta); scores.append(score)
        score=np.stack(scores,1)
        best=np.argmin(score,axis=1)
        selected=np.stack(predictions[1:],1)[np.arange(end-start),best]
        fallback=~np.isfinite(score.min(1))
        selected[fallback]=0
        slopes[start:end]=selected
        choice[start:end]=np.where(fallback,-1,best)
        uncertainty[start:end]=np.where(fallback,1.,1-np.exp(-np.minimum(score.min(1),700)))
        if with_models:
            model_slopes[start:end,1:]=np.stack(predictions[1:],1)
            model_scores[start:end,1:]=score
            # Fronto-parallel hypothesis evaluated on the same immediate stencil.
            delta=np.array([[-1,0],[1,0],[0,-1],[0,1]])
            pix=centers[:,None]+delta
            ok=(pix[:,:,0]>=0)&(pix[:,:,0]<w)&(pix[:,:,1]>=0)&(pix[:,:,1]<h)
            px=np.clip(pix[:,:,0],0,w-1);py=np.clip(pix[:,:,1],0,h-1)
            ok &= valid[py,px]
            residual=inv[py,px]-y0[:,None]
            model_scores[start:end,0]=(soft_score(residual,sigma)*ok).sum(1)/np.maximum(ok.sum(1),1)
    result=(ids,uv,slopes,uncertainty,choice)
    return (*result,model_slopes,model_scores) if with_models else result


def reference_intensity(points,batch_size):
    """Multiscale radial Poisson held-out likelihood, in surface-area units.

    Queries include all centers, not just a thin ring; prevents ring truncation.
    World distances approximate intrinsic distances locally. Sharp folds remain
    an uncertainty of this estimator, not a promised surface reconstruction.
    """
    n=len(points)
    if n<3:
        raise DensityError("no_reference: fewer than three distinct background centers",{"reason":"no_reference"})
    tree=cKDTree(points)
    ks=[k for k in (2,4,8,16,32) if 2*k<n]
    if not ks: ks=[1]
    rho=np.empty(n); var=np.empty(n)
    for start in range(0,n,batch_size):
        end=min(n,start+batch_size)
        distances=tree.query(points[start:end],k=min(2*max(ks)+1,n))[0][:,1:]
        estimates=[];scores=[]
        for k in ks:
            radius=distances[:,k-1]
            area=np.pi*radius**2
            rate=k/area
            outer=np.pi*distances[:,min(2*k,distances.shape[1])-1]**2-area
            count=min(2*k,distances.shape[1])-k
            # Proper count likelihood on a held-out outer shell; auto k weights.
            prediction=rate*outer
            score=prediction-count*np.log(np.maximum(prediction,np.finfo(float).tiny))
            estimates.append(np.log(rate));scores.append(score/max(count,1))
        lograte=np.stack(estimates,1); score=np.stack(scores,1)
        prob=np.exp(-score-logsumexp(-score,axis=1,keepdims=True))
        mean=(prob*lograte).sum(1)
        rho[start:end]=np.exp(mean);var[start:end]=(prob*(lograte-mean[:,None])**2).sum(1)
    if not np.isfinite(rho).all():
        raise DensityError("nonfinite reference intensity",{"reason":"numerical_failure"})
    return rho,var


def reference_field(frames,seed,ref,depth,valid,cfg):
    points,unique=np.unique(ref['xyz'],axis=0,return_index=True)
    if not len(points) or not np.isfinite(points).all():
        raise DensityError('no_reference: invalid/empty retained centers',{'reason':'no_reference'})
    # Each original center gets ONE weight; additional views improve reliability,
    # not the number of centers. No opacity/alpha/depth-residual cutoff.
    reliability=np.zeros(len(points)); seed_pixel=None; seed_weight=None
    for i,frame in enumerate(frames):
        _,pix,z,inside=project(points,frame['camera']);x,y=pix.T
        d=frame['removed'][y,x]
        known=inside & ~frame['mask'][y,x] & np.isfinite(d)&(d>0)
        residual=log_ratio(z,d)
        sigma=noise_scale(residual[known]) if known.any() else 1.
        alpha=np.nan_to_num(np.clip(frame['alpha'][y,x],0,1),nan=0.)
        weights=known*alpha*ref['opacity'][unique]*soft_weight(residual,sigma)
        reliability=np.maximum(reliability,weights)
        if i==seed:
            seed_pixel=pix;seed_weight=weights
    # Density estimation uses every unique retained center, observation weights
    # only set the precision of its density anchor, not a fractional point count.
    rho,var=reference_intensity(points,cfg['batch_size'])
    weight=seed_weight*np.sqrt(reliability)/(1+var)
    x,y=seed_pixel.T
    weight*=valid[y,x]
    h,w=depth.shape; flat=y*w+x
    mass=np.bincount(flat,weights=weight,minlength=h*w)
    logsum=np.bincount(flat,weights=weight*np.log(rho),minlength=h*w)
    if not np.any(mass>0):
        raise DensityError('no_reference: no known visible reference observation',{'reason':'no_reference'})
    anchor=np.divide(logsum,mass,out=np.zeros_like(logsum),where=mass>0)
    ids=np.flatnonzero(valid); mapping=np.full(h*w,-1,np.int64);mapping[ids]=np.arange(len(ids))
    # Continuous inverse-depth graph. No threshold or connected-component cut.
    inv=np.divide(1.,depth,out=np.zeros_like(depth,dtype=float),where=valid)
    a=[];b=[];res=[]
    grid=np.arange(h*w).reshape(h,w)
    for left,right in (((slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                       ((slice(None,-1),slice(None)),(slice(1,None),slice(None)))):
        ok=valid[left]&valid[right]
        a.append(mapping[grid[left][ok]]);b.append(mapping[grid[right][ok]])
        res.append((inv[left]-inv[right])[ok])
    a=np.concatenate(a);b=np.concatenate(b);res=np.concatenate(res)
    sigma=noise_scale(res) if len(res) else 1.
    conductance=soft_weight(res,sigma)
    adjacency=coo_matrix((np.r_[conductance,conductance],(np.r_[a,b],np.r_[b,a])),shape=(len(ids),len(ids))).tocsr()
    precision=mass[ids]/(1+mass[ids])
    # Floating regularization only; unobserved disconnected components are an
    # identifiability failure, not filled from an unrelated global average.
    from scipy.sparse.csgraph import connected_components
    _,labels=connected_components(adjacency,directed=False)
    totals=np.bincount(labels,weights=precision)
    if np.any(totals==0):
        raise DensityError('no_reference: a valid depth component has no observations',{'reason':'no_reference'})
    lap=diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    solution=spsolve(lap+diags(precision),precision*anchor[ids])
    if not np.isfinite(solution).all():
        raise DensityError('density field solve failed',{'reason':'numerical_failure'})
    field=np.zeros(h*w);field[ids]=np.exp(solution)
    return field.reshape(h,w),dict(reference_centers=len(points),observed_centers=int((weight>0).sum()),
        density_quantiles=np.quantile(rho,[.1,.5,.9]).tolist()),(points,reliability)


def view_uncertainty(points,frames,seed):
    """Soft censored residual evidence. Occluded/duplicate views never veto points."""
    total=np.zeros(len(points));norm=np.zeros(len(points))
    center=points.mean(0) if len(points) else np.zeros(3)
    bearings=np.array([center-f['camera']['center'] for f in frames])
    bearings/=np.maximum(np.linalg.norm(bearings,axis=1,keepdims=True),np.finfo(float).tiny)
    # Subtraction gives exactly zero for repeated cameras; 1-dot can turn
    # round-off into spurious independent evidence in a duplicate-view set.
    pair=.5*np.sum((bearings[:,None]-bearings[None,:])**2,axis=-1)
    scale=noise_scale(pair.ravel())
    redundancy=soft_weight(pair,scale).sum(1)
    for i,frame in enumerate(frames):
        if i==seed:continue
        _,pix,z,inside=project(points,frame['camera']);x,y=pix.T
        known=~frame['mask'][y,x]
        d=np.where(known,frame['removed'][y,x],frame['depth'][y,x])
        valid=inside&np.isfinite(d)&(d>0)
        r=log_ratio(z,d)
        sigma=noise_scale(r[valid]) if valid.any() else 1.
        visibility=soft_weight(np.maximum(r,0),sigma)
        novelty=pair[i,seed]/(pair[i,seed]+scale)
        alpha=np.nan_to_num(np.clip(frame['alpha'][y,x],0,1),nan=0.)
        weight=valid*visibility*novelty/redundancy[i]*np.where(known,alpha,1.)
        total+=weight*soft_score(r,sigma);norm+=weight
    return np.divide(total,norm,out=np.ones_like(total),where=norm>0)


def allocate(mass,seed):
    mass=np.asarray(mass,float)
    if not np.isfinite(mass).all() or (mass<0).any():raise ValueError('invalid quota mass')
    n=int(np.rint(mass.sum()))
    if n==0:return np.zeros(len(mass),np.int64)
    scaled=mass*(n/mass.sum())
    phase=np.random.default_rng(seed).random()
    edges=np.floor(np.r_[0.,np.cumsum(scaled)]+phase).astype(np.int64)
    return np.diff(edges)


def sample_mass(frames,seed,ref,cfg,gate):
    frame=frames[seed];cam=frame['camera'];mask=frame['mask']
    depth=np.where(mask,frame['depth'],frame['removed']).astype(float)
    valid=np.isfinite(depth)&(depth>0)
    if not mask.any():raise DensityError('empty seed mask',{'reason':'invalid_input'})
    if np.any(mask & ~valid):
        raise DensityError('hole has undefined depth',{'reason':'geometry_unresolved','invalid_hole_pixels':int((mask&~valid).sum())})
    # Conservative workspace estimate; resource budget never truncates N_new.
    budget=cfg['memory_budget_mb']*1024**2
    if depth.size*1200 + len(ref['xyz'])*500 > budget:
        raise DensityError('resource_limited: increase memory_budget_mb',{'reason':'resource_limited'})
    ids,uv,slope,uncertainty,choice,models,scores=fit_surfels(
        depth,valid,cfg['batch_size'],query_mask=mask,with_models=True)
    rho,ref_report,background=reference_field(frames,seed,ref,depth,valid,cfg)
    d=depth.ravel()[ids]
    # Independent view evidence selects a hypothesis, never averages layer depths
    # or rejects a pixel. Invalid fitted hypotheses are represented by inf scores.
    for model in range(models.shape[1]):
        beta=models[:,model]
        evidence=np.zeros(len(ids))
        for delta in ((-.25,-.25),(.25,.25)):
            probes=surfel_points(uv,d,beta,np.broadcast_to(delta,uv.shape),cam)
            finite=np.isfinite(probes).all(1)
            if finite.any(): evidence[finite]+=view_uncertainty(probes[finite],frames,seed)/2
            evidence[~finite]=np.inf
        scores[:,model]+=evidence
    choice=np.argmin(scores,axis=1)
    slope=models[np.arange(len(ids)),choice]
    posterior=np.exp(-scores-logsumexp(-scores,axis=1,keepdims=True))
    uncertainty=1-posterior.max(1)
    # Equal UV integration bins also define a safe rejection-free sampling atlas.
    # The quadrature order is numerical accuracy, not a surface quality gate.
    axis=(np.arange(4)+.5)/4-.5
    grid=np.array([(x,y) for y in axis for x in axis])
    areas=np.zeros((len(ids),len(grid))); allowed=np.zeros_like(areas,bool)
    full_area=np.zeros(len(ids))
    for j,offset in enumerate(grid):
        off=np.broadcast_to(offset,uv.shape)
        p=surfel_points(uv,d,slope,off,cam)
        good=np.isfinite(p).all(1); ok=np.zeros(len(p),bool);ok[good]=gate(p[good])
        area=surfel_jacobian(uv,d,slope,off,cam)/len(grid)
        full_area+=area
        areas[:,j]=np.where(ok,area,0);allowed[:,j]=ok
    area=areas.sum(1)
    if not np.isfinite(area).all():raise DensityError('nonfinite surfel area',{'reason':'numerical_failure'})
    if not np.any(area>0):
        raise DensityError('geometry_unresolved: no seed surface inside original gate',{'reason':'geometry_unresolved'})
    # Background contribution normalized per original center, never per view.
    pts,reliability=background
    _,pix,z,inside=project(pts,cam);x,y=pix.T
    known_depth=frame['depth'][y,x]
    residual=log_ratio(z,known_depth)
    eligible=inside&mask[y,x]&np.isfinite(known_depth)&(known_depth>0)
    sigma=noise_scale(residual[eligible]) if eligible.any() else 1.
    contribution=eligible*reliability*soft_weight(residual,sigma)
    present=np.bincount(y*cam['w']+x,weights=contribution,minlength=depth.size)[ids]
    target=rho.ravel()[ids]*area; mass=np.maximum(target-present,0)
    needed=int(np.rint(mass.sum()))
    if needed*500+depth.size*1200+len(pts)*500 > budget:
        raise DensityError(f'resource_limited: {needed} points required',{'reason':'resource_limited','required_points':needed})
    quota=allocate(mass,cfg['seed'])
    owner=np.repeat(np.arange(len(ids)),quota)
    offsets=np.empty((needed,2));rng=np.random.default_rng(cfg['seed'])
    cursor=0
    for i in np.flatnonzero(quota):
        count=quota[i]; cdf=np.cumsum(areas[i])/area[i]
        bins=np.searchsorted(cdf,(np.arange(count)+rng.random())/count,side='right')
        # Stratification inside each integration bin, not 3D jitter.
        for cell in np.unique(bins):
            positions=np.flatnonzero(bins==cell); n=len(positions)
            side=int(np.ceil(np.sqrt(n)));index=np.arange(n)
            jitter=np.column_stack(((index%side+rng.random(n))/side,(index//side+rng.random(n))/side))
            offsets[cursor+positions]=grid[cell]+(jitter-.5)/4
        cursor+=count
    points=surfel_points(uv[owner],d[owner],slope[owner],offsets,cam)
    # Gate boundary integration is approximate. Contract toward a feasible
    # quadrature node in the SAME pixel; keep the quota and distinct offsets.
    ok=gate(points) if needed else np.empty(0,bool)
    bad=np.flatnonzero(~ok)
    anchors=np.zeros_like(offsets)
    for j in bad:
        nodes=grid[allowed[owner[j]]]
        anchors[j]=nodes[np.argmin(np.sum((nodes-offsets[j])**2,axis=1))]
    for _ in range(np.finfo(float).nmant+1):
        if not len(bad):break
        offsets[bad]=(offsets[bad]+anchors[bad])/2
        points[bad]=surfel_points(uv[owner[bad]],d[owner[bad]],slope[owner[bad]],offsets[bad],cam)
        bad=bad[~gate(points[bad])]
    if needed and (not np.isfinite(points).all() or not gate(points).all() or len(np.unique(points.astype(np.float32),axis=0))!=needed):
        raise DensityError('geometry_unresolved: gate/float32 layout unresolved',{'reason':'geometry_unresolved'})
    geometry=view_uncertainty(points,frames,seed) if needed else np.empty(0)
    # Pixel-owned colors cannot blend across foreground/background boundaries.
    rgb=frame['rgb'].reshape(-1,3)[ids[owner]]
    data=dict(xyz=points,rgb=rgb,seed_uv=uv[owner]+offsets,spacing=1/np.sqrt(rho.ravel()[ids[owner]]),
        patch=np.zeros(needed,np.int32),quota_id=owner,pixel_id=ids[owner],geometry_uncertainty=geometry)
    fields=dict(pixel_id=ids,uv=uv,depth=d,slope=slope,area=area,full_area=full_area,rho=rho.ravel()[ids],
        geometry_uncertainty=uncertainty,model_choice=choice,gate_fraction=allowed.mean(1))
    quotas=dict(target=target,existing=present,mass=mass,count=quota,pixel_id=ids,
        surplus=np.maximum(present-target,0))
    report=dict(algorithm=ALGORITHM,raw_seed_points=int(mask.sum()),initialized_points=needed,
        target_mass=float(target.sum()),existing_mass=float(present.sum()),new_mass=float(mass.sum()),
        rounding_error=float(needed-mass.sum()),surface_area=float(area.sum()),
        full_hole_surface_area=float(full_area.sum()),surplus_mass=float(quotas['surplus'].sum()),
        gate_excluded_target_mass=float((rho.ravel()[ids]*np.maximum(full_area-area,0)).sum()),
        per_pixel_quota_residual_quantiles=np.quantile(quota-mass,[.1,.5,.9]).tolist(),
        valid_hole_pixels=len(ids),gate_empty_pixels=int((area==0).sum()),
        density_ratio=float((needed+np.minimum(present,target).sum())/target.sum()) if target.sum()>0 else None,
        geometry_uncertainty_quantiles=np.quantile(geometry,[.1,.5,.9]).tolist() if needed else [],
        reference=ref_report,status='complete',note='density_ratio measures quota fulfillment, not independent geometric quality.')
    return data,report,fields,quotas
