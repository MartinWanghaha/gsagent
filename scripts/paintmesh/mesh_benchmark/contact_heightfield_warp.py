"""Continuous monotone generated-object contact warp; independent experiment."""
from pathlib import Path
import time
import numpy as np
from .contact_surface_refine import DEFAULTS, transform, support_envelope, _metrics


def refine_contact_surface_heightfield(canonical_vertices,faces,rigid_pose,
        environment_vertices,environment_faces,evidence=None,config=None,*,up=None):
    """Global continuous monotone underside map, without local pin/freeze cuts.

    A smooth two-dimensional height field comes from the actual support rays.
    All object vertices use the *same* vertical monotone function. At the
    continuous field level this map is injective; the returned straight-triangle
    discretization must still pass exact self-intersection checks. No relative
    original-normal cutoff is imposed: a normal rotating by >90 degrees under a
    valid strong deformation is not sufficient evidence of a topological flip.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
    from scipy.interpolate import RegularGridInterpolator
    cfg={**DEFAULTS,**(config or {})}
    cfg.setdefault('heightfield_grid_size',192)
    cfg.setdefault('heightfield_smoothing_fraction',.025)
    cfg.setdefault('heightfield_tau_fraction',.02)
    cfg.setdefault('heightfield_conservative_quantile',1.)
    cfg.setdefault('heightfield_conservative_mode','global')
    cfg.setdefault('heightfield_local_correction_iterations',10)
    started=time.time();vertices=np.asarray(canonical_vertices,dtype=float)
    faces=np.asarray(faces,dtype=np.int64);pose=np.asarray(rigid_pose,dtype=float)
    world=transform(vertices,pose);diameter=float(np.linalg.norm(np.ptp(world,axis=0)))
    if up is None:
        if evidence is None:raise ValueError('Observed up direction required')
        with np.load(Path(evidence)/'anchors.npz') as f:up=f['up']
    up=np.asarray(up,dtype=float);up/=np.linalg.norm(up)
    axis=np.array([1.,0,0]) if abs(up[0])<.9 else np.array([0.,1,0])
    ax=axis-up*(axis@up);ax/=np.linalg.norm(ax);ay=np.cross(up,ax)
    horizontal=np.c_[world@ax,world@ay];height=world@up
    floor,valid=support_envelope(world,environment_vertices,environment_faces,up,diameter)
    if not valid.any():raise ValueError('No support rays intersect current environment')
    n=int(cfg['heightfield_grid_size'])
    if n<16:raise ValueError('heightfield_grid_size must be >=16')
    lower=horizontal.min(0)-.02*diameter;upper=horizontal.max(0)+.02*diameter
    grid=(horizontal-lower)/(upper-lower)*(n-1)
    ij=np.clip(np.floor(grid).astype(int),0,n-1)
    heights=np.full((n,n),-np.inf)
    # A conservative per-cell maximum comes from all original vertex rays.
    np.maximum.at(heights,(ij[valid,0],ij[valid,1]),floor[valid])
    missing=~np.isfinite(heights)
    nearest=distance_transform_edt(missing,return_distances=False,return_indices=True)
    heights[missing]=heights[tuple(nearest[:,missing])]
    spacing=float(np.min((upper-lower)/(n-1)))
    sigma=max(cfg['heightfield_smoothing_fraction']*diameter/spacing,.5)
    smooth=gaussian_filter(heights,sigma=sigma,mode='nearest')
    axes=(np.linspace(lower[0],upper[0],n),np.linspace(lower[1],upper[1],n))
    interpolator=RegularGridInterpolator(axes,smooth,bounds_error=False,fill_value=None)
    sampled=interpolator(horizontal)
    missing_height=np.maximum(floor[valid]-sampled[valid],0)
    correction=0.;local_corrections=[]
    if cfg['heightfield_conservative_mode']=='local':
        # Raise only neighborhoods that under-estimate an actual ray support.
        # Splat to all four bilinear corners: their interpolation at the source
        # point cannot underweight the requested correction. A positive local
        # Gaussian extension keeps the field continuous without a worst-case
        # offset imposed on unrelated parts of the object.
        baseij=np.clip(np.floor(grid).astype(int),0,n-2)
        for iteration in range(int(cfg['heightfield_local_correction_iterations'])):
            residual=np.maximum(floor[valid]-sampled[valid],0)
            peak=float(residual.max()) if len(residual) else 0.
            if peak<diameter*1e-7:break
            impulse=np.zeros((n,n),dtype=float)
            for dx,dy in [(0,0),(0,1),(1,0),(1,1)]:
                np.maximum.at(impulse,(baseij[valid,0]+dx,baseij[valid,1]+dy),residual)
            positive=maximum_filter(impulse,size=3,mode='nearest')
            positive=gaussian_filter(positive,sigma=max(.5,sigma*.5),mode='nearest')
            # The raw bilinear splat prevents loss of strict support coverage;
            # smoothed positive neighbors provide a continuous transition.
            bump=np.maximum(positive,impulse)
            smooth+=bump
            sampled=RegularGridInterpolator(axes,smooth,bounds_error=False,fill_value=None)(horizontal)
            local_corrections.append(dict(iteration=iteration+1,maximum_residual_before=peak,
                grid_nodes_raised=int(np.sum(bump>0)),maximum_grid_raise=float(bump.max())))
        correction=float(np.max(smooth-heights))
    elif cfg['heightfield_conservative_mode']=='global':
        correction=float(np.quantile(missing_height,cfg['heightfield_conservative_quantile']))
        sampled+=correction
    else:raise ValueError('Unknown heightfield_conservative_mode')
    clearance=(cfg['clearance_fraction']+cfg['minimum_vertical_thickness_fraction'])*diameter
    lowest=float(np.max(sampled-height))
    tau=max(cfg['heightfield_tau_fraction']*diameter,
        max(lowest,0)/max(-np.log(cfg['minimum_vertical_slope']),1.))
    relative=height-sampled-clearance
    new_height=sampled+clearance+tau*np.logaddexp(0.,relative/tau)
    displacement=new_height-height
    candidate=world+displacement[:,None]*up
    corrected=transform(candidate,np.linalg.inv(pose))
    if np.asarray(canonical_vertices).dtype.kind=='f':corrected=corrected.astype(np.asarray(canonical_vertices).dtype)
    actual=transform(corrected,pose)
    before=_metrics(height-floor,valid,np.zeros(len(vertices),dtype=bool),diameter)
    after=_metrics(actual@up-floor,valid,np.zeros(len(vertices),dtype=bool),diameter)
    ratio=1.;degenerate=0
    for start in range(0,len(faces),131072):
        f=faces[start:start+131072];a=world[f];b=actual[f]
        la=np.linalg.norm(np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]),axis=1)
        lb=np.linalg.norm(np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]),axis=1)
        good=la>1e-15
        if good.any():ratio=min(ratio,float((lb[good]/la[good]).min()))
        degenerate+=int(np.sum(good&(lb<1e-12*diameter*diameter)))
    report=dict(complete=True,algorithm='Continuous 2D actual support-ray height field + global strictly monotone softplus vertical map',
        geometry_modified=bool(np.any(displacement>1e-12*diameter)),topology_unchanged=True,
        vertex_count=len(vertices),face_count=len(faces),environment_faces=len(environment_faces),
        vertices_moved=int(np.sum(displacement>1e-12*diameter)),pinned_vertices=0,
        maximum_displacement=float(displacement.max()),maximum_displacement_fraction=float(displacement.max()/diameter),
        before=before,after=after,heightfield_grid_size=n,heightfield_smoothing_sigma=sigma,
        conservative_height_correction_scene_units=correction,transition_width_scene_units=tau,
        conservative_height_correction_mode=cfg['heightfield_conservative_mode'],
        local_correction_history=local_corrections,
        support_envelope_residual_max=float(np.maximum(floor[valid]-sampled[valid],0).max()),
        theoretical_minimum_vertical_derivative=float(1/(1+np.exp(min(float(-relative.min()/tau),700)))),
        minimum_triangle_area_ratio=ratio,new_nearly_degenerate_faces=degenerate,
        local_freeze_used=False,observation_pin_used=False,
        original_asset_modified=False,mesh_simplified=False,colors_uv_materials_changed=False,
        collision_free_certified=False,complete_collision_recheck_required=True,self_intersection_recheck_required=True,
        native_observation_guard_required=True,config=cfg,up=up.tolist(),diameter=diameter,elapsed=time.time()-started,
        limitations=['Continuous monotonicity does not guarantee that discretized straight triangles have no self-intersections.',
                     'No observation pin is used because pin cuts can break vertical ordering; native observation guard must assess the final mesh.',
                     'A conservative high support outlier can raise the whole height field; reject excessive displacement or observation degradation.',
                     'The generated underside changes; it is not a rigid-only pose correction.'])
    return corrected,report
