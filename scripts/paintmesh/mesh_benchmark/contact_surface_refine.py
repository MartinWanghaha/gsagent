"""Occlusion-aware generated-object underside correction on actual triangles.

The environment is immutable and may be open. Rays along observed support up
measure a local height envelope. A monotone soft obstacle map moves penetrating
underside vertices up, with a continuous transition and positive vertical slope;
native observation-visible vertices are pinned. Triangle connectivity, UVs and
colors are not changed. Exact complete-mesh collision and self-intersection
checks remain publication requirements.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy import sparse


DEFAULTS=dict(clearance_fraction=.0003,minimum_vertical_thickness_fraction=.0001,
    transition_fraction=.008,minimum_vertical_slope=.002,
    maximum_displacement_fraction=.15,upper_pin_gap_fraction=.08,
    visibility_tolerance_fraction=.00015,minimum_face_normal_dot=.01,
    minimum_triangle_area_ratio=.002,elastic_iterations=30,
    elastic_strength=.15,available_neighborhood_fraction=.05,
    visibility_pin_mode='all_visible',silhouette_pin_width=2,
    local_safety_backoff_iterations=8,local_safety_freeze_iterations=32)


def transform(points,T):
    return np.asarray(points)@np.asarray(T)[:3,:3].T+np.asarray(T)[:3,3]


def support_envelope(vertices,environment_vertices,environment_faces,up,diameter):
    import open3d as o3d
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(environment_vertices,dtype='float32')),
        o3d.core.Tensor(np.asarray(environment_faces,dtype='uint32')))
    height=vertices@up
    start=float(np.max(height)+.15*diameter)
    origins=vertices+(start-height)[:,None]*up
    parts=[]
    for first in range(0,len(vertices),131072):
        origin=origins[first:first+131072]
        rr=np.c_[origin,np.broadcast_to(-up,origin.shape)].astype('float32')
        parts.append(scene.cast_rays(o3d.core.Tensor(rr),nthreads=4)['t_hit'].numpy())
    distances=np.concatenate(parts);valid=np.isfinite(distances)&(distances>0)
    floor=start-distances
    return floor,valid


def observation_visibility(vertices,faces,evidence,diameter,tolerance_fraction,
                           pin_mode='all_visible',silhouette_width=2):
    """Pin every vertex visible in any original object-mask observation."""
    import open3d as o3d
    evidence=Path(evidence);meta=json.loads((evidence/'observations.json').read_text())
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices,dtype='float32')),
        o3d.core.Tensor(np.asarray(faces,dtype='uint32')))
    pinned=np.zeros(len(vertices),dtype=bool);per_view=[]
    if pin_mode not in ['all_visible','silhouette','none']:
        raise ValueError('visibility_pin_mode must be all_visible, silhouette or none')
    for view in meta['views']:
        with np.load(evidence/'observations'/(Path(view['name']).stem+'.npz')) as data:
            K=data['K'];T=data['w2c'];mask=data['mask'].astype(bool)
        center=np.linalg.inv(T)[:3,3]
        p=vertices@T[:3,:3].T+T[:3,3];uv=p@K.T
        ij=np.floor(np.divide(uv[:,:2],p[:,2:3],out=np.full((len(vertices),2),-1.),where=p[:,2:3]>0)).astype('int64')
        inside=(p[:,2]>0)&(ij[:,0]>=0)&(ij[:,0]<mask.shape[1])&(ij[:,1]>=0)&(ij[:,1]<mask.shape[0])
        ids=np.flatnonzero(inside)
        if pin_mode=='silhouette':
            from scipy.ndimage import binary_erosion
            mask=mask & ~binary_erosion(mask,iterations=max(1,int(silhouette_width)))
        if pin_mode=='none':mask[:]=False
        ids=ids[mask[ij[ids,1],ij[ids,0]]]
        added=0
        for start in range(0,len(ids),131072):
            index=ids[start:start+131072]
            direction=vertices[index]-center;length=np.linalg.norm(direction,axis=1)
            direction/=length[:,None]
            rr=np.c_[np.broadcast_to(center,direction.shape),direction].astype('float32')
            distance=scene.cast_rays(o3d.core.Tensor(rr),nthreads=4)['t_hit'].numpy()
            visible=np.isfinite(distance)&(distance>=length-tolerance_fraction*diameter)
            pinned[index[visible]]=True;added+=int(visible.sum())
        per_view.append(dict(name=view['original'],visible_vertices=added))
    return pinned,per_view


def face_safety(original,changed,faces,cfg):
    min_dot=1.;min_ratio=1.;bad=np.zeros(len(faces),dtype=bool)
    for start in range(0,len(faces),131072):
        f=faces[start:start+131072];a=original[f];b=changed[f]
        na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
        la=np.linalg.norm(na,axis=1);lb=np.linalg.norm(nb,axis=1)
        valid=la>1e-15
        dot=np.divide(np.einsum('ij,ij->i',na,nb),la*lb,out=np.ones(len(la)),where=valid&(lb>0))
        ratio=np.divide(lb,la,out=np.ones(len(la)),where=valid)
        bb=valid&((dot<cfg['minimum_face_normal_dot'])|(ratio<cfg['minimum_triangle_area_ratio']))
        bad[start:start+len(f)]=bb
        if valid.any():min_dot=min(min_dot,float(dot[valid].min()));min_ratio=min(min_ratio,float(ratio[valid].min()))
    return bad,min_dot,min_ratio


def _metrics(gap,valid,pinned,diameter):
    v=gap[valid]/diameter;pen=np.maximum(-v,0)
    return dict(vertices_with_environment_ray=int(valid.sum()),penetrating_vertices=int((v<0).sum()),
        pinned_penetrating_vertices=int((valid&pinned&(gap<0)).sum()),
        maximum_penetration_fraction=float(pen.max()) if len(pen) else None,
        p99_penetration_fraction=float(np.quantile(pen,.99)) if len(pen) else None,
        mean_penetration_fraction=float(pen.mean()) if len(pen) else None)


def refine_contact_surface(canonical_vertices,faces,rigid_pose,environment_vertices,
                           environment_faces,evidence=None,config=None,*,up=None):
    """Return ``(corrected_canonical_vertices, report)`` with fixed topology.

    The original arrays/assets are untouched. The caller must store a new asset
    retaining all original non-position arrays, then verify against the complete
    environment and render all observation/test views before publishing.
    """
    started=time.time();cfg={**DEFAULTS,**(config or {})}
    vertices=np.asarray(canonical_vertices,dtype=float);faces=np.asarray(faces,dtype=np.int64)
    pose=np.asarray(rigid_pose,dtype=float);world=transform(vertices,pose)
    diameter=float(np.linalg.norm(np.ptp(world,axis=0)))
    if not diameter>0:raise ValueError('Nonempty object required')
    if up is None:
        if evidence is None:raise ValueError('Observed support up required')
        with np.load(Path(evidence)/'anchors.npz') as data:up=data['up']
    up=np.asarray(up,dtype=float);up/=np.linalg.norm(up)
    floor,valid=support_envelope(world,environment_vertices,environment_faces,up,diameter)
    gap=world@up-floor
    pinned,views=(observation_visibility(world,faces,evidence,diameter,cfg['visibility_tolerance_fraction'],
                    cfg['visibility_pin_mode'],cfg['silhouette_pin_width'])
                  if evidence else (np.zeros(len(vertices),dtype=bool),[]))
    pinned |= valid&(gap>cfg['upper_pin_gap_fraction']*diameter)
    pinned |= ~valid
    before=_metrics(gap,valid,pinned,diameter)
    # Softplus is a smooth maximum between the support obstacle and the
    # original height. Its positive derivative avoids an exact zero-thickness
    # clamp. Adapt transition width to the deepest allowed underside depth.
    penetration=max(float(np.max(-gap[valid])) if valid.any() else 0.,0)
    tau=max(cfg['transition_fraction']*diameter,
        penetration/max(-np.log(cfg['minimum_vertical_slope']),1.))
    clearance=(cfg['clearance_fraction']+cfg['minimum_vertical_thickness_fraction'])*diameter
    safe_gap=np.where(valid,gap,0.)
    target=tau*np.logaddexp(0.,safe_gap/tau)+clearance-safe_gap
    target=np.clip(target,0,cfg['maximum_displacement_fraction']*diameter)
    target[pinned]=0
    # Elastic transition on the original mesh graph; obstacle lower bounds keep
    # the hidden underside outside the support whenever its trust bound allows.
    edge=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    edge=np.sort(edge,axis=1);edge=np.unique(edge,axis=0)
    length=np.linalg.norm(world[edge[:,0]]-world[edge[:,1]],axis=1)
    weight=1/np.maximum(length,diameter*1e-6)
    row=np.r_[edge[:,0],edge[:,1]];col=np.r_[edge[:,1],edge[:,0]]
    adjacency=sparse.csr_matrix((np.r_[weight,weight],(row,col)),shape=(len(vertices),len(vertices)))
    degree=np.asarray(adjacency.sum(1)).ravel()
    lower=np.maximum(clearance-safe_gap,0);lower[pinned]=0
    lower=np.minimum(lower,cfg['maximum_displacement_fraction']*diameter)
    displacement=target.copy();stiffness=cfg['elastic_strength']
    for _ in range(cfg['elastic_iterations']):
        smooth=np.divide(adjacency@displacement,degree,out=np.zeros_like(displacement),where=degree>0)
        displacement=np.maximum((target+stiffness*smooth)/(1+stiffness),lower)
        displacement[pinned]=0
    displacement=np.minimum(displacement,cfg['maximum_displacement_fraction']*diameter)
    # Pin/obstacle boundaries can conflict. Locally back off only vertices of
    # unsafe faces; never silently flip a face to satisfy the collision proxy.
    corrections=0
    for iteration in range(cfg['local_safety_backoff_iterations']):
        candidate=world+displacement[:,None]*up
        bad,dot,ratio=face_safety(world,candidate,faces,cfg)
        if not bad.any():break
        ids=np.unique(faces[bad]);displacement[ids]*=.5;displacement[pinned]=0
        corrections+=len(ids)
    # Hard-freeze only unsafe face vertices, then propagate the fixed boundary
    # through newly unsafe adjacent faces. A tiny unsafe patch must not discard
    # unrelated safe underside corrections on the other side of the object.
    frozen_for_safety=np.zeros(len(vertices),dtype=bool)
    safety_freeze_waves=0
    for iteration in range(cfg['local_safety_freeze_iterations']):
        candidate=world+displacement[:,None]*up
        bad,dot,ratio=face_safety(world,candidate,faces,cfg)
        if not bad.any():break
        ids=np.unique(faces[bad]);ids=ids[displacement[ids]>0]
        if not len(ids):break
        displacement[ids]=0;frozen_for_safety[ids]=True
        safety_freeze_waves+=1
    candidate=world+displacement[:,None]*up
    bad,dot,ratio=face_safety(world,candidate,faces,cfg)
    if bad.any():
        # At the wave budget, freeze only connected components of still-moving
        # vertices that touch unsafe faces. No face can couple two remaining
        # components, so this closes the safety boundary without a global reset.
        moving=np.flatnonzero(displacement>0)
        from scipy.sparse.csgraph import connected_components
        _,labels=connected_components(adjacency[moving][:,moving],directed=False)
        label_of=np.full(len(vertices),-1,dtype='int64');label_of[moving]=labels
        bad_vertex=np.unique(faces[bad]);affected=np.unique(label_of[bad_vertex]);affected=affected[affected>=0]
        frozen=moving[np.isin(labels,affected)]
        displacement[frozen]=0;frozen_for_safety[frozen]=True
        candidate=world+displacement[:,None]*up
        bad,dot,ratio=face_safety(world,candidate,faces,cfg)
    if bad.any():
        raise RuntimeError('Local contact safety freeze did not restore original face orientation')
    rejected=bool(not np.any(displacement>0))
    corrected=transform(candidate,np.linalg.inv(pose))
    after=_metrics(gap+displacement,valid,pinned,diameter)
    original_dtype=np.asarray(canonical_vertices).dtype
    if original_dtype.kind=='f':corrected=corrected.astype(original_dtype)
    report=dict(complete=True,algorithm='Observed-up original-triangle envelope + visibility-pinned elastic soft obstacle underside map',
        geometry_modified=bool(np.any(displacement>1e-12*diameter)),topology_unchanged=True,
        vertex_count=len(vertices),face_count=len(faces),environment_faces=len(environment_faces),
        vertices_moved=int((displacement>1e-12*diameter).sum()),pinned_vertices=int(pinned.sum()),
        maximum_displacement=float(displacement.max()),maximum_displacement_fraction=float(displacement.max()/diameter),
        displacement_p95=float(np.quantile(displacement[displacement>0],.95)) if np.any(displacement>0) else 0.,
        visibility_pin_max_displacement=float(displacement[pinned].max()) if pinned.any() else 0.,
        transition_width_scene_units=tau,minimum_vertical_slope_requested=cfg['minimum_vertical_slope'],
        before=before,after=after,observation_visibility=views,
        flipped_or_degenerate_faces=int(bad.sum()),minimum_face_normal_dot=dot,
        minimum_triangle_area_ratio=ratio,locally_backed_off_vertex_updates=corrections,
        safety_frozen_vertices=int(frozen_for_safety.sum()),safety_freeze_waves=safety_freeze_waves,
        rejected_for_face_safety=rejected,up=up.tolist(),diameter=diameter,config=cfg,
        original_asset_modified=False,mesh_simplified=False,colors_uv_materials_changed=False,
        collision_free_certified=False,complete_collision_recheck_required=True,
        self_intersection_recheck_required=True,elapsed=time.time()-started,
        limitations=['Configured observation-visible/silhouette vertices and upper vertices are hard pinned; generated geometry changes elsewhere.',
                     'Visible underside penetration cannot be removed while keeping its observations fixed.',
                     'Ray envelope assumes a support height field along observed up; overhang/side-wall cases require independent exact audit.',
                     'Local safety backoff can leave unresolved collision vertices; report does not claim all intersections removed.'])
    return corrected,report


