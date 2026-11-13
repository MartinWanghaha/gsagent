"""Small rigid contact corrections against original environment triangles.

This is a local sequential contact solver, not a global signed-distance field.
It does not simplify either mesh, change scale, or deform vertices.  Up-oriented
support-sheet inequalities are solved in six pose variables; an independent
triangle-intersection audit is still required before publication.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.transform import Rotation


DEFAULTS = dict(iterations=12, clearance_fraction=0.0002,
    max_translation_fraction=.06, max_rotation_radians=.10,
    step_translation_fraction=.025, step_rotation_radians=.04,
    contact_neighborhood_fraction=.15, minimum_support_normal_cosine=.25,
    max_observation_score_increase=.03, max_median_iou_drop=.03,
    depth_observation_weight=2., observed_occlusion_fraction=.01,
    support_gap_fraction=.006, position_weight=1., rotation_weight=4.,
    numerical_fraction=1e-6, available_neighborhood_fraction=.05,
    verify_intersections=False)


def _transform(points, pose):
    return np.asarray(points) @ np.asarray(pose)[:3, :3].T + np.asarray(pose)[:3, 3]


def rigid_delta(initial_pose, delta, center, diameter):
    """Rotate in world coordinates around initial object center; retain scale."""
    delta=np.asarray(delta, dtype=float)
    rotation=Rotation.from_rotvec(delta[:3]).as_matrix()
    out=np.asarray(initial_pose,dtype=float).copy()
    out[:3,:3]=rotation @ out[:3,:3]
    out[:3,3]=rotation @ (out[:3,3]-center)+center+diameter*delta[3:]
    return out


class TriangleSupport:
    def __init__(self, vertices, faces, up, support_points=None):
        import open3d as o3d
        self.o3d=o3d
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=4)
        self.vertices=np.asarray(vertices,dtype=float)
        self.faces=np.asarray(faces,dtype=np.int64)
        if not len(self.faces):raise ValueError('Contact solver needs environment triangles')
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices,dtype='float32')),
            o3d.core.Tensor(np.asarray(faces,dtype='uint32')))
        self.up=np.asarray(up,dtype=float);self.up/=np.linalg.norm(self.up)
        self.support_points=None if support_points is None else np.asarray(support_points,dtype=float)

    def nearest(self, points):
        values=[];normals=[]
        for start in range(0,len(points),131072):
            result=self.scene.compute_closest_points(self.o3d.core.Tensor(
                np.asarray(points[start:start+131072],dtype='float32')),nthreads=4)
            values.append(result['points'].numpy());normals.append(result['primitive_normals'].numpy())
        q=np.concatenate(values).astype(float);n=np.concatenate(normals).astype(float)
        length=np.linalg.norm(n,axis=1)
        n=np.divide(n,length[:,None],out=np.zeros_like(n),where=length[:,None]>1e-12)
        n*=np.where((n@self.up)>=0,1.,-1.)[:,None]
        return q,n

    def constraints(self, points, diameter, cfg):
        q,n=self.nearest(points)
        signed=np.einsum('ij,ij->i',points-q,n)/diameter
        distance=np.linalg.norm(points-q,axis=1)/diameter
        eligible=(distance<cfg['contact_neighborhood_fraction']) & (
            n@self.up>=cfg['minimum_support_normal_cosine'])
        # A nearby upper shelf is not a support sheet merely because its
        # arbitrary triangle orientation can be flipped toward the up vector.
        if self.support_points is not None and len(self.support_points):
            reference=np.quantile(self.support_points@self.up,.95)
            eligible &= q@self.up <= reference+cfg['contact_neighborhood_fraction']*diameter
        return q,n,signed,eligible


class ObservationGuard:
    """Full native observation pixels, fixed ROI, no resized/sampled masks."""
    def __init__(self, vertices, faces, evidence, diameter, cfg):
        import open3d as o3d
        from recompose.geometry import camera_rays
        self.o3d=o3d;self.cfg=cfg;self.diameter=diameter
        self.scene=o3d.t.geometry.RaycastingScene(nthreads=4)
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices,dtype='float32')),
            o3d.core.Tensor(np.asarray(faces,dtype='uint32')))
        evidence=Path(evidence);meta=json.loads((evidence/'observations.json').read_text())
        self.views=[]
        for view in meta['views']:
            path=evidence/'observations'/(Path(view['name']).stem+'.npz')
            with np.load(path) as data:
                mask=data['mask'].astype(bool);depth=data['depth'];K=data['K'];w2c=data['w2c']
            yy,xx=np.nonzero(mask)
            if not len(xx):continue
            margin=max(8,int(round(max(np.ptp(xx),np.ptp(yy))*.25)))
            lo=np.maximum([xx.min()-margin,yy.min()-margin],0)
            hi=np.minimum([xx.max()+margin+1,yy.max()+margin+1],[mask.shape[1],mask.shape[0]])
            y,x=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
            uv=np.column_stack([x.ravel(),y.ravel()])
            self.views.append(dict(name=view['original'],validation=view['validation'],
                rays=camera_rays(K,w2c,uv),truth=mask[y,x].ravel(),depth=depth[y,x].ravel()))

    def score(self, pose):
        inverse=np.linalg.inv(pose);rows=[]
        for view in self.views:
            rays=view['rays']
            rr=np.column_stack([_transform(rays[:,:3],inverse),rays[:,3:]@inverse[:3,:3].T]).astype('float32')
            z=self.scene.cast_rays(self.o3d.core.Tensor(rr),nthreads=4)['t_hit'].numpy()
            depth,truth=view['depth'],view['truth'];hit=np.isfinite(z)&(z>0)
            visible=hit & (truth | (depth<=0) | (z<=depth+self.cfg['observed_occlusion_fraction']*self.diameter))
            iou=float(np.sum(visible&truth)/max(np.sum(visible|truth),1))
            valid=visible&truth&(depth>0)
            residual=float(np.minimum(np.abs(z[valid]-depth[valid])/self.diameter,.25).mean()) if valid.any() else .25
            rows.append(dict(name=view['name'],validation=view['validation'],iou=iou,
                depth_error_fraction=residual,pixels=len(depth)))
        return dict(score=float(np.mean([1-r['iou']+self.cfg['depth_observation_weight']*r['depth_error_fraction'] for r in rows])) if rows else 0.,
            median_iou=float(np.median([r['iou'] for r in rows])) if rows else 1.,
            per_view=rows,all_native_roi_pixels=True)


def _metrics(points,support,diameter,cfg):
    q,n,signed,eligible=support.constraints(points,diameter,cfg)
    values=signed[eligible]
    if not len(values):
        return dict(eligible_vertices=0,maximum_penetration_fraction=None,
            p99_penetration_fraction=None,minimum_gap_fraction=None), (q,n,signed,eligible)
    penetration=np.maximum(-values,0)
    return dict(eligible_vertices=len(values),
        maximum_penetration_fraction=float(penetration.max()),
        p99_penetration_fraction=float(np.quantile(penetration,.99)),
        penetrating_vertices=int((values<-cfg['numerical_fraction']).sum()),
        minimum_gap_fraction=float(values.min())),(q,n,signed,eligible)


def refine_contact_pose(vertices,faces,initial_pose,environment_vertices,environment_faces,
                        evidence=None,config=None,*,up=None,support_points=None):
    """Return ``(object_to_world, report)`` without changing any input arrays.

    ``vertices`` are canonical object coordinates. The initial pose may include
    an existing uniform scale; all corrections are rigid in world coordinates.
    Environment triangles must be originals from a conservative neighborhood
    large enough for the configured pose displacement. Publication must rerun
    the complete-scene broad/narrow collision check at the returned pose.
    """
    started=time.time();cfg={**DEFAULTS,**(config or {})}
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=np.int64)
    pose=np.asarray(initial_pose,dtype=float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError('Triangle mesh arrays required')
    if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('Finite pose required')
    if evidence is not None:
        with np.load(Path(evidence)/'anchors.npz') as anchors:
            if up is None:up=anchors['up']
            if support_points is None:support_points=anchors['support_points']
    if up is None:raise ValueError('Observed support up direction is required')
    initial_world=_transform(vertices,pose);center=np.mean(initial_world,axis=0)
    diameter=float(np.linalg.norm(np.ptp(initial_world,axis=0)))
    if not diameter>0:raise ValueError('Nonempty positive-diameter object required')
    support=TriangleSupport(environment_vertices,environment_faces,up,support_points)
    up=support.up
    guard=ObservationGuard(vertices,faces,evidence,diameter,cfg) if evidence else None
    initial_observation=guard.score(pose) if guard else None
    before,_=_metrics(initial_world,support,diameter,cfg)
    current=pose.copy();delta_total=np.zeros(6);history=[]
    weight=np.array([cfg['rotation_weight']]*3+[cfg['position_weight']]*3)
    global_bounds=np.array([cfg['max_rotation_radians']]*3+[cfg['max_translation_fraction']]*3)
    step_bounds=np.array([cfg['step_rotation_radians']]*3+[cfg['step_translation_fraction']]*3)
    candidates=[(current.copy(),before,delta_total.copy())]
    for iteration in range(cfg['iterations']):
        points=_transform(vertices,current)
        metric,(q,n,signed,eligible)=_metrics(points,support,diameter,cfg)
        active=eligible & (signed<cfg['clearance_fraction']+.015)
        if not active.any():break
        rel=(points[active]-center)/diameter
        matrix=np.column_stack([np.cross(rel,n[active]),n[active]])
        rhs=cfg['clearance_fraction']-signed[active]
        lower=np.maximum(-step_bounds,-global_bounds-delta_total)
        upper=np.minimum(step_bounds,global_bounds-delta_total)
        # Six-variable constrained least squares. All eligible original object
        # vertices participate; no sampled collision proxy is optimized.
        def objective(x):return .5*float(np.dot(weight*(x+delta_total),x+delta_total))
        def jac(x):return weight*(x+delta_total)
        current_center=points.mean(0)
        def trust_region(x):
            rotation=Rotation.from_rotvec(x[:3]).as_matrix()
            candidate_center=rotation@(current_center-center)+center+diameter*x[3:]
            rotation_change=Rotation.from_matrix(rotation@current[:3,:3]@np.linalg.inv(pose[:3,:3])).as_rotvec()
            return np.array([cfg['max_translation_fraction']**2-np.sum(((candidate_center-center)/diameter)**2),
                cfg['max_rotation_radians']**2-np.sum(rotation_change**2)])
        result=minimize(objective,np.zeros(6),jac=jac,method='SLSQP',
            constraints=[LinearConstraint(matrix,rhs,np.inf),{'type':'ineq','fun':trust_region}],bounds=list(zip(lower,upper)),
            options=dict(maxiter=60,ftol=1e-11))
        step=result.x
        if not result.success:
            # A hard contact system can be infeasible on a strongly distorted
            # support surface within the observation/pose trust region. Return
            # the best bounded hinge reduction, but never call it collision-free.
            def soft(x):
                v=np.minimum(matrix@x-rhs,0)
                return .5*float(np.mean(v*v))*1e5+objective(x)
            def soft_jac(x):
                v=np.minimum(matrix@x-rhs,0)
                return 1e5*(matrix.T@v)/max(len(v),1)+jac(x)
            fallback=minimize(soft,np.zeros(6),jac=soft_jac,method='L-BFGS-B',
                bounds=list(zip(lower,upper)),options=dict(maxiter=100,ftol=1e-12))
            step=fallback.x
            # The bounded hinge solve uses a box for speed. Backtrack its step
            # until both true Euclidean translation and rotation remain legal.
            for _ in range(20):
                if np.min(trust_region(step))>=-1e-12:break
                step*=.5
        delta_total+=step
        # Compose around a fixed world center; this preserves the initial Sim3
        # scale exactly even if several rotation updates are needed.
        rotation=Rotation.from_rotvec(step[:3]).as_matrix()
        current[:3,:3]=rotation@current[:3,:3]
        current[:3,3]=rotation@(current[:3,3]-center)+center+diameter*step[3:]
        after,_=_metrics(_transform(vertices,current),support,diameter,cfg)
        history.append(dict(iteration=iteration+1,hard_linear_system_solved=bool(result.success),
            step=step.tolist(),**after))
        candidates.append((current.copy(),after,delta_total.copy()))
        if after['maximum_penetration_fraction'] is not None and after['maximum_penetration_fraction']<cfg['numerical_fraction']:
            break
        if np.linalg.norm(step)<1e-7:break
    # Keep contact: lower a collision-cleared candidate to the first supported
    # original vertex instead of allowing an arbitrary air gap.
    points=_transform(vertices,current)
    _,(q,n,signed,eligible)=_metrics(points,support,diameter,cfg)
    if eligible.any():
        projection=n[eligible]@up
        offset=float(np.min((signed[eligible]-cfg['clearance_fraction'])/projection))
        if offset>0 and offset<cfg['max_translation_fraction']:
            settled=current.copy();settled[:3,3]-=offset*diameter*up
            metric,_=_metrics(_transform(vertices,settled),support,diameter,cfg)
            if metric['maximum_penetration_fraction'] is not None and metric['maximum_penetration_fraction']<=cfg['numerical_fraction']:
                candidates.append((settled,metric,delta_total-offset*np.r_[np.zeros(3),up]))
    # Preserve a Pareto improvement if the full geometric correction conflicts
    # with silhouette/depth observations; an unresolved case must not silently
    # discard every smaller useful correction.
    best=candidates[-1][0]
    best_rotation=Rotation.from_matrix(best[:3,:3]@np.linalg.inv(pose[:3,:3])).as_rotvec()
    centroid_shift=(_transform(vertices,best).mean(0)-center)/diameter
    for fraction in [.75,.5,.25,.125]:
        delta=np.r_[fraction*best_rotation,fraction*centroid_shift]
        candidate=rigid_delta(pose,delta,center,diameter)
        metric,_=_metrics(_transform(vertices,candidate),support,diameter,cfg)
        candidates.append((candidate,metric,delta))
    # Try best geometric candidates against every native observation ROI pixel.
    candidates.sort(key=lambda row:(row[1]['maximum_penetration_fraction'] if row[1]['maximum_penetration_fraction'] is not None else np.inf,
        float(np.dot(weight*row[2],row[2]))))
    accepted=[];rejected=[]
    # Walk the full geometric/pose Pareto ordering until an observation-valid
    # candidate is found; a small useful step cannot disappear behind a fixed
    # top-k cutoff. Original pose remains an explicit final fallback.
    tested=[]
    for candidate,metric,delta in candidates:
        if any(np.allclose(candidate,old,rtol=0,atol=1e-12) for old in tested):continue
        tested.append(candidate)
        candidate_shift=np.linalg.norm(_transform(vertices,candidate).mean(0)-center)/diameter
        candidate_angle=np.linalg.norm(Rotation.from_matrix(candidate[:3,:3]@np.linalg.inv(pose[:3,:3])).as_rotvec())
        if candidate_shift>cfg['max_translation_fraction']+1e-8 or candidate_angle>cfg['max_rotation_radians']+1e-8:
            continue
        observation=guard.score(candidate) if guard else None
        valid=(observation is None or (
            observation['score']<=initial_observation['score']+cfg['max_observation_score_increase'] and
            observation['median_iou']>=initial_observation['median_iou']-cfg['max_median_iou_drop']))
        row=dict(pose=candidate,metric=metric,delta=delta,observation=observation)
        if valid:
            accepted.append(row)
            break
        else:rejected.append(dict(metric=metric,delta=delta.tolist(),observation=observation))
    if accepted:
        chosen=accepted[0]
    else:
        chosen=dict(pose=pose.copy(),metric=before,delta=np.zeros(6),observation=initial_observation)
    current=chosen['pose'];after=chosen['metric']
    initial_scale=float(np.cbrt(np.linalg.det(pose[:3,:3])))
    final_scale=float(np.cbrt(np.linalg.det(current[:3,:3])))
    report=dict(complete=True,algorithm='Original-triangle local support-sheet sequential rigid inequalities + native multi-view observation guard',
        before=before,after=after,history=history,observation_before=initial_observation,
        observation_after=chosen['observation'],observation_rejected_candidates=rejected,
        input_scale=initial_scale,output_scale=final_scale,scale_preserved=bool(np.isclose(initial_scale,final_scale,rtol=1e-10)),
        vertices=len(vertices),faces=len(faces),environment_faces=len(environment_faces),mesh_simplified=False,
        source_vertices_modified=False,up=up.tolist(),diameter=diameter,
        translation_scene_units=(_transform(vertices,current).mean(0)-center).tolist(),
        translation_fraction=float(np.linalg.norm(_transform(vertices,current).mean(0)-center)/diameter),
        canonical_origin_translation_scene_units=(current[:3,3]-pose[:3,3]).tolist(),
        observation_candidates_tested=len(tested),
        rotation_radians=float(np.linalg.norm(Rotation.from_matrix(current[:3,:3]@np.linalg.inv(pose[:3,:3])).as_rotvec())),
        local_sheet_clear=after['maximum_penetration_fraction'] is not None and after['maximum_penetration_fraction']<=cfg['numerical_fraction'],
        collision_free_certified=False,complete_mesh_collision_recheck_required=True,
        environment_neighborhood_extrapolated=bool(np.any(_transform(vertices,current).min(0)<initial_world.min(0)-cfg['available_neighborhood_fraction']*diameter) or np.any(_transform(vertices,current).max(0)>initial_world.max(0)+cfg['available_neighborhood_fraction']*diameter)),
        global_signed_distance_used=False,config=cfg,elapsed=time.time()-started,
        limitations=['Local support-sheet orientation is based on observed up; side-wall collisions require final triangle audit.',
                     'Rigid correction cannot repair a geometrically invalid environment or generated object.',
                     'Observation guard evaluates original capture evidence, not benchmark test RGB.'])
    if cfg['verify_intersections']:
        from mesh_benchmark.mesh_intersection import audit_intersections
        final=audit_intersections(_transform(vertices,current),faces,environment_vertices,environment_faces,
            tolerance=diameter*1e-7)
        report['intersection_counts']=final['counts']
        report['surface_crossings_remaining']=final['counts']['transverse']
        # A local ROI audit is not a full-world containment certificate.
    return current,report


def refine_contact_pose_envelope(vertices,faces,initial_pose,environment_vertices,
                                 environment_faces,evidence=None,config=None,*,up=None):
    """Candidate search against an actual triangle upper support envelope.

    Ray queries use every original object vertex. Candidate construction uses
    quantiles to avoid a single spurious support triangle steering the entire
    rigid pose. Those quantiles are *not* a collision-free certificate: full
    FCL verification remains required. Returned candidate rows allow the caller
    to select with a complete-scene collision objective instead of this proxy.
    """
    import open3d as o3d
    started=time.time();cfg={**DEFAULTS,**(config or {})}
    cfg.setdefault('envelope_lift_fractions',[0.,.005,.01,.015,.02,.03,.04,.05,.06])
    cfg.setdefault('envelope_tilt_radians',.035)
    cfg.setdefault('envelope_max_guard_candidates',12)
    cfg.setdefault('envelope_return_candidates',True)
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=np.int64)
    pose=np.asarray(initial_pose,dtype=float)
    if up is None:
        if evidence is None:raise ValueError('Observed support up is required')
        with np.load(Path(evidence)/'anchors.npz') as f:up=f['up']
    up=np.asarray(up,dtype=float);up/=np.linalg.norm(up)
    world=_transform(vertices,pose);center=world.mean(0);diameter=float(np.linalg.norm(np.ptp(world,axis=0)))
    axis=np.array([1.,0,0])
    if abs(axis@up)>.9:axis=np.array([0.,1,0])
    h0=axis-up*(axis@up);h0/=np.linalg.norm(h0);h1=np.cross(up,h0)
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(environment_vertices,dtype='float32')),
        o3d.core.Tensor(np.asarray(environment_faces,dtype='uint32')))
    start_height=float(np.max(world@up)+.15*diameter)
    def gaps(points):
        heights=points@up
        origins=points+(start_height-heights)[:,None]*up
        chunks=[]
        for start in range(0,len(points),131072):
            rr=np.c_[origins[start:start+131072],np.broadcast_to(-up,(len(origins[start:start+131072]),3))].astype('float32')
            values=scene.cast_rays(o3d.core.Tensor(rr),nthreads=4)['t_hit'].numpy()
            chunks.append(values)
        distance=np.concatenate(chunks);valid=np.isfinite(distance)&(distance>0)
        gap=(heights-(start_height-distance))/diameter
        return gap,valid
    def metrics(points):
        gap,valid=gaps(points);values=gap[valid];pen=np.maximum(-values,0)
        return dict(envelope_valid_vertices=int(valid.sum()),
            maximum_penetration_fraction=float(pen.max()) if len(pen) else None,
            p99_penetration_fraction=float(np.quantile(pen,.99)) if len(pen) else None,
            p999_penetration_fraction=float(np.quantile(pen,.999)) if len(pen) else None,
            mean_penetration_fraction=float(pen.mean()) if len(pen) else None,
            penetrating_vertices=int((pen>cfg['numerical_fraction']).sum()),
            minimum_gap_fraction=float(values.min()) if len(values) else None,
            envelope_support_coverage=float(valid.mean()))
    before=metrics(world)
    # Every candidate is rigid. First-contact lift values arise from actual
    # nonplanar mesh rays, never from a fitted global plane or a shell SDF.
    angles=[np.zeros(3)]
    tilt=min(cfg['envelope_tilt_radians'],cfg['max_rotation_radians'])
    angles += [sign*tilt*axis for axis in [h0,h1] for sign in [-1,1]]
    candidates=[]
    for angle in angles:
        rotated=rigid_delta(pose,np.r_[angle,[0.,0,0]],center,diameter)
        rw=_transform(vertices,rotated);gap,valid=gaps(rw)
        need=-gap[valid]+cfg['clearance_fraction']
        lifts=list(cfg['envelope_lift_fractions'])
        if len(need):lifts += [float(np.quantile(need,q)) for q in [.99,.999,1.]]
        lifts=sorted(set(float(np.clip(v,0,cfg['max_translation_fraction'])) for v in lifts))
        for lift in lifts:
            candidate=rotated.copy();candidate[:3,3]+=up*lift*diameter
            # Vertical translation leaves ray footprint unchanged, hence its
            # complete exact envelope gap is a constant offset of cached gaps.
            values=gap[valid]+lift;pen=np.maximum(-values,0)
            metric=dict(envelope_valid_vertices=int(valid.sum()),
                maximum_penetration_fraction=float(pen.max()) if len(pen) else None,
                p99_penetration_fraction=float(np.quantile(pen,.99)) if len(pen) else None,
                p999_penetration_fraction=float(np.quantile(pen,.999)) if len(pen) else None,
                mean_penetration_fraction=float(pen.mean()) if len(pen) else None,
                penetrating_vertices=int((pen>cfg['numerical_fraction']).sum()),
                minimum_gap_fraction=float(values.min()) if len(values) else None,
                envelope_support_coverage=float(valid.mean()))
            candidates.append(dict(pose=candidate,metric=metric,lift_fraction=lift,
                rotation_radians=float(np.linalg.norm(angle))))
    # Rank actual lower-envelope violations, not an isolated worst signed
    # nearest-face normal. Include small moves so an observation-valid useful
    # correction remains available if every almost-cleared pose is too high.
    candidates.sort(key=lambda c:(c['metric']['mean_penetration_fraction'] or 0,
        c['metric']['p999_penetration_fraction'] or 0,c['lift_fraction']+.1*c['rotation_radians']))
    guard=ObservationGuard(vertices,faces,evidence,diameter,cfg) if evidence else None
    base=guard.score(pose) if guard else None
    accepted=[];rejected=[];guarded=0
    for c in candidates:
        observation=guard.score(c['pose']) if guard else None;guarded+=1
        ok=observation is None or (observation['score']<=base['score']+cfg['max_observation_score_increase']
            and observation['median_iou']>=base['median_iou']-cfg['max_median_iou_drop'])
        row=dict(object_to_world=c['pose'].tolist(),metrics=c['metric'],lift_fraction=c['lift_fraction'],
            rotation_radians=c['rotation_radians'],observation=observation,observation_accepted=ok)
        (accepted if ok else rejected).append(row)
        # Return a small diverse accepted set for full FCL testing. If all
        # low-penetration poses fail observations, continue down the full curve.
        if len(accepted)>=cfg['envelope_max_guard_candidates']:break
    if accepted:
        chosen=accepted[0];result=np.asarray(chosen['object_to_world'])
    else:
        chosen=dict(metrics=before,observation=base,lift_fraction=0.,rotation_radians=0.)
        result=pose.copy()
    final_world=_transform(vertices,result)
    report=dict(complete=True,algorithm='Original-triangle support-ray upper envelope; bounded rigid up/tilt candidate Pareto search',
        before=before,after=chosen['metrics'],observation_before=base,observation_after=chosen['observation'],
        accepted_candidates=accepted,rejected_candidates=rejected,candidates_generated=len(candidates),
        observation_candidates_tested=guarded,translation_fraction=float(np.linalg.norm(final_world.mean(0)-center)/diameter),
        rotation_radians=chosen['rotation_radians'],input_scale=float(np.cbrt(np.linalg.det(pose[:3,:3]))),
        output_scale=float(np.cbrt(np.linalg.det(result[:3,:3]))),scale_preserved=True,
        diameter=diameter,up=up.tolist(),vertices=len(vertices),faces=len(faces),
        environment_faces=len(environment_faces),mesh_simplified=False,source_vertices_modified=False,
        local_sheet_clear=chosen['metrics']['maximum_penetration_fraction'] is not None and chosen['metrics']['maximum_penetration_fraction']<=cfg['numerical_fraction'],
        collision_free_certified=False,complete_mesh_collision_recheck_required=True,
        environment_neighborhood_extrapolated=bool(np.any(final_world.min(0)<world.min(0)-cfg['available_neighborhood_fraction']*diameter) or np.any(final_world.max(0)>world.max(0)+cfg['available_neighborhood_fraction']*diameter)),
        global_signed_distance_used=False,config=cfg,elapsed=time.time()-started,
        limitations=['Support is the first actual surface along observed down; overhangs/remnants can prevent rigid placement.',
                     'Quantile ranking does not certify no collision; complete original-triangle FCL verification is mandatory.',
                     'A rigid feasible pose may not exist under the original silhouette/shape constraints.'])
    return result,report
