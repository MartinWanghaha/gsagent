"""Multistart Sim(3) observation fitting, followed by bounded contact refinement.

Mesh geometry is fixed. Completed support has a weaker weight than observed
support. A physics correction cannot pass publication by sacrificing observations.
"""
from pathlib import Path
import itertools
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .geometry import RayMesh, sim3, perturb, transform, project, camera_rays, surface_samples, check_similarity
from .io import read_json, write_json, ply, xyz
from .texture import sample_colors
from .collision import is_watertight


def huber(x,delta=.02):
    x=np.abs(np.asarray(x));return np.where(x<=delta,.5*x*x,delta*(x-.5*delta))


def central_jacobian(fn,x,step=1e-4):
    # RaycastingScene stores float32. scipy's default 1e-8 step vanishes at x=0.
    cols=[]
    for i in range(len(x)):
        delta=np.zeros_like(x);delta[i]=step
        cols.append((fn(x+delta)-fn(x-delta))/(2*step))
    return np.column_stack(cols)


def physical_constraints_ok(detail,cfg):
    return (detail['contact_gap_fraction']<=cfg['maximum_contact_gap'] and
            detail['oriented_penetration_max']<=cfg['maximum_penetration'] and
            detail['unweighted_penetration_max']<=2*cfg['maximum_penetration'] and
            detail['volumetric_penetration_max']<=2*cfg['maximum_penetration'])


def settle_to_support(problem,T,observation_T):
    """Find a feasible small displacement along observed support normals.

    The mesh remains fixed and nonplanar. A soft objective alone may prefer a
    slightly floating/intersecting solution; enforce the stated constraints
    before choosing the best observation fit among feasible placements.
    """
    p=problem;cfg=p.cfg;before=p.observation(observation_T)
    def details(pose):
        h=transform(p.samples,pose)@p.up
        indices=np.flatnonzero(h<=np.quantile(h,cfg['contact_quantile']))
        return p.physics(pose,indices,return_detail=True)[1]
    current=details(T)
    if physical_constraints_ok(current,cfg):return T,dict(applied=False,reason='already feasible')
    choices=[];bound=cfg['physics_translation_bound']
    for offset in np.linspace(-bound,bound,61):
        candidate=T.copy();candidate[:3,3]+=offset*p.d*p.up
        if np.max(np.abs(candidate[:3,3]-observation_T[:3,3]))>bound*p.d+1e-10:continue
        score=p.observation(candidate)
        if score>before+cfg['max_observation_degradation']:continue
        physical=details(candidate)
        if physical_constraints_ok(physical,cfg):
            choices.append((score+cfg['pose_anchor_weight']*offset**2,offset,candidate))
    if not choices:return T,dict(applied=False,reason='no feasible displacement within observation/translation bounds')
    _,offset,result=min(choices,key=lambda x:x[0])
    return result,dict(applied=True,offset_fraction=float(offset),feasible_candidates=len(choices),
                       method='bounded support-normal translation against actual triangle mesh')


class PoseProblem:
    def __init__(self,evidence,asset,cfg):
        import open3d as o3d
        self.root=Path(evidence);self.meta=read_json(self.root/'observations.json');self.cfg=cfg
        a=np.load(self.root/'anchors.npz');self.points=a['points'];self.up=a['up']
        self.support=a['support_points'];self.support_weight=a['support_confidence']
        self.support_tree=cKDTree(self.support);self.d=self.meta['diameter']
        asset=np.load(asset);self.vertices=asset['vertices'];self.faces=asset['faces'];self.colors=asset['colors']
        self.mesh=RayMesh(self.vertices,self.faces)
        self.samples,self.normals,fi,bary=surface_samples(self.vertices,self.faces,cfg['surface_samples'])
        self.sample_colors=sample_colors(asset,fi,bary)
        self.face_colors=sample_colors(asset,np.arange(len(self.faces)),np.full((len(self.faces),3),1/3))
        self.watertight=is_watertight(self.vertices,self.faces)
        bg=ply(self.root/'background_proxy.ply');self.background=RayMesh(xyz(bg['vertex'].data),np.stack(bg['face'].data['vertex_indices']))
        rng=np.random.default_rng(cfg['seed']);self.views=[]
        for v in self.meta['views']:
            o=np.load(self.root/'observations'/(Path(v['name']).stem+'.npz'))
            mask=o['mask'];ys,xs=np.where(mask)
            if not len(xs):continue
            # Fixed image ROI and rays for all candidates: no pose-dependent evaluation crop.
            margin=max(8,round(max(xs.max()-xs.min(),ys.max()-ys.min())*.25))
            lo=np.maximum([xs.min()-margin,ys.min()-margin],0);hi=np.minimum([xs.max()+margin+1,ys.max()+margin+1],[mask.shape[1],mask.shape[0]])
            y,x=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];uv=np.c_[x.ravel(),y.ravel()]
            if len(uv)>cfg['rays_per_view']:uv=uv[rng.choice(len(uv),cfg['rays_per_view'],replace=False)]
            self.views.append(dict(meta=v,obs={k:o[k] for k in o.files},uv=uv,rays=camera_rays(o['K'],o['w2c'],uv)))

    def anchor_residual(self,T):
        s,_,_=check_similarity(T);q=transform(self.points,np.linalg.inv(T));closest,_,_=self.mesh.closest(q)
        residual=(q-closest)*s/self.d
        # Fixed-length bounded influence; no shrinking the denominator by deleting outliers.
        return np.clip(residual,-.15,.15).ravel()

    def observation(self,T,validation=False,details=False):
        errors=[];stats=[]
        for v in self.views:
            if v['meta']['validation']!=validation:continue
            o=v['obs'];u,y=v['uv'].T;truth=o['mask'][y,u];depth=o['depth'][y,u]
            z,face,_=self.mesh.cast(v['rays'],T);hit=np.isfinite(z)
            # Occlusion by other objects: ignore predicted surfaces behind an observed occluder.
            visible=hit&((depth<=0)|(z<=depth+self.cfg['occlusion_tolerance']*self.d)|truth)
            intersection=(visible&truth).sum();union=(visible|truth).sum();iou=float(intersection/max(union,1))
            both=visible&truth&(depth>0)
            dz=np.abs(z[both]-depth[both])/self.d
            depth_error=float(np.mean(np.minimum(dz,.25))) if both.any() else .25
            # Use texture only where geometry agrees; texture cannot rescue a displaced pose.
            color_valid=both&(np.abs(np.where(hit,z,1e9)-depth)<.05*self.d)
            if color_valid.any():
                rgb=self.face_colors[face[color_valid]];gt=o['rgb'][y[color_valid],u[color_valid]]/255.
                color_error=float(np.mean(np.abs(rgb-gt)))
            else:color_error=1.
            score=(1-iou)+self.cfg['depth_weight']*depth_error+self.cfg['color_weight']*color_error
            errors.append(score);stats.append(dict(view=v['meta']['original'],iou=iou,depth_error_fraction=depth_error,color_l1=color_error))
        if not errors:return (None,[]) if details else 0.
        return (float(np.mean(errors)),stats) if details else float(np.mean(errors))

    def physics(self,T,contact_indices,return_detail=False):
        world=transform(self.samples,T);dist,idx=self.support_tree.query(world)
        # Local oriented sheet tests apply only in a narrow neighborhood of supported surfaces.
        closest,normal,_=self.background.closest(world)
        normal*=np.where((normal@self.up)<0,-1.,1.)[:,None]
        near=(dist<.12*self.d)&(normal@self.up>.35)
        signed=((world-closest)*normal).sum(1)/self.d
        penetration=np.maximum(-signed-self.cfg['contact_tolerance'],0)*near*self.support_weight[idx]
        # Contact is distance to a continuous triangle surface, not to the
        # discrete support samples (whose spacing creates a false air gap).
        # Support samples only establish observed/completed confidence and ROI.
        surface_dist=np.linalg.norm(world-closest,axis=1)/self.d
        contact_dist=np.where(near[contact_indices],surface_dist[contact_indices],.3)
        contact_weight=self.support_weight[idx[contact_indices]]
        contact=float(np.average(np.minimum(contact_dist,.3)**2,weights=np.maximum(contact_weight,.01)))
        volumetric=0.;volumetric_max=0.
        if self.watertight:
            s,_,_=check_similarity(T);local=transform(self.support,np.linalg.inv(T))
            inside=self.mesh.scene.compute_signed_distance(self.mesh.o3d.core.Tensor(local.astype('float32')),nthreads=4,nsamples=3).numpy()*s/self.d
            volumetric=float(np.mean(self.support_weight*np.maximum(-inside-self.cfg['contact_tolerance'],0)**2))
            volumetric_max=float(np.maximum(-inside-self.cfg['contact_tolerance'],0).max())
        loss=self.cfg['contact_weight']*contact+self.cfg['collision_weight']*(float(np.mean(penetration**2))+volumetric)
        detail=dict(contact_gap_fraction=float(np.median(contact_dist)),
                    oriented_penetration_p95=float(np.quantile(penetration,.95)),
                    oriented_penetration_max=float(penetration.max()),
                    unweighted_penetration_max=float((np.maximum(-signed-self.cfg['contact_tolerance'],0)*near).max()),
                    volumetric_penetration_loss=volumetric,volumetric_penetration_max=volumetric_max,closed_object=self.watertight,
                    contact_distance_kind='continuous_triangle_surface',
                    collision_scope='local observed/completed surfaces; not a global watertight scene guarantee')
        return (loss,detail) if return_detail else loss

    def candidates(self,initial=None):
        target_center=np.median(self.points,axis=0);source_center=np.median(self.samples,axis=0)
        def axes(p):return np.linalg.eigh(np.cov(p.T))[1][:,::-1]
        a,b=axes(self.samples),axes(self.points)
        if np.linalg.det(a)<0:a[:,-1]*=-1
        if np.linalg.det(b)<0:b[:,-1]*=-1
        scale=np.linalg.norm(np.quantile(self.points,.95,axis=0)-np.quantile(self.points,.05,axis=0))/max(np.linalg.norm(np.quantile(self.samples,.95,axis=0)-np.quantile(self.samples,.05,axis=0)),1e-12)
        candidates=[]
        if initial is not None:check_similarity(initial);candidates.append(initial)
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product([-1,1],repeat=3):
                p=np.eye(3)[:,perm]@np.diag(signs)
                if np.linalg.det(p)<0:continue
                R=b@p@a.T;candidates.append(sim3(scale,R,target_center-scale*R@source_center))
        return candidates


def optimize_pose(evidence,asset,output,cfg,initial=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);p=PoseProblem(evidence,asset,cfg)
    print(f'Pose: {len(p.vertices)} vertices, {len(p.faces)} faces; closed={p.watertight}',flush=True)
    candidates=[]
    for i,T in enumerate(p.candidates(initial)):
        fn=lambda x:p.anchor_residual(perturb(T,x,p.d))
        result=least_squares(fn,np.zeros(7),jac=lambda x:central_jacobian(fn,x),
                             bounds=([-1.0]*3+[-.4]*3+[-.45],[1.0]*3+[.4]*3+[.45]),
                             loss='soft_l1',f_scale=.02,diff_step=1e-3,max_nfev=cfg['anchor_iterations'])
        pose=perturb(T,result.x,p.d);score=p.observation(pose)+float(huber(p.anchor_residual(pose)).mean())*cfg['anchor_weight']
        candidates.append(dict(index=i,T=pose,score=score))
        print(f'Pose anchor candidate {i}: objective={score:.6f}',flush=True)
    candidates.sort(key=lambda x:x['score']);refined=[]
    for candidate in candidates[:cfg['refine_candidates']]:
        T=candidate['T']
        def objective(x):
            pose=perturb(T,x,p.d)
            return p.observation(pose)+cfg['anchor_weight']*float(huber(p.anchor_residual(pose)).mean())+.01*float(x@x)
        r=minimize(objective,np.zeros(7),method='Powell',bounds=[(-.25,.25)]*3+[(-.1,.1)]*3+[(-.12,.12)],
                   options=dict(maxiter=cfg['pose_iterations'],xtol=1e-4,ftol=1e-5))
        refined.append(dict(index=candidate['index'],T=perturb(T,r.x,p.d),score=float(r.fun)))
        print(f'Pose observation candidate {candidate["index"]}: objective={r.fun:.6f}',flush=True)
    refined.sort(key=lambda x:x['score']);observation_T=refined[0]['T'];T=observation_T.copy()
    world=transform(p.samples,T);height=world@p.up
    contact_indices=np.flatnonzero(height<=np.quantile(height,cfg['contact_quantile']))
    def physical_objective(x):
        pose=perturb(observation_T,x,p.d,scale=False)
        return (p.observation(pose)+cfg['anchor_weight']*float(huber(p.anchor_residual(pose)).mean())+
                p.physics(pose,contact_indices)+cfg['pose_anchor_weight']*float(x@x))
    bound=cfg['physics_translation_bound'];angle=cfg['physics_rotation_bound']
    result=minimize(physical_objective,np.zeros(6),method='Powell',bounds=[(-angle,angle)]*3+[(-bound,bound)]*3,
                    options=dict(maxiter=cfg['physics_iterations'],xtol=1e-4,ftol=1e-5))
    candidate=perturb(observation_T,result.x,p.d,scale=False)
    before=p.observation(observation_T);after=p.observation(candidate)
    physical_accepted=after<=before+cfg['max_observation_degradation']
    if physical_accepted:T=candidate
    T,settle=settle_to_support(p,T,observation_T)
    # Acceptance and later publication must use the same final contact region.
    final_height=transform(p.samples,T)@p.up
    contact_indices=np.flatnonzero(final_height<=np.quantile(final_height,cfg['contact_quantile']))
    train_score,train=p.observation(T,details=True);validation_score,validation=p.observation(T,validation=True,details=True)
    _,physical=p.physics(T,contact_indices,return_detail=True)
    ious=[v['iou'] for v in validation or train]
    reasons=[]
    if float(np.median(ious))<cfg['minimum_iou']:reasons.append('held-out silhouette agreement is insufficient')
    if physical['contact_gap_fraction']>cfg['maximum_contact_gap']:reasons.append('support contact is uncertain or floating')
    if physical['oriented_penetration_max']>cfg['maximum_penetration']:reasons.append('local penetration exceeds tolerance')
    if physical['unweighted_penetration_max']>cfg['maximum_penetration']*2:reasons.append('large intersection with uncertain completed support requires review')
    if physical['volumetric_penetration_max']>cfg['maximum_penetration']*2:reasons.append('background surface is enclosed deeply inside the generated object')
    if p.meta['up_confidence']<cfg['minimum_up_confidence']:reasons.append('support direction is uncertain')
    # Do not silently claim uniqueness for symmetric or indistinguishable hypotheses.
    alternatives=[dict(T=c['T'],score=c['score']) for c in refined if c['score']-refined[0]['score']<cfg['ambiguity_score_gap']]
    report=dict(schema_version=1,kind='paintmesh-object-placement',status='accepted' if not reasons else 'needs_review',
                reasons=reasons,object_to_world=T,observation_pose=observation_T,
                physical_correction_accepted=physical_accepted,physics=physical,
                support_settle=settle,
                train_score=train_score,validation_score=validation_score,train=train,validation=validation,
                alternatives=alternatives,canonical_scale=check_similarity(T)[0],config=cfg,
                input_asset=fingerprint_safe(asset),input_observations=fingerprint_safe(Path(evidence)/'observations.json'))
    write_json(out/'pose.json',report);write_json(out/'candidates.json',dict(candidates=candidates,refined=refined))
    render_review(p,observation_T,T,out/'review')
    return report


def fingerprint_safe(path):
    from .io import fingerprint
    return fingerprint(path)


def render_review(p,before,after,output):
    out=Path(output);out.mkdir(exist_ok=True,parents=True)
    for v in p.views:
        o=v['obs'];h,w=o['mask'].shape;y,x=np.mgrid[:h,:w];rays=camera_rays(o['K'],o['w2c'],np.c_[x.ravel(),y.ravel()]);panels=[]
        for T,title in [(None,'Original mask'),(before,'Observation fit'),(after,'Contact refinement')]:
            rgb=o['rgb'].copy()
            if T is None:mask=o['mask'];color=np.array([30,220,60])
            else:
                z,fi,_=p.mesh.cast(rays,T);z=z.reshape(h,w);fi=fi.reshape(h,w)
                mask=np.isfinite(z)&((o['depth']==0)|(z<=o['depth']+.02*p.d)|o['mask'])
                color=np.zeros_like(rgb);color[mask]=np.clip(p.face_colors[fi[mask]]*255,0,255)
            rgb[mask]=(rgb[mask]*.35+(color if T is None else color[mask])*.65).astype('uint8')
            panel=Image.new('RGB',(w,h+25),'white');panel.paste(Image.fromarray(rgb),(0,25));ImageDraw.Draw(panel).text((8,5),title,fill='black');panels.append(panel)
        page=Image.new('RGB',(w*3,h+25),'white')
        for i,panel in enumerate(panels):page.paste(panel,(i*w,0))
        page.save(out/(Path(v['meta']['name']).stem+'.jpg'))
