"""Partition-consistent local refinement using frozen neighboring anchors.

Only a vertex's complete initial one-ring and frozen training observations
determine its trajectory. Neighbors never move inside another vertex's
objective, so halo truncation cannot propagate through refinement iterations.
A local analytic trust region guarantees positive original projected face
area, without a partition-wide rollback or topology changes.
"""
import gc
import numpy as np

from .fine_tsdf import TRUNCATION
from .normal_surface_fusion import _prepare_frames,_sample_teacher


def geometry_constraints(vertices,faces,max_displacement):
    """Return edges and per-vertex radii preserving original face orientation.

    Let A=||e1 x e2|| and L be the largest sum of two edge lengths of an
    original triangle (invariant to face numbering). If each
    corner moves by at most r=A/(8L), the change of projected doubled area is
    <=2rL+4r² <=(1/4+1/64)A. Thus its projected area stays positive and at
    least 47/64 of the original; no collapse/flip can be introduced.
    """
    vertices=np.asarray(vertices,dtype='float64');faces=np.asarray(faces,dtype='int64')
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():raise ValueError('Finite N,3 vertex positions required')
    if faces.ndim!=2 or faces.shape[1]!=3:raise ValueError('Triangle faces must have shape M,3')
    if len(faces) and (faces.min()<0 or faces.max()>=len(vertices)):raise ValueError('Face index out of range')
    radii=np.full(len(vertices),float(max_displacement),dtype='float64')
    for start in range(0,len(faces),65536):
        indices=faces[start:start+65536];p=vertices[indices]
        e1=p[:,1]-p[:,0];e2=p[:,2]-p[:,0]
        area=np.linalg.norm(np.cross(e1,e2),axis=1)
        lengths=np.stack([np.linalg.norm(e1,axis=1),np.linalg.norm(e2,axis=1),
                          np.linalg.norm(p[:,2]-p[:,1],axis=1)],axis=1)
        length=lengths.sum(1)-lengths.min(1)
        radius=np.divide(area,8*length,out=np.zeros_like(area),where=length>0)
        # A rounding margin protects the analytic bound at Float32 export.
        radius*=.95
        for corner in range(3):np.minimum.at(radii,indices[:,corner],radius)
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0) if len(faces) else np.empty((0,2),dtype='int64')
    return edges,radii


def _teacher_update(points,data,band):
    import torch
    correction=torch.zeros_like(points);normal=torch.zeros_like(points)
    support=torch.zeros(len(points),device=points.device);error=torch.zeros_like(support)
    for frame in data:
        signed,n,w,_,_=_sample_teacher(points,frame)
        valid=torch.isfinite(signed)&(signed.abs()<=band)&(w>0)
        weights=torch.where(valid,w/(1+(signed/(band*.5))**2),0)
        signed=torch.where(valid,signed,torch.zeros_like(signed))
        correction-=weights[:,None]*signed[:,None]*n
        normal+=weights[:,None]*n;support+=weights;error+=weights*signed.abs()
    denominator=support.clamp_min(1e-12)
    correction/=denominator[:,None];error/=denominator
    normal/=torch.linalg.norm(normal,dim=-1,keepdim=True).clamp_min(1e-12)
    return correction,normal,support,error


def refine_vertices_anchored(vertices,frames,faces,band=TRUNCATION,iterations=3,
                            device='CPU:0',chunk_vertices=65536,max_step=.001,
                            max_displacement=TRUNCATION,edge_weight=.25):
    """Move actual vertices using point-to-plane and frozen local tangent goals.

    The target for each directed edge i<-j is the initial teacher projection
    of neighbor j, with the initial teacher normals held fixed. Consequently,
    i depends only on its original one-ring, not on j's later trajectory.
    Complete shared-vertex one-rings produce the same refined positions in
    separate partitions (up to floating-point summation roundoff).

    No face indices, vertex count or colors change. Each actual movement is
    bounded by max_step, max_displacement AND every incident face's analytic
    trust radius. Unobserved vertices remain unchanged.
    """
    import torch
    if band<=0 or iterations<0 or chunk_vertices<1 or max_step<=0 or max_displacement<=0 or edge_weight<0:
        raise ValueError('Invalid local refinement limits')
    initial_numpy=np.asarray(vertices,dtype='float32')
    edges,radii=geometry_constraints(initial_numpy,faces,max_displacement)
    target='cuda:0' if device.startswith('CUDA') else 'cpu'
    data=_prepare_frames(frames,target)
    initial=torch.as_tensor(np.ascontiguousarray(initial_numpy),device=target)
    current=initial.clone();count=len(initial)
    radius=torch.as_tensor(radii.astype('float32'),device=target)
    trust=torch.as_tensor(radii,device=target,dtype=torch.float64)
    normal=torch.zeros_like(initial);anchor=initial.clone();supported=torch.zeros(count,dtype=torch.bool,device=target)
    with torch.no_grad():
        for start in range(0,count,chunk_vertices):
            stop=min(start+chunk_vertices,count)
            correction,n,support,_=_teacher_update(initial[start:stop],data,band)
            anchor[start:stop]+=correction;normal[start:stop]=n;supported[start:stop]=support>0
        edge=torch.as_tensor(edges,device=target,dtype=torch.long)
        a,b=edge[:,0],edge[:,1]
        normals_a=normal[a];normals_b=normal[b]
        valid=supported[a]&supported[b]&((normals_a*normals_b).sum(-1)>=np.cos(np.deg2rad(30.)))
        a,b=a[valid],b[valid]
        nn=normal[a]+normal[b];nn/=torch.linalg.norm(nn,dim=-1,keepdim=True).clamp_min(1e-12)
        # Freeze each vertex's local quadratic tangent objective as H*x=b.
        # Float64 accumulation limits dependence on arbitrary mesh numbering.
        hessian=torch.zeros((count,3,3),device=target,dtype=torch.float64)
        rhs=torch.zeros((count,3),device=target,dtype=torch.float64)
        degree=torch.zeros(count,device=target,dtype=torch.float64)
        for start in range(0,len(a),65536):
            aa=a[start:start+65536];bb=b[start:start+65536];n=nn[start:start+65536].double()
            outer=n[:,:,None]*n[:,None,:]
            hessian.index_add_(0,aa,outer);hessian.index_add_(0,bb,outer)
            rhs.index_add_(0,aa,(anchor[bb].double()*n).sum(-1)[:,None]*n)
            rhs.index_add_(0,bb,(anchor[aa].double()*n).sum(-1)[:,None]*n)
            one=torch.ones(len(aa),device=target,dtype=torch.float64)
            degree.index_add_(0,aa,one);degree.index_add_(0,bb,one)
        summaries=[]
        for iteration in range(iterations):
            updated=current.clone();residual_sum=0.;sample_count=0;quantization_rejected=0
            for start in range(0,count,chunk_vertices):
                stop=min(start+chunk_vertices,count);old=current[start:stop]
                correction,_,support,error=_teacher_update(old,data,band)
                if edge_weight:
                    tangent=rhs[start:stop]-torch.bmm(hessian[start:stop],old.double().unsqueeze(-1)).squeeze(-1)
                    tangent/=degree[start:stop].clamp_min(1)[:,None]
                    # The local tangent residual has bounded influence, while
                    # the trust region bounds displacement from the original.
                    norm=torch.linalg.norm(tangent,dim=-1,keepdim=True)
                    tangent*=torch.clamp(band/norm.clamp_min(1e-12),max=1)
                    correction+=edge_weight*tangent.float()
                correction=torch.where((support>0)[:,None],correction,torch.zeros_like(correction))
                step=torch.linalg.norm(correction,dim=-1,keepdim=True)
                correction*=torch.clamp(max_step/step.clamp_min(1e-12),max=1)
                candidate=old.double()+correction.double()
                displacement=candidate-initial[start:stop].double()
                distance=torch.linalg.norm(displacement,dim=-1,keepdim=True)
                displacement*=torch.clamp(trust[start:stop,None]/distance.clamp_min(1e-15),max=1)
                rounded=(initial[start:stop].double()+displacement).float()
                # The exported mesh stores Float32 positions. At scene
                # coordinates of tens of units, one Float32 ulp can exceed a
                # sliver triangle's analytic trust radius. Check the ACTUAL
                # quantized movement, not just the preceding Float64 proposal.
                actual_displacement=torch.linalg.norm(rounded.double()-initial[start:stop].double(),dim=-1)
                actual_step=torch.linalg.norm(rounded.double()-old.double(),dim=-1)
                accepted=torch.isfinite(rounded).all(-1)&(actual_displacement<=trust[start:stop])&(actual_step<=max_step)
                quantization_rejected+=int((~accepted).sum())
                # The old position passed the same per-vertex check on the
                # preceding iteration. A rejection therefore preserves the
                # trust bound without affecting any neighboring trajectory.
                updated[start:stop]=torch.where(accepted[:,None],rounded,old)
                good=support>0
                if good.any():residual_sum+=float(error[good].sum());sample_count+=int(good.sum())
            movement=torch.linalg.norm(updated.double()-current.double(),dim=-1)
            summaries.append(dict(iteration=iteration+1,supported_vertices=sample_count,
                mean_supported_point_plane_residual=residual_sum/max(sample_count,1),
                max_step=float(movement.max()) if count else 0.,
                float32_quantized_vertex_moves_rejected=quantization_rejected))
            current=updated
    result=current.cpu().numpy().copy();distances=np.linalg.norm(result.astype('float64')-initial_numpy.astype('float64'),axis=1)
    f=np.asarray(faces,dtype='int64');violations=0;minimum_ratio=1.
    for start in range(0,len(f),65536):
        fi=f[start:start+65536];before=initial_numpy[fi].astype('float64');after=result[fi].astype('float64')
        old=np.cross(before[:,1]-before[:,0],before[:,2]-before[:,0]);new=np.cross(after[:,1]-after[:,0],after[:,2]-after[:,0])
        square=(old*old).sum(-1);valid=square>1e-30
        ratio=np.divide((old*new).sum(-1),square,out=np.ones_like(square),where=valid)
        if valid.any():minimum_ratio=min(minimum_ratio,float(ratio[valid].min()))
        violations+=int((valid&(ratio<=0)).sum())
    if violations:raise ValueError('Analytic geometry trust region was violated after numeric export')
    receipt=dict(iterations=iterations,vertices=count,faces=len(f),faces_unchanged=True,vertex_count_unchanged=True,
        actual_vertex_positions_optimized=True,source_gaussians_updated=False,ground_truth_used=False,test_rgb_used=False,
        point_plane_data_term=True,actual_mesh_edge_tangent_term=bool(edge_weight and len(a)),
        neighbors_and_tangent_targets_frozen=True,initial_one_ring_only=True,
        partition_global_rollback=False,partition_independent_objective=True,
        float32_quantized_displacement_bounds_checked=True,
        observation_weight='confidence * incidence; alpha is validity gate only',
        synchronous_updates=True,band=band,max_step_limit=max_step,total_displacement_limit=max_displacement,
        per_face_analytic_trust_region='0.95 * doubled_area / (8 * largest_sum_of_two_edge_lengths)',
        minimum_original_normal_projected_area_ratio=minimum_ratio,flipped_faces=violations,
        maximum_displacement=float(distances.max(initial=0)),moved_vertices=int((distances>1e-8).sum()),
        min_vertex_trust_radius=float(radii.min(initial=max_displacement)),
        differentiable_rasterizer_used=False,global_laplacian_smoothing=False,colors_optimized=False,
        history=summaries,edge_weight=edge_weight)
    del current,initial,data,edge,hessian,rhs,normal,anchor,trust;gc.collect()
    if device.startswith('CUDA'):torch.cuda.empty_cache()
    return result,receipt
