"""Normal-aware fusion and bounded geometric refinement for isolated experiments.

Inputs use camera z-depth in scene units, camera-space normals and world-to-
camera transforms. Normals are normalized and oriented towards the observing
camera. Confidence is a separate scalar image, never the normal's magnitude.
All input frames, original pixels, volume blocks and mesh vertices participate.
This module does not modify production rendering, training or extraction.
"""
import gc
import numpy as np

from .fine_tsdf import VOXEL,TRUNCATION,BLOCK
from .surface_fusion_probe import continuous_cells


def _prepare_frames(frames,device,alpha_min=.5):
    import torch
    prepared=[]
    for frame in frames:
        d=np.asarray(frame['depth'],dtype='float32');h,w=d.shape
        if h<2 or w<2:raise ValueError('At least two rows and columns are required')
        n=np.asarray(frame['normal'],dtype='float32')
        if n.shape==(3,h,w) and n.shape!=(h,w,3):n=n.transpose(1,2,0)
        if n.shape!=(h,w,3):raise ValueError('normal must have shape H,W,3 or 3,H,W')
        alpha=np.asarray(frame.get('alpha',np.ones_like(d)),dtype='float32').squeeze()
        confidence=np.asarray(frame.get('confidence',np.ones_like(d)),dtype='float32').squeeze()
        if alpha.shape!=d.shape or confidence.shape!=d.shape:raise ValueError('Scalar geometry image shape mismatch')
        length=np.linalg.norm(n,axis=-1)
        valid=np.isfinite(d)&(d>0)&np.isfinite(n).all(-1)&(length>1e-8)
        valid&=np.isfinite(alpha)&(alpha>=alpha_min)&np.isfinite(confidence)&(confidence>0)
        n=np.divide(n,length[...,None],out=np.zeros_like(n),where=length[...,None]>1e-8)
        K=np.asarray(frame['K'],dtype='float64');T=np.asarray(frame['w2c'],dtype='float64')
        if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all() or K[0,0]<=0 or K[1,1]<=0:
            raise ValueError('Finite pinhole intrinsics and world-to-camera pose required')
        rgb=np.asarray(frame['rgb'],dtype='float32')
        if rgb.shape!=(h,w,3) or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>255:
            raise ValueError('rgb must have shape H,W,3 and native RGB8 range')
        yy,xx=np.mgrid[:h,:w];rr=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],-1)
        n=np.where((n*rr).sum(-1,keepdims=True)>0,-n,n)
        n[~valid]=0
        agreement=np.ones((h-1,w-1),dtype=bool)
        corners=[n[:-1,:-1],n[:-1,1:],n[1:,:-1],n[1:,1:]]
        for i in range(1,4):agreement&=(corners[0]*corners[i]).sum(-1)>=np.cos(np.deg2rad(30.))
        continuity=continuous_cells(d)&agreement
        continuity&=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
        tensor=lambda a:torch.as_tensor(np.ascontiguousarray(a),device=device)
        prepared.append(dict(depth=tensor(np.where(valid,d,0)),normal=tensor(n),
            alpha=tensor(np.where(valid,np.clip(alpha,0,1),0)),
            confidence=tensor(np.where(valid,confidence,0)),valid=tensor(valid),
            continuity=tensor(continuity),rgb=tensor(rgb),
            R=tensor(T[:3,:3].astype('float32')),t=tensor(T[:3,3].astype('float32')),K=K))
    return prepared


def _sample_teacher(world,frame,minimum_incidence=.05):
    """Depth/normal interpolation only inside mutually continuous valid cells."""
    import torch
    q=world@frame['R'].T+frame['t'];z=q[:,2];K=frame['K'];h,w=frame['depth'].shape
    safe=torch.where(z.abs()>1e-12,z,torch.ones_like(z))
    u=q[:,0]*float(K[0,0])/safe+float(K[0,2]);v=q[:,1]*float(K[1,1])/safe+float(K[1,2])
    valid=(z>0)&torch.isfinite(q).all(-1)&(u>=0)&(u<w)&(v>=0)&(v<h)
    x=torch.floor(u).long().clamp(0,w-1);y=torch.floor(v).long().clamp(0,h-1)
    valid&=frame['valid'][y,x]
    d=frame['depth'][y,x];n=frame['normal'][y,x]
    cx=x.clamp(0,w-2);cy=y.clamp(0,h-2)
    interpolate=valid&(x<w-1)&(y<h-1)&frame['continuity'][cy,cx]
    a=(u-cx).clamp(0,1);b=(v-cy).clamp(0,1)
    weights=torch.stack([(1-a)*(1-b),a*(1-b),(1-a)*b,a*b],-1)
    dd=torch.stack([frame['depth'][cy,cx],frame['depth'][cy,cx+1],frame['depth'][cy+1,cx],frame['depth'][cy+1,cx+1]],-1)
    rho=(weights/dd.clamp_min(1e-12)).sum(-1)
    d=torch.where(interpolate,1/rho.clamp_min(1e-12),d)
    nn=torch.stack([frame['normal'][cy,cx],frame['normal'][cy,cx+1],frame['normal'][cy+1,cx],frame['normal'][cy+1,cx+1]],-2)
    interpolated=(weights[...,None]*nn).sum(-2)
    n=torch.where(interpolate[:,None],interpolated,n)
    n=n/torch.linalg.norm(n,dim=-1,keepdim=True).clamp_min(1e-12)
    ray=torch.stack([(u-float(K[0,2]))/float(K[0,0]),(v-float(K[1,2]))/float(K[1,1]),torch.ones_like(u)],-1)
    n=torch.where((n*ray).sum(-1,keepdim=True)>0,-n,n)
    ray_length=torch.linalg.norm(ray,dim=-1)
    denominator=-(n*ray).sum(-1)
    incidence=denominator/ray_length.clamp_min(1e-12)
    valid&=(d>0)&torch.isfinite(d)&(incidence>=minimum_incidence)
    surface=ray*d[:,None]
    # n faces the observing camera: points before the surface have positive SDF.
    signed=(n*(q-surface)).sum(-1)
    behind_tolerance=TRUNCATION/denominator.clamp_min(1e-6)
    valid&=(d-z>=-behind_tolerance-1e-7)&(signed>=-TRUNCATION-1e-7)
    # confidence already contains the caller's alpha/normal-agreement score.
    # Alpha is only an observation-validity gate, not a second weight factor.
    confidence=frame['confidence'][y,x]*incidence
    confidence=torch.where(valid,confidence,torch.zeros_like(confidence))
    # Row-vector normal rotation from camera to world.
    normal_world=n@frame['R']
    return signed,normal_world,confidence,frame['rgb'][y,x],d-z


def integrate_normal_partition(active,frames,device='CUDA:0',chunk_blocks=64,alpha_min=.5):
    """Fuse observed local tangent planes on the original .002/.008 lattice.

    Returns ``vertices[N,3], faces[M,3], colors[N,3]`` (colors in [0,1]).
    Each frame adds confidence*incidence weight. Alpha only gates valid data.
    The caller's confidence is the complete observation reliability score,
    including any intended alpha and normal-agreement factors. Positive free-space
    evidence is retained; evidence behind a visible surface stops at its
    normal-distance truncation band, rather than fusing an occluded layer.
    RGB uses the same observation weights as geometry; color differences are
    therefore possible even without texture fitting or changes to input RGB.
    Opposing observations of the same zero-thickness two-sided sheet still
    require explicit layer/orientation handling outside this local field;
    this experiment does not claim a globally consistent occupancy model.
    """
    import torch,open3d as o3d
    if chunk_blocks<1:raise ValueError('chunk_blocks must be positive')
    target='cuda:0' if device.startswith('CUDA') else 'cpu';dev=o3d.core.Device(device)
    data=_prepare_frames(frames,target,alpha_min)
    volume=o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),
        attr_dtypes=(o3d.core.float32,)*3,attr_channels=((1,),(1,),(3,)),
        voxel_size=VOXEL,block_resolution=BLOCK,block_count=max(8,len(active)*2),device=dev)
    coords=o3d.core.Tensor(np.ascontiguousarray(active,dtype='int32'),o3d.core.int32,dev)
    volume.hashmap().activate(coords);buffers,_=volume.hashmap().find(coords)
    share=lambda a:torch.utils.dlpack.from_dlpack(a.to_dlpack())
    tsdf=share(volume.attribute('tsdf')).reshape(-1)
    weight=share(volume.attribute('weight')).reshape(-1)
    color=share(volume.attribute('color')).reshape(-1,3)
    with torch.no_grad():
        for start in range(0,len(active),chunk_blocks):
            pos,index=volume.voxel_coordinates_and_flattened_indices(buffers[start:start+chunk_blocks])
            xyz=share(pos);indices=share(index).long()
            total=torch.zeros(len(xyz),device=target);value=torch.zeros_like(total)
            rgb=torch.zeros((len(xyz),3),device=target)
            for frame in data:
                signed,_,weights,sample,_=_sample_teacher(xyz,frame)
                valid=weights>0
                total+=weights
                value+=torch.where(valid,(signed/TRUNCATION).clamp(-1,1),0)*weights
                rgb+=sample*weights[:,None]
            denominator=total.clamp_min(1e-12)
            tsdf[indices]=value/denominator;weight[indices]=total;color[indices]=rgb/denominator[:,None]
            del pos,index,xyz,indices,total,value,rgb,signed,weights,sample
    del data,tsdf,weight,color;gc.collect()
    if device.startswith('CUDA'):torch.cuda.empty_cache()
    mesh=volume.extract_triangle_mesh(weight_threshold=0.)
    if 'positions' not in mesh.vertex or len(mesh.vertex.positions)==0:
        result=(np.empty((0,3),dtype='float32'),np.empty((0,3),dtype='int64'),np.empty((0,3),dtype='float32'))
    else:result=tuple(a.cpu().numpy().copy() for a in [mesh.vertex.positions,mesh.triangle.indices,mesh.vertex.colors])
    del mesh,volume,coords,buffers;gc.collect()
    if device.startswith('CUDA'):o3d.core.cuda.synchronize();o3d.core.cuda.release_cache()
    return result


def refine_vertices_local(vertices,frames,faces=None,band=TRUNCATION,iterations=3,
                          device='CPU:0',chunk_vertices=65536,max_step=.001,
                          max_displacement=TRUNCATION,edge_weight=.25):
    """Bounded point-to-plane plus actual mesh-edge tangent refinement.

    This is a geometric optimization of vertex positions, not a shading-normal
    edit and not full differentiable rasterization. Nearby teacher normals
    constrain ``edge dot normal`` only when their angle is <=30 degrees.
    Every iteration reads the complete old vertex array, computes all updates,
    then writes the next array synchronously. Chunk size never controls which
    neighbors are used. Faces/vertex count and RGB remain unchanged.

    Returns ``(new_vertices, receipt)``. Total and per-step displacement are
    bounded in COLMAP scene units. Teacher evidence is training-frame only.
    """
    import torch
    if band<=0 or iterations<0 or max_step<=0 or max_displacement<=0 or chunk_vertices<1:
        raise ValueError('Invalid refinement limits')
    vertices=np.asarray(vertices)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
        raise ValueError('Finite vertex positions with shape N,3 required')
    target='cuda:0' if device.startswith('CUDA') else 'cpu'
    data=_prepare_frames(frames,target)
    initial=torch.as_tensor(np.ascontiguousarray(vertices,dtype='float32'),device=target)
    current=initial.clone();count=len(vertices)
    if faces is not None and len(faces):
        f=np.asarray(faces,dtype='int64')
        if f.min()<0 or f.max()>=count:raise ValueError('Invalid face indices')
        edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
        edge=torch.as_tensor(edges,device=target,dtype=torch.long)
        triangle=torch.as_tensor(f,device=target,dtype=torch.long)
    else:
        edge=torch.empty((0,2),device=target,dtype=torch.long)
        triangle=torch.empty((0,3),device=target,dtype=torch.long)
    summaries=[]
    with torch.no_grad():
        for iteration in range(iterations):
            delta=torch.zeros_like(current);normal=torch.zeros_like(current)
            support=torch.zeros(count,device=target);errors=torch.zeros(count,device=target)
            # Data updates use one immutable iterate. They do not see any
            # position changes from earlier chunks in the same iteration.
            for start in range(0,count,chunk_vertices):
                stop=min(start+chunk_vertices,count);old=current[start:stop]
                correction=torch.zeros_like(old);normal_sum=torch.zeros_like(old)
                weights_sum=torch.zeros(len(old),device=target);error_sum=torch.zeros_like(weights_sum)
                for frame in data:
                    signed,n,w,_,_=_sample_teacher(old,frame)
                    accepted=torch.isfinite(signed)&(signed.abs()<=band)&(w>0)
                    # Bounded influence across disagreeing but supported views.
                    w=torch.where(accepted,w/(1+(signed/(band*.5))**2),0)
                    correction-=w[:,None]*signed[:,None]*n
                    normal_sum+=w[:,None]*n;weights_sum+=w;error_sum+=w*signed.abs()
                denominator=weights_sum.clamp_min(1e-12)
                delta[start:stop]=correction/denominator[:,None]
                normal[start:stop]=normal_sum/torch.linalg.norm(normal_sum,dim=-1,keepdim=True).clamp_min(1e-12)
                support[start:stop]=weights_sum;errors[start:stop]=error_sum/denominator
            # An unoriented edge must be tangent to the nearby teacher surface.
            # This moves actual positions while excluding sharp normal changes.
            edge_gradient=torch.zeros_like(current);degree=torch.zeros(count,device=target)
            accepted_edges=0
            if len(edge) and edge_weight:
                a,b=edge[:,0],edge[:,1];na,nb=normal[a],normal[b]
                valid=(support[a]>0)&(support[b]>0)&((na*nb).sum(-1)>=np.cos(np.deg2rad(30.)))
                a,b=a[valid],b[valid];n=normal[a]+normal[b]
                n=n/torch.linalg.norm(n,dim=-1,keepdim=True).clamp_min(1e-12)
                residual=((current[b]-current[a])*n).sum(-1)
                # Avoid an inconsistent edge overwhelming reliable depth data.
                residual=residual.clamp(-band,band);gradient=residual[:,None]*n
                edge_gradient.index_add_(0,a,gradient);edge_gradient.index_add_(0,b,-gradient)
                one=torch.ones(len(a),device=target);degree.index_add_(0,a,one);degree.index_add_(0,b,one)
                delta+=edge_weight*edge_gradient/degree.clamp_min(1)[:,None]
                accepted_edges=len(a)
            length=torch.linalg.norm(delta,dim=-1,keepdim=True)
            delta*=torch.clamp(max_step/length.clamp_min(1e-12),max=1)
            next_position=current+delta
            displacement=next_position-initial;length=torch.linalg.norm(displacement,dim=-1,keepdim=True)
            next_position=initial+displacement*torch.clamp(max_displacement/length.clamp_min(1e-12),max=1)
            next_position=torch.where((support>0)[:,None],next_position,current)
            # Preserve triangle orientation and prevent newly collapsed faces.
            # Reject unsafe vertex moves rather than changing any triangle.
            rolled_back=torch.zeros(count,dtype=torch.bool,device=target)
            rejected_faces=0;whole_step_rejected=False
            for attempt in range(9):
                bad_vertices=[];bad_count=0
                for begin in range(0,len(triangle),65536):
                    tri=triangle[begin:begin+65536];before=current[tri];after=next_position[tri]
                    old_cross=torch.linalg.cross(before[:,1]-before[:,0],before[:,2]-before[:,0])
                    new_cross=torch.linalg.cross(after[:,1]-after[:,0],after[:,2]-after[:,0])
                    old_area=torch.linalg.norm(old_cross,dim=-1);new_area=torch.linalg.norm(new_cross,dim=-1)
                    bad=(old_area>1e-16)&(((old_cross*new_cross).sum(-1)<=0)|(new_area<old_area*.1))
                    if bad.any():bad_vertices.append(tri[bad].reshape(-1));bad_count+=int(bad.sum().item())
                if not bad_count:break
                rejected_faces+=bad_count
                if attempt==8:
                    next_position=current.clone();whole_step_rejected=True;break
                affected=torch.unique(torch.cat(bad_vertices));rolled_back[affected]=True
                next_position[affected]=current[affected]
            supported=support>0
            summaries.append(dict(iteration=iteration+1,supported_vertices=int(supported.sum().item()),
                mean_supported_point_plane_residual=float(errors[supported].mean().item()) if supported.any() else None,
                tangent_edges=accepted_edges,max_step=float(torch.linalg.norm(next_position-current,dim=-1).max().item()) if count else 0,
                flipped_or_collapsed_face_attempts=rejected_faces,vertex_moves_rejected=int(rolled_back.sum().item()),
                whole_step_rejected=whole_step_rejected))
            current=next_position
    result=current.cpu().numpy().copy()
    distance=np.linalg.norm(result-np.asarray(vertices),axis=1)
    receipt=dict(iterations=iterations,vertices=count,faces_unchanged=True,vertex_count_unchanged=True,
        source_gaussians_updated=False,ground_truth_used=False,test_rgb_used=False,
        point_plane_data_term=True,actual_mesh_edge_tangent_term=bool(len(edge) and edge_weight),
        observation_weight='confidence * incidence; alpha is a validity gate only',
        differentiable_rasterizer_used=False,global_laplacian_smoothing=False,
        synchronous_updates=True,band=band,max_step_limit=max_step,total_displacement_limit=max_displacement,
        triangle_flip_and_collapse_protection=True,
        maximum_displacement=float(distance.max(initial=0)),moved_vertices=int((distance>1e-8).sum()),
        history=summaries,edge_weight=edge_weight)
    del current,initial,data,edge,triangle;gc.collect()
    if device.startswith('CUDA'):torch.cuda.empty_cache()
    return result,receipt
