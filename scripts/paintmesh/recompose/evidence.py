"""Build object observations from real training images, never LaMa hole masks."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
from scipy.spatial import cKDTree
from .io import complete, read_json, write_json, record_path, fingerprint, ply, xyz, crop_mesh, save_mesh,crop_reconstruction
from .geometry import RayMesh, camera_rays, backproject, project, support_up

SELECTION={'truck':'fitlama','doppelherz':'fitlama','bag':'fitlama','fruits':'fitlama',
           'car':'best_results','cone_red':'fit','cone_yellow':'fit','cube':'fit',
           'garden_toys':'fit','redbull':'fit','toys':'fit'}


def resolve_scene(root,scene):
    if scene not in SELECTION:raise ValueError(f'Unknown scene: {scene}')
    root=Path(root).resolve(); group=SELECTION[scene]
    run=root/group/scene/'run' if group=='best_results' else root/group/'scenes'/scene/'inpaint'
    path=run/'inpaint_manifest.json';m=complete(path)
    workspace=complete(record_path(m['inputs']['workspace_manifest']))
    removal_workspace=complete(record_path(workspace['inputs']['removal_workspace_manifest']))
    smp=record_path(removal_workspace['inputs']['semantic_manifest']);sm=complete(smp)
    return dict(scene=scene,group=group,run=str(run),manifest=str(path),manifest_identity=m['artifact_id'],
                target_ids=m['target_ids'],source=m['parameters']['mesh']['source_path'],
                background_mesh=str(record_path(m['outputs']['pgsr_mesh'])),
                background_gaussians=str(record_path(m['outputs']['inpainted_gaussian_ply'])),
                original_mesh=str(record_path(sm['inputs']['mesh'])),
                original_gaussians=str(record_path(sm['inputs']['gaussian_ply'])),
                gaussian_labels=str(record_path(sm['outputs']['gaussian_label'],smp.parent)),
                gaussian_confidence=str(record_path(sm['outputs']['gaussian_confidence'],smp.parent)),
                semantic_manifest=str(smp),
                mask_root=str(record_path(sm['inputs']['scene_info']).parent),
                residual_labels=m.get('target_residuals'),
                note='semantic label counts are not proof of geometric object remnants')


def original_cameras(repo,source,max_size):
    file=Path(repo)/'submodules/Inpaint360GS/tools/read_write_model.py'
    spec=importlib.util.spec_from_file_location('paintmesh_colmap',file);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    sparse=Path(source)/'sparse/0'
    cameras,images,_=mod.read_model(str(sparse),ext='.bin' if (sparse/'cameras.bin').exists() else '.txt')
    result=[]
    for im in sorted(images.values(),key=lambda x:x.name):
        if Path(im.name).name.lower().startswith('test'):continue
        c=cameras[im.camera_id];p=c.params
        if c.model=='SIMPLE_PINHOLE':fx,fy,cx,cy=p[0],p[0],p[1],p[2]
        elif c.model=='PINHOLE':fx,fy,cx,cy=p
        else:raise ValueError(f'Undistorted PINHOLE cameras required; got {c.model}')
        image=Path(source)/'images'/im.name
        if not image.exists():raise FileNotFoundError(image)
        with Image.open(image) as pic:ow,oh=pic.size
        # COLMAP dimensions and decoded image dimensions must share one pixel coordinate system.
        if (ow,oh)!=(c.width,c.height):raise ValueError(f'COLMAP/image dimensions differ: {image}')
        factor=min(1.,max_size/max(ow,oh));w,h=round(ow*factor),round(oh*factor)
        K=np.array([[fx*w/ow,0,cx*w/ow],[0,fy*h/oh,cy*h/oh],[0,0,1.]])
        T=np.eye(4);T[:3,:3]=im.qvec2rotmat();T[:3,3]=im.tvec
        result.append(dict(name=im.name,image=str(image),width=w,height=h,K=K,w2c=T,
                           native_width=ow,native_height=oh,
                           native_K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]),
                           center=np.linalg.inv(T)[:3,3]))
    return result


def binary_mask(path,target_id,size):
    a=np.array(Image.open(path))
    if a.ndim!=2:raise ValueError(f'Instance labels, not RGB color masks, required: {path}')
    return cv2.resize((a==target_id).astype('uint8'),size,interpolation=cv2.INTER_NEAREST).astype(bool)


def native_generation_view(camera,mask_path,target_id,ray_mesh,output,name,with_pointmap=True,chunk_pixels=262144):
    """Preserve decoded source RGB; let the official model crop BEFORE its 518 resize.

    Pose/validation observations have their own fixed raster and must not change
    with generation resolution. Raycast this same source mesh at native camera
    pixels instead of upsampling low-resolution XYZ (which changes camera rays).
    Chunked rays bound peak host memory, without changing the sampling grid.
    """
    out=Path(output);w,h=camera['native_width'],camera['native_height']
    K=np.asarray(camera['native_K']);T=np.asarray(camera['w2c'])
    with Image.open(camera['image']) as source:
        if source.size!=(w,h):raise ValueError('Native source image/camera dimensions changed')
        rgb=source.convert('RGB');rgb.save(out/'images'/name)
        mask=binary_mask(mask_path,target_id,(w,h))
        rgba=rgb.copy();rgba.putalpha(Image.fromarray(mask.astype('uint8')*255))
        rgba.save(out/'object'/name)
    record=dict(policy='native_rgb_official_crop',width=w,height=h,K=K,w2c=T,
                rgb_resampled=False,mask_pixels=int(mask.sum()),
                pointmap_source='same pre-removal PGSR proxy, native calibrated rays')
    if not with_pointmap:return record,None,None
    depth=np.empty(h*w,dtype='float32');pointmap=np.empty((3,h*w),dtype='float32')
    inverse_K=np.linalg.inv(K)
    for lo in range(0,h*w,chunk_pixels):
        hi=min(lo+chunk_pixels,h*w);index=np.arange(lo,hi)
        uv=np.c_[index%w,index//w]
        z,_,_=ray_mesh.cast(camera_rays(K,T,uv))
        valid=np.isfinite(z)&(z>0);depth[lo:hi]=np.where(valid,z,0)
        camera_points=(np.c_[uv,np.ones(len(uv))]@inverse_K.T)*np.where(valid,z,0)[:,None]
        camera_points[~valid]=np.nan;pointmap[:,lo:hi]=camera_points.T
    return record,pointmap.reshape(3,h,w),depth.reshape(h,w)


def select_views(cameras,mask_root,target_id,center,count,min_area):
    candidates=[]
    for c in cameras:
        path=Path(mask_root)/(Path(c['name']).stem+'.png')
        if not path.exists():continue
        mask=binary_mask(path,target_id,(c['width'],c['height']))
        if mask.sum()<min_area:continue
        border=np.r_[mask[0],mask[-1],mask[:,0],mask[:,-1]].mean()
        direction=c['center']-center;direction/=max(np.linalg.norm(direction),1e-12)
        candidates.append((c,mask,direction,float(np.sqrt(mask.mean())*(1-border))))
    if len(candidates)<count:raise ValueError(f'Need {count} usable distinct training views, found {len(candidates)} for object {target_id}')
    chosen=[int(np.argmax([x[3] for x in candidates]))]
    while len(chosen)<min(count,len(candidates)):
        scores=[]
        for i,c in enumerate(candidates):
            separation=min(1-float(c[2]@candidates[j][2]) for j in chosen)
            scores.append(-1 if i in chosen else separation*np.sqrt(c[3]))
        chosen.append(int(np.argmax(scores)))
    return [(candidates[i][0],candidates[i][1]) for i in chosen]


def mask_consistent_centers(points,cameras,mask_root,target_id):
    """Reject semantic spill before defining the ROI; masks constrain a visual volume."""
    votes=np.zeros(len(points));seen=np.zeros(len(points));used=0
    for c in cameras[::max(1,len(cameras)//32)]:
        path=Path(mask_root)/(Path(c['name']).stem+'.png')
        if not path.exists():continue
        mask=binary_mask(path,target_id,(c['width'],c['height']))
        if mask.sum()<100:continue
        used+=1;uv,z=project(points,c['K'],c['w2c']);u,y=np.rint(uv).astype(int).T
        inside=(z>0)&(u>=0)&(u<c['width'])&(y>=0)&(y<c['height']);idx=np.flatnonzero(inside)
        seen[idx]+=1;votes[idx]+=mask[y[idx],u[idx]]
    keep=(seen>=min(4,used))&(votes>=3)&(votes>=.6*np.maximum(seen,1))
    if keep.sum()<30:raise ValueError('Semantic object disagrees with multiview instance masks; inspect target identity')
    return points[keep]


def prepare(repo,inputs,output,cfg,target_id):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if target_id not in inputs['target_ids']:raise ValueError('Object was not part of this removal')
    label=np.load(inputs['gaussian_labels'],mmap_mode='r');conf=np.load(inputs['gaussian_confidence'],mmap_mode='r')
    v=ply(inputs['original_gaussians'])['vertex'].data
    if len(v)!=len(label):raise ValueError('Gaussian semantics are not aligned')
    centers=xyz(v[(label==target_id)&(conf>=cfg['semantic_min'])]);centers=centers[np.isfinite(centers).all(1)]
    if len(centers)<30:raise ValueError('Insufficient object geometry for a bounded search region')
    cameras=original_cameras(repo,inputs['source'],cfg['observation_max_size'])
    count_before=len(centers);centers=mask_consistent_centers(centers,cameras,inputs['mask_root'],target_id)
    print(f'Object {target_id}: {len(centers)}/{count_before} centers agree with real multiview masks',flush=True)
    lower,upper=np.quantile(centers,[.02,.98],axis=0);diameter=float(np.linalg.norm(upper-lower));center=(lower+upper)/2
    if diameter<=1e-8:raise ValueError('Degenerate object extent')
    lo=lower-diameter*cfg['roi_margin'];hi=upper+diameter*cfg['roi_margin']
    selected=select_views(cameras,inputs['mask_root'],target_id,center,cfg['generation_views']+cfg['validation_views'],cfg['min_mask_pixels'])
    print(f'Object {target_id}: crop original/background local meshes',flush=True)
    original,original_source,original_recovered=crop_reconstruction(inputs['original_mesh'],lo,hi,cfg['proxy_faces'],inputs['original_gaussians'])
    background,background_source,background_recovered=crop_reconstruction(inputs['background_mesh'],lo,hi,cfg['proxy_faces'],inputs['background_gaussians'])
    save_mesh(out/'original_proxy.ply',*original);save_mesh(out/'background_proxy.ply',*background)
    original_ray=RayMesh(*original[:2]);background_ray=RayMesh(*background[:2])
    for d in ['images','object','observations','review']:(out/d).mkdir(exist_ok=True)
    depths=[];pointmaps=[];Ks=[];Ts=[];names=[];views=[];point_chunks=[];normal_chunks=[];ring_normals=[];support_chunks=[];support_weights=[]
    rng=np.random.default_rng(cfg['seed'])
    for i,(c,mask) in enumerate(selected):
        print(f'Object {target_id}: observation {i+1}/{len(selected)} {c["name"]}',flush=True)
        w,h=c['width'],c['height']; yy,xx=np.mgrid[:h,:w];rays=camera_rays(c['K'],c['w2c'],np.c_[xx.ravel(),yy.ravel()])
        depth,face,normal=original_ray.cast(rays);depth=depth.reshape(h,w);normal=normal.reshape(h,w,3)
        bg_depth,bg_face,bg_normal=background_ray.cast(rays);bg_depth=bg_depth.reshape(h,w);bg_normal=bg_normal.reshape(h,w,3)
        # Orient local support normals toward visible cameras, never assume world-Z is up.
        sign=np.where((bg_normal.reshape(-1,3)*rays[:,3:]).sum(1)>0,-1.,1.)
        bg_normal*=sign.reshape(h,w,1)
        inner=cv2.erode(mask.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
        valid=inner&np.isfinite(depth)&(depth>0)
        coverage=float(valid.sum()/max(inner.sum(),1))
        if coverage<cfg['min_depth_coverage']:raise ValueError(f'{c["name"]}: object mask/depth coverage {coverage:.3f}; review masks or increase ROI')
        d=np.where(np.isfinite(depth),depth,0.).astype('float32');points=backproject(d,c['K'],c['w2c'])
        indices=np.argwhere(valid);indices=indices[rng.choice(len(indices),min(cfg['points_per_view'],len(indices)),replace=False)]
        p=points[indices[:,0],indices[:,1]];n=normal[indices[:,0],indices[:,1]]
        is_validation=i>=cfg['generation_views']
        if not is_validation:point_chunks.append(p);normal_chunks.append(n)
        ring=cv2.dilate(mask.astype('uint8'),np.ones((21,21),'uint8')).astype(bool)&~mask
        bgpoints=backproject(np.where(np.isfinite(bg_depth),bg_depth,0),c['K'],c['w2c'])
        difference=np.full_like(depth,1e6);both_valid=np.isfinite(bg_depth)&np.isfinite(depth)
        np.subtract(bg_depth,depth,out=difference,where=both_valid)
        reliable=ring&both_valid&(np.abs(difference)<cfg['support_match_fraction']*diameter)
        if not is_validation:ring_normals.append(bg_normal[reliable])
        # Distinguish observed support from hallucinated completion; labels are not confidence.
        support=mask|ring;support&=np.isfinite(bg_depth)&(bg_depth>0)
        if not is_validation:
            support_chunks.append(bgpoints[support]);support_weights.append(np.where(reliable[support],1.,cfg['completed_support_weight']))
        image=np.asarray(Image.open(c['image']).convert('RGB').resize((w,h),Image.Resampling.LANCZOS))
        name=f'{i:05d}.png'
        print(f'Object {target_id}: native generation RGB {c["native_width"]}x{c["native_height"]}; pose observations {w}x{h}',flush=True)
        generation,pm,native_depth=native_generation_view(c,Path(inputs['mask_root'])/(Path(c['name']).stem+'.png'),
            target_id,original_ray,out,name,with_pointmap=not is_validation)
        np.savez_compressed(out/'observations'/f'{i:05d}.npz',mask=mask,depth=d,background_depth=np.where(np.isfinite(bg_depth),bg_depth,0),K=c['K'],w2c=c['w2c'],rgb=image)
        overlay=image.copy();overlay[mask]=(overlay[mask]*.55+np.array([50,220,60])*.45).astype('uint8')
        Image.fromarray(overlay).save(out/'review'/name)
        views.append(dict(name=name,original=c['name'],K=c['K'],w2c=c['w2c'],width=w,height=h,
                          validation=is_validation,depth_coverage=coverage,mask_pixels=int(mask.sum()),generation=generation,
                          image=fingerprint(c['image']),mask=fingerprint(Path(inputs['mask_root'])/(Path(c['name']).stem+'.png'))))
        views[-1]['prepared_image']=fingerprint(out/'images'/name)
        views[-1]['prepared_mask']=fingerprint(out/'object'/name)
        views[-1]['observation']=fingerprint(out/'observations'/f'{i:05d}.npz')
        if not is_validation:
            depths.append(native_depth);pointmaps.append(pm);Ks.append(c['native_K']);Ts.append(c['w2c']);names.append(name)
    # Votes require at least two actual input cameras; interpolated views never count as extra evidence.
    points=np.concatenate(point_chunks);normals=np.concatenate(normal_chunks);votes=np.zeros(len(points),dtype=int)
    for view in views:
        if view['validation']:continue
        o=np.load(out/'observations'/(Path(view['name']).stem+'.npz'))
        uv,z=project(points,o['K'],o['w2c']);u=np.rint(uv[:,0]).astype(int);vv=np.rint(uv[:,1]).astype(int)
        inside=(z>0)&(u>=0)&(u<view['width'])&(vv>=0)&(vv<view['height']);idx=np.flatnonzero(inside)
        agree=o['mask'][vv[idx],u[idx]]&(o['depth'][vv[idx],u[idx]]>0)&(np.abs(o['depth'][vv[idx],u[idx]]-z[idx])<cfg['depth_agreement_fraction']*diameter)
        votes[idx[agree]]+=1
    keep=votes>=cfg['minimum_views'];points=points[keep];normals=normals[keep]
    if len(points)<cfg['minimum_anchor_points']:raise ValueError('Too few multiview-consistent object anchors')
    n=min(len(points),cfg['anchor_points']);idx=rng.choice(len(points),n,replace=False);points=points[idx];normals=normals[idx]
    sp=np.concatenate(support_chunks);sw=np.concatenate(support_weights);keys=np.floor(sp/(diameter*.005)).astype('int64');_,idx=np.unique(keys,axis=0,return_index=True);sp=sp[idx];sw=sw[idx]
    up,up_conf=support_up(np.concatenate(ring_normals))
    np.savez_compressed(out/'anchors.npz',points=points,normals=normals,support_points=sp,support_confidence=sw,up=up)
    # External pointmaps follow official run_da3.py: CV camera coordinates, N x 3 x H x W.
    if len({d.shape for d in depths})!=1:raise ValueError('Native MV-SAM3D views must have matching raster dimensions; resizing RGB is not permitted')
    np.savez_compressed(out/'pointmaps.npz',pointmaps_sam3d=np.stack(pointmaps),depth=np.stack(depths),intrinsics=np.stack(Ks),extrinsics=np.stack(Ts)[:,:3],image_files=np.array(names))
    manifest=dict(schema_version=1,kind='paintmesh-object-observations',status='complete',complete=True,
                  target_id=target_id,inputs=inputs,diameter=diameter,center=center,up=up,up_confidence=up_conf,
                  views=views,anchor_count=len(points),config=cfg,
                  generation_input_policy='native_rgb_official_crop',
                  original_mesh=fingerprint(original_source),background_mesh=fingerprint(background_source),
                  mesh_recovery=dict(original=original_recovered,background=background_recovered,
                                     background_patch=str((out/'background_proxy.ply').resolve()) if background_recovered else None),
                  semantic_mask_warning='Associated masks are checked geometrically; inspect review images for identity mistakes.',
                  depth_source='pre-removal PGSR mesh raycast, not sensor ground truth')
    write_json(out/'observations.json',manifest);return manifest
