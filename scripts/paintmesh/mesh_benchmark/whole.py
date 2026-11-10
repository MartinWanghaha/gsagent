"""Whole-scene mesh extraction from unchanged PaintMesh Gaussian checkpoints."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
from .core import BASE,REPO,read,write,identity,digest,cameras

ORIGINAL=REPO/'output/paintmesh/Inpaint360GS/recomposition'

def extraction_parameters(scene,resolution=384):
    """Depth range from reliable *training-track* SfM observations, not test RGB."""
    document=read(ORIGINAL/scene/'composition/scene.json');selection=document['input_selection']
    cs=[c for c in cameras(scene) if c['part']=='train']
    centers=np.array([np.linalg.inv(c['w2c'])[:3,3] for c in cs]);center=centers.mean(0)
    radius=float(np.linalg.norm(centers-center,axis=1).max()*1.1)
    spec=importlib.util.spec_from_file_location('whole_colmap',REPO/'submodules/Inpaint360GS/tools/read_write_model.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    _,images,points=mod.read_model(str(Path(selection['source'])/'sparse/0'),ext='.bin')
    names={c['name'] for c in cs};train={i:x for i,x in images.items() if x.name in names}
    depth_rows={i:(x.qvec2rotmat()[2],x.tvec[2]) for i,x in train.items()}
    depths=[];reliable_points=0
    for p in points.values():
        ids=[int(i) for i in p.image_ids if int(i) in train]
        if len(ids)<3 or p.error>2 or not np.isfinite(p.xyz).all():continue
        reliable_points+=1
        depths.extend(float(depth_rows[i][0]@p.xyz+depth_rows[i][1]) for i in ids)
    depths=np.asarray(depths);depths=depths[np.isfinite(depths)&(depths>0)]
    if not len(depths):raise ValueError(f'No reliable training depths: {scene}')
    quantiles=np.quantile(depths,[.5,.9,.99,.995])
    max_depth=max(2*radius,1.2*float(quantiles[-1]))
    voxel=max(.002,radius/resolution)
    return dict(scene=scene,source=selection['source'],gaussian_source=document['background']['gaussians']['path'],
                model_source=str(Path(document['background']['gaussians']['path']).parents[2]),
                iteration=int(Path(document['background']['gaussians']['path']).parent.name.split('_')[-1]),
                source_scene=str(ORIGINAL/scene/'composition/scene.json'),train_views=len(cs),train_names=sorted(names),
                camera_center=center.tolist(),camera_radius=radius,depth_quantiles=quantiles.tolist(),
                reliable_sfm_points=reliable_points,max_depth=max_depth,voxel_size=voxel,
                sdf_trunc=6*voxel,alpha_min=.5,min_component_faces=50,grid_radius_resolution=resolution,
                note='Scene-adaptive finite TSDF over observed background; not infinite/unbounded contraction. No Gaussian retraining.')

def available_ram():
    return next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))

def extract(output,scene,parameters):
    import os,time,resource,gc
    output=Path(output);model=output/'models'/scene;model.mkdir(parents=True,exist_ok=True)
    source=Path(parameters['model_source']);frames=model/'geometry_frames';frames.mkdir(exist_ok=True)
    request=dict(parameters=parameters,checkpoint=identity(parameters['gaussian_source']),config=identity(source/'config.yaml'))
    request_id=digest(request);marker=model/'whole_mesh_complete.json'
    if marker.exists():
        previous=read(marker)
        if previous['request_id']!=request_id:raise ValueError('Whole extraction inputs changed; use a new output')
        return previous
    frame_marker=frames/'complete.json'
    render_keys=['scene','source','gaussian_source','model_source','iteration','train_names','max_depth','alpha_min']
    def render_request(value):
        return dict(checkpoint=value['checkpoint'],config=value['config'],parameters={k:value['parameters'][k] for k in render_keys})
    render_id=digest(render_request(request))
    if frame_marker.exists() and read(frame_marker).get('render_request_id')!=render_id:
        previous=model/'extraction_request.json'
        if not previous.exists() or render_request(read(previous)['request'])!=render_request(request):raise ValueError('Cached depths have different rendering inputs')
        cached=read(frame_marker);cached['render_request_id']=render_id;write(frame_marker,cached)
    write(model/'extraction_request.json',dict(request_id=request_id,request=request,render_request_id=render_id))
    if not frame_marker.exists():
        import sys,torch
        torch.set_num_threads(4)
        free,total=torch.cuda.mem_get_info();cap=6*1024**3
        if free<cap+4*1024**3:raise RuntimeError('Not enough free VRAM for whole-scene rendering')
        torch.cuda.set_per_process_memory_fraction(cap/total)
        edgs=REPO/'submodules/EDGS';sys.path.insert(0,str(edgs))
        spec=importlib.util.spec_from_file_location('whole_edgs_render',edgs/'render.py')
        render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
        cfg=render.load_training_config(source,source_path=Path(parameters['source']),images='images',resolution=4)
        cfg.gs.dataset.data_device='cpu';cfg.gs.dataset.model_path=str(source)
        render.bootstrap_gaussian_splatting()
        from scene import GaussianModel,Scene
        renderer=render.build_renderer({'backend':'pgsr','clamp_rgb':True})
        with torch.no_grad():
            gaussian=GaussianModel(int(cfg.gs.sh_degree));scene_model=Scene(cfg.gs.dataset,gaussian,load_iteration=parameters['iteration'],shuffle=False)
            wanted={Path(n).stem for n in parameters['train_names']}
            views=sorted([c for c in scene_model.getTrainCameras() if Path(c.image_name).stem in wanted],key=lambda c:c.image_name)
            if {Path(v.image_name).stem for v in views}!=wanted:raise ValueError('Fusion cameras differ from declared training split')
            background=torch.zeros(3,device='cuda');records=[]
            for i,c in enumerate(views):
                package=renderer(c,gaussian,cfg.gs.pipe,background,return_plane=True)
                raw=package['plane_depth'].squeeze();alpha=package['rendered_alpha'].squeeze()
                valid=torch.isfinite(raw)&(raw>0)&(raw<=parameters['max_depth'])&(alpha>=parameters['alpha_min'])
                depth=torch.where(valid,raw,0).float().cpu().numpy()
                rgb=np.rint(package['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')
                file=frames/(c.image_name+'.npz')
                K=render.camera_intrinsics(c,device='cpu').numpy();T=c.world_view_transform.T.detach().cpu().numpy()
                np.savez_compressed(file,depth=depth,rgb=rgb,K=K,w2c=T)
                records.append(dict(name=c.image_name,path=str(file),valid_fraction=float(valid.float().mean()),
                       fraction_over_old_depth_limit=float(((raw>5)&torch.isfinite(raw)&(alpha>=.5)).float().mean())))
                if i%10==0:print('WHOLE_RENDER',scene,i+1,'/',len(views),'valid',records[-1]['valid_fraction'],flush=True)
                del package,raw,alpha,valid
            write(frame_marker,dict(complete=True,request_id=request_id,render_request_id=render_id,frames=records,cuda_peak_mib=torch.cuda.max_memory_allocated()/1024**2))
        del views,c,gaussian,scene_model,renderer;gc.collect();torch.cuda.empty_cache()
    recorded=read(frame_marker)
    if recorded.get('render_request_id')!=render_id:raise ValueError('Stale depth cache')
    import open3d as o3d
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=parameters['voxel_size'],
              sdf_trunc=parameters['sdf_trunc'],color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    started=time.time()
    for i,record in enumerate(recorded['frames']):
        with np.load(record['path']) as a:
            d=a['depth'];rgb=a['rgb'];K=a['K'];T=a['w2c']
        h,w=d.shape
        image=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),o3d.geometry.Image(d),
                   depth_scale=1.,depth_trunc=parameters['max_depth'],convert_rgb_to_intensity=False)
        intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2]))
        volume.integrate(image,intrinsic,T)
        if i%10==0:print('WHOLE_TSDF',scene,i+1,'/',len(recorded['frames']),'available RAM GiB',round(available_ram(),1),flush=True)
        if available_ram()<10:raise MemoryError('Whole TSDF stopped before system RAM exhaustion')
    if available_ram()<20:raise MemoryError('Not enough RAM headroom to extract whole mesh')
    mesh=volume.extract_triangle_mesh();del volume;gc.collect()
    if not len(mesh.triangles):raise ValueError('Whole TSDF returned no triangles')
    mesh_dir=output/'meshes'/scene;mesh_dir.mkdir(parents=True,exist_ok=True)
    raw_counts=dict(vertices=len(mesh.vertices),faces=len(mesh.triangles))
    labels,counts,areas=mesh.cluster_connected_triangles();labels=np.asarray(labels);counts=np.asarray(counts)
    remove=counts[labels]<parameters['min_component_faces']
    mesh.remove_triangles_by_mask(remove);mesh.remove_unreferenced_vertices();mesh.remove_degenerate_triangles();mesh.compute_vertex_normals()
    path=mesh_dir/'background.ply'
    if not o3d.io.write_triangle_mesh(str(path),mesh):raise RuntimeError('Could not write whole background mesh')
    result=dict(complete=True,request_id=request_id,request=request,path=str(path),parameters=parameters,
             raw_counts=raw_counts,vertices=len(mesh.vertices),faces=len(mesh.triangles),
             components_total=len(counts),components_retained=int((counts>=parameters['min_component_faces']).sum()),
             bounds=[mesh.get_min_bound().tolist(),mesh.get_max_bound().tolist()],elapsed_fusion=time.time()-started,
             process_peak_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,render_metadata=recorded)
    write(marker,result);print('WHOLE_MESH_COMPLETE',scene,result['vertices'],result['faces'],flush=True)
    return result

def validate_whole(output,scene):
    """Measure coverage, without tuning against RGB/geometry benchmark scores."""
    from .core import MeshScene,rays,image_mask
    from PIL import Image,ImageDraw
    output=Path(output);destination=output/'meshes'/scene
    extraction=read(output/'models'/scene/'whole_mesh_complete.json')
    original=read(ORIGINAL/scene/'composition/scene.json');selection=original['input_selection']
    mesh=MeshScene([dict(path=extraction['path'])]);rows=[];preview=None
    old_root=BASE/'mesh_comparison';choices=old_root/'figures_all_methods/selected_views.json'
    chosen=read(choices)['scenes'][scene]['camera']['name'] if choices.exists() else None
    for camera in [c for c in cameras(scene) if c['part']=='held_out']:
        rgb,foreground=image_mask(camera,selection);allowed=~foreground
        z,color=mesh.cast(rays(camera),color=camera['name']==chosen)
        depth=z.reshape(camera['height'],camera['width'])
        old_path=old_root/'cache/paintmesh'/scene/'background/depth'/(camera['stem']+'.npy')
        old=np.load(old_path) if old_path.exists() else None
        rows.append(dict(name=camera['name'],whole=float(((depth>0)&allowed).sum()/max(allowed.sum(),1)),
                    original=float(((old>0)&allowed).sum()/max(allowed.sum(),1)) if old is not None else None))
        if camera['name']==chosen:
            whole=np.rint(color.reshape(rgb.shape).clip(0,1)*255).astype('uint8')
            old_rgb=old_root/'cache/paintmesh'/scene/'background/native_renders'/(camera['stem']+'.png')
            images=[Image.fromarray(rgb),Image.open(old_rgb).convert('RGB') if old_rgb.exists() else Image.new('RGB',(camera['width'],camera['height'])),Image.fromarray(whole)]
            titles=['GT original-state RGB','Previous background mesh','Whole background mesh']
            width,height=camera['width'],camera['height'];canvas=Image.new('RGB',(width*3,height+38),'white');draw=ImageDraw.Draw(canvas)
            for i,(image,title) in enumerate(zip(images,titles)):
                canvas.paste(image,(i*width,38));draw.text((i*width+8,10),title,fill='black')
            canvas.save(destination/'coverage_preview.jpg',quality=94)
            preview=dict(camera=camera['name'],path=str(destination/'coverage_preview.jpg'))
    whole=float(np.mean([x['whole'] for x in rows]));old_values=[x['original'] for x in rows if x['original'] is not None]
    old=float(np.mean(old_values)) if old_values else None
    result=dict(whole_background_coverage=whole,previous_background_coverage=old,
                change=whole-old if old is not None else None,per_view=rows,preview=preview,
                passed=bool(whole>=.90 or (old is not None and whole>=old+.10)),
                definition='Positive finite triangle depth outside original target mask; no RGB error tuning')
    write(destination/'coverage_validation.json',result);return result

def publish_whole(output,scene):
    import copy
    from recompose.io import fingerprint
    from recompose.mesh_export import export_mesh_scene
    from recompose.texture import transform_glb
    output=Path(output);original=ORIGINAL/scene/'composition/scene.json';document=copy.deepcopy(read(original))
    result=read(output/'models'/scene/'whole_mesh_complete.json')
    document['background']['mesh']=fingerprint(result['path']);document['background'].pop('patches',None)
    document['input_selection']['background_mesh']=result['path']
    document['whole_scene_extraction']=dict(source_scene=str(original),source_checkpoint=result['request']['checkpoint'],
                parameters=result['parameters'],geometry=str(output/'meshes'/scene/'background.ply'),
                gaussian_retrained=False,object_geometry_and_poses_unchanged=True)
    destination=output/'recomposition'/scene/'composition';destination.mkdir(parents=True,exist_ok=True)
    write(destination/'scene.json',document)
    export_mesh_scene(document,destination/'world')
    for node in document['objects']:
        if 'textured_mesh' not in node:continue
        with np.load(node['mesh']['path']) as asset:
            shape=np.eye(4);shape[:3,:3]=asset['baked_shape_transform'] if 'baked_shape_transform' in asset else np.eye(3)
        transform_glb(node['textured_mesh']['path'],destination/'world'/(node['id']+'_textured.glb'),np.asarray(node['object_to_world'])@shape)
    pure=copy.deepcopy(document);pure['objects']=[]
    export_mesh_scene(pure,output/'meshes'/scene/'world')
    write(output/'meshes'/scene/'scene_export.json',dict(background_scene=str(output/'meshes'/scene/'world/mesh_scene.gltf'),
              recomposed_scene=str(destination/'world/mesh_scene.gltf'),source_scene=str(destination/'scene.json')))
    return destination/'scene.json'

def reuse_baseline_caches(output):
    """Read-only reuse after independently checking cameras, masks and geometry."""
    import copy,shutil
    from .cache import common
    from .core import SCENES,METHODS,specs_for
    output=Path(output);previous=BASE/'mesh_comparison';reused=[]
    for scene in SCENES:
        if not (output/'recomposition'/scene/'composition/scene.json').exists():continue
        common(output,scene)
        # Canonicalize the in-memory numpy camera matrices via the manifest
        # writer before comparing them with the serialized source cache.
        new=read(output/'common'/scene/'complete.json');old_file=previous/'common'/scene/'complete.json'
        if not old_file.exists():continue
        old=read(old_file)
        if digest(old['cameras'])!=digest(new['cameras']) or old['request']['masks']!=new['request']['masks']:
            print('BASELINE_REUSE_SKIPPED_INPUT_CHANGE',scene,flush=True);continue
        for method in [m for m in METHODS if m!='paintmesh']:
            specs=specs_for(scene,method,'background')
            if specs is None:continue
            expected=[dict(identity=identity(s['path']),transform=s.get('transform')) for s in specs]
            for state in ['background','recomposition']:
                source=previous/'cache'/method/scene/state;marker=source/'complete.json'
                if not marker.exists():continue
                old_meta=read(marker)
                if old_meta['request']['specs']!=expected:continue
                destination=output/'cache'/method/scene/state;destination.mkdir(parents=True,exist_ok=True)
                for name in ['depth','native_renders','train_pool.npz']:
                    p=source/name;q=destination/name
                    if p.exists() and not q.exists():q.symlink_to(p.resolve(),target_is_directory=p.is_dir())
                meta=copy.deepcopy(old_meta);meta['request']['common']=new['request_id'];meta['reused_immutable_cache']=str(source)
                write(destination/'complete.json',meta)
                old_render=previous/state/method/scene;new_render=output/state/method/scene
                new_render.mkdir(parents=True,exist_ok=True)
                for name in ['color_field.pt','renders']:
                    p=old_render/name;q=new_render/name
                    if p.is_dir():shutil.copytree(p,q,dirs_exist_ok=True)
                    elif p.exists() and not q.exists():shutil.copy2(p,q)
                reused.append(dict(method=method,scene=scene,state=state,source=str(source)))
    write(output/'baseline_reuse.json',dict(reused=reused,validation='Exact camera definitions, target mask identities, and mesh identities; depth/training pools immutable; RGB/field outputs independently copied'))
