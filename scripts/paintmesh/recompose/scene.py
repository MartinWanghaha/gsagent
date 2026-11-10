"""Versioned movable scene graph; canonical assets are never transformed in-place."""
from pathlib import Path
import numpy as np
from .io import read_json, write_json, fingerprint, save_mesh, ply, xyz,crop_mesh
from .geometry import transform, check_similarity, RayMesh, surface_samples
from .gaussian import transform_gaussians,merge_gaussians
from .texture import transform_glb
from .collision import meshes_intersect
from .original_gaussians import movement_transform


def publish(inputs,objects,output,review=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);nodes=[]
    if sorted(int(item['id']) for item in objects)!=sorted(inputs['target_ids']):
        raise ValueError('A scene export requires all requested object assets')
    for item in objects:
        pose=read_json(item['pose'])
        if pose['status']!='accepted' and not review:raise ValueError(f'Object {item["id"]} requires review: {pose["reasons"]}')
        asset_ref=fingerprint(item['asset']);observation_ref=fingerprint(Path(item['evidence'])/'observations.json')
        if 'input_asset' in pose and pose['input_asset']['sha256']!=asset_ref['sha256']:
            raise ValueError('Placement refers to a different generated mesh')
        if 'input_observations' in pose and pose['input_observations']['sha256']!=observation_ref['sha256']:
            raise ValueError('Placement observations have changed')
        check_similarity(pose['object_to_world'])
        nodes.append(dict(id=str(item['id']),object_to_world=pose['object_to_world'],original_pose=pose['object_to_world'],
                          mesh=asset_ref,gaussians=fingerprint(item['gaussians']),
                          gaussian_coordinates='original_world',
                          gaussian_provenance=fingerprint(item['gaussian_provenance']),
                          gaussian_indices=fingerprint(item['gaussian_indices']),pose_report=fingerprint(item['pose']),
                          evidence=str(Path(item['evidence']).resolve())))
        nodes[-1]['evidence_references']=[observation_ref]+[fingerprint(Path(item['evidence'])/name) for name in ['anchors.npz','background_proxy.ply'] if (Path(item['evidence'])/name).exists()]
        native=Path(item['asset']).parent/'generated.glb'
        if native.exists():nodes[-1]['textured_mesh']=fingerprint(native)
    scene=dict(schema_version=2,kind='paintmesh-composable-scene',status='needs_review' if review else 'complete',complete=not review,
               components=dict(gaussians=dict(status='complete',source='inpaint background + exact deleted original Gaussians',training=False),
                               mesh=dict(status='needs_review' if review else 'complete',source='inpaint background + MV-SAM3D generated meshes')),
               coordinate_convention='column vectors, COLMAP world, CV cameras; scale has no assumed metric units',
               background=dict(mesh=fingerprint(inputs['background_mesh']),gaussians=fingerprint(inputs['background_gaussians'])),
               input_selection=inputs,objects=nodes,
               rendering='Globally depth-sort all background/object splats; mesh is a separate geometry representation.',
               lighting='SH rotates with object; dynamic physical relighting/shadows are not implemented.')
    patches=[]
    for item in objects:
        evidence=read_json(Path(item['evidence'])/'observations.json')
        patch=evidence.get('mesh_recovery',{}).get('background_patch')
        if patch and patch not in [p['path'] for p in patches]:patches.append(fingerprint(patch))
    if patches:
        scene['background']['patches']=patches
        scene['background_recovery']='Local unfiltered TSDF patches fill empty component-filter ROIs; original background remains a separate static layer.'
    # Object-object intersection check is also needed when multiple targets share a scene.
    report=validate_scene_placement(scene)
    report['pose_rejections']=[dict(id=str(item['id']),reasons=read_json(item['pose'])['reasons']) for item in objects if read_json(item['pose'])['status']!='accepted']
    report['accepted']=bool(report['accepted'] and not report['pose_rejections'])
    if not report['accepted'] and not review:
        write_json(out/'placement_rejected.json',report);raise ValueError('Inter-object collision or support check failed')
    if review:
        scene['warning']='Diagnostic candidate only: placement/identity/contact checks are unresolved. Not an accepted composition.'
        scene['review_reasons']=report['pose_rejections']
    write_json(out/'placement_validation.json',report);write_json(out/'scene.json',scene)
    export_scene(out/'scene.json',out/'world');return scene


def validate_scene_placement(scene):
    from .optimize import PoseProblem,physical_constraints_ok
    checks=[];transformed=[]
    for node in scene['objects']:
        pose=read_json(node['pose_report']['path']);p=PoseProblem(node['evidence'],node['mesh']['path'],pose['config'])
        T=np.asarray(node['object_to_world']);world=transform(p.samples,T);height=world@p.up
        if not np.allclose(T,node.get('original_pose',T),atol=1e-8):
            # A moved object must use geometry at its destination, not its old support ROI.
            from scipy.spatial import cKDTree
            lower=world.min(0)-.2*p.d;upper=world.max(0)+.2*p.d
            parts=[]
            for ref in [scene['background']['mesh']]+scene['background'].get('patches',[]):
                try:parts.append(crop_mesh(ref['path'],lower,upper,200000))
                except ValueError as error:
                    if 'No triangles in requested ROI:' not in str(error):raise
            if not parts:raise ValueError('No background surface at the movement destination')
            bv=[];bf=[];offset=0
            for vertices,faces,colors in parts:bv.append(vertices);bf.append(faces+offset);offset+=len(vertices)
            bv=np.concatenate(bv);bf=np.concatenate(bf)
            p.background=RayMesh(bv,bf)
            support,_,_,_=surface_samples(bv,bf,30000,42)
            p.support=support;p.support_weight=np.ones(len(support));p.support_tree=cKDTree(support)
        indices=np.flatnonzero(height<=np.quantile(height,p.cfg['contact_quantile']))
        _,physics=p.physics(T,indices,return_detail=True)
        accepted=physical_constraints_ok(physics,p.cfg)
        checks.append(dict(id=node['id'],accepted=bool(accepted),physics=physics));transformed.append((node,p,world))
    pair_checks=[]
    for i,(ni,pi,wi) in enumerate(transformed):
        for nj,pj,wj in transformed[i+1:]:
            # Accurate signed containment when the generated object is watertight; otherwise
            # intersecting triangle meshes are tested without inventing an inside/outside sign.
            import open3d as o3d
            ai=np.load(ni['mesh']['path']);aj=np.load(nj['mesh']['path'])
            intersects=meshes_intersect(transform(ai['vertices'],ni['object_to_world']),ai['faces'],
                                        transform(aj['vertices'],nj['object_to_world']),aj['faces'])
            contained=False
            for node,p,points in [(ni,pi,wj),(nj,pj,wi)]:
                if p.watertight:
                    local=transform(points,np.linalg.inv(node['object_to_world']))
                    signed=p.mesh.scene.compute_signed_distance(o3d.core.Tensor(local.astype('float32')),nthreads=4,nsamples=3).numpy()
                    contained|=bool((signed<-.001).any())
            pair_checks.append(dict(objects=[ni['id'],nj['id']],intersects=bool(intersects or contained)))
    return dict(accepted=all(x['accepted'] for x in checks) and not any(x['intersects'] for x in pair_checks),objects=checks,pairs=pair_checks)


def export_scene(scene_path,output):
    scene=read_json(scene_path);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    verify_scene_assets(scene)
    sources=[scene['background']['gaussians']['path']]
    for node in scene['objects']:
        a=np.load(node['mesh']['path']);T=np.asarray(node['object_to_world'])
        save_mesh(out/f'{node["id"]}_mesh.ply',transform(a['vertices'],T),a['faces'],a['colors'])
        if 'textured_mesh' in node:
            native=Path(node['textured_mesh']['path'])
            shape=np.eye(4);shape[:3,:3]=a['baked_shape_transform'] if 'baked_shape_transform' in a else np.eye(3)
            transform_glb(native,out/f'{node["id"]}_textured.glb',T@shape)
        delta=movement_transform(node)
        if np.array_equal(delta,np.eye(4)):
            # Initial placement must preserve the original coefficients bit-for-bit.
            import shutil
            shutil.copyfile(node['gaussians']['path'],out/f'{node["id"]}_gaussians.ply')
        else:transform_gaussians(node['gaussians']['path'],out/f'{node["id"]}_gaussians.ply',delta)
        sources.append(out/f'{node["id"]}_gaussians.ply')
    ranges=merge_gaussians(sources,out/'composed_gaussians.ply')
    write_json(out/'gaussian_ranges.json',dict(ranges=ranges,object_order=['background']+[n['id'] for n in scene['objects']]))
    write_json(out/'layers.json',dict(background=scene['background'],
                                    objects=[dict(id=n['id'],mesh=f'{n["id"]}_mesh.ply',
                                                  textured_mesh=f'{n["id"]}_textured.glb' if 'textured_mesh' in n else None,
                                                  gaussians=f'{n["id"]}_gaussians.ply') for n in scene['objects']]))
    from .mesh_export import export_mesh_scene
    export_mesh_scene(scene,out)


def verify_scene_assets(scene):
    references=[scene['background'][k] for k in ['mesh','gaussians']]+scene['background'].get('patches',[])
    for node in scene['objects']:
        if 'gaussians' in node:movement_transform(node)
        references.extend(node[k] for k in ['mesh','gaussians','gaussian_provenance','gaussian_indices','pose_report','textured_mesh'] if k in node)
        references.extend(node.get('evidence_references',[]))
    for ref in references:
        if fingerprint(ref['path'])['sha256']!=ref['sha256']:
            raise ValueError(f'Scene asset changed: {ref["path"]}; regenerate composition')


def move_object(scene_path,object_id,T,output):
    check_similarity(T);scene=read_json(scene_path);found=False
    if scene.get('status')=='needs_review':raise ValueError('Resolve candidate placement before exporting a validated movement')
    for node in scene['objects']:
        if node['id']==str(object_id):
            old_scale=check_similarity(node['object_to_world'])[0];new_scale=check_similarity(T)[0]
            if not np.isclose(old_scale,new_scale,rtol=1e-5):raise ValueError('Movement preserves calibrated scale')
            node['object_to_world']=np.asarray(T).tolist();found=True
    if not found:raise ValueError('Object ID not in scene graph')
    if 'background' in scene:verify_scene_assets(scene)
    report=validate_scene_placement(scene)
    if not report['accepted']:raise ValueError(f'Movement rejected: {report}')
    out=Path(output)
    if out.resolve()==Path(scene_path).resolve():raise ValueError('Write a new scene revision for movement')
    scene['parent_scene']=fingerprint(scene_path);write_json(out.with_suffix('.validation.json'),report)
    write_json(out,scene);export_scene(out,out.parent/(out.stem+'_world'))
