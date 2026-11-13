#!/usr/bin/env python3
"""Publish a mesh-registration correction and incrementally refresh its report.

This is a generated-mesh registration correction, not a user movement of the
original object. The initial mesh/GS registration reference moves with the mesh
registration, so the exact deleted original Gaussian rows stay in original
world coordinates. Background triangles and their cached RGB/depth stay fixed.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import gc
import json
import subprocess
import sys
import time
import types
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mesh_benchmark.core import BASE, SCENES, METHODS, STATES, read, write, identity, digest
from mesh_benchmark.layout import work_root, configure_sources
from recompose.io import fingerprint, save_mesh
from recompose.geometry import check_similarity, transform
from recompose.original_gaussians import movement_transform

DEFAULT = BASE/'mesh_comparison_whole'


def compose_native_arrays(background_depth, background_rgb, object_depth, object_rgb):
    """Nearest positive camera-z; never average foreground/background surfaces."""
    bd, od = np.asarray(background_depth), np.asarray(object_depth)
    br, rr = np.asarray(background_rgb), np.asarray(object_rgb)
    if bd.shape != od.shape or br.shape != (*bd.shape, 3) or rr.shape != br.shape:
        raise ValueError('Composition image/depth dimensions differ')
    hit = np.isfinite(od) & (od > 0) & (~np.isfinite(bd) | (bd <= 0) | (od < bd))
    depth, rgb = bd.copy(), br.copy()
    depth[hit], rgb[hit] = od[hit], rr[hit]
    return depth, rgb


def registration_node(node, update, report, accepted):
    """Return a corrected mesh reference while preserving the GS identity delta."""
    result = copy.deepcopy(node)
    if not np.allclose(movement_transform(node), np.eye(4), rtol=0, atol=1e-12):
        raise ValueError('This publication only corrects initial registration, not an already moved GS object')
    target = np.asarray(update['object_to_world'], dtype='float64')
    before = np.asarray(node['object_to_world'], dtype='float64')
    if not np.isclose(check_similarity(target)[0], check_similarity(before)[0], rtol=1e-6, atol=1e-10):
        raise ValueError('Collision correction must preserve generated-object scale')
    reference = result.get('registration_refinement', {})
    original = reference.get('previous_mesh_registration', before.tolist())
    result['object_to_world'] = target.tolist()
    result['original_pose'] = target.tolist()
    geometry_changed = bool(update.get('mesh_path') and
                            Path(update['mesh_path']).resolve()!=Path(node['mesh']['path']).resolve())
    previous_mesh = reference.get('previous_generated_mesh',node.get('mesh'))
    if geometry_changed:
        result['mesh']=fingerprint(update['mesh_path'])
        if update.get('corrected_textured_mesh'):
            result['textured_mesh']=update['corrected_textured_mesh']
        else:
            # Preflight calls validate transforms before GLB generation. They
            # must never carry an old, undeformed native texture mesh forward.
            result.pop('textured_mesh',None)
    result['registration_refinement'] = dict(kind='mesh_registration_correction',
        previous_mesh_registration=original, corrected_mesh_registration=target.tolist(),
        optimization_report=report, accepted=bool(accepted),
        gaussian_policy='Exact deleted original Gaussians remain in original world coordinates; movement delta is identity',
        generated_mesh_geometry_changed=geometry_changed, generated_mesh_scale_changed=False,
        previous_generated_mesh=previous_mesh,
        corrected_mesh_texture_policy=('Corrected GLB reconstructed from original NPZ UV/material/texture data, with no texture rebaking; primary scene glTF retains original vertex colors'
                                       if geometry_changed else 'Original texture assets unchanged'))
    if not np.allclose(movement_transform(result), np.eye(4), rtol=0, atol=1e-12):
        raise AssertionError('Registration correction unexpectedly moved original Gaussians')
    return result


def validation_object(validation, object_id, report):
    values = validation.get('objects', {})
    if isinstance(values, list):
        values = {str(v['id']): v for v in values}
    item = values.get(str(object_id), {}) if isinstance(values, dict) else {}
    # A completed computation is not the same thing as an accepted placement.
    accepted = item.get('accepted') is True or item.get('status') == 'accepted'
    if not item:
        accepted = report.get('accepted') is True or report.get('status') == 'accepted'
    if validation.get('complete') is not True or item.get('unresolved') or report.get('unresolved'):
        accepted = False
    return item, bool(accepted)


def _gaussian_stamps(graph, composition):
    paths = [graph['background']['gaussians']['path']]
    paths += [n['gaussians']['path'] for n in graph['objects']]
    paths += [p for p in (Path(composition)/'world').glob('*gaussians.ply')]
    return [identity(p) for p in paths]


def publish_registration(scene, object_updates, validation, output=DEFAULT):
    """Update both published scene graphs, their pose reports and mesh exports.

    No geometry source mesh, texture, Gaussian file or background mesh changes.
    ``object_updates`` maps an id to {object_to_world, report_path}. ``validation``
    must explicitly mark accepted objects; unresolved objects remain candidates.
    """
    public = Path(output); root = work_root(public); configure_sources(root)
    graph_path = public/'recomposition'/scene/'composition/scene.json'
    graph = read(graph_path)
    object_updates = {str(k): dict(v,object_to_world=np.asarray(v['object_to_world'],dtype='float64').tolist())
                      for k,v in object_updates.items()}
    if not object_updates or not set(object_updates).issubset({str(n['id']) for n in graph['objects']}):
        raise ValueError('No valid object updates for this scene')
    # Corrected assets may live in a dedicated refinement directory; their
    # location cannot be used to infer the original scene root on resume.
    canonical = Path(graph['objects'][0]['pose_report']['path']).parents[3]/'composition/scene.json'
    graph_paths = list(dict.fromkeys([graph_path, canonical]))
    before_gs = []
    for path in graph_paths:
        before_gs += _gaussian_stamps(read(path), path.parent)
    reports = {oid: fingerprint(value['report_path']) for oid, value in object_updates.items()}
    corrected_meshes = {oid:fingerprint(v['mesh_path']) for oid,v in object_updates.items() if v.get('mesh_path')}
    # Validate every graph and transform before writing even the first pose.
    # In particular a malformed Sim(3) must not leave a half-published pose file.
    for path in graph_paths:
        for node in read(path)['objects']:
            oid=str(node['id'])
            if oid in object_updates:
                registration_node(node,object_updates[oid],reports[oid],False)
                replacement=object_updates[oid].get('mesh_path')
                if replacement and Path(replacement).resolve()!=Path(node['mesh']['path']).resolve():
                    with np.load(node['mesh']['path']) as before,np.load(replacement) as after:
                        if before['vertices'].shape!=after['vertices'].shape:
                            raise ValueError('Contact correction changed vertex topology')
                        for key in before.files:
                            if key in ['faces','colors','triangle_uv','triangle_material'] or key.startswith('texture_'):
                                if key not in after or not np.array_equal(before[key],after[key]):
                                    raise ValueError('Contact correction must preserve topology/UV/texture bytes: '+key)
    receipt_folder = root/'collision_refinement_publication'/scene
    receipt_folder.mkdir(parents=True, exist_ok=True)
    request = dict(scene=scene, updates={k:dict(object_to_world=v['object_to_world'], report=reports[k],
        corrected_mesh=corrected_meshes.get(k))
        for k,v in object_updates.items()}, validation=validation, policy='mesh-registration-only, original-GS-world-frozen')
    receipt = receipt_folder/'registration.json'
    if receipt.exists():
        old = read(receipt)
        if old.get('complete') and old['request'] == request:
            for path in graph_paths:
                current = read(path)
                for n in current['objects']:
                    if str(n['id']) in object_updates:
                        if n['object_to_world'] != object_updates[str(n['id'])]['object_to_world']:
                            raise ValueError('Published mesh registration changed after completion')
            return old
    for oid,value in object_updates.items():
        if not value.get('mesh_path'):
            continue
        from mesh_benchmark.asset_gltf import export_textured_asset
        glb=Path(value['mesh_path']).with_suffix('.glb')
        export_textured_asset(value['mesh_path'],glb)
        value['corrected_textured_mesh']=fingerprint(glb)
    statuses = {}
    # Pose files are shared between the canonical and the organized graph.
    for node in graph['objects']:
        oid = str(node['id'])
        if oid not in object_updates:
            continue
        optimization = read(object_updates[oid]['report_path'])
        item, accepted = validation_object(validation, oid, optimization)
        statuses[oid] = accepted
        pose_path = Path(node['pose_report']['path']); pose = read(pose_path)
        pose['object_to_world'] = object_updates[oid]['object_to_world']
        if oid in corrected_meshes:
            prior=pose.setdefault('pre_registration_refinement_diagnostics',{})
            prior.setdefault('input_asset',pose.get('input_asset'))
            pose['input_asset']=corrected_meshes[oid]
        pose['status'] = 'accepted' if accepted else 'needs_review'
        pose['reasons'] = [] if accepted else item.get('reasons', ['Full-resolution registration/contact acceptance unresolved'])
        if 'physics' in pose and 'pre_registration_refinement_physics' not in pose:
            pose['pre_registration_refinement_physics'] = pose['physics']
        prior_diagnostics=pose.setdefault('pre_registration_refinement_diagnostics',{})
        for key in ['train_score','validation_score','train','validation','alternatives','support_settle']:
            if key in pose:
                if key not in prior_diagnostics:
                    prior_diagnostics[key]=pose[key]
                pose.pop(key)
        pose['physical_correction_accepted']=accepted
        pose['physics'] = dict(kind='full_mesh_registration_refinement', validation=item,
            optimization_report=reports[oid], old_proxy_physics_is_not_current_acceptance=True)
        pose['registration_refinement'] = dict(optimization_report=reports[oid], validation=item,
            generated_mesh_only=True, gaussian_world_unchanged=True)
        write(pose_path, pose)
    from mesh_benchmark.incremental_mesh_registration import export_registered_mesh_incremental
    from recompose.texture import transform_glb
    updated = []
    for path in graph_paths:
        document = read(path)
        for i,node in enumerate(document['objects']):
            oid = str(node['id'])
            if oid not in object_updates:
                continue
            node = registration_node(node, object_updates[oid], reports[oid], statuses[oid])
            node['pose_report'] = fingerprint(node['pose_report']['path'])
            document['objects'][i] = node
        all_accepted = all(n.get('registration_refinement', {}).get('accepted',
            read(n['pose_report']['path']).get('status') == 'accepted') for n in document['objects'])
        all_accepted = all_accepted and validation.get('complete') is True
        document['complete'] = bool(all_accepted)
        document['status'] = 'complete' if all_accepted else 'needs_review'
        document.setdefault('components', {}).setdefault('mesh', {})['status'] = document['status']
        document['collision_registration_refinement'] = dict(complete=True, accepted=bool(all_accepted),
            validation=validation, publication_policy=request['policy'], updated=time.time())
        if all_accepted:
            document.pop('warning', None)
        else:
            document['warning'] = 'Candidate registration: unresolved full-mesh contact or observation acceptance remains; see collision_registration_refinement.'
        world = path.parent/'world'
        # Preserve the complete existing background glTF bytes/references.
        # Only generated-object buffers and registration nodes need rewriting.
        export_registered_mesh_incremental(document, world)
        for node in document['objects']:
            oid = str(node['id'])
            if oid not in object_updates:
                continue
            with np.load(node['mesh']['path']) as asset:
                T = np.asarray(node['object_to_world'])
                if (world/f'{oid}_mesh.ply').exists() or path == canonical:
                    save_mesh(world/f'{oid}_mesh.ply', transform(asset['vertices'],T), asset['faces'], asset['colors'])
                if 'textured_mesh' in node:
                    if node['registration_refinement'].get('generated_mesh_geometry_changed'):
                        from mesh_benchmark.asset_gltf import export_textured_asset
                        export_textured_asset(node['mesh']['path'],world/f'{oid}_textured.glb',transform=T)
                    else:
                        shape = np.eye(4)
                        if 'baked_shape_transform' in asset:
                            shape[:3,:3] = asset['baked_shape_transform']
                        transform_glb(node['textured_mesh']['path'], world/f'{oid}_textured.glb', T@shape)
        write(path, document)
        write(path.parent/'placement_validation.json', dict(complete=True, accepted=bool(all_accepted),
            method='Full current environment mesh registration correction', validation=validation,
            generated_mesh_registration_only=True, original_gaussians_unchanged=True))
        updated.append(identity(path))
    if any(identity(s['path']) != s for s in before_gs):
        raise AssertionError('Original or composed Gaussian bytes/stat unexpectedly changed')
    result = dict(complete=True, request=request, graphs=updated, original_gaussians_unchanged=True,
        gaussian_sources=before_gs, background_mesh_unchanged=True, updated=time.time())
    write(receipt, result)
    return result


def retag_background_metadata(root, scene, graph_path):
    """Refresh pose-dependent provenance while preserving background numbers."""
    root = Path(root)
    common = root/'common'/scene
    info = read(common/'complete.json')
    old_id = info['request_id']
    info['request']['selection'] = identity(graph_path)
    info['request_id'] = digest(info['request'])
    new_id = info['request_id']
    if new_id == old_id:
        return info
    write(common/'complete.json', info)
    for method in METHODS:
        for state in STATES:
            path = root/'cache'/method/scene/state/'complete.json'
            meta = read(path)
            if meta['request']['common'] not in (old_id,new_id):
                raise ValueError('Unrelated common inputs changed: '+str(path))
            meta['request']['common'] = new_id
            write(path,meta)
    # Geometry/RGB support pixels are unchanged for the background. Change only
    # the source request chains; never run a new renderer or score this state.
    geo_marker = common/'background_support/complete.json'
    geo = read(geo_marker)
    for row in geo['request']['inputs']:
        row['request'] = read(root/'cache'/row['method']/scene/'background/complete.json')['request']
    geo['request_id'] = digest(geo['request'])
    write(geo_marker,geo)
    for method in METHODS:
        dest = root/'cache'/method/scene/'background'
        g = read(dest/'geometry_metrics.json');g['support_id'] = geo['request_id']
        write(dest/'geometry_metrics.json',g)
        meta = read(dest/'complete.json');meta['geometry'] = g;write(dest/'complete.json',meta)
    from evaluate_native_mesh_rgb import rendering
    native_marker = common/'background_native_support/complete.json'
    native = read(native_marker)
    native['request']['sources'] = {m: rendering(root,m,scene,'background')['request'] for m in METHODS}
    native['request_id'] = digest(native['request']);write(native_marker,native)
    for method in METHODS:
        path = root/'background'/method/scene/'metrics.json'
        metric = read(path)
        metric['request'] = dict(renderer=rendering(root,method,scene,'background')['request'],
            support_id=native['request_id'],geometry_support_id=geo['request_id'])
        metric['geometry'] = read(root/'cache'/method/scene/'background/geometry_metrics.json')
        metric['background_retag_reason'] = 'Only generated-mesh registration changed; original RGB/depth, masks and every numeric background metric preserved'
        write(path,metric)
    return info


def refresh_composed_render(scene, output=DEFAULT):
    public = Path(output); root = work_root(public); configure_sources(root)
    graph_path = public/'recomposition'/scene/'composition/scene.json'
    graph = read(graph_path)
    info = retag_background_metadata(root,scene,graph_path)
    from mesh_benchmark.core import MeshScene, rays, specs_for
    dest = root/'cache/paintmesh'/scene/'recomposition'
    background = root/'cache/paintmesh'/scene/'background'
    old = read(dest/'complete.json')
    specs = specs_for(scene,'paintmesh','recomposition')
    request = copy.deepcopy(old['request'])
    request['specs'] = [dict(identity=identity(s['path']),transform=s.get('transform')) for s in specs]
    request['common'] = info['request_id']
    refresh_marker = root/'collision_refinement_publication'/scene/'render.json'
    signature = dict(request=request,background_request=read(background/'complete.json')['request'],
        implementation=identity(Path(__file__)))
    if refresh_marker.exists():
        prior=read(refresh_marker)
        if prior.get('complete') and prior['signature']==signature:
            if old['request'] != request:
                raise ValueError('Composed cache differs from its registration render receipt')
            return prior
    object_specs = [dict(path=o['mesh']['path'],transform=o['object_to_world']) for o in graph['objects']]
    model = MeshScene(object_specs)
    coverage=[];outputs=[]
    for c in info['cameras']:
        if c['part'] != 'held_out':
            continue
        name = c['stem'];shape=(c['height'],c['width'])
        bd=np.load(background/'depth'/(name+'.npy'))
        br=np.asarray(Image.open(background/'native_renders'/(name+'.png')).convert('RGB'))
        oz,orgb=model.cast(rays(c),color=True)
        oz=oz.reshape(shape)
        orgb=np.rint(np.clip(orgb,0,1).reshape(*shape,3)*255).astype('uint8')
        depth,rgb=compose_native_arrays(bd,br,oz,orgb)
        (dest/'depth').mkdir(exist_ok=True);(dest/'native_renders').mkdir(exist_ok=True)
        np.save(dest/'depth'/(name+'.npy'),depth)
        Image.fromarray(rgb).save(dest/'native_renders'/(name+'.png'))
        coverage.append(dict(name=c['name'],coverage=float((np.isfinite(depth)&(depth>0)).mean()),pixels=int(depth.size)))
        outputs += [identity(dest/'depth'/(name+'.npy')),identity(dest/'native_renders'/(name+'.png'))]
        print('COLLISION_RECOMPOSE_RENDER',scene,name,flush=True)
    bgmeta=read(background/'complete.json')
    old.update(request=request,coverage=coverage,mesh_counts=bgmeta['mesh_counts']+model.counts,
        scene_status=graph['status'],geometry_pending_rescore=True,
        incremental_registration_render=dict(background_reused_exactly=True,
            renderer='Same actual-triangle Embree and native RGB sampling',
            all_object_triangles_rendered=True,gaussian_rgb_substitution=False,updated=time.time()))
    # Native mesh evaluation no longer uses training color-field pools. Keeping
    # the old pose's pool would silently misrepresent its geometry.
    for name in ['train_pool.npz','visible_surface_points.npy']:
        (dest/name).unlink(missing_ok=True)
    old['legacy_train_pool_removed'] = True
    write(dest/'complete.json',old)
    result=dict(complete=True,signature=signature,outputs=outputs,background_reused_exactly=True,
        held_out_views=len(coverage),original_gaussians_unchanged=True,updated=time.time())
    write(refresh_marker,result)
    del model;gc.collect()
    return result


def _single_state(function, state):
    """Bind the existing evaluator to one state without mutating its globals.

    The upstream functions iterate a module-level STATES list. A private global
    namespace preserves their exact implementation while excluding background.
    """
    namespace=dict(function.__globals__);namespace['STATES']=(state,)
    return types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)


def refresh_evaluation(scene, output=DEFAULT, figures=True, verify=True):
    public = Path(output); root = work_root(public); configure_sources(root)
    cfg=read(root/'protocol.json')['config']
    from mesh_benchmark.overlap import prepare_support
    from evaluate_native_mesh_rgb import supports,score,reuse_unchanged_score
    # Changed composed coverage changes the fair domain for all six methods.
    # The background support and its numbers were proven unchanged above.
    marker=root/'common'/scene/'recomposition_support/complete.json'
    marker.unlink(missing_ok=True)
    _single_state(prepare_support,'recomposition')(root,scene,METHODS,cfg)
    _single_state(supports,'recomposition')(root,[scene])
    rescored,reused=[],[]
    for method in METHODS:
        if reuse_unchanged_score(root,method,scene,'recomposition'):
            reused.append(method)
        else:
            score(root,method,scene,'recomposition')
            rescored.append(method)
    dest=root/'cache/paintmesh'/scene/'recomposition/complete.json'
    meta=read(dest);meta.pop('geometry_pending_rescore',None);write(dest,meta)
    from mesh_benchmark.report import update
    update(root)
    if figures:
        from plot_mesh_method_comparison import build
        build(root,public/'figures_all_methods',SCENES)
        subprocess.run([sys.executable,str(HERE/'organize_mesh_subplots.py'),
            '--figures',str(public/'figures_all_methods')],check=True)
        from publish_latest_mesh_geometry import figures as latest_figures
        latest_figures(public,[scene])
    if verify:
        from verify_native_mesh_delivery import verify as verify_delivery
        verify_delivery(public)
    if figures and verify:
        # Existing completion receipts contain exact file identities, not just
        # booleans. Refresh them only after the independent publication verifier
        # has checked the new PNGs, scores and source links.
        delivery_path=public/'figures_all_methods/scenes'/scene/'delivery.json'
        if delivery_path.exists():
            delivery=read(delivery_path)
            for key in ['models','figures','reports']:
                if key in delivery:
                    delivery[key]=[identity(row['path']) for row in delivery[key]]
            if 'metrics' in delivery:
                delivery['metrics']={key:identity(row['path']) for key,row in delivery['metrics'].items()}
            manifest=read(delivery_path.parent/'figure_manifest.json')
            delivery.update(updated=time.time(),figure_signature=manifest['signature'],
                collision_registration_refinement=True)
            write(delivery_path,delivery)
            old_marker=root/'fine_whole/publication'/scene/'evaluated.json'
            if old_marker.exists():
                old=read(old_marker);old.update(complete=True,updated=time.time(),
                    delivery=str(delivery_path),figure_signature=manifest['signature'],
                    collision_registration_refinement=True)
                write(old_marker,old)
    result=dict(complete=True,scene=scene,updated=time.time(),composed_methods_rescored=rescored,
        composed_methods_reused_after_exact_checks=reused,
        background_numeric_scores_unchanged=True,figures_refreshed=figures,delivery_verified=verify)
    write(root/'collision_refinement_publication'/scene/'evaluation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--updates',type=Path,required=True)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--publish-only',action='store_true')
    p.add_argument('--skip-evaluation',action='store_true')
    p.add_argument('--skip-figures',action='store_true')
    p.add_argument('--skip-verification',action='store_true')
    args=p.parse_args();payload=read(args.updates)
    updates=payload.get('object_updates',payload.get('updates'))
    if updates is None:raise ValueError('Updates JSON needs object_updates and validation')
    validation=payload.get('validation',dict(complete=False,objects={}))
    root=work_root(args.output)
    lock=open(root/'collision_refinement_publication.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    publish_registration(args.scene,updates,validation,args.output)
    if args.publish_only:return
    refresh_composed_render(args.scene,args.output)
    if not args.skip_evaluation:
        refresh_evaluation(args.scene,args.output,not args.skip_figures,not args.skip_verification)


if __name__=='__main__':main()
