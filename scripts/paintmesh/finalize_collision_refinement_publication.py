#!/usr/bin/env python3
"""Refresh all comparison figures once after per-scene collision publication.

This finalizer never optimizes, moves or renders scene geometry. Per-scene
publish_collision_refinement.py must first finish current composed metrics.
It then regenerates global figures/subplots and verifies the existing actual
mesh renders, finally recording fresh delivery identities for all scenes.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,METHODS,STATES,read,write,identity
from mesh_benchmark.layout import work_root,configure_sources


def _numerics(value):
    """Drop provenance only, retaining every background metric value."""
    if isinstance(value,dict):
        return {k:_numerics(v) for k,v in value.items() if k not in ['support_id']}
    if isinstance(value,list):return [_numerics(v) for v in value]
    return value


def preflight(public,scenes):
    root=work_root(public);configure_sources(root)
    baseline=read(public/'collision_refinement/before_numeric_and_asset_checks.json')['scenes']
    evidence=[]
    for scene in scenes:
        graph=read(public/'recomposition'/scene/'composition/scene.json')
        publication=root/'collision_refinement_publication'/scene
        receipt=read(publication/'evaluation.json')
        if not receipt.get('complete'):
            raise ValueError('Current composed evaluation is incomplete: '+scene)
        cache=read(root/'cache/paintmesh'/scene/'recomposition/complete.json')
        if cache.get('geometry_pending_rescore'):
            raise ValueError('Composed geometry scores are pending: '+scene)
        for node,spec in zip(graph['objects'],cache['request']['specs'][-len(graph['objects']):]):
            if spec['identity']!=identity(node['mesh']['path']) or spec['transform']!=node['object_to_world']:
                raise ValueError('Composed cache uses stale object geometry/pose: '+scene)
            if node['object_to_world']!=node['original_pose']:
                raise ValueError('Original Gaussian movement must remain identity: '+scene)
        current=read(root/'background/paintmesh'/scene/'metrics.json')
        before=baseline[scene]['background']
        for field in ['mean','per_view','geometry']:
            if _numerics(current[field])!=_numerics(before[field]):
                raise ValueError('Background numeric values changed: '+scene+'/'+field)
        for state in STATES:
            common=root/'common'/scene
            support=read(common/(state+'_native_support')/'complete.json')
            for method in METHODS:
                metrics=read(root/state/method/scene/'metrics.json')
                geometry=read(root/'cache'/method/scene/state/'geometry_metrics.json')
                if not metrics.get('complete') or metrics['geometry']!=geometry:
                    raise ValueError('Incomplete/stale scoring: '+scene+'/'+method+'/'+state)
                if metrics['request']['support_id']!=support['request_id']:
                    raise ValueError('Stale common scoring domain: '+scene+'/'+method+'/'+state)
        for stamp in baseline[scene]['gaussian_sources']:
            stat=Path(stamp['path']).stat()
            if stat.st_size!=stamp['size'] or stat.st_mtime_ns!=stamp['mtime_ns']:
                raise ValueError('Original Gaussian source changed: '+stamp['path'])
        evidence.append(dict(scene=scene,graph=identity(public/'recomposition'/scene/'composition/scene.json'),
            background_numeric_scores_unchanged=True,composed_scores_current=True,
            original_gaussian_source_stats_unchanged=True))
    return root,evidence


def delivery_receipt(public, root, scene):
    """Record actual delivered assets, including historical non-fine stage 1.

    garden_toys has an original stage-1 PLY rather than a fine whole-scene
    stage1_world export. A collision-only update must not invent that asset.
    """
    from publish_fine_whole_meshes import scene_delivery_receipt
    graph=read(public/'recomposition'/scene/'composition/scene.json')
    if graph.get('stage1_whole',{}).get('complete'):
        return scene_delivery_receipt(scene)
    from plot_mesh_method_comparison import input_signature, choose_view
    folder=public/'figures_all_methods/scenes'/scene
    manifest=read(folder/'figure_manifest.json');camera,_,_=choose_view(root,scene)
    signature=input_signature(root,scene,camera)
    if manifest['signature']!=signature or any(r['method'] and r['status']!='complete' for r in manifest['records']):
        raise ValueError('Historical stage-1 scene figures are incomplete or stale: '+scene)
    metrics={}
    for state in STATES:
        support=read(root/'common'/scene/(state+'_native_support/complete.json'))
        for method in METHODS:
            path=root/state/method/scene/'metrics.json';metric=read(path)
            geometry=read(root/'cache'/method/scene/state/'geometry_metrics.json')
            if not metric.get('complete') or metric['request']['support_id']!=support['request_id'] or metric['geometry']!=geometry:
                raise ValueError('Historical stage-1 scene scores are incomplete: '+scene)
            metrics[state+'/'+method]=identity(path)
    assets=[Path(graph['input_selection']['original_mesh']),public/'meshes'/scene/'world/mesh_scene.gltf',
        public/'recomposition'/scene/'composition/world/mesh_scene.gltf']
    pictures=[folder/(name+'.png') for name in ['all_methods','background','recomposition']]
    pictures.append(public/'figures_all_methods/geometry_latest'/(scene+'_all_methods.png'))
    return dict(complete=True,scene=scene,updated=time.time(),camera=camera['name'],figure_signature=signature,
        models=[identity(p) for p in assets],metrics=metrics,figures=[identity(p) for p in pictures],
        reports=[identity(public/'reports'/(s+'.md')) for s in STATES],native_triangles=True,
        additional_image_downsampling=False,mesh_decimation=False,source_stage1_mesh_preserved=True,
        stage1_asset_kind='Existing original stage-1 PLY; no fine whole-scene stage1_world export exists',
        stage1_fine_whole_export_available=False)


def finalize(public,scenes=SCENES,preflight_only=False,verified_figures_resume=False):
    public=Path(public);root=work_root(public);configure_sources(root)
    lock=open(root/'collision_refinement_publication.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    root,evidence=preflight(public,scenes)
    if preflight_only:return dict(complete=True,preflight_only=True,scenes=evidence)
    if verified_figures_resume:
        # Resume only receipt creation after a post-verification metadata
        # failure. Any changed metric/image/scene invalidates this shortcut.
        check_path=public/'docs/NATIVE_RGB_DELIVERY_CHECK.json';check=read(check_path)
        if not check.get('complete') or not check.get('published_png_psnr_independently_recomputed'):
            raise ValueError('No completed independent figure/score verification to resume')
        verified_at=check_path.stat().st_mtime_ns
        relevant=[public/'recomposition'/s/'composition/scene.json' for s in SCENES]
        for scene in SCENES:
            relevant += list((root/'common'/scene).rglob('*.png'))
            relevant += [root/state/method/scene/'metrics.json' for state in STATES for method in METHODS]
            relevant += list((root/'cache/paintmesh'/scene/'recomposition/native_renders').glob('*.png'))
        relevant += list((public/'figures_all_methods').rglob('*.png'))
        if any(p.stat().st_mtime_ns>verified_at for p in relevant):
            raise ValueError('Verified scene/score/figure inputs changed; full refresh is required')
    else:
        # Reports first, then one pass for all full comparison figures; copying
        # method subplots comes after current full-scene panels exist.
        from mesh_benchmark.report import update
        from plot_mesh_method_comparison import build,input_signature,choose_view
        from publish_latest_mesh_geometry import figures as geometry_figures
        from organize_mesh_subplots import build as organize
        update(root)
        build(root,public/'figures_all_methods',SCENES)
        geometry_figures(public,SCENES)
        organize(public/'figures_all_methods',public/'figures_all_methods/subplots')
        # Independent PSNR recomputation from PNG bytes and equal masks.
        from verify_native_mesh_delivery import verify
        verify(public)
    receipts={}
    for scene in scenes:
        receipt=delivery_receipt(public,root,scene)
        receipt.update(collision_registration_refinement=True,updated=time.time())
        destination=public/'figures_all_methods/scenes'/scene/'delivery.json'
        write(destination,receipt);receipts[scene]=identity(destination)
        old_marker=root/'fine_whole/publication'/scene/'evaluated.json'
        if old_marker.exists():
            old=read(old_marker);old.update(complete=True,updated=time.time(),
                delivery=str(destination),figure_signature=receipt['figure_signature'],
                collision_registration_refinement=True)
            write(old_marker,old)
        p=root/'collision_refinement_publication'/scene/'evaluation.json'
        value=read(p);value.update(figures_refreshed=True,delivery_verified=True,
            global_batch_refresh=True,updated=time.time());write(p,value)
    result=dict(complete=True,updated=time.time(),scenes=list(scenes),
        evidence=evidence,delivery_receipts=receipts,
        figures=str(public/'figures_all_methods/overview'),
        geometry_rendering_repeated=False,source_gaussians_changed=False,
        native_mesh_delivery_verified=True,resumed_after_verified_figures=verified_figures_resume)
    write(public/'collision_refinement/final_publication.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=BASE/'mesh_comparison_whole')
    p.add_argument('--scenes',nargs='+',choices=SCENES,default=SCENES)
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();result=finalize(a.output,a.scenes,a.preflight_only)
    print(result,flush=True)


if __name__=='__main__':main()
