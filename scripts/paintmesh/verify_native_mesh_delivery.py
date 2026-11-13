#!/usr/bin/env python3
"""Audit native RGB publication separately from frozen geometry support."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,METHODS,STATES,read,write
from mesh_benchmark.layout import work_root,public_root,configure_sources
from evaluate_native_mesh_rgb import rendering,verified_file_fingerprint
from recompose.io import fingerprint


def verify(output):
    root=work_root(output);configure_sources(root);public=public_root(root)
    counts=dict(colored_meshes=0,native_renderer_jobs=0,mode_metrics=0,
                shared_view_masks=0,published_rgb_links=0,subplots=0)
    for method in ['come','gaussian_wrapping']:
        for scene in SCENES:
            out=BASE/method/scene;receipt=read(out/'mesh_color_complete.json')
            assert receipt['complete'] and receipt['geometry_unchanged']
            assert receipt['training_performed'] is False
            mesh=Path(receipt['colored_mesh']['path']).resolve(strict=True)
            assert (out/'mesh_colored.ply').resolve(strict=True)==mesh
            assert (out/'mesh.ply').resolve(strict=True)==mesh
            assert Path(read(out/'mesh_complete.json')['path']).resolve(strict=True)==mesh
            assert (public/'meshes'/scene/(method+'_colored.ply')).resolve(strict=True)==mesh
            counts['colored_meshes']+=1
    for scene in SCENES:
        info=read(root/'common'/scene/'complete.json')
        graph=read(public/'recomposition'/scene/'composition/scene.json')
        original=Path(graph['input_selection']['original_mesh']).resolve(strict=True)
        fine=graph.get('stage1_whole',{})
        if fine.get('complete'):
            receipt=read(fine['extraction'])
            assert receipt['complete'] and not receipt['geometry_simplified'] and not receipt['connected_components_removed']
            assert receipt['request']['voxel']==.002 and receipt['request']['sdf_trunc']==.008
            expected=[Path(s['path']).resolve(strict=True) for s in fine['mesh_specs']]
            exported=read(public/'meshes'/scene/'stage1_world/mesh_scene.gltf')
            assert len(exported['meshes'])==len(expected)
            assert (public/'meshes'/scene/'stage1_original_truncated.ply').resolve(strict=True)==original
        else:
            assert (public/'meshes'/scene/'stage1.ply').resolve(strict=True)==original
            expected=[original]
        for state in STATES:
            first=read(root/'cache/paintmesh_stage1'/scene/state/'complete.json')
            assert [Path(s['identity']['path']).resolve(strict=True) for s in first['request']['specs']]==expected
            assert [Path(s['path']).resolve(strict=True) for s in first['mesh_counts']]==expected
        cameras=[c for c in info['cameras'] if c['part']=='held_out']
        names=[c['name'] for c in cameras]
        for method in METHODS:
            if method in ['paintmesh','paintmesh_stage1']:continue
            meta=read(root/'native_renderers'/method/scene/'complete.json')
            assert meta['complete'] and not meta['pilot'] and meta['views']==len(cameras)
            assert meta['request']['gaussian_renderer_used'] is False
            assert meta.get('integer_index_precision_preserved') is True
            assert meta['request']['mesh']==verified_file_fingerprint(root,read(BASE/method/scene/'mesh_complete.json')['path'])
            assert meta['request']['adapter']==fingerprint(HERE/'render_native_colored_meshes.py')
            counts['native_renderer_jobs']+=1
        for state in STATES:
            support=read(root/'common'/scene/(state+'_native_support')/'complete.json')
            sources={m:rendering(root,m,scene,state) for m in METHODS}
            metrics={m:read(root/state/m/scene/'metrics.json') for m in METHODS}
            for method,metric in metrics.items():
                assert metric['complete'] and metric['native_mesh_rendering_version']==1
                assert metric['gaussian_renderer_used'] is False and metric['appearance_optimized'] is False
                if method in ['come','gof'] and read(root/'protocol.json').get('native_support_validation',{}).get('enabled'):
                    repair=metric['geometry_postprocessing']
                    assert repair['reference_data_used'] is False and repair['coordinates_and_colors_exact']
                    assert repair['retained_face_records_exact'] and not repair['training_performed']
                assert metric['request']['support_id']==support['request_id']
                geo=read(root/'cache'/method/scene/state/'geometry_metrics.json')
                assert metric['geometry']==geo
                assert metric['request']['geometry_support_id']==geo['support_id']
                assert [r['name'] for r in metric['per_view']]==names
                for key in ['PSNR','SSIM','LPIPS']:
                    values=[r[key] for r in metric['per_view']]
                    assert all(np.isfinite(v) for v in values)
                    assert abs(float(np.mean(values))-metric['mean'][key])<1e-10
                if method=='paintmesh' and state=='recomposition':
                    assert metric['quality_status']==read(public/'recomposition'/scene/'composition/scene.json')['status']
                counts['mode_metrics']+=1
            for camera in cameras:
                stem=camera['stem'];common=root/'common'/scene
                foreground=np.asarray(Image.open(common/'held_out'/(stem+'_foreground.png')))>127
                eligible=~foreground if state=='background' else np.ones(foreground.shape,dtype=bool)
                expected=eligible.copy();own={}
                for method,source in sources.items():
                    depth=np.load(source['depth']/(stem+'.npy'),mmap_mode='r')
                    own[method]=np.isfinite(depth)&(depth>0)&eligible
                    expected&=own[method]
                    actual=root/state/method/scene/'renders'/(stem+'.png')
                    assert actual.resolve(strict=True)==(source['rgb']/(stem+'.png')).resolve(strict=True)
                    with Image.open(actual) as im:assert im.size==(camera['width'],camera['height'])
                    counts['published_rgb_links']+=1
                allowed=np.asarray(Image.open(common/(state+'_native_support')/(stem+'.png')))>127
                np.testing.assert_array_equal(allowed,expected)
                assert allowed.any()
                if state=='background':assert not np.any(allowed&foreground)
                for method,metric in metrics.items():
                    row=next(r for r in metric['per_view'] if r['name']==camera['name'])
                    assert row['evaluation_pixels']==int(allowed.sum())
                    assert row['own_pixels']==int(own[method].sum())
                    mask=root/state/method/scene/'diagnostics'/(stem+'_evaluation_mask.png')
                    np.testing.assert_array_equal(np.asarray(Image.open(mask))>127,allowed)
                    # Recompute from published PNG bytes rather than trusting
                    # that a score belongs to the current linked render.
                    pred=np.asarray(Image.open(root/state/method/scene/'renders'/(stem+'.png')).convert('RGB'))
                    target=np.asarray(Image.open(common/'held_out'/(stem+'_gt.png')).convert('RGB'))
                    difference=pred[allowed].astype('float64')-target[allowed].astype('float64')
                    mse=float(np.sum(difference*difference)/(3*int(allowed.sum()))/255**2)
                    psnr=float(-10*np.log10(max(mse,1e-12)))
                    assert abs(psnr-row['PSNR'])<2e-5,(method,scene,state,stem,'stale/wrong PSNR',psnr,row['PSNR'])
                counts['shared_view_masks']+=1
    figures=public/'figures_all_methods'
    for path in sorted((figures/'subplots').glob('*/*/*/source.json')):
        record=read(path);assert record['status'] in ['complete','reference']
        assert Path(record['source_rgb']).exists()
        np.testing.assert_array_equal(np.asarray(Image.open(path.parent/'evaluation_mask.png')),
                                      np.asarray(Image.open(record['source_evaluation_mask'])))
        with Image.open(path.parent/'rgb.png') as copied,Image.open(record['source_rgb']) as original:
            np.testing.assert_array_equal(np.asarray(copied),np.asarray(original))
        assert (path.parent/'panel.png').exists()
        counts['subplots']+=1
    view_count=sum(read(root/'common'/s/'complete.json')['held_out_count'] for s in SCENES)
    expected=dict(colored_meshes=2*len(SCENES),native_renderer_jobs=4*len(SCENES),
        mode_metrics=len(METHODS)*len(SCENES)*len(STATES),shared_view_masks=view_count*len(STATES),
        published_rgb_links=view_count*len(STATES)*len(METHODS),
        subplots=(len(METHODS)+1)*len(SCENES)*len(STATES))
    assert counts==expected,(counts,expected)
    result=dict(complete=True,updated=time.strftime('%Y-%m-%d %H:%M:%S %Z'),counts=counts,
                original_geometry_and_color_properties='See colored_mesh_delivery_check.json for byte/property verification',
                native_rgb_and_frozen_geometry_supports_separately_verified=True,
                published_png_psnr_independently_recomputed=True,
                gaussian_rgb_substitution=False,appearance_fitting=False)
    write(public/'docs/NATIVE_RGB_DELIVERY_CHECK.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'mesh_comparison_whole')
    verify(p.parse_args().output)
