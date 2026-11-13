#!/usr/bin/env python3
"""Read-only per-view audit of CoMe mesh color integration and selection."""
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import read,write


def inspect_native(scene,query_file,cameras,gaussian,pipe,settings,official,model):
    import torch
    from diff_gaussian_rasterization import ExtendedSettings,GlobalSortOrder
    data=np.load(query_file);xyz=data['xyz'];original=data['original_rgb'];n=len(xyz)
    points=torch.from_numpy(np.asarray(xyz,dtype='float32')).cuda()
    background=torch.zeros(3,device='cuda')
    settings.meshing_settings.return_color=True
    expected_alpha,expected_color=official.evaluate_alpha(points,cameras,gaussian,pipe,background,.003,splat_args=settings)
    native=np.rint(np.clip(expected_color.cpu().numpy(),0,1)*255).astype('uint8')
    opacity=[];colors=[];frustums=[]
    for i,camera in enumerate(cameras):
        config=ExtendedSettings.from_json(str(model/'config.json'))
        config.meshing_settings.return_color=True
        config.meshing_settings.alpha_early_stop=False
        config.sort_settings.sort_order=GlobalSortOrder.PTD_MAX
        alpha=torch.ones(n,device='cuda')
        with torch.no_grad():
            result=official.integrate(points,alpha,camera,gaussian,pipe,background,kernel_size=.003,splat_args=config)
        opacity.append(alpha.cpu().numpy());colors.append(result['color_integrated'].cpu().numpy())
        W=camera.world_view_transform.cpu().numpy();P=np.c_[xyz,np.ones(n)]@W
        u=P[:,0]/np.maximum(P[:,2],1e-6)*camera.focal_x+camera.image_width/2
        v=P[:,1]/np.maximum(P[:,2],1e-6)*camera.focal_y+camera.image_height/2
        frustums.append((P[:,2]>.2)&(u>=0)&(u<camera.image_width)&(v>=0)&(v<camera.image_height))
        if i%10==0:print('COME_COLOR_VIEW',scene,i,'/',len(cameras),flush=True)
    opacity=np.stack(opacity);colors=np.stack(colors);frustums=np.stack(frustums)
    minimum=opacity.argmin(0);maximum=opacity.argmax(0);idx=np.arange(n)
    chosen=colors[minimum,idx];min_alpha=opacity[minimum,idx];max_alpha=opacity[maximum,idx]
    best=colors[np.argmax(np.where(frustums,opacity,-1),0),idx]
    # A constant white Gaussian field measures the sum of the *same* color
    # compositing weights. This distinguishes a dark material from an RGB
    # value still premultiplied by incomplete ray coverage.
    white=torch.ones_like(gaussian.get_xyz)
    weight=np.zeros(n,dtype='float32');white_alpha_error=0.
    for i,camera in enumerate(cameras):
        selected=np.flatnonzero(minimum==i)
        if not len(selected):continue
        config=ExtendedSettings.from_json(str(model/'config.json'))
        config.meshing_settings.return_color=True
        config.meshing_settings.alpha_early_stop=False
        config.sort_settings.sort_order=GlobalSortOrder.PTD_MAX
        alpha=torch.ones(len(selected),device='cuda')
        with torch.no_grad():
            result=official.integrate(points[selected],alpha,camera,gaussian,pipe,background,kernel_size=.003,
                                      splat_args=config,override_color=white)
        response=result['color_integrated'].cpu().numpy()
        np.testing.assert_allclose(response[:,0],response[:,1],atol=1e-6)
        np.testing.assert_allclose(response[:,0],response[:,2],atol=1e-6)
        weight[selected]=response[:,0]
        white_alpha_error=max(white_alpha_error,float(np.abs(alpha.cpu().numpy()-min_alpha[selected]).max()))
    normalized=chosen/np.maximum(weight[:,None],1e-8)
    groups={}
    def stats(x):
        return np.quantile(x,[0,.1,.5,.9,1]).tolist()
    for label,gid in [('black',0),('regular',1)]:
        keep=data['groups']==gid
        if not keep.any():continue
        groups[label]=dict(points=int(keep.sum()),original_mean_rgb=original[keep].mean(0).tolist(),
            native_mean_rgb=native[keep].mean(0).tolist(),
            original_native_mean_byte_difference=float(np.abs(original[keep].astype('float64')-native[keep]).mean()),
            per_view_min_opacity_quantiles=stats(min_alpha[keep]),per_view_max_opacity_quantiles=stats(max_alpha[keep]),
            chosen_color_quantiles=stats(chosen[keep].mean(1)),
            high_opacity_color_quantiles=stats(best[keep].mean(1)),
            selected_zero_color_fraction=float((chosen[keep].max(1)<1e-6).mean()),
            selected_zero_opacity_fraction=float((min_alpha[keep]<1e-6).mean()),
            selected_opacity_below_01_fraction=float((min_alpha[keep]<.1).mean()),
            full_gaussian_rgb_mean=data['gaussian_rgb'][keep].mean(0).tolist(),
            selected_camera_frustum_fraction=float(frustums[minimum,idx][keep].mean()),
            color_compositing_weight_quantiles=stats(weight[keep]),
            normalized_mean_rgb=(np.clip(normalized[keep],0,1)*255).mean(0).tolist(),
            normalized_color_quantiles=stats(normalized[keep].mean(1)),
            native_vs_full_gaussian_mean_byte_error=float(np.abs(native[keep].astype('float64')-data['gaussian_rgb'][keep]).mean()),
            normalized_vs_full_gaussian_mean_byte_error=float(np.abs(np.clip(normalized[keep],0,1)*255-data['gaussian_rgb'][keep]).mean()))
    np.savez(Path(query_file).with_name('per_view_colors.npz'),opacity=opacity,colors=colors,frustums=frustums,
        native_rgb=native,official_field_alpha=expected_alpha.cpu().numpy(),selected_camera=minimum,
        camera_names=np.array([c.image_name for c in cameras]),maximum_camera=maximum,
        selected_color_weight=weight,normalized_color=normalized)
    report=dict(complete=True,scene=scene,updated=time.time(),groups=groups,
        maximum_independent_min_color_difference=float(np.abs(chosen-expected_color.cpu().numpy()).max()),
        maximum_white_field_alpha_difference=white_alpha_error,
        geometry_unchanged=True,colors_unchanged=True,read_only=True,
        source='Unmodified official CoMe integrate; independent per-camera alpha initialized to one')
    write(Path(query_file).with_name('audit.json'),report)
    print('COME_BLACK_COLOR_AUDIT',scene,report,flush=True)
    return report
