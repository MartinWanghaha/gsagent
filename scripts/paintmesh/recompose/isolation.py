"""Disambiguate a reused semantic ID using the actual pre-inpaint removal.

This is used for recorded target-isolation repairs: a semantic ID can cover
multiple objects, while the removal retains one of them. It must not be generated
again. Membership comes from unchanged Gaussian centers, never test RGB/masks.
"""
from pathlib import Path
import hashlib
import json
import shutil
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .io import read_json,write_json,record_path,fingerprint,ply,xyz
from .geometry import RayMesh,camera_rays
from .evidence import original_cameras


def classify_removed_surface(points,removed,retained,maximum_distance):
    """Only geometry nearer the truly deleted component can define its mask."""
    a=cKDTree(removed).query(points,workers=4)[0]
    b=cKDTree(retained).query(points,workers=4)[0] if len(retained) else np.full(len(points),np.inf)
    return (a<b)&(a<maximum_distance)


def isolate_selection(repo,inputs,root,max_size=768):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    manifest=read_json(inputs['manifest']);removal=record_path(manifest['inputs']['removal_manifest'])
    removed_manifest=read_json(removal);remaining=record_path(removed_manifest['outputs']['removed_3dgs'])
    sources={k:fingerprint(inputs[k]) for k in ['original_gaussians','gaussian_labels']}
    sources.update(removal=fingerprint(removal),retained=fingerprint(remaining))
    source_masks=sorted(Path(inputs['mask_root']).glob('*.png'))
    sources['masks']=[fingerprint(p) for p in source_masks]
    sources['implementation']=fingerprint(__file__)
    signature=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest();receipt=root/'isolation.json'
    if receipt.exists():
        prior=read_json(receipt)
        if prior.get('signature')==signature and all(Path(r['path']).exists() and fingerprint(r['path'])['sha256']==r['sha256'] for r in prior['outputs']):
            return dict(inputs,gaussian_labels=str(root/'gaussian_labels.npy'),mask_root=str(root/'masks'),isolation_manifest=str(receipt))
    original=xyz(ply(inputs['original_gaussians'])['vertex'].data);keep=xyz(ply(remaining)['vertex'].data)
    labels=np.load(inputs['gaussian_labels']);out_labels=labels.copy()
    if len(labels)!=len(original):raise ValueError('Original Gaussian labels and centers disagree')
    # Reject comparing unrelated/fine-tuned reconstructions as a deletion record.
    if np.max(cKDTree(original).query(keep,workers=4)[0])>1e-6:raise ValueError('Removal PLY is not an unchanged subset of the original centers')
    remain_tree=cKDTree(keep);deleted=remain_tree.query(original,workers=4)[0]>1e-6
    cameras=original_cameras(repo,inputs['source'],max_size);camera_map={Path(c['name']).stem:c for c in cameras}
    masks=root/'masks';masks.mkdir(exist_ok=True);summary={}
    # Existing broad evidence is used only for ray intersection, not object identity.
    proxies={}
    for target in inputs['target_ids']:
        target_mask=labels==target;selected=original[target_mask&deleted];other=original[target_mask&~deleted]
        if len(selected)<30:raise ValueError(f'Target {target}: too few truly removed centers')
        out_labels[target_mask&~deleted]=np.iinfo(out_labels.dtype).max
        proxy=root.parent/'objects'/str(target)/'evidence/original_proxy.ply'
        if not proxy.exists():raise ValueError('Prepare broad evidence once before target-isolation refinement')
        mesh=ply(proxy);ray=RayMesh(xyz(mesh['vertex'].data),np.ascontiguousarray(mesh['face'].data['vertex_indices'],dtype=np.uint32))
        diameter=float(np.linalg.norm(np.quantile(selected,.98,axis=0)-np.quantile(selected,.02,axis=0)))
        proxies[target]=(selected,other,ray,diameter)
        summary[target]=dict(removed_centers=len(selected),retained_same_label=len(other),diameter=diameter)
    mask_counts={}
    for path in source_masks:
        c=camera_map.get(path.stem)
        if c is None:continue  # test* is deliberately excluded
        raw=np.array(Image.open(path));h,w=c['height'],c['width'];scaled=cv2.resize(raw,(w,h),interpolation=cv2.INTER_NEAREST);result=scaled.copy()
        for target,(selected,other,ray,diameter) in proxies.items():
            ys,xs=np.where(scaled==target)
            if not len(xs):continue
            rays=camera_rays(c['K'],c['w2c'],np.c_[xs,ys]);depth,_,_=ray.cast(rays)
            hit=np.isfinite(depth)&(depth>0);good=np.zeros(len(xs),bool)
            points=rays[hit,:3]+depth[hit,None]*rays[hit,3:]
            good[hit]=classify_removed_surface(points,selected,other,.05*diameter)
            kept_mask=np.zeros((h,w),'uint8');kept_mask[ys[good],xs[good]]=1
            # Close one-pixel ray/mesh gaps, but never expand beyond the semantic mask.
            kept_mask=cv2.morphologyEx(kept_mask,cv2.MORPH_CLOSE,np.ones((3,3),'uint8')).astype(bool)&(scaled==target)
            result[(scaled==target)&~kept_mask]=np.iinfo(result.dtype).max
            mask_counts.setdefault(target,[]).append(dict(view=c['name'],before=len(xs),after=int(kept_mask.sum())))
        Image.fromarray(cv2.resize(result,(raw.shape[1],raw.shape[0]),interpolation=cv2.INTER_NEAREST)).save(masks/path.name)
    scene_info=Path(inputs['mask_root'])/'scene.json'
    if scene_info.exists():shutil.copy2(scene_info,masks/'scene.json')
    np.save(root/'gaussian_labels.npy',out_labels)
    write_json(receipt,dict(signature=signature,sources=sources,targets=summary,mask_counts=mask_counts,
                           reason='Use only centers actually removed before inpainting; exclude retained instances sharing the semantic ID.',
                           outputs=[fingerprint(root/'gaussian_labels.npy')]+[fingerprint(p) for p in sorted(masks.glob('*.png'))]))
    return dict(inputs,gaussian_labels=str(root/'gaussian_labels.npy'),mask_root=str(masks),isolation_manifest=str(receipt))
