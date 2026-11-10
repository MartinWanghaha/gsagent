"""CPU triangle visibility and explicit shared image/geometry scoring conventions."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / 'output/3dgs_with_mesh'
RECOMPOSE = Path(os.environ.get('PAINTMESH_RECOMPOSE_ROOT',str(REPO/'output/paintmesh/Inpaint360GS/recomposition')))
SCENES = ['bag','car','cone_red','cone_yellow','cube','doppelherz','fruits','garden_toys','redbull','toys','truck']
METHODS = ['2dgs','come','gof','gaussian_wrapping','paintmesh']
STATES = ['background','recomposition']

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    from recompose.io import write_json
    write_json(path, data)

def identity(path):
    p = Path(path).resolve(strict=True); s = p.stat()
    return {'path':str(p), 'size_bytes':s.st_size, 'mtime_ns':s.st_mtime_ns}

def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def cameras(scene):
    """Use the original COLMAP K and pose, but baseline's exact decoded image sizes."""
    split = read(BASE/'_data'/scene/'split.json')
    source = Path(split['source'])
    file = REPO/'submodules/Inpaint360GS/tools/read_write_model.py'
    spec = importlib.util.spec_from_file_location('mesh_eval_colmap', file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    intrinsics, extrinsics, _ = mod.read_model(str(source/'sparse/0'), ext='.bin')
    images = {x.name:x for x in extrinsics.values()}
    sizes = {x['name']:(x['width'],x['height']) for x in split['images']}
    result = []
    for part in ['train','held_out']:
        for name in split[part]:
            assert not Path(name).name.lower().startswith('test')
            im = images[name]; c = intrinsics[im.camera_id]; p = c.params
            if c.model == 'SIMPLE_PINHOLE': fx,fy,cx,cy = p[0],p[0],p[1],p[2]
            elif c.model == 'PINHOLE': fx,fy,cx,cy = p
            else: raise ValueError(f'Undistorted pinhole cameras required: {c.model}')
            w,h = sizes[name]
            K = np.array([[fx*w/c.width,0,cx*w/c.width],[0,fy*h/c.height,cy*h/c.height],[0,0,1.]])
            T = np.eye(4); T[:3,:3] = im.qvec2rotmat(); T[:3,3] = im.tvec
            result.append(dict(name=name,stem=Path(name).stem,image=str(source/'images'/name),
                               width=w,height=h,K=K,w2c=T,part=part))
    assert len({c['name'] for c in result}) == len(result)
    return result

def image_mask(camera, selection):
    size = (camera['width'],camera['height'])
    image = np.asarray(Image.open(camera['image']).convert('RGB').resize(size, Image.Resampling.BILINEAR))
    path = Path(selection['mask_root'])/(camera['stem']+'.png')
    labels = np.asarray(Image.open(path))
    if labels.ndim != 2: raise ValueError(f'Instance label image required: {path}')
    mask = np.isin(labels, selection['target_ids']).astype('uint8')*255
    mask = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST)) > 127
    return image, mask

def rays(camera, pixels=None):
    from recompose.geometry import camera_rays
    w,h = camera['width'],camera['height']
    if pixels is None: pixels = np.arange(w*h)
    uv = np.c_[pixels%w+.5,pixels//w+.5]
    return camera_rays(camera['K'], camera['w2c'], uv)

def points_from_depth(camera, depth, pixels=None):
    rr = rays(camera,pixels)
    return rr[:,:3]+rr[:,3:]*np.asarray(depth).reshape(-1,1)

class MeshScene:
    """Embree CPU ray casting of original triangles; no simplification or splats."""
    def __init__(self, specs):
        import open3d as o3d
        self.o3d = o3d; self.scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        self.attributes = {}; self.counts = []
        from recompose.io import ply, xyz
        from recompose.geometry import transform
        for spec in specs:
            path = Path(spec['path']); asset = None
            if path.suffix == '.npz':
                with np.load(path) as a: asset = {k:a[k] for k in a.files}
                vertices = transform(asset['vertices'], spec['transform']).astype('float32')
                faces = np.asarray(asset['faces'], dtype='uint32'); colors = asset.get('colors')
                data = None
            else:
                data = ply(path); v = data['vertex'].data
                vertices = xyz(v).astype('float32'); faces = np.asarray(data['face'].data['vertex_indices'], dtype='uint32', order='C')
                colors = np.column_stack([v[n] for n in ['red','green','blue']])/255. if all(n in v.dtype.names for n in ['red','green','blue']) else None
            if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
                raise ValueError(f'Invalid/empty triangle mesh: {path}')
            if int(faces.max())>=len(vertices):raise ValueError(f'Out-of-range triangle indices: {path}')
            gid = self.scene.add_triangles(o3d.core.Tensor.from_numpy(vertices), o3d.core.Tensor.from_numpy(faces))
            self.attributes[gid] = (faces, colors, asset, data)
            self.counts.append(dict(path=str(path),vertices=len(vertices),faces=len(faces),has_native_color=colors is not None))
            # Open3D retains its own tensor handles. Release temporary vertex copies.
            del vertices

    def cast(self, rr, color=False):
        out = self.scene.cast_rays(self.o3d.core.Tensor.from_numpy(np.ascontiguousarray(rr,dtype='float32')),nthreads=4)
        z = out['t_hit'].numpy(); good = np.isfinite(z)&(z>0)
        rgb = None
        if color:
            rgb = np.zeros((len(z),3),dtype='float32'); geom = out['geometry_ids'].numpy()
            face = out['primitive_ids'].numpy(); uv = out['primitive_uvs'].numpy()
            from recompose.texture import sample_colors
            for gid,(faces,colors,asset,_) in self.attributes.items():
                keep = good&(geom==gid)
                if not keep.any() or colors is None: continue
                bary = np.c_[1-uv[keep].sum(1),uv[keep]]
                if asset is not None: rgb[keep] = sample_colors(asset,face[keep],bary)
                else: rgb[keep] = (colors[faces[face[keep]]]*bary[:,:,None]).sum(1)
        return np.where(good,z,0).astype('float32'), rgb

    def distance(self,points):
        return self.scene.compute_distance(self.o3d.core.Tensor.from_numpy(np.asarray(points,dtype='float32')),nthreads=4).numpy()

def specs_for(scene,method,state):
    document = read(RECOMPOSE/scene/'composition/scene.json')
    if method != 'paintmesh':
        p = BASE/method/scene/'mesh_complete.json'
        if not p.exists(): return None
        m = read(p)
        if not m.get('complete') or not Path(m['path']).exists(): return None
        return [dict(path=m['path'])]
    result = [dict(path=document['background']['mesh']['path'])]
    result += [dict(path=x['path']) for x in document['background'].get('patches',[])]
    if state == 'recomposition':
        result += [dict(path=x['mesh']['path'],transform=x['object_to_world']) for x in document['objects']]
    return result

def chamfer(a,b):
    """Symmetric *half-sum* means; report unsquared and squared definitions explicitly."""
    if not len(a) or not len(b): raise ValueError('Empty surface samples')
    ab = cKDTree(b).query(a,k=1,workers=4)[0]
    ba = cKDTree(a).query(b,k=1,workers=4)[0]
    return {'CD_L1':float((ab.mean()+ba.mean())/2),
            'CD_L2':float((np.mean(ab**2)+np.mean(ba**2))/2),
            'a_to_b':float(ab.mean()),'b_to_a':float(ba.mean()),
            'a_p95':float(np.quantile(ab,.95)),'b_p95':float(np.quantile(ba,.95))}

def surface_metrics(pred,reference,extent,thresholds=(.005,.01,.02)):
    """Thresholded point-cloud F-score on an independently specified support.

    Support selection must never use nearest-neighbor distance or F-score tau.
    """
    if extent<=0 or not len(pred) or not len(reference):raise ValueError('Empty/invalid geometry support')
    ab=cKDTree(reference).query(pred,k=1,workers=4)[0]
    ba=cKDTree(pred).query(reference,k=1,workers=4)[0]
    out=dict(CD_L1=float((ab.mean()+ba.mean())/2),CD_L2=float((np.mean(ab**2)+np.mean(ba**2))/2),
             accuracy=float(ab.mean()),completeness=float(ba.mean()),
             accuracy_p95=float(np.quantile(ab,.95)),completeness_p95=float(np.quantile(ba,.95)),
             reference_points=len(reference),surface_samples=len(pred))
    out['CD_L1_normalized']=out['CD_L1']/extent
    out['F_scores']={}
    for fraction in thresholds:
        tau=float(fraction*extent);p=float((ab<=tau).mean());r=float((ba<=tau).mean())
        out['F_scores'][str(fraction)]=dict(threshold_fraction=fraction,threshold_colmap_units=tau,
                precision=p,recall=r,F1=float(2*p*r/(p+r)) if p+r else 0.)
    out.update(F1=out['F_scores']['0.01']['F1'],precision=out['F_scores']['0.01']['precision'],recall=out['F_scores']['0.01']['recall'])
    return out

def psnr(a,b,mask=None):
    e = np.mean((a-b)**2,axis=-1)
    mse = float(e[mask].mean() if mask is not None else e.mean())
    return float(-10*np.log10(max(mse,1e-12)))

def ssim_map(a,b):
    # Match the Gaussian Splatting 11x11 sigma=1.5 implementation (zero padding).
    import cv2
    k = cv2.getGaussianKernel(11,1.5)
    blur = lambda x: cv2.sepFilter2D(x,-1,k,k,borderType=cv2.BORDER_CONSTANT)
    ma,mb = blur(a),blur(b)
    aa,bb,ab = blur(a*a)-ma*ma,blur(b*b)-mb*mb,blur(a*b)-ma*mb
    return (((2*ma*mb+.01**2)*(2*ab+.03**2))/((ma*ma+mb*mb+.01**2)*(aa+bb+.03**2))).mean(-1)

def masked_ssim_map(a,b,mask):
    """Normalize local Gaussian moments over valid pixels only (including holes)."""
    if np.all(mask):return ssim_map(a,b)
    import cv2
    # Stable moments matter at mask edges: float32 cancellation can otherwise
    # make identical constant images score below one after renormalization.
    a=np.asarray(a,dtype='float64');b=np.asarray(b,dtype='float64')
    k=cv2.getGaussianKernel(11,1.5)
    blur=lambda x:cv2.sepFilter2D(x,-1,k,k,borderType=cv2.BORDER_CONSTANT)
    weights=np.asarray(mask,dtype='float64')[:,:,None]
    denominator=np.maximum(blur(weights[:,:,0]),1e-12)[:,:,None]
    ma=blur(a*weights)/denominator;mb=blur(b*weights)/denominator
    aa=np.maximum(blur(a*a*weights)/denominator-ma*ma,0)
    bb=np.maximum(blur(b*b*weights)/denominator-mb*mb,0)
    ab=blur(a*b*weights)/denominator-ma*mb
    return (((2*ma*mb+.01**2)*(2*ab+.03**2))/((ma*ma+mb*mb+.01**2)*(aa+bb+.03**2))).mean(-1)

def uniform_surface(specs,count=250000,seed=0,box=None):
    """Area-uniform sampling for a supplied dense GT; chunked for 60M-face inputs."""
    from recompose.io import ply,xyz
    from recompose.geometry import transform
    vertices=[]; faces=[]; areas=[]
    for spec in specs:
        p=Path(spec['path'])
        if p.suffix=='.npz':
            a=np.load(p);v=transform(a['vertices'],spec['transform']);f=a['faces']
        else:
            a=ply(p);v=xyz(a['vertex'].data);f=a['face'].data['vertex_indices']
            if 'transform' in spec:v=transform(v,spec['transform'])
        ar=np.empty(len(f),dtype='float64')
        for start in range(0,len(f),100000):
            tri=v[f[start:start+100000]]
            ar[start:start+len(tri)]=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
        vertices.append(v);faces.append(f);areas.append(ar)
    area=np.concatenate(areas);cdf=np.cumsum(area)
    if not np.isfinite(cdf[-1]) or cdf[-1]<=0:raise ValueError('Invalid mesh surface area')
    rng=np.random.default_rng(seed);chunks=[]
    # Rejection sampling preserves area uniformity within an independently declared ROI.
    while sum(map(len,chunks))<count:
        ids=np.searchsorted(cdf,rng.random(count)*cdf[-1]);offset=0;batch=[]
        for v,f in zip(vertices,faces):
            chosen=ids[(ids>=offset)&(ids<offset+len(f))]-offset;offset+=len(f)
            tri=v[f[chosen]];uv=rng.random((len(tri),2));u=np.sqrt(uv[:,0]);b=np.c_[1-u,u*(1-uv[:,1]),u*uv[:,1]]
            pp=(tri*b[:,:,None]).sum(1)
            if box is not None:pp=pp[((pp>=box[0])&(pp<=box[1])).all(1)]
            batch.append(pp)
        x=np.concatenate(batch)
        if len(x)<10:raise ValueError('Too little surface in declared GT crop')
        chunks.append(x)
    return np.concatenate(chunks)[:count].astype('float32')

def reference_cloud(scene,selection,cs):
    """The supplied dataset contains sparse SfM, not an independent laser scan.

    Label a point using its actual COLMAP track observations, rather than projecting
    invisible points through masks. At least two object-present observations are
    required. Background excludes ambiguous votes and target foreground.
    """
    file=REPO/'submodules/Inpaint360GS/tools/read_write_model.py'
    spec=importlib.util.spec_from_file_location('mesh_ref_colmap',file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    sparse=Path(selection['source'])/'sparse/0'
    intr,images,points=mod.read_model(str(sparse),ext='.bin')
    counts={};masks={};sizes={};valid_images={}
    for ident,im in images.items():
        if Path(im.name).name.lower().startswith('test'):continue
        p=Path(selection['mask_root'])/(Path(im.name).stem+'.png')
        labels=np.asarray(Image.open(p));masks[ident]=np.isin(labels,selection['target_ids'])
        c=intr[im.camera_id];sizes[ident]=(c.width,c.height);valid_images[ident]=im
    coordinates=[];background=[];support=[];fractions=[]
    for point in points.values():
        votes=[]
        for imid,idx in zip(point.image_ids,point.point2D_idxs):
            imid=int(imid)
            if imid not in valid_images:continue
            im=valid_images[imid];mask=masks[imid];ow,oh=sizes[imid]
            u,v=im.xys[int(idx)];u=int(np.floor(u*mask.shape[1]/ow));v=int(np.floor(v*mask.shape[0]/oh))
            if 0<=u<mask.shape[1] and 0<=v<mask.shape[0]:votes.append(bool(mask[v,u]))
        if len(votes)<2 or not np.isfinite(point.xyz).all():continue
        fraction=float(np.mean(votes));coordinates.append(point.xyz);background.append(fraction<=.1)
        support.append(len(votes));fractions.append(fraction)
    xyz=np.asarray(coordinates,dtype='float32');bg=np.asarray(background,dtype=bool)
    if not len(xyz) or not bg.any():raise ValueError('Insufficient reference points')
    # Shared ROI from reference ONLY. No method-specific alignment, scaling or crop.
    box=np.quantile(xyz,[.005,.995],axis=0);margin=(box[1]-box[0])*.02;box+=np.stack([-margin,margin])
    inside=((xyz>=box[0])&(xyz<=box[1])).all(1)
    return xyz[inside],xyz[inside&bg],box,dict(
        kind='COLMAP sparse SfM reference, NOT independent dense scan ground truth',
        source=identity(sparse/'points3D.bin'),raw_points=len(points),two_view_points=len(xyz),
        original_state_points=int(inside.sum()),background_points=int((inside&bg).sum()),
        foreground_or_ambiguous_excluded=int((inside&~bg).sum()),
        background_rule='at least 2 non-test* track observations; target-mask vote fraction <= 0.1',
        roi=box.tolist(),units='COLMAP reconstruction units; no verified metric scale',
        caveat='Known observed surfaces only. Does not provide truth for previously occluded inpainted surfaces.')
