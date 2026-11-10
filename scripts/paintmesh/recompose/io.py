"""Strict provenance, atomic outputs and bounded-memory PLY operations."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from plyfile import PlyData, PlyElement


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=encode)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name)
    try:
        with os.fdopen(fd, 'w') as f: f.write(text+'\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8<<20), b''): h.update(block)
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=h.hexdigest())


def complete(path):
    v = read_json(path)
    if v.get('status') != 'complete' or not v.get('complete'):
        raise ValueError(f'Incomplete upstream artifact: {path}')
    return v


def record_path(record, parent=None):
    name = record.get('path', record.get('file'))
    if name is None: raise ValueError('Artifact must declare a path or file')
    p = Path(name)
    return (p if p.is_absolute() else Path(parent)/p).resolve(strict=True)


def ply(path):
    return PlyData.read(str(path), mmap='r', known_list_len={'face': {'vertex_indices': 3}})


def xyz(v):
    return np.column_stack([v[n] for n in ('x','y','z')])


def save_mesh(path, vertices, faces, colors=None):
    v = np.empty(len(vertices), dtype=[(n,'f4') for n in ('x','y','z')] +
                 ([(n,'u1') for n in ('red','green','blue')] if colors is not None else []))
    for k,n in enumerate(('x','y','z')): v[n] = vertices[:,k]
    if colors is not None:
        for k,n in enumerate(('red','green','blue')): v[n] = np.clip(colors[:,k]*255,0,255).astype('u1')
    f = np.empty(len(faces), dtype=[('vertex_indices','i4',(3,))]); f['vertex_indices'] = faces
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    PlyData([PlyElement.describe(v,'vertex'),PlyElement.describe(f,'face')],text=False).write(str(path))


def crop_mesh(path, lower, upper, max_faces=150000, label_path=None, target_id=None):
    """Stream original triangles; include boundary-crossing faces, then QEM simplify.

    No stochastic dropping of triangles (which would invalidate ray visibility).
    ROI is never normalized or translated. Returns original world coordinates.
    """
    import open3d as o3d
    data=ply(path); vertices=data['vertex'].data; faces=data['face'].data
    labels=np.load(label_path,mmap_mode='r') if label_path else None
    if labels is not None and len(labels)!=len(faces): raise ValueError('Face label/topology mismatch')
    chunks=[]
    for start in range(0,len(faces),200000):
        ids=faces['vertex_indices'][start:start+200000]
        if ids.dtype==object: ids=np.stack(ids)
        if labels is not None: ids=ids[labels[start:start+len(ids)]==target_id]
        if len(ids)==0: continue
        tri=xyz(vertices[ids.reshape(-1)]).reshape(-1,3,3)
        keep=(tri.max(1)>=lower).all(1)&(tri.min(1)<=upper).all(1)
        if keep.any(): chunks.append(np.asarray(ids[keep],dtype=np.int64))
    if not chunks: raise ValueError(f'No triangles in requested ROI: {path}')
    faces=np.concatenate(chunks); unique,inverse=np.unique(faces,return_inverse=True)
    v=xyz(vertices[unique]).astype(float); f=inverse.reshape(-1,3)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f))
    if all(n in vertices.dtype.names for n in ('red','green','blue')):
        mesh.vertex_colors=o3d.utility.Vector3dVector(np.column_stack([vertices[n][unique] for n in ('red','green','blue')])/255.)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
    if len(mesh.triangles)>max_faces*4:
        # Millions of TSDF micro-triangles make direct QEM require tens of GB.
        # Spatial clustering preserves topology connectivity instead of dropping faces.
        diagonal=float(np.linalg.norm(np.ptp(v,axis=0)))
        mesh=mesh.simplify_vertex_clustering(max(diagonal/800.,1e-8))
    if len(mesh.triangles)>max_faces: mesh=mesh.simplify_quadric_decimation(max_faces)
    mesh.compute_vertex_normals()
    c=np.asarray(mesh.vertex_colors)
    return np.asarray(mesh.vertices),np.asarray(mesh.triangles),c if len(c) else np.full((len(mesh.vertices),3),.5)


def crop_reconstruction(path,lower,upper,max_faces,gaussian_path):
    """Recover an empty post-filter ROI from the same chronological TSDF export.

    The actual source is recorded and still must pass multiview observation checks.
    Background recovery is exported as a local static layer, never as a silent
    replacement of the user's whole selected background mesh.
    """
    path=Path(path)
    try:return crop_mesh(path,lower,upper,max_faces),path,False
    except ValueError as error:
        if 'No triangles in requested ROI:' not in str(error):raise
        if path.name!='tsdf_fusion_post.ply':raise
        raw=path.with_name('tsdf_fusion.ply')
        if not raw.is_file():raise
        gaussian_path=Path(gaussian_path);model=gaussian_path
        # Semantic distillation saves a later PLY with the same frozen centers.
        # Verify row alignment before using the earlier meshed Gaussian export.
        if raw.stat().st_mtime_ns<model.stat().st_mtime_ns and path.parent.name.startswith('ours_'):
            iteration=path.parent.name.removeprefix('ours_')
            candidate=path.parents[2]/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'
            if candidate.is_file():
                a=ply(gaussian_path)['vertex'].data;b=ply(candidate)['vertex'].data
                if len(a)==len(b) and all(np.array_equal(a[k],b[k]) for k in ['x','y','z']):model=candidate
        if not model.stat().st_mtime_ns<=raw.stat().st_mtime_ns<=path.stat().st_mtime_ns:
            raise ValueError(f'Unfiltered TSDF chronology does not match its Gaussian export: {raw}') from error
        print(f'Recover local ROI from unfiltered TSDF: {raw}',flush=True)
        return crop_mesh(raw,lower,upper,max_faces),raw,True
