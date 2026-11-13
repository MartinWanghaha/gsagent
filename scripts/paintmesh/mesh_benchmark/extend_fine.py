"""Extend an actual fine mesh without changing any original triangle/color.

Whole-scene partitions use the same fine lattice. Remove only redundant new
triangles whose three corners AND center lie within the TSDF uncertainty band
of an interior original surface. A nearest boundary edge does not establish
coverage: retain those faces so an original crop boundary cannot open a seam.
No ground-truth, mask, camera score, plane fitting or mesh decimation is used.
"""
from pathlib import Path
import gc
import fcntl
import numpy as np
from .core import MeshScene,read,write,identity,digest
from recompose.io import fingerprint


def prepare_distance_bound(mesh,original):
    """Any covered point must be within band + longest-edge of a source vertex.

    This is a necessary geometric bound, not a sampled approximation. It lets
    the exact closest-triangle query skip distant free space inside a loose
    source AABB. Every source face participates in the longest-edge bound.
    """
    from recompose.io import ply,xyz
    from scipy.spatial import cKDTree
    data=ply(original);vertices=xyz(data['vertex'].data).astype('float32').astype('float64')
    faces=data['face'].data['vertex_indices'];longest=0.
    for start in range(0,len(faces),100000):
        q=vertices[faces[start:start+100000]]
        longest=max(longest,float(np.linalg.norm(q-np.roll(q,1,axis=1),axis=2).max()))
    mesh.coverage_vertex_tree=cKDTree(vertices)
    mesh.coverage_longest_edge=longest
    print('FINE_EXTENSION_EXACT_BOUND',len(vertices),len(faces),longest,flush=True)


def covered(mesh,points,band=.008):
    points=np.asarray(points,dtype='float32')
    result=np.zeros(len(points),dtype=bool)
    lo,hi=np.asarray(mesh.counts[0]['bounds'])
    eligible=np.where(((points>=lo-band)&(points<=hi+band)).all(1))[0]
    if len(eligible) and hasattr(mesh,'coverage_vertex_tree'):
        margin=1e-5*(1+float(np.max(np.abs(points[eligible]))))
        d,_=mesh.coverage_vertex_tree.query(points[eligible],k=1,
            distance_upper_bound=band+mesh.coverage_longest_edge+margin,workers=4)
        eligible=eligible[np.isfinite(d)]
    for start in range(0,len(eligible),65536):
        ids=eligible[start:start+65536]
        closest=mesh.scene.compute_closest_points(mesh.o3d.core.Tensor.from_numpy(np.ascontiguousarray(points[ids])),nthreads=4)
        p=closest['points'].numpy();uv=closest['primitive_uvs'].numpy()
        # Conservative interior test protects actual open/cropped boundaries.
        interior=(uv[:,0]>1e-5)&(uv[:,1]>1e-5)&(uv.sum(1)<1-1e-5)
        distance=np.linalg.norm(points[ids]-p,axis=1)
        exactly_on_surface=distance<=1e-7*(1+np.abs(points[ids]).max(1))
        result[ids]=(distance<=band)&(interior|exactly_on_surface)
    return result


def extend(original,full_mesh,folder,band=.008):
    original=Path(original);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    lock=open(folder/'extension.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    request=dict(original=identity(original),full_request=full_mesh['request'],
        full_meshes=[identity(s['path']) for s in full_mesh['mesh_specs']],
        voxel=.002,sdf_trunc=.008,overlap_band=band,implementation=identity(Path(__file__)),
        rule='Only new triangles with all corners and center covered by interior source surfaces are redundant')
    marker=folder/'complete.json'
    if marker.exists():
        old=read(marker)
        if old['request']!=request:raise ValueError('Fine extension inputs changed')
        return old
    source=MeshScene([dict(path=str(original))]);specs=[dict(path=str(original))];rows=[]
    prepare_distance_bound(source,original)
    original_faces=source.counts[0]['faces'];removed_total=0;added_faces=0
    for i,spec in enumerate(full_mesh['mesh_specs']):
        path=Path(spec['path']);meta=read(path)
        vertices=np.load(meta['vertices']['path'],mmap_mode='r');faces=np.load(meta['indices']['path'],mmap_mode='r')
        kept=np.empty(len(faces),dtype=bool)
        for start in range(0,len(faces),100000):
            ids=faces[start:start+100000]
            triangles=np.asarray(vertices['position'][ids],dtype='float32')
            points=np.concatenate([triangles.reshape(-1,3),triangles.mean(1)])
            flags=covered(source,points,band)
            duplicate=flags[:len(ids)*3].reshape(-1,3).all(1)&flags[len(ids)*3:]
            kept[start:start+len(ids)]=~duplicate
        removed=int((~kept).sum());count=int(kept.sum());removed_total+=removed;added_faces+=count
        if count==len(faces):specs.append(spec)
        elif count:
            # Reuse every vertex/color byte; write only the nonoverlapping
            # original triangle indices. No vertex motion, fitting or reduction.
            fp=folder/f'{i:05d}.indices.npy';np.save(fp,np.asarray(faces[kept],dtype='<u4'))
            output=folder/f'{i:05d}.mesh.json';updated=dict(meta)
            updated.update(indices=fingerprint(fp),index_identity=identity(fp),face_count=count,
                source_mesh=identity(path),overlap_faces_removed=removed,original_triangle_records_retained_exact=True)
            write(output,updated);specs.append(dict(path=str(output)))
        rows.append(dict(source=identity(path),faces=len(faces),added_faces=count,overlap_faces_removed=removed))
        if i%20==0:print('FINE_EXTENSION',str(original),i+1,len(full_mesh['mesh_specs']),'added',added_faces,'duplicate',removed_total,flush=True)
    assert identity(original)==request['original'],'Original fine mesh changed during extension'
    result=dict(complete=True,request=request,mesh_specs=specs,original_mesh=request['original'],
        original_mesh_unchanged=True,original_faces_preserved=original_faces,added_faces=added_faces,
        redundant_new_faces_removed=removed_total,faces=original_faces+added_faces,
        geometry_simplified=False,connected_components_removed=False,parts=rows,
        no_coarse_background=True,no_reference_data_used=True)
    write(marker,result);del source;gc.collect();return result
