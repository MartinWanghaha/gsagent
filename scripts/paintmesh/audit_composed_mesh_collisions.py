#!/usr/bin/env python3
"""Audit the published composed meshes, without moving or simplifying them.

Every environment partition participates in a conservative broad phase. Only
triangles whose AABBs can touch an object neighborhood enter the narrow phase;
crossing triangles are retained even when all three vertices lie outside it.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mesh_benchmark.core import BASE, SCENES, read, write, identity, digest
from recompose.io import ply, xyz
from recompose.geometry import transform, check_similarity

PUBLIC = BASE / 'mesh_comparison_whole'
DEFAULT = PUBLIC / 'collision_audit'


def overlaps(bounds, lower, upper):
    box = np.asarray(bounds)
    return bool(np.all(box[1] >= lower) and np.all(box[0] <= upper))


def triangle_aabb_mask(triangles, boxes):
    """Conservative test; vertex-only cropping would miss crossing faces."""
    low, high = triangles.min(1), triangles.max(1)
    keep = np.zeros(len(triangles), dtype=bool)
    for lower, upper in boxes:
        keep |= np.all(high >= lower, axis=1) & np.all(low <= upper, axis=1)
    return keep


def source_records(graph):
    specs = [dict(path=graph['background']['mesh']['path'])]
    specs += [dict(path=item['path']) for item in graph['background'].get('patches', [])]
    records = []
    for i, spec in enumerate(specs):
        path = Path(spec['path'])
        row = dict(index=i, source=identity(path), kind='original_near_ply' if i == 0 else 'whole_scene_extension')
        if path.name.endswith('.mesh.json'):
            meta = read(path)
            if meta['format'] != 'paintmesh-native-triangles-v1':
                raise ValueError('Unsupported native triangle format')
            for key, stamp in [('vertices', 'vertex_identity'), ('indices', 'index_identity')]:
                actual = identity(meta[key]['path'])
                if actual != meta[stamp]:
                    raise ValueError('Native mesh buffer changed: '+actual['path'])
                row[key] = actual
            row.update(bounds=meta['bounds'], faces=meta['face_count'], vertices_count=meta['vertex_count'])
        records.append(row)
    return records


def geometry(path):
    path = Path(path)
    if path.name.endswith('.mesh.json'):
        meta = read(path)
        vertices = np.load(meta['vertices']['path'], mmap_mode='r')
        return vertices['position'], np.load(meta['indices']['path'], mmap_mode='r'), vertices['color'][:, :3]
    data = ply(path)
    vertices = data['vertex'].data
    # Preserve source precision. A whole near PLY fits in host memory, while
    # per-face temporaries remain bounded by the streaming chunk size below.
    v = xyz(vertices)
    f = data['face'].data['vertex_indices']
    if f.dtype == object:
        f = np.stack(f)
    c = np.column_stack([vertices[k] for k in ['red', 'green', 'blue']])
    return v, f, c


def prepare(output, scene):
    folder = Path(output) / 'scenes' / scene
    folder.mkdir(parents=True, exist_ok=True)
    graph_path = PUBLIC / 'recomposition' / scene / 'composition/scene.json'
    graph = read(graph_path)
    records = source_records(graph)
    object_records = []
    boxes = []
    for obj in graph['objects']:
        mesh_path = Path(obj['mesh']['path'])
        with np.load(mesh_path) as a:
            vertices = np.asarray(a['vertices'], dtype='float64')
            faces = np.asarray(a['faces'], dtype='int32')
            colors = np.asarray(a['colors'], dtype='float32')
        pose = np.asarray(obj['object_to_world'], dtype='float64')
        check_similarity(pose)
        vertices = transform(vertices, pose)
        diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        if not diagonal > 0:
            raise ValueError('Invalid object size')
        # The extra neighborhood is for contact diagnostics, not for changing
        # collision tolerances or discarding triangles near the object.
        margin = .05 * diagonal
        bounds = [vertices.min(0), vertices.max(0)]
        boxes.append((bounds[0]-margin, bounds[1]+margin))
        destination = folder / 'objects' / str(obj['id'])
        destination.mkdir(parents=True, exist_ok=True)
        object_records.append(dict(id=str(obj['id']), source=identity(mesh_path),
            pose=pose.tolist(), bounds=np.asarray(bounds).tolist(), diameter=diagonal,
            vertices=len(vertices), faces=len(faces), evidence=obj['evidence'],
            pose_report=identity(obj['pose_report']['path']),
            mesh_world=str(destination / 'mesh_world.npz')))
        target = destination / 'mesh_world.npz'
        if not target.exists():
            np.savez(target, vertices=vertices, faces=faces, colors=colors)
    request = dict(scene_graph=identity(graph_path), environment_sources=records,
        objects=object_records, neighborhood_fraction=.05,
        implementation=identity(Path(__file__)), geometry_simplified=False)
    marker = folder / 'candidates.json'
    if marker.exists():
        old = read(marker)
        if old.get('complete') and old['request'] == request and old['roi'] == identity(folder/'background_roi.npz'):
            return old
        raise ValueError('Existing collision ROI inputs changed; refuse to reuse stale audit')
    vertices_out, colors_out, faces_out, face_ids, spec_ids = [], [], [], [], []
    vertex_offset = 0
    scanned_faces = excluded_parts = loaded_parts = 0
    started = time.time()
    for row in records:
        if 'bounds' in row and not any(overlaps(row['bounds'], lo, hi) for lo, hi in boxes):
            excluded_parts += 1
            continue
        v, f, c = geometry(row['source']['path'])
        loaded_parts += 1
        kept = []
        for start in range(0, len(f), 200000):
            ids = np.asarray(f[start:start+200000], dtype='int64')
            triangles = np.asarray(v[ids], dtype='float64')
            keep = triangle_aabb_mask(triangles, boxes)
            if keep.any():
                kept.append(np.flatnonzero(keep)+start)
            scanned_faces += len(ids)
        if kept:
            original_ids = np.concatenate(kept)
            selected = np.asarray(f[original_ids], dtype='int64')
            unique, inverse = np.unique(selected, return_inverse=True)
            vertices_out.append(np.asarray(v[unique], dtype='float64'))
            colors_out.append(np.asarray(c[unique], dtype='uint8'))
            faces_out.append(inverse.reshape(-1, 3).astype('int64')+vertex_offset)
            face_ids.append(original_ids)
            spec_ids.append(np.full(len(original_ids), row['index'], dtype='int32'))
            vertex_offset += len(unique)
        del v, f, c
        gc.collect()
        print('COLLISION_ROI', scene, row['index']+1, len(records),
            'kept_faces', sum(len(a) for a in face_ids), 'elapsed', round(time.time()-started, 2), flush=True)
    vertices = np.concatenate(vertices_out) if vertices_out else np.empty((0, 3))
    faces = np.concatenate(faces_out).astype('int32') if faces_out else np.empty((0, 3), dtype='int32')
    np.savez(folder/'background_roi.npz', vertices=vertices, faces=faces,
        colors=np.concatenate(colors_out) if colors_out else np.empty((0, 3), dtype='uint8'),
        source_spec_indices=np.concatenate(spec_ids) if spec_ids else np.empty(0, dtype='int32'),
        source_face_ids=np.concatenate(face_ids) if face_ids else np.empty(0, dtype='int64'))
    if identity(graph_path) != request['scene_graph']:
        raise ValueError('Published scene graph changed during collision audit')
    result = dict(complete=True, request=request, scene=scene, objects=object_records,
        publication_complete=graph.get('complete'), publication_status=graph.get('status'),
        source_partitions=len(records), excluded_by_conservative_aabb=excluded_parts,
        loaded_partitions=loaded_parts, scanned_faces=scanned_faces,
        total_environment_faces=sum(r.get('faces', 0) for r in records[1:])+len(ply(records[0]['source']['path'])['face'].data),
        retained_original_triangles=len(faces), retained_vertices=len(vertices),
        roi=identity(folder/'background_roi.npz'), elapsed=time.time()-started,
        all_source_partitions_considered=True, decimation=False, vertex_only_crop=False)
    write(marker, result)
    return result


def array_diagnostics(vertices, faces, env_vertices, env_faces, obj, folder):
    """Volume depth only when the object's closed geometry was verified."""
    import open3d as o3d
    from recompose.collision import is_watertight
    closed = is_watertight(vertices, faces)
    tolerance = .003 * obj['diameter']
    result = dict(applicable=closed, closed_object=closed,
        tolerance_scene_units=tolerance, normalized_by_object_diagonal=obj['diameter'],
        method='All retained environment vertices queried against verified closed object; parity signed distance, nsamples=3',
        vertex_measure_not_exact_global_penetration_depth=True)
    if not closed or not len(env_vertices):
        result['reason'] = 'Object is not a closed non-self-intersecting manifold' if not closed else 'No environment near object'
        return result
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype='float32')),
        o3d.core.Tensor(np.asarray(faces, dtype='uint32')))
    depths = []
    for start in range(0, len(env_vertices), 100000):
        signed = scene.compute_signed_distance(o3d.core.Tensor(np.asarray(env_vertices[start:start+100000], dtype='float32')),
            nthreads=4, nsamples=3).numpy()
        depths.append(signed)
    signed = np.concatenate(depths)
    inside = signed < -1e-7*obj['diameter']
    deep = signed < -tolerance
    positive = -signed[inside]
    result.update(environment_vertices_tested=len(signed), environment_vertices_inside_closed_object=int(inside.sum()),
        significant_inside_vertices=int(deep.sum()), maximum_depth=float(positive.max()) if len(positive) else 0.,
        p95_depth=float(np.quantile(positive, .95)) if len(positive) else 0.,
        maximum_depth_fraction=float(positive.max()/obj['diameter']) if len(positive) else 0.)
    np.savez(folder/'inside_environment_vertices.npz', indices=np.flatnonzero(inside),
        signed_distance=signed[inside], significant=np.asarray(deep[inside]))
    return result


def audit(output, scene):
    folder = Path(output)/'scenes'/scene
    prepared = prepare(output, scene)
    from mesh_benchmark.mesh_intersection import audit_intersections
    implementation = identity(HERE/'mesh_benchmark/mesh_intersection.py')
    request = dict(candidates=identity(folder/'candidates.json'), narrow_phase=implementation)
    marker = folder/'audit.json'
    if marker.exists() and read(marker).get('complete') and read(marker)['request'] == request:
        return read(marker)
    with np.load(folder/'background_roi.npz') as data:
        env_vertices, env_faces = data['vertices'], data['faces']
        source_spec_indices = data['source_spec_indices']
    rows = []
    object_geometry = {}
    for obj in prepared['objects']:
        dest = folder/'objects'/obj['id']
        with np.load(obj['mesh_world']) as data:
            vertices, faces = data['vertices'], data['faces']
        object_geometry[obj['id']] = (vertices, faces)
        row_request = dict(scene=request, object=obj)
        old = dest/'audit.json'
        if old.exists() and read(old).get('complete') and read(old)['request'] == row_request:
            rows.append(read(old)); continue
        started = time.time()
        # Broad-phase per object is only acceleration: keep every triangle
        # whose bounds overlap the object's AABB, including spanning faces.
        bounds = np.asarray(obj['bounds']); tol=1e-7*obj['diameter']
        selected=[]
        for start in range(0,len(env_faces),200000):
            tri=env_vertices[env_faces[start:start+200000]]
            keep=triangle_aabb_mask(tri,[(bounds[0]-tol,bounds[1]+tol)])
            if keep.any(): selected.append(np.flatnonzero(keep)+start)
        selected=np.concatenate(selected) if selected else np.empty(0,dtype='int64')
        print('COLLISION_EXACT_BEGIN',scene,obj['id'],'object_faces',len(faces),'environment_faces',len(selected),flush=True)
        result=audit_intersections(vertices,faces,env_vertices,env_faces[selected],tolerance=tol)
        pairs=np.asarray(result['face_pairs'],dtype='int64')
        if len(pairs): pairs[:,1]=selected[pairs[:,1]]
        np.savez(dest/'intersections.npz',face_pairs=pairs,
            classification=np.asarray(result['classification']),segments=np.asarray(result['segments']),
            positions=np.asarray(result['positions']))
        penetration=array_diagnostics(vertices,faces,env_vertices,env_faces,obj,dest)
        counts=result['counts']; transverse=int(counts.get('transverse',0))
        status='transverse_intersection' if transverse else ('boundary_or_coplanar_contact' if len(pairs) else 'no_surface_intersection')
        if penetration.get('significant_inside_vertices',0)>0 and transverse:
            status='confirmed_penetration_beyond_contact_tolerance'
        sources={}
        if len(pairs):
            unique,n=np.unique(source_spec_indices[pairs[:,1]],return_counts=True)
            sources={str(int(k)):int(v) for k,v in zip(unique,n)}
        row=dict(complete=True,request=row_request,id=obj['id'],vertices=len(vertices),faces=len(faces),diameter=obj['diameter'],
            intersection_counts=counts,collision_status=status,closed_object=penetration['closed_object'],
            penetration=penetration,intersected_source_partitions=sources,
            original_pose_report=read(obj['pose_report']['path']).get('status'),
            all_triangle_pairs_enumerated=result.get('complete',False),elapsed=time.time()-started,
            numerical_tolerance_scene_units=tol,contact_tolerance_separate_from_intersection_test=True)
        write(old,row);rows.append(row)
        print('COLLISION_OBJECT_COMPLETE',scene,obj['id'],status,counts,penetration,flush=True)
    object_pairs=[]
    ids=list(object_geometry)
    for i,key in enumerate(ids):
        for other in ids[i+1:]:
            va,fa=object_geometry[key];vb,fb=object_geometry[other]
            tol=1e-7*max(np.linalg.norm(np.ptp(va,axis=0)),np.linalg.norm(np.ptp(vb,axis=0)))
            if not overlaps([va.min(0),va.max(0)],vb.min(0)-tol,vb.max(0)+tol):
                object_pairs.append(dict(a=key,b=other,status='disjoint_aabb',complete=True));continue
            result=audit_intersections(va,fa,vb,fb,tolerance=tol)
            object_pairs.append(dict(a=key,b=other,intersection_counts=result['counts'],complete=result['complete']))
    result=dict(complete=True,scene=scene,request=request,updated=time.time(),objects=rows,object_object_pairs=object_pairs,
        publication_complete=prepared['publication_complete'],publication_status=prepared['publication_status'],
        total_environment_faces=prepared['total_environment_faces'],retained_candidate_faces=prepared['retained_original_triangles'],
        all_source_partitions_considered=True,mesh_simplified=False,pose_changed=False,
        caveats=['Open environment is not treated as a closed solid.',
                 'Transverse intersections are exact face-crossing evidence, not a global minimum translation distance.',
                 'Closed-object volume depths are evaluated on all retained environment vertices, not continuous volume maxima.',
                 'A no-surface-crossing result alone does not exclude containment or wrong-side placement of open surfaces.'])
    write(marker,result)
    return result


def run(args):
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor,as_completed
    function=prepare if args.prepare_only else audit
    write(output/'status.json',dict(complete=False,stage='collision_audit',scenes=args.scenes,started=time.time(),pid=os.getpid()))
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(function,output,s):s for s in args.scenes}
        for job in as_completed(jobs):
            scene=jobs[job]
            try:results.append(job.result())
            except Exception as error:
                failures.append(dict(scene=scene,error=repr(error)))
                print('COLLISION_FAILED',scene,repr(error),flush=True)
            write(output/'status.json',dict(complete=False,stage='preparing' if args.prepare_only else 'collision_audit',
                scenes=args.scenes,completed=[r['scene'] for r in results],failures=failures,updated=time.time(),pid=os.getpid()))
    if failures:raise RuntimeError(f'Collision audit failures: {failures}')
    if not args.prepare_only:
        subprocess.run([sys.executable,str(HERE/'visualize_composed_collisions.py'),'--output',str(output)],check=True)
    write(output/'status.json',dict(complete=True,stage='candidates_ready' if args.prepare_only else 'complete',
        scenes=args.scenes,completed=[r['scene'] for r in results],updated=time.time(),pid=os.getpid(),poses_changed=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--scenes',nargs='+',choices=SCENES,default=SCENES)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if not 1<=args.workers<=3:p.error('workers must be between 1 and 3')
    run(args)


if __name__=='__main__':main()
