#!/usr/bin/env python3
"""Remove unsupported long interpolation faces from trained CoMe/GOF meshes.

No reference cloud, held-out RGB, PSNR or F1 enters the repair. Coordinates,
colors, supported triangles and original files are preserved. The output is
explicitly described as native extraction plus Gaussian-support validation.
"""
import argparse
from pathlib import Path
import shutil
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mesh_benchmark.core import BASE,SCENES,read,write,digest
from mesh_benchmark.color_export import read_mesh,positions
from mesh_benchmark.gaussian_support import GaussianSupport
from recompose.io import fingerprint

POLICY=dict(version=1,native_opacity_cutoff=1/255,
    long_edge_work_threshold_median_sigma_factor=8,
    probes=['centroid','edge01_midpoint','edge12_midpoint','edge20_midpoint'],
    rejection='centroid unsupported AND at least 3 of 4 probes unsupported',
    broad_phase='radius-bin cKDTrees enumerate every potentially containing ellipsoid',
    reference_data_used=False,vertex_positions_changed=False,colors_changed=False)
ROOT=BASE/'mesh_comparison_whole'
WORK=ROOT/'_internal/workspace/native_support_repair'


def filter_faces(source,destination,keep):
    """Copy native vertex bytes and retained native face records exactly."""
    source=Path(source);destination=Path(destination);mesh=read_mesh(source)
    if mesh.text or [e.name for e in mesh.elements]!=['vertex','face']:
        raise ValueError('Expected binary vertex/face PLY for lossless native face selection')
    vertices=mesh['vertex'].data;faces=mesh['face'].data
    if keep.shape!=(len(faces),) or keep.dtype!=bool:raise ValueError('Bad face mask')
    with source.open('rb') as f:
        header=[]
        while True:
            line=f.readline()
            if not line:raise ValueError('No PLY end header')
            if line.startswith(b'element face '):line=f'element face {int(keep.sum())}\n'.encode()
            header.append(line)
            if line.strip()==b'end_header':break
        offset=f.tell()
        destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix('.tmp.ply')
        with temp.open('wb') as out:
            out.writelines(header);remaining=len(vertices)*vertices.dtype.itemsize
            while remaining:
                block=f.read(min(8<<20,remaining))
                if not block:raise ValueError('Truncated vertex buffer')
                out.write(block);remaining-=len(block)
            # plyfile hides each list-length byte in structured dtype padding;
            # a NumPy structured fancy-copy need not preserve that padding.
            # Select whole raw byte records so list lengths/extra attributes
            # remain bit-for-bit native as well as the integer triangle IDs.
            raw=np.memmap(source,mode='r',dtype='uint8',offset=offset+len(vertices)*vertices.dtype.itemsize,
                          shape=(len(faces),faces.dtype.itemsize))
            for begin in range(0,len(faces),262144):
                out.write(raw[begin:begin+262144][keep[begin:begin+262144]].tobytes())
        temp.replace(destination)
    actual=read_mesh(destination)
    for begin in range(0,len(vertices),262144):
        np.testing.assert_array_equal(actual['vertex'].data[begin:begin+262144],vertices[begin:begin+262144])
    cursor=0
    for begin in range(0,len(faces),262144):
        wanted=faces[begin:begin+262144][keep[begin:begin+262144]]
        np.testing.assert_array_equal(actual['face'].data[cursor:cursor+len(wanted)],wanted);cursor+=len(wanted)
    assert cursor==int(keep.sum())


def repair(method,scene,publish=False):
    out=BASE/method/scene;marker=read(out/'mesh_complete.json')
    source=Path(marker.get('native_source_mesh',marker['path']))
    checkpoint=out/'model/point_cloud/iteration_30000/point_cloud.ply'
    request=dict(method=method,scene=scene,source=fingerprint(source),checkpoint=fingerprint(checkpoint),policy=POLICY)
    cache=WORK/method/scene;cache.mkdir(parents=True,exist_ok=True)
    receipt=cache/'complete.json';signature=digest(request)
    destination=source.with_name(source.stem.replace('_support_validated','')+'_support_validated.ply')
    if receipt.exists() and read(receipt)['signature']==signature:
        result=read(receipt)
        if publish:publish_result(result)
        return result
    began=time.time();support=GaussianSupport.from_ply(checkpoint,cutoff=POLICY['native_opacity_cutoff'])
    mesh=read_mesh(source);vertices=mesh['vertex'].data;faces=mesh['face'].data['vertex_indices']
    # This gate limits work to coarse interpolation. The rejection itself is
    # exact source-model support, not edge length or nearest-neighbor distance.
    threshold=POLICY['long_edge_work_threshold_median_sigma_factor']*support.median_sigma
    keep=np.ones(len(faces),dtype=bool);candidates=0;removed=0;records=[]
    progress=cache/'progress.json';saved=read(progress) if progress.exists() else None
    start=saved['completed_faces'] if saved and saved['signature']==signature else 0
    if start:
        keep=np.load(cache/'keep_faces.npy');candidates=saved['candidates'];removed=saved['removed']
    for begin in range(start,len(faces),131072):
        end=min(begin+131072,len(faces));indices=faces[begin:end]
        tri=np.stack([positions(vertices[indices[:,i]]) for i in range(3)],axis=1)
        edge=np.maximum.reduce([np.linalg.norm(tri[:,0]-tri[:,1],axis=1),np.linalg.norm(tri[:,1]-tri[:,2],axis=1),np.linalg.norm(tri[:,2]-tri[:,0],axis=1)])
        selected=np.flatnonzero(edge>threshold);candidates+=len(selected)
        if len(selected):
            t=tri[selected];probes=np.stack([t.mean(1),(t[:,0]+t[:,1])/2,(t[:,1]+t[:,2])/2,(t[:,2]+t[:,0])/2],axis=1)
            covered=support.contains(probes.reshape(-1,3)).reshape(-1,4)
            invalid=(~covered[:,0])&((~covered).sum(1)>=3)
            ids=begin+selected[invalid];keep[ids]=False;removed+=len(ids)
            if len(ids):records.append(dict(begin=begin,removed=len(ids),max_removed_edge=float(edge[selected[invalid]].max())))
        if begin% (131072*8)==0 or end==len(faces):
            checkpoint_temp=cache/'keep_faces.tmp.npy'
            np.save(checkpoint_temp,keep);checkpoint_temp.replace(cache/'keep_faces.npy')
            removed=int((~keep).sum())
            write(progress,dict(signature=signature,completed_faces=end,total_faces=len(faces),candidates=candidates,removed=removed,edge_work_threshold=threshold))
            print('NATIVE_SUPPORT_REPAIR',method,scene,end,'/',len(faces),'candidates',candidates,'removed',removed,'seconds',time.time()-began,flush=True)
    removed=int((~keep).sum())
    if removed:
        filter_faces(source,destination,keep)
    else:destination=source
    result=dict(complete=True,signature=signature,request=request,path=str(destination),native_source_mesh=str(source),
        original_faces=len(faces),faces=int(keep.sum()),vertices=len(vertices),removed_faces=removed,
        removed_face_fraction=float(removed/max(len(faces),1)),tested_faces=candidates,
        edge_work_threshold=threshold,coordinates_and_colors_exact=True,retained_face_records_exact=True,
        training_performed=False,reference_data_used=False,geometry_postprocessed=bool(removed),
        processing_label='Native CoMe/GOF extraction + Gaussian-support validation',
        candidate_mesh=fingerprint(destination),elapsed_seconds=time.time()-began,records=records)
    write(receipt,result)
    if publish:publish_result(result)
    return result


def atomic_link(source,destination):
    import os
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_name('.'+destination.name+'.repair')
    if temp.is_symlink() or temp.exists():temp.unlink()
    temp.symlink_to(os.path.relpath(Path(source).resolve(strict=True),destination.parent.resolve()))
    os.replace(temp,destination)


def publish_result(result):
    method=result['request']['method'];scene=result['request']['scene'];out=BASE/method/scene
    marker=read(out/'mesh_complete.json');path=Path(result['path'])
    if marker.get('support_validation_signature')==result['signature']:return
    geometry_path=str(path) if result['geometry_postprocessed'] else marker.get('geometry_path',str(path))
    geometry_sha=result['candidate_mesh']['sha256'] if result['geometry_postprocessed'] else marker.get('geometry_sha256',marker['sha256'])
    marker.update(path=str(path),sha256=result['candidate_mesh']['sha256'],geometry_path=geometry_path,
        geometry_sha256=geometry_sha,colored_path=str(path),has_vertex_colors=True,
        native_source_mesh=result['native_source_mesh'],vertices=result['vertices'],faces=result['faces'],
        support_validation_signature=result['signature'],geometry_postprocessed=result['geometry_postprocessed'],
        processing_label=result['processing_label'],support_validation_receipt=str(WORK/method/scene/'complete.json'))
    write(out/'mesh_complete.json',marker)
    for name in ['mesh.ply','mesh_colored.ply']:atomic_link(path,out/name)
    if method=='come':atomic_link(path,ROOT/'meshes'/scene/'come_colored.ply')
    color=out/'mesh_color_complete.json';previous=read(color) if color.exists() else {}
    previous.update(complete=True,method=method,scene=scene,source_geometry=fingerprint(path),colored_mesh=fingerprint(path),
        geometry_unchanged=True,training_performed=False,geometry_postprocessed_before_color_export=result['geometry_postprocessed'],
        raw_native_mesh=fingerprint(result['native_source_mesh']),processing_label=result['processing_label'])
    write(color,previous)
    if (out/'result.json').exists():
        item=read(out/'result.json');item.update(mesh=str(path),geometry_mesh=str(path),colored_mesh=str(path),support_validation=result);write(out/'result.json',item)
    print('PUBLISHED_NATIVE_SUPPORT_REPAIR',method,scene,'removed',result['removed_faces'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['come','gof'],required=True);p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--publish',action='store_true');a=p.parse_args();result=repair(a.method,a.scene,a.publish)
    print('NATIVE_SUPPORT_REPAIR_COMPLETE',a.method,a.scene,{k:result[k] for k in ['vertices','original_faces','faces','removed_face_fraction','elapsed_seconds']},flush=True)
