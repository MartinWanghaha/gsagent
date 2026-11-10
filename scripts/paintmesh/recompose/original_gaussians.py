"""Exact reuse of deleted original splats. No sampling, fitting, or training."""
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement
from scipy.spatial import cKDTree
from .io import read_json,write_json,record_path,ply,xyz
from .provenance import Fingerprinter
from .gaussian import merge_gaussians

fingerprint=Fingerprinter()


def sources(inputs):
    manifest=read_json(inputs['manifest'])
    removal_path=record_path(manifest['inputs']['removal_manifest'])
    removal=read_json(removal_path)
    workspace=read_json(record_path(removal['inputs']['workspace_manifest']))
    original=record_path(workspace['inputs']['semantic_gaussian_ply'])
    return dict(original=str(original),retained=str(record_path(removal['outputs']['removed_3dgs'])),
                labels=inputs['gaussian_labels'],removal_manifest=str(removal_path))


def standard_fields(data):
    names=data.dtype.names
    return [n for n in names if n in ['x','y','z','opacity'] or n.startswith(('f_dc_','f_rest_','scale_','rot_'))]


def record_keys(data,fields):
    a=np.empty((len(data),len(fields)),dtype='<f4')
    for i,name in enumerate(fields):a[:,i]=data[name]
    if not np.isfinite(a).all():raise ValueError('Nonfinite Gaussian attributes in removal inputs')
    return a.view(np.dtype((np.void,a.dtype.itemsize*len(fields)))).ravel()


def deleted_indices(original,retained):
    """Exact multiset subtraction, including coincident centers and duplicate rows."""
    fields=standard_fields(retained)
    if not set(fields)<=set(original.dtype.names):raise ValueError('Original/removal Gaussian schemas differ')
    for name in ['x','y','z','opacity','f_dc_0','scale_0','rot_0']:
        if name not in fields:raise ValueError('Removal inputs are not standard Gaussian models')
    a=record_keys(original,fields);b=record_keys(retained,fields)
    order=np.argsort(a,kind='stable');unique,start,counts=np.unique(a[order],return_index=True,return_counts=True)
    bu,bc=np.unique(b,return_counts=True);positions=np.searchsorted(unique,bu)
    valid=positions<len(unique)
    if not valid.all() or not np.array_equal(unique[positions],bu):
        raise ValueError('Removed model is not an exact unchanged subset of the original Gaussian attributes')
    if np.any(bc>counts[positions]):raise ValueError('Removal model contains more copies of a Gaussian than the original')
    keep=np.zeros(len(original),bool)
    for pos,n in zip(positions,bc):keep[order[start[pos]:start[pos]+n]]=True
    result=np.flatnonzero(~keep)
    if len(result)!=len(original)-len(retained):raise AssertionError('Gaussian subtraction count mismatch')
    return result


def extract(inputs,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);src=sources(inputs)
    original=ply(src['original'])['vertex'].data;retained=ply(src['retained'])['vertex'].data
    indices=deleted_indices(original,retained)
    if not len(indices):raise ValueError('The recorded removal deleted no Gaussians')
    labels=np.load(src['labels']);semantic=ply(inputs['original_gaussians'])['vertex'].data
    if len(labels)!=len(original) or len(semantic)!=len(original):raise ValueError('Semantic rows differ from pre-removal Gaussian rows')
    if not np.array_equal(xyz(original),xyz(semantic)):raise ValueError('Semantic labels are not aligned with pre-removal centers')
    target_ids=inputs['target_ids'];deleted_labels=labels[indices].astype('int64')
    if len(target_ids)==1:assignment=np.full(len(indices),target_ids[0],dtype='int64')
    else:
        assignment=deleted_labels.copy();known=np.isin(assignment,target_ids)
        if not known.all():
            if not known.any():raise ValueError('No deleted Gaussian has one of the requested object IDs')
            points=xyz(original[indices]);_,nearest=cKDTree(points[known]).query(points[~known],workers=4)
            assignment[~known]=assignment[known][nearest]
    paths={};counts={}
    for target in target_ids:
        selected=indices[assignment==target]
        if not len(selected):raise ValueError(f'No actual deleted Gaussians belong to object {target}')
        folder=root/str(target);folder.mkdir(exist_ok=True)
        destination=folder/'gaussians_world.ply'
        PlyData([PlyElement.describe(original[selected].copy(),'vertex')],text=False).write(str(destination))
        np.save(folder/'original_indices.npy',selected)
        paths[str(target)]=str(destination);counts[str(target)]=len(selected)
    report=dict(schema_version=2,kind='exact-deleted-original-gaussians',complete=True,training=False,
                sources={k:fingerprint(v) for k,v in src.items()},
                original_count=len(original),retained_count=len(retained),deleted_count=len(indices),
                object_counts=counts,objects=paths,
                assignment='all actual deleted rows; semantic IDs partition multiple targets; unlabeled deleted rows go to nearest deleted labeled component',
                invariants='Source rows including SH/opacity/scales/quaternions are copied without modification. No center tolerance or learned reconstruction.')
    write_json(root/'manifest.json',report)
    return report


def export_initial(inputs,report,output):
    """Publish the complete Gaussian result before any expensive mesh processing."""
    import shutil
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    paths=[inputs['background_gaussians']]
    for target in inputs['target_ids']:
        destination=out/f'{target}_gaussians.ply';shutil.copyfile(report['objects'][str(target)],destination);paths.append(destination)
    ranges=merge_gaussians(paths,out/'composed_gaussians.ply')
    write_json(out/'gaussian_ranges.json',dict(ranges=ranges,object_order=['background']+[str(t) for t in inputs['target_ids']]))
    write_json(out/'gaussians_manifest.json',dict(schema_version=2,complete=True,training=False,
        background=fingerprint(inputs['background_gaussians']),objects=report['objects'],object_counts=report['object_counts'],
        gaussian_coordinates='original COLMAP world; initial movement delta is identity'))


def movement_transform(node):
    """Same world-space rigid movement for the independently represented mesh and GS."""
    if node.get('gaussian_coordinates')!='original_world':raise ValueError('Regenerate scene: only original deleted Gaussian assets are supported')
    current=np.asarray(node['object_to_world']);reference=np.asarray(node['original_pose'])
    if np.array_equal(current,reference):return np.eye(4)
    return current@np.linalg.inv(reference)
