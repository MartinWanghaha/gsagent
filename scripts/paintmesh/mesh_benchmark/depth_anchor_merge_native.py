"""Native execution of the unchanged deterministic surface-anchor merge.

The reference Python implementation remains in depth_surface_anchors.py. This
adapter preserves its NumPy validation, normalization, sorting and scale levels;
only its point-by-point hash-neighbor loop executes in a standalone C++ binary.
No threshold, observation, representative position or color is changed.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

from .depth_surface_anchors import PARAMETERS,ARRAY_FIELDS,_checked_anchors,empty_anchors

SOURCE=Path(__file__).with_suffix('.cpp')
DEFAULT_BUILD=Path(__file__).resolve().parents[3]/'output/3dgs_with_mesh/mesh_comparison_whole/experiments/bag_depth_tetra_20261112/code_build'


def _identity(path):
    path=Path(path).resolve(strict=True);stat=path.stat()
    return dict(path=str(path),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)


def build_binary(build_dir=None):
    """Compile with system g++; binaries and receipts stay in experiment output."""
    build=Path(build_dir or DEFAULT_BUILD);build.mkdir(parents=True,exist_ok=True)
    source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    binary=build/'depth_anchor_merge_native';receipt=build/'depth_anchor_merge_native.json'
    compiler=shutil.which('g++')
    if compiler is None:raise RuntimeError('System g++ is required for exact native merge acceleration')
    compiler=str(Path(compiler).resolve())
    flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off']
    signature=dict(source_sha256=source_sha,compiler=_identity(compiler),flags=flags)
    with (build/'depth_anchor_merge_native.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists() and binary.exists():
            previous=json.loads(receipt.read_text())
            if previous.get('signature')==signature and previous.get('binary')==_identity(binary):return binary
        temporary=build/'.depth_anchor_merge_native.tmp'
        command=[compiler,*flags,str(SOURCE),'-o',str(temporary)]
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Native merge compile failed:\n'+result.stderr)
        os.replace(temporary,binary)
        row=dict(complete=True,signature=signature,binary=_identity(binary),source=_identity(SOURCE),
            compiler_version=subprocess.run([compiler,'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0],
            command=command,updated=time.time(),conda_environment_modified=False)
        receipt.write_text(json.dumps(row,indent=2)+'\n')
    return binary


def merge_anchors(frame_outputs,build_dir=None,work_dir=None):
    """Return the same retained source arrays/order and multiplicities as Python.

    Optional build_dir/work_dir affect storage only. Every input source anchor
    participates in the same confidence-first greedy representative decisions.
    """
    if sys.byteorder!='little':raise RuntimeError('Native raw-array protocol requires little-endian host')
    arrays=[a[0] if isinstance(a,tuple) else a for a in frame_outputs]
    counts=[_checked_anchors(a) for a in arrays]
    if not sum(counts):return empty_anchors(),dict(complete=True,input_anchors=0,output_anchors=0,merged_observations=0,no_point_cap=True)
    joined={key:np.concatenate([a[key] for a in arrays],axis=0) for key in ARRAY_FIELDS}
    count=sum(counts);positions=joined['positions'].astype('float64');normals=joined['normals'].astype('float64')
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    scales=joined['scale'].astype('float64');confidence=joined['confidence']
    order=np.lexsort((positions[:,2],positions[:,1],positions[:,0],joined['pixels'][:,0],joined['pixels'][:,1],
        joined['frame_indices'],scales,-confidence.astype('float64')))
    minimum=float(scales.min());levels=np.floor(np.log2(scales/minimum)).astype('int32')
    max_level=int(levels.max());cells=np.array([minimum*(2.**i)*(2*PARAMETERS['merge_distance_pixels']) for i in range(max_level+2)],dtype='<f8')
    if not np.isfinite(cells).all() or (cells<=0).any():raise ValueError('Scale hash cell width overflows')
    binary=build_binary(build_dir)
    parent=Path(work_dir) if work_dir is not None else binary.parent
    parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    with tempfile.TemporaryDirectory(prefix='merge-raw-',dir=parent) as name:
        folder=Path(name)
        for key,value in [('positions',positions),('normals',normals),('scale',scales),('order',order.astype('<i8')),('levels',levels.astype('<i4')),('cells',cells)]:
            np.ascontiguousarray(value).tofile(folder/(key+'.raw'))
        command=[str(binary),str(folder),str(count),str(max_level),repr(PARAMETERS['merge_distance_pixels']),
            repr(PARAMETERS['merge_point_plane_pixels']),repr(math.cos(math.radians(PARAMETERS['merge_normal_degrees']))),
            repr(PARAMETERS['merge_scale_ratio']),repr(math.degrees(1.))]
        completed=subprocess.run(command,capture_output=True,text=True)
        if completed.returncode:raise RuntimeError('Native anchor merge failed:\n'+completed.stderr)
        if completed.stderr:print(completed.stderr,end='',flush=True)
        ids=np.fromfile(folder/'kept.raw',dtype='<i8');multiplicity=np.fromfile(folder/'multiplicity.raw',dtype='<i4')
        stats=json.loads((folder/'stats.json').read_text())
        if len(ids)!=stats['output_anchors'] or len(ids)!=len(multiplicity) or (ids<0).any() or (ids>=count).any() or int(multiplicity.sum(dtype='int64'))!=count:
            raise ValueError('Native representative protocol output is invalid')
    result={key:joined[key][ids].copy() for key in ARRAY_FIELDS}
    result['observation_counts']=multiplicity
    receipt=dict(complete=True,input_frames=len(arrays),input_anchors=count,output_anchors=len(ids),
        merged_observations=count-len(ids),candidate_pairs_checked=stats['candidate_pairs_checked'],
        maximum_merged_distance_source_pixels=stats['maximum_merged_distance_source_pixels'],
        maximum_merged_point_plane_source_pixels=stats['maximum_merged_point_plane_source_pixels'],
        maximum_merged_normal_angle_degrees=stats['maximum_merged_normal_angle_degrees'],parameters=PARAMETERS,
        deterministic_best_source_observation=True,world_positions_averaged=False,colors_averaged=False,
        opposite_orientations_merged=False,no_point_cap=True,all_frame_anchors_considered=True,
        scale_levels=len(np.unique(levels)),
        scale_policy='Only compatible neighboring scale levels are candidates; distant-scale observations are retained, never discarded',
        layer_limit='Surfaces separated by more than the explicit point-to-plane tolerance cannot merge; subpixel-separated same-facing layers remain observationally ambiguous',
        unknown_topology_preserved=True,test_data_used=False,gaussians_updated=False,
        native_acceleration=dict(binary=_identity(binary),source=_identity(SOURCE),adapter=_identity(Path(__file__)),
            unchanged_reference=_identity(Path(__file__).with_name('depth_surface_anchors.py')),
            loop='Same three scale levels, 27 hash offsets, candidate order and greedy tuple minimum',
            exact_source_order_and_representatives=True,geometry_quality_changed=False,elapsed_seconds=time.time()-started))
    return result,receipt
