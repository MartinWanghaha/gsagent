#!/usr/bin/env python3
"""CPU CGAL Delaunay for exact float32 depth-anchor candidates.

This adapter uses the already built tetra-neRF extension and its matching
PyTorch environment. It constructs no density/SDF and never starts CUDA work.
The complete input point array is passed once, with no point cap or sampling.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
import time

RESEARCH=Path('/home/member/wyc/mesh_gs_research')
NATIVE_PYTHON=RESEARCH/'envs/come_probe/bin/python'
NATIVE_EXTENSION=RESEARCH/'build/tetra_triangulation_20261101/native/tetranerf/utils/extension/tetranerf_cpp_extension.cpython-310-x86_64-linux-gnu.so'


def identity(path):
    path=Path(path).resolve(strict=True);stat=path.stat()
    return dict(path=str(path),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        with os.fdopen(fd,'w') as handle:
            json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def native_environment():
    prefix=Path('/home/member/.conda/envs/aurafusion360')
    env={k:v for k,v in os.environ.items() if k not in ['PYTHONHOME','PYTHONPATH']}
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        LD_LIBRARY_PATH=str(prefix/'lib')+':'+env.get('LD_LIBRARY_PATH',''))
    return env


def validated_points(path):
    import numpy as np
    with np.load(path) as arrays:
        if 'points' not in arrays:raise ValueError('NPZ input must contain points[N,3]')
        points=arrays['points'].copy()
    if points.dtype!=np.dtype('float32') or points.ndim!=2 or points.shape[1]!=3 or len(points)<4:
        raise ValueError('At least four float32 N,3 points are required; no silent precision conversion')
    if not np.isfinite(points).all():raise ValueError('Delaunay input contains nonfinite points')
    return np.ascontiguousarray(points)


def validate_cells(points,cells):
    import numpy as np
    if cells.ndim!=2 or cells.shape[1]!=4 or not len(cells) or not np.issubdtype(cells.dtype,np.integer):
        raise ValueError('Delaunay did not return a nonempty integer M,4 tetrahedral array')
    if cells.min()<0 or cells.max()>=len(points):raise ValueError('Delaunay references points outside its complete input array')
    smallest=None;largest=0.
    for start in range(0,len(cells),262144):
        indices=cells[start:start+262144]
        if np.any(np.diff(np.sort(indices,axis=1),axis=1)==0):raise ValueError('A tetrahedron repeats a vertex')
        p=points[indices].astype('float64')
        volume=np.abs(np.einsum('ij,ij->i',np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),p[:,3]-p[:,0]))/6.
        if not np.isfinite(volume).all() or np.any(volume<=0):raise ValueError('Degenerate or nonfinite Delaunay tetrahedron')
        lo=float(volume.min());smallest=lo if smallest is None else min(smallest,lo);largest=max(largest,float(volume.max()))
    return dict(minimum_tetrahedron_volume=smallest,maximum_tetrahedron_volume=largest,all_cells_nondegenerate=True)


def triangulate(source,destination,receipt=None):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    source=Path(source);destination=Path(destination);stamp=identity(source)
    points=validated_points(source)
    spec=importlib.util.spec_from_file_location('tetranerf_cpp_extension',NATIVE_EXTENSION)
    extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
    start=time.time()
    # CPU input makes the upstream wrapper stay on CPU; CUDA is hidden too.
    tensor=torch.from_numpy(points)
    with torch.no_grad():cells=extension.triangulate(tensor).cpu().numpy().copy()
    elapsed=time.time()-start;quality=validate_cells(points,cells)
    if identity(source)!=stamp:raise ValueError('Candidate points changed during triangulation')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name('.'+destination.name+'.tmp.npy')
    np.save(temporary,cells);os.replace(temporary,destination)
    record=dict(complete=True,input=stamp,output=identity(destination),
        implementation=identity(Path(__file__)),native_extension=identity(NATIVE_EXTENSION),
        python=sys.executable,torch=torch.__version__,cuda_used=False,device='CPU',
        point_count=len(points),tetrahedron_count=len(cells),all_input_points_passed=True,
        point_cap=None,point_subsampling=False,scalar_field_evaluated=False,
        point_array_sha256=hashlib.sha256(memoryview(points)).hexdigest(),
        bounds=[points.min(0).tolist(),points.max(0).tolist()],
        elapsed_seconds=elapsed,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cpu_threads=4,updated=time.time(),**quality)
    atomic_json(receipt or destination.with_suffix('.json'),record)
    print('DEPTH_ANCHOR_DELAUNAY_COMPLETE',len(points),len(cells),'seconds',elapsed,flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path);parser.add_argument('--native-worker',action='store_true')
    args=parser.parse_args()
    if not args.native_worker:
        command=[str(NATIVE_PYTHON),'-u',str(Path(__file__).resolve()),'--input',str(args.input),
            '--output',str(args.output),'--native-worker']
        if args.receipt:command+=['--receipt',str(args.receipt)]
        result=subprocess.run(command,env=native_environment(),check=False)
        raise SystemExit(result.returncode)
    triangulate(args.input,args.output,args.receipt)
