"""Attach native colors while preserving every geometric value and triangle byte."""
from pathlib import Path
import os
import shutil
import tempfile
import numpy as np
from plyfile import PlyData


def read_mesh(path):
    return PlyData.read(str(path),mmap='r',known_list_len={'face':{'vertex_indices':3}})


def positions(data):
    return np.column_stack([data[n] for n in ['x','y','z']])


def exact_color_indices(original,processed,colors):
    """Only exact source vertices are allowed; ambiguous colors are rejected."""
    a=np.asarray(original,dtype='<f8').copy();b=np.asarray(processed,dtype='<f8').copy()
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite mesh positions')
    a[a==0]=0;b[b==0]=0  # Signed zero does not change geometry.
    keytype=np.dtype((np.void,24));keys=a.view(keytype).ravel();query=b.view(keytype).ravel()
    unique,first,inverse,counts=np.unique(keys,return_index=True,return_inverse=True,return_counts=True)
    if (counts>1).any() and not np.array_equal(colors,colors[first[inverse]]):
        raise ValueError('Coincident original vertices have different colors; exact association is ambiguous')
    order=np.searchsorted(unique,query)
    if np.any(order>=len(unique)) or not np.array_equal(unique[np.minimum(order,len(unique)-1)],query):
        raise ValueError('Postprocessed geometry contains a vertex absent from the original mesh')
    indices=first[order];np.testing.assert_array_equal(a[indices],b)
    return indices


def attach_colors(source,destination,colors,chunk=262144):
    """Stream a binary PLY, adding RGB and copying all following elements exactly."""
    source=Path(source);destination=Path(destination);mesh=read_mesh(source)
    if mesh.text:raise ValueError('Binary source mesh required')
    if mesh.elements[0].name!='vertex':raise ValueError('Vertex must be the first PLY element')
    v=mesh['vertex'].data
    if len(colors)!=len(v) or colors.shape!=(len(v),3) or colors.dtype!=np.uint8:
        raise ValueError('Expected one uint8 RGB color per unchanged vertex')
    with source.open('rb') as stream:
        lines=[]
        while True:
            line=stream.readline();lines.append(line)
            if not line:raise ValueError('Missing PLY end_header')
            if line.strip()==b'end_header':break
        offset=stream.tell()
    missing=[n for n in ['red','green','blue'] if n not in v.dtype.names]
    header=[];inside=False;inserted=False
    for line in lines:
        if inside and (line.startswith(b'element ') or line.strip()==b'end_header') and not inserted:
            header += [f'property uchar {n}\n'.encode() for n in missing];inserted=True
        header.append(line)
        if line.startswith(b'element '):inside=line.split()[1]==b'vertex'
    dtype=np.dtype(v.dtype.descr+[(n,'u1') for n in missing])
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=destination.parent,prefix='.colored-')
    try:
        with os.fdopen(fd,'wb') as output,source.open('rb') as original:
            output.writelines(header)
            for start in range(0,len(v),chunk):
                stop=min(start+chunk,len(v));rows=np.empty(stop-start,dtype=dtype)
                for name in v.dtype.names:rows[name]=v[name][start:stop]
                for i,name in enumerate(['red','green','blue']):rows[name]=colors[start:stop,i]
                output.write(rows.tobytes())
            original.seek(offset+len(v)*v.dtype.itemsize)
            shutil.copyfileobj(original,output,8<<20)
            output.flush();os.fsync(output.fileno())
        os.replace(temp,destination)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    check=read_mesh(destination)
    for start in range(0,len(v),chunk):
        for name in v.dtype.names:
            if name not in ['red','green','blue']:
                np.testing.assert_array_equal(v[name][start:start+chunk],check['vertex'].data[name][start:start+chunk])
    for old,new in zip(mesh.elements[1:],check.elements[1:]):
        if old.name!=new.name or old.count!=new.count:raise ValueError('PLY topology/elements changed')
        for start in range(0,old.count,chunk):np.testing.assert_array_equal(old.data[start:start+chunk],new.data[start:start+chunk])
    return dict(vertices=len(v),faces=len(mesh['face'].data),all_original_geometry_and_properties_exact=True)
