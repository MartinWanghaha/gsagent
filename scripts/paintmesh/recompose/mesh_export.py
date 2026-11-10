"""Stream full-resolution, independently movable glTF mesh nodes.

Background geometry is cached by its source hash; no mesh decimation, merging of
instances or multi-GB Python lists are needed. Baked vertex color is unlit.
Native textured object GLBs are additionally retained by scene.export_scene.
"""
from pathlib import Path
import os
import tempfile
import numpy as np
from .io import ply, write_json


def write_buffer(path,vertices,faces,colors):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    n=len(vertices);m=len(faces)
    if n==0 or m==0:raise ValueError('Cannot export an empty mesh')
    lo=np.full(3,np.inf);hi=-lo
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.mesh-')
    try:
        with os.fdopen(fd,'wb') as out:
            for start in range(0,n,65536):
                v=np.asarray(vertices[start:start+65536],dtype='<f4')
                if not np.isfinite(v).all():raise ValueError('Nonfinite mesh coordinates')
                c=np.asarray(colors[start:start+len(v)])
                if np.issubdtype(c.dtype,np.floating):c=np.rint(np.clip(c,0,1)*255)
                chunk=np.empty(len(v),dtype=[('position','<f4',(3,)),('color','u1',(4,))])
                chunk['position']=v;chunk['color'][:,:3]=c[:,:3];chunk['color'][:,3]=255
                out.write(chunk.tobytes());lo=np.minimum(lo,v.min(0));hi=np.maximum(hi,v.max(0))
            for start in range(0,m,65536):
                f=np.asarray(faces[start:start+65536])
                if f.ndim!=2 or f.shape[1]!=3 or f.min()<0 or f.max()>=n:raise ValueError('Invalid triangle indices')
                out.write(f.astype('<u4').tobytes())
            out.flush();os.fsync(out.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return dict(vertex_count=n,face_count=m,minimum=lo,maximum=hi,byte_length=n*16+m*12)


class PlyColumns:
    """Slice structured mmap columns without materializing a full scene array."""
    def __init__(self,data,names,default=None):self.data=data;self.names=names;self.default=default
    def __len__(self):return len(self.data)
    def __getitem__(self,key):
        a=self.data[key]
        if self.default is not None:return np.full((len(a),len(self.names)),self.default,dtype='uint8')
        return np.column_stack([a[n] for n in self.names])


def export_mesh_scene(scene,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cache=out/'mesh_buffers';cache.mkdir(exist_ok=True)
    doc=dict(asset=dict(version='2.0',generator='PaintMesh composition'),scene=0,
             scenes=[dict(nodes=[])],nodes=[],meshes=[],buffers=[],bufferViews=[],accessors=[],
             extensionsUsed=['KHR_materials_unlit'],materials=[dict(name='Baked appearance',doubleSided=True,
             pbrMetallicRoughness=dict(metallicFactor=0,roughnessFactor=1),extensions={'KHR_materials_unlit':{}})],
             extras=dict(status=scene.get('status','unvalidated'),coordinate_convention=scene.get('coordinate_convention','COLMAP world'),
                         note='Full geometry with baked vertex colors. Native object GLBs retain UV textures.'))
    def add(name,buffer,meta,T):
        bi=len(doc['buffers']);vi=len(doc['bufferViews']);ai=len(doc['accessors']);ni=len(doc['nodes'])
        doc['buffers'].append(dict(uri=os.path.relpath(buffer,out),byteLength=meta['byte_length']))
        doc['bufferViews'] += [dict(buffer=bi,byteOffset=0,byteLength=meta['vertex_count']*16,byteStride=16,target=34962),
                              dict(buffer=bi,byteOffset=meta['vertex_count']*16,byteLength=meta['face_count']*12,target=34963)]
        doc['accessors'] += [dict(bufferView=vi,byteOffset=0,componentType=5126,count=meta['vertex_count'],type='VEC3',min=meta['minimum'],max=meta['maximum']),
                             dict(bufferView=vi,byteOffset=12,componentType=5121,normalized=True,count=meta['vertex_count'],type='VEC4'),
                             dict(bufferView=vi+1,byteOffset=0,componentType=5125,count=meta['face_count']*3,type='SCALAR')]
        doc['meshes'].append(dict(name=name,primitives=[dict(attributes={'POSITION':ai,'COLOR_0':ai+1},indices=ai+2,material=0)]))
        doc['nodes'].append(dict(name=name,mesh=ni,matrix=np.asarray(T).T.ravel().tolist()))
        doc['scenes'][0]['nodes'].append(ni)
    refs=[scene['background']['mesh']]+scene['background'].get('patches',[])
    from .io import read_json
    for i,ref in enumerate(refs):
        buffer=cache/f'{ref["sha256"]}.bin';metadata=buffer.with_suffix('.json')
        meta=read_json(metadata) if metadata.exists() else None
        if meta is None or not buffer.exists() or buffer.stat().st_size!=meta['byte_length']:
            mesh=ply(ref['path']);v=mesh['vertex'].data
            colors=PlyColumns(v,['red','green','blue'],default=None if 'red' in v.dtype.names else 180)
            meta=write_buffer(buffer,PlyColumns(v,['x','y','z']),mesh['face'].data['vertex_indices'],colors)
            write_json(metadata,meta)
        add('background' if i==0 else f'background_patch_{i}',buffer,meta,np.eye(4))
    for node in scene['objects']:
        asset=np.load(node['mesh']['path']);buffer=cache/f'object_{node["id"]}_{node["mesh"]["sha256"]}.bin'
        meta=write_buffer(buffer,asset['vertices'],asset['faces'],asset['colors'])
        add('object_'+node['id'],buffer,meta,node['object_to_world'])
    write_json(out/'mesh_scene.gltf',doc)
    return out/'mesh_scene.gltf'
