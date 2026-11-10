"""Preserve generated texture detail for surface sampling and transform textured GLBs."""
import json
import struct
from pathlib import Path
import numpy as np


def sample_colors(asset,face_ids,barycentric):
    faces=asset['faces'][face_ids]
    colors=(asset['colors'][faces]*barycentric[:,:,None]).sum(1)
    if 'triangle_material' not in asset:return colors
    materials=asset['triangle_material'][face_ids]
    uv=(asset['triangle_uv'][face_ids]*barycentric[:,:,None]).sum(1)
    for material in np.unique(materials):
        if material<0:continue
        keep=materials==material;tex=asset[f'texture_{material}'].astype(float)/255.
        h,w=tex.shape[:2];coords=uv[keep]
        # glTF normalized UVs, repeat wrap, linear filtering at texel centers.
        x=np.mod(coords[:,0],1)*w-.5;y=(1-np.mod(coords[:,1],1))*h-.5
        x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);dx=x-x0;dy=y-y0
        a=tex[y0%h,x0%w];b=tex[y0%h,(x0+1)%w]
        c=tex[(y0+1)%h,x0%w];d=tex[(y0+1)%h,(x0+1)%w]
        colors[keep]=((a*(1-dx[:,None])+b*dx[:,None])*(1-dy[:,None])+
                      (c*(1-dx[:,None])+d*dx[:,None])*dy[:,None])*asset[f'texture_factor_{material}']
    return np.clip(colors,0,1)


def transform_glb(source,destination,matrix):
    """Add one parent node; retain embedded textures, materials and original node transforms."""
    data=Path(source).read_bytes()
    if data[:4]!=b'glTF' or struct.unpack_from('<I',data,4)[0]!=2:raise ValueError('Expected binary glTF 2.0')
    chunks=[];offset=12;document=None
    while offset<len(data):
        size,kind=struct.unpack_from('<II',data,offset);payload=data[offset+8:offset+8+size];offset+=8+size
        if kind==0x4E4F534A:document=json.loads(payload)
        else:chunks.append((kind,payload))
    if document is None:raise ValueError('GLB has no JSON document')
    active=document.get('scene',0);scene=document['scenes'][active]
    node=dict(name='PaintMesh object_to_world',matrix=np.asarray(matrix).T.ravel().tolist(),children=scene['nodes'])
    scene['nodes']=[len(document['nodes'])];document['nodes'].append(node)
    payload=json.dumps(document,separators=(',',':')).encode();payload+=b' '*((-len(payload))%4)
    content=struct.pack('<II',len(payload),0x4E4F534A)+payload
    for kind,payload in chunks:content+=struct.pack('<II',len(payload),kind)+payload
    dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(struct.pack('<4sII',b'glTF',2,12+len(content))+content)
