"""Update movable object nodes in an existing complete mesh glTF.

Background buffers, nodes and accessors are preserved byte-for-byte as JSON
values. No background .npy/.ply is opened: its already-published glTF bindings
remain authoritative. Only the small object NPZ and binary buffer are read and
written. This changes scheduling/IO, never geometry or rendering quality.
"""
from pathlib import Path
import copy
import os
import numpy as np

from recompose.io import read_json, write_json
from recompose.geometry import check_similarity
from recompose.mesh_export import write_buffer


def export_registered_mesh_incremental(document, world):
    world=Path(world);gltf=world/'mesh_scene.gltf'
    if not gltf.exists():
        from recompose.mesh_export import export_mesh_scene
        return export_mesh_scene(document,world)
    original=read_json(gltf);result=copy.deepcopy(original)
    names={node.get('name'):i for i,node in enumerate(result['nodes'])}
    if len(names)!=len(result['nodes']):
        raise ValueError('Incremental glTF requires unique named scene nodes')
    expected_background=1+len(document['background'].get('patches',[]))
    background=[n for n in result['nodes'] if n.get('name','').startswith('background')]
    if len(background)!=expected_background:
        raise ValueError('Background partition set changed; a registration-only update cannot rebuild it')
    expected_objects={'object_'+str(obj['id']) for obj in document['objects']}
    present_objects={n.get('name') for n in result['nodes'] if n.get('name','').startswith('object_')}
    if expected_objects!=present_objects:
        raise ValueError('Object node set changed; a registration-only update cannot rebuild it')
    # Prove object edits cannot mutate a buffer/accessor shared with background.
    background_accessors=set()
    for node in background:
        for primitive in result['meshes'][node['mesh']]['primitives']:
            background_accessors.update(primitive.get('attributes',{}).values())
            if 'indices' in primitive:background_accessors.add(primitive['indices'])
    background_views={result['accessors'][i]['bufferView'] for i in background_accessors}
    background_buffers={result['bufferViews'][i]['buffer'] for i in background_views}
    edits=[]
    for obj in document['objects']:
        key='object_'+str(obj['id']);node=result['nodes'][names[key]]
        check_similarity(obj['object_to_world'])
        primitives=result['meshes'][node['mesh']]['primitives']
        if len(primitives)!=1:
            raise ValueError('Expected one native vertex-color primitive per object')
        primitive=primitives[0]
        if set(primitive['attributes'])!={'POSITION','COLOR_0'}:
            raise ValueError('Unexpected object attribute layout; refuse to drop attributes')
        position_i=primitive['attributes']['POSITION'];color_i=primitive['attributes']['COLOR_0'];index_i=primitive['indices']
        accessors=[position_i,color_i,index_i]
        views=[result['accessors'][i]['bufferView'] for i in accessors]
        buffers={result['bufferViews'][i]['buffer'] for i in views}
        if len(buffers)!=1 or set(accessors)&background_accessors or set(views)&background_views or buffers&background_buffers:
            raise ValueError('Object storage unexpectedly shared with background')
        if views[0]!=views[1] or views[0]==views[2]:
            raise ValueError('Unexpected packed object vertex/index storage')
        edits.append((obj,node,position_i,color_i,index_i,views[0],views[2],next(iter(buffers))))
    for obj,node,position_i,color_i,index_i,vertex_view,index_view,buffer_i in edits:
        path=Path(obj['mesh']['path'])
        buffer=world/'mesh_buffers'/f'object_{obj["id"]}_{obj["mesh"]["sha256"]}.bin'
        with np.load(path,allow_pickle=False) as asset:
            meta=write_buffer(buffer,asset['vertices'],asset['faces'],asset['colors'])
        result['buffers'][buffer_i]=dict(uri=os.path.relpath(buffer,world),byteLength=meta['byte_length'])
        result['bufferViews'][vertex_view]=dict(buffer=buffer_i,byteOffset=0,
            byteLength=meta['vertex_count']*16,byteStride=16,target=34962)
        result['bufferViews'][index_view]=dict(buffer=buffer_i,byteOffset=meta['vertex_count']*16,
            byteLength=meta['face_count']*12,target=34963)
        result['accessors'][position_i]=dict(bufferView=vertex_view,byteOffset=0,componentType=5126,
            count=meta['vertex_count'],type='VEC3',min=meta['minimum'],max=meta['maximum'])
        result['accessors'][color_i]=dict(bufferView=vertex_view,byteOffset=12,componentType=5121,
            normalized=True,count=meta['vertex_count'],type='VEC4')
        result['accessors'][index_i]=dict(bufferView=index_view,byteOffset=0,componentType=5125,
            count=meta['face_count']*3,type='SCALAR')
        node['matrix']=np.asarray(obj['object_to_world'],dtype='float64').T.ravel().tolist()
    result.setdefault('extras',{})['status']=document.get('status','unvalidated')
    result['extras']['incremental_object_registration']=True
    # Final structural guard: no accidental background mutation is publishable.
    for key,indices in [('nodes',[names[n['name']] for n in background]),
                        ('meshes',[n['mesh'] for n in background]),
                        ('accessors',background_accessors),('bufferViews',background_views),
                        ('buffers',background_buffers)]:
        if any(result[key][i]!=original[key][i] for i in indices):
            raise AssertionError('Background glTF binding changed during object-only update')
    write_json(gltf,result)
    return gltf
