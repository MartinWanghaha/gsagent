"""Fixed-resolution allocation and partitioned MC must preserve the surface."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.fine_tsdf import blocks_from_depth,partitions,retain_owned_faces,integrate_partition


def triangles(vertices,faces):
    # Hash geometric triangles independently of MC vertex numbering/winding.
    return sorted(tuple(sorted(tuple(np.round(v,7)) for v in t)) for t in vertices[faces])


def test_all_pixels_participate_in_sparse_allocation():
    d=np.zeros((8,8),dtype='float32');d[3,5]=.1
    K=np.array([[30.,0,4],[0,30.,4],[0,0,1]])
    blocks=blocks_from_depth(d,K,np.eye(4))
    p=np.linalg.inv(K)@np.array([5,3,1])*.1
    assert any(np.array_equal(b,np.floor(p/.032).astype('int32')) for b in blocks)
    assert len(blocks)>0


def test_partitioned_native_tsdf_matches_monolithic_geometry_and_rgb():
    h=w=32;yy,xx=np.mgrid[:h,:w]
    d=(.12+.0002*xx).astype('float32');rgb=np.empty((h,w,3),dtype='uint8');rgb[:]=[170,120,90]
    K=np.array([[30.,0,16],[0,30.,16],[0,0,1]])
    frame=dict(depth=d,rgb=rgb,K=K,w2c=np.eye(4))
    blocks=blocks_from_depth(d,K,np.eye(4));vertices,faces,colors=integrate_partition(blocks,[frame],'CPU:0')
    expected=triangles(vertices,faces);actual=[];owners=[]
    for core,halo in partitions(blocks,core_blocks=4,max_blocks=64):
        v,f,c=integrate_partition(halo,[frame],'CPU:0');keep=retain_owned_faces(v,f,core)
        actual+=triangles(v,f[keep]);owners.extend(map(tuple,core.tolist()))
        np.testing.assert_allclose(c[f[keep]],np.broadcast_to(np.array([170,120,90])/255.,c[f[keep]].shape),atol=2e-7)
    assert sorted(actual)==expected
    assert len(owners)==len(set(owners))==len(blocks)
    assert len(actual)==len(set(actual))


def test_partition_does_not_change_fine_lattice_when_capacity_changes():
    coords=np.array([[x,y,z] for x in range(8) for y in range(3) for z in range(2)],dtype='int32')
    a=list(partitions(coords,core_blocks=12,max_blocks=30))
    assert len(a)>1
    np.testing.assert_array_equal(np.unique(np.concatenate([c for c,h in a]),axis=0),coords)
    for core,halo in a:
        assert len(halo)<=30
        assert set(map(tuple,core))<=set(map(tuple,halo))


def test_native_partition_render_and_gltf_share_exact_geometry_buffers(tmp_path):
    from mesh_benchmark.fine_tsdf import fuse
    from mesh_benchmark.core import MeshScene,rays,read
    from recompose.mesh_export import export_mesh_scene
    from recompose.io import fingerprint
    path=tmp_path/'frame.npz';K=np.array([[32.,0,8],[0,32.,8],[0,0,1]])
    np.savez(path,depth=np.full((16,16),.08,dtype='float32'),rgb=np.full((16,16,3),[170,120,90],dtype='uint8'),K=K,w2c=np.eye(4))
    result=fuse(tmp_path/'fusion',[dict(path=str(path))],device='CPU:0')
    spec=result['mesh_specs'][0];meta=read(spec['path'])
    scene=MeshScene([spec]);z,rgb=scene.cast(rays(dict(width=1,height=1,K=[[32,0,.5],[0,32,.5],[0,0,1]],w2c=np.eye(4))),color=True)
    np.testing.assert_allclose(z,[.08],atol=1e-6)
    np.testing.assert_allclose(rgb,[[170/255.,120/255.,90/255.]],atol=1e-7)
    graph=dict(background=dict(mesh=fingerprint(spec['path'])),objects=[])
    output=export_mesh_scene(graph,tmp_path/'world');doc=read(output)
    assert len(doc['buffers'])==2 and len(doc['meshes'])==1
    assert (output.parent/doc['buffers'][0]['uri']).resolve()==Path(meta['vertices']['path']).resolve()
    assert (output.parent/doc['buffers'][1]['uri']).resolve()==Path(meta['indices']['path']).resolve()
    raw=np.load(meta['indices']['path'],mmap_mode='r');view=doc['bufferViews'][1]
    exported=np.frombuffer(Path(meta['indices']['path']).read_bytes(),dtype='<u4',count=raw.size,offset=view['byteOffset'])
    np.testing.assert_array_equal(exported.reshape(-1,3),raw)
    assert list((tmp_path/'world/mesh_buffers').iterdir())==[]
