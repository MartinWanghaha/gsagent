"""Partition-by-partition visibility must equal the complete triangle scene."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_streamed_visibility_matches_monolithic_depth_color_and_objects(tmp_path,monkeypatch):
    from mesh_benchmark import stream_cache as stream
    from mesh_benchmark.core import MeshScene,rays,write,read
    from recompose.io import save_mesh
    specs=[]
    for i in range(5):
        p=tmp_path/f'part_{i}.ply'
        save_mesh(p,np.array([[-8,-8,4+i],[8,-8,4+i],[0,8,4+i]],dtype='float32'),
            np.array([[0,1,2]]),np.array([[.5,.25,.75]]*3))
        specs.append(dict(path=str(p)))
    obj=tmp_path/'object.ply';save_mesh(obj,np.array([[-1,-1,2],[1,-1,2],[0,1,2]],dtype='float32'),
        np.array([[0,1,2]]),np.array([[0,1,0]]*3));object_spec=dict(path=str(obj))
    cameras=[dict(name=f'{part}.png',stem=part,part=part,width=4,height=4,
        K=[[4.,0,2],[0,4.,2],[0,0,1]],w2c=np.eye(4).tolist()) for part in ['train','held_out']]
    document=dict(input_selection={},status='complete')
    info=dict(request_id='same frozen cameras and masks',center=[0,0,0],extent=10.,reference=dict(kind='unit test'))
    root=tmp_path/'comparison';common=root/'common/test';common.mkdir(parents=True)
    reference=np.array([[0,0,4],[1,0,4],[-1,0,4]],dtype='float32')
    np.savez(common/'reference_points.npz',background=reference,recomposition=reference,box=[[-10]*3,[10]*3])
    monkeypatch.setattr(stream,'common',lambda *a:(cameras,document,info))
    monkeypatch.setattr(stream,'specs_for',lambda s,m,state:specs+([object_spec] if state=='recomposition' else []))
    monkeypatch.setattr(stream,'image_mask',lambda c,s:(np.full((4,4,3),128,dtype='uint8'),np.zeros((4,4),dtype=bool)))
    config=dict(seed=0,pool_pixels=32,protocol_version=1)
    stream.prepare_streamed(root,'paintmesh','test',config)
    for state in ['background','recomposition']:
        mesh=MeshScene(specs+([object_spec] if state=='recomposition' else []))
        depth,color=mesh.cast(rays(cameras[1]),color=True);folder=root/'cache/paintmesh/test'/state
        np.testing.assert_array_equal(np.load(folder/'depth/held_out.npy'),depth.reshape(4,4))
        np.testing.assert_array_equal(np.asarray(Image.open(folder/'native_renders/held_out.png')),
            np.rint(np.clip(color,0,1).reshape(4,4,3)*255).astype('uint8'))
        meta=read(folder/'complete.json');assert meta['streamed_all_faces']
        assert len(meta['mesh_counts'])==5+(state=='recomposition')
        np.testing.assert_allclose(meta['geometry']['reference_to_mesh_mean'],mesh.distance(reference).mean(),atol=1e-7)
