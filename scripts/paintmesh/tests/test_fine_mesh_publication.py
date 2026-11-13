"""New fine geometry must replace both modes without corrupting baseline data."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def native_part(folder):
    from recompose.io import fingerprint
    from mesh_benchmark.core import identity,write
    folder.mkdir(parents=True)
    vertices=np.empty(3,dtype=[('position','<f4',(3,)),('color','u1',(4,))])
    vertices['position']=[[-1,-1,3],[1,-1,3],[0,1,3]];vertices['color']=[170,120,90,255]
    vp=folder/'vertices.npy';fp=folder/'indices.npy';path=folder/'part.mesh.json'
    np.save(vp,vertices);np.save(fp,np.array([[0,1,2]],dtype='<u4'))
    write(path,dict(format='paintmesh-native-triangles-v1',vertices=fingerprint(vp),indices=fingerprint(fp),
        vertex_identity=identity(vp),index_identity=identity(fp),vertex_count=3,face_count=1,
        bounds=[[-1,-1,3],[1,1,3]],vertex_stride=16))
    return path


def test_scene_publication_rebinds_common_and_preserves_baseline_depths(tmp_path,monkeypatch):
    import publish_fine_whole_meshes as publication
    from mesh_benchmark.core import read,write,identity
    from recompose.io import save_mesh,fingerprint
    root=tmp_path/'public';work=root/'_internal/workspace';campaign=work/'fine_whole'
    monkeypatch.setattr(publication,'PUBLIC',root);monkeypatch.setattr(publication,'ROOT',work);monkeypatch.setattr(publication,'WORK',campaign)
    old_mesh=root/'meshes/bag/background.ply'
    save_mesh(old_mesh,np.array([[-1,-1,3],[1,-1,3],[0,1,3]]),np.array([[0,1,2]]),np.ones((3,3)))
    original=tmp_path/'original.ply';original.write_bytes(old_mesh.read_bytes())
    checkpoint=tmp_path/'model/point_cloud/iteration_5000/point_cloud.ply';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'immutable 3DGS source')
    local_background=tmp_path/'model/mesh/ours_5000/tsdf_fusion_post.ply';local_background.parent.mkdir(parents=True);local_background.write_bytes(old_mesh.read_bytes())
    (old_mesh.parent/'stage1.ply').symlink_to(original)
    graph=dict(status='complete',input_selection=dict(original_mesh=str(original)),
        background=dict(mesh=fingerprint(old_mesh),gaussians=fingerprint(checkpoint)),objects=[])
    write(root/'recomposition/bag/composition/scene.json',graph)
    for name in ['all_methods','background','recomposition']:
        p=root/'figures_all_methods/scenes/bag'/(name+'.png');p.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(2,2)).save(p)
    write(work/'protocol.json',dict(paintmesh_stage1=dict(enabled=True)))
    write(work/'common/bag/complete.json',dict(request_id='old common',request=dict(selection=identity(root/'recomposition/bag/composition/scene.json'))))
    for state in publication.STATES:
        folder=work/'common/bag'/(state+'_native_support');folder.mkdir(parents=True)
        Image.new('L',(2,2),255).save(folder/'view.png')
    for kind in ['background','stage1']:
        part=native_part(campaign/kind/'bag/parts')
        write(campaign/kind/'bag/complete.json',dict(complete=True,request=dict(voxel=.002,sdf_trunc=.008),mesh_specs=[dict(path=str(part))]))
        write(campaign/kind/'bag/frames/complete.json',dict(request=dict(checkpoint=identity(checkpoint))))
    for method in publication.METHODS:
        for state in publication.STATES:
            folder=work/'cache'/method/'bag'/state;folder.mkdir(parents=True)
            write(folder/'complete.json',dict(request=dict(common='old common')))
            (folder/'native_depth.npy').write_bytes(b'unchanged baseline depth')
            write(folder/'geometry_metrics.json',dict(old=True))
            write(work/state/method/'bag/metrics.json',dict(old=True))
    publication.publish_scene('bag')
    updated=read(root/'recomposition/bag/composition/scene.json')
    assert updated['stage1_whole']['complete']
    assert updated['fine_whole_extraction']['voxel_size']==.002
    assert updated['stage1_whole']['source_checkpoint']==identity(checkpoint)
    assert not old_mesh.exists() and not (old_mesh.parent/'stage1.ply').exists()
    assert (old_mesh.parent/'stage1_original_truncated.ply').resolve()==original
    common=read(work/'common/bag/complete.json')['request_id'];assert common!='old common'
    for state in publication.STATES:
        assert not (work/'common/bag'/(state+'_native_support')).exists()
        np.testing.assert_array_equal(np.asarray(Image.open(campaign/'before/bag/common'/(state+'_native_support')/'view.png')),np.full((2,2),255))
    for method in publication.METHODS:
        for state in publication.STATES:
            folder=work/'cache'/method/'bag'/state
            assert not (work/state/method/'bag/metrics.json').exists()
            if method.startswith('paintmesh'):
                assert not folder.exists()
                assert (campaign/'before/bag/cache'/method/state/'native_depth.npy').read_bytes()==b'unchanged baseline depth'
            else:
                assert (folder/'native_depth.npy').read_bytes()==b'unchanged baseline depth'
                assert read(folder/'complete.json')['request']['common']==common
                assert not (folder/'geometry_metrics.json').exists()
    before=(root/'recomposition/bag/composition/scene.json').read_bytes()
    publication.publish_scene('bag')
    assert (root/'recomposition/bag/composition/scene.json').read_bytes()==before
