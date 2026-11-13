"""A scene update must not rewrite another scene or acknowledge stale figures."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_scoped_native_support_preserves_other_scene(tmp_path,monkeypatch):
    import evaluate_native_mesh_rgb as native
    from mesh_benchmark.core import write,read
    monkeypatch.setattr(native,'METHODS',['first','second'])
    camera=dict(name='test.png',stem='test',part='held_out')
    monkeypatch.setattr(native,'cameras',lambda scene:[camera])
    untouched=tmp_path/'common/car/background_native_support'
    untouched.mkdir(parents=True)
    write(untouched/'complete.json',dict(complete=True,request_id='previous car result'))
    Image.new('L',(2,2),255).save(untouched/'test.png')
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in untouched.iterdir()}
    heldout=tmp_path/'common/bag/held_out';heldout.mkdir(parents=True)
    Image.fromarray(np.array([[255,0],[0,0]],dtype='uint8')).save(heldout/'test_foreground.png')
    for method in native.METHODS:
        folder=tmp_path/'native'/method
        (folder/'depth').mkdir(parents=True);(folder/'rgb').mkdir()
        depth=np.array([[1,1],[1,0 if method=='second' else 1]],dtype='float32')
        np.save(folder/'depth/test.npy',depth)
        Image.new('RGB',(2,2),(110,90,60)).save(folder/'rgb/test.png')
    calls=[]
    def source(root,method,scene,state):
        calls.append(scene)
        folder=root/'native'/method
        return dict(depth=folder/'depth',rgb=folder/'rgb',request=dict(method=method))
    monkeypatch.setattr(native,'rendering',source)
    native.supports(tmp_path,['bag'])
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/'common/bag/background_native_support/test.png')),
                                  np.array([[0,255],[255,0]],dtype='uint8'))
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/'common/bag/recomposition_native_support/test.png')),
                                  np.array([[255,255],[255,0]],dtype='uint8'))
    assert set(calls)=={'bag'}
    assert {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in untouched.iterdir()}==before
    assert read(untouched/'complete.json')['request_id']=='previous car result'
    assert (tmp_path/'background/second/bag/renders/test.png').resolve()==tmp_path/'native/second/rgb/test.png'


def test_delivery_rejects_stale_figure_and_stale_support(tmp_path,monkeypatch):
    import publish_fine_whole_meshes as publication
    import plot_mesh_method_comparison as plotting
    from mesh_benchmark.core import write,read
    public=tmp_path/'public';root=public/'_internal/workspace';campaign=root/'fine_whole'
    monkeypatch.setattr(publication,'PUBLIC',public);monkeypatch.setattr(publication,'ROOT',root)
    monkeypatch.setattr(publication,'WORK',campaign)
    camera=dict(name='test.png',stem='test')
    monkeypatch.setattr(plotting,'choose_view',lambda root,scene:(camera,[0,0,2,2],.2))
    for state in publication.STATES:
        write(root/'common/bag'/(state+'_native_support')/'complete.json',dict(request_id=state))
        for method in publication.METHODS:
            geometry=dict(support_id=state+' geo',CD_L1=.01,F1=.95)
            write(root/'cache'/method/'bag'/state/'geometry_metrics.json',geometry)
            write(root/state/method/'bag/metrics.json',dict(complete=True,geometry=geometry,
                  request=dict(support_id=state)))
    for path in ['meshes/bag/stage1_world/mesh_scene.gltf','meshes/bag/world/mesh_scene.gltf',
                 'recomposition/bag/composition/world/mesh_scene.gltf','reports/background.md','reports/recomposition.md']:
        target=public/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('immutable test artifact')
    figure=public/'figures_all_methods/scenes/bag'
    for name in ['all_methods','background','recomposition']:
        figure.mkdir(parents=True,exist_ok=True);Image.new('RGB',(2,2)).save(figure/(name+'.png'))
    geometry_image=public/'figures_all_methods/geometry_latest/bag_all_methods.png'
    geometry_image.parent.mkdir(parents=True);Image.new('RGB',(2,2)).save(geometry_image)
    manifest=figure/'figure_manifest.json'
    def redraw():
        write(manifest,dict(signature=plotting.input_signature(root,'bag',camera),
                            records=[dict(method=m,status='complete') for m in publication.METHODS]))
    redraw()
    assert publication.scene_delivery_receipt('bag')['complete']
    rendered=root/'background/paintmesh/bag/renders/test.png';rendered.parent.mkdir(parents=True)
    Image.new('RGB',(2,2),(100,90,80)).save(rendered)
    with pytest.raises(ValueError,match='figure inputs have changed'):
        publication.scene_delivery_receipt('bag')
    redraw()
    assert publication.scene_delivery_receipt('bag')['complete']
    metric=root/'background/paintmesh/bag/metrics.json';record=read(metric)
    record['request']['support_id']='stale support';write(metric,record);redraw()
    with pytest.raises(ValueError,match='missing or stale scores'):
        publication.scene_delivery_receipt('bag')
    assert not (campaign/'publication/bag/evaluated.json').exists()


def test_cumulative_scene_delivery_and_parent_scope(tmp_path,monkeypatch):
    import publish_fine_whole_meshes as publication
    import evaluate_native_mesh_rgb as native
    from mesh_benchmark.core import write
    monkeypatch.setattr(publication,'WORK',tmp_path)
    write(tmp_path/'publication/bag/evaluated.json',dict(complete=True))
    write(tmp_path/'publication/cube/evaluated.json',dict(complete=False))
    assert publication.delivered_scenes(['car'])==['bag','car']
    read_bytes=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda path:b'python\0-u\0/home/member/wyc/gsagent/scripts/paintmesh/publish_fine_whole_meshes.py\0--scenes\0bag\0'
                        if str(path).startswith('/proc/') else read_bytes(path))
    assert native.publisher_scenes()==['bag']
    monkeypatch.setattr(Path,'read_bytes',lambda path:b'python\0unrelated.py\0--scenes\0bag\0'
                        if str(path).startswith('/proc/') else read_bytes(path))
    assert native.publisher_scenes() is None
    with pytest.raises(ValueError):native.selected_scenes(['bag','unknown'])
    with pytest.raises(ValueError):native.selected_scenes(['bag','bag'])
