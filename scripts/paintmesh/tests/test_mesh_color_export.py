from pathlib import Path
import sys
import numpy as np
import pytest
from plyfile import PlyData,PlyElement

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_benchmark.color_export import exact_color_indices,attach_colors,read_mesh


def test_exact_mapping_preserves_postprocessed_order_and_rejects_missing_vertices():
    source=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    colors=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype='uint8')
    ids=exact_color_indices(source,source[[3,1,2]],colors)
    np.testing.assert_array_equal(ids,[3,1,2])
    with pytest.raises(ValueError,match='absent'):exact_color_indices(source,[[.001,0,0]],colors)


def test_coincident_vertices_cannot_silently_pick_different_colors():
    points=np.zeros((2,3));colors=np.array([[0,0,0],[255,0,0]],dtype='uint8')
    with pytest.raises(ValueError,match='ambiguous'):exact_color_indices(points,points[:1],colors)


def test_color_writer_preserves_double_positions_extra_attributes_and_faces(tmp_path):
    vertices=np.array([(0.,0,0,7),(1.123456789123,0,0,9),(0,1,0,11)],dtype=[('x','f8'),('y','f8'),('z','f8'),('quality','u2')])
    faces=np.array([([0,1,2],)],dtype=[('vertex_indices','i4',(3,))])
    source=tmp_path/'source.ply';destination=tmp_path/'color.ply'
    PlyData([PlyElement.describe(vertices,'vertex'),PlyElement.describe(faces,'face')],text=False).write(source)
    colors=np.array([[255,0,0],[0,255,0],[0,0,255]],dtype='uint8')
    assert attach_colors(source,destination,colors,chunk=2)['all_original_geometry_and_properties_exact']
    data=read_mesh(destination)['vertex'].data
    for name in vertices.dtype.names:np.testing.assert_array_equal(vertices[name],data[name])
    np.testing.assert_array_equal(np.column_stack([data[n] for n in ['red','green','blue']]),colors)


def test_colored_primary_keeps_the_frozen_geometry_cache_identity(tmp_path,monkeypatch):
    import json
    import mesh_benchmark.core as core
    raw=tmp_path/'raw.ply';colored=tmp_path/'colored.ply';raw.touch();colored.touch()
    marker=tmp_path/'come/bag/mesh_complete.json';marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(dict(complete=True,path=str(colored),geometry_path=str(raw))))
    graph=tmp_path/'scenes/bag/composition/scene.json';graph.parent.mkdir(parents=True);graph.write_text('{}')
    monkeypatch.setattr(core,'BASE',tmp_path);monkeypatch.setattr(core,'RECOMPOSE',tmp_path/'scenes')
    assert core.specs_for('bag','come','recomposition')==[dict(path=str(raw))]
