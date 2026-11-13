"""Refinement publication preserves native topology, colors and precision."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_bag_tsdf_refine import commit_geometry,input_record
from mesh_benchmark.core import read,write,identity


def test_original_double_positions_survive_ply_roundtrip(tmp_path):
    from plyfile import PlyData
    positions=np.array([[20.+1e-9,0,1],[20.01,0,1],[20.,.01,1]],dtype='float64')
    faces=np.array([[0,1,2]],dtype='int64')
    color=np.array([[10,20,30,255],[40,50,60,255],[70,80,90,255]],dtype='uint8')
    spec=commit_geometry(tmp_path,positions,faces,color,None,{'path':'original.ply'},dict(moved_vertices=1))
    data=PlyData.read(spec['path']);v=data['vertex'].data
    actual=np.column_stack([v[n] for n in ['x','y','z']])
    np.testing.assert_array_equal(actual,positions)
    assert actual.dtype==np.float64
    assert actual[0,0]!=np.float32(actual[0,0])
    np.testing.assert_array_equal(data['face'].data['vertex_indices'][0],faces[0])
    np.testing.assert_array_equal(np.column_stack([v[n] for n in ['red','green','blue']]),color[:,:3])


def test_native_publication_reuses_exact_faces_and_colors(tmp_path):
    positions=np.array([[0,0,1],[.01,0,1],[0,.01,1]],dtype='float32')
    faces=np.array([[0,1,2]],dtype='uint32');ip=tmp_path/'original.indices.npy';np.save(ip,faces)
    color=np.array([[10,20,30,255],[40,50,60,255],[70,80,90,255]],dtype='uint8')
    before=identity(ip)
    spec=commit_geometry(tmp_path/'new',positions,faces,color,ip,{'path':'original.mesh.json'},dict(moved_vertices=1))
    meta=read(spec['path']);v=np.load(meta['vertices']['path'])
    assert meta['index_identity']==before==identity(ip)
    np.testing.assert_array_equal(v['position'],positions)
    np.testing.assert_array_equal(v['color'],color)
    assert input_record(spec)['indices']==before


def test_unchanged_partition_remains_an_exact_source_reference(tmp_path):
    spec={'path':str(tmp_path/'unmodified.ply')}
    actual=commit_geometry(tmp_path,np.zeros((3,3)),np.array([[0,1,2]]),np.ones((3,4)),None,spec,dict(moved_vertices=0))
    assert actual==spec
    assert list(tmp_path.iterdir())==[]
