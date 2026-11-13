from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_bag_depth_tetra import save_arrays,task_manifest
from mesh_benchmark.core import read,identity
from mesh_benchmark.depth_tetra_field import prepare_frames,refine_edges,make_pivots,query_field,marching_tetrahedra


class TetraDriverContractTests(unittest.TestCase):
    def test_all_candidate_arrays_keep_exact_order_in_gpu_task_shards(self):
        points=np.arange(39,dtype='float32').reshape(-1,3)
        arrays=dict(points=points,reference_points=points+.125,
            reference_normals=np.tile([0.,0.,-1.],(len(points),1)).astype('float32'),
            reference_scales=np.linspace(.01,.02,len(points),dtype='float32'))
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);manifest=task_manifest(out,'query',arrays,chunk=5)
            self.assertEqual(manifest['count'],13)
            self.assertEqual([t['stop']-t['start'] for t in manifest['tasks']],[5,5,3])
            collected={k:[] for k in arrays}
            for task in manifest['tasks']:
                self.assertEqual(identity(task['input']['path']),task['input'])
                with np.load(task['input']['path']) as data:
                    for k in arrays:collected[k].append(data[k])
            for k in arrays:np.testing.assert_array_equal(np.concatenate(collected[k]),arrays[k])

    def test_real_field_root_sharding_and_final_topology_are_compatible(self):
        import torch
        torch.set_num_threads(2)
        frame=dict(depth=np.ones((16,16),dtype='float32'),normal=np.tile([0,0,-1.],(16,16,1)).astype('float32'),
            rgb=np.full((16,16,3),[123,89,45],dtype='uint8'),alpha=np.ones((16,16),dtype='float32'),
            confidence=np.ones((16,16),dtype='float32'),K=np.array([[40.,0,8],[0,40.,8],[0,0,1.]]),w2c=np.eye(4))
        anchors=dict(positions=np.array([[-.05,-.05,1],[.05,-.05,1],[-.05,.05,1],[.05,.05,1]],dtype='float32'),
            normals=np.tile([0,0,-1.],(4,1)).astype('float32'),scale=np.full(4,.01,dtype='float32'))
        points=np.array([[-.05,-.05,.98],[.05,-.05,1.02],[-.05,.05,1.02],[.05,.05,1.02]],dtype='float32')
        references=points.copy();references[:,2]=1
        prepared=prepare_frames([frame])
        field=query_field(points,anchors,prepared,reference_points=references,
            reference_normals=anchors['normals'],reference_scales=anchors['scale'])
        mesh=marching_tetrahedra(points,np.array([[0,1,2,3]]),field,anchors)
        edges=mesh['crossing_edges']
        result=refine_edges(points[edges[:,0]],points[edges[:,1]],anchors,prepared,
            reference_points=mesh['edge_reference_points'],reference_normals=mesh['edge_reference_normals'],reference_scales=mesh['edge_reference_scales'])
        self.assertTrue(result['support'].all())
        self.assertTrue(np.all(result['support'][mesh['faces']]))
        np.testing.assert_allclose(result['vertices'][:,2],1,atol=.0004)
        np.testing.assert_array_equal(result['colors'],np.tile([123,89,45],(len(edges),1)))


if __name__=='__main__':unittest.main()
