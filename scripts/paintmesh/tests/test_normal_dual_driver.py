from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_bag_normal_mesh_dual import prepare_cached_partitions
from mesh_benchmark import normal_dual_queue as queue
from mesh_benchmark.fine_tsdf import partitions


class BufferedPlanCacheTests(unittest.TestCase):
    def test_exact_geometry_and_final_manifest_keep_original_writer(self):
        coords=np.array([[x,y,z] for x in range(-2,3) for y in range(-2,3) for z in range(-2,3)],dtype='int32')
        original=queue.write
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);calls=[]
            def durable(path,value):
                calls.append(Path(path));return original(path,value)
            with patch.object(queue,'write',durable):
                result=prepare_cached_partitions(out,coords,dict(test='cache writing only'))
                self.assertIs(queue.write,durable)
            self.assertEqual(calls,[out/'dual_queue/manifest.json'])
            expected=list(partitions(coords))
            self.assertEqual(len(expected),result['task_count'])
            for row,(core,active) in zip(result['tasks'],expected):
                with np.load(row['plan']['path']) as data:
                    np.testing.assert_array_equal(data['core'],core)
                    np.testing.assert_array_equal(data['active'],active)
            again=prepare_cached_partitions(out,coords,dict(test='cache writing only'))
            self.assertEqual(again['request_id'],result['request_id'])
        self.assertIs(queue.write,original)

    def test_original_durable_writer_is_restored_on_error(self):
        original=queue.write
        with patch.object(queue,'prepare_tasks',side_effect=RuntimeError('test interruption')):
            with self.assertRaises(RuntimeError):prepare_cached_partitions(Path('/tmp/example'),np.ones((1,3),dtype='int32'),{})
        self.assertIs(queue.write,original)


if __name__=='__main__':unittest.main()
