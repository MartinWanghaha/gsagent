from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_bag_surfel_fssr import inspect_mesh,export_evidence
from recompose.io import save_mesh


class FSSRPilotBoundaryChecks(unittest.TestCase):
    def test_actual_colored_mesh_required_and_bad_triangles_rejected(self):
        vertices=np.array([[0,0,1],[1,0,1],[0,1,1]],dtype='float32')
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'mesh.ply'
            save_mesh(path,vertices,np.array([[0,1,2]]),np.ones((3,3),dtype='float32')*.5)
            counts=inspect_mesh(path)
            self.assertEqual((counts['vertices'],counts['faces']),(3,1))
            save_mesh(path,vertices,np.array([[0,1,3]]),np.ones((3,3),dtype='float32'))
            with self.assertRaisesRegex(ValueError,'indices'):inspect_mesh(path)
            save_mesh(path,vertices,np.array([[0,1,2]]))
            with self.assertRaisesRegex(ValueError,'RGB'):inspect_mesh(path)

    def test_cached_incomplete_or_changed_evidence_cannot_advance(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name)
            for folder in ['frames','teacher_views']:(out/folder).mkdir()
            training=out/'frames/complete.json';teacher=out/'teacher_views/complete.json'
            training.write_text(json.dumps(dict(frames=[{}]*84,all_depth_rgb_camera_arrays_exact=True)))
            teacher.write_text(json.dumps(dict(frames=[{}]*13)))
            with self.assertRaisesRegex(ValueError,'Complete exact'):export_evidence(out)
            training.write_text(json.dumps(dict(frames=[{}]*85,all_depth_rgb_camera_arrays_exact=False)))
            with self.assertRaisesRegex(ValueError,'Complete exact'):export_evidence(out)


if __name__=='__main__':unittest.main()
