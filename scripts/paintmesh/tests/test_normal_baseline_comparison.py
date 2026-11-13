import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'compare_normal_mesh_with_baselines.py'
spec=importlib.util.spec_from_file_location('normal_baseline_comparison',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SharedNativeEvaluationTests(unittest.TestCase):
    def test_background_exclusion_and_all_method_surface_intersection(self):
        foreground=np.array([[True,False,False],[False,False,False]])
        a=np.ones((2,3));b=np.array([[1,np.nan,1],[np.inf,0,1]])
        eligible,shared=module.shared_support([a,b],foreground,True)
        np.testing.assert_array_equal(shared,[[False,False,True],[False,False,True]])
        self.assertEqual(eligible.sum(),5)
        _,composed=module.shared_support([a,b],foreground,False)
        self.assertTrue(composed[0,0])
        with self.assertRaises(ValueError):module.shared_support([a[:1]],foreground,False)

    def test_camera_names_match_but_changed_pose_or_resolution_is_rejected(self):
        import copy
        camera=dict(name='one.JPG',stem='one',width=3,height=2,K=np.eye(3).tolist(),w2c=np.eye(4).tolist())
        other=copy.deepcopy(camera);other.update(name='two.JPG',stem='two')
        module.assert_same_cameras([other,camera],[camera,other])
        for key in ['w2c','width']:
            changed=copy.deepcopy(camera)
            if key=='width':changed[key]=4
            else:changed[key][0][3]=.001
            with self.assertRaises(ValueError):module.assert_same_cameras([changed],[camera])


if __name__=='__main__':unittest.main()
