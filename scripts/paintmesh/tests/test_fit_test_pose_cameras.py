import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fit_test_pose_cameras import interpolate, continuous_order, minimum_frame_count


class CameraTests(unittest.TestCase):
    def test_exact_anchor_centers_and_orientations(self):
        c=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]])
        r=Rotation.from_euler('z',[0,45,90],degrees=True).as_matrix()
        cs,rs,_,idx,_=interpolate(c,r,30)
        self.assertEqual(len(cs),30)
        np.testing.assert_allclose(cs[idx],c,atol=1e-12)
        np.testing.assert_allclose(rs[idx],r,atol=1e-12)
        np.testing.assert_allclose(rs@rs.transpose(0,2,1),np.broadcast_to(np.eye(3),rs.shape),atol=1e-12)

    def test_world_center_and_translation_convention(self):
        c=np.array([[2.,1.,3.],[3.,2.,4.]])
        r=Rotation.from_euler('xyz',[[10,20,30],[35,50,70]],degrees=True).as_matrix()
        cs,rs,_,_,_=interpolate(c,r,40)
        for center,c2w in zip(cs,rs):
            t=-c2w.T@center
            np.testing.assert_allclose(-np.linalg.inv(c2w.T)@t,center,atol=1e-12)
            np.testing.assert_allclose(c2w.T@center+t,0,atol=1e-12)

    def test_short_rotation_across_180_degrees(self):
        r=Rotation.from_euler('z',[179,-179],degrees=True).as_matrix()
        c=np.array([[0.,0.,0.],[1.,0.,0.]])
        _,rs,_,_,_=interpolate(c,r,20)
        angle=(Rotation.from_matrix(rs[:-1]).inv()*Rotation.from_matrix(rs[1:])).magnitude()
        self.assertLess(np.rad2deg(angle).max(),1)

    def test_order_is_deterministic_and_preserves_all_anchors(self):
        c=np.array([[0.,0.,0.],[5.,0.,0.],[1.,0.,0.],[4.,0.,0.],[2.,0.,0.]])
        r=np.tile(np.eye(3),(len(c),1,1))
        order=continuous_order(c,r)
        self.assertEqual(sorted(order.tolist()),list(range(len(c))))
        np.testing.assert_array_equal(order,continuous_order(c,r))
        self.assertLess(np.linalg.norm(np.diff(c[order],axis=0),axis=1).sum(),np.linalg.norm(np.diff(c,axis=0),axis=1).sum())

    def test_minimum_budget_enforces_continuity(self):
        c=np.array([[0.,0.,0.],[1.,0.,0.]])
        r=Rotation.from_euler('y',[0,170],degrees=True).as_matrix()
        n=minimum_frame_count(c,r)
        with self.assertRaises(ValueError):interpolate(c,r,n-1)
        cs,rs,_,_,radius=interpolate(c,r,n)
        angle=(Rotation.from_matrix(rs[:-1]).inv()*Rotation.from_matrix(rs[1:])).magnitude()
        self.assertLessEqual(np.rad2deg(angle).max(),15+1e-10)
        self.assertLessEqual(np.linalg.norm(np.diff(cs,axis=0),axis=1).max()/radius,.2+1e-10)


if __name__=='__main__':unittest.main()
