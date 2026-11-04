"""Geometry gates must reject occlusion and preserve camera conventions."""
import unittest
import numpy as np
from shared_reference_completion import correspondence, depth_valid


def cam(x=0.):
    c=np.eye(4);c[0,3]=x
    return dict(c2w=c,w2c=np.linalg.inv(c),k=np.array([[20.,0,15],[0,20.,15],[0,0,1.]]),h=32,w=32)


def frame(z):
    depth=np.full((32,32),z,dtype=float)
    return dict(depth=depth,valid=depth_valid(depth,.05))


class GeometryTests(unittest.TestCase):
    def test_plane_translated_camera_correspondence(self):
        uv=np.array([[10.,10.],[15.,15.],[20.,20.]])
        q,good=correspondence(uv,frame(2),frame(2),cam(),cam(.2))
        np.testing.assert_allclose(q,uv+[-2.,0.],atol=1e-10)
        self.assertTrue(good.all())

    def test_in_front_occluder_rejected(self):
        _,good=correspondence(np.array([[15.,15.]]),frame(2),frame(1),cam(),cam(.2))
        self.assertFalse(good.any())

    def test_depth_holes_and_discontinuity_rejected(self):
        d=np.ones((32,32));d[:,16:]=2;d[4,4]=0
        valid=depth_valid(d,.05)
        self.assertFalse(valid[4,4]);self.assertFalse(valid[10,15]);self.assertFalse(valid[10,16])
        self.assertTrue(valid[10,10]);self.assertTrue(valid[10,20])

    def test_out_of_frame_not_clamped_into_guide(self):
        _,good=correspondence(np.array([[1.,15.],[15.,15.]]),frame(2),frame(2),cam(),cam(.5))
        np.testing.assert_array_equal(good,[False,True])

    def test_cycle_rejects_small_depth_error_with_large_baseline(self):
        a=cam();b=cam(1.5);a['k'][0,0]=b['k'][0,0]=200
        a['w']=b['w']=512
        d=np.full((32,512),2.);e=d*1.019
        target=dict(depth=d,valid=depth_valid(d,.05));reference=dict(depth=e,valid=depth_valid(e,.05))
        _,good=correspondence(np.array([[200.,15.]]),target,reference,a,b)
        self.assertFalse(good.any())


if __name__=='__main__':unittest.main()
