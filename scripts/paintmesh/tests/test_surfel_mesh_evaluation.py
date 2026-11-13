from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate_surfel_mesh_experiment as evaluator
from evaluate_normal_mesh_experiment import render_specs,output_files


class SurfelEvaluationTests(unittest.TestCase):
    def test_actual_colored_triangles_compose_by_positive_nearest_depth(self):
        camera=dict(name='tiny.JPG',stem='tiny',width=3,height=3,K=np.array([[1.,0,1.5],[0,1.,1.5],[0,0,1.]]).tolist(),w2c=np.eye(4).tolist())
        def mesh(path,z,rgb,size):
            rows=[f'{-size} {-size} {z} {rgb[0]} {rgb[1]} {rgb[2]}',
                  f'{size} {-size} {z} {rgb[0]} {rgb[1]} {rgb[2]}',
                  f'0 {size} {z} {rgb[0]} {rgb[1]} {rgb[2]}']
            path.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n'+'\n'.join(rows)+'\n3 0 1 2\n')
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);background=out/'bg.ply';objects=out/'obj.ply'
            mesh(background,2,(255,0,0),4);mesh(objects,1,(0,255,0),.5)
            render_specs(out/'background',[dict(path=str(background))],[camera])
            render_specs(out/'objects',[dict(path=str(objects))],[camera])
            marker=evaluator.combine_render_folders(out/'background',out/'objects',out/'composed',[camera],dict(test='fixed pose'))
            depth=np.load(out/'composed/tiny_depth.npy');rgb=np.asarray(Image.open(out/'composed/tiny.png'))
            self.assertTrue(marker['complete']);self.assertAlmostEqual(float(depth[1,1]),1.)
            np.testing.assert_array_equal(rgb[1,1],[0,255,0])
            self.assertEqual(marker['outputs'],output_files(out/'composed',[camera]))
            self.assertAlmostEqual(float(depth[0,0]),2.)
            np.testing.assert_array_equal(rgb[0,0],[255,0,0])
            # A corrupt cached RGB must be regenerated, never accepted by name.
            Image.fromarray(np.zeros((3,3,3),dtype='uint8')).save(out/'composed/tiny.png')
            evaluator.combine_render_folders(out/'background',out/'objects',out/'composed',[camera],dict(test='fixed pose'))
            np.testing.assert_array_equal(np.asarray(Image.open(out/'composed/tiny.png'))[1,1],[0,255,0])

    def test_shared_surface_mask_and_exact_camera_identity(self):
        foreground=np.array([[True,False,False],[False,False,False]])
        a=np.ones((2,3));b=np.array([[1,np.nan,1],[0,np.inf,1]])
        eligible,shared=evaluator.covered_support([a,b],foreground,True)
        np.testing.assert_array_equal(shared,[[False,False,True],[False,False,True]])
        self.assertEqual(int(eligible.sum()),5)
        self.assertTrue(evaluator.covered_support([a,b],foreground,False)[1][0,0])
        with self.assertRaises(ValueError):evaluator.covered_support([a[:1]],foreground,False)
        camera=dict(name='one.JPG',stem='one',width=3,height=2,K=np.eye(3).tolist(),w2c=np.eye(4).tolist())
        same=json.loads(json.dumps(camera));evaluator.same_cameras([same],[camera])
        same['w2c'][0][3]=1e-8
        with self.assertRaises(ValueError):evaluator.same_cameras([same],[camera])


if __name__=='__main__':unittest.main()
