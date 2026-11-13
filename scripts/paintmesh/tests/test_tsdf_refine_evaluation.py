from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_tsdf_refine_experiment as evaluator
from evaluate_normal_mesh_experiment import render_specs


class TsdfRefineEvaluationTests(unittest.TestCase):
    def test_slab_parallel_inside_outside_and_existing_occluder(self):
        # Rays parallel to the x/y slabs must remain valid when on the box.
        rr = np.array([[0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 1], [0, 0, 4, 0, 0, 1],
            [0, 0, 4, 0, 0, -1]], dtype='float32')
        box = [[-1, -1, 2], [1, 1, 3]]
        np.testing.assert_array_equal(evaluator.possible_ray_indices(rr, box), [0, 4])
        np.testing.assert_array_equal(evaluator.possible_ray_indices(rr, box,
            np.array([1, 0, 0, 0, 0])), [4])
        # A same-depth boundary hit is kept, protecting tie and rounding cases.
        self.assertIn(0, evaluator.possible_ray_indices(rr, box, np.array([2, 0, 0, 0, 0])))

    def test_accelerated_render_matches_all_rays_for_colored_triangle_union(self):
        camera = dict(name='tiny.JPG', stem='tiny', width=11, height=9,
            K=[[7., 0, 5.5], [0, 7., 4.5], [0, 0, 1]], w2c=np.eye(4).tolist())
        def mesh(path, z, rgb, size, shift=0):
            from recompose.io import save_mesh
            v = np.array([[-size+shift, -size, z], [size+shift, -size, z], [shift, size, z]], dtype='float32')
            c = np.broadcast_to(np.array(rgb, dtype='float32') / 255, (3, 3)).copy()
            save_mesh(path, v, np.array([[0, 1, 2]], dtype='int32'), c)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            mesh(p/'red.ply', 3, [255, 0, 0], 3)
            mesh(p/'green.ply', 1.5, [0, 255, 0], .5)
            mesh(p/'offscreen.ply', 2, [0, 0, 255], .5, shift=30)
            specs = [dict(path=str(p/name)) for name in ['green.ply', 'red.ply', 'offscreen.ply']]
            render_specs(p/'reference', specs, [camera], checkpoint_parts=1)
            evaluator.render_streamed(p/'fast', specs, [camera], checkpoint_parts=1)
            for suffix in ['_depth.npy', '_normal.npy']:
                np.testing.assert_array_equal(np.load(p/'reference'/('tiny'+suffix)), np.load(p/'fast'/('tiny'+suffix)))
            np.testing.assert_array_equal(np.asarray(Image.open(p/'reference/tiny.png')),
                np.asarray(Image.open(p/'fast/tiny.png')))
            # Exact cache reuse should leave native output identity untouched.
            stamp = evaluator.identity(p/'fast/tiny.png')
            evaluator.render_streamed(p/'fast', specs, [camera], checkpoint_parts=1)
            self.assertEqual(stamp, evaluator.identity(p/'fast/tiny.png'))

    def test_support_excludes_target_and_nonpositive_nonfinite_depth(self):
        fg = np.array([[1, 0, 0], [0, 0, 0]], dtype=bool)
        a = np.ones((2, 3))
        b = np.array([[1, np.nan, 1], [0, np.inf, 1]])
        eligible, shared = evaluator.covered_support([a, b], fg, True)
        np.testing.assert_array_equal(shared, [[False, False, True], [False, False, True]])
        self.assertEqual(int(eligible.sum()), 5)
        self.assertTrue(evaluator.covered_support([a, b], fg, False)[1][0, 0])

    def test_depth_normals_are_real_geometry_and_do_not_fill_holes(self):
        c = dict(K=[[10., 0, 3], [0, 10., 3], [0, 0, 1]])
        d = np.ones((7, 7), dtype='float32') * 2
        image = evaluator.depth_normals(d, c)
        np.testing.assert_array_equal(image[3, 3], [128, 128, 0])
        d[3, 3] = 0
        image = evaluator.depth_normals(d, c)
        np.testing.assert_array_equal(image[3, 3], [0, 0, 0])
        np.testing.assert_array_equal(image[3, 2], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
