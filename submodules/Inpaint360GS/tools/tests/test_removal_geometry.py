from __future__ import annotations

import unittest

import torch

from edit_object_removal import points_inside_convex_hull
from scene.gaussian_model import GaussianModel


class ConvexHullFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.1, 0.1],
            ]
        )

    def test_tetrahedron_includes_interior_point(self) -> None:
        mask = torch.tensor([True, True, True, True, False])
        inside, radius = points_inside_convex_hull(
            self.points,
            mask,
            remove_outliers=False,
        )
        self.assertEqual(inside.tolist(), [True, True, True, True, True])
        self.assertGreater(radius, 0.0)

    def test_degenerate_selection_preserves_probability_mask(self) -> None:
        mask = torch.tensor([True, True, True, False, False])
        inside, _ = points_inside_convex_hull(
            self.points,
            mask,
            remove_outliers=False,
        )
        self.assertTrue(torch.equal(inside, mask))

    def test_empty_selection_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty mask"):
            points_inside_convex_hull(
                self.points,
                torch.zeros(len(self.points), dtype=torch.bool),
            )


class ExclusiveObjectPartitionTest(unittest.TestCase):
    def test_target_owns_overlap_with_surrounding_object(self) -> None:
        model = GaussianModel(0)
        count = 5
        model._xyz = torch.nn.Parameter(
            torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
        )
        model._features_dc = torch.nn.Parameter(torch.zeros(count, 1, 3))
        model._features_rest = torch.nn.Parameter(torch.zeros(count, 0, 3))
        model._opacity = torch.nn.Parameter(torch.zeros(count, 1))
        model._scaling = torch.nn.Parameter(torch.zeros(count, 3))
        model._rotation = torch.nn.Parameter(torch.zeros(count, 4))
        model._objects_dc = torch.nn.Parameter(torch.zeros(count, 1, 16))

        masks = {
            14: {
                "mask3d": torch.tensor(
                    [True, True, False, False, False]
                )[:, None, None]
            },
            24: {
                "mask3d": torch.tensor(
                    [False, True, True, False, False]
                )[:, None, None]
            },
        }
        objects = model.removal_setup(masks)

        self.assertEqual(len(objects[14]._xyz), 2)
        self.assertEqual(len(objects[24]._xyz), 1)
        self.assertEqual(len(model._xyz), 2)
        partition = torch.cat((objects[14]._xyz, objects[24]._xyz, model._xyz))
        self.assertEqual(len(torch.unique(partition[:, 0])), count)


if __name__ == "__main__":
    unittest.main()
