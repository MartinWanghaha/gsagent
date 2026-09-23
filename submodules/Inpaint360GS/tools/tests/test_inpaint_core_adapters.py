"""Focused contracts for the run-local Inpaint360GS adapters."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import edit_object_inpaint as inpaint
import edit_object_removal_plyfusion as fusion
from tools.combine_gaussian_scene import _infer_removed_iteration, parse_object_ids
from utils.compose_utils import (
    _extra_feature_names,
    _load_rest_features,
    _sh_degree_from_args,
)
from utils.virtual_camera_manifest import (
    build_virtual_camera_manifest,
    load_virtual_camera_manifest,
    validate_virtual_camera_manifest,
    write_virtual_camera_manifest,
)


class _Camera:
    def __init__(self, index: int):
        self.image_name = f"{index:05d}"
        self.R = np.eye(3, dtype=np.float64)
        self.T = np.array([index / 100.0, 0.0, 1.0], dtype=np.float64)
        self.FoVx = 0.8
        self.FoVy = 0.6
        self.image_width = 32
        self.image_height = 24
        self.znear = 0.01
        self.zfar = 100.0
        self.trans = np.zeros(3, dtype=np.float64)
        self.scale = 1.0


class _PlyElement:
    def __init__(self, values):
        self._values = values
        self.properties = [SimpleNamespace(name=name) for name in values]

    def __getitem__(self, name):
        return self._values[name]


def _record(frame: str, key: str) -> dict:
    return {
        "frame": frame,
        key: {
            "path": f"/{key}/{frame}",
            "size_bytes": 10,
            "mtime_ns": 1,
            "sha256": "a" * 64,
        },
    }


class VirtualCameraManifestTests(unittest.TestCase):
    def test_round_trip_has_stable_completion_identity(self):
        views = [_Camera(index) for index in range(30)]
        payload = build_virtual_camera_manifest(
            views, iteration=2000, circle_radius=0.123456789012345
        )
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(len(payload["artifact_id"]), 64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            write_virtual_camera_manifest(
                path, views, iteration=2000, circle_radius=0.123456789012345
            )
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["artifact_id"], payload["artifact_id"])
            loaded = load_virtual_camera_manifest(
                path, expected_iteration=2000, expected_frame_count=30
            )
            self.assertEqual(loaded["circle_radius"], 0.123456789012345)
            self.assertEqual(loaded["cameras"][29]["image_name"], "00029")

            fixed_mtime_ns = 1_700_000_000_000_000_000
            os.utime(path, ns=(fixed_mtime_ns, fixed_mtime_ns))
            write_virtual_camera_manifest(
                path, views, iteration=2000, circle_radius=0.123456789012345
            )
            self.assertEqual(path.stat().st_mtime_ns, fixed_mtime_ns)

    def test_rejects_non_orthonormal_rotation(self):
        payload = build_virtual_camera_manifest(
            [_Camera(index) for index in range(30)],
            iteration=2000,
            circle_radius=0.1,
        )
        payload["cameras"][0]["R"][0][0] = 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_virtual_camera_manifest(payload)

    def test_uniformly_scaled_rotation_round_trips_without_normalization(self):
        angle = 0.37
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        expected = 0.22666207697214255 * rotation
        views = [_Camera(index) for index in range(30)]
        for view in views:
            view.R = expected.copy()

        payload = build_virtual_camera_manifest(
            views, iteration=2000, circle_radius=0.965473259059761
        )
        np.testing.assert_array_equal(payload["cameras"][0]["R"], expected)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scaled_cameras.json"
            write_virtual_camera_manifest(
                path,
                views,
                iteration=2000,
                circle_radius=0.965473259059761,
            )
            loaded = load_virtual_camera_manifest(path, expected_iteration=2000)
            np.testing.assert_array_equal(loaded["cameras"][0]["R"], expected)

    def test_rejects_anisotropic_scale(self):
        payload = build_virtual_camera_manifest(
            [_Camera(index) for index in range(30)],
            iteration=2000,
            circle_radius=0.1,
        )
        payload["cameras"][0]["R"] = np.diag([0.2, 0.25, 0.2]).tolist()
        with self.assertRaisesRegex(ValueError, "uniform positive scale"):
            validate_virtual_camera_manifest(payload)

    def test_rejects_shear(self):
        payload = build_virtual_camera_manifest(
            [_Camera(index) for index in range(30)],
            iteration=2000,
            circle_radius=0.1,
        )
        payload["cameras"][0]["R"] = [
            [0.2, 0.03, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ]
        with self.assertRaisesRegex(ValueError, "uniform positive scale"):
            validate_virtual_camera_manifest(payload)

    def test_rejects_reflection(self):
        payload = build_virtual_camera_manifest(
            [_Camera(index) for index in range(30)],
            iteration=2000,
            circle_radius=0.1,
        )
        payload["cameras"][0]["R"] = np.diag([0.2, 0.2, -0.2]).tolist()
        with self.assertRaisesRegex(ValueError, "determinant"):
            validate_virtual_camera_manifest(payload)

    def test_rejects_singular_scaled_rotation(self):
        payload = build_virtual_camera_manifest(
            [_Camera(index) for index in range(30)],
            iteration=2000,
            circle_radius=0.1,
        )
        payload["cameras"][0]["R"] = np.zeros((3, 3)).tolist()
        with self.assertRaisesRegex(ValueError, "singular"):
            validate_virtual_camera_manifest(payload)


class MaskAndInputContractTests(unittest.TestCase):
    def test_temporary_ply_defaults_survive_combined_args_dropping_none(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            args = SimpleNamespace(model_path=str(model), finetune_iteration=5000)

            temporary = inpaint._prepare_temporary_ply(args)

            output = (
                model
                / "point_cloud_object_inpaint_virtual"
                / "iteration_5000"
                / "point_cloud.ply"
            ).resolve()
            self.assertEqual(Path(args.inpaint_output_ply), output)
            self.assertEqual(Path(args.temp_ply), temporary)
            self.assertEqual(temporary.parent, output.parent)
            self.assertFalse(temporary.exists())

    def test_lama_mask_name_takes_precedence_over_rgb_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
            Image.fromarray(mask).save(root / "00000_mask.png")
            self.assertEqual(
                fusion._existing_mask(root, "00000").name, "00000_mask.png"
            )
            self.assertEqual(
                inpaint._find_virtual_mask(root, "00000").name, "00000_mask.png"
            )
            Image.fromarray(mask).save(root / "00000.png")
            self.assertEqual(
                fusion._existing_mask(root, "00000").name, "00000_mask.png"
            )
            self.assertEqual(
                inpaint._find_virtual_mask(root, "00000").name, "00000_mask.png"
            )

    def test_multiple_explicit_lama_masks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
            Image.fromarray(mask).save(root / "00000_mask.png")
            Image.fromarray(mask).save(root / "00000_mask.PNG")
            with self.assertRaisesRegex(ValueError, "multiple canonical"):
                fusion._existing_mask(root, "00000")
            with self.assertRaisesRegex(ValueError, "multiple canonical"):
                inpaint._find_virtual_mask(root, "00000")

    def test_16_bit_mask_labels_are_not_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "00000_mask.png"
            labels = np.array([[0, 300], [0, 0]], dtype=np.uint16)
            Image.fromarray(labels).save(path)
            fusion_mask = fusion._read_mask(path, (2, 2))
            inpaint_mask = inpaint._load_virtual_mask(path, (2, 2))
            self.assertTrue(fusion_mask[0, 1])
            self.assertEqual(int(inpaint_mask[0, 1]), 255)

    def test_depth_rejects_non_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.npy"
            np.save(path, np.array([[1.0, np.nan]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "finite numeric"):
                fusion._read_depth(path, (1, 2), "completed depth")

    def test_empty_bbox_is_safe(self):
        self.assertIsNone(inpaint.mask_to_bbox(torch.zeros((4, 4), dtype=torch.bool)))

    def test_object_ids_above_255_remain_valid(self):
        self.assertEqual(inpaint._selected_object_ids([14, 300], 512), [14, 300])
        with self.assertRaisesRegex(ValueError, r"\[0, 511\]"):
            inpaint._selected_object_ids([512], 512)


class FusionManifestTests(unittest.TestCase):
    def _records(self):
        frames = [f"{index:05d}" for index in range(30)]
        inputs = [_record(frame, "completed_rgb") for frame in frames]
        outputs = [_record(frame, "fused_mask_ply") for frame in frames]
        return inputs, outputs

    def test_matching_manifest_refreshes_but_identity_mismatch_is_refused(self):
        inputs, outputs = self._records()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fusion.json"
            first = fusion._write_fusion_manifest(
                path,
                iteration=2000,
                frame_count=30,
                circle_radius=0.2,
                write_hole_ply=False,
                camera_manifest=None,
                lama_manifest=None,
                inputs=inputs,
                outputs=outputs,
            )
            inputs[0]["completed_rgb"]["mtime_ns"] = 999
            second = fusion._write_fusion_manifest(
                path,
                iteration=2000,
                frame_count=30,
                circle_radius=0.2,
                write_hole_ply=False,
                camera_manifest=None,
                lama_manifest=None,
                inputs=inputs,
                outputs=outputs,
            )
            self.assertEqual(first["artifact_id"], second["artifact_id"])
            self.assertEqual(first["created_at"], second["created_at"])

            with self.assertRaisesRegex(ValueError, "INPAINT_RUN_NAME"):
                fusion._write_fusion_manifest(
                    path,
                    iteration=2001,
                    frame_count=30,
                    circle_radius=0.2,
                    write_hole_ply=False,
                    camera_manifest=None,
                    lama_manifest=None,
                    inputs=inputs,
                    outputs=outputs,
                )


class ModelCompatibilityTests(unittest.TestCase):
    def test_sh_degree_and_object_id_parsing(self):
        self.assertEqual(_sh_degree_from_args(SimpleNamespace(sh_degree=2)), 2)
        self.assertEqual(parse_object_ids("[11, 22,11]"), [11, 22])

    def test_rest_features_follow_source_sh_degree(self):
        coefficient_count = (2 + 1) ** 2 - 1
        values = {
            f"f_rest_{index}": np.arange(2, dtype=np.float32) + index
            for index in range(3 * coefficient_count)
        }
        plydata = SimpleNamespace(elements=[_PlyElement(values)])
        reference = {"features_rest": torch.zeros((5, coefficient_count, 3))}
        names = _extra_feature_names(plydata, reference)
        loaded = _load_rest_features(plydata, 2, names)
        self.assertEqual(loaded.shape, (2, 3, coefficient_count))

    def test_removal_iteration_is_not_hardcoded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "point_cloud_object_removal" / "iteration_731"
            root.mkdir(parents=True)
            (root / "point_cloud_11.ply").touch()
            self.assertEqual(_infer_removed_iteration(directory, [11]), 731)


if __name__ == "__main__":
    unittest.main()
