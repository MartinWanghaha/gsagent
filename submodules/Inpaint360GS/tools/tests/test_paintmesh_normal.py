from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from tools import paintmesh_normal as normal
from tools.prepare_paintmesh_lama_data import (
    LamaDataError, _artifact, _atomic_json, _sha256,
    prepare_lama_inputs, validate_lama_outputs,
)
from tools.tests.test_paintmesh_lama_data import LamaFixture

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
from virtual_render_io import FrameWriter


class NormalFixture(LamaFixture):
    def __init__(self, root, frames=2, shape=(8, 9)):
        super().__init__(root, frames)
        self.render_root = root / "removed"
        self.render_root.mkdir()
        self.rgb.rename(self.render_root / "renders")
        self.depth.rename(self.render_root / "depth")
        self.rgb = self.render_root / "renders"
        self.depth = self.render_root / "depth"
        self.normal_input = self.depth_input.parent / "normal"
        self.normal_output = self.depth_output.parent / "normal"
        h, w = shape
        self.camera_path = root / "cameras.json"
        records = [dict(image_name=f"{i:05d}", image_height=h, image_width=w, FoVx=1.1, FoVy=0.9,
                        R=np.eye(3).tolist(), T=[0., 0., 0.], znear=0.01, zfar=100., trans=[0., 0., 0.], scale=1.) for i in range(frames)]
        cameras = dict(schema_version=1, kind="inpaint360gs-virtual-cameras", frame_count=frames, iteration=2000, circle_radius=1., cameras=records)
        cameras.update(artifact_id=normal.identity(cameras), complete=True, status="complete")
        _atomic_json(self.camera_path, cameras)
        self.cameras = {c["image_name"]: c for c in records}
        writer = FrameWriter(self.render_root, "edgs-pgsr", cameras, {})
        yy, xx = np.mgrid[:h, :w]
        for i in range(frames):
            stem = f"{i:05d}"
            labels = np.zeros(shape, np.uint8)
            rh, rw = max(1, h // 8), max(1, w // 8)
            labels[h//2-rh:h//2+rh, w//2-rw:w//2+rw] = 255
            Image.fromarray(labels).save(self.masks / f"{stem}.png")
            np.save(self.reference / f"{stem}.npy", (2 + yy * .05 + xx * .1).astype(np.float32))
            alpha = np.ones((1, h, w), np.float32)
            alpha[0, labels != 0] = 0
            alpha[0, 0, 0] = 0  # invalid outside the hole must remain invalid
            normals = np.zeros((3, h, w), np.float32)
            normals[2] = -alpha[0]
            writer.write(SimpleNamespace(image_name=stem), {
                "render": np.full((3, h, w), .3, np.float32),
                "plane_depth": (1 + yy * .05 + xx * .1).astype(np.float32)[None],
                "rendered_normal": normals, "rendered_alpha": alpha,
            })
        writer.finish()

    def prepare(self):
        return prepare_lama_inputs(self.masks, self.rgb, self.depth, self.reference,
                                   self.color_input, self.depth_input, self.input_manifest,
                                   frames=self.frames, min_area=3, dilation=1,
                                   camera_manifest=self.camera_path)

    def make_normal_outputs(self):
        frames = {}
        for i in range(self.frames):
            stem = f"{i:05d}"
            source, valid = normal.read_normal(self.normal_input / f"{stem}.npy", self.normal_input / "valid" / f"{stem}.png")
            hole = normal.read_mask(self.normal_input / f"{stem}_mask.png")
            prediction = np.broadcast_to(np.array([.5, .5, 1], np.float32), source.shape).copy()
            completed, completed_valid = normal.compose_normal(source, valid, hole, prediction, self.cameras[stem])
            for directory in (self.normal_output, self.normal_output / "valid", self.normal_output / "vis"):
                directory.mkdir(parents=True, exist_ok=True)
            np.save(self.normal_output / f"{stem}.npy", completed)
            Image.fromarray(completed_valid.astype(np.uint8) * 255).save(self.normal_output / "valid" / f"{stem}.png")
            Image.fromarray(normal.normal_preview(completed, completed_valid)).save(self.normal_output / "vis" / f"{stem}.png")
            frames[stem] = dict(normal=_artifact(self.normal_output / f"{stem}.npy"),
                                normal_valid=_artifact(self.normal_output / "valid" / f"{stem}.png"),
                                normal_vis=_artifact(self.normal_output / "vis" / f"{stem}.png"))
        inputs = json.loads(self.input_manifest.read_text())
        receipt = dict(kind="paintmesh-normal-prediction", schema_version=1, complete=True, status="complete",
                       input_artifact_id=inputs["artifact_id"], input_sha256=_sha256(self.input_manifest),
                       model=dict(config=_sha256(self.model / "config.yaml"), checkpoint=_sha256(self.model / "models/best.ckpt")),
                       encoding=normal.ENCODING, method=normal.METHOD, refine=True,
                       prediction_config_sha256=_sha256(REPO / "submodules/Inpaint360GS/LaMa/configs/prediction/default.yaml"), frames=frames)
        receipt["artifact_id"] = normal.identity(receipt)
        _atomic_json(self.normal_output / "prediction.json", receipt)

    def validate(self):
        return validate_lama_outputs(self.color_input, self.depth_input, self.color_output, self.depth_output,
                                     self.model, self.input_manifest, self.completion_manifest, frames=self.frames)


class NormalPipelineTests(unittest.TestCase):
    def test_automatic_inputs_common_mask_padding_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp))
            p = f.prepare()
            self.assertEqual(p["parameters"]["required_modalities"], ["rgb", "depth", "normal"])
            self.assertEqual(p["artifact_id"], f.prepare()["artifact_id"])
            np.testing.assert_array_equal(normal.read_mask(f.normal_input / "00000_mask.png"), normal.read_mask(f.depth_input / "00000_mask.png"))
            dataset = normal.NormalInpaintingDataset(f.normal_input, ["00000"])
            batch = dataset[0]
            self.assertEqual(batch["image"].shape, (3, 8, 16))
            self.assertEqual(batch["unpad_to_size"], (8, 9))
            np.testing.assert_array_equal(batch["image"][:, 0, 1], [.5, .5, 0])
            self.assertEqual(batch["mask"][0, 0, 0], 1)
            self.assertFalse(normal.read_mask(f.normal_input / "00000_mask.png")[0, 0])

    def test_completed_normals_are_verified_and_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp))
            f.prepare(); f.make_valid_outputs(); f.make_normal_outputs()
            result = f.validate()
            self.assertEqual(result["artifact_id"], f.validate()["artifact_id"])
            self.assertIn("normal", result["frames"]["00000"]["outputs"])
            self.assertEqual(result["frames"]["00000"]["normal_hole_valid_fraction"], 1)

    def test_workspace_symlink_layout_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp))
            linked = Path(tmp) / "workspace/virtual/ours_object_removal/iteration_2000"
            linked.mkdir(parents=True)
            for path in f.render_root.iterdir():
                (linked / path.name).symlink_to(path, target_is_directory=path.is_dir())
            f.rgb = linked / "renders"; f.depth = linked / "depth"
            self.assertIn("normal_input", f.prepare()["roots"])

    def test_publisher_validates_normal_chain_and_rejects_tampering(self):
        from tools.publish_inpainted_edgs_model import _validate_lama_chain, ArtifactError
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp), frames=30)
            f.prepare(); f.make_valid_outputs(); f.make_normal_outputs()
            result = f.validate()
            _validate_lama_chain(f.completion_manifest, result)
            path = f.normal_output / "00000.npy"
            value = np.load(path); value[0, 1] = [1, 0, 0]; np.save(path, value)
            with self.assertRaises(ArtifactError):
                _validate_lama_chain(f.completion_manifest, result)

    def test_missing_normal_cannot_publish_rgb_depth_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp))
            f.prepare(); f.make_valid_outputs()
            with self.assertRaises((ValueError, OSError, LamaDataError)):
                f.validate()
            self.assertFalse(f.completion_manifest.exists())

    def test_changed_normal_outside_hole_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp))
            f.prepare(); f.make_valid_outputs(); f.make_normal_outputs()
            path = f.normal_output / "00000.npy"
            value = np.load(path); value[0, 1] = [1, 0, 0]; np.save(path, value)
            with self.assertRaisesRegex(LamaDataError, "outside the mask"):
                f.validate()

    def test_missing_or_modified_render_and_camera_rejected(self):
        for kind in ("missing", "modified", "camera", "declaration"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                f = NormalFixture(Path(tmp))
                if kind == "missing":
                    (f.render_root / "normal" / "00001.npy").unlink()
                elif kind == "modified":
                    np.save(f.render_root / "normal" / "00000.npy", np.zeros((8, 9, 3), np.float32))
                elif kind == "camera":
                    p = json.loads(f.camera_path.read_text()); p["cameras"][0]["FoVx"] = .8; _atomic_json(f.camera_path, p)
                else:
                    p = json.loads((f.render_root / "render_manifest.json").read_text()); p["capabilities"].remove("normal"); _atomic_json(f.render_root / "render_manifest.json", p)
                with self.assertRaises((ValueError, OSError, LamaDataError)):
                    f.prepare()
                self.assertFalse(f.input_manifest.exists())

    def test_no_manifest_normal_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp))
            (f.render_root / "render_manifest.json").unlink()
            with self.assertRaisesRegex(LamaDataError, "requires a verified"):
                f.prepare()

    def test_model_change_invalidates_normal_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp))
            f.prepare(); f.make_valid_outputs(); f.make_normal_outputs()
            (f.model / "models/best.ckpt").write_bytes(b"another checkpoint")
            with self.assertRaisesRegex(LamaDataError, "provenance mismatch"):
                f.validate()

    def test_legacy_rgb_depth_cache_cannot_be_upgraded_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp))
            p = f.prepare(); f.make_valid_outputs(); f.make_normal_outputs()
            completion = f.validate()
            completion["artifact_id"] = "0" * 64  # old/different completion identity
            _atomic_json(f.completion_manifest, completion)
            with self.assertRaisesRegex(LamaDataError, "different outputs"):
                f.validate()
            p["parameters"].pop("required_modalities")
            _atomic_json(f.input_manifest, p)
            with self.assertRaisesRegex(LamaDataError, "absent from LaMa input modalities"):
                f.validate()

    def test_changed_inference_mask_and_partial_normal_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp))
            f.prepare(); f.make_valid_outputs(); f.make_normal_outputs()
            (f.normal_output / "valid" / "00001.png").unlink()
            with self.assertRaisesRegex(LamaDataError, "exactly 2 canonical frames"):
                f.validate()
            Image.fromarray(np.ones((8, 9), np.uint8) * 255).save(f.normal_input / "inference_mask" / "00000.png")
            with self.assertRaisesRegex(LamaDataError, "changed"):
                f.prepare()

    def test_symlink_output_directory_rejected_before_inference(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("normal_predictor_test", REPO / "submodules/Inpaint360GS/LaMa/bin/predict_normal.py")
        module = module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            f = NormalFixture(Path(tmp)); f.prepare()
            f.normal_output.parent.mkdir(parents=True)
            f.normal_output.symlink_to(f.normal_input, target_is_directory=True)
            with self.assertRaises((LamaDataError, ValueError)):
                module.predict(SimpleNamespace(input_dir=f.normal_input, output_dir=f.normal_output, input_manifest=f.input_manifest, model_path=f.model))

    def test_degenerate_nonfinite_and_direction(self):
        source = np.zeros((3, 5, 3), np.float32); source[..., 2] = -1
        valid = np.ones((3, 5), bool)
        hole = np.zeros((3, 5), bool); hole[1, :] = True
        prediction = np.full_like(source, .5)
        prediction[1, 0] = [1, .5, 1]
        prediction[1, 1] = np.nan
        prediction[1, 2] = np.inf
        camera = dict(image_height=3, image_width=5, FoVx=1.1, FoVy=1.)
        result, mask = normal.compose_normal(source, valid, hole, prediction, camera)
        self.assertTrue(mask[1, 0]); self.assertFalse(mask[1, 1:].any())
        np.testing.assert_array_equal(result[~hole], source[~hole])
        np.testing.assert_array_equal(result[1, 1:], 0)
        self.assertLessEqual(np.dot(result[1, 0], normal.camera_rays(camera, hole.shape)[1, 0]), 0)
        with self.assertRaisesRegex(ValueError, "no valid normals"):
            normal.compose_normal(source, valid, hole, np.full_like(source, .5), camera)
        with self.assertRaisesRegex(ValueError, "no valid context"):
            normal.compose_normal(source, hole, hole, prediction, camera)

    def test_known_normal_must_be_float32_unit_or_zero(self):
        for invalid in (np.nan, 2.0):
            with self.subTest(value=invalid), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                n = np.zeros((2, 3, 3), np.float32); n[..., 2] = invalid
                np.save(path / "n.npy", n); Image.fromarray(np.full((2, 3), 255, np.uint8)).save(path / "v.png")
                with self.assertRaises(ValueError):
                    normal.read_normal(path / "n.npy", path / "v.png")


@unittest.skipUnless(os.environ.get("PAINTMESH_NORMAL_GPU_MODEL"), "opt-in real LaMa GPU smoke")
class NormalGpuSmoke(unittest.TestCase):
    def test_real_lama_predictor_and_completion_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            shape = tuple(map(int, os.environ.get("PAINTMESH_NORMAL_GPU_SHAPE", "64x81").split("x")))
            f = NormalFixture(Path(tmp), frames=int(os.environ.get("PAINTMESH_NORMAL_GPU_FRAMES", "1")), shape=shape)
            f.model = Path(os.environ["PAINTMESH_NORMAL_GPU_MODEL"])
            f.prepare(); f.make_valid_outputs()
            subprocess.run([
                os.environ.get("PAINTMESH_LAMA_PYTHON", sys.executable),
                str(REPO / "submodules/Inpaint360GS/LaMa/bin/predict_normal.py"),
                "--input-dir", str(f.normal_input), "--output-dir", str(f.normal_output),
                "--input-manifest", str(f.input_manifest), "--model-path", str(f.model),
            ], check=True)
            self.assertTrue(f.validate()["complete"])


if __name__ == "__main__":
    unittest.main()
