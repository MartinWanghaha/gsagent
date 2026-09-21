from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from virtual_render_io import (
    FrameWriter,
    check_backend,
    decode,
    read_json,
    sha256,
    validate_render,
    verify_tracking_render,
    write_json,
)


def package():
    return {
        "render": np.full((3, 2, 3), 0.25, np.float32),
        "plane_depth": np.full((1, 2, 3), 2, np.float32),
        "rendered_alpha": np.full((1, 2, 3), 0.5, np.float32),
        "rendered_normal": np.broadcast_to(
            np.array([0, 0, -0.5], np.float32)[:, None, None], (3, 2, 3)
        ).copy(),
    }


class GeometryContract(unittest.TestCase):
    def test_premultiplied_normal_and_validity(self):
        p = package()
        p["rendered_alpha"][0, 0, 0] = 0
        p["rendered_normal"][:, 0, 1] = 0
        p["rendered_normal"][:, 0, 2] = np.nan
        value = decode(p, "edgs-pgsr")
        np.testing.assert_array_equal(value["normal"][0], np.zeros((3, 3)))
        np.testing.assert_allclose(value["normal"][1], [[0, 0, -1]] * 3)
        self.assertEqual(value["normal"].dtype, np.float32)
        self.assertFalse(value["normal_valid"][0].any())

    def test_native_keeps_depth_definition_and_no_fake_normal(self):
        p = package()
        p["depth_3dgs"] = p["plane_depth"] * 0.3
        p["alpha"] = p["rendered_alpha"]
        value = decode(p, "inpaint360gs")
        np.testing.assert_array_equal(value["depth"], p["depth_3dgs"][0])
        self.assertNotIn("normal", value)

    def test_invalid_depth_is_zero_and_normal_invalid(self):
        p = package()
        p["plane_depth"][0, 0] = [np.nan, -1, np.inf]
        value = decode(p, "edgs-pgsr")
        np.testing.assert_array_equal(value["depth"][0], [0, 0, 0])
        self.assertFalse(value["normal_valid"][0].any())

    def test_shape_and_alpha_errors(self):
        p = package()
        p["rendered_alpha"] = np.ones((1, 1, 3), np.float32)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            decode(p, "edgs-pgsr")
        with self.assertRaisesRegex(ValueError, "alpha_min"):
            decode(package(), "edgs-pgsr", float("nan"))

    def test_writer_tamper_and_stale_frame_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cameras = {
                "artifact_id": "camera",
                "cameras": [
                    {"image_name": "00000", "image_height": 2, "image_width": 3}
                ],
            }
            writer = FrameWriter(root, "edgs-pgsr", cameras, {})
            self.assertFalse(read_json(root / "render_manifest.json")["complete"])
            p = package()
            p["rendered_alpha"][0, 0, 0] = 0
            writer.write(SimpleNamespace(image_name="00000"), p)
            writer.finish()
            np.testing.assert_array_equal(
                np.asarray(Image.open(root / "normal_vis/00000.png"))[0, 0], [0, 0, 0]
            )
            validate_render(root, "edgs-pgsr")
            np.save(root / "normal/99999.npy", np.zeros(3))
            with self.assertRaisesRegex(ValueError, "extra or missing"):
                validate_render(root)
            (root / "normal/99999.npy").unlink()
            np.save(root / "normal/00000.npy", np.zeros((2, 3, 3), np.float32))
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_render(root)

    def test_switch_refuses_legacy_and_locked_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ours_2000").mkdir()
            check_backend(root, "inpaint360gs")
            with self.assertRaisesRegex(ValueError, "legacy"):
                check_backend(root, "edgs-pgsr")
            write_json(root / "render_backend.json", {"backend": "edgs-pgsr"})
            check_backend(root, "edgs-pgsr")
            with self.assertRaisesRegex(ValueError, "changed"):
                check_backend(root, "inpaint360gs")

    def test_incomplete_frame_set_not_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = FrameWriter(
                Path(tmp),
                "edgs-pgsr",
                {
                    "artifact_id": "a",
                    "cameras": [
                        {"image_name": "00000", "image_height": 2, "image_width": 3}
                    ],
                },
                {},
            )
            with self.assertRaisesRegex(ValueError, "missing frames"):
                writer.finish()
            self.assertFalse(read_json(Path(tmp) / "render_manifest.json")["complete"])


@unittest.skipUnless(
    os.environ.get("PAINTMESH_GPU_SMOKE_MODEL"),
    "set PAINTMESH_GPU_SMOKE_MODEL to an existing removal work_model",
)
class GPUModelSmoke(unittest.TestCase):
    def assert_upright_hemisphere(self, generated, baseline):
        trajectory = generated.get("trajectory", {})
        if trajectory.get("type") != "hemisphere":
            return
        self.assertEqual(trajectory["orientation"], "scene_up")
        self.assertEqual(generated["cameras"][0], baseline["cameras"][0])
        transform = np.asarray(trajectory["world_to_pca"])
        for c in generated["cameras"]:
            w2c = np.eye(4)
            w2c[:3, :3], w2c[:3, 3] = np.asarray(c["R"]).T, c["T"]
            c2w = transform @ np.linalg.inv(w2c)
            axes = c2w[:3, :3]
            axes /= np.linalg.norm(axes, axis=0)
            right = np.cross(trajectory["scene_up_pca"], -axes[:, 2])
            right /= np.linalg.norm(right)
            np.testing.assert_allclose(axes[:, 0], right, atol=1e-9)

    def test_pose_generation_only(self):
        source = Path(os.environ["PAINTMESH_GPU_SMOKE_MODEL"]).resolve()
        cameras = read_json(source.parent / "tracker/virtual_cameras.json")
        with tempfile.TemporaryDirectory(prefix="paintmesh-pose-smoke-") as tmp:
            root = Path(tmp)
            model = root / "work_model"
            model.mkdir()
            for name in ("cfg_args", "point_cloud"):
                (model / name).symlink_to(source / name)
            configs = root / "config"
            shutil.copytree(source.parent / "config", configs)
            removal_config = next((configs / "object_removal").rglob("*.json"))
            project = SCRIPTS.parents[1] / "submodules/Inpaint360GS"
            camera_path = root / "virtual_cameras.json"
            subprocess.run(
                [
                    sys.executable,
                    str(project / "tools/virtual_pose.py"),
                    "--source_path",
                    os.environ["PAINTMESH_GPU_SMOKE_SCENE"],
                    "--model_path",
                    str(model),
                    "--iteration",
                    str(cameras["iteration"]),
                    "--resolution",
                    os.environ.get("PAINTMESH_GPU_SMOKE_RESOLUTION", "8"),
                    "--config_file",
                    str(removal_config),
                    "--camera_manifest",
                    str(camera_path),
                    "--tracker_archive",
                    str(root / "images.zip"),
                    "--poses-only",
                    "--camera-path", os.environ.get("PAINTMESH_GPU_SMOKE_CAMERA_PATH", "circle"),
                    "--camera-count", os.environ.get("PAINTMESH_GPU_SMOKE_CAMERA_COUNT", "30"),
                ],
                cwd=project,
                env=dict(os.environ, PYTHONPATH=str(project)),
                check=True,
            )
            generated = read_json(camera_path)
            self.assert_upright_hemisphere(generated, cameras)
            self.assertEqual(generated["frame_count"], int(os.environ.get("PAINTMESH_GPU_SMOKE_CAMERA_COUNT", "30")))
            self.assertTrue(generated["complete"])
            self.assertFalse((root / "images.zip").exists())
            self.assertFalse((model / "virtual").exists())
            self.assertTrue((root / "camera_trajectory.svg").is_file())
            if generated.get("trajectory", {}).get("type") == "hemisphere":
                stats = read_json(root / "camera_trajectory.json")
                self.assertLess(stats["scene_up_roll"]["max_abs_deg"], 1e-8)
                self.assertEqual(stats["scene_up_roll"]["undefined_frames"], 0)

    def test_both_backends_on_same_real_cameras(self):
        source = Path(os.environ["PAINTMESH_GPU_SMOKE_MODEL"]).resolve()
        camera_path = source.parent / "tracker/virtual_cameras.json"
        cameras = read_json(camera_path)
        iteration = cameras["iteration"]
        scene = os.environ["PAINTMESH_GPU_SMOKE_SCENE"]
        edgs = os.environ["PAINTMESH_GPU_SMOKE_EDGS"]
        resolution = os.environ.get("PAINTMESH_GPU_SMOKE_RESOLUTION", "8")
        with tempfile.TemporaryDirectory(prefix="paintmesh-virtual-smoke-") as tmp:
            if os.environ.get("PAINTMESH_GPU_SMOKE_CAMERA_PATH"):
                root = Path(tmp)
                work = root / "camera-work"
                work.mkdir()
                for name in ("cfg_args", "point_cloud"):
                    (work / name).symlink_to(source / name)
                shutil.copytree(source.parent / "config", root / "config")
                config = next((root / "config/object_removal").rglob("*.json"))
                project = SCRIPTS.parents[1] / "submodules/Inpaint360GS"
                camera_path = root / "virtual_cameras.json"
                subprocess.run([sys.executable, str(project / "tools/virtual_pose.py"),
                    "--source_path", scene, "--model_path", str(work), "--iteration", str(iteration),
                    "--resolution", resolution, "--config_file", str(config),
                    "--camera_manifest", str(camera_path), "--poses-only",
                    "--camera-path", os.environ["PAINTMESH_GPU_SMOKE_CAMERA_PATH"],
                    "--camera-count", os.environ.get("PAINTMESH_GPU_SMOKE_CAMERA_COUNT", "30")],
                    cwd=project, env=dict(os.environ, PYTHONPATH=str(project)), check=True)
                generated = read_json(camera_path)
                self.assert_upright_hemisphere(generated, cameras)
                cameras = generated
            for backend in ("inpaint360gs", "edgs-pgsr"):
                with self.subTest(backend=backend):
                    model = Path(tmp) / backend
                    model.mkdir()
                    for name in (
                        "cfg_args",
                        "point_cloud",
                        "point_cloud_object_removal",
                    ):
                        (model / name).symlink_to(source / name)
                    argv = [
                        "--backend",
                        backend,
                        "--model-path",
                        str(model),
                        "--iteration",
                        str(iteration),
                        "--edgs-model-path",
                        edgs,
                        "--source-path",
                        scene,
                        "--resolution",
                        resolution,
                        "--camera-manifest",
                        str(camera_path),
                        "--tracker-archive",
                        str(model / "images.zip"),
                    ]
                    subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPTS / "render_virtual_views.py"),
                            *argv,
                        ],
                        check=True,
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPTS / "render_virtual_views.py"),
                            *argv,
                            "--validate-only",
                        ],
                        check=True,
                    )
                    removed = (
                        model / "virtual/ours_object_removal" / f"iteration_{iteration}"
                    )
                    value = validate_render(removed, backend)
                    self.assertEqual(len(value["frames"]), cameras["frame_count"])
                    if backend == "edgs-pgsr":
                        for index in sorted({0, min(10, cameras["frame_count"]-1),
                                             min(23, cameras["frame_count"]-1), cameras["frame_count"]-1}):
                            name = f"{index:05d}"
                            n = np.load(removed / f"normal/{name}.npy")
                            valid = np.asarray(Image.open(removed / f"normal_valid/{name}.png")) > 0
                            self.assertTrue(valid.any())
                            self.assertTrue(np.isfinite(n).all())
                            np.testing.assert_allclose(np.linalg.norm(n[valid], axis=-1), 1, atol=1e-5)
                    session = {
                        "input_virtual_render": {
                            "path": str(removed / "render_manifest.json"),
                            "sha256": sha256(removed / "render_manifest.json"),
                            "backend": backend,
                            "artifact_id": value["artifact_id"],
                        }
                    }
                    verify_tracking_render(session)


if __name__ == "__main__":
    unittest.main()
