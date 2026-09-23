from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.init_configs import _atomic_write_text, _validate_scene_key, setup_configs
from tools.prepare_removal_workspace import ArtifactError, prepare_workspace
from tools.publish_removed_edgs_model import publish_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _write_gaussian_ply(path: Path, *, embedding_dimensions: int = 16) -> None:
    properties = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    properties.extend(f"obj_dc_{index}" for index in range(embedding_dimensions))
    lines = ["ply", "format ascii 1.0", "element vertex 1"]
    lines.extend(f"property float {name}" for name in properties)
    lines.extend(("end_header", " ".join("0" for _ in properties)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class RemovalArtifactFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dataset = root / "dataset"
        self.dataset.mkdir()
        self.bridge_root = root / "bridge"
        self.bridge_root.mkdir()
        self.edgs_root = root / "edgs"
        self.edgs_root.mkdir()
        self.config = self.edgs_root / "config.yaml"
        self.config.write_text("gs:\n  sh_degree: 0\n", encoding="utf-8")
        self.mesh = self.edgs_root / "mesh.ply"
        self.mesh.write_bytes(b"test-mesh")

        self.semantic_model = root / "semantic_3dgs"
        iteration_root = self.semantic_model / "point_cloud" / "iteration_2"
        self.gaussian = iteration_root / "point_cloud.ply"
        self.classifier = iteration_root / "classifier.pth"
        _write_gaussian_ply(self.gaussian)
        self.classifier.write_bytes(b"test-classifier")
        self.cfg_args = self.semantic_model / "cfg_args"
        self.cfg_args.write_text(
            "Namespace("
            "sh_degree=0, "
            f"source_path={str(self.dataset)!r}, "
            f"model_path={str(self.semantic_model)!r}, "
            f"vanilla_3dgs_path={str(self.bridge_root)!r}, "
            "num_classes=20)\n",
            encoding="utf-8",
        )

        self.bridge_manifest = self.bridge_root / "bridge_manifest.json"
        _write_json(
            self.bridge_manifest,
            {
                "schema_version": 1,
                "kind": "edgs-inpaint360gs-bridge",
                "complete": True,
                "artifact_id": "bridge-test-id",
                "dataset": {"source_path": str(self.dataset)},
                "bridge": {"root": str(self.bridge_root)},
                "edgs": {
                    "config_path": str(self.config),
                    "config_sha256": _sha256(self.config),
                    "mesh_path": str(self.mesh),
                },
            },
        )

        self.semantic_root = root / "semantic_mesh"
        self.semantic_manifest = self.semantic_root / "semantic_manifest.json"
        _write_json(
            self.semantic_manifest,
            {
                "schema_version": 1,
                "complete": True,
                "status": "complete",
                "inputs": {
                    "gaussian_ply": _record(self.gaussian),
                    "classifier": _record(self.classifier),
                    "mesh": _record(self.mesh),
                },
                "counts": {
                    "gaussians": 1,
                    "classes": 20,
                    "embedding_dimensions": 16,
                },
            },
        )


class SceneKeyValidationTest(unittest.TestCase):
    def test_nested_dataset_is_allowed_but_traversal_is_rejected(self) -> None:
        _validate_scene_key("mip-nerf/360_v2", "kitchen")
        for dataset_name in ("../escape", "/absolute", "bad//path", "bad/"):
            with self.subTest(dataset_name=dataset_name):
                with self.assertRaises(ValueError):
                    _validate_scene_key(dataset_name, "kitchen")
        for scene in ("../kitchen", "bad/scene", ".", ".."):
            with self.subTest(scene=scene):
                with self.assertRaises(ValueError):
                    _validate_scene_key("mip-nerf/360_v2", scene)


class AtomicConfigWriteTest(unittest.TestCase):
    def test_run_local_configs_recover_from_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "config"
            malformed = output / "object_removal" / "dataset" / "scene.json"
            malformed.parent.mkdir(parents=True)
            malformed.write_text('{"truncated":', encoding="utf-8")

            generated = setup_configs(
                "dataset",
                "scene",
                [14],
                [24],
                output_root=output,
                removal_thresh=0.7,
            )

            self.assertEqual(len(generated), 2)
            for path in generated:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["target_id"], [14])
                self.assertEqual(payload["surrounding_ids"], [24])
                self.assertEqual(payload["select_obj_id"], [14, 24])
                self.assertEqual(payload["removal_thresh"], 0.7)

    def test_atomic_writer_preserves_existing_file_if_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text("old", encoding="utf-8")
            previous_mode = path.stat().st_mode

            with mock.patch(
                "tools.init_configs.os.replace",
                side_effect=OSError("simulated interrupted commit"),
            ):
                with self.assertRaisesRegex(OSError, "interrupted commit"):
                    _atomic_write_text(path, "new")

            self.assertEqual(path.read_text(encoding="utf-8"), "old")
            self.assertEqual(path.stat().st_mode, previous_mode)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


class PrepareRemovalWorkspaceTest(unittest.TestCase):
    def test_create_refresh_and_reject_changed_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RemovalArtifactFixture(Path(temporary))
            output = fixture.root / "work_model"
            result = prepare_workspace(
                fixture.semantic_model,
                2,
                fixture.bridge_manifest,
                fixture.semantic_manifest,
                output,
            )
            self.assertEqual(result["action"], "create")
            self.assertTrue((output / "cfg_args").is_symlink())
            self.assertTrue((output / "point_cloud").is_symlink())
            self.assertFalse(os.path.isabs(os.readlink(output / "cfg_args")))
            self.assertFalse(os.path.isabs(os.readlink(output / "point_cloud")))
            manifest = json.loads((output / "workspace_manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            manifest_before = (output / "workspace_manifest.json").read_bytes()

            downstream = output / "point_cloud_object_removal"
            downstream.mkdir()
            (output / "cfg_args").unlink()
            refreshed = prepare_workspace(
                fixture.semantic_model,
                2,
                fixture.bridge_manifest,
                fixture.semantic_manifest,
                output,
            )
            self.assertEqual(refreshed["action"], "refresh")
            self.assertTrue(downstream.is_dir())
            self.assertTrue((output / "cfg_args").is_symlink())
            self.assertEqual(
                (output / "workspace_manifest.json").read_bytes(), manifest_before
            )

            bridge = json.loads(fixture.bridge_manifest.read_text())
            bridge["artifact_id"] = "different-bridge"
            _write_json(fixture.bridge_manifest, bridge)
            with self.assertRaisesRegex(
                ArtifactError, "different parameters or inputs"
            ):
                prepare_workspace(
                    fixture.semantic_model,
                    2,
                    fixture.bridge_manifest,
                    fixture.semantic_manifest,
                    output,
                )

    def test_recovers_managed_partial_links_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RemovalArtifactFixture(Path(temporary))
            output = fixture.root / "partial_workspace"
            output.mkdir()
            os.symlink(fixture.cfg_args, output / "cfg_args")

            result = prepare_workspace(
                fixture.semantic_model,
                2,
                fixture.bridge_manifest,
                fixture.semantic_manifest,
                output,
            )
            self.assertEqual(result["action"], "recover")
            self.assertTrue((output / "workspace_manifest.json").is_file())

            unsafe = fixture.root / "unsafe_workspace"
            unsafe.mkdir()
            (unsafe / "unmanaged.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "non-empty"):
                prepare_workspace(
                    fixture.semantic_model,
                    2,
                    fixture.bridge_manifest,
                    fixture.semantic_manifest,
                    unsafe,
                )


class PublishRemovedModelTest(unittest.TestCase):
    def test_create_refresh_and_reject_changed_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RemovalArtifactFixture(Path(temporary))
            output = fixture.root / "removed_3dgs"
            result = publish_model(
                fixture.gaussian,
                fixture.classifier,
                fixture.config,
                fixture.cfg_args,
                2,
                "4,3",
                "none",
                fixture.bridge_manifest,
                fixture.semantic_manifest,
                output,
                removal_threshold=0.7,
            )
            self.assertEqual(result["action"], "create")
            self.assertEqual(result["target_ids"], [3, 4])
            for path in (
                output / "config.yaml",
                output / "cfg_args",
                output / "point_cloud" / "iteration_2" / "point_cloud.ply",
                output / "point_cloud" / "iteration_2" / "classifier.pth",
            ):
                self.assertTrue(path.is_symlink())
                self.assertFalse(os.path.isabs(os.readlink(path)))
            manifest = json.loads((output / "model_manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["gaussian"]["point_count"], 1)
            manifest_before = (output / "model_manifest.json").read_bytes()

            refreshed = publish_model(
                fixture.gaussian,
                fixture.classifier,
                fixture.config,
                fixture.cfg_args,
                2,
                "3,4",
                "none",
                fixture.bridge_manifest,
                fixture.semantic_manifest,
                output,
                removal_threshold=0.7,
            )
            self.assertEqual(refreshed["action"], "refresh")
            self.assertEqual(
                (output / "model_manifest.json").read_bytes(), manifest_before
            )
            with self.assertRaisesRegex(
                ArtifactError, "different parameters or inputs"
            ):
                publish_model(
                    fixture.gaussian,
                    fixture.classifier,
                    fixture.config,
                    fixture.cfg_args,
                    2,
                    "3,4",
                    "none",
                    fixture.bridge_manifest,
                    fixture.semantic_manifest,
                    output,
                    removal_threshold=0.8,
                )
            with self.assertRaisesRegex(
                ArtifactError, "different parameters or inputs"
            ):
                publish_model(
                    fixture.gaussian,
                    fixture.classifier,
                    fixture.config,
                    fixture.cfg_args,
                    2,
                    "5",
                    "none",
                    fixture.bridge_manifest,
                    fixture.semantic_manifest,
                    output,
                    removal_threshold=0.7,
                )

    def test_rejects_missing_object_embedding_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RemovalArtifactFixture(Path(temporary))
            invalid = fixture.root / "invalid_removed.ply"
            _write_gaussian_ply(invalid, embedding_dimensions=15)
            with self.assertRaisesRegex(ArtifactError, "obj_dc_0..obj_dc_15"):
                publish_model(
                    invalid,
                    fixture.classifier,
                    fixture.config,
                    fixture.cfg_args,
                    2,
                    "3",
                    "none",
                    fixture.bridge_manifest,
                    fixture.semantic_manifest,
                    fixture.root / "invalid_output",
                    removal_threshold=0.7,
                )

    def test_recovers_managed_partial_model_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RemovalArtifactFixture(Path(temporary))
            output = fixture.root / "partial_model"
            iteration_root = output / "point_cloud" / "iteration_2"
            iteration_root.mkdir(parents=True)
            os.symlink(fixture.gaussian, iteration_root / "point_cloud.ply")

            result = publish_model(
                fixture.gaussian,
                fixture.classifier,
                fixture.config,
                fixture.cfg_args,
                2,
                "3",
                "none",
                fixture.bridge_manifest,
                fixture.semantic_manifest,
                output,
                removal_threshold=0.7,
            )
            self.assertEqual(result["action"], "recover")
            self.assertTrue((output / "model_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
