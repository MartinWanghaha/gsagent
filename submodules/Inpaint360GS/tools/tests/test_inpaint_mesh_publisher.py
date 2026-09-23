from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.prepare_removal_workspace import ArtifactError
from tools.publish_inpaint_mesh import publish_mesh


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class PublishInpaintMeshTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.gaussian = (
            self.model / "point_cloud" / "iteration_5000" / "point_cloud.ply"
        )
        _write(self.gaussian, "ply\nformat ascii 1.0\nend_header\n")
        self.model_manifest = self.model / "model_manifest.json"
        _write(
            self.model_manifest,
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "paintmesh-inpainted-edgs-model",
                    "complete": True,
                    "status": "complete",
                    "artifact_id": "a" * 64,
                    "parameters": {"output_iteration": 5000},
                    "model": {
                        "point_cloud": "point_cloud/iteration_5000/point_cloud.ply"
                    },
                }
            ),
        )
        self.mesh = self.model / "mesh" / "ours_5000" / "tsdf_fusion_post.ply"
        _write(
            self.mesh,
            "\n".join(
                (
                    "ply",
                    "format binary_little_endian 1.0",
                    "element vertex 3",
                    "property float x",
                    "property float y",
                    "property float z",
                    "element face 1",
                    "property list uchar int vertex_indices",
                    "end_header",
                )
            )
            + "\n",
        )
        self.train_manifest = (
            self.model / "train" / "ours_5000" / "render_manifest.json"
        )
        _write(
            self.train_manifest,
            json.dumps(
                {
                    "backend": "pgsr",
                    "complete": True,
                    "iteration": 5000,
                    "num_views": 1,
                    "split": "train",
                    "views": {"image": "image.png"},
                }
            ),
        )
        self.scene = self.root / "scene"
        self.scene.mkdir()
        self.output = self.mesh.parent / "mesh_manifest.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _publish(
        self,
        resolution: int = 8,
        *,
        test_render_manifest: Path | None = None,
    ):
        return publish_mesh(
            model_manifest_path=self.model_manifest,
            gaussian_ply=self.gaussian,
            mesh=self.mesh,
            train_render_manifest=self.train_manifest,
            test_render_manifest=test_render_manifest,
            output=self.output,
            iteration=5000,
            source_path=self.scene,
            images="images",
            resolution=resolution,
            max_depth=5.0,
            voxel_size=0.002,
            num_clusters=1,
            use_depth_filter=False,
        )

    def test_publish_and_reuse(self) -> None:
        first = self._publish()
        committed = self.output.read_bytes()
        fixed_mtime_ns = 1_700_000_000_000_000_000
        os.utime(self.output, ns=(fixed_mtime_ns, fixed_mtime_ns))
        second = self._publish()
        self.assertEqual(first["artifact_id"], second["artifact_id"])
        self.assertEqual(second["created_at"], first["created_at"])
        self.assertEqual(self.output.read_bytes(), committed)
        self.assertEqual(self.output.stat().st_mtime_ns, fixed_mtime_ns)
        self.assertEqual(first["kind"], "paintmesh-pgsr-inpaint-mesh")
        self.assertEqual(first["identity_version"], 2)
        self.assertEqual(first["counts"]["mesh_vertices"], 3)
        self.assertEqual(first["counts"]["mesh_triangles"], 1)
        self.assertEqual(first["inputs"]["model_manifest"]["artifact_id"], "a" * 64)

    def test_existing_identity_mismatch_is_rejected(self) -> None:
        self._publish()
        with self.assertRaisesRegex(ArtifactError, "different inputs/settings"):
            self._publish(resolution=4)

    def test_model_manifest_bytes_do_not_define_mesh_identity(self) -> None:
        first = self._publish()
        model = json.loads(self.model_manifest.read_text(encoding="utf-8"))
        model["refreshed_at"] = "later"
        _write(self.model_manifest, json.dumps(model))

        second = self._publish()
        self.assertEqual(second["artifact_id"], first["artifact_id"])
        self.assertEqual(second["created_at"], first["created_at"])

    def test_model_artifact_id_change_is_rejected(self) -> None:
        self._publish()
        model = json.loads(self.model_manifest.read_text(encoding="utf-8"))
        model["artifact_id"] = "b" * 64
        _write(self.model_manifest, json.dumps(model))

        with self.assertRaisesRegex(ArtifactError, "different inputs/settings"):
            self._publish()

    def test_gaussian_content_change_is_rejected(self) -> None:
        self._publish()
        _write(self.gaussian, "ply\nformat ascii 1.0\ncomment changed\nend_header\n")

        with self.assertRaisesRegex(ArtifactError, "different inputs/settings"):
            self._publish()

    def test_mesh_content_change_is_rejected(self) -> None:
        self._publish()
        with self.mesh.open("a", encoding="utf-8") as stream:
            stream.write("changed\n")

        with self.assertRaisesRegex(ArtifactError, "different inputs/settings"):
            self._publish()

    def test_train_render_manifest_content_change_is_rejected(self) -> None:
        self._publish()
        render = json.loads(self.train_manifest.read_text(encoding="utf-8"))
        render["refreshed_at"] = "later"
        _write(self.train_manifest, json.dumps(render))

        with self.assertRaisesRegex(ArtifactError, "different inputs/settings"):
            self._publish()

    def test_test_render_manifest_is_bound_to_identity(self) -> None:
        test_manifest = self.model / "test" / "ours_5000" / "render_manifest.json"
        _write(
            test_manifest,
            json.dumps(
                {
                    "backend": "pgsr",
                    "complete": True,
                    "iteration": 5000,
                    "num_views": 1,
                    "split": "test",
                    "views": {"image": "image.png"},
                }
            ),
        )
        self._publish(test_render_manifest=test_manifest)
        render = json.loads(test_manifest.read_text(encoding="utf-8"))
        render["refreshed_at"] = "later"
        _write(test_manifest, json.dumps(render))

        with self.assertRaisesRegex(ArtifactError, "different inputs/settings"):
            self._publish(test_render_manifest=test_manifest)

    def test_legacy_identity_is_not_migrated(self) -> None:
        self._publish()
        legacy = json.loads(self.output.read_text(encoding="utf-8"))
        legacy.pop("identity_version")
        _write(self.output, json.dumps(legacy))

        with self.assertRaisesRegex(ArtifactError, "legacy mesh identity"):
            self._publish()


if __name__ == "__main__":
    unittest.main()
