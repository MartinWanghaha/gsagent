"""New geometry artifacts must survive workspace isolation and reject stale tracking."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "paintmesh"))
from virtual_render_io import (
    FrameWriter,
    identity,
    read_json,
    sha256,
    validate_render,
    write_json,
)
from tools.prepare_inpaint_workspace import ArtifactError, prepare_inpaint_workspace
from tools.tests.test_inpaint_artifacts import InpaintArtifactFixture, record


class VirtualGeometryWorkspaceTest(unittest.TestCase):
    def fixture(self, root, backend):
        with patch.dict("os.environ", {"VIRTUAL_RENDERER": "inpaint360gs"}):
            f = InpaintArtifactFixture(root)
        cameras = read_json(f.camera_manifest)
        for camera in cameras["cameras"]:
            camera.update(image_height=2, image_width=3)
        cameras["artifact_id"] = identity(
            {
                k: v
                for k, v in cameras.items()
                if k not in {"artifact_id", "complete", "status"}
            }
        )
        write_json(f.camera_manifest, cameras)
        p = {
            "render": np.full((3, 2, 3), 0.5, np.float32),
            "plane_depth": np.ones((1, 2, 3), np.float32),
            "depth_3dgs": np.ones((1, 2, 3), np.float32),
            "rendered_alpha": np.ones((1, 2, 3), np.float32),
            "alpha": np.ones((1, 2, 3), np.float32),
            "rendered_normal": np.broadcast_to(
                np.array([0, 0, -1], np.float32)[:, None, None], (3, 2, 3)
            ),
        }
        virtual = f.removal_workspace / "virtual"
        write_json(virtual / "render_backend.json", {"backend": backend})
        ids = {}
        for label, directory in (
            ("full", virtual / "ours_2"),
            ("removed", virtual / "ours_object_removal/iteration_2"),
        ):
            writer = FrameWriter(
                directory,
                backend,
                cameras,
                {"camera_sha256": sha256(f.camera_manifest), "files": {}},
            )
            for camera in cameras["cameras"]:
                writer.write(SimpleNamespace(image_name=camera["image_name"]), p)
            writer.finish()
            ids[label] = writer.payload["artifact_id"]
        archive = root / "tracker/images.zip"
        archive.write_bytes(b"fixture archive")
        write_json(
            virtual / "virtual_render_manifest.json",
            {
                "complete": True,
                "backend": backend,
                **ids,
                "archive_sha256": sha256(archive),
            },
        )
        session = read_json(f.tracking_session)
        manifest = virtual / "ours_object_removal/iteration_2/render_manifest.json"
        session.update(
            input_cameras={
                **record(f.camera_manifest),
                "artifact_id": cameras["artifact_id"],
            },
            input_archive=record(archive),
            input_virtual_render={
                "path": str(manifest),
                "sha256": sha256(manifest),
                "backend": backend,
                "artifact_id": ids["removed"],
            },
        )
        write_json(f.tracking_session, session)
        return f

    def prepare(self, f):
        return prepare_inpaint_workspace(
            f.removal_workspace,
            f.removal_manifest,
            f.tracking_session,
            f.camera_manifest,
            f.tracking_masks,
            2,
            f.root / "new_inpaint/work_model",
        )

    def test_both_backend_workspaces_and_refresh(self):
        for backend in ("inpaint360gs", "edgs-pgsr"):
            with self.subTest(
                backend=backend
            ), tempfile.TemporaryDirectory() as tmp, patch.dict(
                "os.environ", {"VIRTUAL_RENDERER": backend}
            ):
                f = self.fixture(Path(tmp), backend)
                first = self.prepare(f)
                workspace = Path(first["output"])
                removed = workspace / "virtual/ours_object_removal/iteration_2"
                self.assertTrue((removed / "alpha").is_symlink())
                self.assertEqual(
                    (removed / "normal").is_symlink(), backend == "edgs-pgsr"
                )
                self.assertTrue((removed / "render_manifest.json").is_symlink())
                validate_render(removed, backend)
                validate_render(workspace / "virtual/ours_2", backend)
                self.assertEqual(self.prepare(f)["artifact_id"], first["artifact_id"])
                np.save(
                    f.removal_workspace
                    / "virtual/ours_object_removal/iteration_2/alpha/00000.npy",
                    np.zeros((2, 3), np.float32),
                )
                with self.assertRaisesRegex(ValueError, "changed"):
                    self.prepare(f)

    def test_mismatched_backend_and_unbound_tracking_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self.fixture(Path(tmp), "edgs-pgsr")
            with patch.dict("os.environ", {"VIRTUAL_RENDERER": "inpaint360gs"}):
                with self.assertRaisesRegex(ArtifactError, "VIRTUAL_RENDERER"):
                    self.prepare(f)
            session = read_json(f.tracking_session)
            session.pop("input_virtual_render")
            write_json(f.tracking_session, session)
            with patch.dict("os.environ", {"VIRTUAL_RENDERER": "edgs-pgsr"}):
                with self.assertRaisesRegex(ArtifactError, "not bound"):
                    self.prepare(f)


if __name__ == "__main__":
    unittest.main()
