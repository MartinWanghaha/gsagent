from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.finalize_inpaint_result import ArtifactError, finalize_inpaint_result
from tools.tests.test_inpaint_artifacts import (
    InpaintArtifactFixture,
    record,
    strict_manifest,
    write_json,
)


def write_triangle_mesh(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "3 0 1 2",
            )
        )
        + "\n",
        encoding="ascii",
    )


class InpaintGeometryFixture(InpaintArtifactFixture):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.publish()
        self.model_manifest = self.model_output / "model_manifest.json"
        model_payload = json.loads(self.model_manifest.read_text())
        self.published_ply = (
            self.model_output / "point_cloud" / "iteration_5" / "point_cloud.ply"
        ).resolve(strict=True)

        self.mesh = self.model_output / "mesh" / "ours_5" / "tsdf_fusion_post.ply"
        write_triangle_mesh(self.mesh)
        self.mesh_manifest = self.mesh.parent / "mesh_manifest.json"
        strict_manifest(
            self.mesh_manifest,
            "paintmesh-pgsr-inpaint-mesh",
            "mesh-id",
            parameters={
                "iteration": 5,
                "renderer": "pgsr",
                "resolution": 8,
            },
            inputs={
                "model_manifest": {
                    "path": str(self.model_manifest.resolve()),
                    "artifact_id": model_payload["artifact_id"],
                },
                "gaussian_ply": record(self.published_ply),
            },
            outputs={"mesh": record(self.mesh)},
        )

        self.semantic_root = root / "inpaint" / "inpainted_mesh"
        self.semantic_root.mkdir()
        self.geometry = self.semantic_root / "geometry.ply"
        os.symlink(
            os.path.relpath(self.mesh, start=self.geometry.parent), self.geometry
        )
        arrays = {
            "gaussian_label": np.asarray([14], dtype=np.uint16),
            "gaussian_confidence": np.asarray([0.9], dtype=np.float32),
            "vertex_label": np.asarray([14, 0, 0], dtype=np.uint16),
            "vertex_confidence": np.asarray([0.8, 0.9, 0.7], dtype=np.float32),
            "face_label": np.asarray([14], dtype=np.uint16),
            "face_confidence": np.asarray([0.6], dtype=np.float32),
        }
        filenames = {
            "gaussian_label": "gaussian_instance_id.npy",
            "gaussian_confidence": "gaussian_confidence.npy",
            "vertex_label": "vertex_instance_id.npy",
            "vertex_confidence": "vertex_confidence.npy",
            "face_label": "face_instance_id.npy",
            "face_confidence": "face_confidence.npy",
        }
        outputs = {}
        for name, array in arrays.items():
            path = self.semantic_root / filenames[name]
            np.save(path, array)
            outputs[name] = {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        palette = self.semantic_root / "palette.json"
        write_json(palette, {"classes": []})
        outputs["palette"] = {
            "file": palette.name,
            "size_bytes": palette.stat().st_size,
        }
        self.semantic_manifest = self.semantic_root / "semantic_manifest.json"
        write_json(
            self.semantic_manifest,
            {
                "schema_version": 1,
                "complete": True,
                "status": "complete",
                "inputs": {
                    "gaussian_ply": record(self.published_ply),
                    "mesh": record(self.mesh),
                },
                "parameters": {"write_colored_ply": False},
                "counts": {
                    "gaussians": 1,
                    "mesh_vertices": 3,
                    "mesh_triangles": 1,
                },
                "outputs": outputs,
            },
        )
        self.result_manifest = root / "inpaint" / "inpaint_manifest.json"

    def finalize(self) -> dict[str, object]:
        return finalize_inpaint_result(
            self.model_manifest,
            self.mesh_manifest,
            self.semantic_manifest,
            self.removal_manifest,
            self.workspace_manifest,
            self.lama_manifest,
            self.fusion_manifest,
            self.published_ply,
            self.mesh,
            self.geometry,
            self.result_manifest,
        )


class FinalizeInpaintResultTest(unittest.TestCase):
    def test_target_residuals_are_recorded_without_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintGeometryFixture(Path(temporary))
            summary = fixture.finalize()
            residuals = summary["target_residuals"]
            self.assertEqual(residuals["gaussians"]["14"], 1)
            self.assertEqual(residuals["mesh_vertices"]["14"], 1)
            self.assertEqual(residuals["mesh_triangles"]["14"], 1)
            payload = json.loads(fixture.result_manifest.read_text())
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["kind"], "paintmesh-object-inpaint")
            self.assertEqual(fixture.finalize()["action"], "refresh")

    def test_changed_sidecar_rejects_existing_result_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintGeometryFixture(Path(temporary))
            fixture.finalize()
            np.save(
                fixture.semantic_root / "gaussian_instance_id.npy",
                np.asarray([0], dtype=np.uint16),
            )
            with self.assertRaisesRegex(
                ArtifactError, "different inputs or parameters"
            ):
                fixture.finalize()

    def test_geometry_must_be_relative_link_to_pgsr_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintGeometryFixture(Path(temporary))
            fixture.geometry.unlink()
            other = fixture.root / "other_mesh.ply"
            write_triangle_mesh(other)
            os.symlink(
                os.path.relpath(other, fixture.geometry.parent), fixture.geometry
            )
            with self.assertRaisesRegex(ArtifactError, "does not resolve"):
                fixture.finalize()

    def test_sidecar_length_must_match_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintGeometryFixture(Path(temporary))
            path = fixture.semantic_root / "vertex_instance_id.npy"
            np.save(path, np.asarray([14, 0], dtype=np.uint16))
            semantic = json.loads(fixture.semantic_manifest.read_text())
            semantic["outputs"]["vertex_label"].update(
                {
                    "size_bytes": path.stat().st_size,
                    "shape": [2],
                }
            )
            write_json(fixture.semantic_manifest, semantic)
            with self.assertRaisesRegex(ArtifactError, r"expected \(3,\)"):
                fixture.finalize()


if __name__ == "__main__":
    unittest.main()
