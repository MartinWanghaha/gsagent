from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.prepare_inpaint_workspace import (
    ArtifactError,
    prepare_inpaint_workspace,
)
from tools.prepare_removal_workspace import _identity
from tools.publish_inpainted_edgs_model import publish_inpainted_model
from utils.fusion_manifest_identity import (
    FUSION_IDENTITY_VERSION,
    fusion_artifact_id,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(resolved),
    }


def upgrade_fusion_manifest_to_v2(path: Path) -> dict[str, object]:
    """Upgrade a fixture manifest using the same identity code as production."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["identity_version"] = FUSION_IDENTITY_VERSION
    payload["upstream_artifact_ids"] = {
        key: value["artifact_id"] for key, value in payload["upstream"].items()
    }
    payload["artifact_id"] = fusion_artifact_id(payload)
    write_json(path, payload)
    return payload


def strict_manifest(path: Path, kind: str, artifact_id: str, **values: object) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "kind": kind,
            "complete": True,
            "status": "complete",
            "artifact_id": artifact_id,
            **values,
        },
    )


def write_gaussian_ply(path: Path, xyz: tuple[float, float, float] = (0, 0, 0)) -> None:
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
        *(f"obj_dc_{index}" for index in range(16)),
    ]
    values = [*xyz, *([0.0] * (len(properties) - 3))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                *(f"property float {name}" for name in properties),
                "end_header",
                " ".join(map(str, values)),
            ]
        )
        + "\n",
        encoding="ascii",
    )


class InpaintArtifactFixture:
    source_iteration = 2
    output_iteration = 5
    target_ids = [14]
    surrounding_ids: list[int] = []

    def __init__(self, root: Path, frame_count=30) -> None:
        self.frame_count = frame_count
        self.root = root
        self.removal_workspace = root / "removal_work_model"
        self.removal_workspace.mkdir()
        self.cfg_args = self.removal_workspace / "cfg_args"
        self.cfg_args.write_text(
            "Namespace(sh_degree=0, num_classes=20)\n", encoding="utf-8"
        )
        semantic_iteration = (
            self.removal_workspace
            / "point_cloud"
            / f"iteration_{self.source_iteration}"
        )
        self.semantic_ply = semantic_iteration / "point_cloud.ply"
        self.classifier = semantic_iteration / "classifier.pth"
        write_gaussian_ply(self.semantic_ply)
        self.classifier.write_bytes(b"classifier")

        removed_iteration = (
            self.removal_workspace
            / "point_cloud_object_removal"
            / f"iteration_{self.source_iteration}"
        )
        self.removed_ply = removed_iteration / "point_cloud.ply"
        write_gaussian_ply(self.removed_ply)

        baseline = (
            self.removal_workspace
            / "virtual"
            / f"ours_{self.source_iteration}"
            / "depth"
        )
        removed_virtual = (
            self.removal_workspace
            / "virtual"
            / "ours_object_removal"
            / f"iteration_{self.source_iteration}"
        )
        for index in range(self.frame_count):
            name = f"{index:05d}"
            baseline.mkdir(parents=True, exist_ok=True)
            (baseline / f"{name}.npy").write_bytes(f"base-{index}".encode())
            for folder, suffix, value in (
                (removed_virtual / "renders", ".png", f"rgb-{index}"),
                (removed_virtual / "depth", ".npy", f"depth-{index}"),
            ):
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{name}{suffix}").write_bytes(value.encode())

        self.removal_workspace_manifest = (
            self.removal_workspace / "workspace_manifest.json"
        )
        strict_manifest(
            self.removal_workspace_manifest,
            "paintmesh-removal-workspace",
            "removal-workspace-id",
            parameters={"iteration": self.source_iteration},
        )

        self.edgs_config = root / "edgs" / "config.yaml"
        self.edgs_config.parent.mkdir()
        self.edgs_config.write_text("gs:\n  sh_degree: 0\n", encoding="utf-8")
        self.removed_model_manifest = root / "removed_3dgs" / "model_manifest.json"
        strict_manifest(
            self.removed_model_manifest,
            "paintmesh-removed-edgs-model",
            "removed-model-id",
            parameters={
                "iteration": self.source_iteration,
                "target_ids": self.target_ids,
                "surrounding_ids": self.surrounding_ids,
                "removal_threshold": 0.7,
            },
            inputs={
                "classifier": record(self.classifier),
                "edgs_config": record(self.edgs_config),
                "cfg_args": record(self.cfg_args),
            },
        )

        self.removal_manifest = root / "removal_manifest.json"
        strict_manifest(
            self.removal_manifest,
            "paintmesh-object-removal",
            "removal-id",
            target_ids=self.target_ids,
            surrounding_ids=self.surrounding_ids,
            parameters={
                "distill_iteration": self.source_iteration,
                "removal_threshold": 0.7,
            },
            inputs={
                "workspace_manifest": {
                    "path": str(self.removal_workspace_manifest.resolve()),
                    "artifact_id": "removal-workspace-id",
                },
                "model_manifest": {
                    "path": str(self.removed_model_manifest.resolve()),
                    "artifact_id": "removed-model-id",
                },
            },
        )

        self.camera_manifest = root / "tracker" / "virtual_cameras.json"
        camera_values = {
            "frame_count": self.frame_count,
            "iteration": self.source_iteration,
            "circle_radius": 1.0,
            "cameras": [dict(image_name=f"{index:05d}", image_width=32, image_height=24,
                             R=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], T=[0., 0., 0.], FoVx=1., FoVy=1.,
                             znear=.01, zfar=100., trans=[0., 0., 0.], scale=1.) for index in range(self.frame_count)],
        }
        self.camera_artifact_id = _identity(
            {
                "schema_version": 1,
                "kind": "inpaint360gs-virtual-cameras",
                **camera_values,
            }
        )
        strict_manifest(
            self.camera_manifest,
            "inpaint360gs-virtual-cameras",
            self.camera_artifact_id,
            **camera_values,
        )
        camera_record = record(self.camera_manifest)
        self.tracking_masks = root / "tracker" / "masks"
        mask_records = []
        for index in range(self.frame_count):
            path = self.tracking_masks / f"{index:05d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"mask-{index}".encode())
            mask_records.append({"file": path.name, **record(path)})
            mask_records[-1].pop("path")
        self.tracking_session = root / "tracker" / "tracking_session.json"
        write_json(
            self.tracking_session,
            {
                "schema_version": 2,
                "kind": "paintmesh-tracking-session",
                "complete": True,
                "status": "complete",
                "input_cameras": {
                    **camera_record,
                    "artifact_id": self.camera_artifact_id,
                },
                "expected_masks": [f"{index:05d}.png" for index in range(self.frame_count)],
                "outputs": {"masks": mask_records},
            },
        )
        self.workspace = root / "inpaint" / "work_model"
        prepare_inpaint_workspace(
            self.removal_workspace,
            self.removal_manifest,
            self.tracking_session,
            self.camera_manifest,
            self.tracking_masks,
            self.source_iteration,
            self.workspace,
        )
        self.workspace_manifest = self.workspace / "workspace_manifest.json"

        frame_names = [f"{index:05d}" for index in range(self.frame_count)]
        lama_root = root / "inpaint" / "lama"
        color_input = lama_root / "color_input"
        depth_input = lama_root / "depth_input"
        color_output = lama_root / "color_output"
        depth_output = lama_root / "depth_output"
        lama_input_frames = {}
        lama_completion_frames = {}
        fusion_inputs = []
        fusion_outputs = []
        for index, stem in enumerate(frame_names):
            prepared = {}
            for key, path in (
                ("color", color_input / f"{stem}.png"),
                ("color_mask", color_input / f"{stem}_mask.png"),
                ("depth", depth_input / f"{stem}.npy"),
                ("depth_mask", depth_input / f"{stem}_mask.png"),
                (
                    "reference_depth",
                    depth_input / "depth_original" / f"{stem}.npy",
                ),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"prepared-{key}-{stem}".encode())
                prepared[key] = record(path)

            completed_color = color_output / f"{stem}.png"
            completed_depth = depth_output / f"{stem}.npy"
            for path, value in (
                (completed_color, f"completed-color-{stem}"),
                (completed_depth, f"completed-depth-{stem}"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value.encode())
            completed = {
                "color": record(completed_color),
                "depth": record(completed_depth),
            }

            removed_rgb = removed_virtual / "renders" / f"{stem}.png"
            removed_depth = removed_virtual / "depth" / f"{stem}.npy"
            reference_depth = baseline / f"{stem}.npy"
            source_mask = self.tracking_masks / f"{stem}.png"
            lama_input_frames[stem] = {
                "shape": [1, 1],
                "mask_foreground_before": 1,
                "mask_foreground_after": 1,
                "removed_depth_range": [0.0, 1.0],
                "reference_depth_range": [0.0, 1.0],
                "inputs": {
                    "mask": record(source_mask),
                    "removed_rgb": record(removed_rgb),
                    "removed_depth": record(removed_depth),
                    "reference_depth": record(reference_depth),
                },
                "outputs": prepared,
            }
            lama_completion_frames[stem] = {
                "shape": [1, 1],
                "completed_depth_range": [0.0, 1.0],
                "outputs": completed,
            }

            fused_mask = root / "inpaint" / "fusion" / f"{stem}.ply"
            fused_hole = root / "inpaint" / "fusion_hole" / f"{stem}.ply"
            for path, value in (
                (fused_mask, f"fused-mask-{stem}"),
                (fused_hole, f"fused-hole-{stem}"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value.encode())
            fusion_inputs.append(
                {
                    "frame": stem,
                    "completed_rgb": completed["color"],
                    "inpaint_mask": prepared["color_mask"],
                    "completed_depth": completed["depth"],
                    "removed_rgb": record(removed_rgb),
                    "removed_depth": record(removed_depth),
                }
            )
            fusion_outputs.append(
                {
                    "frame": stem,
                    "fused_mask_ply": record(fused_mask),
                    "fused_hole_ply": record(fused_hole),
                }
            )

        self.lama_input_manifest = lama_root / "input_manifest.json"
        strict_manifest(
            self.lama_input_manifest,
            "paintmesh-lama-inputs",
            "lama-input-id",
            parameters={
                "frames": self.frame_count,
                "frame_names": frame_names,
                "min_area": 0,
                "dilation": 0,
                "mask_rule": "original_index_nonzero",
            },
            roots={
                "color_input": str(color_input.resolve()),
                "depth_input": str(depth_input.resolve()),
            },
            frames=lama_input_frames,
        )
        lama_model_config = lama_root / "model" / "config.yaml"
        lama_checkpoint = lama_root / "model" / "models" / "best.ckpt"
        lama_model_config.parent.mkdir(parents=True, exist_ok=True)
        lama_model_config.write_text("model: test\n", encoding="utf-8")
        lama_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        lama_checkpoint.write_bytes(b"checkpoint")
        self.lama_manifest = lama_root / "completion_manifest.json"
        strict_manifest(
            self.lama_manifest,
            "paintmesh-lama-completion",
            "lama-id",
            input_manifest=record(self.lama_input_manifest),
            input_artifact_id="lama-input-id",
            model={
                "config": record(lama_model_config),
                "checkpoint": record(lama_checkpoint),
            },
            parameters={
                "frames": self.frame_count,
                "frame_names": frame_names,
                "recursive_guide": False,
                "outside_mask_policy": "preserve_input_exactly",
            },
            roots={
                "color_output": str(color_output.resolve()),
                "depth_output": str(depth_output.resolve()),
            },
            frames=lama_completion_frames,
        )
        self.fusion_manifest = root / "inpaint" / "fusion_manifest.json"
        strict_manifest(
            self.fusion_manifest,
            "paintmesh-rgbd-fusion",
            "fusion-id",
            parameters={
                "iteration": self.source_iteration,
                "frame_count": self.frame_count,
                "circle_radius": 1.0,
                "write_hole_ply": True,
            },
            upstream={
                "camera_manifest": {
                    "artifact": record(self.camera_manifest),
                    "artifact_id": self.camera_artifact_id,
                },
                "lama_completion_manifest": {
                    "artifact": record(self.lama_manifest),
                    "artifact_id": "lama-id",
                },
            },
            inputs=fusion_inputs,
            outputs=fusion_outputs,
        )
        self.inpaint_config = root / "inpaint" / "config.json"
        write_json(
            self.inpaint_config,
            {
                "finetune_iteration": self.output_iteration,
                "target_id": self.target_ids,
                "surrounding_ids": self.surrounding_ids,
                "select_obj_id": self.target_ids,
                "removal_thresh": 0.7,
                "lambda_dssim": 0.8,
            },
        )
        self.inpainted_ply = root / "inpaint" / "raw" / "point_cloud.ply"
        write_gaussian_ply(self.inpainted_ply, (1, 2, 3))
        self.model_output = root / "inpaint" / "inpainted_3dgs"

    def publish(self, *, fusion_seed_frame: int = 4) -> dict[str, object]:
        return publish_inpainted_model(
            self.inpainted_ply,
            self.classifier,
            self.edgs_config,
            self.cfg_args,
            self.source_iteration,
            self.output_iteration,
            "14",
            "none",
            self.removed_model_manifest,
            self.removal_manifest,
            self.workspace_manifest,
            self.tracking_session,
            self.lama_manifest,
            self.fusion_manifest,
            self.inpaint_config,
            self.model_output,
            fusion_seed_frame=fusion_seed_frame,
        )


class PrepareInpaintWorkspaceTest(unittest.TestCase):
    def test_creates_isolated_relative_link_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            for relative in (
                "cfg_args",
                "point_cloud",
                "point_cloud_object_removal",
                "tracking_masks",
                "virtual/cameras.json",
                "virtual/ours_2/depth",
                "virtual/ours_object_removal/iteration_2/renders",
                "virtual/ours_object_removal/iteration_2/depth",
            ):
                path = fixture.workspace / relative
                self.assertTrue(path.is_symlink(), relative)
                self.assertFalse(os.path.isabs(os.readlink(path)))
            manifest = json.loads(fixture.workspace_manifest.read_text())
            self.assertEqual(manifest["kind"], "paintmesh-inpaint-workspace")
            self.assertTrue(manifest["complete"])

            refreshed = prepare_inpaint_workspace(
                fixture.removal_workspace,
                fixture.removal_manifest,
                fixture.tracking_session,
                fixture.camera_manifest,
                fixture.tracking_masks,
                fixture.source_iteration,
                fixture.workspace,
            )
            self.assertEqual(refreshed["action"], "refresh")

    def test_refresh_ignores_volatile_upstream_manifest_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            workspace_before = fixture.workspace_manifest.read_bytes()
            artifact_before = json.loads(workspace_before)["artifact_id"]

            upstream = json.loads(fixture.removal_workspace_manifest.read_text())
            upstream["updated_at"] = "2099-01-01T00:00:00+00:00"
            write_json(fixture.removal_workspace_manifest, upstream)

            refreshed = prepare_inpaint_workspace(
                fixture.removal_workspace,
                fixture.removal_manifest,
                fixture.tracking_session,
                fixture.camera_manifest,
                fixture.tracking_masks,
                fixture.source_iteration,
                fixture.workspace,
            )
            self.assertEqual(refreshed["action"], "refresh")
            self.assertEqual(refreshed["artifact_id"], artifact_before)
            self.assertEqual(fixture.workspace_manifest.read_bytes(), workspace_before)

    def test_migrates_a_valid_legacy_workspace_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            workspace = json.loads(fixture.workspace_manifest.read_text())
            legacy_identity = {
                "kind": workspace["kind"],
                "schema_version": workspace["schema_version"],
                "source_iteration": workspace["parameters"]["source_iteration"],
                "removal_workspace_artifact_id": workspace["upstream"][
                    "removal_workspace_manifest"
                ]["artifact_id"],
                "removal_artifact_id": workspace["upstream"]["removal_manifest"][
                    "artifact_id"
                ],
                "tracking_artifact_id": workspace["upstream"]["tracking_session"][
                    "artifact_id"
                ],
                "camera_artifact_id": workspace["upstream"]["camera_manifest"][
                    "artifact_id"
                ],
                "input_hashes": {
                    name: record["sha256"]
                    for name, record in workspace["inputs"].items()
                },
                "frame_set_artifact_ids": {
                    name: record["artifact_id"]
                    for name, record in workspace["frame_sets"].items()
                },
            }
            workspace.pop("identity_version")
            workspace["artifact_id"] = _identity(legacy_identity)
            write_json(fixture.workspace_manifest, workspace)

            migrated = prepare_inpaint_workspace(
                fixture.removal_workspace,
                fixture.removal_manifest,
                fixture.tracking_session,
                fixture.camera_manifest,
                fixture.tracking_masks,
                fixture.source_iteration,
                fixture.workspace,
            )
            self.assertEqual(migrated["action"], "migrate")
            current = json.loads(fixture.workspace_manifest.read_text())
            self.assertEqual(current["identity_version"], 2)
            self.assertEqual(current["artifact_id"], migrated["artifact_id"])

    def test_does_not_migrate_legacy_workspace_with_downstream_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            workspace = json.loads(fixture.workspace_manifest.read_text())
            legacy_identity = {
                "kind": workspace["kind"],
                "schema_version": workspace["schema_version"],
                "source_iteration": workspace["parameters"]["source_iteration"],
                "removal_workspace_artifact_id": workspace["upstream"][
                    "removal_workspace_manifest"
                ]["artifact_id"],
                "removal_artifact_id": workspace["upstream"]["removal_manifest"][
                    "artifact_id"
                ],
                "tracking_artifact_id": workspace["upstream"]["tracking_session"][
                    "artifact_id"
                ],
                "camera_artifact_id": workspace["upstream"]["camera_manifest"][
                    "artifact_id"
                ],
                "input_hashes": {
                    name: record["sha256"]
                    for name, record in workspace["inputs"].items()
                },
                "frame_set_artifact_ids": {
                    name: record["artifact_id"]
                    for name, record in workspace["frame_sets"].items()
                },
            }
            workspace.pop("identity_version")
            workspace["artifact_id"] = _identity(legacy_identity)
            write_json(fixture.workspace_manifest, workspace)
            downstream = (
                fixture.workspace / "point_cloud_object_inpaint_virtual" / "iteration_5"
            )
            downstream.mkdir(parents=True)

            with self.assertRaisesRegex(
                ArtifactError, "different parameters or inputs"
            ):
                prepare_inpaint_workspace(
                    fixture.removal_workspace,
                    fixture.removal_manifest,
                    fixture.tracking_session,
                    fixture.camera_manifest,
                    fixture.tracking_masks,
                    fixture.source_iteration,
                    fixture.workspace,
                )

    def test_rejects_camera_changed_after_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            payload = json.loads(fixture.camera_manifest.read_text())
            payload["cameras"][0]["image_name"] = "99999"
            write_json(fixture.camera_manifest, payload)
            with self.assertRaisesRegex(ArtifactError, "input_cameras|camera names|image_name"):
                prepare_inpaint_workspace(
                    fixture.removal_workspace,
                    fixture.removal_manifest,
                    fixture.tracking_session,
                    fixture.camera_manifest,
                    fixture.tracking_masks,
                    fixture.source_iteration,
                    fixture.root / "different_workspace",
                )


class PublishInpaintedModelTest(unittest.TestCase):
    def test_non30_workspace_and_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary), frame_count=7)
            result = fixture.publish()
            self.assertTrue(result["artifact_id"])
            with self.assertRaisesRegex(ArtifactError, "outside"):
                fixture.publish(fusion_seed_frame=7)

    def test_v2_fusion_accepts_mtime_only_output_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            fusion = upgrade_fusion_manifest_to_v2(fixture.fusion_manifest)
            for key in ("fused_mask_ply", "fused_hole_ply"):
                output = Path(fusion["outputs"][0][key]["path"])
                stat = output.stat()
                os.utime(output, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

            summary = fixture.publish()

            self.assertEqual(summary["output_iteration"], 5)

    def test_v2_fusion_still_rejects_output_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            fusion = upgrade_fusion_manifest_to_v2(fixture.fusion_manifest)
            output = Path(fusion["outputs"][0]["fused_mask_ply"]["path"])
            original = output.read_bytes()
            output.write_bytes(bytes([original[0] ^ 1]) + original[1:])

            with self.assertRaisesRegex(ArtifactError, "sha256"):
                fixture.publish()

    def test_v2_fusion_rejects_record_change_with_stale_artifact_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            fusion = upgrade_fusion_manifest_to_v2(fixture.fusion_manifest)
            output_record = fusion["outputs"][0]["fused_mask_ply"]
            output = Path(output_record["path"])
            original = output.read_bytes()
            output.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            fusion["outputs"][0]["fused_mask_ply"] = record(output)
            write_json(fixture.fusion_manifest, fusion)

            with self.assertRaisesRegex(ArtifactError, "content-addressed payload"):
                fixture.publish()

    def test_legacy_fusion_still_rejects_mtime_only_output_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            fusion = json.loads(fixture.fusion_manifest.read_text())
            output = Path(fusion["outputs"][0]["fused_mask_ply"]["path"])
            stat = output.stat()
            os.utime(output, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

            with self.assertRaisesRegex(ArtifactError, "mtime_ns"):
                fixture.publish()

    def test_source_and_output_iterations_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            summary = fixture.publish()
            self.assertEqual(summary["source_iteration"], 2)
            self.assertEqual(summary["output_iteration"], 5)
            for relative in (
                "config.yaml",
                "cfg_args",
                "point_cloud/iteration_5/classifier.pth",
            ):
                path = fixture.model_output / relative
                self.assertTrue(path.is_symlink(), relative)
                self.assertFalse(os.path.isabs(os.readlink(path)))
            point_cloud = (
                fixture.model_output / "point_cloud" / "iteration_5" / "point_cloud.ply"
            )
            self.assertTrue(point_cloud.is_file())
            self.assertFalse(point_cloud.is_symlink())
            self.assertFalse(os.path.samefile(point_cloud, fixture.inpainted_ply))
            self.assertEqual(sha256(point_cloud), sha256(fixture.inpainted_ply))
            manifest = json.loads(
                (fixture.model_output / "model_manifest.json").read_text()
            )
            self.assertEqual(manifest["parameters"]["source_iteration"], 2)
            self.assertEqual(manifest["parameters"]["output_iteration"], 5)
            self.assertEqual(manifest["parameters"]["fusion_seed_frame"], 4)
            self.assertEqual(
                manifest["model"]["point_cloud_storage"]["type"],
                "regular_copy",
            )
            self.assertEqual(fixture.publish()["action"], "refresh")

    def test_refresh_reuses_intact_materialized_point_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            fixture.publish()
            point_cloud = (
                fixture.model_output / "point_cloud" / "iteration_5" / "point_cloud.ply"
            )
            before = point_cloud.stat()
            manifest_path = fixture.model_output / "model_manifest.json"
            manifest_before = manifest_path.read_bytes()

            summary = fixture.publish()

            after = point_cloud.stat()
            self.assertEqual(summary["action"], "refresh")
            self.assertEqual(summary["point_cloud_action"], "reuse")
            self.assertEqual(
                (after.st_dev, after.st_ino), (before.st_dev, before.st_ino)
            )
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
            self.assertEqual(manifest_path.read_bytes(), manifest_before)

    def test_refresh_repairs_corrupted_materialized_point_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            fixture.publish()
            point_cloud = (
                fixture.model_output / "point_cloud" / "iteration_5" / "point_cloud.ply"
            )
            manifest_path = fixture.model_output / "model_manifest.json"
            manifest_before = manifest_path.read_bytes()
            damaged = bytearray(point_cloud.read_bytes())
            damaged[-1] ^= 1
            point_cloud.write_bytes(damaged)

            summary = fixture.publish()

            self.assertEqual(summary["point_cloud_action"], "materialize")
            self.assertFalse(point_cloud.is_symlink())
            self.assertFalse(os.path.samefile(point_cloud, fixture.inpainted_ply))
            self.assertEqual(sha256(point_cloud), sha256(fixture.inpainted_ply))
            self.assertEqual(manifest_path.read_bytes(), manifest_before)

    def test_refresh_materializes_legacy_point_cloud_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            fixture.publish()
            point_cloud = (
                fixture.model_output / "point_cloud" / "iteration_5" / "point_cloud.ply"
            )
            point_cloud.unlink()
            os.symlink(
                os.path.relpath(fixture.inpainted_ply, start=point_cloud.parent),
                point_cloud,
            )

            summary = fixture.publish()

            self.assertEqual(summary["point_cloud_action"], "materialize")
            self.assertTrue(point_cloud.is_file())
            self.assertFalse(point_cloud.is_symlink())
            self.assertFalse(os.path.samefile(point_cloud, fixture.inpainted_ply))
            self.assertEqual(sha256(point_cloud), sha256(fixture.inpainted_ply))

    def test_recovers_interrupted_publish_with_regular_point_cloud(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            iteration_root = fixture.model_output / "point_cloud" / "iteration_5"
            iteration_root.mkdir(parents=True)
            for source, destination in (
                (fixture.edgs_config, fixture.model_output / "config.yaml"),
                (fixture.cfg_args, fixture.model_output / "cfg_args"),
                (fixture.classifier, iteration_root / "classifier.pth"),
            ):
                os.symlink(
                    os.path.relpath(source, start=destination.parent), destination
                )
            point_cloud = iteration_root / "point_cloud.ply"
            point_cloud.write_bytes(fixture.inpainted_ply.read_bytes())

            summary = fixture.publish()

            self.assertEqual(summary["action"], "recover")
            self.assertEqual(summary["point_cloud_action"], "reuse")
            self.assertTrue(point_cloud.is_file())
            self.assertFalse(point_cloud.is_symlink())
            self.assertTrue((fixture.model_output / "model_manifest.json").is_file())

    def test_existing_output_rejects_changed_fusion_seed_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            fixture.publish(fusion_seed_frame=4)
            with self.assertRaisesRegex(
                ArtifactError, "different parameters or inputs"
            ):
                fixture.publish(fusion_seed_frame=5)

    def test_existing_output_rejects_changed_fusion_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            fixture.publish()
            fusion = json.loads(fixture.fusion_manifest.read_text())
            fusion["artifact_id"] = "different-fusion"
            write_json(fixture.fusion_manifest, fusion)
            with self.assertRaisesRegex(
                ArtifactError, "different parameters or inputs"
            ):
                fixture.publish()

    def test_rejects_fusion_with_wrong_lama_upstream_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InpaintArtifactFixture(Path(temporary))
            fusion = json.loads(fixture.fusion_manifest.read_text())
            fusion["upstream"]["lama_completion_manifest"]["artifact_id"] = "wrong"
            write_json(fixture.fusion_manifest, fusion)
            with self.assertRaisesRegex(ArtifactError, "artifact_id mismatch"):
                fixture.publish()


if __name__ == "__main__":
    unittest.main()
