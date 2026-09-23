"""Producer dispatch tests; numerical joint receipts are tested with real PGSR."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/paintmesh"))
from tools.tests.test_inpaint_artifacts import InpaintArtifactFixture, record, write_json
from tools.tests.test_inpaint_geometry import InpaintGeometryFixture
from tools.publish_inpainted_edgs_model import publish_inpainted_model
from tools.finalize_inpaint_result import finalize_inpaint_result
from tools.prepare_removal_workspace import ArtifactError


def test_joint_publisher_requires_own_producer_not_fusion(tmp_path):
    f = InpaintArtifactFixture(tmp_path)
    # Joint iterations intentionally differ from the old config's RGB budget.
    initial_path = tmp_path / "init.json"
    initial = {"inputs": {k: record(p) for k, p in (("classifier", f.classifier),
        ("removed_model", f.removed_model_manifest), ("inpaint_config", f.inpaint_config))}}
    write_json(initial_path, initial)
    joint_path = tmp_path / "joint.json"
    joint = dict(artifact_id="joint-id", parameters={"iterations": 7},
                 inputs=dict(initialization=record(initial_path), edgs_config=record(f.edgs_config)))
    write_json(joint_path, joint)
    args = [f.inpainted_ply, f.classifier, f.edgs_config, f.cfg_args, f.source_iteration,
        7, "14", "none", f.removed_model_manifest, f.removal_manifest, f.workspace_manifest,
        f.tracking_session, f.lama_manifest, None, f.inpaint_config, f.model_output]
    with patch("tools.publish_inpainted_edgs_model._validate_edgs_joint", return_value=joint), patch("edgs_inpaint_io.validate_init", return_value=initial):
        summary = publish_inpainted_model(*args, edgs_joint_manifest_path=joint_path)
        again = publish_inpainted_model(*args, edgs_joint_manifest_path=joint_path)
    assert summary["artifact_id"] == again["artifact_id"]
    model = json.loads((f.model_output / "model_manifest.json").read_text())
    assert model["parameters"]["pipeline"] == "edgs-pgsr"
    assert model["parameters"]["output_iteration"] == 7
    assert "fusion_manifest" not in model["inputs"]
    assert "fusion_seed_frame" not in model["parameters"]
    assert model["upstream_artifact_ids"]["edgs_joint"] == "joint-id"
    with pytest.raises(ArtifactError):
        f.publish()
    args[13] = f.fusion_manifest
    with pytest.raises(ArtifactError, match="old-path"):
        publish_inpainted_model(*args, edgs_joint_manifest_path=joint_path)


def test_joint_finalizer_dispatch_and_rejects_old_fusion(tmp_path):
    f = InpaintGeometryFixture(tmp_path)
    joint_path = tmp_path / "joint.json"
    joint = dict(artifact_id="joint-id", parameters={"iterations": 5})
    write_json(joint_path, joint)
    model = json.loads(f.model_manifest.read_text())
    model["parameters"].update(pipeline="edgs-pgsr", joint_iterations=5)
    model["inputs"].pop("fusion_manifest")
    model["upstream_artifact_ids"].pop("fusion")
    model["inputs"]["edgs_joint_manifest"] = record(joint_path)
    model["upstream_artifact_ids"]["edgs_joint"] = "joint-id"
    write_json(f.model_manifest, model)
    with pytest.raises(ArtifactError, match="must not supply fusion"):
        f.finalize()
    args = [f.model_manifest, f.mesh_manifest, f.semantic_manifest, f.removal_manifest,
        f.workspace_manifest, f.lama_manifest, None, f.published_ply, f.mesh,
        f.semantic_root / "geometry.ply", tmp_path / "final.json"]
    with patch("tools.finalize_inpaint_result._validate_edgs_joint", return_value=joint):
        finalize_inpaint_result(*args)
    result = json.loads((tmp_path / "final.json").read_text())
    assert result["upstream_artifact_ids"]["edgs_joint"] == "joint-id"
    assert "fusion_manifest" not in result["inputs"]
