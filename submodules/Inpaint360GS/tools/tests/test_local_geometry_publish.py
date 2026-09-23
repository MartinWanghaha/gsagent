"""Publisher/finalizer integration; heavy geometry validation has its own tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/paintmesh"))

from tools.tests.test_inpaint_artifacts import InpaintArtifactFixture, record, write_json
from tools.tests.test_inpaint_geometry import InpaintGeometryFixture
from tools.publish_inpainted_edgs_model import publish_inpainted_model
from tools.prepare_removal_workspace import ArtifactError


def test_publisher_binds_local_result_and_refuses_rgb_only_reuse(tmp_path):
    fixture = InpaintArtifactFixture(tmp_path)
    rgb_manifest = tmp_path / "rgb_manifest.json"
    write_json(rgb_manifest, {"inputs": {key: record(path) for key, path in (
        ("classifier", fixture.classifier), ("fusion", fixture.fusion_manifest),
        ("inpaint_config", fixture.inpaint_config))}})
    local_path = tmp_path / "local.json"
    local = dict(artifact_id="local-result-id", parameters=dict(
        rgb_iterations=fixture.output_iteration, local_geometry_iterations=1000),
        inputs=dict(rgb_manifest=record(rgb_manifest), edgs_config=record(fixture.edgs_config)))
    write_json(local_path, local)
    arguments = [fixture.inpainted_ply, fixture.classifier, fixture.edgs_config,
        fixture.cfg_args, fixture.source_iteration, fixture.output_iteration, "14", "none",
        fixture.removed_model_manifest, fixture.removal_manifest, fixture.workspace_manifest,
        fixture.tracking_session, fixture.lama_manifest, fixture.fusion_manifest,
        fixture.inpaint_config, fixture.model_output]
    with patch("tools.publish_inpainted_edgs_model._validate_local_geometry", return_value=local) as validate:
        publish_inpainted_model(*arguments, local_geometry_manifest_path=local_path)
        validate.assert_called_once()
    model = json.loads((fixture.model_output / "model_manifest.json").read_text())
    assert model["parameters"]["local_geometry_refine"] is True
    assert model["parameters"]["local_geometry_iterations"] == 1000
    assert model["upstream_artifact_ids"]["local_geometry"] == "local-result-id"
    assert model["inputs"]["local_geometry_manifest"]["sha256"] == record(local_path)["sha256"]
    with pytest.raises(ArtifactError):
        fixture.publish()


def test_finalizer_requires_enabled_local_receipt(tmp_path):
    fixture = InpaintGeometryFixture(tmp_path)
    model = json.loads(fixture.model_manifest.read_text())
    model["parameters"].update(local_geometry_refine=True, rgb_iterations=5, local_geometry_iterations=1000)
    write_json(fixture.model_manifest, model)
    with pytest.raises(ArtifactError, match="local_geometry"):
        fixture.finalize()


def test_finalizer_rejects_contradictory_disabled_local_metadata(tmp_path):
    fixture = InpaintGeometryFixture(tmp_path)
    model = json.loads(fixture.model_manifest.read_text())
    model["upstream_artifact_ids"]["local_geometry"] = "unexpected"
    write_json(fixture.model_manifest, model)
    with pytest.raises(ArtifactError, match="contradicts"):
        fixture.finalize()


def test_publisher_binds_density_without_enabling_geometry(tmp_path):
    mode='mass_adaptive'
    fixture = InpaintArtifactFixture(tmp_path)
    density_path, rgb_path = tmp_path/"density.json", tmp_path/"rgb.json"
    write_json(density_path,{"artifact_id":"density-id"})
    write_json(rgb_path,{"artifact_id":"rgb-id"})
    arguments = [fixture.inpainted_ply, fixture.classifier, fixture.edgs_config,
        fixture.cfg_args, fixture.source_iteration, fixture.output_iteration, "14", "none",
        fixture.removed_model_manifest, fixture.removal_manifest, fixture.workspace_manifest,
        fixture.tracking_session, fixture.lama_manifest, fixture.fusion_manifest,
        fixture.inpaint_config, fixture.model_output]
    with patch("tools.publish_inpainted_edgs_model._validate_density",return_value={"artifact_id":"density-id","parameters":{"mode":mode}}):
        publish_inpainted_model(*arguments,density_manifest_path=density_path,rgb_manifest_path=rgb_path)
    model=json.loads((fixture.model_output/"model_manifest.json").read_text())
    assert model["parameters"]["support_density_mode"]==mode
    assert not model["parameters"].get("local_geometry_refine",False)
    assert model["upstream_artifact_ids"]["support_density"]=="density-id"
    with pytest.raises(ArtifactError):fixture.publish()


def test_finalizer_rejects_density_without_mode(tmp_path):
    fixture=InpaintGeometryFixture(tmp_path)
    model=json.loads(fixture.model_manifest.read_text())
    model["upstream_artifact_ids"]["support_density"]="unexpected"
    write_json(fixture.model_manifest,model)
    with pytest.raises(ArtifactError,match="density provenance"):
        fixture.finalize()
