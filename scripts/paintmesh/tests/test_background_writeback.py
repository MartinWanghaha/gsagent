"""All-row gradients, official selection parity, exact restore and ablation scope."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
sys.path.insert(0, str(REPO / "submodules/EDGS"))
from edgs_inpaint_io import check_preserved, load_config
from source.paintmesh_joint_data import JointGaussians
from source.paintmesh_joint_writeback import select_writeback, restore_background
from run_background_writeback_ablation import make_variant, WritebackBatch


def fixture(tmp_path):
    fields = [*"xyz", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "obj_dc_0"]
    vertex = np.zeros(6, dtype=[(key, "f4") for key in fields])
    vertex["x"] = np.linspace(-.5, .5, 6)
    vertex["z"], vertex["rot_0"], vertex["opacity"], vertex["obj_dc_0"] = 2, 1, 2, 1
    vertex["f_dc_0"] = .5
    path = tmp_path / "initial.ply"
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))
    new = np.array([False] * 4 + [True] * 2)
    return path, vertex, new


def test_all_rows_train_prune_resume_restore_and_exact_saved_background(tmp_path):
    path, initial, new = fixture(tmp_path)
    model = JointGaussians(path, new, device="cpu", freeze_background=False)
    optimizer = model.optimizer(load_config()["optimizer"])
    before = model.xyz.detach().clone()
    sum(p.square().sum() for p in model.parameters()).backward()
    assert model.xyz.grad[:4].abs().sum() > 0 and model.xyz.grad[4:].abs().sum() > 0
    optimizer.step()
    assert not torch.equal(model.xyz[:4], before[:4])
    with torch.no_grad():
        model.opacity[0] = -12  # Original old row must survive for restoration.
        model.opacity[4] = -12  # New row is still pruned with baseline threshold.
    assert model.prune(optimizer, preserve_old=True) == 1
    assert model.indices.tolist() == [0, 1, 2, 3, 5]
    state = copy.deepcopy(dict(local=model.local_state(), optimizer=optimizer.state_dict()))
    restored = JointGaussians(path, new, device="cpu", freeze_background=False)
    restored.restore_local(state["local"])
    restored_optimizer = restored.optimizer(load_config()["optimizer"])
    restored_optimizer.load_state_dict(state["optimizer"])
    for key in model.parameter_names:
        torch.testing.assert_close(getattr(model, key), getattr(restored, key), rtol=0, atol=0)
    # Only old row 1 passes the post-training first-view AND 3D gate.
    permitted, stats = restore_background(restored, np.array([False, True, False, False, False]))
    assert permitted.tolist() == [False, True, False, False, True, True]
    assert stats["selected_old"] == 1 and stats["restored_old"] == 3
    assert stats["changed_old_before_restore"] == 4 and stats["changed_old_kept"] == 1
    output = tmp_path / "final.ply"
    restored.save(output)
    rows = restored.retained_rows.numpy()
    check_preserved(path, output, new, retained_rows=rows, freeze_background=False, trainable_mask=permitted)
    result = PlyData.read(str(output))["vertex"].data.copy()
    np.testing.assert_array_equal(result[[0, 2, 3]], initial[[0, 2, 3]])
    assert result["z"][1] != initial["z"][1] and result["z"][-1] != initial["z"][-1]
    result["x"][0] += 1
    PlyData([PlyElement.describe(result, "vertex")]).write(str(output))
    with pytest.raises(ValueError, match="frozen field"):
        check_preserved(path, output, new, retained_rows=rows, freeze_background=False, trainable_mask=permitted)


def test_projection_and_3d_gate_match_actual_official_function(tmp_path):
    official = REPO.parent / "Inpaint360GS"
    if not official.is_dir():
        pytest.skip("optional official source checkout unavailable")
    namespace = dict(torch=torch, np=np, cv2=cv2, os=__import__("os"), PlyData=PlyData)
    for source, wanted in ((official / "utils/point_utils.py", {"project_3d_points", "ndc_to_pixel"}),
                           (official / "edit_object_inpaint.py", {"get_projected_gaussians"})):
        tree = ast.parse(source.read_text())
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    rng = np.random.default_rng(42)
    xyz = torch.from_numpy(rng.uniform(-1.4, 1.4, (2000, 3)).astype(np.float32))
    xyz[:, 2] += .6
    mask = np.zeros((100, 120), bool); mask[30:70, 40:80] = True
    seed = rng.normal(0, .2, (200, 3)).astype(np.float32); seed[:, 2] += .5
    vertex = np.empty(len(seed), dtype=[(key, "f4") for key in "xyz"])
    for i, key in enumerate("xyz"):
        vertex[key] = seed[:, i]
    seed_path = tmp_path / "seed.ply"
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(seed_path))
    namespace["args"] = SimpleNamespace(supp_ply=str(seed_path))
    projection = torch.eye(4)
    projection[2, 3] = .3
    view = SimpleNamespace(full_proj_transform=projection, image_width=120, image_height=100,
        objects=torch.from_numpy(mask.astype(np.uint8) * 255))
    expected = namespace["get_projected_gaussians"](SimpleNamespace(get_xyz=xyz), view)["p_inside_obj_mask"].numpy()
    actual, stats = select_writeback(xyz, projection, mask, seed, load_config()["background_writeback"])
    np.testing.assert_array_equal(actual, expected)
    assert 0 < actual.sum() < len(xyz) and stats["expanded_area"] >= int(mask.sum() * 1.1)


def test_original_losses_and_all_other_configuration_are_preserved(tmp_path):
    cfg = load_config(train_depth=True, train_normal=True)
    cfg["local_background"]["enabled"] = True
    cfg["loss"].update(ssim=.2, lpips=.005)
    before = copy.deepcopy(cfg)
    variant = make_variant(cfg)
    assert cfg == before
    result = copy.deepcopy(variant["config"])
    result["freeze_background"] = True
    result["local_background"]["enabled"] = True
    result["background_writeback"]["enabled"] = False
    assert result == before
    runner = WritebackBatch.__new__(WritebackBatch)
    shared = {key: dict(path=f"/shared/{key}") for key in ("initialization", "lama", "camera", "edgs_config")}
    pub = {key: dict(path=f"/shared/{key}") for key in ("classifier", "edgs_config", "cfg_args",
        "removed_model_manifest", "removal_manifest", "workspace_manifest", "tracking_session", "lama_manifest", "inpaint_config")}
    runner.req = dict(python="/env/python", reference_config=cfg, training_inputs=shared, publish_inputs=pub,
        source="/data/bag", publish_parameters=dict(source_iteration=2000, target_ids=[36], surrounding_ids=[]))
    commands = runner.commands(tmp_path)
    train, publish, evaluate = [command for _, command, _ in commands]
    assert runner.req["training_inputs"] == shared
    assert "--local-selection" not in train
    assert train[train.index("--initialization") + 1] == "/shared/initialization"
    assert publish[publish.index("--output-iteration") + 1] == "5000"
    assert evaluate[evaluate.index("--iteration") + 1] == "5000"


@pytest.mark.parametrize("setting", ["enabled: 1", "mask_area_factor: 1", "seed_std_factor: .nan"])
def test_invalid_writeback_configuration(tmp_path, setting):
    path = tmp_path / "bad.yaml"
    path.write_text("background_writeback:\n  " + setting + "\n")
    with pytest.raises(ValueError, match="background_writeback"):
        load_config(path)


def test_invalid_combination_and_default_disabled(tmp_path):
    assert load_config()["background_writeback"]["enabled"] is False
    path = tmp_path / "bad.yaml"
    path.write_text("background_writeback:\n  enabled: true\n")
    with pytest.raises(ValueError, match="all old/new"):
        load_config(path)
    path.write_text("freeze_background: false\nbackground_writeback:\n  enabled: true\nadaptive_density:\n  enabled: true\n")
    with pytest.raises(ValueError, match="prune-only"):
        load_config(path)


def test_joint_receipt_enforces_exact_restored_background(tmp_path, monkeypatch):
    import edgs_inpaint_io as io
    path, _, new = fixture(tmp_path)
    mask_path = tmp_path / "new.npy"; np.save(mask_path, new)
    cfg = load_config()
    cfg["freeze_background"] = False
    cfg["background_writeback"]["enabled"] = True
    inputs = {}
    for key in ("initialization", "lama", "camera"):
        placeholder = tmp_path / (key + ".json")
        io.write_json(placeholder, {})
        inputs[key] = io.record(placeholder)
    initial = dict(artifact_id="initial", point_count=len(new),
        request=dict(config={key: cfg[key] for key in ("init", "matcher", "seed")}),
        inputs={key: inputs[key] for key in ("lama", "camera")},
        outputs=dict(ply=io.record(path), editable_mask=io.record(mask_path)))
    monkeypatch.setattr(io, "validate_init", lambda *args, **kwargs: initial)
    monkeypatch.setattr(io, "verify_targets", lambda *args, **kwargs: None)
    model = JointGaussians(path, new, device="cpu", freeze_background=False)
    with torch.no_grad():
        model.xyz.add_(.01)
    permitted, stats = restore_background(model, np.array([False, True, False, False, False, False]))
    output = tmp_path / "result.ply"; model.save(output)
    mapping = tmp_path / "rows.npy"; np.save(mapping, model.retained_rows.numpy())
    permitted_path = tmp_path / "writeback.npy"; np.save(permitted_path, permitted)
    summary = tmp_path / "summary.json"; io.write_json(summary, dict(config=cfg["background_writeback"], **stats))

    def receipt():
        return io.seal(io.JOINT_KIND, inputs=inputs,
            outputs=dict(ply=io.record(output), retained_rows=io.record(mapping),
                writeback_mask=io.record(permitted_path), writeback_summary=io.record(summary)),
            request=dict(inputs=inputs, config=cfg), initialization_id="initial", point_count=len(new),
            parameters=dict(iterations=cfg["iterations"], pruning=cfg["pruning"], freeze_background=False,
                background_writeback=cfg["background_writeback"]))
    assert io.validate_joint(receipt())["point_count"] == len(new)
    vertex = PlyData.read(str(output))["vertex"].data.copy()
    vertex["opacity"][0] += 1
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(output))
    with pytest.raises(ValueError, match="frozen field: opacity"):
        io.validate_joint(receipt())
