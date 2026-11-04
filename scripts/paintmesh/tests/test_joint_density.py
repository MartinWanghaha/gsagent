"""Dynamic topology must preserve frozen rows, Adam history and resume state."""
import copy
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData, PlyElement
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
sys.path.insert(0, str(REPO / "submodules/EDGS"))
from edgs_inpaint_io import check_preserved, load_config
from source.paintmesh_joint_density import AdaptiveJointGaussians


def fixture(tmp_path):
    fields = [*"xyz", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "obj_dc_0"]
    data = np.zeros(5, dtype=[(key, "f4") for key in fields])
    data["z"], data["rot_0"], data["opacity"], data["obj_dc_0"] = 2, 1, 2, 1
    for key in ("scale_0", "scale_1", "scale_2"):
        data[key] = np.log([.02, .02, .02, .001, .04])
    path = tmp_path / "initial.ply"
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))
    new = np.array([False, False, False, True, True])
    trainable = new.copy(); trainable[1] = True
    model = AdaptiveJointGaussians(path, new, device="cpu", trainable_mask=trainable)
    return model, model.optimizer(load_config()["optimizer"]), path, new, trainable


def step(model, optimizer):
    optimizer.zero_grad()
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()


def test_clone_split_optimizer_resume_and_frozen_ply(tmp_path):
    model, optimizer, source, new, trainable = fixture(tmp_path)
    step(model, optimizer)
    moment = optimizer.state[model.features]["exp_avg"].clone()
    before_old = model.xyz[0].detach().clone()
    # High gradients on local OLD and both NEW points: old must never grow.
    model.gradient_sum.fill_(1); model.gradient_count.fill_(1)
    cfg = dict(grad_threshold=.0002, percent_dense=.01, max_new_points=10)
    torch.manual_seed(17)
    counts = model.densify(optimizer, cfg, extent=1.)
    assert counts["cloned"] == 1 and counts["split_parents"] == 1 and counts["split_children"] == 2
    assert len(model.indices) == 5 and len(model.retained_rows) == 7
    assert int((model.indices == 1).sum()) == 1
    torch.testing.assert_close(model.xyz[0], before_old, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[model.features]["exp_avg"][:2], moment[:2])
    assert optimizer.state[model.features]["exp_avg"][2:].count_nonzero() == 0
    assert not model.gradient_sum.any()
    assert len(optimizer.state) == len(model.parameter_names)

    # Drop a child, keep a local old row, then resume with moments and statistics.
    with torch.no_grad():
        model.opacity[-1] = -12
    assert model.prune(optimizer) == 1
    model.gradient_sum.fill_(.03); model.gradient_count.fill_(2)
    checkpoint = copy.deepcopy(dict(local=model.local_state(), optimizer=optimizer.state_dict()))
    restored_dir = tmp_path / "restored"
    restored_dir.mkdir()
    restored, _, _, _, _ = fixture(restored_dir)
    restored.restore_local(checkpoint["local"])
    resumed_optimizer = restored.optimizer(load_config()["optimizer"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    for value in (model, restored):
        assert value.density_counts["pruned_new"] == 1
    step(model, optimizer); step(restored, resumed_optimizer)
    for key in model.parameter_names:
        torch.testing.assert_close(getattr(model, key), getattr(restored, key), rtol=0, atol=0)
    torch.testing.assert_close(model.gradient_sum, restored.gradient_sum, rtol=0, atol=0)
    output = tmp_path / "trained.ply"
    model.save(output)
    rows = model.retained_rows.numpy()
    check_preserved(source, output, new, retained_rows=rows, trainable_mask=trainable, new_descendants=True)
    result = PlyData.read(str(output))["vertex"].data.copy()
    initial = PlyData.read(str(source))["vertex"].data
    np.testing.assert_array_equal(result[:2], initial[[0, 2]])
    with pytest.raises(ValueError, match="mapping"):
        check_preserved(source, output, new, retained_rows=rows, trainable_mask=trainable)
    result["x"][0] += 1
    PlyData([PlyElement.describe(result, "vertex")]).write(str(output))
    with pytest.raises(ValueError, match="frozen field"):
        check_preserved(source, output, new, retained_rows=rows, trainable_mask=trainable, new_descendants=True)


def test_density_budget_and_actual_screen_gradients(tmp_path):
    model, optimizer, *_ = fixture(tmp_path)
    viewspace = torch.zeros((5, 3), requires_grad=True)
    (viewspace[:, 0] * torch.tensor([0., 0., 3., 4., 5.])).sum().backward()
    model.accumulate(dict(viewspace_points=viewspace, visibility_filter=torch.ones(5, dtype=torch.bool),
                          radii=torch.ones(5)))
    cfg = dict(grad_threshold=.0002, percent_dense=.01, max_new_points=3)
    counts = model.densify(optimizer, cfg, extent=1.)
    assert counts["split_parents"] == 1 and counts["cloned"] == 0
    assert counts["budget_limited"] == 1
    assert int(torch.as_tensor(model.new_mask)[model.indices].sum()) == 3
    counts = model.densify(optimizer, cfg, extent=1.)
    assert counts["events"] == 2 and counts["split_parents"] == 1


@pytest.mark.parametrize("section", ["enabled: 1", "interval: 0", "grad_threshold: .nan", "percent_dense: -1",
                                     "max_new_points: 0", "from_iter: 5000"])
def test_bad_density_config_rejected(tmp_path, section):
    path = tmp_path / "bad.yaml"
    path.write_text("adaptive_density:\n  " + section + "\n")
    with pytest.raises(ValueError, match="adaptive_density"):
        load_config(path)


def test_default_keeps_density_disabled():
    assert load_config()["adaptive_density"]["enabled"] is False
