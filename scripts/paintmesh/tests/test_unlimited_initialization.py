"""A disabled resource cap must preserve geometry and not invent another cap."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
sys.path.insert(0, str(REPO / "submodules/EDGS"))
from edgs_inpaint_io import load_config
from source.paintmesh_edgs_init import initialize_candidates
from test_edgs_inpaint import view, target, gaussian_ply, fake_matcher


def test_yaml_null_disables_cap_and_invalid_caps_fail(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("init:\n  max_points: null\n")
    assert load_config(path)["init"]["max_points"] is None
    for value in (0, -1, True, "unlimited", 1.5):
        path.write_text("init:\n  max_points: " + str(value).lower() + "\n")
        with pytest.raises(ValueError, match="init.max_points"):
            load_config(path)


@pytest.mark.parametrize("depth", [False, True])
def test_unlimited_keeps_same_candidates_and_quality_filters(tmp_path, depth):
    background = gaussian_ply(tmp_path / "background.ply")
    background = np.column_stack([background[k] for k in "xyz"])
    views = [view(0), view(1)]
    targets = [target(depth=depth, normal=False)] * 2
    finite = load_config(init_depth=depth, init_normal=False)
    unlimited = copy.deepcopy(finite)
    unlimited["init"]["max_points"] = None
    expected, old_report = initialize_candidates(views, targets, background, finite, fake_matcher)
    actual, new_report = initialize_candidates(views, targets, background, unlimited, fake_matcher,
                                              rgb_pairs=[])
    assert len(actual["xyz"]) > 1
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert old_report == new_report
    tiny = copy.deepcopy(finite); tiny["init"]["max_points"] = 1
    with pytest.raises(ValueError, match="max_points"):
        initialize_candidates(views, targets, background, tiny, fake_matcher)


def test_active_scene_recipe_has_no_global_cap():
    cfg = load_config(REPO / "scripts/paintmesh/configs/inpaint360_balanced_10000.yaml")
    assert cfg["init"]["max_points"] is None
    assert cfg["init"]["confidence_min"] == .7
    assert cfg["pruning"]["enabled"] and not cfg["adaptive_density"]["enabled"]
