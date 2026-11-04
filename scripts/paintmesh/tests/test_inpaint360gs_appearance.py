"""Contracts for official appearance supervision, independent of GS geometry."""
import ast
import json
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
sys.path.insert(0, str(REPO / "submodules/EDGS"))

from edgs_inpaint_io import load_config
from source.paintmesh_joint_losses import OfficialPatchLPIPS, joint_losses


def official_config(tmp_path, **overrides):
    value = dict(appearance=dict(mode="inpaint360gs"),
                 loss=dict(rgb_hole=0., rgb_known_preserve=.2, ssim=.8, lpips=.0005))
    for section, changes in overrides.items():
        value[section].update(changes)
    path = tmp_path / "official.json"
    path.write_text(json.dumps(value))
    return load_config(path)


def test_mode_is_explicit_and_validates_zero_hole_rgb(tmp_path):
    assert load_config()["appearance"]["mode"] == "masked"
    cfg = official_config(tmp_path)
    assert cfg["loss"]["rgb_hole"] == 0
    with pytest.raises(ValueError, match="rgb_hole=0"):
        official_config(tmp_path, loss=dict(rgb_hole=1.))
    with pytest.raises(ValueError, match="positive outside L1"):
        official_config(tmp_path, loss=dict(ssim=0.))
    with pytest.raises(ValueError, match="appearance.mode"):
        official_config(tmp_path, appearance=dict(mode="typo"))
    with pytest.raises(ValueError, match="RGB loss"):
        official_config(tmp_path, appearance=dict(mode="masked"))


def test_official_l1_uses_lama_outside_and_has_no_hole_gradient(tmp_path):
    cfg = official_config(tmp_path)
    cfg["loss"] = {key: 0. for key in cfg["loss"]}
    cfg["loss"]["rgb_known_preserve"] = .2
    mask = torch.zeros(20, 20, dtype=torch.bool)
    mask[5:15, 5:15] = True
    reference = torch.full((3, 20, 20), .4)
    rgb = torch.full_like(reference, .7, requires_grad=True)
    baseline = dict(rgb=torch.full_like(reference, .1), alpha=torch.ones(20, 20))
    target = dict(rgb=reference, mask=mask, rgb_weight=torch.zeros(20, 20))
    package = dict(render=rgb, plane_depth=torch.ones(1, 20, 20),
                   rendered_normal=torch.zeros(3, 20, 20), rendered_alpha=torch.ones(1, 20, 20))
    loss, metrics = joint_losses(package, target, baseline, 0, cfg)
    torch.testing.assert_close(loss, torch.tensor(.06))
    assert metrics["rgb_hole"] == 0
    loss.backward()
    assert rgb.grad[:, mask].abs().sum() == 0
    assert rgb.grad[:, ~mask].abs().sum() > 0


class PatchDistance(torch.nn.Module):
    def forward(self, a, b):
        self.a, self.b = a.detach().clone(), b.detach().clone()
        return (a - b).square().mean((1, 2, 3), keepdim=True)


def perceptual_stub():
    module = OfficialPatchLPIPS.__new__(OfficialPatchLPIPS)
    torch.nn.Module.__init__(module)
    module.network = PatchDistance()
    return module


def official_patch_functions():
    # Execute only the release's pure helpers, without importing its CUDA app.
    path = REPO / "submodules/Inpaint360GS/edit_object_inpaint.py"
    tree = ast.parse(path.read_text())
    names = {"mask_to_bbox", "crop_using_bbox", "divide_into_patches"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == 3
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_native_bbox_patches_match_release_and_update_context():
    module = perceptual_stub()
    rgb = torch.linspace(.1, .9, 3 * 97 * 105).reshape(3, 97, 105).requires_grad_()
    reference = torch.full_like(rgb, .4)
    mask = torch.zeros(97, 105, dtype=torch.bool)
    # Odd-sized 75x81 bbox, including context outside this irregular hole.
    mask[10, 12] = True
    mask[84, 92] = True
    loss = module(rgb, reference, mask)
    functions = official_patch_functions()
    bbox = functions["mask_to_bbox"](mask)
    crop = functions["crop_using_bbox"](rgb, bbox)
    expected = functions["divide_into_patches"](crop[None], 2).squeeze(0) * 2 - 1
    torch.testing.assert_close(module.network.a, expected, rtol=0, atol=0)
    assert module.network.a.shape == (4, 3, 37, 40)
    loss.backward()
    assert rgb.grad[:, 40, 40].abs().sum() > 0  # Context is supervised too.
    assert rgb.grad[:, 0, 0].abs().sum() == 0
    assert rgb.grad[:, 84, :].abs().sum() == 0  # Release omits odd trailing row.
    assert rgb.grad[:, :, 92].abs().sum() == 0


@pytest.mark.parametrize("empty", [False, True])
def test_empty_or_small_bbox_skips_lpips_without_breaking_backward(empty):
    module = perceptual_stub()
    rgb = torch.full((3, 64, 70), .8, requires_grad=True)
    mask = torch.zeros(64, 70, dtype=torch.bool)
    if not empty:
        mask[:62, :64] = True  # 31-pixel patch height, below release threshold.
    loss = module(rgb, torch.full_like(rgb, .4), mask)
    assert loss == 0 and not hasattr(module.network, "a")
    loss.backward()
    assert rgb.grad.abs().sum() == 0
