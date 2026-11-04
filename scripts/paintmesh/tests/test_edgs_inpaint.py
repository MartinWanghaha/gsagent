from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/paintmesh"))
sys.path.insert(0, str(REPO / "submodules/EDGS"))

from edgs_inpaint_io import load_config, config_arguments, resolve_config, check_preserved, claim, validate_init, validate_joint, record, seal, write_json, read_json, INIT_KIND
from source.paintmesh_edgs_init import camera, project, triangulate, unproject, compose, initialize_candidates, select_pairs, filter_roma_matches
from source.paintmesh_joint_data import JointGaussians, read_targets
from source.paintmesh_joint_losses import joint_losses, geometry_ramp
from source.paintmesh_joint_debug import save_debug


def view(index=0, size=24):
    matrix = torch.eye(4)
    matrix[3, 0] = -index * .15
    return SimpleNamespace(world_view_transform=matrix, image_height=size, image_width=size,
                           FoVx=1., FoVy=1., image_name=f"{index:05d}")


def target(size=24, depth=True, normal=True):
    mask = torch.zeros(size, size, dtype=torch.bool)
    mask[2:-2, 2:-2] = True
    t = dict(rgb=torch.full((3, size, size), .4), mask=mask)
    if depth:
        t["depth"] = torch.full((size, size), 2.)
    if normal:
        n = torch.zeros(3, size, size)
        n[2] = -1
        t.update(normal=n, normal_valid=torch.ones(size, size, dtype=torch.bool))
    return t


def gaussian_ply(path):
    fields = [*"xyz", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    fields += [f"obj_dc_{i}" for i in range(16)]
    xy = np.array([(x, y) for x in np.linspace(-1.1, 1.1, 12) for y in np.linspace(-1.1, 1.1, 12)
                   if abs(x) > .5 or abs(y) > .5])
    a = np.zeros(len(xy), dtype=[(key, "f4") for key in fields])
    a["x"], a["y"], a["z"] = xy[:, 0], xy[:, 1], 2
    a["rot_0"], a["opacity"] = 1, 2
    a["scale_0"] = a["scale_1"] = np.log(.15)
    a["scale_2"] = np.log(.02)
    a["obj_dc_0"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(a, "vertex")]).write(str(path))
    return a


@pytest.mark.parametrize("depth,normal", [(False, False), (True, False), (True, True)])
def test_config_modes_and_inherited_supervision(depth, normal):
    cfg = load_config(init_depth=depth, init_normal=normal)
    assert cfg["supervision"] == dict(use_depth=depth, use_normal=normal)
    assert cfg["debug"]["enabled"]
    assert cfg["loss"]["depth"] > 0 if depth else cfg["loss"]["depth"] == 0
    assert cfg["loss"]["lama_normal"] > 0 if normal else cfg["loss"]["lama_normal"] == 0


def test_normal_only_training_and_invalid_initialization():
    cfg = load_config(train_normal=True, train_depth=False)
    assert not cfg["init"]["use_normal"] and cfg["supervision"]["use_normal"]
    with pytest.raises(ValueError, match="requires depth"):
        load_config(init_normal=True)
    with pytest.raises(ValueError):
        load_config(iterations=0)
    for ramp in (0, 2):
        cfg = load_config(geometry_from_iter=2, geometry_ramp_iters=ramp)
        assert geometry_ramp(1, cfg) == 0 and geometry_ramp(4, cfg) == 1


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf"), True, "0.8"])
def test_invalid_match_confidence(value):
    with pytest.raises(ValueError, match="confidence_min"):
        load_config(match_confidence_min=value)


def test_match_threshold_config_and_cli_override(tmp_path):
    import argparse
    config = tmp_path / "thresholds.yaml"
    config.write_text("init:\n  confidence_min: 0.7\n  cycle_pixels: 2.0\n  reprojection_pixels: 1.5\n")
    assert load_config()["init"]["confidence_min"] == .7
    assert load_config(config)["init"]["confidence_min"] == .7
    parser = argparse.ArgumentParser()
    config_arguments(parser)
    cfg = resolve_config(parser.parse_args(["--config", str(config), "--match-confidence-min", "0.9",
        "--match-cycle-pixels", "1", "--match-reprojection-pixels", "0.5"]))
    assert [cfg["init"][k] for k in ("confidence_min", "cycle_pixels", "reprojection_pixels")] == [.9, 1., .5]
    for edge in (0, 1):
        assert load_config(match_confidence_min=edge)["init"]["confidence_min"] == edge
    for override in (dict(match_cycle_pixels=0), dict(match_reprojection_pixels=float("nan"))):
        with pytest.raises(ValueError):
            load_config(**override)


def identity_warp(size=4):
    y, x = np.mgrid[:size, :size]
    xy = (np.stack((x, y), -1) + .5) * 2 / size - 1
    return np.concatenate((xy, xy), -1)


def test_roma_confidence_gate_uses_both_raw_directions_before_sampling():
    warp = identity_warp()
    forward, reverse = np.full((4, 4), .95), np.full((4, 4), .94)
    forward[1, 1], reverse[2, 2] = .2, .1
    mask = np.ones((4, 4), bool)
    cfg = load_config()["init"]
    xy, _, scores, stats = filter_roma_matches(warp, forward, warp, reverse, mask, mask, cfg, np.random.default_rng(0))
    assert len(xy) == 14 < cfg["samples_per_pair"]  # No padding with rejected low-score matches.
    assert not any(np.array_equal(p, [1, 1]) or np.array_equal(p, [2, 2]) for p in xy)
    np.testing.assert_allclose(scores, .94)
    assert stats["hole_candidates"] == 16 and stats["confidence_passed"] == stats["sampled"] == 14
    cfg["confidence_min"] = 1.
    xy, uv, scores, stats = filter_roma_matches(warp, forward, warp, reverse, mask, mask, cfg, np.random.default_rng(0))
    assert xy.shape == uv.shape == (0, 2) and scores.shape == (0,)
    assert stats["sampled"] == 0


def test_roma_reverse_score_is_sampled_at_target_coordinate():
    forward, reverse = identity_warp(), identity_warp()
    forward[..., 2] += .5  # One pixel to the right in B.
    reverse[..., 2] -= .5
    fc, rc = np.full((4, 4), .95), np.full((4, 4), .95)
    rc[1, 2] = .1
    mask = np.ones((4, 4), bool)
    xy, uv, _, stats = filter_roma_matches(forward, fc, reverse, rc, mask, mask, load_config()["init"], np.random.default_rng(0))
    assert len(xy) == 11  # Four out-of-bounds targets and one low reverse score.
    assert not any(np.array_equal(p, [1, 1]) for p in xy)
    assert any(np.array_equal(p, [2, 1]) for p in xy)
    np.testing.assert_allclose(uv - xy, np.tile([1, 0], (11, 1)))
    assert stats["confidence_passed"] == 11


def test_roma_score_interpolation_between_image_and_match_grid():
    forward, reverse = identity_warp(4), identity_warp(8)
    fc, rc = np.full((4, 4), .95), np.full((8, 8), .95)
    rc[:2, :2] = .2
    mask = np.ones((16, 16), bool)
    xy, _, scores, stats = filter_roma_matches(forward, fc, reverse, rc, mask, mask, load_config()["init"], np.random.default_rng(0))
    assert len(xy) == 15 and stats["sampled"] == 15
    assert not any(np.array_equal(p, [1.5, 1.5]) for p in xy)
    assert (scores >= .8).all()


def test_roma_rejects_invalid_scores_coordinates_and_cycle():
    f, r = identity_warp(), identity_warp()
    fc, rc = np.full((4, 4), .95), np.full((4, 4), .95)
    fc.flat[0], rc.flat[1], fc.flat[2], rc.flat[3] = np.nan, np.inf, 0., 1.1
    f[1, 0, 2] = np.nan
    f[1, 1, 0] = 2.  # Invalid source must not be rescued by nearest-boundary sampling.
    f[1, 2, 2] = 2.
    r[1, 3, 2] = np.nan
    r[2, 0, 2] += .5  # One-pixel cycle error.
    cfg = load_config(match_cycle_pixels=.5)["init"]
    mask = np.ones((4, 4), bool)
    xy, _, scores, stats = filter_roma_matches(f, fc, r, rc, mask, mask, cfg, np.random.default_rng(0))
    assert len(xy) > 0 and np.isfinite(scores).all()
    assert not any(np.array_equal(p, [0, 2]) for p in xy)
    assert stats["sampled"] < stats["confidence_passed"] < stats["hole_candidates"]


def test_roma_sampling_never_weakens_threshold():
    warp, mask = identity_warp(8), np.ones((8, 8), bool)
    confidence = np.linspace(.1, 1, 64).reshape(8, 8)
    cfg = load_config(match_confidence_min=.8)["init"]
    cfg["samples_per_pair"] = 2
    result = filter_roma_matches(warp, confidence, warp, confidence, mask, mask, cfg, np.random.default_rng(1))
    repeat = filter_roma_matches(warp, confidence, warp, confidence, mask, mask, cfg, np.random.default_rng(1))
    assert len(result[2]) == 2 and (result[2] >= .8).all()
    np.testing.assert_array_equal(result[0], repeat[0])


def test_triangulation_pixel_convention_and_depth_roundtrip():
    cams = [camera(view(i)) for i in range(2)]
    xyz = np.array([[0., 0., 2.], [.2, -.1, 2.5], [-.1, .15, 3.]])
    uv1, z, _ = project(xyz, cams[0])
    uv2, _, _ = project(xyz, cams[1])
    recovered, valid = triangulate(*cams, uv1, uv2, np.ones(len(xyz)), load_config()["init"])
    assert valid.all()
    np.testing.assert_allclose(recovered, xyz, atol=1e-6)
    np.testing.assert_allclose(unproject(uv1, z, cams[0]), xyz, atol=1e-6)


def test_stricter_reprojection_threshold_rejects_inconsistent_match():
    cams = [camera(view(i)) for i in range(2)]
    xyz = np.array([[0., 0., 2.]])
    uv1, _, _ = project(xyz, cams[0])
    uv2, _, _ = project(xyz, cams[1])
    uv2[:, 1] += 2.
    assert triangulate(*cams, uv1, uv2, np.ones(1), load_config()["init"])[1].all()
    assert not triangulate(*cams, uv1, uv2, np.ones(1), load_config(match_reprojection_pixels=.5)["init"])[1].any()


def fake_matcher(a, b, cfg, rng):
    xyz = np.array([(x, y, 2.) for x in np.linspace(-.45, .45, 9) for y in np.linspace(-.4, .4, 9)])
    return project(xyz, camera(view(0)))[0], project(xyz, camera(view(1)))[0], np.ones(len(xyz))


@pytest.mark.parametrize("depth,normal", [(False, False), (True, False), (True, True)])
def test_candidate_modes_and_compose_preserve_background(tmp_path, depth, normal):
    source = tmp_path / "background.ply"
    bg = gaussian_ply(source)
    cfg = load_config(init_depth=depth, init_normal=normal)
    support, report = initialize_candidates([view(0), view(1)], [target(depth=depth, normal=normal)] * 2,
        np.column_stack([bg[k] for k in "xyz"]), cfg, fake_matcher)
    assert report["rgb_candidates"] > 0 and report["initialized_points"] > 0
    assert support["normal_valid"].any() == normal
    output = tmp_path / "initial.ply"
    editable = compose(source, support, cfg["init"], output)
    points = PlyData.read(str(output))["vertex"].data
    np.testing.assert_array_equal(points[:len(bg)], bg)
    assert editable.sum() == report["initialized_points"]
    if normal:
        assert (points["scale_2"][editable] < points["scale_0"][editable]).all()
        np.testing.assert_allclose(points["rot_1"][editable], 1, atol=1e-6)


def test_rgb_only_has_no_depth_fallback(tmp_path):
    bg = gaussian_ply(tmp_path / "bg.ply")
    no_matches = lambda *args: (np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
    with pytest.raises(ValueError, match="no depth fallback"):
        initialize_candidates([view(0), view(1)], [target(depth=False, normal=False)] * 2,
            np.column_stack([bg[k] for k in "xyz"]), load_config(), no_matches)


def test_initializer_cannot_admit_low_confidence_rgb_candidates(tmp_path):
    bg = gaussian_ply(tmp_path / "bg.ply")
    def weak_matcher(*args):
        uv, other, confidence = fake_matcher(*args)
        return uv, other, confidence * .4
    with pytest.raises(ValueError, match="no depth fallback"):
        initialize_candidates([view(0), view(1)], [target(depth=False, normal=False)] * 2,
            np.column_stack([bg[k] for k in "xyz"]), load_config(), weak_matcher)


@pytest.mark.parametrize("use_depth", [False, True])
def test_disabled_modalities_are_never_opened(tmp_path, monkeypatch, use_depth):
    rgb, mask = tmp_path / "rgb.png", tmp_path / "mask.png"
    Image.fromarray(np.full((24, 24, 3), 100, np.uint8)).save(rgb)
    Image.fromarray(target()["mask"].numpy().astype(np.uint8) * 255).save(mask)
    outputs = dict(color={"path": str(rgb)}, depth={"path": "DO_NOT_OPEN_DEPTH"}, normal={"path": "DO_NOT_OPEN_NORMAL"})
    lama = {"frames": {"00000": {"outputs": outputs}}}
    inputs = {"frames": {"00000": {"outputs": {"color_mask": {"path": str(mask)}}}}}
    def guarded_load(path, **kwargs):
        if use_depth and path == "DO_NOT_OPEN_DEPTH":
            return np.full((24, 24), 2, np.float32)
        pytest.fail("disabled geometry loader was called")
    monkeypatch.setattr(np, "load", guarded_load)
    result = read_targets(lama, inputs, [view()], use_depth=use_depth, use_normal=False)
    assert set(result[0]) == ({"rgb", "mask", "rgb_weight"} | ({"depth", "depth_weight"} if use_depth else set()))


def test_disabled_arrays_cannot_influence_initialization(tmp_path):
    bg = gaussian_ply(tmp_path / "bg.ply")
    xyz = np.column_stack([bg[k] for k in "xyz"])
    a, b = target(), target()
    first, _ = initialize_candidates([view(0), view(1)], [a, a], xyz, load_config(), fake_matcher)
    b["depth"].fill_(float("nan"))
    b["normal"].fill_(float("nan"))
    second, _ = initialize_candidates([view(0), view(1)], [b, b], xyz, load_config(), fake_matcher)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_missing_enabled_geometry_fails_not_silently_falls_back(tmp_path):
    rgb, mask = tmp_path / "rgb.png", tmp_path / "mask.png"
    Image.fromarray(np.zeros((24, 24, 3), np.uint8)).save(rgb)
    Image.fromarray(np.full((24, 24), 255, np.uint8)).save(mask)
    lama = {"frames": {"00000": {"outputs": {"color": {"path": str(rgb)}}}}}
    inputs = {"frames": {"00000": {"outputs": {"color_mask": {"path": str(mask)}}}}}
    with pytest.raises(KeyError, match="depth"):
        read_targets(lama, inputs, [view()], use_depth=True)


def test_projection_with_scaled_camera_and_normal_orientation(tmp_path):
    v = view()
    v.world_view_transform[:3, :3] *= 2
    cam = camera(v)
    xyz = np.array([[.1, .2, 2.]])
    uv, depth, valid = project(xyz, cam)
    np.testing.assert_allclose(unproject(uv, depth, cam), xyz)
    assert valid.all()
    source = tmp_path / "bg.ply"
    gaussian_ply(source)
    n = np.array([[.6, 0, -.8]], np.float32)
    support = dict(xyz=xyz, rgb=np.full((1, 3), .3), normal=n, normal_valid=np.array([True]), spacing=np.array(.1))
    output = tmp_path / "initial.ply"
    compose(source, support, load_config(init_depth=True, init_normal=True)["init"], output)
    q = np.array([PlyData.read(str(output))["vertex"].data[f"rot_{i}"][-1] for i in range(4)])
    from scipy.spatial.transform import Rotation
    rotated = Rotation.from_quat(q[[1, 2, 3, 0]]).apply([0, 0, 1])
    np.testing.assert_allclose(rotated, n[0], atol=1e-6)


@pytest.mark.parametrize("normal", [False, True])
def test_optional_losses_and_debug_without_depth(tmp_path, normal):
    cfg = load_config(train_normal=normal, geometry_from_iter=0, geometry_ramp_iters=0)
    t = target(depth=False, normal=normal)
    package = dict(render=torch.full((3, 24, 24), .3, requires_grad=True),
        plane_depth=torch.full((1, 24, 24), 2.2, requires_grad=True),
        rendered_alpha=torch.full((1, 24, 24), .4, requires_grad=True),
        rendered_normal=torch.cat((torch.full((1, 24, 24), .1), torch.zeros(1, 24, 24), -torch.ones(1, 24, 24))).requires_grad_())
    package["depth_normal"] = target()["normal"]
    baseline = dict(rgb=torch.full((3, 24, 24), .3), alpha=torch.zeros(24, 24))
    loss, metrics = joint_losses(package, t, baseline, 0, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["depth_pixels"] == 0
    assert metrics["normal_pixels"] > 0 if normal else metrics["normal_pixels"] == 0
    assert package["rendered_alpha"].grad.sum() < 0
    save_debug(tmp_path, package, t, metrics, 0)
    assert (tmp_path / "debug/step_000000.jpg").is_file()


def test_masked_perceptual_gradients_and_normalized_inputs():
    from source.paintmesh_joint_losses import MaskedLPIPS
    class Distance(torch.nn.Module):
        def forward(self, a, b):
            assert a.shape[-2] >= 32 and a.shape[-1] >= 32
            assert a.min() >= -1 and a.max() <= 1
            return (a - b).square().mean(1, keepdim=True)
    module = MaskedLPIPS.__new__(MaskedLPIPS)
    torch.nn.Module.__init__(module)
    module.network = Distance();module.padding = 16;module.max_side = 512
    rgb = torch.full((3, 40, 40), .8, requires_grad=True)
    reference = torch.full_like(rgb, .4)
    mask = torch.zeros((40, 40), dtype=torch.bool);mask[10:30, 10:30] = True
    loss = module(rgb, reference, mask)
    assert loss > 0
    loss.backward()
    assert rgb.grad[:, mask].abs().sum() > 0
    assert rgb.grad[:, ~mask].abs().sum() == 0
    assert module(reference, reference, mask) == 0
    assert module(rgb, reference, torch.zeros_like(mask)) == 0


def test_full_image_ssim_constrains_pixels_outside_hole():
    cfg = load_config()
    cfg["loss"] = {k: 0. for k in cfg["loss"]}
    cfg["loss"]["ssim"] = .2
    t = target(size=32, depth=False, normal=False)
    t["mask"][:] = False;t["mask"][12:20, 12:20] = True
    rgb = t["rgb"].clone();rgb[:, 2:6, 2:6] = .9;rgb.requires_grad_()
    p = dict(render=rgb, rendered_alpha=torch.ones(1, 32, 32),
             plane_depth=torch.ones(1, 32, 32), rendered_normal=torch.zeros(3, 32, 32))
    baseline = dict(rgb=t["rgb"], alpha=torch.ones(32, 32))
    loss, metrics = joint_losses(p, t, baseline, 0, cfg)
    assert metrics["rgb_hole"] == 0 and metrics["ssim"] > 0
    loss.backward()
    assert rgb.grad[:, ~t["mask"]].abs().sum() > 0


def test_joint_optimizer_updates_color_opacity_only_new_rows(tmp_path):
    source = tmp_path / "input.ply"
    original = gaussian_ply(source)
    mask = np.arange(len(original)) >= len(original) - 3
    model = JointGaussians(source, mask, device="cpu")
    optimizer = model.optimizer(load_config()["optimizer"])
    before = model.local_state()
    for _ in range(3):
        loss = model.get_features.sum() + model.get_opacity.sum() + model.get_xyz.sum() + model.get_scaling.sum() + model.get_rotation.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for name in ("features", "opacity", "xyz", "scaling", "rotation"):
        assert not torch.equal(getattr(model, name), before[name])
        assert getattr(model, name).shape[0] == 3
    out = tmp_path / "out.ply"
    model.save(out)
    check_preserved(source, out, mask)
    np.testing.assert_array_equal(PlyData.read(str(source))["vertex"].data, original)
    model.restore_local(before)
    for name in before:
        torch.testing.assert_close(getattr(model, name), before[name])


def test_pruning_preserves_background_adam_and_resume(tmp_path):
    source = tmp_path / "input.ply"
    original = gaussian_ply(source)
    # A low-opacity old Gaussian MUST survive, even below the pruning threshold.
    original["opacity"][0] = -12
    PlyData([PlyElement.describe(original, "vertex")]).write(str(source))
    mask = np.arange(len(original)) >= len(original) - 5
    model = JointGaussians(source, mask, device="cpu")
    cfg = load_config()["optimizer"]
    optimizer = model.optimizer(cfg)

    def update(m, opt):
        loss = sum((getattr(m, name).square().sum() for name in m.parameter_names))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    update(model, optimizer)
    with torch.no_grad():
        model.opacity[[0, 3]] = -12
    keep = torch.tensor([False, True, True, False, True])
    before = model.local_state()
    moments = {name: {k: v.clone() for k, v in optimizer.state[getattr(model, name)].items()}
               for name in model.parameter_names}
    assert model.prune(optimizer) == 2
    assert model.prune(optimizer) == 0
    torch.testing.assert_close(model.indices, before["indices"][keep])
    np.testing.assert_array_equal(model.retained_rows[:(~mask).sum()].numpy(), np.flatnonzero(~mask))
    for name in model.parameter_names:
        parameter = getattr(model, name)
        torch.testing.assert_close(parameter, before[name][keep])
        for key, previous in moments[name].items():
            torch.testing.assert_close(optimizer.state[parameter][key],
                                       previous[keep] if previous.shape == before[name].shape else previous)
    # A compacted model must train identically after reconstructing the checkpoint.
    state, opt_state = model.local_state(), optimizer.state_dict()
    restored = JointGaussians(source, mask, device="cpu")
    restored.restore_local(state)
    resumed_optimizer = restored.optimizer(cfg)
    import copy
    resumed_optimizer.load_state_dict(copy.deepcopy(opt_state))
    update(model, optimizer)
    update(restored, resumed_optimizer)
    for key, value in model.local_state().items():
        torch.testing.assert_close(value, restored.local_state()[key], rtol=0, atol=0)
    out = tmp_path / "pruned.ply"
    model.save(out)
    rows = model.retained_rows.numpy()
    result = PlyData.read(str(out))["vertex"].data.copy()
    assert len(result) == len(original) - 2
    assert model.get_xyz.shape[0] == model.get_features.shape[0] == model.get_opacity.shape[0] == len(result)
    np.testing.assert_array_equal(result[:(~mask).sum()], original[~mask])
    np.testing.assert_array_equal(PlyData.read(str(source))["vertex"].data, original)
    check_preserved(source, out, mask, retained_rows=rows)
    with pytest.raises(ValueError):
        check_preserved(source, out, mask)  # Never accept a shrink without ownership.
    with pytest.raises(ValueError, match="frozen background"):
        check_preserved(source, out, mask, retained_rows=np.arange(2, len(original), dtype=np.int64))
    result["obj_dc_0"][-1] += 1
    PlyData([PlyElement.describe(result, "vertex")]).write(str(out))
    with pytest.raises(ValueError, match="frozen field"):
        check_preserved(source, out, mask, retained_rows=rows)


def test_pruning_without_adam_state_and_all_new_failure(tmp_path):
    source = tmp_path / "input.ply"
    original = gaussian_ply(source)
    mask = np.arange(len(original)) >= len(original) - 3
    model = JointGaussians(source, mask, device="cpu")
    optimizer = model.optimizer(load_config()["optimizer"])
    with torch.no_grad():
        model.opacity[1] = -12
    assert model.prune(optimizer) == 1
    assert not optimizer.state
    with torch.no_grad():
        model.opacity.fill_(-12)
    before = model.local_state()
    with pytest.raises(ValueError, match="all new Gaussians"):
        model.prune(optimizer)
    for key, value in before.items():
        torch.testing.assert_close(model.local_state()[key], value)


def test_unfrozen_background_updates_prunes_and_resumes(tmp_path):
    import copy
    source = tmp_path / "input.ply"
    original = gaussian_ply(source)
    mask = np.arange(len(original)) >= len(original) - 3
    model = JointGaussians(source, mask, device="cpu", freeze_background=False)
    optimizer = model.optimizer(load_config()["optimizer"])

    def update(m, opt):
        loss = sum((getattr(m, key).square() + getattr(m, key)).sum() for key in m.parameter_names)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    before = model.local_state()
    update(model, optimizer)
    assert len(model.indices) == len(original)
    for key in model.parameter_names:
        assert not torch.equal(getattr(model, key)[~mask], before[key][~mask])
    # Remove one old and one new row, preserving original identity and Adam.
    with torch.no_grad():
        model.opacity[[0, len(original) - 1]] = -12
    assert model.prune(optimizer) == 2
    state = model.local_state()
    restored = JointGaussians(source, mask, device="cpu", freeze_background=False)
    restored.restore_local(state)
    resumed_optimizer = restored.optimizer(load_config()["optimizer"])
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    update(model, optimizer)
    update(restored, resumed_optimizer)
    for key, value in model.local_state().items():
        torch.testing.assert_close(value, restored.local_state()[key], rtol=0, atol=0)
    out = tmp_path / "unfrozen.ply"
    model.save(out)
    rows = model.retained_rows.numpy()
    assert 0 not in rows and len(original) - 1 not in rows
    check_preserved(source, out, mask, retained_rows=rows, freeze_background=False)
    with pytest.raises(ValueError, match="frozen background"):
        check_preserved(source, out, mask, retained_rows=rows)
    np.testing.assert_array_equal(PlyData.read(str(source))["vertex"].data, original)
    # Unfreezing geometry/appearance never permits semantic-field corruption.
    data = PlyData.read(str(out))["vertex"].data.copy()
    data["obj_dc_0"][0] += 1
    PlyData([PlyElement.describe(data, "vertex")]).write(str(out))
    with pytest.raises(ValueError, match="frozen field"):
        check_preserved(source, out, mask, retained_rows=rows, freeze_background=False)


def test_invalid_freeze_configuration(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("freeze_background: 0\n")
    with pytest.raises(ValueError, match="freeze_background"):
        load_config(config)


def test_local_selection_rejects_distant_projection_and_keeps_footprint_overlap(tmp_path):
    from source.paintmesh_joint_region import select_local_rows
    vertex = gaussian_ply(tmp_path / "region.ply")[:5].copy()
    xyz = np.array([[0, 0, 2], [.55, 0, 2], [0, 0, 20], [-3, 0, 2], [0, 0, 2]], np.float32)
    for i, k in enumerate("xyz"):
        vertex[k] = xyz[:, i]
    for k in ("scale_0", "scale_1", "scale_2"):
        vertex[k] = np.log(.1)
        vertex[k][2] = np.log(50.)  # A giant far Gaussian cannot defeat the 3D gate.
    new = np.array([False, False, False, False, True])
    k = np.array([[30, 0, 20], [0, 30, 20], [0, 0, 1]], float)
    cam = dict(w2c=np.eye(4), k=k, h=40, w=40)
    mask = np.zeros((40, 40), bool); mask[18:23, 18:23] = True
    cfg = load_config()["local_background"];cfg.update(enabled=True, mask_dilation=0)
    trainable, votes, stats = select_local_rows(vertex, new, [cam, cam], [mask, mask], xyz[new], .1, cfg)
    np.testing.assert_array_equal(trainable, [True, True, False, False, True])
    assert votes[1] == 2 and stats["frozen_old"] == 2


def test_local_background_freezes_far_rows_through_update_prune_save_resume(tmp_path):
    source = tmp_path / "input.ply";original = gaussian_ply(source)
    new = np.arange(len(original)) >= len(original) - 3
    trainable = new.copy();trainable[[4, 5]] = True
    original["opacity"][0] = -12
    PlyData([PlyElement.describe(original, "vertex")]).write(str(source))
    model = JointGaussians(source, new, device="cpu", trainable_mask=trainable)
    optimizer = model.optimizer(load_config()["optimizer"])
    loss = sum(getattr(model, k).sum() for k in model.parameter_names)
    loss.backward();optimizer.step()
    with torch.no_grad():
        model.opacity[model.indices == 4] = -12
        model.opacity[model.indices == len(original) - 1] = -12
    assert model.prune(optimizer) == 2
    restored = JointGaussians(source, new, device="cpu", trainable_mask=trainable)
    restored.restore_local(model.local_state())
    assert 0 in restored.retained_rows and 4 not in restored.retained_rows
    out = tmp_path / "local.ply";restored.save(out)
    rows = restored.retained_rows.numpy()
    check_preserved(source, out, new, retained_rows=rows, trainable_mask=trainable)
    result = PlyData.read(str(out))["vertex"].data.copy()
    np.testing.assert_array_equal(result[~trainable[rows]], original[~trainable])
    assert result["x"][np.flatnonzero(rows == 5)[0]] != original["x"][5]
    bad = restored.local_state();bad["indices"] = bad["indices"].clone();bad["indices"][0] = 0
    with pytest.raises(ValueError, match="row ownership"):
        restored.restore_local(bad)
    result["x"][0] += 1
    PlyData([PlyElement.describe(result, "vertex")]).write(str(out))
    with pytest.raises(ValueError, match="frozen field"):
        check_preserved(source, out, new, retained_rows=rows, trainable_mask=trainable)


@pytest.mark.parametrize("setting", ["enabled: 1", "min_views: 0", "mask_dilation: -1", "support_distance_factor: .nan"])
def test_invalid_local_configuration(tmp_path, setting):
    p = tmp_path / "local.yaml";p.write_text("local_background:\n  " + setting + "\n")
    with pytest.raises(ValueError, match="local_background"):
        load_config(p)


@pytest.mark.parametrize("indices", [[0, 1], [-1, 2], [999], [1, 1], [], [2, 1]])
def test_resume_rejects_invalid_pruned_ownership(tmp_path, indices):
    source = tmp_path / "input.ply"
    original = gaussian_ply(source)
    mask = np.arange(len(original)) >= len(original) - 3
    model = JointGaussians(source, mask, device="cpu")
    state = model.local_state()
    state["indices"] = torch.tensor(indices, dtype=torch.int64)
    with pytest.raises(ValueError, match="row ownership"):
        model.restore_local(state)


@pytest.mark.parametrize("setting", ["enabled: 1", "until_iter: -1", "until_iter: true",
                                   "min_opacity: 1", "min_opacity: -.1", "min_opacity: .nan"])
def test_invalid_pruning_configuration(tmp_path, setting):
    config = tmp_path / "pruning.yaml"
    config.write_text("pruning:\n  " + setting + "\n")
    with pytest.raises(ValueError, match="pruning"):
        load_config(config)


def test_joint_receipt_checks_pruned_ownership_and_legacy_outputs(tmp_path, monkeypatch):
    import copy
    import edgs_inpaint_io as io
    source = tmp_path / "initial.ply"
    original = gaussian_ply(source)
    mask = np.arange(len(original)) >= len(original) - 3
    mask_path = tmp_path / "editable.npy"
    np.save(mask_path, mask)
    cfg = load_config()
    placeholders = {key: tmp_path / (key + ".json") for key in ("initialization", "lama", "camera")}
    for path in placeholders.values():
        write_json(path, {})
    inputs = {key: record(path) for key, path in placeholders.items()}
    initial = dict(artifact_id="initial-id", point_count=len(original),
        request=dict(config={k: cfg[k] for k in ("init", "matcher", "seed")}),
        inputs={k: inputs[k] for k in ("lama", "camera")},
        outputs=dict(ply=record(source), editable_mask=record(mask_path)))
    monkeypatch.setattr(io, "validate_init", lambda path: initial)
    monkeypatch.setattr(io, "verify_targets", lambda *args, **kwargs: None)
    model = JointGaussians(source, mask, device="cpu")
    optimizer = model.optimizer(cfg["optimizer"])
    with torch.no_grad():
        model.opacity[1] = -12
    model.prune(optimizer)
    out, mapping = tmp_path / "pruned.ply", tmp_path / "retained.npy"
    model.save(out)
    np.save(mapping, model.retained_rows.numpy())
    outputs = dict(ply=record(out), retained_rows=record(mapping))

    def receipt(config=cfg, produced=outputs, count=len(original)-1):
        parameters = dict(iterations=config["iterations"])
        if "pruning" in config:
            parameters["pruning"] = config["pruning"]
        return seal(io.JOINT_KIND, inputs=inputs, outputs=produced,
            request=dict(inputs=inputs, config=config), initialization_id="initial-id",
            point_count=count, parameters=parameters)

    assert validate_joint(receipt())["point_count"] == len(original)-1
    with pytest.raises(ValueError, match="requires retained row mapping"):
        validate_joint(receipt(produced=dict(ply=record(out)), count=len(original)))
    disabled = copy.deepcopy(cfg)
    disabled["pruning"]["enabled"] = False
    with pytest.raises(ValueError, match="pruning disabled"):
        validate_joint(receipt(config=disabled))
    with pytest.raises(ValueError, match="point count"):
        validate_joint(receipt(count=len(original)))
    # Old fixed-topology manifests remain valid and do not require a new sidecar.
    legacy = copy.deepcopy(cfg)
    legacy.pop("pruning")
    assert validate_joint(receipt(config=legacy, produced=dict(ply=record(source)),
                                  count=len(original)))["point_count"] == len(original)
    unfrozen = copy.deepcopy(cfg)
    unfrozen["freeze_background"] = False
    with pytest.raises(ValueError, match="freezing configuration"):
        validate_joint(receipt(config=unfrozen))
    # The explicit unfrozen producer contract may change/delete old rows.
    all_model = JointGaussians(source, mask, device="cpu", freeze_background=False)
    all_optimizer = all_model.optimizer(cfg["optimizer"])
    with torch.no_grad():
        all_model.opacity[0] = -12
        all_model.xyz.add_(.01)
    all_model.prune(all_optimizer)
    all_model.save(out)
    np.save(mapping, all_model.retained_rows.numpy())
    produced = dict(ply=record(out), retained_rows=record(mapping))
    unfrozen_receipt = seal(io.JOINT_KIND, inputs=inputs, outputs=produced,
        request=dict(inputs=inputs, config=unfrozen), initialization_id="initial-id",
        point_count=len(original)-1, parameters=dict(iterations=cfg["iterations"],
        pruning=cfg["pruning"], freeze_background=False))
    assert validate_joint(unfrozen_receipt)["point_count"] == len(original)-1


def test_request_refuses_changed_inputs_and_unowned_outputs(tmp_path):
    root = tmp_path / "run"
    claim(root, {"mode": "rgb"})
    with pytest.raises(ValueError, match="changed"):
        claim(root, {"mode": "depth"})
    with pytest.raises(ValueError, match="incomplete"):
        claim(tmp_path / "missing", {}, True)


def test_shell_syntax_and_conflicting_pipeline_flags():
    import os
    script = REPO / "scripts/paintmesh/run_inpaint.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(["bash", str(script), "mip-nerf/360_v2", "kitchen", "8", "14", "none", "1"],
        env={**os.environ, "INPAINT_PIPELINE": "edgs-pgsr", "LOCAL_GEOMETRY_REFINE": "true"}, text=True, capture_output=True)
    assert result.returncode and "peer path" in result.stderr


@pytest.mark.skipif(os.environ.get("PAINTMESH_JOINT_GPU_TEST") != "1", reason="opt-in PGSR CUDA integration")
@pytest.mark.parametrize("depth,normal,freeze_background,local", [
    (False, False, True, False), (True, False, True, False), (True, True, True, False), (False, True, True, False),
    (True, True, False, False), (True, True, True, True)])
def test_gpu_joint_worker_resume_and_optional_targets(tmp_path, depth, normal, freeze_background, local,
                                                    adaptive=False, density=False, scale=False):
    sys.path.insert(0, str(REPO / "submodules/Inpaint360GS"))
    from tools.tests.test_paintmesh_normal import NormalFixture
    from edgs_inpaint_io import identity, verify_targets
    fixture = NormalFixture(tmp_path, frames=12 if adaptive else 2, shape=(24, 24))
    # Two distinct exact cameras; update producer identity before completion.
    cam = read_json(fixture.camera_path)
    if adaptive:
        from tools.tests.test_adaptive_camera import fixture as camera_fixture, records as camera_records
        from utils.pose_utils import generate_virtual_path
        from utils.virtual_camera_manifest import write_virtual_camera_manifest
        source_views, geometry = camera_fixture()
        shift = np.array([0., 0., 2.])  # Align the support plane with gaussian_ply().
        geometry["target_xyz"] += shift
        geometry["background_xyz"] += shift
        geometry["mesh_vertices"] += shift
        for v in source_views:
            v.T = v.T - v.R.T @ shift
        poses, trajectory = generate_virtual_path(source_views, 12, "adaptive",
            # Sparse 12-frame fixture; production uses 180 frames and 5 deg.
            adaptive_options=dict(tracking_max_rotation_deg=80, tracking_max_motion_fraction=1,
                                  tracking_max_scale_ratio=3), adaptive_geometry=geometry)
        cameras = camera_records(poses)
        for v in cameras:
            v.image_width = v.image_height = 24
        write_virtual_camera_manifest(fixture.camera_path, cameras, iteration=2000,
                                      circle_radius=1., trajectory=trajectory)
        cam = read_json(fixture.camera_path)
        fixture.cameras = {c["image_name"]: c for c in cam["cameras"]}
    else:
        cam["cameras"][1]["T"][0] = -.15
        cam["artifact_id"] = identity({k: v for k, v in cam.items() if k not in ("artifact_id", "complete", "status")})
        write_json(fixture.camera_path, cam)
    render_path = fixture.render_root / "render_manifest.json"
    render = read_json(render_path)
    render["camera_artifact_id"] = cam["artifact_id"]
    render["artifact_id"] = identity({k: v for k, v in render.items() if k != "artifact_id"})
    write_json(render_path, render)
    fixture.prepare()
    fixture.make_valid_outputs()
    fixture.make_normal_outputs()
    fixture.validate()
    verify_targets(fixture.completion_manifest, fixture.camera_path, use_depth=depth, use_normal=normal)
    root = tmp_path / "init"
    bg_path = tmp_path / "bg.ply"
    bg = gaussian_ply(bg_path)
    if not freeze_background or local:
        bg["opacity"][0] = -12
        if local:
            bg["opacity"][1] = -12  # Far frozen row must survive the same threshold.
        PlyData([PlyElement.describe(bg, "vertex")]).write(str(bg_path))
    xyz = np.array([(x, y, 2.) for x in np.linspace(-.4, .4, 5) for y in np.linspace(-.4, .4, 5)], np.float32)
    support = dict(xyz=xyz, rgb=np.full_like(xyz, .4), spacing=np.array(.2))
    cfg = load_config(train_depth=depth, train_normal=normal, iterations=3, geometry_from_iter=0, geometry_ramp_iters=1)
    initial_ply = root / "point_cloud.ply"
    editable = compose(bg_path, support, cfg["init"], initial_ply)
    # Force a real deletion before the first checkpoint; background stays frozen.
    data = PlyData.read(str(initial_ply))["vertex"].data.copy()
    data["opacity"][np.flatnonzero(editable)[:3]] = -12
    if not freeze_background or local:
        data["opacity"][0] = -12
    PlyData([PlyElement.describe(data, "vertex")]).write(str(initial_ply))
    np.save(root / "editable_mask.npy", editable)
    edgs_config = tmp_path / "edgs.yaml"
    edgs_config.write_text("gs:\n  renderer:\n    backend: pgsr\n  dataset:\n    white_background: false\n")
    removed = tmp_path / "removed.json"
    write_json(removed, {"inputs": {"edgs_config": record(edgs_config)}})
    inputs = dict(background=record(bg_path), lama=record(fixture.completion_manifest),
                  camera=record(fixture.camera_path), removed_model=record(removed))
    request = dict(inputs=inputs, config={k: cfg[k] for k in ("init", "matcher", "seed")})
    receipt = seal(INIT_KIND, request=request, inputs=inputs, point_count=len(editable),
        outputs=dict(ply=record(initial_ply), editable_mask=record(root / "editable_mask.npy")))
    manifest = tmp_path / "init.json"
    write_json(manifest, receipt)
    validate_init(manifest)
    config = tmp_path / "joint.yaml"
    config.write_text("checkpoint_interval: 1\nfreeze_background: " + str(freeze_background).lower() + "\ndebug:\n  interval: 1\n")
    if density:
        config.write_text(config.read_text() + "adaptive_density:\n  enabled: true\n  from_iter: 0\n  until_iter: 3\n  interval: 1\n  grad_threshold: 0.000000000001\n")
    if scale:
        config.write_text(config.read_text() + "appearance:\n  mode: inpaint360gs\nloss:\n  rgb_hole: 0\n  rgb_known_preserve: 0.2\n  ssim: 0.8\n  lpips: 0.0005\n  scale_flatten: 100\n")
    joint_root, joint_manifest = tmp_path / "joint", tmp_path / "joint.json"
    command = [sys.executable, str(REPO / "submodules/EDGS/tools/train_paintmesh_pgsr.py"),
        "--initialization", str(manifest), "--lama", str(fixture.completion_manifest),
        "--camera", str(fixture.camera_path), "--edgs-config", str(edgs_config), "--config", str(config),
        "--output-root", str(joint_root), "--manifest", str(joint_manifest), "--iterations", "3",
        "--geometry-from-iter", "0", "--geometry-ramp-iters", "1",
        "--train-depth", str(depth), "--train-normal", str(normal)]
    trainable = None
    if local:
        from edgs_inpaint_io import SELECTION_KIND
        cfg["local_background"]["enabled"] = True
        config.write_text(config.read_text() + "local_background:\n  enabled: true\nloss:\n  ssim: 0.2\n  lpips: 0.005\n")
        trainable = editable.copy();trainable[np.arange(len(bg))[::3]] = True
        trainable_path = tmp_path / "trainable.npy";np.save(trainable_path, trainable)
        selection_inputs = dict(initialization=record(manifest), lama=record(fixture.completion_manifest), camera=record(fixture.camera_path))
        selection = seal(SELECTION_KIND, request=dict(inputs=selection_inputs, config=cfg["local_background"]),
            inputs=selection_inputs, initialization_id=receipt["artifact_id"], outputs=dict(trainable_mask=record(trainable_path)))
        selection_path = tmp_path / "selection.json";write_json(selection_path, selection)
        command.extend(["--local-selection", str(selection_path)])
    launcher = """
import pathlib, runpy, sys
worker = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(worker.parents[3] / 'scripts/paintmesh'))
import edgs_inpaint_io
original = edgs_inpaint_io.atomic_write
def interrupt(path, callback):
    original(path, callback)
    if pathlib.Path(path).name == 'latest.pth':
        raise RuntimeError('simulated checkpoint interruption')
edgs_inpaint_io.atomic_write = interrupt
sys.argv = sys.argv[1:]
runpy.run_path(str(worker), run_name='__main__')
"""
    interrupted = subprocess.run([sys.executable, "-c", launcher, *command[1:]], text=True, capture_output=True)
    assert interrupted.returncode == 2, interrupted.stdout + interrupted.stderr
    assert "simulated checkpoint" in interrupted.stderr, interrupted.stdout + interrupted.stderr
    assert not joint_manifest.exists()
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    valid = validate_joint(joint_manifest)
    if scale:
        stats = read_json(joint_root / "diagnostics/step_000000.json")
        assert stats["scale_flatten"] > 0 and stats["weight_scale_flatten"] == 100
        assert valid["request"]["config"]["appearance"]["mode"] == "inpaint360gs"
    assert valid["parameters"]["iterations"] == 3
    rows = np.load(valid["outputs"]["retained_rows"]["path"])
    if density:
        density_stats = read_json(joint_root / "diagnostics/final.json")["adaptive_density"]["counters"]
        assert density_stats["events"] == 2
        assert density_stats["split_children"] + density_stats["cloned"] > 0
        assert valid["point_count"] == len(rows)
    else:
        assert valid["point_count"] == len(editable) - 3 - int(not freeze_background or local) == len(rows)
    assert valid["parameters"]["freeze_background"] is freeze_background
    if adaptive:
        import random
        order = list(range(12))
        random.Random(cfg["seed"]).shuffle(order)
        diagnostics = read_json(joint_root / "diagnostics/step_000000.json")
        assert diagnostics["view_weight"] == pytest.approx(trajectory["frames"][order[-1]]["weight"])
    if local:
        assert valid["request"]["appearance_weights"]["vgg_backbone"]["sha256"]
        diagnostics = read_json(joint_root / "diagnostics/step_000000.json")
        assert diagnostics["ssim"] >= 0 and diagnostics["lpips"] > 0
    if freeze_background and not local:
        np.testing.assert_array_equal(rows[:len(bg)], np.arange(len(bg)))
    else:
        assert 0 not in rows
        result_data = PlyData.read(valid["outputs"]["ply"]["path"])["vertex"].data
        old = ~editable[rows]
        assert np.any(result_data["f_dc_0"][old] != data["f_dc_0"][rows][old])
    check_preserved(initial_ply, valid["outputs"]["ply"]["path"], editable, retained_rows=rows,
                    freeze_background=freeze_background, trainable_mask=trainable, new_descendants=density)
    assert (joint_root / "debug/final.jpg").exists()
    assert (joint_root / "debug/step_000001.jpg").exists()
    before = joint_manifest.read_bytes()
    result = subprocess.run([*command, "--validate-only"], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert joint_manifest.read_bytes() == before
    result = subprocess.run([*command, "--iterations", "4", "--validate-only"], text=True, capture_output=True)
    assert result.returncode != 0


@pytest.mark.skipif(os.environ.get("PAINTMESH_JOINT_GPU_TEST") != "1", reason="opt-in adaptive PGSR CUDA integration")
def test_gpu_adaptive_camera_joint_training_and_resume(tmp_path):
    test_gpu_joint_worker_resume_and_optional_targets(tmp_path, False, False, True, False, adaptive=True)


@pytest.mark.skipif(os.environ.get("PAINTMESH_JOINT_GPU_TEST") != "1", reason="opt-in ADC PGSR CUDA integration")
def test_gpu_density_topology_and_checkpoint_resume(tmp_path):
    test_gpu_joint_worker_resume_and_optional_targets(tmp_path, False, False, True, False, density=True)


@pytest.mark.skipif(os.environ.get("PAINTMESH_JOINT_GPU_TEST") != "1", reason="opt-in scale PGSR CUDA integration")
def test_gpu_official_appearance_scale_and_checkpoint_resume(tmp_path):
    test_gpu_joint_worker_resume_and_optional_targets(tmp_path, False, False, True, False, scale=True)


@pytest.mark.skipif(os.environ.get("PAINTMESH_ROMA_GPU_TEST") != "1", reason="opt-in real RoMa initialization (cached weights)")
def test_real_roma_initialization_worker_and_reuse(tmp_path):
    sys.path.insert(0, str(REPO / "submodules/Inpaint360GS"))
    from tools.tests.test_paintmesh_normal import NormalFixture
    from edgs_inpaint_io import identity
    from source.paintmesh_edgs_init import matcher_weights
    weights = matcher_weights(load_config()["matcher"], download=False)
    fixture = NormalFixture(tmp_path, frames=2, shape=(24, 24))
    cam = read_json(fixture.camera_path)
    cam["cameras"][1]["T"][0] = -.15
    cam["artifact_id"] = identity({k: v for k, v in cam.items() if k not in ("artifact_id", "complete", "status")})
    write_json(fixture.camera_path, cam)
    rp = fixture.render_root / "render_manifest.json"
    r = read_json(rp)
    r["camera_artifact_id"] = cam["artifact_id"]
    r["artifact_id"] = identity({k: v for k, v in r.items() if k != "artifact_id"})
    write_json(rp, r)
    fixture.prepare(); fixture.make_valid_outputs(); fixture.make_normal_outputs(); fixture.validate()
    bg, classifier = tmp_path / "bg.ply", tmp_path / "classifier.pth"
    gaussian_ply(bg)
    classifier.write_bytes(b"frozen semantic classifier")
    removed, selection = tmp_path / "removed.json", tmp_path / "selection.json"
    write_json(removed, dict(kind="paintmesh-removed-edgs-model", complete=True, status="complete",
        parameters=dict(target_ids=[14], surrounding_ids=[]), inputs=dict(removed_gaussian_ply=record(bg), classifier=record(classifier))))
    write_json(selection, dict(target_id=[14], surrounding_ids=[]))
    config = tmp_path / "config.yaml"
    import yaml
    config.write_text(yaml.safe_dump(dict(matcher={k: v["path"] for k, v in weights.items()})))
    manifest, root = tmp_path / "init.json", tmp_path / "init"
    command = [sys.executable, str(REPO / "submodules/EDGS/tools/initialize_paintmesh_edgs.py"),
        "--removed-model", str(removed), "--inpaint-config", str(selection), "--lama", str(fixture.completion_manifest),
        "--camera", str(fixture.camera_path), "--output-root", str(root), "--manifest", str(manifest),
        "--config", str(config), "--init-depth", "true", "--init-normal", "true"]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = validate_init(manifest)
    assert receipt["report"]["initialized_points"] > 0
    assert receipt["report"]["normal_oriented_points"] > 0
    assert receipt["report"]["pairs"]
    for pair in receipt["report"]["pairs"]:
        stats = pair["matching"]
        assert stats["confidence_min"] == .7
        assert stats["sampled"] <= stats["cycle_passed"] <= stats["confidence_passed"] <= stats["hole_candidates"]
    with np.load(root / "support.npz", allow_pickle=False) as support:
        assert (support["confidence"][support["source_kind"] == 0] >= .7).all()
    before = manifest.read_bytes()
    result = subprocess.run([*command, "--validate-only"], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert manifest.read_bytes() == before
    changed = subprocess.run([*command, "--match-confidence-min", "0.9", "--validate-only"], text=True, capture_output=True)
    assert changed.returncode != 0 and "request differs" in changed.stderr
    assert manifest.read_bytes() == before
