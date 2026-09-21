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
    assert load_config()["init"]["confidence_min"] == .5
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
@pytest.mark.parametrize("depth,normal", [(False, False), (True, False), (True, True), (False, True)])
def test_gpu_joint_worker_resume_and_optional_targets(tmp_path, depth, normal):
    sys.path.insert(0, str(REPO / "submodules/Inpaint360GS"))
    from tools.tests.test_paintmesh_normal import NormalFixture
    from edgs_inpaint_io import identity, verify_targets
    fixture = NormalFixture(tmp_path, frames=2, shape=(24, 24))
    # Two distinct exact cameras; update producer identity before completion.
    cam = read_json(fixture.camera_path)
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
    xyz = np.array([(x, y, 2.) for x in np.linspace(-.4, .4, 5) for y in np.linspace(-.4, .4, 5)], np.float32)
    support = dict(xyz=xyz, rgb=np.full_like(xyz, .4), spacing=np.array(.2))
    cfg = load_config(train_depth=depth, train_normal=normal, iterations=3, geometry_from_iter=0, geometry_ramp_iters=1)
    initial_ply = root / "point_cloud.ply"
    editable = compose(bg_path, support, cfg["init"], initial_ply)
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
    config.write_text("checkpoint_interval: 1\ndebug:\n  interval: 1\n")
    joint_root, joint_manifest = tmp_path / "joint", tmp_path / "joint.json"
    command = [sys.executable, str(REPO / "submodules/EDGS/tools/train_paintmesh_pgsr.py"),
        "--initialization", str(manifest), "--lama", str(fixture.completion_manifest),
        "--camera", str(fixture.camera_path), "--edgs-config", str(edgs_config), "--config", str(config),
        "--output-root", str(joint_root), "--manifest", str(joint_manifest), "--iterations", "3",
        "--geometry-from-iter", "0", "--geometry-ramp-iters", "1",
        "--train-depth", str(depth), "--train-normal", str(normal)]
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
    assert valid["parameters"]["iterations"] == 3
    assert (joint_root / "debug/final.jpg").exists()
    assert (joint_root / "debug/step_000001.jpg").exists()
    before = joint_manifest.read_bytes()
    result = subprocess.run([*command, "--validate-only"], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert joint_manifest.read_bytes() == before
    result = subprocess.run([*command, "--iterations", "4", "--validate-only"], text=True, capture_output=True)
    assert result.returncode != 0


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
        assert stats["confidence_min"] == .5
        assert stats["sampled"] <= stats["cycle_passed"] <= stats["confidence_passed"] <= stats["hole_candidates"]
    with np.load(root / "support.npz", allow_pickle=False) as support:
        assert (support["confidence"][support["source_kind"] == 0] >= .5).all()
    before = manifest.read_bytes()
    result = subprocess.run([*command, "--validate-only"], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert manifest.read_bytes() == before
    changed = subprocess.run([*command, "--match-confidence-min", "0.9", "--validate-only"], text=True, capture_output=True)
    assert changed.returncode != 0 and "request differs" in changed.stderr
    assert manifest.read_bytes() == before
