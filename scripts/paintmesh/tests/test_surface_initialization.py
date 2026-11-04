"""Nonplanar RGB-anchored depth initialization, independent of training losses."""
import argparse
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData, PlyElement
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts/paintmesh"), str(REPO / "submodules/EDGS")]
from edgs_inpaint_io import (load_config, config_arguments, resolve_config,
    initialization_config, check_initialization_config)
from source.paintmesh_edgs_init import camera, project, compose, unproject
from source.paintmesh_surface_init import (correct_surface_support, rgb_anchors,
    triangulate_tracks, depth_field, fit_correction, grid_basis, close_camera_pair_graph)


def fixture():
    views, targets = [], []
    y, x = np.mgrid[:128, :128]
    pixels = np.column_stack((x.ravel(), y.ravel())).astype(float)
    for i in range(3):
        matrix = torch.eye(4); matrix[3, 0] = (i - 1) * .5
        views.append(SimpleNamespace(world_view_transform=matrix, image_height=128,
            image_width=128, FoVx=1.2, FoVy=1.2, image_name=f"{i:05d}"))
        cam = camera(views[-1])
        rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(cam["k"]).T
        cx, cy = cam["c2w"][:2, 3]; rx, ry = rays[:, 0], rays[:, 1]
        a = .2*rx**2 + .1*ry**2; b = .4*cx*rx + .2*cy*ry - 1
        c = 2 + .2*cx**2 + .1*cy**2; discriminant = b*b - 4*a*c
        hit = discriminant >= 0
        d = np.zeros(len(pixels)); d[hit] = 2*c / (-b[hit] + np.sqrt(discriminant[hit]))
        world = unproject(pixels, d, cam)
        mask = (hit & (np.abs(world[:, 0]) < .65) & (np.abs(world[:, 1]) < .65)).reshape(128, 128)
        depth = d.reshape(128, 128).astype(np.float32); depth[mask] += .15
        rgb = np.stack((x/127., y/127., np.full(x.shape, .4))).astype(np.float32)
        targets.append(dict(mask=torch.from_numpy(mask), original_mask=mask.copy(),
            depth=torch.from_numpy(depth), rgb=torch.from_numpy(rgb)))
    x, y = np.meshgrid(np.linspace(-.55, .55, 19), np.linspace(-.55, .55, 19))
    points = np.column_stack((x.ravel(), y.ravel(), 2+.2*x.ravel()**2+.1*y.ravel()**2))
    cams = [camera(v) for v in views]
    pairs = [(a, b, project(points, cams[a])[0], project(points, cams[b])[0], np.ones(len(points)))
             for a, b in ((0, 1), (0, 2), (1, 2))]
    support = dict(xyz=(points + [0, 0, .4]).astype(np.float32), frame=np.zeros(len(points), int),
        pixel_uv=pairs[0][2], other_frame=np.ones(len(points), int), other_uv=pairs[0][3],
        confidence=np.ones(len(points)), source_kind=np.zeros(len(points), np.int8), spacing=np.array(.03),
        normal=np.zeros_like(points), normal_valid=np.zeros(len(points), bool), rgb=np.ones_like(points)*.4)
    x, y = np.meshgrid(np.arange(-1., 1.01, .03), np.arange(-1., 1.01, .03))
    outside = (np.abs(x) > .65) | (np.abs(y) > .65)
    bg = np.column_stack((x[outside], y[outside], 2+.2*x[outside]**2+.1*y[outside]**2))
    cfg = load_config(init_depth=True, init_normal=False, surface_init_mode="depth_anchor")
    cfg["surface_init"].update(grid_pixels=8, anchor_radius_pixels=24)
    return support, bg, views, targets, cfg, pairs


def initialize(f):
    support, bg, views, targets, cfg, pairs = f
    return correct_surface_support(support, bg, views, targets, cfg["surface_init"],
                                   init_config=cfg["init"], rgb_pairs=pairs)


def test_curved_surface_correction_rays_colors_and_no_input_mutation():
    data = fixture(); support, bg, views, targets, cfg, pairs = data
    before = copy.deepcopy(support); before_depth = [t["depth"].clone() for t in targets]
    result, report, model = initialize(data)
    assert len(result["xyz"]) > 100
    xyz = result["xyz"]
    true_z = 2 + .2*xyz[:, 0]**2 + .1*xyz[:, 1]**2
    assert np.median(np.abs(xyz[:, 2] - true_z)) < .025
    # Curvature must survive; fitting a single plane cannot pass this criterion.
    assert np.ptp(true_z) > .045
    assert result["spacing"] == support["spacing"] and not result["normal_valid"].any()
    assert (result["source_kind"] == 2).all() and (result["other_frame"] == -1).all()
    assert (result["depth_votes"] >= 2).all() and model["anchor_in_original_mask"] > 0
    assert model["corrected_depth_usage"].startswith("initialization only")
    for key in before:
        np.testing.assert_array_equal(support[key], before[key])
    for i, target in enumerate(targets):
        assert torch.equal(target["depth"], before_depth[i])
        rows = result["frame"] == i
        uv, _, _ = project(xyz[rows], camera(views[i]))
        np.testing.assert_allclose(uv, result["pixel_uv"][rows], atol=1e-5)
        np.testing.assert_allclose(result["rgb"][rows, :2], uv/127., atol=1e-6)


def test_third_edge_closure_and_depth_independent_anchor_coordinates():
    support, _, views, _, cfg, pairs = fixture(); cams = [camera(v) for v in views]
    anchors, stats = rgb_anchors(support, cams, cfg["surface_init"], cfg["init"], pairs)
    assert stats["accepted"] > 200
    support["xyz"][:] += 100  # Old depth-refined XYZ has no say in RGB triangulation.
    again, _ = rgb_anchors(support, cams, cfg["surface_init"], cfg["init"], pairs)
    np.testing.assert_array_equal(anchors["xyz"], again["xyz"])
    a, b, uv, other, score = pairs[-1]
    pairs[-1] = (a, b, uv, other + 10, score)
    rejected, _ = rgb_anchors(support, cams, cfg["surface_init"], cfg["init"], pairs)
    assert len(rejected["xyz"]) == 0


def test_new_matching_completes_triangle_without_assuming_valid_tracks():
    views = fixture()[2]; cams = [camera(v) for v in views]
    original = [(0, 1), (0, 2)]
    assert close_camera_pair_graph(original, cams) == [(0, 1), (0, 2), (1, 2)]
    assert original == [(0, 1), (0, 2)]


def test_no_anchors_never_fills_with_plane_or_raw_depth_and_writes_diagnostics():
    support, bg, views, targets, cfg, pairs = fixture()
    records = []
    with pytest.raises(ValueError, match="no sufficiently anchored"):
        correct_surface_support(support, bg, views, targets, cfg["surface_init"],
            init_config=cfg["init"], rgb_pairs=pairs[:2], artifact_callback=lambda m, a: records.append((m, a)))
    assert records[0][0]["status"] == "insufficient_support"
    assert records[0][0]["output_points"] == 0


def test_outside_mask_and_unanchored_regions_are_not_corrected():
    support, _, views, targets, cfg, pairs = fixture(); cams = [camera(v) for v in views]
    anchors, _ = rgb_anchors(support, cams, cfg["surface_init"], cfg["init"], pairs)
    changed, domain, field, stats = depth_field(cams[0], targets[0], anchors, 0, .03, cfg["surface_init"])
    mask = targets[0]["mask"].numpy(); raw = targets[0]["depth"].numpy()
    assert domain.any() and not domain[~mask].any()
    np.testing.assert_array_equal(changed[~mask], raw[~mask])
    empty = copy.deepcopy(anchors); empty["uv"][:] = [1, 1]
    changed, domain, field, stats = depth_field(cams[0], targets[0], empty, 0, .03, cfg["surface_init"])
    assert not domain.any(); np.testing.assert_array_equal(changed, raw)


def test_depth_step_does_not_export_left_anchors_to_another_layer():
    support, _, views, targets, cfg, pairs = fixture(); cams = [camera(v) for v in views]
    anchors, _ = rgb_anchors(support, cams, cfg["surface_init"], cfg["init"], pairs)
    # Anchor only the left layer, with a depth discontinuity across the image.
    keep = anchors["uv"][:, 0, 0] < 64
    anchors = {k: v[keep] for k, v in anchors.items()}
    target = copy.deepcopy(targets[0]); target["depth"][:, 64:] += 1
    changed, domain, _, _ = depth_field(cams[0], target, anchors, 0, .03, cfg["surface_init"])
    assert domain[:, :64].any() and not domain[:, 64:].any()
    np.testing.assert_array_equal(changed[:, 64:], target["depth"].numpy()[:, 64:])


def test_cross_view_inconsistency_cannot_publish_points(monkeypatch):
    import source.paintmesh_surface_init as module
    original = module.depth_field
    def inconsistent(cam, target, anchors, frame, gap, cfg):
        depth, domain, field, stats = original(cam, target, anchors, frame, gap, cfg)
        if frame == 2:
            depth[domain] += .8
        return depth, domain, field, stats
    monkeypatch.setattr(module, "depth_field", inconsistent)
    with pytest.raises(ValueError, match="no sufficiently anchored"):
        initialize(fixture())


def test_repeated_camera_is_degenerate_not_high_confidence():
    data = fixture(); cam = camera(data[2][0]); uv = data[5][0][2]
    xyz, errors, cov = triangulate_tracks([cam, cam, cam], [uv, uv, uv])
    assert not np.isfinite(errors).any()
    assert (np.linalg.eigvalsh(cov)[:, -1] > 1e20).all()


def test_scaled_camera_reprojection_and_covariance():
    data = fixture(); cameras = [camera(v) for v in data[2]]
    points = data[0]["xyz"].astype(float) - [0, 0, .4]
    for c in cameras:
        c["w2c"][:3] *= .171
        c["c2w"] = np.linalg.inv(c["w2c"])
    uv = [project(points, c)[0] for c in cameras]
    xyz, errors, cov = triangulate_tracks(cameras, uv)
    np.testing.assert_allclose(xyz, points, atol=1e-7)
    assert errors.max() < 1e-7 and (np.linalg.eigvalsh(cov) > 0).all()


def test_robust_correction_rejects_bad_anchors_without_flattening_depth():
    rng = np.random.default_rng(8)
    uv = rng.uniform(15, 85, (500, 2)); origin = np.array([0., 0.]); shape = (8, 8); step = 16
    residual = 3 + .01*uv[:, 0] - .02*uv[:, 1]
    noisy = residual.copy(); noisy[:100] += rng.choice([-30, 30], 100)
    control, inlier = fit_correction(uv, noisy, np.full(500, .5), origin, shape, step, np.empty((0, 2)))
    predicted = grid_basis(uv, origin, shape, step) @ control
    assert np.sqrt(np.mean((predicted-residual)**2)) < .3
    assert inlier[100:].mean() > .95 and inlier[:100].mean() < .05


def test_compose_preserves_old_rows_scales_rotation_opacity_color(tmp_path):
    data = fixture(); support, bg, views, targets, cfg, pairs = data
    result, _, _ = initialize(data)
    names = [*"xyz", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "obj_dc_0"]
    vertex = np.zeros(len(bg), dtype=[(k, "f4") for k in names])
    for i, key in enumerate("xyz"): vertex[key] = bg[:, i]
    vertex["rot_0"], vertex["obj_dc_0"] = 1, 2
    source = tmp_path/"background.ply"; output = tmp_path/"result.ply"
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(source))
    editable = compose(source, result, cfg["init"], output)
    produced = PlyData.read(str(output))["vertex"].data
    np.testing.assert_array_equal(produced[~editable], vertex)
    for i in range(3):
        np.testing.assert_allclose(np.exp(produced[f"scale_{i}"][editable]), .03, rtol=1e-6)
        np.testing.assert_allclose(produced[f"f_dc_{i}"][editable], (result["rgb"][:, i]-.5)/.28209479177387814, atol=1e-6)
    np.testing.assert_array_equal(produced["rot_0"][editable], 1.)
    for key in ("rot_1", "rot_2", "rot_3"): assert not produced[key][editable].any()
    np.testing.assert_allclose(produced["opacity"][editable], np.log(.1/.9), atol=1e-6)


def test_legacy_disabled_mode_and_cli_validation(tmp_path):
    cfg = load_config(); legacy = {k: cfg[k] for k in ("init", "matcher", "seed")}
    assert initialization_config(cfg) == legacy
    initial = dict(request=dict(config=legacy)); check_initialization_config(initial, cfg)
    parser = argparse.ArgumentParser(); config_arguments(parser)
    updated = resolve_config(parser.parse_args(["--surface-init-mode", "depth_anchor", "--init-depth", "true"]))
    assert updated["surface_init"]["mode"] == "depth_anchor"
    with pytest.raises(ValueError, match="initialization config"): check_initialization_config(initial, updated)
    with pytest.raises(ValueError, match="use_depth"): resolve_config(parser.parse_args(["--surface-init-mode", "depth_anchor"]))
    with pytest.raises(ValueError, match="use_normal"): resolve_config(parser.parse_args([
        "--surface-init-mode", "depth_anchor", "--init-depth", "true", "--init-normal", "true"]))
    path = tmp_path/"legacy.yaml"
    path.write_text("surface_init:\n  mode: none\n  boundary_pixels: 12\n  fit_distance_factor: 1\n  min_inlier_fraction: .75\n")
    assert load_config(path) == cfg


@pytest.mark.parametrize("value", ["mode: plane", "grid_pixels: 0", "smooth_weight: .nan",
    "min_cross_depth_fraction: 1.1", "sampling_distance_factor: -1", "min_anchor_observations: 2"])
def test_invalid_surface_config_rejected(tmp_path, value):
    path = tmp_path/"bad.yaml"; path.write_text("surface_init:\n  " + value + "\n")
    with pytest.raises(ValueError, match="surface_init"): load_config(path)


def test_depth_anchor_manifest_validates_rays_and_rejects_changed_depth(tmp_path, monkeypatch):
    import edgs_inpaint_io as io
    data = fixture(); support, bg, views, targets, cfg, pairs = data
    evidence = []
    result, report, model = correct_surface_support(support, bg, views, targets, cfg["surface_init"],
        init_config=cfg["init"], rgb_pairs=pairs, artifact_callback=lambda m, a: evidence.append(a))
    names = [*"xyz", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "obj_dc_0"]
    vertex = np.zeros(len(bg), dtype=[(k, "f4") for k in names])
    for i, key in enumerate("xyz"): vertex[key] = bg[:, i]
    vertex["rot_0"] = 1
    source = tmp_path/"background.ply"; output = tmp_path/"result.ply"
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(source))
    editable = compose(source, result, cfg["init"], output)
    np.save(tmp_path/"editable.npy", editable)
    np.savez_compressed(tmp_path/"support.npz", **result)
    np.savez_compressed(tmp_path/"surface_evidence.npz", **evidence[0])
    io.write_json(tmp_path/"model.json", model)
    io.write_json(tmp_path/"camera.json", {})
    io.write_json(tmp_path/"lama.json", {})
    records = {k: io.record(tmp_path/p) for k, p in (
        ("background", "background.ply"), ("camera", "camera.json"), ("lama", "lama.json"))}
    outputs = {k: io.record(tmp_path/p) for k, p in (
        ("ply", "result.ply"), ("editable_mask", "editable.npy"), ("support", "support.npz"),
        ("surface_model", "model.json"), ("surface_evidence", "surface_evidence.npz"))}
    request = dict(config=initialization_config(cfg), inputs=records)
    # Target identity validation has its own integration tests; isolate the new output contract.
    monkeypatch.setattr(io, "verify_targets", lambda *args, **kwargs: None)
    def receipt():
        return io.seal(io.INIT_KIND, request=request, inputs=records, outputs=outputs, point_count=len(editable))
    assert io.validate_init(receipt())["point_count"] == len(editable)
    result["corrected_depth"] += .1
    np.savez_compressed(tmp_path/"support.npz", **result)
    outputs["support"] = io.record(tmp_path/"support.npz")
    with pytest.raises(ValueError, match="corrected camera rays"):
        io.validate_init(receipt())


def test_boundary_only_result_is_not_published_as_hole_completion():
    data = fixture()
    for target in data[3]:
        target["original_mask"][:] = False
    diagnostics = []
    with pytest.raises(ValueError, match="only cover the dilated boundary"):
        correct_surface_support(data[0], data[1], data[2], data[3], data[4]["surface_init"],
            init_config=data[4]["init"], rgb_pairs=data[5], artifact_callback=lambda m, a: diagnostics.append(m))
    assert diagnostics[0]["status"] == "insufficient_support"
    assert diagnostics[0]["output_in_original_mask"] == 0
