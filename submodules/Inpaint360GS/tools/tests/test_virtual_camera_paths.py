"""CPU geometry, exact camera contract, and configurable frame-count regressions."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.pose_utils import generate_ellipse_path, generate_virtual_path, viewmatrix
from utils.virtual_camera_manifest import (
    build_virtual_camera_manifest, check_camera_request, load_virtual_camera_manifest,
    write_virtual_camera_manifest, require_declared_camera_manifest, trajectory_diagnostics,
    trajectory_svg,
    hemisphere_spiral_layout, HEMISPHERE_ALGORITHM_REVISION,
)


def training_views():
    views = []
    for angle in np.linspace(0, 2*np.pi, 25)[:-1]:
        p = np.array([3*np.cos(angle), 2*np.sin(angle), .3*np.sin(2*angle)+.5])
        c2w = np.eye(4)
        c2w[:3] = viewmatrix(p - [0, 0, .1], [0, 0, 1], p)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        views.append(SimpleNamespace(R=w2c[:3, :3].T, T=w2c[:3, 3], FoVx=1., FoVy=.8))
    return views


def records(poses):
    return [SimpleNamespace(R=p[:3, :3].T, T=p[:3, 3], FoVx=1., FoVy=.8,
        image_name=f"{i:05d}", image_width=32, image_height=24, znear=.01, zfar=100.,
        trans=np.zeros(3), scale=1.) for i, p in enumerate(poses)]


@pytest.mark.parametrize("count", [2, 5, 7, 30, 60, 90, 120])
def test_circle_legacy_preservation(count):
    views = training_views()
    old = generate_ellipse_path(views, n_frames=count, is_circle=True, circle_radius=.7)
    circle, _ = generate_virtual_path(views, count, circle_radius=.7)
    np.testing.assert_array_equal(circle, old)


@pytest.mark.parametrize("count", [2, 5, 7, 30, 60, 90, 120, 180])
@pytest.mark.parametrize("elevation", [1., 85., 89.9999])
def test_hemisphere_geometry_and_absolute_scene_up(count, elevation):
    views = training_views()
    old, baseline = generate_virtual_path(views, 30, circle_radius=.7)
    poses, info = generate_virtual_path(views, count, "hemisphere", .7, elevation)
    np.testing.assert_array_equal(poses[0], old[0])
    assert info["center_pca"] == baseline["center_pca"]
    assert info["radius_pca"] == baseline["radius_pca"]
    c2w = np.linalg.inv(poses)
    normalized = np.asarray(info["world_to_pca"]) @ c2w
    positions = normalized[:, :3, 3]
    center, radius = np.asarray(info["center_pca"]), info["radius_pca"]
    np.testing.assert_allclose(np.linalg.norm(positions-center, axis=-1), radius, atol=1e-12)
    h = (positions[:, 2]-center[2])/radius
    assert len(poses) == count
    expected_h = np.linspace(0., np.sin(np.deg2rad(elevation)), count)
    np.testing.assert_allclose(h, expected_h, atol=1e-12)
    assert np.all(np.diff(h) >= -1e-12)
    assert abs(h[-1]-np.sin(np.deg2rad(elevation))) < 1e-12
    assert np.sum(abs(h) < 1e-12) == 1
    rotations = normalized[:, :3, :3]
    np.testing.assert_allclose(rotations.transpose(0,2,1) @ rotations,
                               np.broadcast_to(np.eye(3), rotations.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-12)
    look = np.asarray(info["focus_pca"])-positions
    look /= np.linalg.norm(look, axis=-1, keepdims=True)
    np.testing.assert_allclose(rotations[:, :, 2], look, atol=1e-12)
    assert np.isfinite(poses).all()
    # Absolute scene-up orientation, not just adjacent continuity: catches roll drift.
    right = np.cross(info['scene_up_pca'], -look)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    np.testing.assert_allclose(rotations[:, :, 0], right, atol=1e-9)
    assert np.unique(np.round(positions, 12), axis=0).shape[0] == count
    phi = np.arcsin(expected_h)
    phi[-1] = np.deg2rad(elevation)
    expected = info['start_azimuth_rad'] + 2*np.pi*phi/info['pitch_rad']
    delta = positions-center
    np.testing.assert_allclose(delta[:, :2]/np.linalg.norm(delta[:, :2], axis=1, keepdims=True),
                               np.column_stack([np.cos(expected),np.sin(expected)]), atol=1e-6)
    # Build the same poses independently from absolute up, with no history.
    for i, p in enumerate(positions):
        basis = viewmatrix(p-info['focus_pca'], info['scene_up_pca'], p)[:, :3]
        basis[:, 1:3] *= -1
        np.testing.assert_allclose(rotations[i], basis, atol=1e-9)


def test_new_schema_roundtrip_and_no_torch_dependency(tmp_path):
    poses, info = generate_virtual_path(training_views(), 60, "hemisphere")
    path = tmp_path / "cameras.json"
    write_virtual_camera_manifest(path, records(poses), iteration=2000, circle_radius=1., trajectory=info)
    payload = load_virtual_camera_manifest(path)
    assert payload["schema_version"] == 2 and payload["frame_count"] == 60
    check_camera_request(payload, "hemisphere", 60, 85)
    for mode, count, angle in [("circle", 60, 85), ("hemisphere", 90, 85), ("hemisphere", 60, 80)]:
        with pytest.raises(ValueError, match="changed"):
            check_camera_request(payload, mode, count, angle)
    script = ("from utils.virtual_camera_manifest import load_virtual_camera_manifest; "
              "import sys; load_virtual_camera_manifest(sys.argv[1]); assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", script, str(path)], check=True)
    raw = json.loads(path.read_text())
    raw["cameras"][0]["T"][0] += .01
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="artifact_id"):
        load_virtual_camera_manifest(path)


@pytest.mark.parametrize("mode,count,angle", [("bad",30,85), ("circle",True,85),
    ("hemisphere",1,85), ("hemisphere",0,85), ("hemisphere",2.5,85),
    ("hemisphere",90,90), ("hemisphere",90,0), ("hemisphere",90,float('nan'))])
def test_invalid_requests(mode, count, angle):
    with pytest.raises(ValueError):
        generate_virtual_path(training_views(), count, mode, max_elevation_deg=angle)


def test_old_identity_and_unused_options():
    views = training_views()
    poses, _ = generate_virtual_path(views)
    baseline = generate_ellipse_path(views, n_frames=30, is_circle=True)
    assert build_virtual_camera_manifest(records(poses), iteration=2000, circle_radius=1.) == \
        build_virtual_camera_manifest(records(baseline), iteration=2000, circle_radius=1.)
    check_camera_request(build_virtual_camera_manifest(records(poses), iteration=2000, circle_radius=1.),
                         "circle", 30, "unused")


def test_nondefault_fallback_and_diagnostics(tmp_path):
    for config in ({"virtual_camera_path": "hemisphere"}, {"virtual_camera_count": 7}):
        with pytest.raises(ValueError, match="no circle fallback"):
            require_declared_camera_manifest(config, None)
    poses, info = generate_virtual_path(training_views(), 60, "hemisphere")
    views = records(poses)
    path = tmp_path / "cameras.json"
    write_virtual_camera_manifest(path, views, iteration=2000, circle_radius=1., trajectory=info)
    require_declared_camera_manifest(dict(virtual_camera_path="hemisphere", virtual_camera_count=60), path)
    with pytest.raises(ValueError, match="changed"):
        require_declared_camera_manifest(dict(virtual_camera_path="hemisphere", virtual_camera_count=90), path)
    stats = trajectory_diagnostics(views, info)
    assert len(stats["frames"]) == 60
    assert abs(stats["frames"][-1]["elevation_deg"]-85) < 1e-10
    assert "00059" in trajectory_svg(stats)
    assert stats["nearest_camera_angle_deg"]["minimum"] > 0
    assert max(abs(f['scene_up_roll_deg']) for f in stats['frames']) < 1e-8
    assert stats['scene_up_roll']['max_abs_deg'] < 1e-8
    assert stats['scene_up_roll']['undefined_frames'] == 0
    assert stats['surface_coverage']['cell_area_cv'] < .25
    assert np.all(np.diff([f['azimuth_unwrapped_deg'] for f in stats['frames']]) > 0)


def test_spherical_coverage_improves_with_count():
    hmax = np.sin(np.deg2rad(85))
    h, theta = np.meshgrid((np.arange(64)+.5)/64*hmax,
                          (np.arange(256)+.5)/256*2*np.pi, indexing="ij")
    h, theta = h.ravel(), theta.ravel()
    r = np.sqrt(1-h*h)
    probes = np.stack([r*np.cos(theta), r*np.sin(theta), h], -1)
    worst = []
    for count in (30, 60, 90, 120):
        poses, info = generate_virtual_path(training_views(), count, "hemisphere")
        matrices = np.asarray(info["world_to_pca"]) @ np.linalg.inv(poses)
        unit = (matrices[:, :3, 3]-info["center_pca"])/info["radius_pca"]
        dots = probes @ unit.T
        worst.append(np.arccos(np.clip(dots.max(1), -1, 1)).max())
        areas = np.bincount(dots.argmax(1), minlength=count)
        assert areas.min() > 0
        assert areas.std()/areas.mean() < .25
    assert all(b < a for a, b in zip(worst, worst[1:]))
    assert worst[-1] < .6 * worst[0]


def test_sampling_metadata_and_revision_cannot_be_reused():
    poses, info = generate_virtual_path(training_views(), 90, 'hemisphere')
    for key, value in [('algorithm_revision', HEMISPHERE_ALGORITHM_REVISION-1),
                       ('orientation', 'unknown'), ('sampling', 'unknown'),
                       ('scene_up_pca', [0, 0, 2]), ('pitch_rad', .5), ('turns', 1.)]:
        invalid = copy.deepcopy(info)
        invalid[key] = value
        with pytest.raises(ValueError):
            build_virtual_camera_manifest(records(poses), iteration=2000,
                                          circle_radius=1., trajectory=invalid)
    invalid = dict(frame_count=90, trajectory=dict(info, algorithm_revision=HEMISPHERE_ALGORITHM_REVISION-1))
    with pytest.raises(ValueError, match='new RUN_NAME'):
        check_camera_request(invalid, 'hemisphere', 90)


def test_all_integer_budgets():
    for n in range(2, 401):
        offsets, pitch = hemisphere_spiral_layout(n)
        assert offsets.shape == (n, 3) and pitch > 0
        np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), 1.)
        np.testing.assert_allclose(offsets[:, 2], np.linspace(0, np.sin(np.deg2rad(85)), n))


def test_scene_up_under_world_rotation_and_large_focus_offset(monkeypatch):
    import utils.pose_utils as pose_utils
    # A tilted/translated world must not be mistaken for PCA +z; the focus is
    # far below the equator, making the old cumulative roll particularly large.
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    views = training_views()
    for v in views:
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = v.R.T, v.T
        c2w = np.linalg.inv(w2c)
        c2w[:3, :3] = rotation @ c2w[:3, :3]
        c2w[:3, 3] = rotation @ c2w[:3, 3] + [4, -2, 8]
        w2c = np.linalg.inv(c2w)
        v.R, v.T = w2c[:3, :3].T, w2c[:3, 3]
    monkeypatch.setattr(pose_utils, 'focus_point_fn', lambda _: np.array([.2, -.1, -1.]))
    poses, info = generate_virtual_path(views, 120, 'hemisphere')
    stats = trajectory_diagnostics(records(poses), info)
    assert stats['scene_up_roll']['max_abs_deg'] < 1e-9
    assert stats['scene_up_roll']['undefined_frames'] == 0
    assert info['turns'] > 6


def test_scene_up_singularity_is_explicit(monkeypatch):
    # Numerical pole singularities must not switch to an accumulating mode.
    offsets, pitch = hemisphere_spiral_layout(30)
    offsets[-1] = [0, 0, 1]
    import utils.virtual_camera_manifest as contract
    monkeypatch.setattr(contract, 'hemisphere_spiral_layout', lambda *_: (offsets, pitch))
    with pytest.raises(ValueError, match='scene up is parallel'):
        generate_virtual_path(training_views(), 30, 'hemisphere')


def test_lama_and_local_targets_non30(tmp_path):
    from tools.tests.test_paintmesh_normal import NormalFixture
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/paintmesh"))
    from local_geometry_io import verify_targets
    f = NormalFixture(tmp_path, frames=60)
    poses, trajectory = generate_virtual_path(training_views(), 60, "hemisphere")
    camera_views = records(poses)
    for c in camera_views:
        c.image_width, c.image_height = 9, 8
    write_virtual_camera_manifest(f.camera_path, camera_views, iteration=2000,
                                  circle_radius=1., trajectory=trajectory)
    # Rebind the synthetic render to the new, validated camera artifact.
    from virtual_render_io import identity, write_json
    render_path = f.render_root / "render_manifest.json"
    render = json.loads(render_path.read_text())
    render["camera_artifact_id"] = json.loads(f.camera_path.read_text())["artifact_id"]
    render.pop("artifact_id")
    render["artifact_id"] = identity(render)
    write_json(render_path, render)
    f.prepare(); f.make_valid_outputs(); f.make_normal_outputs(); f.validate()
    assert len(verify_targets(f.completion_manifest, f.camera_path)[0]["frames"]) == 60
