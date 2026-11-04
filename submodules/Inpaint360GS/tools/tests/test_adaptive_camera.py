"""Surface planning, exact-camera provenance and peer-path regressions."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.pose_utils import generate_virtual_path, viewmatrix
from utils.virtual_camera_manifest import (
    adaptive_policy, build_virtual_camera_manifest, check_camera_request,
    load_virtual_camera_manifest, require_declared_camera_manifest,
    trajectory_diagnostics, trajectory_svg, write_virtual_camera_manifest,
)
from tools.tests.test_virtual_camera_paths import records as legacy_records


def records(poses):
    result = legacy_records(poses)
    for r in result:
        r.FoVx, r.FoVy = 1.2, .9
    return result


def fixture(tilted=False):
    grid = np.linspace(-3, 3, 55)
    x, y = np.meshgrid(grid, grid)
    keep = (abs(x) > .4) | (abs(y) > .35)
    background = np.column_stack([x[keep], y[keep], np.zeros(keep.sum())])
    x, y, z = np.meshgrid(np.linspace(-.35, .35, 7), np.linspace(-.3, .3, 7), [.1, .3, .6])
    target = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]) if tilted else np.eye(3)
    shift = np.array([3., -2., 5.]) if tilted else np.zeros(3)
    views = []
    for i, a in enumerate(np.linspace(0, 2 * np.pi, 18, endpoint=False)):
        p = np.array([3 * np.cos(a), 2.5 * np.sin(a), 2. + .1 * np.cos(2 * a)])
        c2w = np.eye(4)
        c2w[:3] = viewmatrix(p - [0, 0, .1], [0, 0, 1], p)
        c2w[:3, :3] = rotation @ c2w[:3, :3]
        c2w[:3, 3] = rotation @ p + shift
        c2w[:3, 1:3] *= -1
        w = np.linalg.inv(c2w)
        views.append(SimpleNamespace(R=w[:3, :3].T, T=w[:3, 3], FoVx=1.2, FoVy=.9,
                                     image_name=f"train_{i:03d}"))
    mesh_vertices = np.array([[-3.,-3.,0.], [3.,-3.,0.], [3.,3.,0.], [-3.,3.,0.]]) @ rotation.T + shift
    geometry = dict(mesh_vertices=mesh_vertices, mesh_triangles=np.array([[0,1,2],[0,2,3]]),
                    target_object_radius=.65, target_xyz=target @ rotation.T + shift, background_xyz=background @ rotation.T + shift,
                    inputs=[dict(path=f"/synthetic/{n}.ply", sha256=str(i) * 64, size_bytes=100)
                            for i, n in enumerate(("target", "background"))])
    geometry["inputs"].append(dict(path="/synthetic/mesh.ply", sha256="f"*64, kind="retained_mesh", size_bytes=100))
    return views, geometry


def generate(count=18, tilted=False, **overrides):
    views, geometry = fixture(tilted)
    cfg = adaptive_policy().options(dict(dict(tracking_max_rotation_deg=180,
        tracking_max_motion_fraction=1, tracking_max_scale_ratio=3), **overrides))
    poses, trajectory = generate_virtual_path(views, count, "adaptive", adaptive_options=cfg,
                                              adaptive_geometry=geometry)
    return poses, trajectory


@pytest.mark.parametrize("count", [6, 18, 30, 90])
def test_repair_surface_budget_and_actual_camera_geometry(count):
    poses, t = generate(count)
    p = build_virtual_camera_manifest(records(poses), iteration=2000, circle_radius=1., trajectory=t)
    assert p["schema_version"] == 2 and p["frame_count"] == count
    assert t["surface_fit"]["kind"] == "support_plane"
    assert abs(t["surface_normal_pca"][2]) > .99
    assert all(f["role"] == "sweep" for f in t["frames"])
    assert np.mean([f["weight"] for f in t["frames"]]) == pytest.approx(1)
    matrices = np.asarray(t["world_to_pca"]) @ np.linalg.inv(poses)
    assert len(np.unique(np.round(matrices[:, :3, 3], 10), axis=0)) == count
    # All modes preserve the same similarity scale in the persisted R blocks.
    scale = np.cbrt(np.linalg.det(np.asarray(t["world_to_pca"])[:3, :3]))
    np.testing.assert_allclose(np.linalg.det(np.asarray(poses)[:, :3, :3]), scale ** 3)
    for frame in t["frames"]:
        assert frame["visible_fraction"] >= t["options"]["min_visible_fraction"]
    assert t["geometry_pairs"]
    assert not t["planner"]["appearance_consistency_checked"]
    stats = trajectory_diagnostics(records(poses), t)
    assert stats["scene_up_roll"]["max_abs_deg"] < 1e-8
    assert len(stats["adaptive_planning"]["frames"]) == count
    assert f"{count - 1:05d}" in trajectory_svg(stats)


def test_deterministic_and_tilted_world():
    a, x = generate(18, tilted=True)
    b, y = generate(18, tilted=True)
    np.testing.assert_array_equal(a, b)
    assert x == y
    build_virtual_camera_manifest(records(a), iteration=2000, circle_radius=1., trajectory=x)
    assert x["scene_up_pca"] != [0., 0., 0.]


def test_roundtrip_policy_changes_and_cpu_only_reader(tmp_path):
    poses, t = generate()
    path = tmp_path / "cameras.json"
    write_virtual_camera_manifest(path, records(poses), iteration=2000, circle_radius=1., trajectory=t)
    loaded = load_virtual_camera_manifest(path)
    check_camera_request(loaded, "adaptive", 18, "unused", t["options"])
    require_declared_camera_manifest(dict(virtual_camera_path="adaptive", virtual_camera_count=18,
        virtual_adaptive_options=t["options"]), path)
    for mode, count, cfg in (("circle", 18, None), ("adaptive", 30, t["options"]),
                             ("adaptive", 18, dict(t["options"], inclination_range_deg=[25, 45]))):
        with pytest.raises(ValueError, match="changed"):
            check_camera_request(loaded, mode, count, adaptive_options=cfg)
    with pytest.raises(ValueError, match="no circle fallback"):
        require_declared_camera_manifest(dict(virtual_camera_path="adaptive"), None)
    subprocess.run([sys.executable, "-c",
        "from utils.virtual_camera_manifest import load_virtual_camera_manifest; import sys; "
        "load_virtual_camera_manifest(sys.argv[1]); assert 'torch' not in sys.modules and 'open3d' not in sys.modules", str(path)], check=True)


@pytest.mark.parametrize("change", [
    lambda t: t["frames"][0].update(weight=2),
    lambda t: t["frames"][0].update(visible_roi_indices=[-1]),
    lambda t: t["geometry_pairs"][0].update(j=999),
    lambda t: t["geometry_pairs"][0].update(angle_deg=89),
    lambda t: t.update(surface_normal_pca=[0, 0, 2]),
    lambda t: t.update(geometry_inputs=[]),
    lambda t: t.update(algorithm_revision=0),
    lambda t: t["frames"][0].update(incidence_deg=0),
])
def test_invalid_or_stale_planning_metadata_cannot_be_published(change):
    poses, t = generate()
    change(t)
    with pytest.raises(ValueError):
        build_virtual_camera_manifest(records(poses), iteration=2000, circle_radius=1., trajectory=t)


@pytest.mark.parametrize("cfg", [dict(unknown=True), dict(train_fraction=1), dict(train_fraction=True),
    dict(candidate_multiplier=0), dict(texture_angles_deg=[90]), dict(roi_samples=8),
    dict(distance_ratio_bounds=[2, 1]), dict(use_view_weights="true"), dict(pair_min_angle_deg=50),
    dict(min_roi_fraction=.5), dict(target_roi_fraction=float("nan")), dict(inclination_range_deg=[50,30])])
def test_invalid_options_fail_early(cfg):
    with pytest.raises(ValueError):
        adaptive_policy().options(cfg)


def test_explicit_missing_geometry_and_disabled_weights():
    views, geometry = fixture()
    with pytest.raises(ValueError, match="require target"):
        generate_virtual_path(views, 18, "adaptive")
    for n in (2, 5, 513):
        with pytest.raises(ValueError, match="6,512"):
            generate_virtual_path(views, n, "adaptive", adaptive_geometry=geometry)
    poses, t = generate(use_view_weights=False)
    assert all(f["weight"] == 1 for f in t["frames"])
    build_virtual_camera_manifest(records(poses), iteration=2000, circle_radius=1., trajectory=t)


def test_weak_surface_fit_is_declared_not_silently_called_a_plane():
    views, geometry = fixture()
    rng = np.random.default_rng(1)
    geometry["background_xyz"] = rng.uniform(-3, 3, (2500, 3))
    surface = adaptive_policy().repair_surface(geometry["target_xyz"], geometry["background_xyz"],
        np.array([[2, 0, 2], [0, 2, 2], [-2, 0, 2]]), np.array([0, 0, 1.]), adaptive_policy().options())
    assert surface["fit"]["kind"] == "target_proxy"


def test_occluder_rejection_is_based_on_geometry():
    _, geometry = fixture()
    policy = adaptive_policy()
    cfg = policy.options()
    surface = policy.repair_surface(geometry["target_xyz"], geometry["background_xyz"],
        np.array([[2, 0, 2], [0, 2, 2], [-2, 0, 2]]), np.array([0, 0, 1.]), cfg)
    p = np.array([1., 0., 2.])
    clear, _ = policy.assess_candidate(p, surface, geometry["background_xyz"], [0, 0, 1], [1.2, .9], cfg)
    assert clear is not None
    x, y = np.meshgrid(np.linspace(-2, 2, 80), np.linspace(-2, 2, 80))
    roof = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
    blocked, reason = policy.assess_candidate(p, surface, np.concatenate([geometry["background_xyz"], roof]),
                                               [0, 0, 1], [1.2, .9], cfg)
    assert blocked is None and reason == "occlusion"


def test_edgs_consumes_planned_pairs_and_weights_without_legacy_fallback():
    import torch
    from utils.virtual_camera_manifest import virtual_views_from_manifest
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "submodules/EDGS"))
    from source.paintmesh_edgs_init import camera, select_pairs
    poses, t = generate()
    payload = build_virtual_camera_manifest(records(poses), iteration=2000, circle_radius=1., trajectory=t)
    base = SimpleNamespace(world_view_transform=torch.eye(4), image_width=32, image_height=24)
    views = virtual_views_from_manifest(base, payload)
    np.testing.assert_allclose([v.paintmesh_view_weight for v in views], [f["weight"] for f in t["frames"]])
    pairs = select_pairs([camera(v) for v in views], 4)
    permitted = {(p["i"], p["j"]) for p in t["geometry_pairs"]}
    assert set(pairs) <= permitted and pairs
    assert all(a < b for a, b in pairs)
    for v in views:
        v.paintmesh_geometry_neighbors = []
    with pytest.raises(ValueError, match="no usable adaptive"):
        select_pairs([camera(v) for v in views], 4)


def test_lama_and_joint_targets_reuse_exact_adaptive_cameras(tmp_path):
    from tools.tests.test_paintmesh_normal import NormalFixture
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/paintmesh"))
    from edgs_inpaint_io import verify_targets
    from virtual_render_io import identity, write_json
    f = NormalFixture(tmp_path, frames=18)
    poses, t = generate()
    views = records(poses)
    for v in views:
        v.image_width, v.image_height = 9, 8
    write_virtual_camera_manifest(f.camera_path, views, iteration=2000, circle_radius=1., trajectory=t)
    p = f.render_root / "render_manifest.json"
    render = json.loads(p.read_text())
    render["camera_artifact_id"] = json.loads(f.camera_path.read_text())["artifact_id"]
    render.pop("artifact_id")
    render["artifact_id"] = identity(render)
    write_json(p, render)
    f.prepare(); f.make_valid_outputs(); f.make_normal_outputs(); f.validate()
    _, _, cameras = verify_targets(f.completion_manifest, f.camera_path, use_depth=True, use_normal=True)
    assert cameras["frame_count"] == 18 and cameras["trajectory"]["type"] == "adaptive"


def test_shipped_config_matches_defaults_and_legacy_clears_planning_attributes(tmp_path):
    import torch
    from utils.virtual_camera_manifest import virtual_views_from_manifest
    repo = Path(__file__).resolve().parents[4]
    assert adaptive_policy().load_options(repo / "scripts/paintmesh/configs/adaptive_cameras.json") == adaptive_policy().options()
    poses, t = generate()
    adaptive = build_virtual_camera_manifest(records(poses), iteration=2000, circle_radius=1., trajectory=t)
    base = SimpleNamespace(world_view_transform=torch.eye(4), image_width=32, image_height=24)
    view = virtual_views_from_manifest(base, adaptive)[0]
    legacy_poses, _ = generate_virtual_path(fixture()[0])
    legacy = build_virtual_camera_manifest(records(legacy_poses), iteration=2000, circle_radius=1.)
    old = virtual_views_from_manifest(view, legacy)
    assert all(not hasattr(v, "paintmesh_geometry_neighbors") and not hasattr(v, "paintmesh_view_weight") for v in old)


def test_geometry_loader_balances_multiple_targets_and_records_inputs(tmp_path):
    from plyfile import PlyData, PlyElement
    policy = adaptive_policy()
    model = tmp_path / "work_model"
    root = model / "point_cloud_object_removal/iteration_2000"
    root.mkdir(parents=True)
    rng = np.random.default_rng(2)
    def write(name, xyz):
        a = np.zeros(len(xyz), dtype=[(k, "f4") for k in ("x", "y", "z", "opacity")])
        for i, key in enumerate("xyz"):
            a[key] = xyz[:, i]
        a["opacity"] = 2.
        PlyData([PlyElement.describe(a, "vertex")]).write(str(root / name))
    write("point_cloud_1.ply", rng.normal(size=(2000, 3)) * .1 + [-2, 0, .5])
    write("point_cloud_2.ply", rng.normal(size=(20, 3)) * .1 + [2, 0, .5])
    write("point_cloud.ply", rng.normal(size=(500, 3)))
    mesh_root = tmp_path / "removed_mesh"
    mesh_root.mkdir(exist_ok=True)
    mesh_path = mesh_root / "geometry.ply"
    mesh_path.write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n-3 -3 0\n3 -3 0\n0 3 0\n3 0 1 2\n")
    def record(path):
        return dict(path=str(path), size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    (mesh_root / "semantic_manifest.json").write_text(json.dumps(dict(complete=True,inputs=dict(
        gaussian_ply=record(root / "point_cloud.ply"), mesh=record(mesh_path)))))
    cfg = policy.options(dict(target_samples=32, background_samples=128))
    geometry = policy.load_geometry(model, 2000, [1, 2], cfg)
    assert len(geometry["target_xyz"]) == 32
    assert (geometry["target_xyz"][:, 0] > 0).sum() == 16
    assert len(geometry["inputs"]) == 5
    before = copy.deepcopy(geometry["inputs"])
    write("point_cloud_2.ply", rng.normal(size=(20, 3)) * .1 + [2, 0, .5])
    after = policy.load_geometry(model, 2000, [1, 2], cfg)["inputs"]
    assert before[0] == after[0] and before[1]["sha256"] != after[1]["sha256"]
    declared = mesh_root / "semantic_manifest.json"
    original = json.loads(declared.read_text())
    changed = copy.deepcopy(original)
    changed["inputs"]["gaussian_ply"]["path"] = str(root / "point_cloud_1.ply")
    declared.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="all-selected"):
        policy.load_geometry(model, 2000, [1, 2], cfg)
    changed = copy.deepcopy(original)
    changed["inputs"]["mesh"]["size_bytes"] += 1
    declared.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="stale"):
        policy.load_geometry(model, 2000, [1, 2], cfg)


def test_custom_json_controls_layout_without_changing_total_budget(tmp_path):
    p = tmp_path / "custom.json"
    p.write_text(json.dumps(dict(inclination_range_deg=[25,45], use_view_weights=False,
        tracking_max_rotation_deg=180, tracking_max_motion_fraction=1, tracking_max_scale_ratio=3)))
    cfg = adaptive_policy().load_options(p)
    views, geometry = fixture()
    poses, t = generate_virtual_path(views, 18, "adaptive", adaptive_options=cfg, adaptive_geometry=geometry)
    assert len(poses) == 18 and all(f["role"] == "sweep" for f in t["frames"])
    assert all(f["weight"] == 1 for f in t["frames"])
    assert [t["frames"][i]["incidence_deg"] for i in (0,-1)] == pytest.approx([25,45])
    build_virtual_camera_manifest(records(poses), iteration=2000, circle_radius=1., trajectory=t)


@pytest.mark.parametrize("tilted", [False, True])
def test_open_double_sweep_actual_cameras_and_no_closing_edge(tilted):
    views, geometry = fixture(tilted)
    poses, t = generate_virtual_path(views, path_type="adaptive", adaptive_geometry=geometry)
    build_virtual_camera_manifest(records(poses), iteration=2000, circle_radius=1., trajectory=t)
    repeat, again = generate_virtual_path(views, 180, "adaptive", adaptive_geometry=geometry)
    np.testing.assert_array_equal(poses, repeat)
    assert t == again and t["algorithm_revision"] == 3
    assert t["planner"]["sweep_views"] == 180
    assert t["continuous_path"]["closed"] is False
    assert t["continuous_path"]["turns"] == 2
    assert len(t["tracking_edges"]) == 179
    assert [(e["i"],e["j"]) for e in t["tracking_edges"]] == [(i,i+1) for i in range(179)]
    assert [t["frames"][i]["incidence_deg"] for i in (0,-1)] == pytest.approx([30,50])
    assert all(e["rotation_deg"] <= 5 + 1e-7 and e["scale_ratio"] <= 1.03 + 1e-7 and
               e["overlap"] >= .85 and e["motion_fraction"] <= .02 for e in t["tracking_edges"])
    assert t["frames"][-1]["azimuth_rad"]-t["frames"][0]["azimuth_rad"] == pytest.approx(4*np.pi)
    assert trajectory_diagnostics(records(poses), t)["scene_up_roll"]["max_abs_deg"] < 1e-8


@pytest.mark.parametrize("cfg", [dict(tracking_max_rotation_deg=1),
    dict(tracking_max_motion_fraction=1e-6), dict(tracking_max_scale_ratio=1), dict(min_context_support=1)])
def test_sweep_rejects_bad_frames_or_edges_without_another_strategy(cfg):
    views, geometry = fixture()
    with pytest.raises(ValueError, match="no fallback"):
        generate_virtual_path(views, 180, "adaptive", adaptive_options=cfg, adaptive_geometry=geometry)


@pytest.mark.parametrize("change", [
    lambda t: t["continuous_path"].update(reference_distance_pca=2),
    lambda t: t["continuous_path"].update(start_phase_rad=.1),
    lambda t: t["continuous_path"].update(closed=True),
    lambda t: t["continuous_path"].update(turns=1),
    lambda t: t["continuous_path"].update(distance_scale=2),
    lambda t: t["tracking_edges"].insert(0,dict(t["tracking_edges"][0], i=17,j=0)),
    lambda t: t["tracking_edges"][0].update(rotation_deg=181),
    lambda t: t["tracking_edges"][0].update(motion_fraction=0),
    lambda t: t["tracking_edges"][0].update(scale_ratio=1),
    lambda t: t["frames"][0].update(roi_fraction=.02),
    lambda t: t["frames"][0].update(context_support=.1),
    lambda t: t["frames"][0].update(path_parameter=.01),
    lambda t: t["planner"]["mesh"].update(source_triangles=0),
])
def test_sweep_metadata_cannot_hide_geometry_or_a_closing_edge(change):
    poses,t=generate()
    change(t)
    with pytest.raises(ValueError):
        build_virtual_camera_manifest(records(poses), iteration=2000, circle_radius=1., trajectory=t)


@pytest.mark.parametrize("revision", [1,2])
def test_historical_adaptive_is_readable_but_has_no_generation_entry(revision):
    from virtual_render_io import read_camera_manifest
    path=Path(__file__).with_name("fixtures")/f"adaptive_revision{revision}.json"
    raw=json.loads(path.read_text());t=raw["trajectory"]
    loaded=load_virtual_camera_manifest(path)
    assert loaded["trajectory"]==t
    assert read_camera_manifest(path)["artifact_id"]==raw["artifact_id"]
    require_declared_camera_manifest(dict(virtual_camera_path="adaptive",virtual_camera_count=raw["frame_count"],
        virtual_adaptive_options=t["options"]),path)
    with pytest.raises(ValueError,match="changed"):
        check_camera_request(loaded,"adaptive",raw["frame_count"])
    with pytest.raises(ValueError,match="known keys"):
        adaptive_policy().options(t["options"])


@pytest.mark.parametrize("cfg", [dict(path_strategy="selected"),dict(path_strategy="continuous_ring"),
    dict(ring_angles_deg=[20,40]),dict(tracking_max_rotation_deg=0),dict(min_context_support=1.1),
    dict(tracking_max_scale_ratio=.9),dict(tracking_min_overlap=False),dict(inclination_range_deg=[30,55])])
def test_removed_strategies_and_invalid_continuity_options(cfg):
    with pytest.raises(ValueError):
        adaptive_policy().options(cfg)


def test_mesh_queries_reject_an_occluder_and_crossing_segments():
    policy=adaptive_policy()
    vertices=np.array([[-10.,-10.,1.],[10.,-10.,1.],[10.,10.,1.],[-10.,10.,1.]])
    mesh=policy.MeshGuard(dict(vertices=vertices,triangles=np.array([[0,1,2],[0,2,3]])),
                          [np.array([-1.,-1.,0.]),np.array([1.,1.,2.])])
    # All triangle vertices lie outside the crop; their interiors still block rays.
    assert mesh.stats["queried_triangles"]==2
    assert not mesh.visible(np.array([0.,0.,2.]),np.array([[0.,0.,0.]]),.01)[0]
    assert not mesh.clear_segments(np.array([[0.,0.,2.],[0.,0.,0.]]),.01)[0]
    assert mesh.clear_segments(np.array([[0.,0.,2.],[.5,0.,2.]]),.01)[0]


def test_missing_mesh_is_not_replaced_by_point_geometry():
    views,g=fixture()
    g.pop("mesh_vertices");g.pop("mesh_triangles")
    with pytest.raises(ValueError,match="requires a retained triangle mesh"):
        generate_virtual_path(views,180,"adaptive",adaptive_geometry=g)


def test_mesh_without_local_faces_has_no_known_surface_hits():
    policy=adaptive_policy()
    mesh=policy.MeshGuard(dict(vertices=np.array([[-10.,-10.,5.],[10.,-10.,5.],[0.,10.,5.]]),
                              triangles=np.array([[0,1,2]])),
                          [np.array([-1.,-1.,0.]),np.array([1.,1.,2.])])
    assert mesh.stats["source_triangles"]==1 and mesh.stats["queried_triangles"]==0
    assert mesh.visible(np.array([0.,0.,2.]),np.array([[0.,0.,0.]]),.01).all()
    assert mesh.clear_segments(np.array([[0.,0.,2.],[0.,0.,0.]]),.01).all()
    with pytest.raises(ValueError,match="leaves"):
        mesh.visible(np.array([0.,0.,6.]),np.array([[0.,0.,0.]]),.01)
