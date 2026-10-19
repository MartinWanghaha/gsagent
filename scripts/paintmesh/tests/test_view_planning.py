"""Synthetic safety, coordinates, exact route budget and schema-3 contracts."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
INPAINT = SCRIPTS.parents[1] / "submodules/Inpaint360GS"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(INPAINT))

from virtual_render_io import camera_contract, identity, write_json
from view_planning_io import load_config, file_record, validate_published_plan, validate_records
from view_planning_geometry import (MeshScene, ObservedFreeSpace, camera_center,
    near_radius, project, world_to_camera, image_rays, certify_segment)
from view_planning_candidates import look_at, sample_target, evaluate_candidate, preview_metrics
from view_planning_selection import select_sequence


def camera(index=0):
    return dict(image_name=f"{index:05d}", R=np.eye(3).tolist(), T=[0., 0., 0.],
        FoVx=1., FoVy=1., image_width=48, image_height=48, znear=.01, zfar=100.,
        trans=[0., 0., 0.], scale=1.)


def plane_z(z):
    return np.array([[-5, -5, z], [5, -5, z], [5, 5, z], [-5, 5, z]]), np.array([[0, 1, 2], [0, 2, 3]])


def scene(vertices, triangles):
    pytest.importorskip("open3d")
    return MeshScene(vertices, triangles)


class Everywhere:
    def contains_ball(self, center, radius):
        return True


def certificate(geometry, a, b, free=None):
    return certify_segment(geometry, free or Everywhere(), a, b, .05,
        margin=1e-5, min_step=.0001, max_depth=20)


def test_scaled_camera_preserves_translation_intrinsics_and_near_plane():
    template = camera()
    template.update(R=(np.eye(3)*3).tolist(), T=[2., -3., 4.], trans=[.2, -.4, .6], scale=2.5)
    center = np.array([2., -1., .3])
    focus = np.array([.1, .5, 1.])
    result = look_at(center, focus, [0, 0, 1], template, 0)
    np.testing.assert_allclose(camera_center(result), center, atol=1e-12)
    np.testing.assert_allclose(project(result, [focus])[0], [[.5, .5]], atol=1e-12)
    np.testing.assert_allclose(np.linalg.svd(world_to_camera(result)[:3, :3])[1], [3, 3, 3])
    expected = template["znear"]/3 * np.sqrt(1+2*np.tan(.5)**2)
    assert near_radius(result) == pytest.approx(expected)
    assert result["trans"] == template["trans"] and result["scale"] == template["scale"]
    origin, directions, lengths = image_rays(result)
    yy, xx = 12, 30
    point = origin + directions[yy, xx]*lengths[yy, xx]*2
    uv, z = project(result, [point])
    np.testing.assert_allclose(uv, [[xx/48, yy/48]], atol=1e-12)
    assert z[0] == pytest.approx(2.)  # z is not Euclidean ray distance


def test_open_thin_wall_blocks_safe_endpoints_and_offcentre_crossing():
    # Non-watertight, zero-thickness wall; midpoint need not lie on the wall.
    vertices = np.array([[.27, -1, -1], [.27, 1, -1], [.27, 1, 1], [.27, -1, 1]])
    wall = scene(vertices, [[0, 1, 2], [0, 2, 3]])
    assert np.all(wall.distance([[-1, 0, 0], [1, 0, 0]]) > .05)
    assert certificate(wall, [-1, 0, 0], [1, 0, 0]) is None
    assert certificate(wall, [-1, 2, 0], [1, 2, 0]) is not None


def test_unknown_interval_is_not_skipped_by_endpoint_safety():
    class ClearMesh:
        def distance(self, points):
            return np.full(len(points), 100.)
    free = ObservedFreeSpace()
    free.origins = [[-1, 0, 0], [1, 0, 0]]
    free.origin_radii = [.2, .2]
    assert free.contains_ball([-1, 0, 0], .05)
    assert free.contains_ball([1, 0, 0], .05)
    assert certificate(ClearMesh(), [-1, 0, 0], [1, 0, 0], free) is None


def test_free_space_uses_only_corroborated_depth_and_not_behind_hits():
    geometry = scene(*plane_z(5))
    config = load_config()["free_space"]
    free = ObservedFreeSpace()
    free.add_view(geometry, camera(), np.full((48,48), 5.), np.ones((48,48)), 1., config, .05)
    assert free.contains_ball([0, 0, 2], .02)
    assert free.contains_ball([0, 0, 0], .05)  # separately validated real-camera anchor
    assert not free.contains_ball([0, 0, 5.1], .01)
    assert not free.contains_ball([10, 0, 2], .01)
    for depth, alpha in ((np.zeros((48,48)), np.ones((48,48))),
                         (np.full((48,48), 5.), np.zeros((48,48))),
                         (np.full((48,48), 8.), np.ones((48,48)))):
        unknown = ObservedFreeSpace()
        unknown.add_view(geometry, camera(), depth, alpha, 1., config, .05)
        assert not unknown.contains_ball([0,0,2], .01)
        assert not unknown.origins


def test_observed_depth_scale_and_no_hit_do_not_carve_empty_space():
    config = load_config()["free_space"]
    template = camera()
    template["R"] = (np.eye(3)*2).tolist()
    geometry = scene(*plane_z(5))
    free = ObservedFreeSpace()
    free.add_view(geometry, template, np.full((48,48), 10.), np.ones((48,48)), 1., config, .05)
    assert free.contains_ball([0,0,2], .02)
    class Miss:
        def cast(self, origin, directions):
            return np.full(directions.shape[:-1], np.inf)
        def distance(self, points):
            return np.full(len(points), 100.)
    unknown = ObservedFreeSpace()
    unknown.add_view(Miss(), camera(), np.full((48,48), 5.), np.ones((48,48)), 1., config, .05)
    assert not unknown.origins and not len(unknown.planes)


def test_near_plane_and_foreground_occlusion_are_hard_rejections():
    config = load_config()
    target_v = np.array([[-.5, -.5, 5], [.5, -.5, 5], [.5, .5, 5], [-.5, .5, 5]])
    target_t = np.array([[0,1,2], [0,2,3]])
    pytest.importorskip("open3d")
    roi = sample_target(target_v, target_t, target_v, 256, 0)
    foreground_v, foreground_t = plane_z(3)
    geometry = scene(np.vstack((target_v, foreground_v)), np.vstack((target_t, foreground_t+4)))
    metric, reason = evaluate_candidate(geometry, Everywhere(), roi, np.zeros(3), camera(), config, .05, 1e-5)
    assert metric is None and reason == "target_occluded"
    template = camera(); template["znear"] = .1
    geometry = scene(*plane_z(.04))
    metric, reason = evaluate_candidate(geometry, Everywhere(), roi, np.zeros(3), template, config, near_radius(template), 1e-5)
    assert metric is None and reason == "camera_collision"


def test_bent_route_keeps_every_turn_and_fails_without_distinct_budget():
    graph = {0:{1:{}}, 1:{0:{},2:{}}, 2:{1:{},3:{}}, 3:{2:{}}}
    for neighbors in graph.values():
        for edge in neighbors.values():
            edge.update(rotation_deg=1., length=1.)
    metrics = [dict(visible=np.array([True, True]), score=1.) for _ in graph]
    config = load_config()
    positions = np.array([[-1,0,0], [-1,2,0], [1,2,0], [1,0,0]])
    route, _, _ = select_sequence(graph, metrics, positions, [0], 4, config)
    assert route in ([0,1,2,3], [3,2,1,0])
    assert len(route) == len(set(route)) == 4
    with pytest.raises(ValueError):
        select_sequence(graph, metrics, positions, [0], 5, config)
    with pytest.raises(ValueError):
        select_sequence(graph, metrics, positions, [0], 4, config, banned={1})


def test_preview_allows_low_alpha_inside_hole_but_requires_context():
    config = load_config()
    shape = (48,48)
    target = np.zeros(shape); target[18:30,18:30] = 1
    removed = np.ones(shape); removed[target > 0] = 0
    arrays = {"target_alpha":target, "target_depth":target*2,
        "full_alpha":np.ones(shape), "full_depth":np.where(target,2.,3.),
        "removed_alpha":removed, "removed_depth":removed*3}
    assert preview_metrics(arrays, config, 1., camera())["passed"]
    arrays["removed_alpha"][:] = 0
    assert not preview_metrics(arrays, config, 1., camera())["passed"]


def schema3():
    records = [camera(i) for i in range(2)]
    records[1]["T"][0] = -.1
    trajectory = dict(type="collision_aware", algorithm_revision=1, frame_count=2,
        target_id=14, config_sha256="a"*64, input_sha256="b"*64, evidence_sha256="c"*64,
        focus_world=[0,0,1], scene_up_world=[0,0,1], safe_radius_world=.05,
        selected_candidate_ids=[10,20], unknown_space="reject", path_model="certified_piecewise_linear")
    return camera_contract().build_collision_camera_manifest(records, iteration=2000, trajectory=trajectory)


def test_schema3_cpu_readers_normal_publisher_and_identity(tmp_path):
    from tools.paintmesh_normal import read_cameras
    from tools.publish_inpainted_edgs_model import _validate_camera_artifact_id
    manifest = schema3()
    path = tmp_path / "virtual_cameras.json"
    write_json(path, manifest)
    value = camera_contract().load_virtual_camera_manifest(path)
    assert value["schema_version"] == 3 and "circle_radius" not in value
    assert _validate_camera_artifact_id(manifest) == manifest["artifact_id"]
    raw, cameras = read_cameras(path, ["00000", "00001"])
    assert raw["artifact_id"] == manifest["artifact_id"] and len(cameras) == 2
    camera_contract().check_camera_request(raw, "collision_aware", 2)
    with pytest.raises(ValueError):
        camera_contract().check_camera_request(raw, "circle", 2)
    changed = copy.deepcopy(manifest); changed["cameras"][1]["T"][0] += .001
    write_json(path, changed)
    with pytest.raises(ValueError, match="artifact_id"):
        camera_contract().load_virtual_camera_manifest(path)
    for key, value in (("unknown_space", "allow"), ("selected_candidate_ids", [10,10])):
        changed = copy.deepcopy(manifest); changed["trajectory"][key] = value
        with pytest.raises(ValueError):
            camera_contract().validate_virtual_camera_manifest(changed)


def test_evidence_rejects_changed_geometry_config_and_camera_binding(tmp_path):
    manifest = schema3()
    geometry = tmp_path / "mesh.ply"; geometry.write_text("mesh-version-1")
    config = load_config()
    inputs = dict(backend="edgs-pgsr", files={"collision_mesh_full_raw":file_record(geometry)})
    safety = dict(complete=True, unknown_space="reject", files={}, positions=[[0,0,0],[.1,0,0]],
        edges=[dict(observed_free=True, swept_sphere=True, certified_leaves=1,
            minimum_mesh_distance_lower_bound=.2, length=.1)], safe_radius_world=.05,
        numeric_margin_world=.0001, mesh_distances=[.3,.3],
        cameras_sha256=identity(manifest["cameras"]), selected_candidate_ids=[10,20])
    root = tmp_path / "view_planning"
    for name, value in (("input_manifest", inputs), ("resolved_config", config), ("safety_report", safety)):
        write_json(root/(name+".json"), value)
    manifest["trajectory"].update(config_sha256=identity(config), input_sha256=identity(inputs), evidence_sha256=identity(safety))
    manifest["artifact_id"] = camera_contract().camera_artifact_id(manifest)
    path = tmp_path / "virtual_cameras.json"; write_json(path, manifest)
    validate_published_plan(path, backend="edgs-pgsr", config=config)
    with pytest.raises(ValueError, match="backend"):
        validate_published_plan(path, backend="inpaint360gs")
    changed = copy.deepcopy(config); changed["seed"] = 1
    with pytest.raises(ValueError, match="configuration"):
        validate_published_plan(path, config=changed)
    geometry.write_text("mesh-version-2")
    with pytest.raises(ValueError, match="input changed"):
        validate_published_plan(path)


@pytest.mark.parametrize("text", ["collision: {unknown_space: allow}", "seed: true", "radius_multipliers: [.nan]",
    "candidate_directions: 100000", "view_quality: {min_target_visible_ratio: 1.1}", "typo: 1"])
def test_invalid_configuration_fails_closed(tmp_path, text):
    path = tmp_path/"config.yaml"; path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_json_source_formatting_is_not_a_geometry_change(tmp_path):
    path = tmp_path/"removal.json"; path.write_text('{"target_id": [14]}')
    records = {"removal_config":file_record(path, canonical_json=True)}
    path.write_text(json.dumps(dict(target_id=[14]), indent=4))
    validate_records(records)
    path.write_text('{"target_id": [15]}')
    with pytest.raises(ValueError):
        validate_records(records)


def test_exact_record_reconstruction_matches_renderer_float32_matrices():
    torch = pytest.importorskip("torch")
    from utils.graphics_utils import getWorld2View2
    template = camera()
    template.update(R=(np.eye(3)*2.3).tolist(), trans=[.2,-.3,.7], scale=1.8)
    record = look_at(np.array([3.,-2.,1.]), [0,0,0], [0,0,1], template, 0)
    base = SimpleNamespace(world_view_transform=torch.eye(4), image_width=48, image_height=48)
    view = camera_contract().camera_views_from_records(base, [record])[0]
    expected = getWorld2View2(np.array(record["R"]), np.array(record["T"]),
        np.array(record["trans"]), record["scale"])
    np.testing.assert_array_equal(view.world_view_transform.numpy(), expected.T)
    np.testing.assert_allclose(view.camera_center.numpy(), camera_center(record), atol=1e-6)
    assert view.image_name == record["image_name"] and view.znear == record["znear"]
