"""Planner-to-consumer integration with real triangle queries.

Only CUDA rendering is replaced by analytic first-hit images. This tests the
actual CLI orchestration/contracts, not the quality of Gaussian rasterization.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[1]/"submodules/Inpaint360GS"))


def test_synthetic_room_plan_publish_resume_and_invalidate(tmp_path, monkeypatch):
    o3d = pytest.importorskip("open3d")
    from plyfile import PlyData, PlyElement
    import plan_virtual_views as planner
    from view_planning_candidates import look_at
    from view_planning_geometry import MeshScene, image_rays
    from view_planning_io import file_record, validate_published_plan
    from virtual_render_io import atomic_write, read_json, write_json, camera_contract
    from tools.prepare_inpaint_workspace import _strict_manifest, CAMERA_KIND
    from tools.paintmesh_normal import read_cameras

    target = o3d.geometry.TriangleMesh.create_box(1,1,1).translate([-.5,-.5,-.5])
    room = o3d.geometry.TriangleMesh.create_box(10,10,10).translate([-5,-5,-5])
    full = target+room
    raw = tmp_path/"edgs/mesh/ours_30000/tsdf_fusion.ply"
    raw.parent.mkdir(parents=True)
    o3d.io.write_triangle_mesh(str(raw), full)
    post = raw.with_name("tsdf_fusion_post.ply")
    o3d.io.write_triangle_mesh(str(post), full)
    full_scene = MeshScene(full.vertices, full.triangles)
    target_scene = MeshScene(target.vertices, target.triangles)
    room_scene = MeshScene(room.vertices, room.triangles)
    model = tmp_path/"run/work_model"
    source = tmp_path/"edgs/point_cloud/iteration_30000/point_cloud.ply"
    source.parent.mkdir(parents=True); source.write_text("source-model")
    (tmp_path/"edgs/config.yaml").write_text("gs: {renderer: {backend: pgsr}}")
    for name in ("point_cloud/iteration_2000/point_cloud.ply", "point_cloud/iteration_2000/classifier.pth",
                 "point_cloud_object_removal/iteration_2000/point_cloud.ply", "cfg_args"):
        path = model/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    xyz = np.asarray(target.vertices)
    values = np.empty(len(xyz), dtype=[("x","f4"),("y","f4"),("z","f4")])
    for i,k in enumerate(("x","y","z")):
        values[k] = xyz[:,i]
    target_ply = model/"point_cloud_object_removal/iteration_2000/point_cloud_14.ply"
    PlyData([PlyElement.describe(values, "vertex")], text=True).write(str(target_ply))
    semantic = tmp_path/"semantic/semantic_manifest.json"
    semantic.parent.mkdir()
    labels = np.zeros(len(full.triangles), dtype=np.uint16)
    labels[:len(target.triangles)] = 14
    np.save(semantic.parent/"face_instance_id.npy", labels)
    write_json(semantic, dict(complete=True, inputs=dict(mesh=file_record(post),
        gaussian_ply=file_record(model/"point_cloud/iteration_2000/point_cloud.ply")),
        outputs=dict(face_label=dict(file="face_instance_id.npy"))))
    bridge = tmp_path/"bridge.json"
    write_json(bridge, dict(complete=True, edgs=dict(model_path=str(tmp_path/"edgs"), iteration=30000,
        point_cloud_path=str(source), point_cloud_sha256=file_record(source)["sha256"])))
    removal = tmp_path/"run/config/object_removal/test/room.json"
    inpaint = tmp_path/"run/config/object_inpaint/test/room.json"
    for path in (removal, inpaint):
        write_json(path, dict(target_id=[14], surrounding_ids=[], select_obj_id=[14]))
    template = dict(R=np.eye(3).tolist(), T=[0,0,0], trans=[0,0,0], scale=1.,
        FoVx=1., FoVy=1., image_width=64, image_height=64, znear=.01, zfar=100.)
    training = [look_at(np.array([3*np.cos(a),3*np.sin(a),1.]), np.zeros(3), [0,0,1], template, i)
        for i,a in enumerate(np.linspace(0,2*np.pi,16,endpoint=False))]
    context = tmp_path/"run/tracker/view_planning/training_context.json"
    write_json(context, dict(kind="paintmesh-planning-context", schema_version=1, iteration=2000,
        cameras=training, model_path=str(model), source_path=str(tmp_path/"dataset")))
    config = tmp_path/"planner.yaml"
    config.write_text("candidate_directions: 32\nsurface_probes: 256\n"
        "free_space: {tile_pixels: 8, max_training_views: 16, max_depth_spread_over_target_radius: 1.0}\n"
        "selection: {minimum_common_target_samples: 8, max_neighbor_translation_over_target_radius: 0.4}\n")
    calls = []
    def analytic_preview(args, records, directory, *, target=False):
        calls.append(target)
        files = {}
        variants = [("full", full_scene)] + ([("target", target_scene),("removed", room_scene)] if target else [])
        for camera in records:
            origin, directions, lengths = image_rays(camera)
            arrays = {}
            for label, geometry in variants:
                depth = geometry.cast(origin, directions)/lengths
                arrays[label+"_alpha"] = np.isfinite(depth).astype(np.float32)
                arrays[label+"_depth"] = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
            path = directory/(camera["image_name"]+".npz")
            atomic_write(path, lambda stream, arrays=arrays: np.savez_compressed(stream, **arrays))
            files[camera["image_name"]] = file_record(path)
        value = dict(complete=True, backend=args.backend, cameras=records, files=files)
        write_json(directory/"preview.json", value)
        return value
    monkeypatch.setattr(planner, "preview", analytic_preview)
    args = SimpleNamespace(config_path=config, count=5, output=tmp_path/"run/tracker/virtual_cameras.json",
        validate_only=False, context=context, model_path=model, edgs_model_path=tmp_path/"edgs",
        bridge_manifest=bridge, semantic_manifest=semantic, removal_config=removal,
        target_id=14, iteration=2000, backend="edgs-pgsr")
    planner.run(args)
    assert calls[0] is False and calls[1] is True
    manifest = read_json(args.output)
    assert manifest["frame_count"] == 5 and "circle_radius" not in manifest
    assert camera_contract().load_virtual_camera_manifest(args.output)["schema_version"] == 3
    _strict_manifest(args.output, "virtual cameras", CAMERA_KIND)
    read_cameras(args.output, [f"{i:05d}" for i in range(5)])
    validate_published_plan(args.output, backend="edgs-pgsr")
    before = args.output.read_bytes()
    planner.run(args)
    assert args.output.read_bytes() == before and len(calls) == 2
    # Geometry mutation invalidates reuse without overwriting approved cameras.
    with raw.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="input changed"):
        planner.run(args)
    assert args.output.read_bytes() == before
