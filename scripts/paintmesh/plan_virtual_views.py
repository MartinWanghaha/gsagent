#!/usr/bin/env python3
"""Plan object-centred collision-aware cameras, then freeze a schema-3 route."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from virtual_render_io import camera_contract, identity, read_json, write_json, atomic_write
from view_planning_io import file_record, load_config, validate_published_plan, validate_records, resize_planning_cameras
from view_planning_geometry import MeshScene, ObservedFreeSpace, near_radius, camera_center
from view_planning_candidates import generate_candidates, sample_target, evaluate_candidate, preview_metrics
from view_planning_selection import build_graph, select_sequence

REPO = Path(__file__).resolve().parents[2]


def preview(args, records, directory, *, target=False):
    request = directory / "request.json"
    write_json(request, dict(cameras=records))
    project = REPO / "submodules" / ("EDGS" if args.backend == "edgs-pgsr" else "Inpaint360GS")
    command = [args.edgs_python if args.backend == "edgs-pgsr" else args.inpaint_python,
        str(Path(__file__).with_name("render_virtual_worker.py")), "--backend", args.backend,
        "--model-path", str(args.model_path.resolve()), "--edgs-model-path", str(args.edgs_model_path.resolve()),
        "--iteration", str(args.iteration), "--planning-cameras", str(request.resolve()),
        "--planning-output", str(directory.resolve()), "--planning-max-size", str(args.config["free_space"]["max_size"])]
    if target:
        command += ["--planning-target-ply", str(args.target_ply.resolve())]
    subprocess.run(command, cwd=project, env=dict(os.environ, PYTHONPATH=str(project)), check=True)
    result = read_json(directory / "preview.json")
    if result.get("complete") is not True or result.get("backend") != args.backend:
        raise ValueError("incomplete Gaussian preview")
    if identity(result["cameras"]) != identity(resize_planning_cameras(records, args.config["free_space"]["max_size"])):
        raise ValueError("Gaussian preview camera matrices/intrinsics differ from requested cameras")
    validate_records(result["files"])
    validate_records(result["models"])
    if result["request_sha256"] != file_record(request)["sha256"]:
        raise ValueError("Gaussian preview/request mismatch")
    return result


def load_inputs(args):
    import open3d as o3d
    from plyfile import PlyData
    context = read_json(args.context)
    if (context.get("kind") != "paintmesh-planning-context" or context.get("iteration") != args.iteration
            or Path(context["model_path"]).resolve() != args.model_path.resolve()):
        raise ValueError("planning context/model/iteration mismatch")
    training = context["cameras"]
    if not training:
        raise ValueError("no real training cameras")
    # Reuse the CPU record validator, while keeping internal real camera names.
    camera_contract().validate_virtual_camera_manifest(dict(schema_version=1,
        kind=camera_contract().MANIFEST_KIND, frame_count=len(training), iteration=args.iteration,
        circle_radius=1., cameras=training))
    bridge = read_json(args.bridge_manifest)
    semantic = read_json(args.semantic_manifest)
    if bridge.get("complete") is not True or semantic.get("complete") is not True:
        raise ValueError("bridge/semantic inputs are incomplete")
    source = bridge["edgs"]
    if args.edgs_model_path.resolve() != Path(source["model_path"]).resolve():
        raise ValueError("collision geometry and Gaussian source model differ")
    raw = args.edgs_model_path / "mesh" / f"ours_{source['iteration']}" / "tsdf_fusion.ply"
    if not raw.is_file():
        raise ValueError(f"full raw collision mesh missing: {raw}; rerun full-scene mesh extraction, never use removed/post mesh")
    semantic_mesh = Path(semantic["inputs"]["mesh"]["path"])
    # This sidecar is bound to THIS post mesh, never the raw mesh's indices.
    record = semantic["inputs"]["mesh"]
    for key, actual in (("size_bytes", semantic_mesh.stat().st_size), ("mtime_ns", semantic_mesh.stat().st_mtime_ns),
                        ("sha256", file_record(semantic_mesh)["sha256"])):
        if key in record and record[key] != actual:
            raise ValueError(f"semantic source mesh changed: {key}")
    labels_path = args.semantic_manifest.parent / semantic["outputs"]["face_label"]["file"]
    labels = np.load(labels_path, allow_pickle=False)
    mesh = o3d.io.read_triangle_mesh(str(semantic_mesh))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if labels.shape != (len(triangles),) or labels.dtype.kind not in "ui":
        raise ValueError("semantic face labels do not match their source mesh")
    target_triangles = triangles[labels == args.target_id]
    if not len(target_triangles):
        raise ValueError("target has no supported semantic mesh faces")
    args.target_ply = args.model_path / "point_cloud_object_removal" / f"iteration_{args.iteration}" / f"point_cloud_{args.target_id}.ply"
    ply = PlyData.read(str(args.target_ply))["vertex"].data
    gaussian_xyz = np.column_stack([ply[k] for k in ("x", "y", "z")])
    if len(gaussian_xyz) < 4 or not np.isfinite(gaussian_xyz).all():
        raise ValueError("insufficient actual removed target Gaussians")
    roi = sample_target(vertices, target_triangles, gaussian_xyz, args.config["surface_probes"], args.config["seed"])
    paths = {"context": args.context, "bridge": args.bridge_manifest, "semantic": args.semantic_manifest,
        "collision_mesh_full_raw": raw, "semantic_mesh": semantic_mesh, "face_labels": labels_path,
        "target_gaussians": args.target_ply, "edgs_config": args.edgs_model_path / "config.yaml",
        "edgs_source_gaussians": Path(source["point_cloud_path"]),
        "full_gaussians": args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply",
        "classifier": args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "classifier.pth",
        "removed_all_selected": args.target_ply.with_name("point_cloud.ply"),
        "model_config": args.model_path / "cfg_args", "removal_config": args.removal_config}
    inputs = dict(kind="paintmesh-view-planning-input", backend=args.backend, target_id=args.target_id,
        source_cameras_sha256=identity(training), files={key: file_record(path) for key, path in paths.items()})
    inputs["files"]["removal_config"] = file_record(args.removal_config, canonical_json=True)
    if Path(semantic["inputs"]["gaussian_ply"]["path"]).resolve() != Path(inputs["files"]["full_gaussians"]["path"]):
        raise ValueError("semantic mesh and full Gaussian assets have different sources")
    for name in ("scene_info", "classifier", "gaussian_ply"):
        record = semantic["inputs"].get(name)
        if record:
            inputs["files"]["semantic_" + name] = file_record(record["path"])
    sparse = Path(context["source_path"]) / "sparse" / "0"
    for path in sorted(sparse.glob("*")):
        if path.is_file() and path.stem in ("cameras", "images", "points3D"):
            inputs["files"]["colmap_" + path.name] = file_record(path)
    if inputs["files"]["edgs_source_gaussians"]["sha256"] != source["point_cloud_sha256"]:
        raise ValueError("source EDGS geometry changed since bridge creation")
    return training, roi, MeshScene.from_file(raw), inputs


def save_trajectory(path, positions, focus):
    # Separate non-spherical diagnostics; no fabricated PCA/circle metadata.
    from virtual_render_io import atomic_write
    points = np.asarray(positions)-focus
    scale = max(float(np.abs(points).max()), 1e-12)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="520">', '<rect width="1000" height="520" fill="white"/>']
    for panel, (a, b, label) in enumerate(((0, 1, "World XY"), (0, 2, "World XZ"))):
        xy = points[:, [a, b]]/scale*210 * [1, -1] + [250+500*panel, 270]
        parts += [f'<text x="{panel*500+25}" y="30" font-size="18">{label}: certified route</text>',
            '<polyline fill="none" stroke="#16837b" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in xy)+'"/>']
        for i, (x, y) in enumerate(xy):
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#1d4c70"/><text x="{x+4:.2f}" y="{y-4:.2f}" font-size="9">{i:05d}</text>')
    atomic_write(path, lambda stream: stream.write(('\n'.join(parts)+"\n</svg>\n").encode()))


def save_contact_sheet(rendered, path):
    """Up to 12 evenly spaced frames; full / isolated target / removed RGB."""
    from PIL import Image, ImageDraw, ImageOps
    records = rendered["cameras"]
    indices = np.unique(np.linspace(0, len(records)-1, min(len(records), 12)).astype(int))
    cards = []
    for index in indices:
        name = records[index]["image_name"]
        with np.load(rendered["files"][name]["path"], allow_pickle=False) as arrays:
            if not all(label+"_rgb" in arrays for label in ("full", "target", "removed")):
                return False  # analytic geometry fixtures have no RGB
            card = Image.new("RGB", (318, 136), "white")
            draw = ImageDraw.Draw(card)
            draw.text((6, 2), f"{name}   full / target / removed", fill="#24394b")
            for col, label in enumerate(("full", "target", "removed")):
                rgb = (np.clip(arrays[label+"_rgb"], 0, 1)*255).astype(np.uint8)
                tile = ImageOps.contain(Image.fromarray(rgb), (102, 102))
                card.paste(tile, (6+col*104+(102-tile.width)//2, 26+(102-tile.height)//2))
            cards.append(card)
    sheet = Image.new("RGB", (318*3, 136*((len(cards)+2)//3)), "#eeeeee")
    for i, card in enumerate(cards):
        sheet.paste(card, ((i%3)*318, (i//3)*136))
    atomic_write(path, lambda stream: sheet.save(stream, format="PNG"))
    return True


def run(args):
    args.config = load_config(args.config_path)
    if args.count < 2:
        raise ValueError("camera count must be >= 2")
    root = args.output.parent / "view_planning"
    if args.output.exists():
        camera_contract().load_virtual_camera_manifest(args.output, expected_iteration=args.iteration)
        camera_contract().check_camera_request(read_json(args.output), "collision_aware", args.count)
        validate_published_plan(args.output, backend=args.backend, config=args.config)
        # Context re-export must reproduce source camera matrices, not merely
        # keep a previously valid context file cached elsewhere.
        recorded = read_json(root / "input_manifest.json")
        if recorded["target_id"] != args.target_id:
            raise ValueError("planning target changed; use a new RUN_NAME")
        for label, path in (("bridge", args.bridge_manifest), ("semantic", args.semantic_manifest),
                ("edgs_config", args.edgs_model_path / "config.yaml"), ("removal_config", args.removal_config),
                ("full_gaussians", args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply")):
            if Path(recorded["files"][label]["path"]).resolve() != path.resolve():
                raise ValueError(f"planning {label} source changed; use a new RUN_NAME")
        if recorded["files"]["context"] != file_record(args.context):
            raise ValueError("training cameras changed; use a new RUN_NAME")
        print(f"Reusing validated collision-aware cameras: {args.output}")
        return
    if args.validate_only:
        raise ValueError("collision-aware cameras are missing; run removal Stage 4")
    removal = read_json(args.removal_config)
    if removal.get("target_id") != [args.target_id]:
        raise ValueError("collision_aware v1 requires exactly one target ID (surrounding IDs remain obstacles)")
    for path in (args.removal_config, Path(str(args.removal_config).replace("object_removal", "object_inpaint"))):
        value = read_json(path)
        value.update(virtual_camera_path="collision_aware", virtual_camera_count=args.count)
        write_json(path, value)
    training, roi, scene, inputs = load_inputs(args)
    print(f"Planning target {args.target_id}: radius={roi['radius']:.6g}, full raw mesh loaded", flush=True)
    write_json(root / "input_manifest.json", inputs)
    write_json(root / "resolved_config.json", args.config)
    write_json(root / "target_roi.json", dict(focus_world=roi["focus"].tolist(), radius_world=roi["radius"],
        bounds_world=roi["bounds"].tolist(), actual_removed_gaussian_bounds=roi["gaussian_bounds"],
        known_surface_area=roi["area"]))
    indices = np.unique(np.linspace(0, len(training)-1, min(len(training), args.config["free_space"]["max_training_views"])).astype(int))
    observed = preview(args, [training[i] for i in indices], root / "observations")
    safe_radius = max(near_radius(training[0]), roi["radius"]*args.config["collision"]["body_radius_over_target_radius"])
    safe_radius += roi["radius"]*args.config["collision"]["clearance_over_target_radius"]
    camera_precision = max(float(np.abs(camera_center(c)).max()) for c in training) * np.finfo(np.float32).eps * 32
    margin = max(scene.numeric_floor, camera_precision,
                 roi["radius"]*args.config["collision"]["numeric_margin_over_target_radius"])
    free = ObservedFreeSpace()
    for camera in observed["cameras"]:
        with np.load(observed["files"][camera["image_name"]]["path"], allow_pickle=False) as arrays:
            free.add_view(scene, camera, arrays["full_depth"], arrays["full_alpha"], roi["radius"],
                          args.config["free_space"], safe_radius+margin)
    write_json(root / "observed_space.json", dict(policy="corroborated_depth_tile_frusta_and_validated_real_origins", views=free.stats))
    if not free.origins:
        raise ValueError("no safe real-camera anchors with corroborated depth; inspect observed_space.json")
    print(f"Observed space: {len(free.origins)} anchors, {len(free.planes)} trusted depth tiles", flush=True)
    positions, cameras, up = generate_candidates(roi, training, args.config)
    valid_positions, valid_cameras, metrics, ids = [], [], [], []
    rejected = Counter()
    for i, (position, camera) in enumerate(zip(positions, cameras)):
        metric, reason = evaluate_candidate(scene, free, roi, position, camera, args.config, safe_radius, margin)
        if reason:
            rejected[reason] += 1
        else:
            valid_positions.append(position); valid_cameras.append(camera); metrics.append(metric); ids.append(i)
    write_json(root / "candidate_metrics.json", dict(total=len(cameras), valid=len(metrics), rejected=dict(rejected)))
    if len(metrics) < args.count:
        raise ValueError(f"only {len(metrics)} valid candidates for {args.count} frames; rejection details in candidate_metrics.json")
    print(f"Candidates: {len(cameras)} evaluated, {len(metrics)} valid; certifying graph edges", flush=True)
    positions = np.asarray(valid_positions)
    graph = build_graph(scene, free, positions, valid_cameras, metrics, roi, args.config, safe_radius, margin)
    atomic_write(root / "candidates.npz", lambda stream: np.savez_compressed(stream,
        positions=positions, candidate_ids=np.asarray(ids), target_probes=roi["probes"],
        visible=np.stack([metric["visible"] for metric in metrics])))
    write_json(root / "path_edges.json", [{"a": ids[a], "b": ids[b], **edge}
        for a, neighbors in graph.items() for b, edge in neighbors.items() if a < b])
    anchors = [i for i,p in enumerate(positions) if np.any(np.linalg.norm(np.asarray(free.origins)-p, axis=1) <= 1e-6*roi["radius"])]
    if not anchors:
        raise ValueError("no target-visible candidate at a validated real-camera anchor")
    banned = set()
    files = {"observations": file_record(root / "observations" / "preview.json"),
        **{f"observed_{name}": record for name,record in observed["files"].items()}}
    for attempt in range(args.config["max_refinement_rounds"]+1):
        print(f"Selecting and pre-rendering {args.count} frames (attempt {attempt+1})", flush=True)
        route, coverage, score = select_sequence(graph, metrics, positions, anchors, args.count, args.config, banned=banned)
        selected = [camera_contract()._camera_record(SimpleNamespace(**dict(
            valid_cameras[i], image_name=f"{frame:05d}"))) for frame,i in enumerate(route)]
        directory = root / f"verification_{attempt}"
        rendered = preview(args, selected, directory, target=True)
        checks = []
        for camera, node in zip(rendered["cameras"], route):
            with np.load(rendered["files"][camera["image_name"]]["path"], allow_pickle=False) as arrays:
                check = preview_metrics(arrays, args.config, roi["radius"], camera)
            checks.append(dict(candidate_id=ids[node], **check))
            if not check["passed"]:
                banned.add(node)
        write_json(directory / "quality.json", checks)
        if all(check["passed"] for check in checks):
            files.update({"verification": file_record(directory / "preview.json"),
                "preview_quality": file_record(directory / "quality.json"),
                **{f"preview_{name}": record for name, record in rendered["files"].items()}})
            break
    else:
        raise ValueError("Gaussian/mesh visibility or context disagreement after bounded replanning")
    # Recheck every published position and edge. No trajectory smoothing or
    # resampling is permitted between this check and the manifest writer.
    edges = [graph[a][b] for a,b in zip(route, route[1:])]
    distances = scene.distance(positions[route])
    if (distances < safe_radius+margin).any() or not all(free.contains_ball(p, safe_radius+margin) for p in positions[route]):
        raise ValueError("final camera safety recheck failed")
    safety = dict(kind="paintmesh-view-safety", complete=True, unknown_space="reject", files=files,
        cameras_sha256=identity(selected), safe_radius_world=safe_radius, numeric_margin_world=margin,
        positions=positions[route].tolist(), mesh_distances=distances.tolist(), edges=edges,
        selected_candidate_ids=[ids[i] for i in route])
    if save_contact_sheet(rendered, root / "preview_contact_sheet.png"):
        files["preview_contact_sheet"] = file_record(root / "preview_contact_sheet.png")
    write_json(root / "safety_report.json", safety)
    write_json(root / "coverage_report.json", dict(known_target_surface_coverage=coverage, score=score,
        target_area_world=roi["area"], target_gaussian_bounds=roi["gaussian_bounds"], preview_quality=checks,
        note="Coverage is over reconstructed semantic surface, not unseen geometry ground truth."))
    trajectory = dict(type="collision_aware", algorithm_revision=1, frame_count=args.count,
        target_id=args.target_id, seed=args.config["seed"], config_sha256=identity(args.config),
        input_sha256=identity(inputs), evidence_sha256=identity(safety), focus_world=roi["focus"].tolist(),
        scene_up_world=up.tolist(), safe_radius_world=safe_radius, unknown_space="reject",
        path_model="certified_piecewise_linear", selected_candidate_ids=[ids[i] for i in route])
    manifest = camera_contract().build_collision_camera_manifest(selected, iteration=args.iteration, trajectory=trajectory)
    if identity(manifest["cameras"]) != safety["cameras_sha256"]:
        raise ValueError("normalized cameras differ from the safety certificate")
    # All data must still match immediately before publishing complete=true.
    validate_records(inputs["files"])
    validate_records(files)
    save_trajectory(args.output.parent / "camera_trajectory.svg", positions[route], roi["focus"])
    write_json(args.output.parent / "camera_trajectory.json", dict(trajectory=trajectory,
        positions_world=positions[route].tolist(), edges=edges))
    write_json(args.output, manifest)
    validate_published_plan(args.output, backend=args.backend, config=args.config)
    print(f"collision_aware: {args.count} distinct certified cameras; known target coverage={coverage:.3f} -> {args.output}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("context", "model-path", "edgs-model-path", "bridge-manifest", "semantic-manifest", "removal-config", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--config", dest="config_path", type=Path)
    p.add_argument("--target-id", type=int, required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--backend", choices=("inpaint360gs", "edgs-pgsr"), default="inpaint360gs")
    p.add_argument("--inpaint-python", default=sys.executable)
    p.add_argument("--edgs-python", default=sys.executable)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        write_json(args.output.parent / "view_planning" / "failure.json",
            dict(complete=False, error_type=type(exc).__name__, reason=str(exc)))
        raise


if __name__ == "__main__":
    main()
