#!/usr/bin/env python3
"""Independent PaintMesh Stage 5b. Never invokes EDGS global training."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from types import SimpleNamespace

EDGS = Path(__file__).resolve().parents[1]
REPO = EDGS.parents[1]
sys.path.insert(0, str(EDGS))
sys.path.insert(0, str(REPO / "scripts/paintmesh"))

from local_geometry_io import (  # noqa: E402
    LOCAL_KIND, atomic_write, gate_membership, identity, load_config, read_json,
    record, seal, sha256, validate_local, validate_rgb, verify_targets, write_json,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("rgb-manifest", "lama", "camera", "edgs-config", "config", "output-root", "manifest"):
        result.add_argument("--" + name, type=Path, required=True)
    for name in ("iterations", "geometry-from-iter", "geometry-ramp-iters"):
        result.add_argument("--" + name, type=int)
    result.add_argument("--validate-only", action="store_true")
    return result


def prepare(args):
    cfg = load_config(args.config, iterations=args.iterations,
        geometry_from_iter=args.geometry_from_iter, geometry_ramp_iters=args.geometry_ramp_iters)
    rgb = validate_rgb(args.rgb_manifest)
    lama, inputs = verify_targets(args.lama, args.camera)
    if cfg["debug"]["view_index"] >= len(lama["frames"]):
        raise ValueError("debug.view_index is outside the virtual camera frame set")
    for key, path in (("lama", args.lama), ("camera", args.camera)):
        if record(path) != rgb["inputs"][key]:
            raise ValueError(f"Stage 5a and Stage 5b {key} inputs differ")
    import yaml
    edgs_config = yaml.safe_load(args.edgs_config.read_text())
    if edgs_config["gs"]["renderer"]["backend"] != "pgsr":
        raise ValueError("Stage 5b requires an EDGS-PGSR source configuration")
    records = {name: record(getattr(args, name)) for name in (
        "rgb_manifest", "lama", "camera", "edgs_config", "config")}
    implementation = {str(path.relative_to(REPO)): sha256(path) for path in (
        Path(__file__), EDGS / "source/paintmesh_local_data.py",
        EDGS / "source/paintmesh_local_losses.py", EDGS / "source/renderers/pgsr.py",
        EDGS / "source/paintmesh_local_debug.py", EDGS / "source/pgsr_debug.py",
        EDGS / "source/pgsr_geometry.py", REPO / "scripts/paintmesh/local_geometry_io.py")}
    request = dict(version=1, enabled=True, inputs=records, config=cfg, implementation=implementation,
                   output_root=str(args.output_root.resolve()), manifest=str(args.manifest.resolve()))
    output = args.output_root / "point_cloud" / f"iteration_{cfg['iterations']}" / "point_cloud.ply"
    if output.resolve() == Path(rgb["outputs"]["ply"]["path"]).resolve():
        raise ValueError("local geometry output must not overwrite Stage 5a")
    if args.manifest.exists():
        validate_local(args.manifest, selected_ply=output, lama_path=args.lama,
                       camera_path=args.camera, request=request)
        print(f"Reused validated local geometry: {output}")
        return None
    if args.validate_only:
        raise ValueError("Stage 5b is incomplete; resume run_inpaint from Stage 5")
    request_path = args.output_root / "request.json"
    if request_path.exists() and read_json(request_path) != request:
        raise ValueError("local geometry request changed; choose a new INPAINT_RUN_NAME")
    if not request_path.exists() and output.exists():
        raise ValueError("unbound local geometry output exists; choose a new INPAINT_RUN_NAME")
    write_json(request_path, request)
    atomic_write(args.output_root / "config.resolved.yaml",
                 lambda stream: stream.write(yaml.safe_dump(cfg, sort_keys=True).encode()))
    return cfg, rgb, lama, inputs, edgs_config, request, output


def train(args, prepared):
    import numpy as np
    import torch
    from source.paintmesh_local_data import LocalGaussians, make_views, read_targets
    from source.paintmesh_local_losses import geometry_ramp, local_losses, local_depth_normal, unit_normal
    from source.pgsr_geometry import depth_to_normal
    from source.renderers.pgsr import PGSRRenderer
    from source.paintmesh_local_debug import LocalGeometryDebug

    cfg, rgb, lama, inputs, edgs_config, request, output = prepared
    if not torch.cuda.is_available():
        raise RuntimeError("local PGSR refinement requires CUDA")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    editable = np.load(rgb["outputs"]["editable_mask"]["path"], allow_pickle=False)
    model = LocalGaussians(rgb["outputs"]["ply"]["path"], editable)
    views = make_views(args.camera)
    targets = read_targets(lama, inputs, views)
    renderer = PGSRRenderer()
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                               debug=False, antialiasing=False)
    background = torch.tensor([1., 1., 1.] if edgs_config["gs"]["dataset"]["white_background"]
                              else [0., 0., 0.], device="cuda")
    optimizer = model.optimizer(cfg["optimizer"])
    checkpoint = args.output_root / "checkpoints/latest.pth"
    request_id = identity(request)
    start, order, empty_steps = 0, [], 0

    def render(view, need_depth_normal=False):
        package = renderer.render(view, model, pipeline, background, return_plane=True)
        if need_depth_normal:
            package["depth_normal"] = local_depth_normal(view, package["plane_depth"])
        return package

    def on_device(target):
        return {key: value.cuda() for key, value in target.items()}

    baselines, baseline_coverage = [], []
    # Always compute I0/A0 from Stage 5a, BEFORE restoring a local checkpoint.
    with torch.no_grad():
        for view, target in zip(views, targets):
            package = render(view)
            baseline = dict(rgb=package["render"].cpu(), alpha=package["rendered_alpha"].squeeze(0).cpu())
            baselines.append(baseline)
            _, stats = local_losses(package, on_device(target), on_device(baseline), 0, cfg)
            baseline_coverage.append(float(stats["coverage"]))

    if checkpoint.exists():
        # Run-owned checkpoint, not a user-supplied arbitrary pickle.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["request_id"] != request_id:
            raise ValueError("checkpoint belongs to different geometry inputs/settings")
        model.restore_local(state["local"])
        optimizer.load_state_dict(state["optimizer"])
        start, order, empty_steps = state["next_step"], state["view_order"], state["empty_steps"]
        if not 0 <= start <= cfg["iterations"] or any(not 0 <= i < len(views) for i in order):
            raise ValueError("invalid checkpoint step/view order")
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming local geometry at step {start}", flush=True)

    debug = LocalGeometryDebug(args.output_root, cfg["debug"],
                               targets[cfg["debug"]["view_index"]], cfg["validity"]["render_alpha_min"])

    @torch.no_grad()
    def capture_debug(completed_steps, final=False):
        if not cfg["debug"]["enabled"]:
            return
        index = cfg["debug"]["view_index"]
        target = on_device(targets[index])
        package = render(views[index], True)
        # The model is post-update; weights correspond to the last update.
        _, metrics = local_losses(package, target, on_device(baselines[index]),
                                   max(0, completed_steps - 1), cfg)
        debug.save(package, target, metrics, completed_steps, final=final)

    if debug.due(start):
        capture_debug(start)

    def save_checkpoint(next_step):
        state = dict(request_id=request_id, local=model.local_state(),
                     optimizer=optimizer.state_dict(), next_step=next_step,
                     view_order=order, empty_steps=empty_steps, python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all())
        atomic_write(checkpoint, lambda stream: torch.save(state, stream))

    for step in range(start, cfg["iterations"]):
        if not order:
            order = list(range(len(views)))
            random.shuffle(order)
        index = order.pop()
        need_consistency = geometry_ramp(step, cfg) > 0 and cfg["loss"]["depth_normal_consistency"] > 0
        package = render(views[index], need_consistency)
        loss, stats = local_losses(package, on_device(targets[index]),
                                   on_device(baselines[index]), step, cfg)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite local loss at step {step}")
        if geometry_ramp(step, cfg) > 0:
            supervised = sum(stats[key] for name, key in (
                ("depth", "depth_pixels"), ("lama_normal", "normal_pixels"),
                ("depth_normal_consistency", "consistency_pixels")) if cfg["loss"][name] > 0)
            empty_steps = empty_steps + 1 if int(supervised) == 0 else 0
            if empty_steps >= cfg["validity"]["max_empty_steps"]:
                raise ValueError("persistent empty local geometry supervision; no result committed")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError(f"nonfinite local gradient at step {step}")
        optimizer.step()
        if debug.due(step + 1):
            capture_debug(step + 1)
        if (step + 1) % cfg["diagnostic_interval"] == 0 or step == start:
            metrics = {key: float(value) for key, value in stats.items()}
            metrics.update(local_step=step, frame=views[index].image_name)
            write_json(args.output_root / "diagnostics" / f"step_{step:06d}.json", metrics)
            print(f"Local PGSR {step + 1}/{cfg['iterations']} loss={float(loss):.6f} "
                  f"ramp={geometry_ramp(step, cfg):.3f} coverage={float(stats['coverage']):.3f}", flush=True)
        if (step + 1) % cfg["checkpoint_interval"] == 0 or step + 1 == cfg["iterations"]:
            save_checkpoint(step + 1)

    # Reject escaping updates, preserving the original Stage 5a gate contract.
    with torch.no_grad():
        allowed = gate_membership(model.xyz.detach().cpu().numpy(),
                                  rgb["outputs"]["gate"]["path"], rgb["inputs"]["support"]["path"])
        rejected = torch.as_tensor(~allowed, device="cuda")
        model.restore_rows(rejected)
        capture_debug(cfg["iterations"], final=True)
        final_stats = []
        for index, (view, target) in enumerate(zip(views, targets)):
            package = render(view, True)
            device_target = on_device(target)
            _, stats = local_losses(package, device_target, on_device(baselines[index]), cfg["iterations"] - 1, cfg)
            values = {key: float(value) for key, value in stats.items()}
            if (not all(np.isfinite(list(values.values()))) or values["normal_pixels"] <= 0 or
                    values["coverage"] < baseline_coverage[index] * cfg["validity"]["min_coverage_ratio"]):
                raise ValueError(f"invalid/collapsed geometry coverage in {view.image_name}; not publishing")
            normal, _ = unit_normal(package["rendered_normal"])
            target_derived = depth_to_normal(view, device_target["depth"])
            angular_valid = device_target["mask"] & device_target["normal_valid"] & (target_derived.norm(dim=0) > 1e-6)
            cosine = (target_derived * device_target["normal"]).sum(0).clamp(-1, 1)
            angles = torch.rad2deg(torch.acos(cosine[angular_valid]))
            values["lama_vs_completed_depth_angle_deg"] = float(angles.mean()) if angles.numel() else None
            values["baseline_coverage"] = baseline_coverage[index]
            final_stats.append(values)
            arrays = dict(rgb=package["render"].permute(1, 2, 0).cpu().numpy(),
                          depth=package["plane_depth"].squeeze(0).cpu().numpy(),
                          normal=normal.permute(1, 2, 0).cpu().numpy(),
                          alpha=package["rendered_alpha"].squeeze(0).cpu().numpy())
            atomic_write(args.output_root / "diagnostics" / f"{view.image_name}.npz",
                         lambda stream, arrays=arrays: np.savez_compressed(stream, **arrays))
        model.save(output)
        if "density" in rgb["inputs"]:
            from support_density_io import audit_density_xyz
            audit_density_xyz(rgb["inputs"]["density"]["path"], model.get_xyz.detach().cpu().numpy(),
                              args.output_root / "diagnostics/density_after_geometry.json")
    # Refuse to commit if an upstream artifact changed during optimization.
    from local_geometry_io import verify_record
    for value in request["inputs"].values():
        verify_record(value)
    validate_rgb(args.rgb_manifest)
    verify_targets(args.lama, args.camera)
    metrics_path = args.output_root / "diagnostics/final.json"
    write_json(metrics_path, dict(frames=final_stats, gate_reverted_rows=int(rejected.sum())))
    receipt = seal(LOCAL_KIND, request=request, inputs=request["inputs"],
        outputs={"ply": record(output), "diagnostics": record(metrics_path),
                 "resolved_config": record(args.output_root / "config.resolved.yaml")},
        parameters=dict(enabled=True, rgb_iterations=rgb["parameters"]["iterations"],
                        local_geometry_iterations=cfg["iterations"], config=cfg),
        rgb_artifact_id=rgb["artifact_id"], point_count=rgb["point_count"],
        gate_reverted_rows=int(rejected.sum()))
    write_json(args.manifest, receipt)
    print(f"Local geometry complete: {output}", flush=True)


def main():
    args = parser().parse_args()
    try:
        prepared = prepare(args)
        if prepared is not None:
            train(args, prepared)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
