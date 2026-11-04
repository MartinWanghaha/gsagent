#!/usr/bin/env python3
"""Stage 5: RGB/geometry joint training; background freezing is configurable."""
import argparse
from pathlib import Path
import random
import sys
from types import SimpleNamespace

EDGS = Path(__file__).resolve().parents[1]
REPO = EDGS.parents[1]
sys.path.insert(0, str(EDGS))
sys.path.insert(0, str(REPO / "scripts/paintmesh"))

from edgs_inpaint_io import (JOINT_KIND, atomic_write, claim, config_arguments, identity,
    implementation, read_json, record, resolve_config, seal, validate_init, validate_joint,
    verify_record, verify_targets, write_json, validate_selection, appearance_weights,
    check_initialization_config)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("initialization", "lama", "camera", "edgs-config", "output-root", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--local-selection", type=Path)
    p.add_argument("--save-iterations", type=int, nargs="+", default=[],
                   help="Export intermediate PLY snapshots of this same training trajectory")
    config_arguments(p)
    return p


def validate_snapshots(root, request, before_step=None):
    from local_geometry_io import read_receipt
    for step in request.get("save_iterations", []):
        if before_step is not None and step >= before_step:
            continue
        saved = read_receipt(root / "snapshots" / f"iteration_{step}/manifest.json",
                             "paintmesh-joint-snapshot")
        if (saved["request"] != request or saved["request_id"] != identity(request)
                or saved["iterations"] != step or saved["training_budget"] != request["config"]["iterations"]):
            raise ValueError("intermediate snapshot provenance differs from this training trajectory")


def prepare(args):
    import yaml
    cfg = resolve_config(args)
    initial = validate_init(args.initialization)
    lama, inputs, cameras = verify_targets(args.lama, args.camera, **cfg["supervision"])
    if cfg["debug"]["view_index"] >= cameras["frame_count"]:
        raise ValueError("debug.view_index outside camera frame set")
    for key in ("lama", "camera"):
        if initial["inputs"][key] != record(getattr(args, key)):
            raise ValueError(f"initialization/training {key} differs")
    check_initialization_config(initial, cfg)
    edgs = yaml.safe_load(args.edgs_config.read_text())
    if edgs["gs"]["renderer"]["backend"] != "pgsr":
        raise ValueError("joint training requires EDGS-PGSR model configuration")
    removed = read_json(initial["inputs"]["removed_model"]["path"])
    if record(args.edgs_config)["sha256"] != removed["inputs"]["edgs_config"]["sha256"]:
        raise ValueError("joint training EDGS config differs from removal")
    records = {k: record(getattr(args, k)) for k in ("initialization", "lama", "camera", "edgs_config")}
    if cfg["local_background"]["enabled"]:
        if args.local_selection is None:
            raise ValueError("local background requires --local-selection")
        selection = validate_selection(args.local_selection, initial=initial, config=cfg["local_background"])
        if selection["inputs"]["initialization"] != records["initialization"]:
            raise ValueError("local selection refers to a different initialization")
        records["local_selection"] = record(args.local_selection)
    elif args.local_selection is not None:
        raise ValueError("local selection supplied while local background disabled")
    request = dict(pipeline="edgs-pgsr", inputs=records, config=cfg, implementation=implementation(),
                   appearance_weights=appearance_weights(cfg),
                   output_root=str(args.output_root.resolve()), manifest=str(args.manifest.resolve()))
    saves = sorted(set(args.save_iterations))
    if any(step <= 0 or step >= cfg["iterations"] for step in saves):
        raise ValueError("intermediate save iterations must be positive and below the training budget")
    if saves and cfg["background_writeback"]["enabled"]:
        raise ValueError("intermediate snapshots do not support final background writeback")
    if saves:
        request["save_iterations"] = saves
    output = args.output_root / "point_cloud" / f"iteration_{cfg['iterations']}" / "point_cloud.ply"
    if args.manifest.exists():
        validate_joint(args.manifest, selected_ply=output, request=request)
        validate_snapshots(args.output_root, request)
        print(f"Reused PGSR joint output: {output}")
        return None
    claim(args.output_root, request, args.validate_only)
    if args.validate_only:
        raise ValueError("PGSR joint training incomplete; resume Stage 5")
    write_json(args.output_root / "config.resolved.json", cfg)
    return cfg, initial, lama, inputs, edgs, request, output


def train(args, prepared):
    import numpy as np
    import torch
    from source.paintmesh_joint_data import JointGaussians, make_views, read_targets
    from source.paintmesh_edgs_init import camera, select_pairs
    from source.paintmesh_joint_losses import joint_losses, multiview_loss, geometry_ramp, build_perceptual
    from source.paintmesh_local_losses import local_depth_normal
    from source.paintmesh_joint_debug import save_debug
    from source.renderers.pgsr import PGSRRenderer
    cfg, initial, lama, inputs, edgs, request, output = prepared
    if not torch.cuda.is_available():
        raise RuntimeError("PGSR joint training requires CUDA")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    views = make_views(args.camera)
    targets = read_targets(lama, inputs, views, **cfg["supervision"])
    editable = np.load(initial["outputs"]["editable_mask"]["path"], allow_pickle=False)
    trainable = None
    if cfg["local_background"]["enabled"]:
        selection = validate_selection(args.local_selection, initial=initial, config=cfg["local_background"])
        trainable = np.load(selection["outputs"]["trainable_mask"]["path"], allow_pickle=False)
    density = cfg["adaptive_density"]
    writeback = cfg["background_writeback"]
    if writeback["enabled"]:
        print("Background policy: train ALL old/new rows; retain old identities; "
              "restore old background then write back first-view + seed-neighborhood selection", flush=True)
    extent = None
    model_type = JointGaussians
    if density["enabled"]:
        from source.paintmesh_joint_density import AdaptiveJointGaussians, camera_extent
        if int(editable.sum()) > density["max_new_points"]:
            raise ValueError("adaptive density budget is smaller than the initialized new population")
        model_type, extent = AdaptiveJointGaussians, camera_extent(views)
        print(f"Adaptive density: {density} extent={extent:.6f}; new descendants only", flush=True)
    model = model_type(initial["outputs"]["ply"]["path"], editable,
                           freeze_background=cfg["freeze_background"], trainable_mask=trainable)
    print(f"Ownership: new={int(editable.sum())} trainable_old={int((model.editable & ~editable).sum())} "
          f"frozen_old={int((~model.editable).sum())}", flush=True)
    renderer = PGSRRenderer()
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    background = torch.tensor([1., 1., 1.] if edgs["gs"]["dataset"]["white_background"] else [0., 0., 0.], device="cuda")
    perceptual = build_perceptual(cfg, "cuda")
    def render(view, consistency=False, scene=model):
        package = renderer.render(view, scene, pipeline, background, return_plane=True)
        if consistency:
            package["depth_normal"] = local_depth_normal(view, package["plane_depth"])
        return package
    def device(target):
        return {k: v.cuda() for k, v in target.items()}
    # The preservation target is ALWAYS the initial removed model, including
    # in unfrozen mode. Rebuild from base buffers, never the updated Parameters.
    frozen = SimpleNamespace(get_xyz=model.xyz_base[~torch.as_tensor(editable, device="cuda")],
        get_scaling=model.scaling_base[~torch.as_tensor(editable, device="cuda")].exp(),
        get_rotation=torch.nn.functional.normalize(model.rotation_base[~torch.as_tensor(editable, device="cuda")], dim=-1),
        get_features=model.features_base[~torch.as_tensor(editable, device="cuda")],
        get_opacity=model.opacity_base[~torch.as_tensor(editable, device="cuda")].sigmoid(),
        active_sh_degree=model.active_sh_degree)
    with torch.no_grad():
        baselines = []
        for view in views:
            p = render(view, scene=frozen)
            baselines.append(dict(rgb=p["render"].cpu(), alpha=p["rendered_alpha"].squeeze(0).cpu()))
    del frozen
    neighbors = {i: [] for i in range(len(views))}
    for i, j in select_pairs([camera(v) for v in views], cfg["init"]["neighbors"]):
        neighbors[i].append(j)
        neighbors[j].append(i)
    request_id = identity(request)
    checkpoint = args.output_root / "checkpoints/latest.pth"
    start, order, state = 0, [], None
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["request_id"] != request_id:
            raise ValueError("joint checkpoint belongs to different inputs/config")
        model.restore_local(state["local"])
        start, order = state["next_step"], state["view_order"]
        if not 0 <= start <= cfg["iterations"] or any(not 0 <= x < len(views) for x in order):
            raise ValueError("invalid joint checkpoint step/order")
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    # Pruning changes Parameter shapes. Restore row ownership first, then bind
    # Adam to the restored Parameters and load its correspondingly sliced state.
    optimizer = model.optimizer(cfg["optimizer"])
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
    validate_snapshots(args.output_root, request, before_step=start)
    @torch.no_grad()
    def capture(step, final=False):
        if not cfg["debug"]["enabled"]:
            return
        i = cfg["debug"]["view_index"]
        p = render(views[i], True)
        _, metrics = joint_losses(p, device(targets[i]), device(baselines[i]), max(0, step - 1), cfg, perceptual, gaussians=model)
        save_debug(args.output_root, p, device(targets[i]), metrics, step, final)
    @torch.no_grad()
    def snapshot(step):
        # Serialization only: no rendering, optimizer changes, or RNG sampling.
        # model.save also verifies that all protected background rows are intact.
        ply = args.output_root / "point_cloud" / f"iteration_{step}" / "point_cloud.ply"
        rows = args.output_root / "snapshots" / f"iteration_{step}" / "retained_rows.npy"
        model.save(ply)
        atomic_write(rows, lambda f: np.save(f, model.retained_rows.cpu().numpy(), allow_pickle=False))
        write_json(rows.parent / "manifest.json", seal("paintmesh-joint-snapshot",
            request=request, request_id=request_id, inputs=request["inputs"],
            initialization_id=initial["artifact_id"], iterations=step,
            training_budget=cfg["iterations"], point_count=len(model.retained_rows),
            outputs=dict(ply=record(ply), retained_rows=record(rows))))
        print(f"Saved same-run PGSR snapshot: {step}/{cfg['iterations']} {ply}", flush=True)
    capture(start)
    if start in request.get("save_iterations", []):
        snapshot(start)  # Recover a crash after checkpoint commit but before export.
    for step in range(start, cfg["iterations"]):
        if not order:
            order = list(range(len(views)))
            random.shuffle(order)
        index = order.pop()
        ramp = geometry_ramp(step, cfg)
        package = render(views[index], ramp > 0 and cfg["loss"]["depth_normal_consistency"] > 0)
        loss, metrics = joint_losses(package, device(targets[index]), device(baselines[index]), step, cfg, perceptual, gaussians=model)
        if ramp > 0 and cfg["loss"]["multiview"] > 0 and neighbors[index]:
            j = neighbors[index][step % len(neighbors[index])]
            other = render(views[j])
            mv = multiview_loss(package, other, views[index], views[j], targets[index]["mask"].cuda(), cfg)
            loss = loss + ramp * cfg["loss"]["multiview"] * mv
            metrics["multiview"] = mv.detach()
        # Only the adaptive camera mode declares these fixed planning weights.
        # Legacy circle/hemisphere losses and the global seg Trainer are untouched.
        if hasattr(views[index], "paintmesh_view_weight"):
            weight = float(views[index].paintmesh_view_weight)
            loss = loss * weight
            metrics["view_weight"] = loss.new_tensor(weight)
        metrics["total"] = loss.detach()
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite joint loss at {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError(f"nonfinite joint gradient at {step}")
        if density["enabled"] and step + 1 < density["until_iter"]:
            model.accumulate(package)
        optimizer.step()
        pruning = cfg["pruning"]
        if pruning["enabled"] and step + 1 < pruning["until_iter"]:
            if writeback["enabled"]:
                model.prune(optimizer, pruning["min_opacity"], preserve_old=True)
            else:
                model.prune(optimizer, pruning["min_opacity"])
        if (density["enabled"] and density["from_iter"] < step + 1 < density["until_iter"]
                and (step + 1) % density["interval"] == 0):
            counts = model.densify(optimizer, density, extent)
            print(f"ADC {step + 1}/{cfg['iterations']}: {counts} live={len(model.retained_rows)}", flush=True)
        if (step + 1) % cfg["debug"]["interval"] == 0:
            capture(step + 1)
        if step % cfg["diagnostic_interval"] == 0:
            values = {k: float(v) for k, v in metrics.items()}
            rows = model.retained_rows.cpu().numpy()
            new_count = int(editable[rows].sum())
            values.update(new_gaussians=new_count,
                          pruned_new_total=int(editable.sum()) - new_count,
                          old_gaussians=len(rows) - new_count,
                          pruned_old_total=int((~editable).sum()) - (len(rows) - new_count))
            if density["enabled"]:
                values.update(adaptive_density=dict(model.density_counts),
                              pruned_new_total=model.density_counts["pruned_new"],
                              net_new_growth=new_count - int(editable.sum()))
            write_json(args.output_root / "diagnostics" / f"step_{step:06d}.json", values)
            print(f"PGSR joint {step + 1}/{cfg['iterations']} loss={float(loss):.6f} "
                  f"coverage={float(metrics['coverage']):.3f} new={new_count} "
                  f"old={values['old_gaussians']} pruned_new={values['pruned_new_total']} "
                  f"pruned_old={values['pruned_old_total']}", flush=True)
        if (step + 1) % cfg["checkpoint_interval"] == 0 or step + 1 == cfg["iterations"]:
            state = dict(request_id=request_id, local=model.local_state(), optimizer=optimizer.state_dict(),
                next_step=step + 1, view_order=order, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            atomic_write(checkpoint, lambda f: torch.save(state, f))
        if step + 1 in request.get("save_iterations", []):
            snapshot(step + 1)
    writeback_outputs, writeback_summary = {}, None
    if writeback["enabled"]:
        from source.paintmesh_joint_writeback import select_writeback, restore_background
        seed = model.xyz_base[torch.as_tensor(editable, device="cuda")].cpu().numpy()
        selected, projection = select_writeback(model.get_xyz, views[0].full_proj_transform,
            targets[0]["mask"].numpy(), seed, writeback)
        permitted, restored = restore_background(model, selected)
        writeback_summary = dict(config=writeback, first_view=views[0].image_name,
            seed_source="unchanged initialization new Gaussian xyz", **projection, **restored)
        mask_path = args.output_root / "writeback_mask.npy"
        summary_path = args.output_root / "writeback_summary.json"
        atomic_write(mask_path, lambda f: np.save(f, permitted, allow_pickle=False))
        write_json(summary_path, writeback_summary)
        writeback_outputs = dict(writeback_mask=record(mask_path), writeback_summary=record(summary_path))
        print(f"Background writeback: {writeback_summary}", flush=True)
    capture(cfg["iterations"], final=True)
    final = []
    with torch.no_grad():
        for i, view in enumerate(views):
            p = render(view, True)
            _, stats = joint_losses(p, device(targets[i]), device(baselines[i]), cfg["iterations"] - 1, cfg, perceptual, gaussians=model)
            values = {k: float(v) for k, v in stats.items()}
            if not all(np.isfinite(list(values.values()))):
                raise ValueError("nonfinite joint final metrics")
            final.append(dict(frame=view.image_name, **values))
    if not any(v["coverage"] > 0 for v in final):
        raise ValueError("joint model has no valid hole coverage; not committing")
    model.save(output)
    retained_path = args.output_root / "retained_rows.npy"
    atomic_write(retained_path, lambda f: np.save(f, model.retained_rows.cpu().numpy(), allow_pickle=False))
    for value in request["inputs"].values():
        verify_record(value)
    for value in request["appearance_weights"].values():
        verify_record(value)
    validate_init(args.initialization)
    verify_targets(args.lama, args.camera, **cfg["supervision"])
    if implementation() != request["implementation"]:
        raise ValueError("joint implementation changed during training; result not committed")
    diagnostics = args.output_root / "diagnostics/final.json"
    rows = model.retained_rows.cpu().numpy()
    new_count = int(editable[rows].sum())
    old_count = len(rows) - new_count
    write_json(diagnostics, dict(frames=final, initialized_points=int(editable.sum()),
        fixed_topology=not cfg["pruning"]["enabled"] and not density["enabled"],
        adaptive_density=dict(**density, extent=extent,
                              counters=dict(model.density_counts) if density["enabled"] else {}),
        freeze_background=cfg["freeze_background"],
        local_background=cfg["local_background"],
        background_writeback=writeback_summary,
        pruning=dict(**cfg["pruning"], remaining_new=new_count,
                     pruned_new=model.density_counts["pruned_new"] if density["enabled"] else int(editable.sum()) - new_count,
                     remaining_old=old_count, pruned_old=int((~editable).sum()) - old_count,
                     trainable_old_initial=int((model.editable & ~editable).sum()),
                     frozen_background=int((~model.editable).sum()))))
    receipt = seal(JOINT_KIND, request=request, inputs=request["inputs"], initialization_id=initial["artifact_id"],
        parameters=dict(pipeline="edgs-pgsr", iterations=cfg["iterations"], pruning=cfg["pruning"],
                        adaptive_density=density,
                        freeze_background=cfg["freeze_background"], local_background=cfg["local_background"],
                        background_writeback=writeback),
        outputs=dict(ply=record(output), diagnostics=record(diagnostics), retained_rows=record(retained_path),
                     **writeback_outputs),
        point_count=len(model.retained_rows))
    validate_joint(receipt)
    write_json(args.manifest, receipt)
    print(f"PGSR joint training complete: {output}", flush=True)


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
