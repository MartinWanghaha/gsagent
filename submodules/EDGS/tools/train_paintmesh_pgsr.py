#!/usr/bin/env python3
"""Stage 5: new-Gaussian RGB/geometry joint training with frozen background."""
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
    verify_record, verify_targets, write_json)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("initialization", "lama", "camera", "edgs-config", "output-root", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    config_arguments(p)
    return p


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
    if any(initial["request"]["config"][k] != cfg[k] for k in ("init", "matcher", "seed")):
        raise ValueError("initialization config differs from joint request")
    edgs = yaml.safe_load(args.edgs_config.read_text())
    if edgs["gs"]["renderer"]["backend"] != "pgsr":
        raise ValueError("joint training requires EDGS-PGSR model configuration")
    removed = read_json(initial["inputs"]["removed_model"]["path"])
    if record(args.edgs_config)["sha256"] != removed["inputs"]["edgs_config"]["sha256"]:
        raise ValueError("joint training EDGS config differs from removal")
    records = {k: record(getattr(args, k)) for k in ("initialization", "lama", "camera", "edgs_config")}
    request = dict(pipeline="edgs-pgsr", inputs=records, config=cfg, implementation=implementation(),
                   output_root=str(args.output_root.resolve()), manifest=str(args.manifest.resolve()))
    output = args.output_root / "point_cloud" / f"iteration_{cfg['iterations']}" / "point_cloud.ply"
    if args.manifest.exists():
        validate_joint(args.manifest, selected_ply=output, request=request)
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
    from source.paintmesh_joint_losses import joint_losses, multiview_loss, geometry_ramp
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
    model = JointGaussians(initial["outputs"]["ply"]["path"], editable)
    optimizer = model.optimizer(cfg["optimizer"])
    renderer = PGSRRenderer()
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    background = torch.tensor([1., 1., 1.] if edgs["gs"]["dataset"]["white_background"] else [0., 0., 0.], device="cuda")
    def render(view, consistency=False, scene=model):
        package = renderer.render(view, scene, pipeline, background, return_plane=True)
        if consistency:
            package["depth_normal"] = local_depth_normal(view, package["plane_depth"])
        return package
    def device(target):
        return {k: v.cuda() for k, v in target.items()}
    # Render only the frozen removed rows for the known-region baseline. This
    # cannot contain the initialized hole and is recreated before resume.
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
    start, order = 0, []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["request_id"] != request_id:
            raise ValueError("joint checkpoint belongs to different inputs/config")
        model.restore_local(state["local"])
        optimizer.load_state_dict(state["optimizer"])
        start, order = state["next_step"], state["view_order"]
        if not 0 <= start <= cfg["iterations"] or any(not 0 <= x < len(views) for x in order):
            raise ValueError("invalid joint checkpoint step/order")
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    @torch.no_grad()
    def capture(step, final=False):
        if not cfg["debug"]["enabled"]:
            return
        i = cfg["debug"]["view_index"]
        p = render(views[i], True)
        _, metrics = joint_losses(p, device(targets[i]), device(baselines[i]), max(0, step - 1), cfg)
        save_debug(args.output_root, p, device(targets[i]), metrics, step, final)
    capture(start)
    for step in range(start, cfg["iterations"]):
        if not order:
            order = list(range(len(views)))
            random.shuffle(order)
        index = order.pop()
        ramp = geometry_ramp(step, cfg)
        package = render(views[index], ramp > 0 and cfg["loss"]["depth_normal_consistency"] > 0)
        loss, metrics = joint_losses(package, device(targets[index]), device(baselines[index]), step, cfg)
        if ramp > 0 and cfg["loss"]["multiview"] > 0 and neighbors[index]:
            j = neighbors[index][step % len(neighbors[index])]
            other = render(views[j])
            mv = multiview_loss(package, other, views[index], views[j], targets[index]["mask"].cuda(), cfg)
            loss = loss + ramp * cfg["loss"]["multiview"] * mv
            metrics["multiview"] = mv.detach()
        metrics["total"] = loss.detach()
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite joint loss at {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError(f"nonfinite joint gradient at {step}")
        optimizer.step()
        if (step + 1) % cfg["debug"]["interval"] == 0:
            capture(step + 1)
        if step % cfg["diagnostic_interval"] == 0:
            values = {k: float(v) for k, v in metrics.items()}
            write_json(args.output_root / "diagnostics" / f"step_{step:06d}.json", values)
            print(f"PGSR joint {step + 1}/{cfg['iterations']} loss={float(loss):.6f} coverage={float(metrics['coverage']):.3f}", flush=True)
        if (step + 1) % cfg["checkpoint_interval"] == 0 or step + 1 == cfg["iterations"]:
            state = dict(request_id=request_id, local=model.local_state(), optimizer=optimizer.state_dict(),
                next_step=step + 1, view_order=order, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            atomic_write(checkpoint, lambda f: torch.save(state, f))
    capture(cfg["iterations"], final=True)
    final = []
    with torch.no_grad():
        for i, view in enumerate(views):
            p = render(view, True)
            _, stats = joint_losses(p, device(targets[i]), device(baselines[i]), cfg["iterations"] - 1, cfg)
            values = {k: float(v) for k, v in stats.items()}
            if not all(np.isfinite(list(values.values()))):
                raise ValueError("nonfinite joint final metrics")
            final.append(dict(frame=view.image_name, **values))
    if not any(v["coverage"] > 0 for v in final):
        raise ValueError("joint model has no valid hole coverage; not committing")
    model.save(output)
    for value in request["inputs"].values():
        verify_record(value)
    validate_init(args.initialization)
    verify_targets(args.lama, args.camera, **cfg["supervision"])
    if implementation() != request["implementation"]:
        raise ValueError("joint implementation changed during training; result not committed")
    diagnostics = args.output_root / "diagnostics/final.json"
    write_json(diagnostics, dict(frames=final, initialized_points=int(editable.sum()), fixed_topology=True))
    receipt = seal(JOINT_KIND, request=request, inputs=request["inputs"], initialization_id=initial["artifact_id"],
        parameters=dict(pipeline="edgs-pgsr", iterations=cfg["iterations"]),
        outputs=dict(ply=record(output), diagnostics=record(diagnostics)), point_count=len(editable))
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
