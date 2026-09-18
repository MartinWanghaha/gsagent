"""Prepare optional, run-local density-matched initialization; never alter gate support."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from local_geometry_io import (REPO, atomic_write, gate_membership, identity, read_json,
    record, seal, sha256, verify_record, verify_targets, write_json)
from support_geometry import DensityError, camera
from support_mass import load_config, sample_mass
from support_density_io import KIND, validate_density, validate_density_payload, write_support
from virtual_render_io import validate_render


def prepare(args):
    mode = getattr(args, "mode", "mass_adaptive")
    if mode != "mass_adaptive":
        raise ValueError("unsupported density mode; use mass_adaptive")
    cfg = load_config(args.config)
    if not 0 <= args.seed_frame < 30:
        raise ValueError("seed frame must be in [0,29]")
    lama, inputs = verify_targets(args.lama, args.camera)
    cameras = read_json(args.camera)
    metadata = inputs["parameters"]["normal"]
    render_path = verify_record(metadata["render_manifest"])
    render = validate_render(render_path.parent, "edgs-pgsr")
    if render["camera_artifact_id"] != cameras["artifact_id"]:
        raise ValueError("density render/camera mismatch")
    fusion = read_json(args.fusion)
    sys.path.insert(0, str(REPO / "submodules/Inpaint360GS"))
    from utils.fusion_manifest_identity import fusion_artifact_id
    if (fusion.get("kind") != "paintmesh-rgbd-fusion" or not fusion.get("complete") or
            fusion.get("artifact_id") != fusion_artifact_id(fusion) or
            fusion["upstream_artifact_ids"]["camera_manifest"] != cameras["artifact_id"] or
            fusion["upstream_artifact_ids"]["lama_completion_manifest"] != lama["artifact_id"]):
        raise ValueError("density fusion identity mismatch")
    stem = f"{args.seed_frame:05d}"
    if verify_record(fusion["outputs"][args.seed_frame]["fused_mask_ply"]) != args.support.resolve():
        raise ValueError("gate support is not the original fusion seed")
    deps, frames = [record(render_path), record(verify_record(lama["input_manifest"]))], []
    for i, c in enumerate(cameras["cameras"]):
        name = f"{i:05d}"
        if c["image_name"] != name:
            raise ValueError("unordered camera manifest")
        paths = {"depth": verify_record(lama["frames"][name]["outputs"]["depth"]),
                 "rgb": verify_record(lama["frames"][name]["outputs"]["color"]),
                 "mask": verify_record(inputs["frames"][name]["outputs"]["color_mask"]),
                 "removed": render_path.parent / "depth" / (name+".npy"),
                 "alpha": render_path.parent / "alpha" / (name+".npy")}
        for key in ("removed", "alpha"):
            relative = str(paths[key].relative_to(render_path.parent))
            if render["frames"][name]["outputs"].get(relative) != sha256(paths[key]):
                raise ValueError(f"unbound density input {paths[key]}")
        deps.extend(record(p) for p in paths.values())
        frame = {k: np.load(p, allow_pickle=False) for k, p in paths.items() if k not in ("rgb", "mask")}
        frame.update(rgb=np.array(Image.open(paths["rgb"]).convert("RGB")),
                     mask=np.array(Image.open(paths["mask"]).convert("L")) > 0, camera=camera(c))
        shape = (c["image_height"], c["image_width"])
        if any(frame[k].shape != shape for k in ("depth", "removed", "alpha", "mask")) or frame["rgb"].shape != (*shape,3):
            raise ValueError(f"density frame shape mismatch: {name}")
        frames.append(frame)
    exporter = REPO / "submodules/Inpaint360GS/tools/export_density_reference.py"
    implementations = [Path(__file__), Path(__file__).with_name("support_geometry.py"),
        Path(__file__).with_name("support_density_io.py"), exporter,
        REPO / "submodules/Inpaint360GS/edit_object_inpaint.py",
        REPO / "submodules/Inpaint360GS/utils/compose_utils.py",
        REPO / "submodules/Inpaint360GS/scene/gaussian_model.py"]
    implementations.append(Path(__file__).with_name("support_mass.py"))
    request = seal(KIND+"-request", inputs={k: record(getattr(args,k)) for k in
        ("source_ply", "classifier", "inpaint_config", "camera", "lama", "fusion", "support")},
        dependencies=deps, parameters=dict(config=cfg, seed_frame=args.seed_frame, mode=mode),
        implementation={str(p.relative_to(REPO)): sha256(p) for p in implementations})
    root = args.output_root.resolve()
    request_path = root / "request.json"
    if request_path.exists() and read_json(request_path) != request:
        raise ValueError("density inputs/settings changed; choose a new INPAINT_RUN_NAME")
    if args.manifest.exists():
        old = validate_density(args.manifest)
        if old["request_id"] != request["artifact_id"]:
            raise ValueError("density output belongs to another request")
        print(f"Reusing density support: {old['report']['initialized_points']} points")
        return
    if args.validate_only:
        raise ValueError("density support missing; run Stage 4 first")
    if not request_path.exists() and root.exists() and any(root.iterdir()):
        raise ValueError("refusing to replace unowned density directory")
    write_json(request_path, request)
    refpath = root / "reference.npz"
    subprocess.run([sys.executable, str(exporter), "--source-ply", str(args.source_ply),
        "--classifier", str(args.classifier), "--inpaint-config", str(args.inpaint_config),
        "--output", str(refpath), "--camera", str(args.camera), "--seed-frame", str(args.seed_frame),
        "--mask", inputs["frames"][stem]["outputs"]["color_mask"]["path"], "--support", str(args.support)],
        cwd=exporter.parents[1], check=True)
    with np.load(refpath, allow_pickle=False) as source:
        ref = dict(source)
    gate = lambda points: gate_membership(points, refpath, args.support)
    try:
        result, report, fields, quotas = sample_mass(frames,args.seed_frame,ref,cfg,gate)
        atomic_write(root / "density_field.npz",lambda f: np.savez_compressed(f,**fields))
        atomic_write(root / "quota.npz",lambda f: np.savez_compressed(f,**quotas))
    except DensityError as exc:
        write_json(root / "diagnostics.json", dict(status="failed", error=str(exc), **exc.report))
        raise
    write_support(root / "init_support.ply", result)
    atomic_write(root / "support.npz", lambda f: np.savez_compressed(f, **result))
    write_json(root / "diagnostics.json", report)
    if cfg["debug"]:
        frame = frames[args.seed_frame]
        preview = frame["rgb"].copy()
        uv = np.rint(result["seed_uv"]).astype(int)
        preview[uv[:,1],uv[:,0]] = [0,255,0]
        atomic_write(root / "debug/seed_support.png", lambda f: Image.fromarray(preview).save(f, format="PNG"))
        import cv2
        for name,values in (("target_density",fields["rho"]),("quota",quotas["count"]),
                            ("uncertainty",fields["geometry_uncertainty"])):
            canvas=np.zeros(frames[args.seed_frame]['depth'].shape,np.float64)
            transformed=np.log1p(values) if name!='uncertainty' else values
            span=float(np.ptp(transformed))
            normalized=(transformed-transformed.min())/span if span else np.ones_like(transformed)
            canvas.ravel()[fields['pixel_id']]=normalized
            colored=cv2.applyColorMap(np.rint(canvas*255).astype(np.uint8),cv2.COLORMAP_TURBO)
            colored[~frame['mask']]=0
            atomic_write(root/f"debug/{name}.png",lambda f,im=colored:Image.fromarray(im[:,:,::-1]).save(f,format='PNG'))
        atomic_write(root / "debug/reference_density.npz", lambda f: np.savez_compressed(f, **fields))
    receipt = seal(KIND, request_id=request["artifact_id"], parameters=request["parameters"],
        inputs={"request":record(request_path), **request["inputs"]}, report=report,
        outputs={"support": record(root/"init_support.ply"), "samples":record(root/"support.npz"),
                 "reference":record(refpath), "diagnostics":record(root/"diagnostics.json")})
    receipt['outputs'].update(field=record(root/'density_field.npz'),quota=record(root/'quota.npz'))
    receipt['artifact_id']=identity({k:v for k,v in receipt.items() if k!='artifact_id'})
    for p in implementations:
        if sha256(p) != request['implementation'][str(p.relative_to(REPO))]:
            raise ValueError('density implementation changed during preparation; choose a new run')
    validate_density_payload(receipt)
    write_json(args.manifest, receipt)
    print(f"Density support ready: {len(result['xyz'])} points; {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-ply", "classifier", "inpaint-config", "camera", "lama", "fusion", "support", "config", "output-root", "manifest"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--seed-frame", type=int, default=4)
    parser.add_argument("--mode",choices=("mass_adaptive",),default="mass_adaptive")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
