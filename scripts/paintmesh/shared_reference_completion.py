#!/usr/bin/env python3
"""Fixed training-reference RGB propagation; depth/normal/initial splats are unchanged.

The gates measure agreement of pseudo depth, not ground-truth confidence. Test
images and masks are deliberately not inputs to reference selection/completion.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, maximum_filter, minimum_filter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "submodules/EDGS"))
from source.paintmesh_local_data import make_views
from source.paintmesh_edgs_init import camera, project, unproject, sample
from run_geometry_ablation import read, write, now
from edgs_inpaint_io import (record, verify_record, verify_targets, validate_init,
    validate_selection, seal, INIT_KIND, SELECTION_KIND, identity)

POLICY = dict(reference_candidates=18, selection_samples_per_view=512,
    relative_depth_tolerance=0.02, cycle_pixels=1.5, depth_edge_relative=0.05,
    max_ray_angle_deg=60.0, trusted_erosion_pixels=1,
    reference_selection="maximum mean valid hole coverage over training views",
    reference_source="fixed existing independent LaMa completion",
    uncertain_policy="LaMa fills residual mask; zero-coverage views reuse baseline",
    uses_test_data=False, seed=0)


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def mask(path):
    return np.asarray(Image.open(path).convert("L")) > 0


def depth_valid(depth, tolerance):
    valid = np.isfinite(depth) & (depth > 0)
    clean = np.where(valid, depth, 0)
    spread = maximum_filter(clean, size=3) - minimum_filter(clean, size=3)
    return valid & (spread <= tolerance * np.maximum(clean, 1e-8))


def correspondence(uv, target, reference, target_cam, reference_cam, policy=POLICY):
    """Inverse warp with visibility, depth discontinuity, parallax and cycle gates."""
    d = sample(target["depth"], uv)
    xyz = unproject(uv, d, target_cam)
    q, z, inside = project(xyz, reference_cam)
    rd = sample(reference["depth"], q)
    good = inside & (d > 0) & (rd > 0) & np.isfinite(d) & np.isfinite(rd)
    good &= sample(target["valid"].astype(np.uint8), uv, 0) > 0
    good &= sample(reference["valid"].astype(np.uint8), q, 0) > 0
    good &= np.abs(z - rd) <= policy["relative_depth_tolerance"] * np.maximum(rd, 1e-8)
    back_xyz = unproject(q, rd, reference_cam)
    back_uv, _, back_inside = project(back_xyz, target_cam)
    good &= back_inside & (np.linalg.norm(back_uv - uv, axis=1) <= policy["cycle_pixels"])
    a = xyz - target_cam["c2w"][:3, 3]
    b = xyz - reference_cam["c2w"][:3, 3]
    cosine = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    good &= cosine >= np.cos(np.deg2rad(policy["max_ray_angle_deg"]))
    return q, good


def load_scene(row):
    lama, inputs, _ = verify_targets(row["inputs"]["lama"]["path"], row["inputs"]["camera"]["path"],
                                     use_depth=True, use_normal=True)
    cams = [camera(v) for v in make_views(row["inputs"]["camera"]["path"], device="cpu")]
    stems = lama["parameters"]["frame_names"]
    data = []
    for stem, cam in zip(stems, cams):
        out = lama["frames"][stem]["outputs"]
        inp = inputs["frames"][stem]["outputs"]
        d = np.load(out["depth"]["path"], allow_pickle=False)
        a = dict(depth=d, valid=depth_valid(d, POLICY["depth_edge_relative"]),
                 mask=mask(inp["color_mask"]["path"]), rgb=rgb(out["color"]["path"]),
                 original=rgb(inp["color"]["path"]))
        if d.shape != (cam["h"], cam["w"]) or a["mask"].shape != d.shape or a["rgb"].shape[:2] != d.shape:
            raise ValueError("Camera/depth/RGB dimensions differ: " + stem)
        if not np.array_equal(a["rgb"][~a["mask"]], a["original"][~a["mask"]]):
            raise ValueError("Baseline completion altered known pixels")
        data.append(a)
    if len(cams) != len(stems):
        raise ValueError("Camera/target counts differ")
    return lama, inputs, cams, stems, data


def select_reference(data, cams):
    rng = np.random.default_rng(POLICY["seed"])
    candidates = np.unique(np.linspace(0, len(data) - 1, min(len(data), POLICY["reference_candidates"])).astype(int))
    samples = []
    for frame in data:
        y, x = np.where(frame["mask"])
        rows = rng.choice(len(x), min(len(x), POLICY["selection_samples_per_view"]), replace=False)
        samples.append(np.column_stack((x[rows], y[rows])))
    scores = []
    for j in candidates:
        fractions = []
        for i, uv in enumerate(samples):
            if i == j:
                continue
            _, good = correspondence(uv, data[i], data[j], cams[i], cams[j])
            fractions.append(float(good.mean()) if len(good) else 0.)
        scores.append(dict(index=int(j), mean_coverage=float(np.mean(fractions))))
    best = max(scores, key=lambda a: (a["mean_coverage"], -a["index"]))
    return best["index"], scores


def prepare(root):
    root = Path(root)
    row = read(root / "source.json")
    for value in row["inputs"].values():
        verify_record(value)
    lama, inputs, cams, stems, data = load_scene(row)
    ref, scores = select_reference(data, cams)
    print(f"Selected training reference {stems[ref]}, estimated coverage {max(x['mean_coverage'] for x in scores):.3f}", flush=True)
    for folder in ("guide/input", "guide/trusted", "lama/output/color"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    evidence = dict(created_at=now(), policy=POLICY, source_lama=row["inputs"]["lama"],
        camera=row["inputs"]["camera"], producer=record(Path(__file__)), reference=stems[ref],
        reference_index=ref, candidates=scores, frames={})
    for i, (stem, frame) in enumerate(zip(stems, data)):
        trusted = np.zeros_like(frame["mask"])
        condition = frame["original"].copy()
        if i != ref:
            y, x = np.where(frame["mask"])
            uv = np.column_stack((x, y))
            q, good = correspondence(uv, frame, data[ref], cams[i], cams[ref])
            trusted[y[good], x[good]] = True
            trusted = binary_erosion(trusted, iterations=POLICY["trusted_erosion_pixels"])
            good = trusted[y, x]
            colors = np.stack([sample(data[ref]["rgb"][..., c].astype(float), q) for c in range(3)], axis=-1)
            condition[y[good], x[good]] = np.clip(np.rint(colors[good]), 0, 255).astype(np.uint8)
        residual = frame["mask"] & ~trusted
        guided = bool(trusted.any()) and i != ref
        if guided and residual.any():
            Image.fromarray(condition).save(root / "guide/input" / f"{stem}.png")
            Image.fromarray(residual.astype(np.uint8) * 255).save(root / "guide/input" / f"{stem}_mask.png")
            mode = "warp_and_lama_residual"
        else:
            if guided:
                Image.fromarray(condition).save(root / "lama/output/color" / f"{stem}.png")
                mode = "fully_supported_warp"
            else:
                shutil.copyfile(lama["frames"][stem]["outputs"]["color"]["path"], root / "lama/output/color" / f"{stem}.png")
                mode = "fixed_reference" if i == ref else "baseline_no_reliable_overlap"
        trusted_path = root / "guide/trusted" / f"{stem}.png"
        Image.fromarray(trusted.astype(np.uint8) * 255).save(trusted_path)
        row_evidence = dict(mode=mode, hole_pixels=int(frame["mask"].sum()), trusted_pixels=int(trusted.sum()),
            trusted_fraction=float(trusted.sum() / max(frame["mask"].sum(), 1)), trusted=record(trusted_path))
        if mode == "warp_and_lama_residual":
            row_evidence["condition"] = record(root / "guide/input" / f"{stem}.png")
            row_evidence["residual_mask"] = record(root / "guide/input" / f"{stem}_mask.png")
        evidence["frames"][stem] = row_evidence
    evidence["mean_hole_coverage_excluding_reference"] = float(np.mean([
        v["trusted_fraction"] for k, v in evidence["frames"].items() if k != stems[ref]]))
    evidence["guided_views"] = sum(v["mode"] == "warp_and_lama_residual" for v in evidence["frames"].values())
    write(root / "guide/evidence.json", evidence)
    print(f"Guide ready: {evidence['guided_views']}/{len(stems)} views, mean reliable coverage={evidence['mean_hole_coverage_excluding_reference']:.3f}", flush=True)


def consistency(cams, stems, data, completed):
    # Different view pairs/samples from reference selection. This measures target
    # compatibility only; an internally consistent hallucination can be wrong.
    rng = np.random.default_rng(123)
    before, after = [], []
    for i in range(1, len(data), 7):
        y, x = np.where(data[i]["mask"])
        rows = rng.choice(len(x), min(1024, len(x)), replace=False)
        uv = np.column_stack((x[rows], y[rows]))
        for offset in (3, 11, -11):
            j = (i + offset) % len(data)
            q, good = correspondence(uv, data[i], data[j], cams[i], cams[j])
            good &= sample(data[j]["mask"].astype(np.uint8), q, 0) > 0
            if not good.any():
                continue
            for images, dest in (([data[i]["rgb"], data[j]["rgb"]], before),
                                 ([completed[i], completed[j]], after)):
                color = np.stack([sample(images[1][..., c].astype(float), q[good]) for c in range(3)], axis=-1)
                source = images[0][y[rows][good], x[rows][good]].astype(float)
                dest.extend(np.abs(color - source).mean(1).tolist())
    return dict(samples=len(before), independent_mae_255=float(np.mean(before)) if before else None,
        shared_mae_255=float(np.mean(after)) if after else None,
        interpretation="Pseudo-depth-consistent training RGB disagreement; not geometric/GT accuracy")


def visualization(root, stems, data, completed, evidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = sorted(range(len(stems)), key=lambda i: -evidence["frames"][stems[i]]["trusted_fraction"])
    chosen = sorted(set(chosen[:3] + list(np.linspace(0, len(stems)-1, 4).astype(int))))
    fig, axs = plt.subplots(len(chosen), 4, figsize=(16, len(chosen)*3), squeeze=False)
    for row, i in enumerate(chosen):
        ys, xs = np.where(data[i]["mask"])
        x0, x1 = max(0, xs.min()-24), min(data[i]["mask"].shape[1], xs.max()+25)
        y0, y1 = max(0, ys.min()-24), min(data[i]["mask"].shape[0], ys.max()+25)
        trusted = mask(evidence["frames"][stems[i]]["trusted"]["path"])
        for col, (im, title) in enumerate(((data[i]["rgb"], "Independent LaMa"),
            (trusted, "Accepted reference warp"), (completed[i], "Shared reference + residual LaMa"),
            (np.abs(completed[i].astype(float)-data[i]["rgb"].astype(float)).mean(2), "RGB change (0-64)"))):
            axs[row,col].imshow(im[y0:y1,x0:x1], cmap="magma" if col==3 else "gray", **({"vmin":0,"vmax":64} if col==3 else {}))
            axs[row,col].set_title(f"{stems[i]} | {title}");axs[row,col].axis("off")
    fig.suptitle(f"{root.name}: fixed training reference {evidence['reference']}; views selected by coverage / uniform index")
    fig.tight_layout();fig.savefig(root / "guide/target_comparison.png", dpi=130);plt.close(fig)
    Image.fromarray(data[evidence["reference_index"]]["rgb"]).save(root / "guide/reference.png")


def finalize(root):
    root = Path(root); row = read(root / "source.json")
    evidence = read(root / "guide/evidence.json")
    if evidence["policy"] != POLICY or evidence["producer"] != record(Path(__file__)):
        raise ValueError("Guide policy/implementation changed")
    lama, inputs, cams, stems, data = load_scene(row)
    for val in evidence["frames"].values():
        for key in ("trusted", "condition", "residual_mask"):
            if key in val: verify_record(val[key])
    completed = []
    for i, stem in enumerate(stems):
        p = root / "lama/output/color" / f"{stem}.png"
        new = rgb(p)
        if new.shape != data[i]["rgb"].shape or not np.array_equal(new[~data[i]["mask"]], data[i]["original"][~data[i]["mask"]]):
            raise ValueError("Output changed dimensions or known pixels: " + stem)
        ev = evidence["frames"][stem]
        if "condition" in ev:
            trust = mask(ev["trusted"]["path"])
            if not np.array_equal(new[trust], rgb(ev["condition"]["path"])[trust]):
                raise ValueError("LaMa overwrote trusted reference pixels")
        completed.append(new)
    report = dict(complete=True, scene=row["scene"], reference=evidence["reference"],
        mean_hole_coverage=evidence["mean_hole_coverage_excluding_reference"], guided_views=evidence["guided_views"],
        outside_mask_byte_exact=True, depth_and_normal_unchanged=True,
        consistency=consistency(cams, stems, data, completed))
    write(root / "guide/validation.json", report)
    visualization(root, stems, data, completed, evidence)
    # Explicit derived-target receipt: same source masks/depth/normal/model, RGB
    # produced by the declared guide process plus the original LaMa refiner.
    derived = copy.deepcopy(lama)
    derived.update(created_at=now())
    derived["roots"]["color_output"] = str((root / "lama/output/color").resolve())
    derived["parameters"]["shared_reference"] = dict(policy=POLICY, evidence=record(root / "guide/evidence.json"),
        validation=record(root / "guide/validation.json"), baseline=row["inputs"]["lama"],
        producer=record(Path(__file__)), reference=evidence["reference"],
        predictor=record(REPO / "submodules/Inpaint360GS/LaMa/bin/predict_color.py"),
        refiner=record(REPO / "submodules/Inpaint360GS/LaMa/saicinpainting/evaluation/refinement.py"))
    for stem in stems:
        output_path = root / "lama/output/color" / f"{stem}.png"
        # Legacy publisher checks timestamps on completion image records.
        derived["frames"][stem]["outputs"]["color"] = dict(record(output_path), mtime_ns=output_path.stat().st_mtime_ns)
    derived["artifact_id"] = identity(dict(kind=derived["kind"], schema_version=derived["schema_version"],
        input_artifact_id=derived["input_artifact_id"], model={k:v["sha256"] for k,v in derived["model"].items()},
        parameters=derived["parameters"], outputs={s:{k:v["sha256"] for k,v in f["outputs"].items()} for s,f in derived["frames"].items()}))
    dest = root / "manifests/lama_completion_manifest.json"
    write(dest, derived)
    verify_targets(dest, row["inputs"]["camera"]["path"], use_depth=True, use_normal=True)
    # The initialization and local eligibility are deliberately *reused*, not
    # regenerated from the new targets. The parent receipt remains explicit.
    old = validate_init(row["inputs"]["initialization"]["path"])
    initial_inputs = dict(old["inputs"], lama=record(dest), proposal_initialization=row["inputs"]["initialization"])
    request = copy.deepcopy(old["request"])
    request.update(inputs=initial_inputs, variant="fixed_initialization_rgb_target_ablation",
        output_root=str(root / "edgs_init"), manifest=str(root / "manifests/edgs_init_manifest.json"),
        reuse_policy="All original Gaussian fields/support/ownership byte-identical; initialized using parent LaMa targets")
    init = seal(INIT_KIND, request=request, inputs=initial_inputs, outputs=old["outputs"],
        point_count=old["point_count"], parameters=old["parameters"], report=old["report"])
    validate_init(init)
    init_path = root / "manifests/edgs_init_manifest.json"; write(init_path, init)
    new_row = copy.deepcopy(row)
    new_row["inputs"].update(lama=record(dest), initialization=record(init_path))
    if "local_selection" in row["inputs"]:
        old_sel = validate_selection(row["inputs"]["local_selection"]["path"], initial=old)
        si = dict(old_sel["inputs"], lama=record(dest), initialization=record(init_path), baseline_selection=row["inputs"]["local_selection"])
        sr = copy.deepcopy(old_sel["request"]); sr.update(inputs=si, output_root=str(root / "local_selection"),
            policy="Reuse identical trainable rows/votes: geometry, masks and initial splats unchanged")
        selection = seal(SELECTION_KIND, request=sr, inputs=si, outputs=old_sel["outputs"], initialization_id=init["artifact_id"])
        validate_selection(selection, initial=init, config=row["variants"][0]["config"]["local_background"])
        sp=root / "manifests/local_selection_manifest.json";write(sp, selection)
        new_row["inputs"]["local_selection"] = record(sp)
    new_row["publish"]["inputs"]["lama_manifest"] = record(dest)
    write(root / "training_source.json", new_row)
    print("Shared-reference targets validated; initialization and geometry unchanged:", report, flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "finalize"));p.add_argument("--root", type=Path, required=True)
    a=p.parse_args();globals()[a.action](a.root.resolve())
