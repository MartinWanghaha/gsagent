#!/usr/bin/env python3
"""Pilot validation of mixed candidates; no plane fit, surface prior or training.

The output is conditional geometric consistency, NOT calibrated accuracy.
Even/odd camera sets separate selection and diagnostic validation. Each
candidate's original source pair is excluded from correspondence evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/paintmesh"), str(REPO / "submodules/EDGS")]

import numpy as np
from scipy.ndimage import map_coordinates

DEFAULT_BASE = REPO / ("output/paintmesh/Inpaint360GS/inpaint360/bag/removal/"
    "table1_adaptive180_sweep/inpaint/edgs_depth_local_prune_perceptual")
POLICY = dict(roma_confidence=.7, cycle_pixels=3., reprojection_pixels=2.,
    min_parallax_degrees=5., depth_tolerance_spacing=2., min_depth_views=6,
    min_depth_agreement=.75, max_pseudo_front_fraction=.10, max_known_front_views=1,
    min_rgb_edges=2, min_rgb_views=3, assumed_pixel_sigma=1., max_sigma_spacing=1.,
    anchor_count=12, pair_angles=[20., 35.], perturbation_spacing=6., controls=5000, seed=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def interpolate(array, uv, order=1):
    return map_coordinates(array, [uv[:, 1], uv[:, 0]], order=order, mode="nearest")


def warp_observations(forward, fc, reverse, rc, uv, shape_a, shape_b):
    """RoMa align_corners=False warp sampling, with an explicit valid domain."""
    ha, wa = shape_a; hb, wb = shape_b
    q = (uv + .5) * [forward.shape[1] / wa, forward.shape[0] / ha] - .5
    norm = np.column_stack([interpolate(forward[..., k], q) for k in (2, 3)])
    target = (norm + 1) * [wb / 2, hb / 2] - .5
    valid = np.isfinite(target).all(1)
    valid &= (uv >= 0).all(1) & (uv <= [wa-1, ha-1]).all(1)
    valid &= (target >= 0).all(1) & (target <= [wb-1, hb-1]).all(1)
    safe = np.where(valid[:, None], target, 0.)
    back_q = (safe + .5) * [reverse.shape[1] / wb, reverse.shape[0] / hb] - .5
    back = np.column_stack([interpolate(reverse[..., k], back_q) for k in (2, 3)])
    back = (back + 1) * [wa / 2, ha / 2] - .5
    certainty = np.minimum(interpolate(fc, q), interpolate(rc, back_q))
    return target, certainty, np.linalg.norm(back - uv, axis=1), valid


def point_information(xyz, cam, pixel_sigma=1.):
    """Conditional position information in world units, including camera scale."""
    rot = cam["w2c"][:3, :3]
    p = xyz @ rot.T + cam["w2c"][:3, 3]
    z = np.maximum(p[:, 2], 1e-9)
    jac = np.stack([cam["k"][k, k] * (rot[k][None] / z[:, None]
        - p[:, k, None] * rot[2][None] / z[:, None]**2) for k in range(2)], axis=1)
    return np.einsum("nki,nkj->nij", jac, jac) / pixel_sigma**2


def sigma_max(information):
    eig = np.linalg.eigvalsh(information)
    result = np.full(len(eig), np.inf)
    good = eig[:, 0] > 1e-8
    result[good] = 1 / np.sqrt(eig[good, 0])
    return result


def select_points(depth, rgb, policy=POLICY):
    ratio = depth["agree"] / np.maximum(depth["available"], 1)
    front = depth["front"] / np.maximum(depth["available"], 1)
    return ((depth["agree"] >= policy["min_depth_views"])
        & (ratio >= policy["min_depth_agreement"])
        & (front <= policy["max_pseudo_front_fraction"])
        & (depth["known_front"] <= policy["max_known_front_views"])
        & (rgb["edges"] >= policy["min_rgb_edges"])
        & (rgb["views"].sum(1) >= policy["min_rgb_views"])
        & (rgb["sigma_spacing"] <= policy["max_sigma_spacing"]))


def choose_pairs(cameras, xyz, policy):
    center = np.median(xyz, axis=0)
    directions = np.array([cam["c2w"][:3, 3] - center for cam in cameras])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1)))
    anchors = np.linspace(0, len(cameras), policy["anchor_count"], endpoint=False).astype(int)
    pairs = []
    for index, a in enumerate(anchors):
        # Disjoint camera sets, in addition to excluding each point's source pair.
        role = "fit" if index % 2 == 0 else "check"
        wanted_parity = 0 if role == "fit" else 1
        a += (wanted_parity - a) % 2
        a %= len(cameras)
        pool = np.arange(wanted_parity, len(cameras), 2)
        pool = pool[(pool != a) & (angles[a, pool] >= policy["min_parallax_degrees"])]
        chosen = []
        for desired in policy["pair_angles"]:
            candidates = pool[~np.isin(pool, chosen)]
            if not len(candidates):
                continue
            b = int(candidates[np.argmin(np.abs(angles[a, candidates] - desired))])
            chosen.append(b)
            pairs.append(dict(a=int(a), b=b, role=role, centroid_parallax_degrees=float(angles[a,b])))
    return pairs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", type=Path, default=DEFAULT_BASE)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    args = p.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import torch
    from PIL import Image
    from plyfile import PlyData, PlyElement
    from edgs_inpaint_io import validate_init, verify_targets, record, verify_record
    from source.paintmesh_local_data import make_views
    from source.paintmesh_edgs_init import camera, project, RomaMatcher

    started = time.time(); root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if (root / "request.json").exists():
        raise ValueError("Use a fresh output root; preserve the registered policy and diagnostic split")
    manifest_path = args.baseline_run.resolve() / "manifests/edgs_init_manifest.json"
    initial = validate_init(manifest_path)
    lama, inputs, _ = verify_targets(initial["inputs"]["lama"]["path"], initial["inputs"]["camera"]["path"], use_depth=True)
    support = dict(np.load(initial["outputs"]["support"]["path"], allow_pickle=False))
    original = support["xyz"].astype(float); count = len(original); gap = float(support["spacing"])
    cameras = [camera(v) for v in make_views(initial["inputs"]["camera"]["path"], device="cpu")]
    stems = sorted(lama["frames"])
    rng = np.random.default_rng(POLICY["seed"])
    parent = rng.choice(count, min(POLICY["controls"], count), replace=False)
    source_c = np.array([cameras[int(support["frame"][k])]["c2w"][:3, 3] for k in parent])
    displacement = original[parent] - source_c
    displacement /= np.linalg.norm(displacement, axis=1, keepdims=True)
    sign = rng.choice([-1., 1.], len(parent))
    perturbed = original[parent] + displacement * (sign * gap * POLICY["perturbation_spacing"])[:, None]
    xyz = np.concatenate((original, perturbed))
    source_frames = np.concatenate((support["frame"], support["frame"][parent]))
    other_frames = np.concatenate((support["other_frame"], support["other_frame"][parent]))
    n = len(xyz); pairs = choose_pairs(cameras, original, POLICY)
    write(root / "request.json", dict(created_at=datetime.now().astimezone().isoformat(), policy=POLICY,
        baseline=record(manifest_path), inputs=initial["inputs"], support=initial["outputs"]["support"],
        script=record(__file__), matcher_weights=initial["request"]["matcher_weights"], pairs=pairs,
        candidate_count=count, spacing_world=gap, gpu=args.gpu,
        scope="candidate evidence filtering only; no surface reconstruction/training/GT",
        caveat="same synthetic completion; split observations are correlated and not blind ground truth"))
    write(root / "status.json", dict(state="running", phase="depth", original_points=count))
    print(f"CANDIDATES {count}; controls={len(parent)}; registered pairs={len(pairs)}", flush=True)
    depths = {role: {key: np.zeros(n, dtype=np.uint16) for key in
        ("available", "agree", "front", "known_front")} for role in ("fit", "check")}
    masks, images = {}, {}
    used_frames = {pair[k] for pair in pairs for k in ("a", "b")}
    for i, stem in enumerate(stems):
        cam = cameras[i]; uv, z, inside = project(xyz, cam)
        safe = np.where(inside[:, None], uv, 0.)
        depth = np.load(lama["frames"][stem]["outputs"]["depth"]["path"], allow_pickle=False)
        with Image.open(inputs["frames"][stem]["outputs"]["color_mask"]["path"]) as im:
            mask = np.asarray(im.convert("L")) > 0
        d = interpolate(depth, safe); hole = interpolate(mask.astype(float), safe, 0) > 0
        valid = inside & (d > 0) & (i != source_frames) & (i != other_frames)
        tol = POLICY["depth_tolerance_spacing"] * gap * np.linalg.norm(cam["w2c"][2, :3])
        # Completed depths provide soft consistency only. Unedited alpha/depth
        # are a model-based free-space proxy, NOT a real sensor measurement.
        part = depths["fit" if i % 2 == 0 else "check"]
        part["available"] += valid & hole
        part["agree"] += valid & hole & (np.abs(z-d) <= tol)
        part["front"] += valid & hole & (z < d-tol)
        observed = np.load(inputs["frames"][stem]["inputs"]["removed_depth"]["path"], allow_pickle=False)
        alpha = np.load(inputs["frames"][stem]["inputs"]["removed_alpha"]["path"], allow_pickle=False)
        observed_d = interpolate(observed, safe)
        trusted = valid & ~hole & (observed_d > 0) & (interpolate(alpha, safe) >= .95)
        part["known_front"] += trusted & (z < observed_d-tol)
        if i in used_frames:
            masks[i] = mask
            images[i] = Image.open(lama["frames"][stem]["outputs"]["color"]["path"]).convert("RGB").copy()
        if i % 30 == 0:
            print(f"DEPTH {i+1}/{len(stems)}", flush=True)
    np.savez_compressed(root / "depth_evidence.npz", **{role+"_"+key: val
        for role, data in depths.items() for key, val in data.items()})
    rgb = {role: dict(edges=np.zeros(n, dtype=np.uint16), eligible=np.zeros(n, dtype=np.uint16),
        views=np.zeros((n,len(cameras)), dtype=bool), information=np.zeros((n,3,3)),
        error_sum=np.zeros(n), certainty_sum=np.zeros(n)) for role in ("fit", "check")}
    for value in initial["request"]["matcher_weights"].values():
        verify_record(value)
    print("Loading cached RoMa weights on GPU",args.gpu,flush=True)
    matcher = RomaMatcher(initial["request"]["config"]["matcher"], initial["request"]["matcher_weights"])
    cache = root / "match_cache"; cache.mkdir()
    pair_stats = []
    for edge, pair in enumerate(pairs):
        a, b, role = [pair[k] for k in ("a", "b", "role")]
        with torch.no_grad():
            forward, fc = matcher.model.match(images[a], images[b], device="cuda")
            reverse, rc = matcher.model.match(images[b], images[a], device="cuda")
        forward, fc, reverse, rc = [v.detach().cpu().numpy() for v in (forward,fc,reverse,rc)]
        np.savez_compressed(cache / f"{role}_{a:05d}_{b:05d}.npz", forward=forward,
                            forward_confidence=fc, reverse=reverse, reverse_confidence=rc)
        auv, _, av = project(xyz, cameras[a]); buv, _, bv = project(xyz, cameras[b])
        safe_a = np.where(av[:,None], auv, 0.); safe_b = np.where(bv[:,None], buv, 0.)
        matched, confidence, cycle, valid = warp_observations(forward,fc,reverse,rc,safe_a,
            masks[a].shape,masks[b].shape)
        in_holes = (interpolate(masks[a].astype(float),safe_a,0)>0)
        in_holes &= interpolate(masks[b].astype(float),safe_b,0)>0
        in_holes &= interpolate(masks[b].astype(float),np.where(valid[:,None],matched,0.),0)>0
        # Never validate a candidate using its original two-view source edge.
        novel = ~(((source_frames==a)&(other_frames==b)) | ((source_frames==b)&(other_frames==a)))
        directions = [xyz-cameras[k]["c2w"][:3,3] for k in (a,b)]
        angle = np.degrees(np.arccos(np.clip(np.einsum('ni,ni->n',*directions) /
            (np.linalg.norm(directions[0],axis=1)*np.linalg.norm(directions[1],axis=1)), -1,1)))
        eligible = av & bv & valid & in_holes & novel & (angle>=POLICY["min_parallax_degrees"])
        error = np.linalg.norm(matched-buv,axis=1)
        accepted = (eligible & (confidence>=POLICY["roma_confidence"])
            & (cycle<=POLICY["cycle_pixels"]) & (error<=POLICY["reprojection_pixels"]))
        part=rgb[role]; part["eligible"] += eligible; part["edges"] += accepted
        part["error_sum"] += np.where(accepted,error,0.)
        part["certainty_sum"] += np.where(accepted,confidence,0.)
        for k in (a,b):
            added = accepted & ~part["views"][:,k]
            part["information"][added] += point_information(xyz[added],cameras[k],POLICY["assumed_pixel_sigma"])
            part["views"][accepted,k]=True
        stats=dict(**pair, eligible=int(eligible[:count].sum()), accepted=int(accepted[:count].sum()))
        pair_stats.append(stats);write(root / "pair_statistics.json",pair_stats)
        write(root / "status.json",dict(state="running",phase="RoMa validation",pair=edge+1,total_pairs=len(pairs)))
        print(f"MATCH {edge+1}/{len(pairs)} {role} {a}/{b}: {stats['accepted']}/{stats['eligible']} points",flush=True)
    del matcher
    for part in rgb.values():
        part["sigma_spacing"] = sigma_max(part.pop("information"))/gap
    fit = select_points(depths["fit"],rgb["fit"])
    check = select_points(depths["check"],rgb["check"])
    strict = fit & check
    score = np.sqrt(np.prod([depths[r]["agree"] / np.maximum(depths[r]["available"],1)
        * rgb[r]["edges"] / np.maximum(rgb[r]["eligible"],1) for r in ("fit","check")],axis=0))
    arrays=dict(xyz=xyz, source_kind=np.concatenate((support["source_kind"],support["source_kind"][parent])),
        fit_pass=fit,check_pass=check,strict_pass=strict,consistency_score=score,control_parent=parent)
    for role in ("fit","check"):
        arrays.update({role+"_depth_"+key:value for key,value in depths[role].items()})
        arrays.update({role+"_rgb_"+key:value for key,value in rgb[role].items()})
    np.savez_compressed(root / "point_evidence.npz",**arrays)
    selected=np.flatnonzero(strict[:count]); selected_fit=np.flatnonzero(fit[:count])
    vertex=PlyData.read(initial["outputs"]["ply"]["path"],mmap=True)["vertex"].data
    ownership=np.load(initial["outputs"]["editable_mask"]["path"])
    new=vertex[ownership]
    for name,rows in (("high_confidence_gaussians.ply",selected),("fit_selected_gaussians.ply",selected_fit)):
        PlyData([PlyElement.describe(new[rows].copy(),"vertex")]).write(str(root/name))
    point_dtype=[(k,"f4") for k in "xyz"]+[(k,"u1") for k in ("red","green","blue")]+[("consistency_score","f4")]
    for name,rows,confidence_colors in (("original_mixed_points.ply",np.arange(count),False),
        ("high_confidence_points.ply",selected,False),("high_confidence_colored.ply",selected,True)):
        pts=np.zeros(len(rows),dtype=point_dtype)
        for k,key in enumerate("xyz"):pts[key]=original[rows,k]
        color=support["rgb"][rows]
        if confidence_colors:
            color=np.column_stack((1-score[rows],score[rows],np.zeros(len(rows))))
        for k,key in enumerate(("red","green","blue")):pts[key]=np.rint(np.clip(color[:,k],0,1)*255).astype("u1")
        pts["consistency_score"]=score[rows]
        PlyData([PlyElement.describe(pts,"vertex")]).write(str(root/name))
    def population(mask):
        return dict(count=int(mask.sum()),fit_selected=int((fit&mask).sum()),
            fit_selected_check_pass_fraction=float((strict&mask).sum()/max(1,(fit&mask).sum())),
            strict_selected=int((strict&mask).sum()),strict_fraction=float((strict&mask).sum()/max(1,mask.sum())))
    original_mask=np.arange(n)<count
    counts=dict(mixed=population(original_mask),perturbed=population(~original_mask))
    for label,kind in (("rgb",0),("depth",1)):
        counts[label]=population(original_mask&(arrays["source_kind"]==kind))
    # Compact synthetic-observation diagnostics, not Table-1 evaluation.
    selected_stats={}
    for role in ("fit","check"):
        part=rgb[role]; mean_error=part["error_sum"] / np.maximum(part["edges"],1)
        selected_stats[role]=dict(distinct_rgb_views_median=float(np.median(part["views"][selected].sum(1))) if len(selected) else None,
            reprojection_pixels_median=float(np.median(mean_error[selected])) if len(selected) else None,
            conditional_sigma_spacing_p95=float(np.quantile(part["sigma_spacing"][selected],.95)) if len(selected) else None,
            depth_agreement_median=float(np.median(depths[role]["agree"][selected]/np.maximum(depths[role]["available"][selected],1))) if len(selected) else None)
    summary=dict(complete=True,scope="evidence screening pilot; no surface fit, mesh or training",policy=POLICY,
        populations=counts,selected_statistics=selected_stats,elapsed_seconds=time.time()-started,
        verdict="consistent subset exists" if len(selected) else "strict evidence insufficient; thresholds were not relaxed",
        confidence_is_calibrated_probability=False,ground_truth_used=False,background_modified=False,
        limitations=["Synthetic views and LaMa completions have correlated systematic errors.",
            "Even/odd checks are additional correspondence evidence, not a blind test or real geometry accuracy.",
            "Pixel-noise covariance is conditional; it omits hallucination bias and camera errors.",
            "Perturbed points may fall on valid surfaces; their pass rate is a control, not false-positive accuracy.",
            "Surface reconstruction, missing-region completion and post-training stability remain untested."])
    make_plots(root,original,support["rgb"],selected,selected_fit,score,cameras,images,depths,rgb,count)
    write(root/"summary.json",summary)
    lines=["# 混合初始化候选的几何置信度初步验证","",summary["verdict"],"",
        "只筛选原混合候选；没有拟合平面、重建曲面、改变坐标或进行训练。fit 使用偶数相机，check 使用奇数相机；每个点排除原始匹配边。", "",
        "这里的置信度表示对当前补全数据的条件一致性，不是经真实几何标定的正确概率。", "",
        "| 来源 | 候选 | fit通过 | fit点在check中通过率 | 两组均通过 | 最终保留率 |",
        "|---|---:|---:|---:|---:|---:|"]
    for label,value in counts.items():
        lines.append(f"| {label} | {value['count']} | {value['fit_selected']} | {value['fit_selected_check_pass_fraction']:.2%} | {value['strict_selected']} | {value['strict_fraction']:.2%} |")
    lines += ["", "固定阈值：`"+json.dumps(POLICY,ensure_ascii=False)+"`。", "",
        "扰动对照为同一批候选沿源相机射线正负移动6个世界点间距，不代表全部一定为错误点。条件位置方差假设1像素观测噪声；其余偏差见summary.json。", "",
        "![几何分布对照](geometry_comparison.png)","", "![虚拟视图投影](projection_checks.png)","",
        "可导入SuperSplat的高置信度候选：high_confidence_gaussians.ply；常规RGB点云：high_confidence_points.ply；逐点证据：point_evidence.npz。", "",
        "未使用GT或测试图像。旧高斯保持原样且不包含在这些新点PLY中。后续必须单独验证曲面覆盖、真实几何精度以及训练漂移。"]
    (root/"report.zh-CN.md").write_text("\n".join(lines)+"\n")
    write(root/"status.json",dict(state="complete",summary=str(root/"summary.json")))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def make_plots(root,xyz,colors,selected,fit_selected,score,cameras,images,depths,rgb,count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from source.paintmesh_edgs_init import project
    center=np.median(xyz,axis=0)
    positions=np.array([c["c2w"][:3,3] for c in cameras]); _,_,basis=np.linalg.svd(positions-positions.mean(0),full_matrices=False)
    coords=(xyz-center)@basis.T
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    lo,hi=np.quantile(coords,[.005,.995],axis=0)
    for col,(title,rows) in enumerate((("Original mixed candidates",np.arange(count)),
        ("Selected on fit cameras",fit_selected),("Passed fit + check",selected))):
        rows=rows[::max(1,len(rows)//25000)]
        for ax,pair in ((axes[0,col],(0,1)),(axes[1,col],(0,2))):
            ax.scatter(coords[rows,pair[0]],coords[rows,pair[1]],c=colors[rows],s=1,alpha=.6)
            ax.set(xlim=(lo[pair[0]],hi[pair[0]]),ylim=(lo[pair[1]],hi[pair[1]]),
                title=f"{title}: {len(selected) if col==2 else len(fit_selected) if col==1 else count:,}",
                xlabel=f"Camera-PCA axis {pair[0]}",ylabel=f"Camera-PCA axis {pair[1]}")
            ax.set_aspect("equal",adjustable="box")
    fig.savefig(root/"geometry_comparison.png",dpi=150);plt.close(fig)
    frames=sorted(images)[::max(1,len(images)//6)][:6]
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,frame in zip(axes.flat,frames):
        ax.imshow(images[frame]);uv,_,inside=project(xyz[selected],cameras[frame]);uv=uv[inside]
        ax.scatter(uv[:,0],uv[:,1],s=1,c="#00ff80",alpha=.5)
        ax.set_title(f"Frame {frame:05d}: {len(uv)} selected projections");ax.axis("off")
    fig.savefig(root/"projection_checks.png",dpi=140);plt.close(fig)


if __name__=="__main__":
    main()
