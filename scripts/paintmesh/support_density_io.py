"""Content-addressed optional Stage 4b support and producer-chain validation."""
from pathlib import Path

import numpy as np

from local_geometry_io import (atomic_write, identity, read_receipt,
                               record, verify_record, write_json)

KIND = "paintmesh-support-density"


def validate_density(path):
    return validate_density_payload(read_receipt(path, KIND))


def validate_density_payload(result):
    """Also usable before atomically publishing the completion manifest."""
    if result.get('artifact_id') != identity({k:v for k,v in result.items() if k!='artifact_id'}):
        raise ValueError('density receipt identity mismatch')
    for section in ('inputs','outputs'):
        for value in result.get(section,{}).values():verify_record(value)
    request = read_receipt(result["inputs"]["request"]["path"], KIND + "-request")
    if result["request_id"] != request["artifact_id"]:
        raise ValueError("density request identity mismatch")
    for key, val in request["inputs"].items():
        if result["inputs"].get(key) != val:
            raise ValueError(f"density producer edge mismatch: {key}")
    if result["parameters"] != request["parameters"]:
        raise ValueError("density parameters differ from request")
    # Snapshot every target/render file hash, not only the containing manifest.
    for value in request["dependencies"]:
        verify_record(value)
    if result['parameters'].get('mode') != 'mass_adaptive':
        raise ValueError('unsupported density mode; prepare a new mass_adaptive run')
    from support_mass import ALGORITHM, allocate
    if result['parameters']['config'].get('algorithm') != ALGORITHM:
        raise ValueError('unknown mass sampling algorithm')
    from plyfile import PlyData
    with np.load(result['outputs']['quota']['path'],allow_pickle=False) as quota, \
         np.load(result['outputs']['field']['path'],allow_pickle=False) as field, \
         np.load(result['outputs']['samples']['path'],allow_pickle=False) as samples:
        count=quota['count']; expected=allocate(quota['mass'],result['parameters']['config']['seed'])
        n=len(samples['xyz'])
        if (count.dtype.kind not in 'iu' or samples['quota_id'].dtype.kind not in 'iu' or
                np.any(samples['quota_id']<0) or np.any(samples['quota_id']>=len(count)) or
                not np.array_equal(count,expected) or count.sum()!=n or
                n!=result['report']['initialized_points'] or
                not np.array_equal(np.bincount(samples['quota_id'],minlength=len(count)),count) or
                not np.allclose(quota['mass'],np.maximum(quota['target']-quota['existing'],0)) or
                not np.array_equal(quota['pixel_id'],field['pixel_id']) or
                not np.array_equal(samples['pixel_id'],quota['pixel_id'][samples['quota_id']]) or
                not np.allclose(quota['target'],field['rho']*field['area']) or
                np.any(field['area']<0) or not np.isfinite(field['rho']).all() or
                not np.isfinite(samples['xyz']).all()):
            raise ValueError('mass quota/output conservation failure')
        vertex=PlyData.read(result['outputs']['support']['path'])['vertex'].data
        if len(vertex)!=n or not np.array_equal(np.column_stack([vertex[k] for k in 'xyz']),samples['xyz'].astype(np.float32)):
            raise ValueError('support PLY differs from mass samples')
    return result


def validate_density_rgb(path, rgb_manifest, *, selected_ply=None, local_manifest=None,
                         lama_path=None, camera_path=None, fusion_path=None, config_path=None):
    from local_geometry_io import validate_rgb, validate_local
    density = validate_density(path)
    rgb = validate_rgb(rgb_manifest)
    if verify_record(rgb["inputs"]["density"]) != Path(path).resolve():
        raise ValueError("RGB result used a different density artifact")
    for key, expected in (("support", density["inputs"]["support"]),
                          ("init_support", density["outputs"]["support"])):
        if rgb["inputs"][key] != expected:
            raise ValueError(f"RGB {key} differs from density support")
    for key, expected in (("lama", lama_path), ("camera", camera_path), ("fusion", fusion_path),
                          ("inpaint_config", config_path)):
        if expected is not None and verify_record(density["inputs"][key]) != Path(expected).resolve():
            raise ValueError(f"density {key} differs from publish chain")
    if local_manifest:
        local = validate_local(local_manifest, selected_ply=selected_ply)
        if verify_record(local["inputs"]["rgb_manifest"]) != Path(rgb_manifest).resolve():
            raise ValueError("local result belongs to a different density RGB result")
    elif selected_ply is not None and record(selected_ply)["sha256"] != rgb["outputs"]["ply"]["sha256"]:
        raise ValueError("published RGB differs from density RGB receipt")
    return density


def write_support(path, data):
    from plyfile import PlyData, PlyElement
    vertex = np.empty(len(data["xyz"]), dtype=[(k, "f4") for k in "xyz"] + [(k, "u1") for k in ("red","green","blue")])
    for i, k in enumerate("xyz"):
        vertex[k] = data["xyz"][:,i]
    for i, k in enumerate(("red","green","blue")):
        vertex[k] = np.clip(np.rint(data["rgb"][:,i]), 0, 255).astype(np.uint8)
    atomic_write(path, lambda f: PlyData([PlyElement.describe(vertex, "vertex")]).write(f))


def audit_density_xyz(manifest, xyz, destination, *, retained_prefix=None):
    """Comparable support-domain occupancy audit; not a claim of true geometry."""
    from scipy.spatial import cKDTree
    receipt = validate_density(manifest)
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("nonfinite Gaussian positions in density audit")
    with np.load(receipt["outputs"]["samples"]["path"], allow_pickle=False) as sample:
        if not len(sample['xyz']):
            write_json(destination,dict(total_gaussians=len(xyz),initialized_points=0,note='No new support required.'))
            return
        distance, nearest = cKDTree(sample["xyz"]).query(xyz)
        near = distance <= 2 * sample["spacing"][nearest]
        reverse = cKDTree(xyz).query(sample["xyz"])[0]
        report = dict(total_gaussians=len(xyz), support_near_gaussians=int(near.sum()),
            retained_prefix=int(retained_prefix) if retained_prefix is not None else None,
            sample_coverage=float((reverse <= 2*sample["spacing"]).mean()),
            note="Support-neighborhood occupancy; samples are not independent ground truth.", patches=[])
        for patch in np.unique(sample["patch"]):
            selected = near & (sample["patch"][nearest] == patch)
            pts = xyz[selected]
            stats = dict(patch=int(patch), gaussians=int(selected.sum()))
            if len(pts) > 1:
                nn = cKDTree(pts).query(pts,k=2)[0][:,1]
                stats["nearest_spacing_quantiles"] = np.quantile(nn,[.1,.5,.9]).tolist()
            report["patches"].append(stats)
    write_json(destination, report)
