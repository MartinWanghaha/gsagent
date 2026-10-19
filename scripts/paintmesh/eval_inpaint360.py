"""Evaluate object-free COLMAP views using Inpaint360GS' released metric definitions.

Rendering and metrics use separate processes because EDGS and Inpaint360GS
both vendor top-level packages named scene and utils.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
EDGS = REPO / "submodules/EDGS"
INPAINT = REPO / "submodules/Inpaint360GS"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
METRICS = ("PSNR", "masked_PSNR", "SSIM", "masked_SSIM", "LPIPS", "masked_LPIPS", "FID")
PROTOCOL = {
    "masked_definition": "zero_outside_mask_then_full_image_metric",
    "mask_threshold": 127,
    "lpips": "Inpaint360GS.lpipsPyTorch.vgg.v0.1.inputs_0_1",
    "fid": "pytorch_fid.inception_2048.full_images",
    "crop": 1.0,
    "resize": 1.0,
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_record(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def image_index(directory, reference_only=False):
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if reference_only and not path.name.lower().startswith("test_"):
            continue
        if path.stem in result:
            raise ValueError(f"duplicate image stem: {path.stem} in {directory}")
        result[path.stem] = path
    return result


def match_images(renders, ground_truth, masks):
    pred = image_index(renders)
    gt = image_index(ground_truth)
    mask = image_index(masks, reference_only=True)
    if not pred or set(pred) != set(gt) or set(pred) != set(mask):
        raise ValueError(
            "render/GT/official mask names must match exactly; "
            f"renders={len(pred)}, GT={len(gt)}, masks={len(mask)}, "
            f"missing renders={sorted(set(mask) - set(pred))}, "
            f"missing masks={sorted(set(pred) - set(mask))}"
        )
    if len(pred) < 2:
        raise ValueError("FID requires at least two reference images")
    return [(name, pred[name], gt[name], mask[name]) for name in sorted(pred)]


def masked_array(array, mask):
    """Match the author's apply_mask: black outside, full canvas retained."""
    return array * (mask > 127)[..., None]


def image_tensor(array, device):
    import torch

    # Match torchvision.to_tensor's contiguous CHW layout; the author's PSNR
    # implementation flattens its input using Tensor.view.
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous().float().unsqueeze(0).to(device) / 255


def render_run(args):
    import torch

    sys.path.insert(0, str(EDGS))
    from source.vendor import bootstrap_gaussian_splatting
    from source.renderers import build_renderer
    from render import load_training_config, resolve_iteration, _save_png

    model = args.model_path.resolve(strict=True)
    source = args.source_path.resolve(strict=True)
    output = (args.output_root or model / "evaluation" / f"inpaint360_r{args.resolution}").resolve()
    if output == model or output in model.parents or output == source or source in output.parents or output in source.parents:
        raise ValueError("evaluation output must not overlap source data or replace the model directory")
    iteration = resolve_iteration(model, args.iteration)
    config = load_training_config(model, source_path=source, resolution=args.resolution, images="images")
    saved = load_training_config(model)
    if Path(saved.gs.dataset.source_path).resolve() != source:
        raise ValueError("evaluation source differs from the model's saved source")
    renderer_config = config.gs.get("renderer", {"backend": "native"})
    backend = args.renderer if args.renderer != "auto" else renderer_config.get("backend", "native")
    references = image_index(source / "images", reference_only=True)
    masks = image_index(source / "unseen_mask", reference_only=True)
    if not references or set(references) != set(masks):
        raise ValueError("all test_* images require corresponding official unseen_mask files")
    request = {
        "model": file_record(model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"),
        "config": file_record(model / "config.yaml"),
        "colmap_images": file_record(source / "sparse/0/images.bin"),
        "colmap_cameras": file_record(source / "sparse/0/cameras.bin"),
        "source_path": str(source), "scene": source.name, "iteration": iteration,
        "resolution": args.resolution, "renderer": backend,
        "frames": sorted(references),
        "training_resolution": int(saved.gs.dataset.resolution),
        "training_eval": bool(saved.gs.dataset.eval),
        "training_exclude_test_images": bool(saved.gs.dataset.get("exclude_test_images", False)),
        "reference_images": [file_record(references[name]) for name in sorted(references)],
        "official_masks": [file_record(masks[name]) for name in sorted(masks)],
    }
    manifest_path = output / "render_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("request") != request or not manifest.get("complete"):
            raise ValueError("existing evaluation has different inputs; use a new --output-root")
        match_images(output / "renders", output / "gt", source / "unseen_mask")
        print(f"Reusing reference renders: {output}", flush=True)
    else:
        if args.metrics_only:
            raise ValueError("--metrics-only requires an existing complete evaluation render manifest")
        bootstrap_gaussian_splatting()
        from scene import GaussianModel
        from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
        from scene.dataset_readers import readColmapCameras
        from utils.camera_utils import loadCam

        extrinsics = read_extrinsics_binary(source / "sparse/0/images.bin")
        selected = {key: value for key, value in extrinsics.items()
                    if value.name.lower().startswith("test_")}
        if {Path(value.name).stem for value in selected.values()} != set(references):
            raise ValueError("reference images and COLMAP test_* cameras differ")
        intrinsics = read_intrinsics_binary(source / "sparse/0/cameras.bin")
        infos = readColmapCameras(selected, intrinsics, None, str(source / "images"), "", [])
        infos.sort(key=lambda view: view.image_name)
        if not torch.cuda.is_available():
            raise RuntimeError("rendering requires CUDA")
        gaussians = GaussianModel(int(config.gs.sh_degree))
        renderer = build_renderer({"backend": backend, "clamp_rgb": True})
        gaussians.load_ply(request["model"]["path"], False)
        background = torch.tensor([1., 1., 1.] if config.gs.dataset.white_background else [0., 0., 0.], device="cuda")
        camera_args = SimpleNamespace(resolution=args.resolution, data_device="cpu", train_test_exp=False)
        for folder in ["renders", "gt", "masks"]:
            (output / folder).mkdir(parents=True, exist_ok=True)
        from tqdm import tqdm
        with torch.no_grad():
            for index, info in enumerate(tqdm(infos, desc="Object-free reference views")):
                view = loadCam(camera_args, index, info, 1.0, False, True)
                name = Path(info.image_name).stem + ".png"
                package = renderer.render(view, gaussians, config.gs.pipe, background)
                _save_png(package["render"], output / "renders" / name)
                _save_png(view.original_image, output / "gt" / name)
                with Image.open(masks[Path(info.image_name).stem]) as mask:
                    mask = mask.convert("L").resize((view.image_width, view.image_height), Image.Resampling.NEAREST)
                    mask.save(output / "masks" / name)
                del package, view
        match_images(output / "renders", output / "gt", source / "unseen_mask")
        write_json(manifest_path, {"complete": True, "request": request})
        del gaussians, renderer
        torch.cuda.empty_cache()
    if not args.render_only:
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "metrics",
            "--renders-dir", str(output / "renders"), "--gt-dir", str(output / "gt"),
            "--mask-dir", str(source / "unseen_mask"), "--output", str(output / "results.json"),
            "--render-manifest", str(manifest_path), "--batch-size", str(args.batch_size),
        ], check=True)


def evaluate_metrics(args):
    import torch
    from tqdm import tqdm
    from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3

    sys.path.insert(0, str(INPAINT))
    from lpipsPyTorch.modules.lpips import LPIPS
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    pairs = match_images(args.renders_dir, args.gt_dir, args.mask_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual = LPIPS(net_type="vgg", version="0.1").to(device).eval()
    per_view = {}
    with torch.no_grad():
        for name, render, ground_truth, mask_path in tqdm(pairs, desc="Official metric definitions"):
            with Image.open(render) as image:
                predicted = np.array(image.convert("RGB"))
            with Image.open(ground_truth) as image:
                gt = np.array(image.convert("RGB"))
            if predicted.shape != gt.shape:
                raise ValueError(f"render/GT dimensions differ for {name}")
            with Image.open(mask_path) as image:
                mask = np.array(image.convert("L").resize((predicted.shape[1], predicted.shape[0]), Image.Resampling.NEAREST))
            if not (mask > 127).any():
                raise ValueError(f"official mask is empty for {name}")
            pred, target = image_tensor(predicted, device), image_tensor(gt, device)
            pred_masked = image_tensor(masked_array(predicted, mask), device)
            target_masked = image_tensor(masked_array(gt, mask), device)
            per_view[name] = {
                "PSNR": float(psnr(pred, target).item()),
                "masked_PSNR": float(psnr(pred_masked, target_masked).item()),
                "SSIM": float(ssim(pred, target).item()),
                "masked_SSIM": float(ssim(pred_masked, target_masked).item()),
                "LPIPS": float(perceptual(pred, target).item()),
                "masked_LPIPS": float(perceptual(pred_masked, target_masked).item()),
                "mask_fraction": float((mask > 127).mean()),
            }
    del perceptual
    torch.cuda.empty_cache()
    # Pass the validated file lists explicitly: directory globbing in
    # pytorch_fid otherwise silently omits upper-case .JPG baseline exports.
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
    pred_mean, pred_cov = calculate_activation_statistics(
        [str(pair[1]) for pair in pairs], inception, args.batch_size, 2048, device, num_workers=0)
    gt_mean, gt_cov = calculate_activation_statistics(
        [str(pair[2]) for pair in pairs], inception, args.batch_size, 2048, device, num_workers=0)
    fid = float(calculate_frechet_distance(pred_mean, pred_cov, gt_mean, gt_cov))
    scores = {key: float(np.mean([view[key] for view in per_view.values()])) for key in METRICS[:-1]}
    scores["FID"] = fid
    metadata = json.loads(args.render_manifest.read_text())["request"] if args.render_manifest else {}
    protocol = dict(PROTOCOL, evaluation_resolution=metadata.get("resolution"))
    result = {"complete": True, "protocol": protocol, "inputs": metadata,
              "frame_count": len(pairs), "metrics": scores, "per_view": per_view}
    write_json(args.output, result)
    print(json.dumps(scores, indent=2), flush=True)
    print(f"Saved: {args.output}", flush=True)


def aggregate_results(paths):
    results = [json.loads(Path(path).read_text()) for path in paths]
    if not results or any(not result.get("complete") for result in results):
        raise ValueError("all scene evaluations must be complete")
    if any(result["protocol"] != results[0]["protocol"] for result in results):
        raise ValueError("cannot aggregate different metric definitions or evaluation resolutions")
    for key in ["training_resolution", "training_eval", "training_exclude_test_images"]:
        if any(result["inputs"].get(key) != results[0]["inputs"].get(key) for result in results):
            raise ValueError(f"cannot aggregate different training policies: {key}")
    scenes = [result["inputs"].get("scene", str(path)) for result, path in zip(results, paths)]
    if len(set(scenes)) != len(scenes):
        raise ValueError("aggregate exactly one result per scene and experiment variant")
    count = sum(result["frame_count"] for result in results)
    image_mean = {key: sum(result["metrics"][key] * result["frame_count"] for result in results) / count
                  for key in METRICS[:-1]}
    image_mean["FID"] = float(np.mean([result["metrics"]["FID"] for result in results]))
    scene_mean = {key: float(np.mean([result["metrics"][key] for result in results])) for key in METRICS}
    return {"protocol": results[0]["protocol"], "scene_count": len(results), "frame_count": count,
            "official_code_average": image_mean, "scene_macro_average": scene_mean,
            "per_scene": {scene: result["metrics"] for scene, result in zip(scenes, results)}}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="render only test_* COLMAP cameras, then compute seven metrics")
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--source-path", type=Path, required=True)
    run.add_argument("--resolution", type=int, choices=[1, 2, 4, 8], default=4)
    run.add_argument("--iteration", type=int, default=-1)
    run.add_argument("--renderer", choices=["auto", "pgsr", "native"], default="auto")
    run.add_argument("--output-root", type=Path)
    exclusive = run.add_mutually_exclusive_group()
    exclusive.add_argument("--render-only", action="store_true")
    exclusive.add_argument("--metrics-only", action="store_true")
    run.add_argument("--batch-size", type=int, default=8, help="FID feature extraction batch size")
    metric = sub.add_parser("metrics", help="evaluate an existing matched render/GT/official-mask triplet")
    for name in ["renders-dir", "gt-dir", "mask-dir", "output"]:
        metric.add_argument("--" + name, type=Path, required=True)
    metric.add_argument("--render-manifest", type=Path)
    metric.add_argument("--batch-size", type=int, default=8)
    aggregate = sub.add_parser("aggregate", help="report official-code image means and scene macro means")
    aggregate.add_argument("--results", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if getattr(args, "batch_size", 1) <= 0:
        raise ValueError("--batch-size must be positive")
    if args.command == "run":
        render_run(args)
    elif args.command == "metrics":
        evaluate_metrics(args)
    else:
        result = aggregate_results(args.results)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
