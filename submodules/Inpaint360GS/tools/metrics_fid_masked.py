# # Modified metrics_fid_masked.py with masked / non-masked region evaluation

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_fid import fid_score
import numpy as np

def center_crop_array(arr, crop_ratio=0.8):
    h, w = arr.shape[:2]
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return arr[top:top+crop_h, left:left+crop_w]

def center_crop_tensor(img_tensor, crop_ratio=0.8):
    _, _, h, w = img_tensor.shape
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return img_tensor[:, :, top:top+crop_h, left:left+crop_w]

def readImages(renders_dir, gt_dir, crop_ratio=1.0, resize_ratio = 1.0):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname).convert("RGB")
        gt = Image.open(gt_dir / fname).convert("RGB")

        new_size = (int(render.width * resize_ratio), int(render.height * resize_ratio))
        render = render.resize(new_size, Image.Resampling.BICUBIC)
        gt = gt.resize(new_size, Image.Resampling.BICUBIC)

        render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()

        if crop_ratio < 1.0:
            render_tensor = center_crop_tensor(render_tensor, crop_ratio)
            gt_tensor = center_crop_tensor(gt_tensor, crop_ratio)

        renders.append(render_tensor)
        gts.append(gt_tensor)
        image_names.append(fname)
    return renders, gts, image_names

def apply_mask(img_tensor, mask_array):
    mask = torch.from_numpy((mask_array > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(img_tensor.device)
    return img_tensor * mask

def evaluate(model_paths, crop_ratio=1.0, resize=1.0):
    
    for scene_dir in model_paths:
        if not Path(scene_dir).exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    full_dict = {}
    per_view_dict = {}

    # Global metrics list (masked / non-masked / full)
    global_ssims_masked, global_psnrs_masked, global_lpipss_masked = [], [], []
    global_ssims_nonmasked, global_psnrs_nonmasked, global_lpipss_nonmasked = [], [], []
    global_ssims_full, global_psnrs_full, global_lpipss_full = [], [], []
    global_fid_full = []

    for scene_dir in model_paths:
        dataset_name = scene_dir.split("/")[-1]
        inpaint_dir = os.path.join(scene_dir, "inpaint")
        waiting_eval_list = []
       
        for method in os.listdir(inpaint_dir):
            for iter in os.listdir(os.path.join(inpaint_dir, method)):
                waiting_eval_list.append(os.path.join(inpaint_dir, method, iter))

        print(f"\n    Waiting_eval_list is {waiting_eval_list}")
        for test_dir in waiting_eval_list:

            print("\nScene:", scene_dir)
            test_dir_str = str(test_dir)
            full_dict[test_dir_str] = {}
            per_view_dict[test_dir_str] = {}

            test_dir = Path(test_dir) 
            mask_dir = Path("data") / "inpaint360" / dataset_name / "unseen_mask"

            full_dict[test_dir_str] = {}
            per_view_dict[test_dir_str] = {}
            
            gt_dir = test_dir / "gt"
            renders_dir = test_dir / "renders"

            renders, gts, image_names = readImages(renders_dir, gt_dir, crop_ratio, resize)

            ssims_masked, psnrs_masked, lpipss_masked = [], [], []
            ssims_nonmasked, psnrs_nonmasked, lpipss_nonmasked = [], [], []
            ssims_full, psnrs_full, lpipss_full = [], [], []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                fname = image_names[idx]

                possible_exts = [".png", ".jpg", ".jpeg", ".JPG"]
                mask_path = None
                for ext in possible_exts:
                    candidate = mask_dir / Path(fname).with_suffix(ext).name
                    if candidate.exists():
                        mask_path = candidate
                        break

                ssim_full = ssim(renders[idx], gts[idx])
                psnr_full = psnr(renders[idx], gts[idx])
                lpips_full = lpips(renders[idx], gts[idx], net_type='vgg')

                ssims_full.append(ssim_full)
                psnrs_full.append(psnr_full)
                lpipss_full.append(lpips_full)

                if mask_path is not None:
                    mask = Image.open(mask_path).convert("L")
                    target_width = int(renders[idx].shape[-1])
                    target_height = int(renders[idx].shape[-2])
                    mask = mask.resize((target_width, target_height), Image.Resampling.NEAREST)

                    mask = np.array(mask)
                    if crop_ratio < 1.0:
                        mask = center_crop_array(mask, crop_ratio)

                    render_masked = apply_mask(renders[idx], mask)
                    gt_masked = apply_mask(gts[idx], mask)

                    render_nonmasked = apply_mask(renders[idx], 255 - mask)
                    gt_nonmasked = apply_mask(gts[idx], 255 - mask)

                    ssims_masked.append(ssim(render_masked, gt_masked))
                    psnrs_masked.append(psnr(render_masked, gt_masked))
                    lpipss_masked.append(lpips(render_masked, gt_masked, net_type='vgg'))

                    ssims_nonmasked.append(ssim(render_nonmasked, gt_nonmasked))
                    psnrs_nonmasked.append(psnr(render_nonmasked, gt_nonmasked))
                    lpipss_nonmasked.append(lpips(render_nonmasked, gt_nonmasked, net_type='vgg'))
                else:
                    print(f"  \u26a0 No mask for {fname} with any known extension, skipping masked evaluation")

            torch.cuda.empty_cache()

            def safe_mean(tlist):
                return float(torch.tensor(tlist).mean().item()) if len(tlist) > 0 else float('nan')

            full_dict[test_dir_str].update({
                "SSIM_masked": safe_mean(ssims_masked),
                "PSNR_masked": safe_mean(psnrs_masked),
                "LPIPS_masked": safe_mean(lpipss_masked),
                "SSIM_nonmasked": safe_mean(ssims_nonmasked),
                "PSNR_nonmasked": safe_mean(psnrs_nonmasked),
                "LPIPS_nonmasked": safe_mean(lpipss_nonmasked),
                "SSIM_full": safe_mean(ssims_full),
                "PSNR_full": safe_mean(psnrs_full),
                "LPIPS_full": safe_mean(lpipss_full)
            })

            print("    SSIM_masked:", full_dict[test_dir_str]["SSIM_masked"])
            print("    PSNR_masked:", full_dict[test_dir_str]["PSNR_masked"])
            print("    LPIPS_masked:", full_dict[test_dir_str]["LPIPS_masked"])
            print("    SSIM_nonmasked:", full_dict[test_dir_str]["SSIM_nonmasked"])
            print("    PSNR_nonmasked:", full_dict[test_dir_str]["PSNR_nonmasked"])
            print("    LPIPS_nonmasked:", full_dict[test_dir_str]["LPIPS_nonmasked"])
            print("    SSIM_full:", full_dict[test_dir_str]["SSIM_full"])
            print("    PSNR_full:", full_dict[test_dir_str]["PSNR_full"])
            print("    LPIPS_full:", full_dict[test_dir_str]["LPIPS_full"])


            fid_value = fid_score.calculate_fid_given_paths(
                [str(renders_dir), str(gt_dir)],
                batch_size=50,
                device=torch.device("cuda:0"),
                dims=2048
            )
            full_dict[test_dir_str]["FID"] = fid_value

            print("    FID:", fid_value)

            # full image evaluation
            global_ssims_masked.extend(ssims_masked)
            global_psnrs_masked.extend(psnrs_masked)
            global_lpipss_masked.extend(lpipss_masked)

            global_ssims_nonmasked.extend(ssims_nonmasked)
            global_psnrs_nonmasked.extend(psnrs_nonmasked)
            global_lpipss_nonmasked.extend(lpipss_nonmasked)

            global_ssims_full.extend(ssims_full)
            global_psnrs_full.extend(psnrs_full)
            global_lpipss_full.extend(lpipss_full)
            global_fid_full.append(fid_value)

            target_dir = "/".join(scene_dir.split("/")[:3])
            with open(f"{target_dir}/inpaint_evaluation_results.json", 'w') as fp:
                json.dump(full_dict[test_dir_str], fp, indent=2)

    print("\n==================== Overall Average Metrics ====================")
    print("  SSIM (masked):      ", float(torch.tensor(global_ssims_masked).mean()) if global_ssims_masked else "N/A")
    print("  PSNR (masked):      ", float(torch.tensor(global_psnrs_masked).mean()) if global_psnrs_masked else "N/A")
    print("  LPIPS (masked):     ", float(torch.tensor(global_lpipss_masked).mean()) if global_lpipss_masked else "N/A")
    print("  SSIM (non-masked):  ", float(torch.tensor(global_ssims_nonmasked).mean()) if global_ssims_nonmasked else "N/A")
    print("  PSNR (non-masked):  ", float(torch.tensor(global_psnrs_nonmasked).mean()) if global_psnrs_nonmasked else "N/A")
    print("  LPIPS (non-masked): ", float(torch.tensor(global_lpipss_nonmasked).mean()) if global_lpipss_nonmasked else "N/A")
    print("  SSIM (full):        ", float(torch.tensor(global_ssims_full).mean()) if global_ssims_full else "N/A")
    print("  PSNR (full):        ", float(torch.tensor(global_psnrs_full).mean()) if global_psnrs_full else "N/A")
    print("  LPIPS (full):       ", float(torch.tensor(global_lpipss_full).mean()) if global_lpipss_full else "N/A")
    print("  FID_full:           ", float(torch.tensor(global_fid_full).mean()) if global_fid_full else "N/A")

    all_results = {
    "per_scene_results": full_dict,
    "overall_average": {
        "SSIM_masked":      float(torch.tensor(global_ssims_masked).mean()) if global_ssims_masked else None,
        "PSNR_masked":      float(torch.tensor(global_psnrs_masked).mean()) if global_psnrs_masked else None,
        "LPIPS_masked":     float(torch.tensor(global_lpipss_masked).mean()) if global_lpipss_masked else None,
        "SSIM_nonmasked":   float(torch.tensor(global_ssims_nonmasked).mean()) if global_ssims_nonmasked else None,
        "PSNR_nonmasked":   float(torch.tensor(global_psnrs_nonmasked).mean()) if global_psnrs_nonmasked else None,
        "LPIPS_nonmasked":  float(torch.tensor(global_lpipss_nonmasked).mean()) if global_lpipss_nonmasked else None,
        "SSIM_full":        float(torch.tensor(global_ssims_full).mean()) if global_ssims_full else None,
        "PSNR_full":        float(torch.tensor(global_psnrs_full).mean()) if global_psnrs_full else None,
        "LPIPS_full":       float(torch.tensor(global_lpipss_full).mean()) if global_lpipss_full else None,
        "FID_full":         float(np.mean(global_fid_full)) if global_fid_full else None
        }
    }

    mean_result_path = "output/inpaint360/all_scene_evaluation_results.json"
    with open(mean_result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Saved full results to {mean_result_path}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Evaluation script with masked/non-masked regions + resize + optional center crop")
    parser.add_argument('--model_paths', '-m', nargs="+", type=str)
    parser.add_argument('--crop', type=float, default=1.0, help='Center crop ratio (e.g., 0.8 for 80% of center)')
    parser.add_argument('--resize', type=float, default=1.0, help='resize image for evalution')
    args = parser.parse_args()

    evaluate(args.model_paths, crop_ratio=args.crop, resize=args.resize)