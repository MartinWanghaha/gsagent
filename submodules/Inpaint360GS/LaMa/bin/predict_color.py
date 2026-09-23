# This file is part of inpaint360gs: Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
# Project page: https://dfki-av.github.io/inpaint360gs/
#
# Copyright 2024-2026 Shaoxiang Wang <shaoxiang.wang@dfki.de>
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0

# Modified from codes in LaMa https://github.com/advimman/lama

# This file contains original research code and modified components from the 
# aforementioned projects. It is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

#!/usr/bin/env python3

import logging
import os
import sys
import traceback
import pdb
import io
import tempfile
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
import argparse
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

LOGGER = logging.getLogger(__name__)


def _resolve_paths(args):
    explicit = args.input_dir is not None or args.output_dir is not None
    if explicit and (args.input_dir is None or args.output_dir is None):
        raise ValueError("--input-dir and --output-dir must be provided together")
    if explicit:
        indir = Path(args.input_dir).expanduser().resolve(strict=True)
        outdir = Path(args.output_dir).expanduser().resolve()
    else:
        if not args.data_name:
            raise ValueError(
                "use --input-dir/--output-dir, or provide legacy --data_name"
            )
        indir = Path("./data/color") / args.data_name
        outdir = Path("./output/color") / args.data_name
        indir = indir.resolve(strict=True)
        outdir = outdir.resolve()
    model_path = Path(args.model_path).expanduser().resolve(strict=True)
    if outdir == indir or indir in outdir.parents or outdir in indir.parents:
        raise ValueError("input and output directories must be independent")
    if explicit:
        lama_root = Path(__file__).resolve().parents[1]
        for shared_root in (lama_root / "data", lama_root / "output"):
            if outdir == shared_root or shared_root in outdir.parents:
                raise ValueError(
                    "explicit PaintMesh output cannot use shared LaMa data/output"
                )
    outdir.mkdir(parents=True, exist_ok=True)
    return str(indir), str(outdir), str(model_path)


def _read_index_mask(path, shape):
    with Image.open(path) as image:
        if image.mode not in {"P", "L", "1"}:
            raise ValueError(
                f"mask must use Pillow mode P, L, or 1; got {image.mode!r}: {path}"
            )
        mask = np.asarray(image) != 0
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match RGB shape {shape}: {path}")
    return mask


def _atomic_png(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_composited_prediction(prediction_rgb, image_path, mask_path, output_path):
    with Image.open(image_path) as image:
        original_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if prediction_rgb.shape != original_rgb.shape:
        raise ValueError(
            f"prediction shape {prediction_rgb.shape} does not match input "
            f"{original_rgb.shape}: {image_path}"
        )
    mask = _read_index_mask(mask_path, original_rgb.shape[:2])
    # Preserve every outside-mask byte explicitly.  This makes the completion
    # contract independent of the neural model's behavior in known regions.
    result = original_rgb.copy()
    result[mask] = prediction_rgb[mask]
    _atomic_png(output_path, result)


def _prediction_to_uint8(prediction, label):
    if not np.isfinite(prediction).all():
        raise ValueError(f"LaMa returned non-finite RGB values for {label}")
    return np.clip(prediction * 255, 0, 255).astype(np.uint8)


def main(args):
    from saicinpainting.evaluation.utils import move_to_device
    from saicinpainting.evaluation.refinement import refine_predict
    from saicinpainting.training.data.datasets import make_default_val_dataset
    from saicinpainting.training.trainers import load_checkpoint
    from saicinpainting.utils import register_debug_signal_handlers

    indir, outdir, model_path = _resolve_paths(args)

    default_config = OmegaConf.load('./configs/prediction/default.yaml')
    default_config.dataset.img_suffix = '.png' 
    
    custom_config = OmegaConf.create({
        'refine': True,
        'model': {
            'path': model_path
        },
        'indir': indir,
        'outdir': outdir
    })
    
    predict_config = OmegaConf.merge(default_config, custom_config)

    print(predict_config)  

    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device("cpu")

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'
        
        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)

        if args.recursive_guide:
            # Note: Recursive guide is experimental and may not suit all scenes. Suggest to disable it in default."
            prev_out_fname = None
            for img_i in tqdm.trange(len(dataset)):
                img_fname = dataset.img_filenames[img_i]
                mask_fname = dataset.mask_filenames[img_i]
                
                cur_out_fname = os.path.join(
                    predict_config.outdir,
                    os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                item = dataset[img_i]
                curr_img = item['image']  # Tensor (3, H, W)
                curr_mask = item['mask']   # Tensor (1, H, W)

                if isinstance(curr_img, np.ndarray):
                    curr_img = torch.from_numpy(curr_img).float()
                if isinstance(curr_mask, np.ndarray):
                    curr_mask = torch.from_numpy(curr_mask).float()

                if img_i > 0 and prev_out_fname is not None and os.path.exists(prev_out_fname):
                    prev_res_bgr = cv2.imread(prev_out_fname)
                    prev_res_rgb = cv2.cvtColor(prev_res_bgr, cv2.COLOR_BGR2RGB)
                    
                    prev_res_tensor = torch.from_numpy(prev_res_rgb).permute(2, 0, 1).float() / 255.0
                    
                    curr_c, curr_h, curr_w = curr_img.shape
                    prev_c, prev_h, prev_w = prev_res_tensor.shape

                    if prev_h != curr_h or prev_w != curr_w:
                
                        prev_res_tensor = torch.nn.functional.interpolate(
                            prev_res_tensor.unsqueeze(0), 
                            size=(curr_h, curr_w), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0)

                    combined_img = torch.cat([prev_res_tensor, curr_img], dim=2)
                    prev_mask_blank = torch.zeros_like(curr_mask)
                    combined_mask = torch.cat([prev_mask_blank, curr_mask], dim=2)
                    
                    batch_item = item.copy()
                    batch_item['image'] = combined_img
                    batch_item['mask'] = combined_mask
                    
                    if 'unpad_to_size' in batch_item:
                        h, w = batch_item['unpad_to_size']
                        batch_item['unpad_to_size'] = (curr_h, curr_w * 2)
                else:

                    batch_item = item.copy()
                    batch_item['image'] = curr_img
                    batch_item['mask'] = curr_mask


                batch = default_collate([batch_item])
                
                if predict_config.get('refine', False):
                    cur_res = refine_predict(batch, model, **predict_config.refiner)
                    cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        batch = move_to_device(batch, device)
                        batch['mask'] = (batch['mask'] > 0) * 1
                        batch = model(batch)
                        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                        
                # --- Cropping results and high-resolution local restoration ---
                if img_i > 0:
                    h_low, total_w_low, _ = cur_res.shape
                    cur_res = cur_res[:, (total_w_low // 2):, :]

                with Image.open(img_fname) as original_image:
                    orig_w, orig_h = original_image.size
                inpainted_rgb = _prediction_to_uint8(cur_res, img_fname)
                inpainted_full_rgb = cv2.resize(
                    inpainted_rgb,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                _save_composited_prediction(
                    inpainted_full_rgb, img_fname, mask_fname, cur_out_fname
                )
                
                prev_out_fname = cur_out_fname
        
        else:
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                img_fname = dataset.img_filenames[img_i]

                cur_out_fname = os.path.join(
                    predict_config.outdir,
                    os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5] + out_ext
                )

                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                batch = default_collate([dataset[img_i]])
                if predict_config.get('refine', False):
                    assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                    # image unpadding is taken care of in the refiner, so that output image
                    # is same size as the input image
                    cur_res = refine_predict(batch, model, **predict_config.refiner)
                    cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy() # here is inpainted image！   
                else:
                    with torch.no_grad():
                        batch = move_to_device(batch, device)
                        batch['mask'] = (batch['mask'] > 0) * 1
                        batch = model(batch)                    
                        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                        unpad_to_size = batch.get('unpad_to_size', None)
                        if unpad_to_size is not None:
                            orig_height, orig_width = unpad_to_size
                            cur_res = cur_res[:orig_height, :orig_width]

                cur_res = _prediction_to_uint8(cur_res, img_fname)
                _save_composited_prediction(
                    cur_res, img_fname, mask_fname, cur_out_fname
                )
                # print(f"The currrent output is saved at {cur_out_fname}.")

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LaMa inpainting prediction")
    parser.add_argument(
        "--data_name",
        type=str,
        default=None,
        help="Legacy name under ./data/color and ./output/color",
    )
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="./big-lama")
    parser.add_argument(
        "--recursive_guide",
        "--recursive-guide",
        dest="recursive_guide",
        action='store_true',
        help="Enable recursive guidance",
    )
    args = parser.parse_args()

    main(args)

# python bin/predict_color.py --data_name 360_fruits_virtual
