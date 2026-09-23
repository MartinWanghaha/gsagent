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
        indir = (Path("./data/depth") / args.data_name).resolve(strict=True)
        outdir = (Path("./output/depth") / args.data_name).resolve()
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
        raise ValueError(
            f"mask shape {mask.shape} does not match depth shape {shape}: {path}"
        )
    return mask


def _load_depth(path, label):
    depth = np.load(path, allow_pickle=False)
    if depth.ndim != 2 or min(depth.shape) <= 0:
        raise ValueError(f"{label} must be a non-empty 2D array: {path}")
    if not np.issubdtype(depth.dtype, np.floating):
        depth = depth.astype(np.float32)
    if not np.isfinite(depth).all():
        raise ValueError(f"{label} contains NaN or infinity: {path}")
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_min < 0.0 or depth_max <= depth_min:
        raise ValueError(
            f"{label} must have a finite non-negative, non-zero range; "
            f"got [{depth_min}, {depth_max}]: {path}"
        )
    return depth


def _atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path, array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    _atomic_bytes(path, buffer.getvalue())


def _atomic_cv2(path, array):
    suffix = Path(path).suffix or ".png"
    success, encoded = cv2.imencode(suffix, array)
    if not success:
        raise ValueError(f"failed to encode image: {path}")
    _atomic_bytes(path, encoded.tobytes())


def main(args):
    from saicinpainting.evaluation.utils import move_to_device
    from saicinpainting.evaluation.refinement import refine_predict
    from saicinpainting.training.data.datasets import make_default_val_dataset
    from saicinpainting.training.trainers import load_checkpoint
    from saicinpainting.utils import register_debug_signal_handlers

    indir, outdir, model_path = _resolve_paths(args)

    default_config = OmegaConf.load('./configs/prediction/default.yaml')
    default_config.dataset.img_suffix = '.npy' 
    

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


        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            img_fname = dataset.img_filenames[img_i]
            os.makedirs(os.path.join(predict_config.outdir, "vis"), exist_ok=True)

            cur_out_fname = os.path.join(
                predict_config.outdir, "vis",
                os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5] + out_ext
            )

            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', False):
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()

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

            if "npy" in default_config.dataset.img_suffix:
                depth_original_path = os.path.join(predict_config.indir, "depth_original", os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5]+".npy")    # depth

                depth_original = _load_depth(depth_original_path, "reference depth")
                depth_source = _load_depth(img_fname, "removed depth")
                if depth_original.shape != depth_source.shape:
                    raise ValueError(
                        f"reference depth shape {depth_original.shape} does not match "
                        f"removed depth {depth_source.shape}: {img_fname}"
                    )
                mask = _read_index_mask(mask_fname, depth_source.shape)
                depth_max = float(depth_original.max())
                depth_min = float(depth_original.min())

                if not np.isfinite(cur_res).all():
                    raise ValueError(f"LaMa returned non-finite depth for {img_fname}")
                depth_prediction = cur_res[:, :, 0] * (depth_max - depth_min) + depth_min
                depth_prediction = depth_prediction.astype(depth_source.dtype, copy=False)
                depth_completed = depth_source.copy()
                depth_completed[mask] = depth_prediction[mask]
                if not np.isfinite(depth_completed).all():
                    raise ValueError(f"completed depth is non-finite for {img_fname}")
                depth_npy_path = os.path.join(
                                        predict_config.outdir, 
                                        os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5]+".npy")
                _atomic_npy(depth_npy_path, depth_completed)

                normalized_completed = np.clip(
                    (depth_completed - depth_min) / (depth_max - depth_min),
                    0.0,
                    1.0,
                )
                depth_jet = (normalized_completed * 255.0).astype(np.uint8)
                depth_jet = cv2.applyColorMap(depth_jet, cv2.COLORMAP_JET)  # three channel

                depth_jet_path = os.path.join(
                                        predict_config.outdir, "vis",
                                        os.path.splitext(mask_fname[len(predict_config.indir):])[0][:-5]+"_jet.png")
                _atomic_cv2(depth_jet_path, depth_jet)


            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')

            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            _atomic_cv2(cur_out_fname, cur_res)
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
        help="Legacy name under ./data/depth and ./output/depth",
    )
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="./big-lama")
    args = parser.parse_args()

    main(args)

# python bin/predict_depth.py --data_name 360_doppelherz_virtual
