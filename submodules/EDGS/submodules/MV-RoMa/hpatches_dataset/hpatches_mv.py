"""
Simply load images from a folder or nested folders (does not have any split).
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
import cv2
from pdb import set_trace as bb
from contextlib import contextmanager

from pathlib import Path
from typing import Optional, Tuple

import random
from .image_process import ImagePreprocessor

logger = logging.getLogger(__name__)

def plot_image_grid(
    imgs,
    titles=None,
    cmaps="gray",
    dpi=100,
    pad=0.5,
    fig=None,
    adaptive=True,
    figs=2.0,
    return_fig=False,
    set_lim=False,
):
    """Plot a grid of images.
    Args:
        imgs: a list of lists of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    nr, n = len(imgs), len(imgs[0])
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs[0]]  # W / H
    else:
        ratios = [4 / 3] * n

    figsize = [sum(ratios) * figs, nr * figs]
    if fig is None:
        fig, axs = plt.subplots(
            nr, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
        )
    else:
        axs = fig.subplots(nr, n, gridspec_kw={"width_ratios": ratios})
        fig.figure.set_size_inches(figsize)
    if nr == 1:
        axs = [axs]

    for j in range(nr):
        for i in range(n):
            ax = axs[j][i]
            ax.imshow(imgs[j][i], cmap=plt.get_cmap(cmaps[i]))
            ax.set_axis_off()
            if set_lim:
                ax.set_xlim([0, imgs[j][i].shape[1]])
                ax.set_ylim([imgs[j][i].shape[0], 0])
            if titles:
                ax.set_title(titles[j][i])
    if isinstance(fig, plt.Figure):
        fig.tight_layout(pad=pad)
    if return_fig:
        return fig, axs
    else:
        return axs


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_random_state(with_cuda):
    pth_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    if torch.cuda.is_available() and with_cuda:
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    return pth_state, np_state, py_state, cuda_state


def set_random_state(state):
    pth_state, np_state, py_state, cuda_state = state
    torch.set_rng_state(pth_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    if (
        cuda_state is not None
        and torch.cuda.is_available()
        and len(cuda_state) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_state)

@contextmanager
def fork_rng(seed=None, with_cuda=True):
    state = get_random_state(with_cuda)
    if seed is not None:
        set_seed(seed)
    try:
        yield
    finally:
        set_random_state(state)


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def load_image(path: Path, grayscale=False) -> torch.Tensor:
    image = read_image(path, grayscale=grayscale)
    orig_size = image.shape[:2]
    cv2.imwrite(str(path).replace('.ppm', '.jpg'), image[...,::-1])
    return numpy_image_to_torch(image), str(path).replace('.ppm', '.jpg'), orig_size


def read_homography(path):
    with open(path) as f:
        result = []
        for line in f.readlines():
            while "  " in line:  # Remove double spaces
                line = line.replace("  ", " ")
            line = line.replace(" \n", "").replace("\n", "")
            # Split and discard empty strings
            elements = list(filter(lambda s: s, line.split(" ")))
            if elements:
                result.append(elements)
        return np.array(result).astype(float)


class HPatches(torch.utils.data.Dataset):
    default_conf = {
        "data_dir": "hpatches-sequences-release",
        "subset": None,
        "ignore_large_images": True,
        "grayscale": False,
    }

    # Large images that were ignored in previous papers
    ignored_scenes = (
        "i_contruction",
        "i_crownnight",
        "i_dc",
        "i_pencils",
        "i_whitebuilding",
        "v_artisans",
        "v_astronautis",
        "v_talent",
        "v_soldiers"
    )
    url = "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz"

    def __init__(self, data_root: str = "/cephfs/jongmin/cvpr2025_matcher/hpatches/hpatches-sequences-release"):

        self.root = Path(data_root)
        self.sequences = sorted([x.name for x in self.root.iterdir()])
        if not self.sequences:
            raise ValueError("No image found!")

        self.image_process = ImagePreprocessor()
        self.items = []  # (seq, q_idx, is_illu)
        for seq in self.sequences:
            if seq in self.ignored_scenes:
                continue
            self.items.append([(seq, 1, seq[0] == "i")])
            for i in range(2, 7):
                self.items[-1].append((seq, i, seq[0] == "i"))


    def _read_image(self, seq: str, idx: int) -> dict:
        img, path, orig_size = load_image(self.root / seq / f"{idx}.ppm", False)
        return img, path, orig_size

    def __getitem__(self, idx):
        scene_info = self.items[idx]
        seq, _, is_illu = scene_info[0]
        data0, path0, orig_size = self._read_image(seq, 1)

        q_idxs = [2,3,4,5,6]

        q_paths = []
        H_set = []

        orig_sizes = []
        orig_sizes.append(orig_size)

        for q_idx in q_idxs:
            data_qidx, path_qidx, orig_size = self._read_image(seq, q_idx)
            orig_sizes.append(orig_size)        
            H = read_homography(self.root / seq / f"H_1_{q_idx}")
            q_paths.append(path_qidx)
            H_set.append(H)

        return {
            "H_0to1": H.astype(np.float32),
            "scene": seq,
            "idx": idx,
            "is_illu": is_illu,
            "name": f"{seq}",
            # "view0": data0,
            # "view1": data1,
            "path0": path0,
            "path_queries": q_paths,
            "H_set": H_set,
            "orig_sizes": orig_sizes,
        }

    def __len__(self):
        return len(self.items)


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = HPatches()
    print("The dataset has %d elements.", len(dataset))

    with fork_rng(seed=42):
        images = []
        print(len(dataset))
        for idx in range(107):
            data = dataset[idx]

    bb()
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)