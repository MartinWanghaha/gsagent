"""Build the optional CUDA backend.

The Python package remains fully functional without running this setup script;
in that case it selects the differentiable PyTorch reference implementation.
"""

from __future__ import annotations

import os

from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


DEFAULT_HEADLESS_CUDA_ARCH_LIST = "7.5;8.0;8.6+PTX"


def configure_cuda_arch_list() -> str | None:
    """Give PyTorch an explicit target when no CUDA device is visible.

    ``torch.utils.cpp_extension`` normally infers architectures from visible
    devices. In headless build nodes and containers that produces an empty
    list and, on several PyTorch releases, an ``IndexError``. An explicit
    ``TORCH_CUDA_ARCH_LIST`` always wins; otherwise visible-GPU builds retain
    PyTorch's native detection and headless builds use a documented portable
    default covering Turing, Ampere, and forward-compatible PTX from sm_86.
    """

    configured = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if configured:
        return configured
    if torch.cuda.is_available():
        return None
    configured = os.environ.get(
        "SGW_HEADLESS_CUDA_ARCH_LIST",
        DEFAULT_HEADLESS_CUDA_ARCH_LIST,
    ).strip()
    if not configured:
        raise RuntimeError(
            "no CUDA device is visible and SGW_HEADLESS_CUDA_ARCH_LIST is empty; "
            "set TORCH_CUDA_ARCH_LIST to the target compute capability"
        )
    os.environ["TORCH_CUDA_ARCH_LIST"] = configured
    print(
        "No CUDA device is visible; building for "
        f"TORCH_CUDA_ARCH_LIST={configured}. Override it for the deployment GPU if known.",
        flush=True,
    )
    return configured


configure_cuda_arch_list()

setup(
    name="diff-semantic-gaussian-rasterization",
    version="0.1.0",
    description="Joint RGB, semantic, depth, alpha and normal Gaussian rasterizer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
    ext_modules=[
        CUDAExtension(
            name="diff_semantic_gaussian_rasterization._C",
            sources=[
                "ext.cpp",
                "cuda_rasterizer/rasterize.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    zip_safe=False,
)
