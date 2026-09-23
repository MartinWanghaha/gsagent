#!/usr/bin/env bash
set -euo pipefail

# Install EDGS and the PGSR plane rasterizer into one Conda environment.
# Override the default with: EDGS_CONDA_ENV=my_env bash install.sh
readonly ENV_NAME="${EDGS_CONDA_ENV:-paintmesh}"
readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PGSR_COMMIT="de24f1a38b350387e8d8fe381b2cd70c1ae946e7"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found; install Miniconda/Anaconda first." >&2
    exit 1
fi

if ! conda run -n "${ENV_NAME}" python -c "pass" >/dev/null 2>&1; then
    echo "Creating Conda environment '${ENV_NAME}' with PyTorch/CUDA 11.8."
    conda create -y -n "${ENV_NAME}" python=3.10 pip
fi

if ! conda run -n "${ENV_NAME}" python -c "import torch" >/dev/null 2>&1; then
    echo "Installing PyTorch/CUDA 11.8 into '${ENV_NAME}'."
    conda install -y -n "${ENV_NAME}" \
        pytorch=2.0.0 torchvision=0.15.1 torchaudio=2.0.1 pytorch-cuda=11.8 \
        -c pytorch -c nvidia
    conda install -y -n "${ENV_NAME}" nvidia/label/cuda-11.8.0::cuda-toolkit
else
    echo "Using existing Conda environment '${ENV_NAME}'; PyTorch will not be replaced."
fi

run_in_env() {
    conda run --no-capture-output -n "${ENV_NAME}" "$@"
}

cd "${REPO_ROOT}"
git submodule sync --recursive
git submodule update --init --recursive
if ! git -C submodules/PGSR rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "PGSR submodule is missing; initialize submodules before installing." >&2
    exit 1
fi
ACTUAL_PGSR_COMMIT="$(git -C submodules/PGSR rev-parse HEAD)"
if [[ "${ACTUAL_PGSR_COMMIT}" != "${PGSR_COMMIT}" ]]; then
    echo "Unexpected PGSR commit: ${ACTUAL_PGSR_COMMIT}; expected ${PGSR_COMMIT}." >&2
    exit 1
fi

# Prefer the toolkit already installed in paintmesh. Otherwise PyTorch's
# extension builder falls back to CUDA_HOME or the system nvcc on PATH.
ENV_PREFIX="$(run_in_env python -c 'import sys; print(sys.prefix)')"
if [[ -x "${ENV_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="${ENV_PREFIX}"
fi

TORCH_CUDA="$(run_in_env python -c 'import torch; print(torch.version.cuda or "")')"
if [[ -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
    NVCC_BIN="${CUDA_HOME}/bin/nvcc"
else
    NVCC_BIN="$(command -v nvcc || true)"
fi
if [[ -z "${NVCC_BIN}" ]]; then
    echo "nvcc was not found; install a CUDA toolkit matching PyTorch ${TORCH_CUDA}." >&2
    exit 1
fi
NVCC_CUDA="$("${NVCC_BIN}" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n 1)"
if [[ "${TORCH_CUDA}" != "${NVCC_CUDA}" ]]; then
    echo "CUDA ABI mismatch: PyTorch=${TORCH_CUDA}, nvcc=${NVCC_CUDA} (${NVCC_BIN})." >&2
    exit 1
fi

run_in_env python -c '
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("EDGS training requires an available CUDA GPU")
print("GPU:", torch.cuda.get_device_name(0))
'

run_in_env python -m pip install \
    wandb hydra-core tqdm torchmetrics lpips matplotlib rich plyfile \
    imageio imageio-ffmpeg pycolmap

# Compile all CUDA extensions from temporary source copies. This binds them to
# the selected PyTorch/CUDA ABI without leaving build products in submodules.
# PGSR's colliding scene/model packages and its own simple-knn are not installed.
EXTENSION_BUILD_ROOT="$(mktemp -d)"
cleanup_extension_build() {
    case "${EXTENSION_BUILD_ROOT}" in
        /tmp/*) rm -rf -- "${EXTENSION_BUILD_ROOT}" ;;
    esac
}
trap cleanup_extension_build EXIT

install_cuda_extension() {
    local source_path="$1"
    local build_name="$2"
    cp -a "${source_path}" "${EXTENSION_BUILD_ROOT}/${build_name}"
    run_in_env python -m pip install \
        --force-reinstall --no-deps --no-build-isolation \
        "${EXTENSION_BUILD_ROOT}/${build_name}"
}

install_cuda_extension \
    submodules/gaussian-splatting/submodules/diff-gaussian-rasterization \
    diff-gaussian-rasterization
install_cuda_extension \
    submodules/gaussian-splatting/submodules/simple-knn \
    simple-knn
install_cuda_extension \
    submodules/PGSR/submodules/diff-plane-rasterization \
    diff-plane-rasterization

# Resolve RoMa under the dependency line already validated for EDGS and the
# shared paintmesh environment, instead of allowing unconstrained upgrades.
run_in_env python -m pip install -e submodules/RoMa \
    "numpy==1.26.4" "pydantic==1.10.13" \
    "albumentations==1.3.1" "opencv-python-headless==4.8.1.78"

# Preserve the visualization/notebook dependencies provided by EDGS's original
# installer only when explicitly requested, keeping the training install small.
if [[ "${EDGS_INSTALL_DEMO:-0}" == "1" ]]; then
    run_in_env python -m pip install \
        "gradio==3.39.0" plotly scikit-learn "moviepy==2.1.1" \
        ffmpeg open3d jupyter
fi

run_in_env python -m pip check

echo "Installation complete. Activate it with: conda activate ${ENV_NAME}"
