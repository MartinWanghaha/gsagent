#!/usr/bin/env bash

set -Eeuo pipefail

# Inpaint360GS 2D RGB/depth completion and 3D Gaussian inpainting pipeline.
#
# Usage:
#   bash scripts/inpaint360gs/run_inpaint.sh \
#       [dataset_name] [scene] [resolution] [start_stage]
#
# Examples:
#   bash scripts/inpaint360gs/run_inpaint.sh inpaint360 doppelherz 2 1
#   bash scripts/inpaint360gs/run_inpaint.sh "mip-nerf/360_v2" kitchen 8 1
#   bash scripts/inpaint360gs/run_inpaint.sh "mip-nerf/360_v2" kitchen 8 3
#
# start_stage: 1=prepare LaMa inputs, 2=LaMa RGB/depth completion,
#              3=import LaMa outputs, 4=RGB-D point-cloud fusion,
#              5=3DGS inpainting optimization, 6=evaluation.
#
# The main environment is normally `paintmesh`. LaMa inference is launched in
# the conda environment selected by LAMA_ENV (default: `lama`).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INPAINT360GS_ROOT="${REPO_ROOT}/submodules/Inpaint360GS"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_BIN="${CONDA_BIN:-conda}"
LAMA_ENV="${LAMA_ENV:-lama}"
LAMA_ENV_PREFIX="${LAMA_ENV_PREFIX:-}"

DATASET_NAME="${1:-${DATASET_NAME:-inpaint360}}"
SCENE="${2:-${SCENE:-doppelherz}}"
RESOLUTION="${3:-${RESOLUTION:-2}}"
START_STAGE="${4:-${START_STAGE:-1}}"

usage() {
    sed -n '5,21p' "${BASH_SOURCE[0]}"
}

if [[ "${DATASET_NAME}" == "-h" || "${DATASET_NAME}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# > 4 )); then
    usage >&2
    exit 2
fi

case "${RESOLUTION}" in
    1|2|4|8)
        ;;
    *)
        echo "Error: resolution must be one of 1, 2, 4, or 8; got '${RESOLUTION}'." >&2
        exit 2
        ;;
esac

case "${START_STAGE}" in
    1|2|3|4|5|6)
        ;;
    *)
        echo "Error: start_stage must be an integer from 1 to 6; got '${START_STAGE}'." >&2
        exit 2
        ;;
esac

SCENE_ROOT="${DATA_ROOT}/${DATASET_NAME}/${SCENE}"
MODEL_ROOT="${OUTPUT_ROOT}/${DATASET_NAME}/${SCENE}"
LAMA_ROOT="${INPAINT360GS_ROOT}/LaMa"
LAMA_MODEL_SOURCE="${CKPT_ROOT}/big-lama"
LAMA_MODEL_TARGET="${LAMA_ROOT}/big-lama"
LAMA_DATA_NAME="360_${SCENE}_virtual"
TRACKING_MASK_ROOT="${INPAINT360GS_ROOT}/Segment-and-Track-Anything/tracking_results/images/images_masks"
REMOVAL_CONFIG="${INPAINT360GS_ROOT}/config/object_removal/${DATASET_NAME}/${SCENE}.json"
INPAINT_CONFIG="${INPAINT360GS_ROOT}/config/object_inpaint/${DATASET_NAME}/${SCENE}.json"
DISTILL_CONFIG="${INPAINT360GS_ROOT}/config/object_distill/train_distill.json"

fail() {
    echo "Error: $*" >&2
    exit 1
}

require_dir() {
    [[ -d "$1" ]] || fail "required directory does not exist: $1"
}

require_file() {
    [[ -f "$1" ]] || fail "required file does not exist: $1"
}

require_matching_file() {
    local pattern="$1"
    local description="$2"

    compgen -G "${pattern}" >/dev/null || fail "${description} was not found: ${pattern}"
}

require_file_count() {
    local pattern="$1"
    local expected="$2"
    local description="$3"
    local matches=()

    mapfile -t matches < <(compgen -G "${pattern}" || true)
    (( ${#matches[@]} == expected )) || \
        fail "${description}: expected ${expected} files, found ${#matches[@]} (${pattern})"
}

link_lama_model() {
    require_dir "${LAMA_MODEL_SOURCE}"
    require_file "${LAMA_MODEL_SOURCE}/config.yaml"
    require_file "${LAMA_MODEL_SOURCE}/models/best.ckpt"

    if [[ ! -e "${LAMA_MODEL_TARGET}" && ! -L "${LAMA_MODEL_TARGET}" ]]; then
        ln -s "${LAMA_MODEL_SOURCE}" "${LAMA_MODEL_TARGET}"
    fi

    require_file "${LAMA_MODEL_TARGET}/config.yaml"
    require_file "${LAMA_MODEL_TARGET}/models/best.ckpt"
}

run_lama_python() {
    env \
        LD_LIBRARY_PATH="${LAMA_ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        TORCH_HOME="${LAMA_ROOT}" \
        PYTHONPATH="${LAMA_ROOT}" \
        "${CONDA_BIN}" run --no-capture-output -n "${LAMA_ENV}" \
        python "$@"
}

trap 'echo "Error: Inpaint360GS run_inpaint failed at line ${LINENO}." >&2' ERR

require_dir "${INPAINT360GS_ROOT}"
require_dir "${DATA_ROOT}"
require_dir "${SCENE_ROOT}"
require_dir "${MODEL_ROOT}"
require_dir "${LAMA_ROOT}"
require_dir "${CKPT_ROOT}"
require_file "${REMOVAL_CONFIG}"
require_file "${INPAINT_CONFIG}"
require_file "${DISTILL_CONFIG}"

if (( START_STAGE <= 2 )); then
    if [[ -z "${LAMA_ENV_PREFIX}" ]]; then
        LAMA_ENV_PREFIX="$("${CONDA_BIN}" run --no-capture-output -n "${LAMA_ENV}" \
            python -c 'import sys; print(sys.prefix)')"
    fi
    require_dir "${LAMA_ENV_PREFIX}/lib"
fi

DISTILL_ITERATION="$("${PYTHON_BIN}" -c \
    'import json, sys; print(int(json.load(open(sys.argv[1]))["iterations"]))' \
    "${DISTILL_CONFIG}")"

FINAL_ITERATION="$("${PYTHON_BIN}" -c \
    'import json, sys; print(int(json.load(open(sys.argv[1]))["finetune_iteration"]))' \
    "${INPAINT_CONFIG}")"

export PYTHONPATH="${INPAINT360GS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Inpaint360GS root : ${INPAINT360GS_ROOT}"
echo "Dataset           : ${SCENE_ROOT}"
echo "Checkpoints       : ${CKPT_ROOT}"
echo "Output            : ${MODEL_ROOT}"
echo "Resolution        : ${RESOLUTION}"
echo "LaMa environment  : ${LAMA_ENV}"
echo "Distill iteration : ${DISTILL_ITERATION}"
echo "Final iteration   : ${FINAL_ITERATION}"
echo "Start stage       : ${START_STAGE}"

cd "${INPAINT360GS_ROOT}"

if (( START_STAGE <= 1 )); then
    echo "[1/6] Preparing virtual RGB, depth, and masks for LaMa"
    require_file_count "${TRACKING_MASK_ROOT}/*.png" 30 "tracked virtual-view masks"
    require_file_count \
        "${MODEL_ROOT}/virtual/ours_${DISTILL_ITERATION}/depth/*.npy" 30 \
        "full-scene virtual depths"
    require_file_count \
        "${MODEL_ROOT}/virtual/ours_object_removal/iteration_${DISTILL_ITERATION}/renders/*.png" 30 \
        "object-removal virtual renders"
    require_file_count \
        "${MODEL_ROOT}/virtual/ours_object_removal/iteration_${DISTILL_ITERATION}/depth/*.npy" 30 \
        "object-removal virtual depths"

    "${PYTHON_BIN}" tools/prepare_lama_data.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${MODEL_ROOT}" \
        --resolution "${RESOLUTION}" \
        --inpaint2lama
else
    echo "[1/6] Skipping LaMa input preparation"
fi

if (( START_STAGE <= 2 )); then
    echo "[2/6] Completing virtual RGB and depth with LaMa"
    require_file_count \
        "${LAMA_ROOT}/data/color/${LAMA_DATA_NAME}/*_mask.png" 30 \
        "LaMa color masks"
    require_file_count \
        "${LAMA_ROOT}/data/depth/${LAMA_DATA_NAME}/*_mask.png" 30 \
        "LaMa depth masks"
    require_file "${LAMA_ROOT}/saicinpainting/training/data/datasets.py"
    link_lama_model

    run_lama_python -c \
        'from saicinpainting.training.data.datasets import make_default_val_dataset; from saicinpainting.training.trainers import load_checkpoint; print("LaMa imports: OK")'

    cd "${LAMA_ROOT}"
    run_lama_python bin/predict_color.py --data_name "${LAMA_DATA_NAME}"
    run_lama_python bin/predict_depth.py --data_name "${LAMA_DATA_NAME}"
    cd "${INPAINT360GS_ROOT}"
else
    echo "[2/6] Skipping LaMa RGB/depth completion"
fi

if (( START_STAGE <= 3 )); then
    echo "[3/6] Importing LaMa outputs into the scene"
    require_file_count \
        "${LAMA_ROOT}/output/color/${LAMA_DATA_NAME}/*.png" 30 \
        "LaMa completed RGB images"
    require_file_count \
        "${LAMA_ROOT}/output/depth/${LAMA_DATA_NAME}/*.npy" 30 \
        "LaMa completed depth maps"

    "${PYTHON_BIN}" tools/prepare_lama_data.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${MODEL_ROOT}" \
        --resolution "${RESOLUTION}"
else
    echo "[3/6] Skipping LaMa output import"
fi

if (( START_STAGE <= 4 )); then
    echo "[4/6] Fusing completed RGB-D views into point clouds"
    require_file_count \
        "${SCENE_ROOT}/inpaint_2d_unseen_mask_virtual/*.png" 30 \
        "scene inpainting masks"
    require_file_count \
        "${MODEL_ROOT}/virtual/ours_object_removal/iteration_${DISTILL_ITERATION}/depth_completed/*.npy" 30 \
        "scene completed depths"
    require_matching_file \
        "${SCENE_ROOT}/images_inpaint_unseen_virtual/*" \
        "completed virtual RGB images"

    "${PYTHON_BIN}" edit_object_removal_plyfusion.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${MODEL_ROOT}" \
        --resolution "${RESOLUTION}" \
        --config_file "${REMOVAL_CONFIG}"
else
    echo "[4/6] Skipping RGB-D point-cloud fusion"
fi

if (( START_STAGE <= 5 )); then
    echo "[5/6] Optimizing the inpainted 3D Gaussian scene"
    require_file \
        "${MODEL_ROOT}/virtual/ours_object_removal/iteration_${DISTILL_ITERATION}/fused_mask_col_dep_ply/00004.ply"

    "${PYTHON_BIN}" edit_object_inpaint.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${MODEL_ROOT}" \
        --config_file "${INPAINT_CONFIG}" \
        --resolution "${RESOLUTION}" \
        --render_video
else
    echo "[5/6] Skipping 3D Gaussian inpainting optimization"
fi

if (( START_STAGE <= 6 )); then
    if [[ "${DATASET_NAME}" == "inpaint360" ]]; then
        echo "[6/6] Evaluating the inpainted scene"
        require_file \
            "${MODEL_ROOT}/point_cloud_object_inpaint_virtual/iteration_${FINAL_ITERATION}/point_cloud.ply"
        "${PYTHON_BIN}" tools/metrics_fid_masked.py --model_paths "${MODEL_ROOT}"
    else
        echo "[6/6] Skipping evaluation: metrics require the official inpaint360 ground truth"
    fi
else
    echo "[6/6] Skipping evaluation"
fi

echo "Inpaint360GS inpainting completed: ${MODEL_ROOT}"
