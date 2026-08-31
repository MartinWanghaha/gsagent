#!/usr/bin/env bash

set -Eeuo pipefail

# Inpaint360GS object-aware Gaussian training pipeline.
#
# Usage:
#   bash scripts/inpaint360gs/run_seg.sh [dataset_name] [scene] [resolution] [start_stage]
#
# Examples:
#   bash scripts/inpaint360gs/run_seg.sh
#   bash scripts/inpaint360gs/run_seg.sh inpaint360 doppelherz 2
#   bash scripts/inpaint360gs/run_seg.sh "mip-nerf/360_v2" kitchen 8 2
#   bash scripts/inpaint360gs/run_seg.sh "mip-nerf/360_v2" kitchen 8 3 \
#       2>&1 | tee -a logs/inpaint360gs_kitchen.log
#
# start_stage: 1=vanilla 3DGS, 2=2D masks, 3=3D association,
#              4=label preview, 5=object-feature distillation, 6=rendering.
#
# The Python environment must already contain the dependencies installed by
# submodules/Inpaint360GS/install.sh (normally the `inpaint360gs` conda env).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INPAINT360GS_ROOT="${REPO_ROOT}/submodules/Inpaint360GS"

# Repository-level storage requested by this workspace.
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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

case "${START_STAGE}" in
    1|2|3|4|5|6)
        ;;
    *)
        echo "Error: start_stage must be an integer from 1 to 6; got '${START_STAGE}'." >&2
        exit 2
        ;;
esac

case "${RESOLUTION}" in
    1)
        SEGMENTATION_IMAGE_FOLDER="images"
        ;;
    2|4|8)
        SEGMENTATION_IMAGE_FOLDER="images_${RESOLUTION}"
        ;;
    *)
        echo "Error: resolution must be one of 1, 2, 4, or 8; got '${RESOLUTION}'." >&2
        exit 2
        ;;
esac

SCENE_ROOT="${DATA_ROOT}/${DATASET_NAME}/${SCENE}"
SCENE_OUTPUT_ROOT="${OUTPUT_ROOT}/${DATASET_NAME}/${SCENE}"
VANILLA_MODEL_ROOT="${SCENE_OUTPUT_ROOT}/3dgs_output"
SEG_WEIGHT_ROOT="${INPAINT360GS_ROOT}/seg/weight"

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

link_checkpoint_if_missing() {
    local checkpoint_name="$1"
    local checkpoint_source="${CKPT_ROOT}/${checkpoint_name}"
    local checkpoint_target="${SEG_WEIGHT_ROOT}/${checkpoint_name}"

    require_file "${checkpoint_source}"

    if [[ ! -e "${checkpoint_target}" ]]; then
        ln -s "${checkpoint_source}" "${checkpoint_target}"
    fi

    require_file "${checkpoint_target}"
}

trap 'echo "Error: Inpaint360GS run_seg failed at line ${LINENO}." >&2' ERR

require_dir "${INPAINT360GS_ROOT}"
require_dir "${DATA_ROOT}"
require_dir "${SCENE_ROOT}"
require_dir "${SCENE_ROOT}/images"
require_dir "${SCENE_ROOT}/${SEGMENTATION_IMAGE_FOLDER}"
require_dir "${SCENE_ROOT}/sparse/0"
require_file "${SCENE_ROOT}/sparse/0/cameras.bin"
require_file "${SCENE_ROOT}/sparse/0/images.bin"
require_file "${SCENE_ROOT}/sparse/0/points3D.bin"

mkdir -p "${SCENE_OUTPUT_ROOT}" "${SEG_WEIGHT_ROOT}"

# Stages 1 and 2 will run CropFormer. raw_mask_sam.py reads fixed
# project-relative weight paths from seg_config.json, so keep the checkpoints
# in the repository-level ckpt directory and expose them through symlinks.
if (( START_STAGE <= 2 )); then
    require_dir "${CKPT_ROOT}"
    link_checkpoint_if_missing "CropFormer_hornet_3x_03823a.pth"

    if [[ -f "${CKPT_ROOT}/sam_vit_h_4b8939.pth" ]]; then
        if [[ ! -e "${SEG_WEIGHT_ROOT}/sam_vit_h_4b8939.pth" ]]; then
            ln -s "${CKPT_ROOT}/sam_vit_h_4b8939.pth" \
                "${SEG_WEIGHT_ROOT}/sam_vit_h_4b8939.pth"
        fi
    fi
fi

# Validate the artifacts that should have been produced by skipped stages.
if (( START_STAGE > 1 && START_STAGE <= 5 )); then
    require_file "${VANILLA_MODEL_ROOT}/cfg_args"
    require_matching_file \
        "${VANILLA_MODEL_ROOT}/point_cloud/iteration_*/point_cloud.ply" \
        "a vanilla 3DGS checkpoint"
fi

if (( START_STAGE == 3 )); then
    require_matching_file \
        "${SCENE_ROOT}/raw_hqsam/*.png" \
        "stage-2 raw instance masks"
fi

if (( START_STAGE == 4 || START_STAGE == 5 )); then
    require_file "${SCENE_ROOT}/associated_hqsam/scene.json"
    require_matching_file \
        "${SCENE_ROOT}/associated_hqsam/*.png" \
        "stage-3 associated instance masks"
fi

if (( START_STAGE == 6 )); then
    require_file "${SCENE_ROOT}/associated_hqsam/scene.json"
    require_file "${SCENE_OUTPUT_ROOT}/cfg_args"
    require_matching_file \
        "${SCENE_OUTPUT_ROOT}/point_cloud/iteration_*/point_cloud.ply" \
        "a distilled object-aware 3DGS checkpoint"
    require_matching_file \
        "${SCENE_OUTPUT_ROOT}/point_cloud/iteration_*/classifier.pth" \
        "a distilled object classifier"
fi

# add_label_num_hqsam.py always looks for images_<resolution>. Provide its
# expected alias when running at native resolution.
if [[ "${RESOLUTION}" == "1" && ! -e "${SCENE_ROOT}/images_1" ]]; then
    ln -s "images" "${SCENE_ROOT}/images_1"
fi

export PYTHONPATH="${INPAINT360GS_ROOT}:${INPAINT360GS_ROOT}/seg/detectron2${PYTHONPATH:+:${PYTHONPATH}}"

echo "Inpaint360GS root : ${INPAINT360GS_ROOT}"
echo "Dataset           : ${SCENE_ROOT}"
echo "Checkpoints       : ${CKPT_ROOT}"
echo "Output            : ${SCENE_OUTPUT_ROOT}"
echo "Resolution        : ${RESOLUTION}"
echo "Start stage       : ${START_STAGE}"

cd "${INPAINT360GS_ROOT}"

if (( START_STAGE <= 1 )); then
    echo "[1/6] Training vanilla 3D Gaussian Splatting"
    "${PYTHON_BIN}" gaussian_splatting/train.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${VANILLA_MODEL_ROOT}" \
        --init_mode sparse \
        --eval \
        --resolution "${RESOLUTION}"
else
    echo "[1/6] Skipping vanilla 3D Gaussian Splatting"
fi

if (( START_STAGE <= 2 )); then
    echo "[2/6] Generating per-view CropFormer instance masks"
    "${PYTHON_BIN}" seg/raw_mask_sam.py \
        --dataset_path "${DATA_ROOT}/${DATASET_NAME}" \
        --scene_name "${SCENE}" \
        --image_folder "${SEGMENTATION_IMAGE_FOLDER}" \
        --method hqsam
else
    echo "[2/6] Skipping per-view CropFormer instance masks"
fi

if (( START_STAGE <= 3 )); then
    echo "[3/6] Associating instance masks across views through 3D Gaussians"
    "${PYTHON_BIN}" seg/mask_associate.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${VANILLA_MODEL_ROOT}" \
        --resolution "${RESOLUTION}" \
        --mask_generator hqsam \
        --eval
else
    echo "[3/6] Skipping 3D instance association"
fi

if (( START_STAGE <= 4 )); then
    echo "[4/6] Rendering instance IDs on the source images"
    "${PYTHON_BIN}" tools/add_label_num_hqsam.py \
        --source_path "${SCENE_ROOT}" \
        --resolution "${RESOLUTION}" \
        --mask_generator hqsam
else
    echo "[4/6] Skipping instance-ID preview rendering"
fi

if (( START_STAGE <= 5 )); then
    echo "[5/6] Distilling 2D instance labels into Gaussian object features"
    "${PYTHON_BIN}" seg/distillation.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${SCENE_OUTPUT_ROOT}" \
        --vanilla_3dgs_path "${VANILLA_MODEL_ROOT}" \
        --resolution "${RESOLUTION}" \
        --object_path associated_hqsam \
        --eval
else
    echo "[5/6] Skipping Gaussian object-feature distillation"
fi

echo "[6/6] Rendering the object-aware scene and trajectory video"
"${PYTHON_BIN}" render.py \
    --model_path "${SCENE_OUTPUT_ROOT}" \
    --render_video

echo "Inpaint360GS segmentation stage completed: ${SCENE_OUTPUT_ROOT}"
