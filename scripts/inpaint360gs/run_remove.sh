#!/usr/bin/env bash

set -Eeuo pipefail

# Inpaint360GS object-removal and interactive mask-refinement pipeline.
#
# Usage:
#   bash scripts/inpaint360gs/run_remove.sh \
#       [dataset_name] [scene] [resolution] [target_id] \
#       [surrounding_ids] [start_stage]
#
# Examples:
#   bash scripts/inpaint360gs/run_remove.sh inpaint360 doppelherz 2 26 "24,10"
#   bash scripts/inpaint360gs/run_remove.sh \
#       "mip-nerf/360_v2" kitchen 8 14 none 1
#   bash scripts/inpaint360gs/run_remove.sh \
#       "mip-nerf/360_v2" kitchen 8 14 none 3
#
# surrounding_ids: comma-separated object IDs, or "none".
# start_stage: 1=config initialization, 2=object removal,
#              3=virtual camera generation, 4=interactive mask refinement.
#
# The Python environment must already contain the dependencies installed by
# submodules/Inpaint360GS/install.sh (for this workspace, normally `paintmesh`).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INPAINT360GS_ROOT="${REPO_ROOT}/submodules/Inpaint360GS"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_NAME="${1:-${DATASET_NAME:-inpaint360}}"
SCENE="${2:-${SCENE:-doppelherz}}"
RESOLUTION="${3:-${RESOLUTION:-2}}"
TARGET_ID="${4:-${TARGET_ID:-26}}"
SURROUNDING_IDS="${5:-${SURROUNDING_IDS:-none}}"
START_STAGE="${6:-${START_STAGE:-1}}"

usage() {
    sed -n '5,24p' "${BASH_SOURCE[0]}"
}

if [[ "${DATASET_NAME}" == "-h" || "${DATASET_NAME}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# > 6 )); then
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
    1|2|3|4)
        ;;
    *)
        echo "Error: start_stage must be an integer from 1 to 4; got '${START_STAGE}'." >&2
        exit 2
        ;;
esac

if [[ ! "${TARGET_ID}" =~ ^[0-9]+([,][0-9]+)*$ ]]; then
    echo "Error: target_id must be an integer or comma-separated integers; got '${TARGET_ID}'." >&2
    exit 2
fi

if [[ "${SURROUNDING_IDS,,}" != "none" && \
      ! "${SURROUNDING_IDS}" =~ ^[0-9]+([,][0-9]+)*$ ]]; then
    echo "Error: surrounding_ids must be 'none' or comma-separated integers; got '${SURROUNDING_IDS}'." >&2
    exit 2
fi

SCENE_ROOT="${DATA_ROOT}/${DATASET_NAME}/${SCENE}"
MODEL_ROOT="${OUTPUT_ROOT}/${DATASET_NAME}/${SCENE}"
REMOVAL_CONFIG="${INPAINT360GS_ROOT}/config/object_removal/${DATASET_NAME}/${SCENE}.json"
INPAINT_CONFIG="${INPAINT360GS_ROOT}/config/object_inpaint/${DATASET_NAME}/${SCENE}.json"
TRACKER_ROOT="${INPAINT360GS_ROOT}/Segment-and-Track-Anything"
TRACKER_CKPT_ROOT="${TRACKER_ROOT}/ckpt"

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

link_tracker_checkpoint() {
    local checkpoint_name="$1"
    local checkpoint_source="${CKPT_ROOT}/${checkpoint_name}"
    local checkpoint_target="${TRACKER_CKPT_ROOT}/${checkpoint_name}"

    require_file "${checkpoint_source}"

    if [[ ! -e "${checkpoint_target}" && ! -L "${checkpoint_target}" ]]; then
        ln -s "${checkpoint_source}" "${checkpoint_target}"
    fi

    require_file "${checkpoint_target}"
}

trap 'echo "Error: Inpaint360GS run_remove failed at line ${LINENO}." >&2' ERR

require_dir "${INPAINT360GS_ROOT}"
require_dir "${DATA_ROOT}"
require_dir "${SCENE_ROOT}"
require_dir "${MODEL_ROOT}"
require_dir "${TRACKER_ROOT}"

# Object removal and virtual-view generation require the object-aware 3DGS
# produced by run_seg.sh, including its per-Gaussian object classifier.
if (( START_STAGE <= 3 )); then
    require_file "${SCENE_ROOT}/associated_hqsam/scene.json"
    require_file "${MODEL_ROOT}/cfg_args"
    require_matching_file \
        "${MODEL_ROOT}/point_cloud/iteration_*/point_cloud.ply" \
        "an object-aware 3DGS checkpoint"
    require_matching_file \
        "${MODEL_ROOT}/point_cloud/iteration_*/classifier.pth" \
        "an object classifier"
fi

# Skipping configuration initialization requires both generated configs.
if (( START_STAGE > 1 && START_STAGE <= 3 )); then
    require_file "${REMOVAL_CONFIG}"
    require_file "${INPAINT_CONFIG}"
fi

# Skipping removal requires its saved Gaussian scene.
if (( START_STAGE == 3 )); then
    require_matching_file \
        "${MODEL_ROOT}/point_cloud_object_removal/iteration_*/point_cloud.ply" \
        "a stage-2 object-removal checkpoint"
fi

# The interactive application resolves both checkpoints relative to its own
# working directory, so expose the repository-level weights through symlinks.
if (( START_STAGE <= 4 )); then
    require_dir "${CKPT_ROOT}"
    mkdir -p "${TRACKER_CKPT_ROOT}"
    link_tracker_checkpoint "sam_vit_b_01ec64.pth"
    link_tracker_checkpoint "R50_DeAOTL_PRE_YTB_DAV.pth"
    link_tracker_checkpoint "groundingdino_swint_ogc.pth"
fi

export PYTHONPATH="${INPAINT360GS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Inpaint360GS root : ${INPAINT360GS_ROOT}"
echo "Dataset           : ${SCENE_ROOT}"
echo "Checkpoints       : ${CKPT_ROOT}"
echo "Output            : ${MODEL_ROOT}"
echo "Resolution        : ${RESOLUTION}"
echo "Target IDs        : ${TARGET_ID}"
echo "Surrounding IDs   : ${SURROUNDING_IDS}"
echo "Start stage       : ${START_STAGE}"

cd "${INPAINT360GS_ROOT}"

if (( START_STAGE <= 1 )); then
    echo "[1/4] Initializing object-removal and inpainting configuration"
    "${PYTHON_BIN}" tools/init_configs.py \
        --dataset_name "${DATASET_NAME}" \
        --scene "${SCENE}" \
        --target_id "${TARGET_ID}" \
        --target_surronding_id "${SURROUNDING_IDS}"
    require_file "${REMOVAL_CONFIG}"
    require_file "${INPAINT_CONFIG}"
else
    echo "[1/4] Skipping configuration initialization"
fi

if (( START_STAGE <= 2 )); then
    echo "[2/4] Removing the selected object Gaussians"
    "${PYTHON_BIN}" edit_object_removal.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${MODEL_ROOT}" \
        --resolution "${RESOLUTION}" \
        --config_file "${REMOVAL_CONFIG}" \
        --render_video
else
    echo "[2/4] Skipping object removal"
fi

if (( START_STAGE <= 3 )); then
    echo "[3/4] Generating the virtual camera trajectory"
    "${PYTHON_BIN}" tools/virtual_pose.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${MODEL_ROOT}" \
        --resolution "${RESOLUTION}" \
        --config_file "${REMOVAL_CONFIG}"
else
    echo "[3/4] Skipping virtual camera generation"
fi

require_file "${TRACKER_ROOT}/assets/images.zip"

echo "[4/4] Launching interactive mask refinement"
echo "Stop the interface with Ctrl+C after exporting the refined masks."
cd "${TRACKER_ROOT}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export GRADIO_SHARE="${GRADIO_SHARE:-false}"
echo "Local interface URL: http://${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT}"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u app.py
