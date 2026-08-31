#!/usr/bin/env bash

set -Eeuo pipefail

# PaintMesh object-aware RGB/depth and 3D Gaussian inpainting pipeline.
#
# Usage:
#   scripts/paintmesh/run_inpaint \
#       [dataset_name] [scene] [resolution] target_ids \
#       [surrounding_ids] [start_stage]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INVOCATION_ROOT="$(pwd -P)"
INPAINT360GS_ROOT="${REPO_ROOT}/submodules/Inpaint360GS"
EDGS_ROOT="${REPO_ROOT}/submodules/EDGS"
LAMA_ROOT="${INPAINT360GS_ROOT}/LaMa"

fail() {
    echo "Error: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  scripts/paintmesh/run_inpaint \
      [dataset_name] [scene] [resolution] target_ids \
      [surrounding_ids] [start_stage]

Example:
  GPU=0 scripts/paintmesh/run_inpaint \
      "mip-nerf/360_v2" kitchen 8 14 none 1

Arguments:
  target_ids       target instance ID(s), comma-separated
  surrounding_ids  temporary occluder ID(s), or "none"
  start_stage       first stage to execute, in [1, 8]

Stages:
  1 = validate removal/tracker inputs and create an isolated workspace
  2 = prepare run-local LaMa RGB/depth/mask inputs
  3 = complete RGB/depth with LaMa and validate every frame
  4 = back-project the completed RGB-D views into support point clouds
  5 = optimize the inpainted object-aware 3D Gaussian scene
  6 = publish an EDGS-loadable inpainted 3DGS
  7 = PGSR rendering and TSDF mesh reconstruction
  8 = semantic lifting and final artifact validation

Important environment overrides:
  PAINTMESH_ENV=paintmesh
  LAMA_ENV=lama
  GPU=0
  END_STAGE=8
  DISTILL_ITERATION=2000
  FINETUNE_ITERATION=5000
  FUSION_SEED_FRAME=4
  MASK_MIN_AREA=50
  MASK_DILATION=10
  RECURSIVE_GUIDE=false
  RENDER_INPAINT_VIDEO=false
  RENDER_INPAINT_TRAIN=false
  RENDER_INPAINT_TEST=false
  RENDER_EDGS_TEST=false
  WRITE_HOLE_PLY=false
  WRITE_COLORED_MESH=true
  INPAINT_RUN_NAME=default
  REMOVAL_ROOT=/absolute/removal/run
  INPAINT_RUN_ROOT=/absolute/inpaint/run
EOF
}

resolve_from_invocation() {
    local value="$1"
    if [[ "${value}" == /* ]]; then
        realpath -m -- "${value}"
    else
        realpath -m -- "${INVOCATION_ROOT}/${value}"
    fi
}

require_file() {
    [[ -f "$1" ]] || fail "required file does not exist: $1"
}

require_regular_file() {
    [[ -f "$1" && ! -L "$1" ]] || \
        fail "required independent regular file does not exist: $1"
}

require_dir() {
    [[ -d "$1" ]] || fail "required directory does not exist: $1"
}

require_boolean() {
    case "${1,,}" in
        1|true|yes|on|0|false|no|off) ;;
        *) fail "expected a boolean value, got '$1'" ;;
    esac
}

is_true() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        0|false|no|off) return 1 ;;
        *) fail "expected a boolean value, got '$1'" ;;
    esac
}

normalize_id_list() {
    local raw="$1"
    local allow_none="$2"
    local label="$3"
    local output_name="$4"
    local item joined
    local -a items normalized sorted
    local -A seen=()

    if is_true "${allow_none}" && [[ "${raw,,}" == "none" ]]; then
        printf -v "${output_name}" '%s' "none"
        return
    fi
    [[ -n "${raw}" ]] || fail "${label} cannot be empty"
    IFS=',' read -r -a items <<< "${raw}"
    for item in "${items[@]}"; do
        [[ "${item}" =~ ^[1-9][0-9]*$ ]] || \
            fail "${label} must contain canonical positive integers; got '${raw}'"
        [[ -z "${seen[${item}]+present}" ]] || \
            fail "${label} contains duplicate ID ${item}"
        seen["${item}"]=1
        normalized+=("${item}")
    done
    mapfile -t sorted < <(printf '%s\n' "${normalized[@]}" | sort -n)
    local IFS=','
    joined="${sorted[*]}"
    printf -v "${output_name}" '%s' "${joined}"
}

ids_overlap() {
    local left="$1"
    local right="$2"
    local item
    local -a left_values right_values
    local -A selected=()
    [[ "${right}" != "none" ]] || return 1
    IFS=',' read -r -a left_values <<< "${left}"
    IFS=',' read -r -a right_values <<< "${right}"
    for item in "${left_values[@]}"; do selected["${item}"]=1; done
    for item in "${right_values[@]}"; do
        [[ -z "${selected[${item}]+present}" ]] || return 0
    done
    return 1
}

validate_scene_key() {
    local dataset_name="$1"
    local scene="$2"
    local component
    local -a components
    [[ -n "${dataset_name}" && "${dataset_name}" != /* && \
       "${dataset_name}" != */ && "${dataset_name}" != *//* ]] || \
        fail "dataset_name must be a non-empty relative path"
    IFS='/' read -r -a components <<< "${dataset_name}"
    for component in "${components[@]}"; do
        [[ -n "${component}" && "${component}" != "." && "${component}" != ".." ]] || \
            fail "dataset_name contains an unsafe path component: '${dataset_name}'"
    done
    [[ -n "${scene}" && "${scene}" != "." && "${scene}" != ".." && \
       "${scene}" != */* ]] || fail "scene must be one safe path component"
}

should_run() {
    local stage="$1"
    (( START_STAGE <= stage && END_STAGE >= stage ))
}

DATA_ROOT="$(resolve_from_invocation "${DATA_ROOT:-${REPO_ROOT}/data}")"
CKPT_ROOT="$(resolve_from_invocation "${CKPT_ROOT:-${REPO_ROOT}/ckpt}")"
OUTPUT_ROOT="$(resolve_from_invocation "${OUTPUT_ROOT:-${REPO_ROOT}/output}")"
CONDA_BIN="${CONDA_BIN:-conda}"
if [[ "${CONDA_BIN}" == */* ]]; then
    CONDA_BIN="$(resolve_from_invocation "${CONDA_BIN}")"
fi
PAINTMESH_ENV="${PAINTMESH_ENV:-paintmesh}"
# LaMa's pinned Lightning/WebDataset stack is isolated from the newer main
# environment. It can be set to paintmesh once those optional packages exist.
LAMA_ENV="${LAMA_ENV:-lama}"
GPU="${GPU:-}"

DATASET_NAME="${1:-${DATASET_NAME:-inpaint360}}"
SCENE="${2:-${SCENE:-doppelherz}}"
RESOLUTION="${3:-${RESOLUTION:-2}}"
TARGET_IDS_RAW="${4:-${TARGET_IDS:-}}"
SURROUNDING_IDS_RAW="${5:-${SURROUNDING_IDS:-none}}"
START_STAGE="${6:-${START_STAGE:-1}}"
END_STAGE="${END_STAGE:-8}"

if [[ "${DATASET_NAME}" == "-h" || "${DATASET_NAME}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# > 6 )); then usage >&2; exit 2; fi
validate_scene_key "${DATASET_NAME}" "${SCENE}"
[[ -n "${TARGET_IDS_RAW}" ]] || { usage >&2; fail "target_ids is required"; }
case "${RESOLUTION}" in 1|2|4|8) ;; *) fail "resolution must be 1, 2, 4, or 8" ;; esac
case "${START_STAGE}" in 1|2|3|4|5|6|7|8) ;; *) fail "start_stage must be in [1, 8]" ;; esac
case "${END_STAGE}" in 1|2|3|4|5|6|7|8) ;; *) fail "END_STAGE must be in [1, 8]" ;; esac
(( START_STAGE <= END_STAGE )) || fail "start_stage cannot exceed END_STAGE"

normalize_id_list "${TARGET_IDS_RAW}" false "target_ids" TARGET_IDS
normalize_id_list "${SURROUNDING_IDS_RAW}" true "surrounding_ids" SURROUNDING_IDS
ids_overlap "${TARGET_IDS}" "${SURROUNDING_IDS}" && \
    fail "target_ids and surrounding_ids must be disjoint"

TARGET_KEY="${TARGET_IDS//,/-}"
REMOVAL_RUN_NAME="target_${TARGET_KEY}"
if [[ "${SURROUNDING_IDS}" != "none" ]]; then
    REMOVAL_RUN_NAME+="__surrounding_${SURROUNDING_IDS//,/-}"
fi
INPAINT_RUN_NAME="${INPAINT_RUN_NAME:-default}"
[[ "${INPAINT_RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    fail "INPAINT_RUN_NAME may contain only letters, digits, '.', '_' and '-'"

DISTILL_ITERATION="${DISTILL_ITERATION:-2000}"
[[ "${DISTILL_ITERATION}" =~ ^[1-9][0-9]*$ ]] || \
    fail "DISTILL_ITERATION must be a canonical positive integer"
FINETUNE_ITERATION_REQUESTED="${FINETUNE_ITERATION:-}"
if [[ -n "${FINETUNE_ITERATION_REQUESTED}" ]]; then
    [[ "${FINETUNE_ITERATION_REQUESTED}" =~ ^[1-9][0-9]*$ ]] || \
        fail "FINETUNE_ITERATION must be a canonical positive integer"
fi
FUSION_SEED_FRAME="${FUSION_SEED_FRAME:-4}"
[[ "${FUSION_SEED_FRAME}" =~ ^([0-9]|[12][0-9])$ ]] || \
    fail "FUSION_SEED_FRAME must be an integer in [0, 29]"
printf -v FUSION_SEED_NAME '%05d' "${FUSION_SEED_FRAME}"

MASK_MIN_AREA="${MASK_MIN_AREA:-50}"
MASK_DILATION="${MASK_DILATION:-10}"
[[ "${MASK_MIN_AREA}" =~ ^[1-9][0-9]*$ ]] || fail "MASK_MIN_AREA must be positive"
[[ "${MASK_DILATION}" =~ ^[0-9]+$ ]] || fail "MASK_DILATION must be non-negative"

RECURSIVE_GUIDE="${RECURSIVE_GUIDE:-false}"
RENDER_INPAINT_VIDEO="${RENDER_INPAINT_VIDEO:-false}"
RENDER_INPAINT_TRAIN="${RENDER_INPAINT_TRAIN:-false}"
RENDER_INPAINT_TEST="${RENDER_INPAINT_TEST:-false}"
RENDER_EDGS_TEST="${RENDER_EDGS_TEST:-false}"
WRITE_HOLE_PLY="${WRITE_HOLE_PLY:-false}"
WRITE_COLORED_MESH="${WRITE_COLORED_MESH:-true}"
USE_DEPTH_FILTER="${USE_DEPTH_FILTER:-false}"
for flag in \
    "${RECURSIVE_GUIDE}" "${RENDER_INPAINT_VIDEO}" \
    "${RENDER_INPAINT_TRAIN}" "${RENDER_INPAINT_TEST}" \
    "${RENDER_EDGS_TEST}" "${WRITE_HOLE_PLY}" \
    "${WRITE_COLORED_MESH}" "${USE_DEPTH_FILTER}"; do
    require_boolean "${flag}"
done

EDGS_IMAGES="${EDGS_IMAGES:-images}"
MAX_DEPTH="${MAX_DEPTH:-5.0}"
VOXEL_SIZE="${VOXEL_SIZE:-0.002}"
NUM_CLUSTERS="${NUM_CLUSTERS:-1}"
MESH_NEIGHBORS="${MESH_NEIGHBORS:-8}"
MESH_CHUNK_SIZE="${MESH_CHUNK_SIZE:-32768}"
MESH_WORKERS="${MESH_WORKERS:--1}"
MESH_OPACITY_MIN="${MESH_OPACITY_MIN:-0.01}"
MESH_SUPPORT_SIGMA="${MESH_SUPPORT_SIGMA:-3.0}"
MESH_NORMAL_POWER="${MESH_NORMAL_POWER:-2.0}"
MESH_MIN_CONFIDENCE="${MESH_MIN_CONFIDENCE:-0.10}"
MESH_MIN_MARGIN="${MESH_MIN_MARGIN:-0.02}"
MESH_UNKNOWN_ID="${MESH_UNKNOWN_ID:-65535}"

PIPELINE_ROOT="$(resolve_from_invocation "${PIPELINE_ROOT:-${OUTPUT_ROOT}/paintmesh/${DATASET_NAME}/${SCENE}}")"
REMOVAL_ROOT="$(resolve_from_invocation "${REMOVAL_ROOT:-${PIPELINE_ROOT}/removal/${REMOVAL_RUN_NAME}}")"
INPAINT_RUN_ROOT="$(resolve_from_invocation "${INPAINT_RUN_ROOT:-${REMOVAL_ROOT}/inpaint/${INPAINT_RUN_NAME}}")"
SCENE_ROOT="${DATA_ROOT}/${DATASET_NAME}/${SCENE}"

REMOVAL_WORK_MODEL="${REMOVAL_ROOT}/work_model"
REMOVAL_WORKSPACE_MANIFEST="${REMOVAL_WORK_MODEL}/workspace_manifest.json"
REMOVAL_MANIFEST="${REMOVAL_ROOT}/removal_manifest.json"
REMOVED_MODEL_MANIFEST="${REMOVAL_ROOT}/removed_3dgs/model_manifest.json"
TRACKING_SESSION="${REMOVAL_ROOT}/tracker/tracking_session.json"
TRACKING_MASK_ROOT="${REMOVAL_ROOT}/tracker/results/images/images_masks"
VIRTUAL_CAMERA_MANIFEST="${REMOVAL_ROOT}/tracker/virtual_cameras.json"
INPAINT_CONFIG="${REMOVAL_ROOT}/config/object_inpaint/${DATASET_NAME}/${SCENE}.json"

WORK_MODEL="${INPAINT_RUN_ROOT}/work_model"
WORKSPACE_MANIFEST="${WORK_MODEL}/workspace_manifest.json"
MANIFEST_ROOT="${INPAINT_RUN_ROOT}/manifests"
LAMA_INPUT_ROOT="${INPAINT_RUN_ROOT}/lama/input"
LAMA_OUTPUT_ROOT="${INPAINT_RUN_ROOT}/lama/output"
LAMA_COLOR_INPUT="${LAMA_INPUT_ROOT}/color"
LAMA_DEPTH_INPUT="${LAMA_INPUT_ROOT}/depth"
LAMA_COLOR_OUTPUT="${LAMA_OUTPUT_ROOT}/color"
LAMA_DEPTH_OUTPUT="${LAMA_OUTPUT_ROOT}/depth"
LAMA_INPUT_MANIFEST="${MANIFEST_ROOT}/lama_input_manifest.json"
LAMA_COMPLETION_MANIFEST="${MANIFEST_ROOT}/lama_completion_manifest.json"
LAMA_MODEL_PATH="$(resolve_from_invocation "${LAMA_MODEL_PATH:-${CKPT_ROOT}/big-lama}")"

FUSED_ROOT="${INPAINT_RUN_ROOT}/fused/mask"
HOLE_ROOT="${INPAINT_RUN_ROOT}/fused/hole"
FUSION_MANIFEST="${MANIFEST_ROOT}/fusion_manifest.json"
SUPPORT_PLY="${FUSED_ROOT}/${FUSION_SEED_NAME}.ply"

INPAINTED_WORK_PLY=""
INPAINTED_GS_ROOT="${INPAINT_RUN_ROOT}/inpainted_3dgs"
INPAINTED_MODEL_MANIFEST="${INPAINTED_GS_ROOT}/model_manifest.json"
INPAINTED_MESH_ROOT="${INPAINT_RUN_ROOT}/inpainted_mesh"
FINAL_MANIFEST="${INPAINT_RUN_ROOT}/inpaint_manifest.json"

run_python() {
    local workdir="$1"
    local pythonpath="$2"
    local environment_name="$3"
    shift 3
    local -a environment=("PYTHONPATH=${pythonpath}")
    if [[ -n "${GPU}" ]]; then environment+=("CUDA_VISIBLE_DEVICES=${GPU}"); fi
    (
        cd "${workdir}"
        env "${environment[@]}" \
            "${CONDA_BIN}" run --no-capture-output -n "${environment_name}" \
            python "$@"
    )
}

run_inpaint() {
    run_python \
        "${INPAINT360GS_ROOT}" \
        "${INPAINT360GS_ROOT}:${INPAINT360GS_ROOT}/seg/detectron2" \
        "${PAINTMESH_ENV}" "$@"
}

run_edgs() {
    run_python "${EDGS_ROOT}" "${EDGS_ROOT}" "${PAINTMESH_ENV}" "$@"
}

run_lama() {
    local -a environment=(
        "PYTHONPATH=${LAMA_ROOT}"
        "TORCH_HOME=${LAMA_ROOT}"
    )
    if [[ -n "${GPU}" ]]; then environment+=("CUDA_VISIBLE_DEVICES=${GPU}"); fi
    if [[ -n "${LAMA_ENV_PREFIX:-}" ]]; then
        environment+=("LD_LIBRARY_PATH=${LAMA_ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}")
    fi
    (
        cd "${LAMA_ROOT}"
        env "${environment[@]}" \
            "${CONDA_BIN}" run --no-capture-output -n "${LAMA_ENV}" python "$@"
    )
}

require_manifest_kind() {
    local path="$1"
    local kind="$2"
    run_inpaint - "${path}" "${kind}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
kind = sys.argv[2]
if not path.is_file():
    raise SystemExit(f"required manifest is missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("kind") != kind:
    raise SystemExit(f"unexpected manifest kind in {path}: {payload.get('kind')!r}")
if payload.get("complete") is not True or payload.get("status") != "complete":
    raise SystemExit(f"manifest is incomplete: {path}")
if not isinstance(payload.get("artifact_id"), str) or len(payload["artifact_id"]) != 64:
    raise SystemExit(f"manifest has no valid artifact_id: {path}")
print(f"Complete {kind}: {path}")
PY
}

validate_lama_completion() {
    local -a arguments=(
        tools/prepare_paintmesh_lama_data.py validate-output
        --color-input "${LAMA_COLOR_INPUT}"
        --depth-input "${LAMA_DEPTH_INPUT}"
        --color-output "${LAMA_COLOR_OUTPUT}"
        --depth-output "${LAMA_DEPTH_OUTPUT}"
        --model-path "${LAMA_MODEL_PATH}"
        --input-manifest "${LAMA_INPUT_MANIFEST}"
        --manifest "${LAMA_COMPLETION_MANIFEST}"
        --frames 30
    )
    if is_true "${RECURSIVE_GUIDE}"; then arguments+=(--recursive-guide); fi
    run_inpaint "${arguments[@]}"
}

read_final_iteration() {
    run_inpaint - "${INPAINT_CONFIG}" "${FINETUNE_ITERATION_REQUESTED}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
requested = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
value = payload.get("finetune_iteration")
if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise SystemExit(f"finetune_iteration must be a positive integer in {path}")
if requested and int(requested) != value:
    raise SystemExit(
        f"FINETUNE_ITERATION={requested} differs from {path}: {value}; "
        "use a run-local config variant instead of overriding checkpoint identity"
    )
print(value)
PY
}

read_manifest_input_path() {
    local manifest="$1"
    local key="$2"
    run_inpaint - "${manifest}" "${key}" <<'PY'
import json
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
key = sys.argv[2]
payload = json.loads(manifest.read_text(encoding="utf-8"))
record = payload.get("inputs", {}).get(key, {})
path_text = record.get("path")
if not isinstance(path_text, str) or not path_text:
    raise SystemExit(f"manifest has no inputs.{key}.path: {manifest}")
path = Path(path_text).expanduser().resolve(strict=True)
if not path.is_file():
    raise SystemExit(f"manifest input is not a file: {path}")
print(path)
PY
}

validate_numeric_settings() {
    run_inpaint - \
        "${MAX_DEPTH}" "${VOXEL_SIZE}" "${NUM_CLUSTERS}" \
        "${MESH_NEIGHBORS}" "${MESH_CHUNK_SIZE}" \
        "${MESH_OPACITY_MIN}" "${MESH_SUPPORT_SIGMA}" \
        "${MESH_NORMAL_POWER}" "${MESH_MIN_CONFIDENCE}" \
        "${MESH_MIN_MARGIN}" "${MESH_UNKNOWN_ID}" <<'PY'
import math
import sys

max_depth, voxel_size = map(float, sys.argv[1:3])
num_clusters, neighbors, chunk_size = map(int, sys.argv[3:6])
opacity, sigma, normal_power, confidence, margin = map(float, sys.argv[6:11])
unknown = int(sys.argv[11])
values = (max_depth, voxel_size, opacity, sigma, normal_power, confidence, margin)
if not all(math.isfinite(value) for value in values):
    raise SystemExit("mesh settings must be finite")
if max_depth <= 0 or voxel_size <= 0 or sigma <= 0:
    raise SystemExit("MAX_DEPTH, VOXEL_SIZE and MESH_SUPPORT_SIGMA must be positive")
if min(num_clusters, neighbors, chunk_size) <= 0:
    raise SystemExit("NUM_CLUSTERS, MESH_NEIGHBORS and MESH_CHUNK_SIZE must be positive")
if not 0 <= opacity <= 1 or normal_power < 0:
    raise SystemExit("invalid opacity or normal weighting setting")
if not 0 <= confidence <= 1 or not 0 <= margin <= 1:
    raise SystemExit("semantic confidence settings must be in [0, 1]")
if not 1 <= unknown <= 2**32 - 1:
    raise SystemExit("MESH_UNKNOWN_ID is outside uint32")
PY
}

publish_mesh_manifest() {
    local -a arguments=(
        tools/publish_inpaint_mesh.py
        --model-manifest "${INPAINTED_MODEL_MANIFEST}"
        --gaussian-ply "${INPAINTED_PLY}"
        --mesh "${INPAINTED_RAW_MESH}"
        --train-render-manifest "${INPAINTED_RENDER_MANIFEST}"
        --output "${INPAINTED_MESH_STAGE_MANIFEST}"
        --iteration "${FINETUNE_ITERATION}"
        --source-path "${SCENE_ROOT}"
        --images "${EDGS_IMAGES}"
        --resolution "${RESOLUTION}"
        --max-depth "${MAX_DEPTH}"
        --voxel-size "${VOXEL_SIZE}"
        --num-clusters "${NUM_CLUSTERS}"
    )
    if is_true "${RENDER_EDGS_TEST}"; then
        arguments+=(--test-render-manifest "${INPAINTED_TEST_RENDER_MANIFEST}")
    fi
    if is_true "${USE_DEPTH_FILTER}"; then arguments+=(--use-depth-filter); fi
    run_inpaint "${arguments[@]}"
}

ensure_geometry_link() {
    run_inpaint - "${INPAINTED_RAW_MESH}" "${INPAINTED_MESH_ROOT}/geometry.ply" <<'PY'
import os
from pathlib import Path
import sys
import uuid

target = Path(sys.argv[1]).resolve(strict=True)
link = Path(sys.argv[2]).absolute()
link.parent.mkdir(parents=True, exist_ok=True)
relative = os.path.relpath(target, start=link.parent)
if link.is_symlink():
    if link.resolve(strict=True) == target:
        raise SystemExit(0)
elif link.exists():
    raise SystemExit(f"managed geometry path is not a symlink: {link}")
temporary = link.with_name(f".{link.name}.{uuid.uuid4().hex}.tmp")
try:
    os.symlink(relative, temporary)
    os.replace(temporary, link)
finally:
    temporary.unlink(missing_ok=True)
PY
}

validate_semantic_reuse() {
    run_inpaint - \
        "${INPAINTED_SEMANTIC_MANIFEST}" \
        "${MESH_NEIGHBORS}" "${MESH_CHUNK_SIZE}" "${MESH_WORKERS}" \
        "${MESH_OPACITY_MIN}" "${MESH_SUPPORT_SIGMA}" \
        "${MESH_NORMAL_POWER}" "${MESH_MIN_CONFIDENCE}" \
        "${MESH_MIN_MARGIN}" "${MESH_UNKNOWN_ID}" \
        "${WRITE_COLORED_MESH}" <<'PY'
import json
import math
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("complete") is not True or payload.get("status") != "complete":
    raise SystemExit(f"semantic manifest is incomplete: {path}")
parameters = payload.get("parameters", {})
expected = {
    "neighbors_requested": int(sys.argv[2]),
    "chunk_size": int(sys.argv[3]),
    "workers": int(sys.argv[4]),
    "opacity_min": float(sys.argv[5]),
    "support_sigma": float(sys.argv[6]),
    "normal_power": float(sys.argv[7]),
    "min_confidence": float(sys.argv[8]),
    "min_margin": float(sys.argv[9]),
    "unknown_id": int(sys.argv[10]),
    "write_colored_ply": sys.argv[11].lower() in {"1", "true", "yes", "on"},
}
for key, value in expected.items():
    actual = parameters.get(key)
    matches = (
        math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12)
        if isinstance(value, float) and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        else actual == value
    )
    if not matches:
        raise SystemExit(
            f"semantic setting {key}={actual!r}, expected {value!r}; "
            "choose a new INPAINT_RUN_NAME for a parameter variant"
        )
PY
}

trap 'echo "Error: paintmesh run_inpaint failed at line ${LINENO}." >&2' ERR

require_dir "${INPAINT360GS_ROOT}"
require_dir "${EDGS_ROOT}"
require_dir "${LAMA_ROOT}"
require_dir "${SCENE_ROOT}"
require_dir "${REMOVAL_ROOT}"
require_file "${REMOVAL_WORKSPACE_MANIFEST}"
require_file "${REMOVAL_MANIFEST}"
require_file "${REMOVED_MODEL_MANIFEST}"
require_file "${TRACKING_SESSION}"
require_file "${VIRTUAL_CAMERA_MANIFEST}"
require_dir "${TRACKING_MASK_ROOT}"
require_file "${INPAINT_CONFIG}"
if (( END_STAGE >= 3 )); then
    require_file "${LAMA_MODEL_PATH}/config.yaml"
    require_file "${LAMA_MODEL_PATH}/models/best.ckpt"
fi

mkdir -p "${INPAINT_RUN_ROOT}" "${MANIFEST_ROOT}"
run_inpaint -c \
    'import cv2, lpips, numpy, open3d, plyfile, scipy, torch; print("paintmesh inpaint imports: OK")'
validate_numeric_settings
FINETUNE_ITERATION="$(read_final_iteration | tail -n 1)"
EDGS_CONFIG="$(read_manifest_input_path "${REMOVED_MODEL_MANIFEST}" edgs_config | tail -n 1)"
INPAINTED_WORK_PLY="${WORK_MODEL}/point_cloud_object_inpaint_virtual/iteration_${FINETUNE_ITERATION}/point_cloud.ply"
INPAINTED_ITERATION_ROOT="${INPAINTED_GS_ROOT}/point_cloud/iteration_${FINETUNE_ITERATION}"
INPAINTED_PLY="${INPAINTED_ITERATION_ROOT}/point_cloud.ply"
INPAINTED_CLASSIFIER="${INPAINTED_ITERATION_ROOT}/classifier.pth"
INPAINTED_RAW_MESH="${INPAINTED_GS_ROOT}/mesh/ours_${FINETUNE_ITERATION}/tsdf_fusion_post.ply"
INPAINTED_RENDER_MANIFEST="${INPAINTED_GS_ROOT}/train/ours_${FINETUNE_ITERATION}/render_manifest.json"
INPAINTED_TEST_RENDER_MANIFEST="${INPAINTED_GS_ROOT}/test/ours_${FINETUNE_ITERATION}/render_manifest.json"
INPAINTED_MESH_STAGE_MANIFEST="${INPAINTED_GS_ROOT}/mesh/ours_${FINETUNE_ITERATION}/mesh_manifest.json"
INPAINTED_SEMANTIC_MANIFEST="${INPAINTED_MESH_ROOT}/semantic_manifest.json"

echo "PaintMesh object inpaint"
echo "Dataset               : ${SCENE_ROOT}"
echo "Removal run           : ${REMOVAL_ROOT}"
echo "Inpaint run           : ${INPAINT_RUN_ROOT}"
echo "Main conda environment: ${PAINTMESH_ENV}"
echo "LaMa environment      : ${LAMA_ENV}"
echo "Source/final iteration: ${DISTILL_ITERATION}/${FINETUNE_ITERATION}"
echo "Stages                : ${START_STAGE}..${END_STAGE}"

if should_run 1; then
    echo "[1/8] Validating removal/tracker artifacts and preparing workspace"
else
    echo "[1/8] Validating the existing inpaint workspace"
fi
run_inpaint tools/prepare_inpaint_workspace.py \
    --removal-workspace "${REMOVAL_WORK_MODEL}" \
    --removal-manifest "${REMOVAL_MANIFEST}" \
    --tracking-session "${TRACKING_SESSION}" \
    --camera-manifest "${VIRTUAL_CAMERA_MANIFEST}" \
    --tracking-masks "${TRACKING_MASK_ROOT}" \
    --source-iteration "${DISTILL_ITERATION}" \
    --output "${WORK_MODEL}"
require_manifest_kind "${WORKSPACE_MANIFEST}" "paintmesh-inpaint-workspace"

if (( END_STAGE >= 2 )); then
if should_run 2; then
    echo "[2/8] Preparing isolated LaMa RGB/depth inputs"
else
    echo "[2/8] Validating prepared LaMa inputs"
fi
run_inpaint tools/prepare_paintmesh_lama_data.py prepare \
    --tracking-masks "${WORK_MODEL}/tracking_masks" \
    --removed-rgb "${WORK_MODEL}/virtual/ours_object_removal/iteration_${DISTILL_ITERATION}/renders" \
    --removed-depth "${WORK_MODEL}/virtual/ours_object_removal/iteration_${DISTILL_ITERATION}/depth" \
    --reference-depth "${WORK_MODEL}/virtual/ours_${DISTILL_ITERATION}/depth" \
    --color-input "${LAMA_COLOR_INPUT}" \
    --depth-input "${LAMA_DEPTH_INPUT}" \
    --manifest "${LAMA_INPUT_MANIFEST}" \
    --frames 30 \
    --min-area "${MASK_MIN_AREA}" \
    --dilation "${MASK_DILATION}"
require_manifest_kind "${LAMA_INPUT_MANIFEST}" "paintmesh-lama-inputs"
fi

if (( END_STAGE >= 3 )); then
if should_run 3; then
    echo "[3/8] Completing and validating virtual RGB/depth with LaMa"
    if [[ -f "${LAMA_COMPLETION_MANIFEST}" ]]; then
        validate_lama_completion
        echo "      Reused manifest-matched LaMa outputs"
    else
        LAMA_ENV_PREFIX="$("${CONDA_BIN}" run --no-capture-output -n "${LAMA_ENV}" \
            python -c 'import sys; print(sys.prefix)')"
        run_lama -c \
            'import pytorch_lightning, webdataset; from saicinpainting.training.trainers import load_checkpoint; print("LaMa imports: OK")'
        color_arguments=(
            bin/predict_color.py
            --input-dir "${LAMA_COLOR_INPUT}"
            --output-dir "${LAMA_COLOR_OUTPUT}"
            --model-path "${LAMA_MODEL_PATH}"
        )
        if is_true "${RECURSIVE_GUIDE}"; then color_arguments+=(--recursive-guide); fi
        run_lama "${color_arguments[@]}"
        run_lama bin/predict_depth.py \
            --input-dir "${LAMA_DEPTH_INPUT}" \
            --output-dir "${LAMA_DEPTH_OUTPUT}" \
            --model-path "${LAMA_MODEL_PATH}"
        validate_lama_completion
    fi
else
    echo "[3/8] Validating completed LaMa RGB/depth"
    validate_lama_completion
fi
require_manifest_kind "${LAMA_COMPLETION_MANIFEST}" "paintmesh-lama-completion"
fi

if (( END_STAGE >= 4 )); then
if should_run 4; then
    echo "[4/8] Back-projecting the completed RGB-D virtual views"
    fusion_arguments=(
        edit_object_removal_plyfusion.py
        --source_path "${SCENE_ROOT}"
        --model_path "${WORK_MODEL}"
        --resolution "${RESOLUTION}"
        --iteration "${DISTILL_ITERATION}"
        --source_iteration "${DISTILL_ITERATION}"
        --config_file "${INPAINT_CONFIG}"
        --inpainted_color_dir "${LAMA_COLOR_OUTPUT}"
        --inpaint_mask_dir "${LAMA_COLOR_INPUT}"
        --completed_depth_dir "${LAMA_DEPTH_OUTPUT}"
        --fused_output_dir "${FUSED_ROOT}"
        --hole_output_dir "${HOLE_ROOT}"
        --camera_manifest "${VIRTUAL_CAMERA_MANIFEST}"
        --lama_manifest "${LAMA_COMPLETION_MANIFEST}"
        --manifest "${FUSION_MANIFEST}"
    )
    if ! is_true "${WRITE_HOLE_PLY}"; then fusion_arguments+=(--skip_hole_ply); fi
    run_inpaint "${fusion_arguments[@]}"
else
    echo "[4/8] Reusing RGB-D support point clouds"
fi
require_manifest_kind "${FUSION_MANIFEST}" "paintmesh-rgbd-fusion"
require_file "${SUPPORT_PLY}"
fi

if (( END_STAGE >= 5 )); then
if should_run 5; then
    echo "[5/8] Optimizing the inpainted object-aware 3DGS"
    inpaint_arguments=(
        edit_object_inpaint.py
        --source_path "${SCENE_ROOT}"
        --model_path "${WORK_MODEL}"
        --images "${EDGS_IMAGES}"
        --resolution "${RESOLUTION}"
        --iteration "${DISTILL_ITERATION}"
        --source_iteration "${DISTILL_ITERATION}"
        --config_file "${INPAINT_CONFIG}"
        --supp_ply "${SUPPORT_PLY}"
        --fusion_dir "${FUSED_ROOT}"
        --fusion_seed_frame "${FUSION_SEED_FRAME}"
        --inpaint_output_ply "${INPAINTED_WORK_PLY}"
        --inpainted_color_dir "${LAMA_COLOR_OUTPUT}"
        --inpaint_mask_dir "${LAMA_COLOR_INPUT}"
        --camera_manifest "${VIRTUAL_CAMERA_MANIFEST}"
        --removal_source_iteration "${DISTILL_ITERATION}"
        --skip_surrounding_filter
    )
    if ! is_true "${RENDER_INPAINT_TRAIN}"; then inpaint_arguments+=(--skip_train); fi
    if ! is_true "${RENDER_INPAINT_TEST}"; then inpaint_arguments+=(--skip_test); fi
    if is_true "${RENDER_INPAINT_VIDEO}"; then inpaint_arguments+=(--render_video); fi
    run_inpaint "${inpaint_arguments[@]}"
else
    echo "[5/8] Reusing optimized inpainted Gaussians"
fi
require_file "${INPAINTED_WORK_PLY}"
fi

if (( END_STAGE >= 6 )); then
if should_run 6; then
    echo "[6/8] Publishing an EDGS-loadable inpainted 3DGS"
    run_inpaint tools/publish_inpainted_edgs_model.py \
        --inpainted-ply "${INPAINTED_WORK_PLY}" \
        --classifier "${WORK_MODEL}/point_cloud/iteration_${DISTILL_ITERATION}/classifier.pth" \
        --edgs-config "${EDGS_CONFIG}" \
        --cfg-args "${WORK_MODEL}/cfg_args" \
        --source-iteration "${DISTILL_ITERATION}" \
        --output-iteration "${FINETUNE_ITERATION}" \
        --target-ids "${TARGET_IDS}" \
        --surrounding-ids "${SURROUNDING_IDS}" \
        --removed-model-manifest "${REMOVED_MODEL_MANIFEST}" \
        --removal-manifest "${REMOVAL_MANIFEST}" \
        --workspace-manifest "${WORKSPACE_MANIFEST}" \
        --tracking-session "${TRACKING_SESSION}" \
        --lama-manifest "${LAMA_COMPLETION_MANIFEST}" \
        --fusion-manifest "${FUSION_MANIFEST}" \
        --fusion-seed-frame "${FUSION_SEED_FRAME}" \
        --inpaint-config "${INPAINT_CONFIG}" \
        --output "${INPAINTED_GS_ROOT}"
else
    echo "[6/8] Reusing published inpainted 3DGS"
fi
require_manifest_kind "${INPAINTED_MODEL_MANIFEST}" "paintmesh-inpainted-edgs-model"
require_regular_file "${INPAINTED_PLY}"
require_file "${INPAINTED_CLASSIFIER}"
fi

if (( END_STAGE >= 7 )); then
if should_run 7; then
    echo "[7/8] Rendering with PGSR and reconstructing the inpainted TSDF mesh"
    mesh_reusable=false
    if [[ -f "${INPAINTED_RAW_MESH}" && \
          -f "${INPAINTED_RENDER_MANIFEST}" && \
          -f "${INPAINTED_MESH_STAGE_MANIFEST}" ]]; then
        if is_true "${RENDER_EDGS_TEST}"; then
            [[ -f "${INPAINTED_TEST_RENDER_MANIFEST}" ]] && mesh_reusable=true
        else
            mesh_reusable=true
        fi
    fi
    if is_true "${mesh_reusable}"; then
        publish_mesh_manifest
        echo "      Reused manifest-matched PGSR mesh"
    else
        edgs_arguments=(
            render.py
            --model-path "${INPAINTED_GS_ROOT}"
            --source-path "${SCENE_ROOT}"
            --images "${EDGS_IMAGES}"
            --resolution "${RESOLUTION}"
            --iteration "${FINETUNE_ITERATION}"
            --renderer pgsr
            --extract-mesh
            --max-depth "${MAX_DEPTH}"
            --voxel-size "${VOXEL_SIZE}"
            --num-clusters "${NUM_CLUSTERS}"
        )
        if ! is_true "${RENDER_EDGS_TEST}"; then edgs_arguments+=(--skip-test); fi
        if is_true "${USE_DEPTH_FILTER}"; then edgs_arguments+=(--use-depth-filter); fi
        run_edgs "${edgs_arguments[@]}"
        publish_mesh_manifest
    fi
else
    echo "[7/8] Reusing PGSR render and TSDF mesh"
fi
require_manifest_kind "${INPAINTED_MESH_STAGE_MANIFEST}" "paintmesh-pgsr-inpaint-mesh"
require_file "${INPAINTED_RAW_MESH}"
fi

if (( END_STAGE >= 8 )); then
if should_run 8; then
    echo "[8/8] Lifting semantics and committing the final inpaint result"
    semantic_reusable=false
    if [[ -f "${INPAINTED_SEMANTIC_MANIFEST}" ]]; then
        validate_semantic_reuse
        semantic_reusable=true
    fi
    if ! is_true "${semantic_reusable}"; then
        semantic_arguments=(
            tools/lift_gaussian_semantics_to_mesh.py
            --gaussian-ply "${INPAINTED_PLY}"
            --classifier "${INPAINTED_CLASSIFIER}"
            --scene-info "${SCENE_ROOT}/associated_hqsam/scene.json"
            --mesh "${INPAINTED_RAW_MESH}"
            --output-dir "${INPAINTED_MESH_ROOT}"
            --neighbors "${MESH_NEIGHBORS}"
            --chunk-size "${MESH_CHUNK_SIZE}"
            --workers "${MESH_WORKERS}"
            --opacity-min "${MESH_OPACITY_MIN}"
            --support-sigma "${MESH_SUPPORT_SIGMA}"
            --normal-power "${MESH_NORMAL_POWER}"
            --min-confidence "${MESH_MIN_CONFIDENCE}"
            --min-margin "${MESH_MIN_MARGIN}"
            --unknown-id "${MESH_UNKNOWN_ID}"
            --overwrite
        )
        if is_true "${WRITE_COLORED_MESH}"; then semantic_arguments+=(--write-colored-ply); fi
        run_inpaint "${semantic_arguments[@]}"
    fi
    ensure_geometry_link
    finalize_arguments=(
        tools/finalize_inpaint_result.py
        --model-manifest "${INPAINTED_MODEL_MANIFEST}"
        --mesh-manifest "${INPAINTED_MESH_STAGE_MANIFEST}"
        --semantic-manifest "${INPAINTED_SEMANTIC_MANIFEST}"
        --removal-manifest "${REMOVAL_MANIFEST}"
        --workspace-manifest "${WORKSPACE_MANIFEST}"
        --lama-manifest "${LAMA_COMPLETION_MANIFEST}"
        --fusion-manifest "${FUSION_MANIFEST}"
        --gaussian-ply "${INPAINTED_PLY}"
        --mesh "${INPAINTED_RAW_MESH}"
        --geometry "${INPAINTED_MESH_ROOT}/geometry.ply"
        --output "${FINAL_MANIFEST}"
    )
    run_inpaint "${finalize_arguments[@]}"
else
    echo "[8/8] Reusing semantic inpaint result"
fi
require_manifest_kind "${FINAL_MANIFEST}" "paintmesh-object-inpaint"
fi

echo "PaintMesh object inpainting completed."
echo "Inpainted 3DGS : ${INPAINTED_GS_ROOT}"
echo "Inpainted mesh : ${INPAINTED_MESH_ROOT}"
echo "Final manifest : ${FINAL_MANIFEST}"
