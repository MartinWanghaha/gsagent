#!/usr/bin/env bash

set -Eeuo pipefail

# PaintMesh object-removal pipeline.
#
# Usage:
#   scripts/paintmesh/run_remove \
#       [dataset_name] [scene] [resolution] target_ids \
#       [surrounding_ids] [start_stage]
#
# Example:
#   scripts/paintmesh/run_remove "mip-nerf/360_v2" kitchen 8 14 none 1
#
# Stages:
#   1 = initialize run-local configs and the Inpaint360GS work model
#   2 = remove Gaussians and publish an EDGS-loadable target-removed 3DGS
#   3 = render the removed 3DGS, extract its PGSR mesh, and lift semantics
#   4 = generate virtual views and a run-local tracker archive
#   5 = optionally launch interactive mask refinement

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INVOCATION_ROOT="$(pwd -P)"
EDGS_ROOT="${REPO_ROOT}/submodules/EDGS"
INPAINT_ROOT="${REPO_ROOT}/submodules/Inpaint360GS"
TRACKER_SOURCE_ROOT="${INPAINT_ROOT}/Segment-and-Track-Anything"

resolve_from_invocation() {
    local path="$1"

    if [[ "${path}" == /* ]]; then
        realpath -m -- "${path}"
    else
        realpath -m -- "${INVOCATION_ROOT}/${path}"
    fi
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  scripts/paintmesh/run_remove \
      [dataset_name] [scene] [resolution] target_ids \
      [surrounding_ids] [start_stage]

Examples:
  scripts/paintmesh/run_remove "mip-nerf/360_v2" kitchen 8 14 none 1
  GPU=0 scripts/paintmesh/run_remove \
      "mip-nerf/360_v2" kitchen 8 14 "10,24" 1
  END_STAGE=3 LAUNCH_REFINER=false scripts/paintmesh/run_remove \
      "mip-nerf/360_v2" kitchen 8 14 none 1

Arguments:
  target_ids       one positive instance ID or a comma-separated list
  surrounding_ids  temporary occluder IDs, or "none" (default)
  start_stage       first stage to execute, in [1, 5] (default: 1)

Stages:
  1 = run-local configs and Inpaint360GS work model
  2 = target-removed 3DGS
  3 = PGSR/TSDF removed mesh and semantic lifting
  4 = virtual views and tracker image archive
  5 = interactive mask refinement

Important environment overrides:
  PAINTMESH_ENV=paintmesh
  GPU=0
  END_STAGE=5
  DISTILL_ITERATION=2000
  REMOVAL_THRESHOLD=0.7
  RENDER_VIDEO=false
  RENDER_OBJECT_VIDEOS=false
  RENDER_REMOVAL_TRAIN=false
  RENDER_REMOVAL_TEST=false
  RENDER_EDGS_TEST=false
  WRITE_DEBUG_PLY=false
  LAUNCH_REFINER=true
  WRITE_COLORED_MESH=true
  RUN_NAME=target_14_t080
  REMOVAL_ROOT=/absolute/output/path
EOF
}

require_dir() {
    [[ -d "$1" ]] || fail "required directory does not exist: $1"
}

require_file() {
    [[ -f "$1" ]] || fail "required file does not exist: $1"
}

require_positive_integer() {
    local value="$1"
    local name="$2"

    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || \
        fail "${name} must be a canonical positive integer; got '${value}'"
}

require_boolean() {
    case "${1,,}" in
        1|true|yes|on|0|false|no|off)
            return 0
            ;;
        *)
            fail "expected a boolean value, got '$1'"
            ;;
    esac
}

is_true() {
    case "${1,,}" in
        1|true|yes|on)
            return 0
            ;;
        0|false|no|off)
            return 1
            ;;
        *)
            fail "expected a boolean value, got '$1'"
            ;;
    esac
}

normalize_id_list() {
    local raw="$1"
    local allow_none="$2"
    local description="$3"
    local output_name="$4"
    local item
    local joined
    local -a items normalized sorted
    local -A seen=()

    if is_true "${allow_none}" && [[ "${raw,,}" == "none" ]]; then
        printf -v "${output_name}" '%s' "none"
        return
    fi
    [[ -n "${raw}" ]] || fail "${description} cannot be empty"

    IFS=',' read -r -a items <<< "${raw}"
    for item in "${items[@]}"; do
        [[ "${item}" =~ ^[1-9][0-9]*$ ]] || \
            fail "${description} must contain canonical positive integers; got '${raw}'"
        [[ -z "${seen[${item}]+present}" ]] || \
            fail "${description} contains duplicate ID ${item}"
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
    local value
    local -a left_values right_values
    local -A selected=()

    [[ "${right}" != "none" ]] || return 1
    IFS=',' read -r -a left_values <<< "${left}"
    IFS=',' read -r -a right_values <<< "${right}"
    for value in "${left_values[@]}"; do
        selected["${value}"]=1
    done
    for value in "${right_values[@]}"; do
        [[ -z "${selected[${value}]+present}" ]] || return 0
    done
    return 1
}

should_run() {
    local stage="$1"
    (( START_STAGE <= stage && END_STAGE >= stage ))
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
    [[ -n "${scene}" && "${scene}" != "." && "${scene}" != ".." && "${scene}" != */* ]] || \
        fail "scene must be one safe path component; got '${scene}'"
}

DATA_ROOT="$(resolve_from_invocation "${DATA_ROOT:-${REPO_ROOT}/data}")"
CKPT_ROOT="$(resolve_from_invocation "${CKPT_ROOT:-${REPO_ROOT}/ckpt}")"
OUTPUT_ROOT="$(resolve_from_invocation "${OUTPUT_ROOT:-${REPO_ROOT}/output}")"
CONDA_BIN="${CONDA_BIN:-conda}"
if [[ "${CONDA_BIN}" == */* ]]; then
    CONDA_BIN="$(resolve_from_invocation "${CONDA_BIN}")"
fi
PAINTMESH_ENV="${PAINTMESH_ENV:-paintmesh}"
GPU="${GPU:-}"

DATASET_NAME="${1:-${DATASET_NAME:-inpaint360}}"
SCENE="${2:-${SCENE:-doppelherz}}"
RESOLUTION="${3:-${RESOLUTION:-2}}"
TARGET_IDS_RAW="${4:-${TARGET_IDS:-}}"
SURROUNDING_IDS_RAW="${5:-${SURROUNDING_IDS:-none}}"
START_STAGE="${6:-${START_STAGE:-1}}"
END_STAGE="${END_STAGE:-5}"

if [[ "${DATASET_NAME}" == "-h" || "${DATASET_NAME}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# > 6 )); then
    usage >&2
    exit 2
fi
validate_scene_key "${DATASET_NAME}" "${SCENE}"
[[ -n "${TARGET_IDS_RAW}" ]] || {
    usage >&2
    fail "target_ids is required"
}

case "${RESOLUTION}" in
    1|2|4|8)
        ;;
    *)
        fail "resolution must be one of 1, 2, 4, or 8; got '${RESOLUTION}'"
        ;;
esac
case "${START_STAGE}" in
    1|2|3|4|5)
        ;;
    *)
        fail "start_stage must be an integer from 1 to 5; got '${START_STAGE}'"
        ;;
esac
case "${END_STAGE}" in
    1|2|3|4|5)
        ;;
    *)
        fail "END_STAGE must be an integer from 1 to 5; got '${END_STAGE}'"
        ;;
esac
(( START_STAGE <= END_STAGE )) || \
    fail "start_stage (${START_STAGE}) cannot be greater than END_STAGE (${END_STAGE})"

normalize_id_list "${TARGET_IDS_RAW}" false "target_ids" TARGET_IDS
normalize_id_list \
    "${SURROUNDING_IDS_RAW}" true "surrounding_ids" SURROUNDING_IDS
if ids_overlap "${TARGET_IDS}" "${SURROUNDING_IDS}"; then
    fail "target_ids and surrounding_ids must be disjoint"
fi

TARGET_KEY="${TARGET_IDS//,/-}"
if [[ -z "${RUN_NAME+x}" ]]; then
    RUN_NAME="target_${TARGET_KEY}"
    if [[ "${SURROUNDING_IDS}" != "none" ]]; then
        RUN_NAME+="__surrounding_${SURROUNDING_IDS//,/-}"
    fi
fi
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    fail "RUN_NAME may contain only letters, digits, '.', '_' and '-'"

DISTILL_ITERATION="${DISTILL_ITERATION:-2000}"
EDGS_IMAGES="${EDGS_IMAGES:-images}"
REMOVAL_THRESHOLD="${REMOVAL_THRESHOLD:-0.7}"
RENDER_VIDEO="${RENDER_VIDEO:-false}"
RENDER_OBJECT_VIDEOS="${RENDER_OBJECT_VIDEOS:-false}"
RENDER_REMOVAL_TRAIN="${RENDER_REMOVAL_TRAIN:-false}"
RENDER_REMOVAL_TEST="${RENDER_REMOVAL_TEST:-false}"
RENDER_EDGS_TEST="${RENDER_EDGS_TEST:-false}"
WRITE_DEBUG_PLY="${WRITE_DEBUG_PLY:-false}"
LAUNCH_REFINER="${LAUNCH_REFINER:-true}"
WRITE_COLORED_MESH="${WRITE_COLORED_MESH:-true}"
USE_DEPTH_FILTER="${USE_DEPTH_FILTER:-false}"

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
EDGS_MODEL_ROOT="$(resolve_from_invocation "${EDGS_MODEL_ROOT:-${PIPELINE_ROOT}/edgs}")"
EDGS_BRIDGE_ROOT="$(resolve_from_invocation "${EDGS_BRIDGE_ROOT:-${PIPELINE_ROOT}/edgs_bridge}")"
SEMANTIC_GS_ROOT="$(resolve_from_invocation "${SEMANTIC_GS_ROOT:-${PIPELINE_ROOT}/semantic_3dgs}")"
SEMANTIC_MESH_ROOT="$(resolve_from_invocation "${SEMANTIC_MESH_ROOT:-${PIPELINE_ROOT}/semantic_mesh}")"
REMOVAL_ROOT="$(resolve_from_invocation "${REMOVAL_ROOT:-${PIPELINE_ROOT}/removal/${RUN_NAME}}")"

SCENE_ROOT="${DATA_ROOT}/${DATASET_NAME}/${SCENE}"
CONFIG_ROOT="${REMOVAL_ROOT}/config"
WORK_MODEL_ROOT="${REMOVAL_ROOT}/work_model"
REMOVED_GS_ROOT="${REMOVAL_ROOT}/removed_3dgs"
REMOVED_MESH_ROOT="${REMOVAL_ROOT}/removed_mesh"
TRACKER_RUN_ROOT="${REMOVAL_ROOT}/tracker"
TRACKER_ARCHIVE="${TRACKER_RUN_ROOT}/images.zip"
VIRTUAL_CAMERA_MANIFEST="${TRACKER_RUN_ROOT}/virtual_cameras.json"
TRACKER_ASSETS_ROOT="${TRACKER_RUN_ROOT}/assets"
TRACKING_RESULTS_ROOT="${TRACKER_RUN_ROOT}/results"
TRACKING_MASK_ROOT="${TRACKING_RESULTS_ROOT}/images/images_masks"
TRACKING_SESSION_MANIFEST="${TRACKER_RUN_ROOT}/tracking_session.json"

REMOVAL_CONFIG="${CONFIG_ROOT}/object_removal/${DATASET_NAME}/${SCENE}.json"
INPAINT_CONFIG="${CONFIG_ROOT}/object_inpaint/${DATASET_NAME}/${SCENE}.json"
BRIDGE_MANIFEST="${EDGS_BRIDGE_ROOT}/bridge_manifest.json"
BASE_SEMANTIC_MANIFEST="${SEMANTIC_MESH_ROOT}/semantic_manifest.json"
SCENE_INFO="${SCENE_ROOT}/associated_hqsam/scene.json"
SEMANTIC_ITERATION_ROOT="${SEMANTIC_GS_ROOT}/point_cloud/iteration_${DISTILL_ITERATION}"
SEMANTIC_PLY="${SEMANTIC_ITERATION_ROOT}/point_cloud.ply"
CLASSIFIER_PATH="${SEMANTIC_ITERATION_ROOT}/classifier.pth"
WORKSPACE_MANIFEST="${WORK_MODEL_ROOT}/workspace_manifest.json"

ALL_SELECTED_REMOVED_ROOT="${WORK_MODEL_ROOT}/point_cloud_object_removal/iteration_${DISTILL_ITERATION}"
TARGET_ONLY_REMOVED_ROOT="${WORK_MODEL_ROOT}/point_cloud_object_removal/iteration_${DISTILL_ITERATION}_removal_target"
ALL_SELECTED_REMOVED_PLY="${ALL_SELECTED_REMOVED_ROOT}/point_cloud.ply"
if [[ "${SURROUNDING_IDS}" == "none" ]]; then
    FINAL_REMOVED_PLY="${ALL_SELECTED_REMOVED_PLY}"
else
    FINAL_REMOVED_PLY="${TARGET_ONLY_REMOVED_ROOT}/point_cloud.ply"
fi

REMOVED_ITERATION_ROOT="${REMOVED_GS_ROOT}/point_cloud/iteration_${DISTILL_ITERATION}"
REMOVED_GS_PLY="${REMOVED_ITERATION_ROOT}/point_cloud.ply"
REMOVED_CLASSIFIER="${REMOVED_ITERATION_ROOT}/classifier.pth"
REMOVED_MODEL_MANIFEST="${REMOVED_GS_ROOT}/model_manifest.json"
REMOVED_RAW_MESH="${REMOVED_GS_ROOT}/mesh/ours_${DISTILL_ITERATION}/tsdf_fusion_post.ply"
REMOVED_RENDER_MANIFEST="${REMOVED_GS_ROOT}/train/ours_${DISTILL_ITERATION}/render_manifest.json"
REMOVED_TEST_RENDER_MANIFEST="${REMOVED_GS_ROOT}/test/ours_${DISTILL_ITERATION}/render_manifest.json"
REMOVED_MESH_STAGE_MANIFEST="${REMOVED_GS_ROOT}/mesh/ours_${DISTILL_ITERATION}/mesh_manifest.json"
REMOVED_SEMANTIC_MANIFEST="${REMOVED_MESH_ROOT}/semantic_manifest.json"
REMOVAL_MANIFEST="${REMOVAL_ROOT}/removal_manifest.json"

run_python() {
    local workdir="$1"
    local pythonpath="$2"
    shift 2

    local -a environment=("PYTHONPATH=${pythonpath}")
    if [[ -n "${GPU}" ]]; then
        environment+=("CUDA_VISIBLE_DEVICES=${GPU}")
    fi

    (
        cd "${workdir}"
        env "${environment[@]}" \
            "${CONDA_BIN}" run --no-capture-output \
            -n "${PAINTMESH_ENV}" python "$@"
    )
}

run_inpaint() {
    run_python \
        "${INPAINT_ROOT}" \
        "${INPAINT_ROOT}:${INPAINT_ROOT}/seg/detectron2" \
        "$@"
}

run_edgs() {
    run_python "${EDGS_ROOT}" "${EDGS_ROOT}" "$@"
}

require_complete_manifest() {
    local manifest="$1"

    require_file "${manifest}"
    run_inpaint - "${manifest}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("complete") is not True:
    raise SystemExit(f"manifest is not complete: {path}")
print(f"Complete manifest: {path}")
PY
}

validate_numeric_configuration() {
    run_inpaint - \
        "${REMOVAL_THRESHOLD}" \
        "${MAX_DEPTH}" \
        "${VOXEL_SIZE}" \
        "${MESH_OPACITY_MIN}" \
        "${MESH_SUPPORT_SIGMA}" \
        "${MESH_NORMAL_POWER}" \
        "${MESH_MIN_CONFIDENCE}" \
        "${MESH_MIN_MARGIN}" <<'PY'
import math
import sys

names = (
    "REMOVAL_THRESHOLD",
    "MAX_DEPTH",
    "VOXEL_SIZE",
    "MESH_OPACITY_MIN",
    "MESH_SUPPORT_SIGMA",
    "MESH_NORMAL_POWER",
    "MESH_MIN_CONFIDENCE",
    "MESH_MIN_MARGIN",
)
values = {name: float(raw) for name, raw in zip(names, sys.argv[1:])}
if not all(math.isfinite(value) for value in values.values()):
    raise SystemExit("all floating-point settings must be finite")
if not 0.0 <= values["REMOVAL_THRESHOLD"] <= 1.0:
    raise SystemExit("REMOVAL_THRESHOLD must be in [0, 1]")
if values["MAX_DEPTH"] <= 0 or values["VOXEL_SIZE"] <= 0:
    raise SystemExit("MAX_DEPTH and VOXEL_SIZE must be positive")
if not 0.0 <= values["MESH_OPACITY_MIN"] <= 1.0:
    raise SystemExit("MESH_OPACITY_MIN must be in [0, 1]")
if values["MESH_SUPPORT_SIGMA"] <= 0:
    raise SystemExit("MESH_SUPPORT_SIGMA must be positive")
if values["MESH_NORMAL_POWER"] < 0:
    raise SystemExit("MESH_NORMAL_POWER must be non-negative")
for name in ("MESH_MIN_CONFIDENCE", "MESH_MIN_MARGIN"):
    if not 0.0 <= values[name] <= 1.0:
        raise SystemExit(f"{name} must be in [0, 1]")
print("Removal numeric configuration: OK")
PY
}

validate_config_contract() {
    run_inpaint - \
        "${REMOVAL_CONFIG}" \
        "${INPAINT_CONFIG}" \
        "${TARGET_IDS}" \
        "${SURROUNDING_IDS}" \
        "${REMOVAL_THRESHOLD}" <<'PY'
import json
import math
from pathlib import Path
import sys

removal_path = Path(sys.argv[1])
inpaint_path = Path(sys.argv[2])
target_ids = [int(value) for value in sys.argv[3].split(",")]
surrounding_ids = (
    [] if sys.argv[4] == "none" else [int(value) for value in sys.argv[4].split(",")]
)
threshold = float(sys.argv[5])
expected = {
    "target_id": target_ids,
    "surrounding_ids": surrounding_ids,
    "select_obj_id": target_ids + surrounding_ids,
}

for path in (removal_path, inpaint_path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SystemExit(
                f"config contract mismatch in {path}: {key}={payload.get(key)!r}, "
                f"expected {value!r}"
            )
    actual_threshold = payload.get("removal_thresh")
    if isinstance(actual_threshold, bool) or not isinstance(actual_threshold, (int, float)):
        raise SystemExit(f"config removal_thresh is not numeric: {path}")
    if not math.isclose(float(actual_threshold), threshold, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"config threshold mismatch in {path}: {actual_threshold}, expected {threshold}"
        )
print("Removal config contract: OK")
PY
}

validate_input_contract() {
    run_inpaint - \
        "${BRIDGE_MANIFEST}" \
        "${BASE_SEMANTIC_MANIFEST}" \
        "${SCENE_INFO}" \
        "${SCENE_ROOT}" \
        "${SEMANTIC_PLY}" \
        "${CLASSIFIER_PATH}" \
        "${SEMANTIC_MESH_ROOT}/gaussian_instance_id.npy" \
        "${TARGET_IDS}" \
        "${SURROUNDING_IDS}" \
        "${RESOLUTION}" \
        "${MESH_UNKNOWN_ID}" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np

(
    bridge_path,
    semantic_manifest_path,
    scene_path,
    scene_root,
    semantic_ply,
    classifier,
    gaussian_labels_path,
) = map(Path, sys.argv[1:8])
target_ids = [int(value) for value in sys.argv[8].split(",")]
surrounding_ids = (
    [] if sys.argv[9] == "none" else [int(value) for value in sys.argv[9].split(",")]
)
resolution = int(sys.argv[10])
unknown_id = int(sys.argv[11])

bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
semantic = json.loads(semantic_manifest_path.read_text(encoding="utf-8"))
scene = json.loads(scene_path.read_text(encoding="utf-8"))
if bridge.get("complete") is not True or semantic.get("complete") is not True:
    raise SystemExit("run_seg bridge and semantic manifests must both be complete")

bridge_source_text = bridge.get("dataset", {}).get("source_path")
if not isinstance(bridge_source_text, str) or not bridge_source_text:
    raise SystemExit("bridge manifest has no dataset.source_path")
bridge_source = Path(bridge_source_text).resolve(strict=True)
if bridge_source != scene_root.resolve(strict=True):
    raise SystemExit(
        f"scene/bridge mismatch: requested {scene_root.resolve()}, "
        f"bridge records {bridge_source}"
    )
bridge_resolution = bridge.get("dataset", {}).get("resolution")
if isinstance(bridge_resolution, bool) or bridge_resolution != resolution:
    raise SystemExit(
        f"resolution/bridge mismatch: requested {resolution}, "
        f"run_seg bridge records {bridge_resolution!r}"
    )

num_classes = int(scene["num_classes"])
if unknown_id < num_classes or unknown_id > 2**32 - 1:
    raise SystemExit(
        f"MESH_UNKNOWN_ID must be in [{num_classes}, {2**32 - 1}], got {unknown_id}"
    )
for class_id in target_ids + surrounding_ids:
    if not 0 < class_id < num_classes:
        raise SystemExit(
            f"instance ID {class_id} is outside the valid range [1, {num_classes - 1}]"
        )

inputs = semantic.get("inputs", {})
expected = {
    "gaussian_ply": semantic_ply.resolve(),
    "classifier": classifier.resolve(),
    "scene_info": scene_path.resolve(),
}
for name, expected_path in expected.items():
    record = inputs.get(name, {})
    actual = Path(record.get("path", "")).resolve()
    if actual != expected_path:
        raise SystemExit(
            f"semantic manifest {name} mismatch: actual={actual}, expected={expected_path}"
        )
    stat = expected_path.stat()
    for key, observed in (
        ("size_bytes", int(stat.st_size)),
        ("mtime_ns", int(stat.st_mtime_ns)),
    ):
        recorded = record.get(key)
        if recorded is not None and recorded != observed:
            raise SystemExit(
                f"semantic manifest {name}.{key}={recorded}, expected {observed}"
            )

labels = np.load(gaussian_labels_path, mmap_mode="r")
expected_count = int(semantic.get("counts", {}).get("gaussians", -1))
if labels.ndim != 1 or len(labels) != expected_count:
    raise SystemExit(
        f"Gaussian label sidecar mismatch: shape={labels.shape}, expected={expected_count}"
    )
for class_id in target_ids + surrounding_ids:
    count = int(np.count_nonzero(labels == class_id))
    if count < 4:
        raise SystemExit(
            f"instance ID {class_id} has only {count} labeled Gaussians; "
            "at least four are required for 3D removal"
        )
    role = "target" if class_id in target_ids else "surrounding"
    print(f"{role} instance {class_id}: {count} semantic Gaussians")
PY
}

validate_tracker_archive() {
    run_inpaint - "${TRACKER_ARCHIVE}" <<'PY'
from io import BytesIO
from pathlib import Path
import sys
import zipfile

from PIL import Image

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"tracker archive is missing: {path}")
with zipfile.ZipFile(path) as archive:
    members = [info for info in archive.infolist() if not info.is_dir()]
    names = [info.filename for info in members]
    expected = [f"{index:05d}.png" for index in range(30)]
    if sorted(names) != expected:
        raise SystemExit(
            "tracker archive must contain exactly 00000.png..00029.png at its root; "
            f"found {names}"
        )
    dimensions = []
    for info in members:
        with Image.open(BytesIO(archive.read(info))) as image:
            image.verify()
        with Image.open(BytesIO(archive.read(info))) as image:
            dimensions.append(image.size)
if not dimensions or any(size != dimensions[0] for size in dimensions):
    raise SystemExit(f"tracker archive images have inconsistent dimensions: {dimensions}")
if dimensions[0][0] <= 0 or dimensions[0][1] <= 0:
    raise SystemExit("tracker archive contains an empty image")
print(f"Tracker archive: 30 images at {dimensions[0][0]}x{dimensions[0][1]}")
PY
}

begin_tracking_session() {
    run_inpaint - \
        "${TRACKING_SESSION_MANIFEST}" \
        "${TRACKER_ARCHIVE}" \
        "${VIRTUAL_CAMERA_MANIFEST}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

manifest_path = Path(sys.argv[1])
archive_path = Path(sys.argv[2]).resolve(strict=True)
camera_path = Path(sys.argv[3]).resolve(strict=True)
stat = archive_path.stat()
digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
camera_stat = camera_path.stat()
camera_payload = json.loads(camera_path.read_text(encoding="utf-8"))
if (
    camera_payload.get("kind") != "inpaint360gs-virtual-cameras"
    or camera_payload.get("complete") is not True
    or camera_payload.get("frame_count") != 30
):
    raise SystemExit(f"virtual-camera manifest is incomplete or invalid: {camera_path}")
payload = {
    "schema_version": 2,
    "kind": "paintmesh-tracking-session",
    "complete": False,
    "status": "in_progress",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "started_at_ns": time.time_ns(),
    "input_archive": {
        "path": str(archive_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    },
    "input_cameras": {
        "path": str(camera_path),
        "size_bytes": int(camera_stat.st_size),
        "mtime_ns": int(camera_stat.st_mtime_ns),
        "sha256": hashlib.sha256(camera_path.read_bytes()).hexdigest(),
        "artifact_id": camera_payload.get("artifact_id"),
    },
    "expected_masks": [f"{index:05d}.png" for index in range(30)],
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    dir=manifest_path.parent, prefix=".tracking-session-", suffix=".json"
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)
finally:
    temporary.unlink(missing_ok=True)
print(f"Tracking session: {manifest_path}")
PY
}

validate_tracking_masks() {
    run_inpaint - \
        "${TRACKING_MASK_ROOT}" \
        "${TRACKER_ARCHIVE}" \
        "${VIRTUAL_CAMERA_MANIFEST}" \
        "${TRACKING_SESSION_MANIFEST}" <<'PY'
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from PIL import Image

root = Path(sys.argv[1])
archive_path = Path(sys.argv[2])
camera_path = Path(sys.argv[3])
session_path = Path(sys.argv[4])
session = json.loads(session_path.read_text(encoding="utf-8"))
if session.get("kind") != "paintmesh-tracking-session":
    raise SystemExit(f"unexpected tracking-session manifest: {session_path}")
if session.get("complete") is not False or session.get("status") != "in_progress":
    raise SystemExit(f"tracking session is not active: {session_path}")
archive_record = session.get("input_archive", {})
archive_stat = archive_path.resolve(strict=True).stat()
archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
expected_archive = {
    "path": str(archive_path.resolve(strict=True)),
    "size_bytes": int(archive_stat.st_size),
    "mtime_ns": int(archive_stat.st_mtime_ns),
    "sha256": archive_digest,
}
if archive_record != expected_archive:
    raise SystemExit("tracker archive changed after the interactive session started")
camera_stat = camera_path.resolve(strict=True).stat()
expected_cameras = {
    "path": str(camera_path.resolve(strict=True)),
    "size_bytes": int(camera_stat.st_size),
    "mtime_ns": int(camera_stat.st_mtime_ns),
    "sha256": hashlib.sha256(camera_path.read_bytes()).hexdigest(),
    "artifact_id": json.loads(camera_path.read_text(encoding="utf-8")).get(
        "artifact_id"
    ),
}
if session.get("input_cameras") != expected_cameras:
    raise SystemExit("virtual cameras changed after the interactive session started")

expected_names = [f"{index:05d}.png" for index in range(30)]
if session.get("expected_masks") != expected_names:
    raise SystemExit("tracking session has an unexpected mask contract")
missing = [name for name in expected_names if not (root / name).is_file()]
if missing:
    raise SystemExit(f"refined tracker masks are missing from {root}: {missing}")

with zipfile.ZipFile(archive_path) as archive:
    source_sizes = {}
    for name in expected_names:
        with Image.open(BytesIO(archive.read(name))) as image:
            source_sizes[name] = image.size

mask_records = []
started_at_ns = int(session["started_at_ns"])
for name in expected_names:
    mask_path = root / name
    mask_stat = mask_path.stat()
    if mask_stat.st_mtime_ns < started_at_ns:
        raise SystemExit(
            f"refined mask predates the current tracking session: {mask_path}"
        )
    with Image.open(mask_path) as mask:
        mask.verify()
    with Image.open(mask_path) as mask:
        if mask.size != source_sizes[name]:
            raise SystemExit(
                f"refined mask size mismatch for {name}: {mask.size}, "
                f"expected {source_sizes[name]}"
            )
    mask_records.append(
        {
            "file": name,
            "size_bytes": int(mask_stat.st_size),
            "mtime_ns": int(mask_stat.st_mtime_ns),
            "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        }
    )

session["complete"] = True
session["status"] = "complete"
session["completed_at"] = datetime.now(timezone.utc).isoformat()
session["outputs"] = {"masks": mask_records}
descriptor, temporary_name = tempfile.mkstemp(
    dir=session_path.parent, prefix=".tracking-session-", suffix=".json"
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(session, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, session_path)
finally:
    temporary.unlink(missing_ok=True)
print(f"Refined tracker masks: {len(expected_names)} exact frame masks")
PY
}

tracking_session_matches() {
    local expected_state="$1"

    [[ -f "${TRACKING_SESSION_MANIFEST}" && \
       -f "${TRACKER_ARCHIVE}" && \
       -f "${VIRTUAL_CAMERA_MANIFEST}" ]] || return 1
    if [[ "${expected_state}" == "complete" ]]; then
        [[ -d "${TRACKING_MASK_ROOT}" ]] || return 1
    fi

    run_inpaint - \
        "${expected_state}" \
        "${TRACKING_SESSION_MANIFEST}" \
        "${TRACKER_ARCHIVE}" \
        "${VIRTUAL_CAMERA_MANIFEST}" \
        "${TRACKING_MASK_ROOT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

expected_state = sys.argv[1]
session_path = Path(sys.argv[2])
archive_path = Path(sys.argv[3])
camera_path = Path(sys.argv[4])
mask_root = Path(sys.argv[5])

try:
    if expected_state not in {"in_progress", "complete"}:
        raise ValueError("unsupported tracking state")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    archive_path = archive_path.resolve(strict=True)
    archive_stat = archive_path.stat()
    expected_archive = {
        "path": str(archive_path),
        "size_bytes": int(archive_stat.st_size),
        "mtime_ns": int(archive_stat.st_mtime_ns),
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    camera_path = camera_path.resolve(strict=True)
    camera_stat = camera_path.stat()
    camera_payload = json.loads(camera_path.read_text(encoding="utf-8"))
    expected_cameras = {
        "path": str(camera_path),
        "size_bytes": int(camera_stat.st_size),
        "mtime_ns": int(camera_stat.st_mtime_ns),
        "sha256": hashlib.sha256(camera_path.read_bytes()).hexdigest(),
        "artifact_id": camera_payload.get("artifact_id"),
    }
    expected_names = [f"{index:05d}.png" for index in range(30)]
    if (
        session.get("schema_version") != 2
        or session.get("kind") != "paintmesh-tracking-session"
        or session.get("input_archive") != expected_archive
        or session.get("input_cameras") != expected_cameras
        or session.get("expected_masks") != expected_names
    ):
        raise ValueError("tracking-session contract mismatch")

    if expected_state == "in_progress":
        if (
            session.get("complete") is not False
            or session.get("status") != "in_progress"
            or not isinstance(session.get("started_at_ns"), int)
            or session["started_at_ns"] <= 0
        ):
            raise ValueError("tracking session is not resumable")
        raise SystemExit(0)

    if session.get("complete") is not True or session.get("status") != "complete":
        raise ValueError("tracking session is incomplete")
    records = session["outputs"]["masks"]
    if [record.get("file") for record in records] != expected_names:
        raise ValueError("tracked mask list mismatch")
    for record in records:
        mask_path = mask_root / record["file"]
        stat = mask_path.stat()
        current = {
            "file": record["file"],
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        }
        if current != record:
            raise ValueError("tracked mask changed")
except (
    AttributeError,
    KeyError,
    OSError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
):
    raise SystemExit(1)
PY
}

tracking_session_is_active() {
    tracking_session_matches in_progress
}

tracking_session_is_complete() {
    tracking_session_matches complete
}

mesh_stage_contract() {
    local mode="$1"

    run_inpaint - \
        "${mode}" \
        "${REMOVED_MESH_STAGE_MANIFEST}" \
        "${REMOVED_MODEL_MANIFEST}" \
        "${REMOVED_GS_PLY}" \
        "${REMOVED_RAW_MESH}" \
        "${REMOVED_RENDER_MANIFEST}" \
        "${REMOVED_TEST_RENDER_MANIFEST}" \
        "${DISTILL_ITERATION}" \
        "${SCENE_ROOT}" \
        "${EDGS_IMAGES}" \
        "${RESOLUTION}" \
        "${MAX_DEPTH}" \
        "${VOXEL_SIZE}" \
        "${NUM_CLUSTERS}" \
        "${USE_DEPTH_FILTER}" \
        "${RENDER_EDGS_TEST}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

(
    mode,
    manifest_text,
    model_manifest_text,
    gaussian_text,
    mesh_text,
    render_manifest_text,
    test_render_manifest_text,
    iteration_text,
    source_text,
    images,
    resolution_text,
    max_depth_text,
    voxel_size_text,
    clusters_text,
    depth_filter_text,
    render_test_text,
) = sys.argv[1:]
if mode not in {"write", "validate"}:
    raise SystemExit(f"unsupported mesh contract mode: {mode}")

manifest_path = Path(manifest_text)
model_manifest_path = Path(model_manifest_text)
gaussian_path = Path(gaussian_text)
mesh_path = Path(mesh_text)
render_manifest_path = Path(render_manifest_text)
test_render_manifest_path = Path(test_render_manifest_text)

def load_complete(path, label):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise SystemExit(f"{label} is incomplete: {path}")
    return payload

def artifact(path):
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SystemExit(f"artifact is missing or empty: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }

def parse_bool(raw):
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"invalid boolean: {raw}")

model = load_complete(model_manifest_path, "removed model manifest")
render = load_complete(render_manifest_path, "EDGS train render manifest")
if render.get("backend") != "pgsr":
    raise SystemExit("removed mesh requires a PGSR render manifest")
iteration = int(iteration_text)
if int(render.get("iteration", -1)) != iteration or render.get("split") != "train":
    raise SystemExit("EDGS render manifest does not describe the requested train split")

model_artifact_id = model.get("artifact_id")
if not isinstance(model_artifact_id, str) or not model_artifact_id:
    raise SystemExit("removed model manifest has no artifact_id")
parameters = {
    "iteration": iteration,
    "renderer": "pgsr",
    "source_path": str(Path(source_text).resolve(strict=True)),
    "images": images,
    "resolution": int(resolution_text),
    "max_depth": float(max_depth_text),
    "voxel_size": float(voxel_size_text),
    "num_clusters": int(clusters_text),
    "use_depth_filter": parse_bool(depth_filter_text),
    "render_test": parse_bool(render_test_text),
}
test_render_artifact = None
if parameters["render_test"]:
    test_render = load_complete(test_render_manifest_path, "EDGS test render manifest")
    if (
        test_render.get("backend") != "pgsr"
        or int(test_render.get("iteration", -1)) != iteration
        or test_render.get("split") != "test"
    ):
        raise SystemExit("EDGS test render manifest does not match the requested run")
    test_render_artifact = artifact(test_render_manifest_path)
gaussian = artifact(gaussian_path)
mesh = artifact(mesh_path)
render_artifact = artifact(render_manifest_path)
identity_payload = {
    "kind": "paintmesh-pgsr-removal-mesh",
    "schema_version": 1,
    "model_artifact_id": model_artifact_id,
    "parameters": parameters,
    "gaussian": gaussian,
    "mesh": mesh,
    "render_manifest": render_artifact,
    "test_render_manifest": test_render_artifact,
}
artifact_id = hashlib.sha256(
    json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

if mode == "write":
    payload = {
        "schema_version": 1,
        "kind": "paintmesh-pgsr-removal-mesh",
        "complete": True,
        "status": "complete",
        "artifact_id": artifact_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "model_manifest": {
                "path": str(model_manifest_path.resolve(strict=True)),
                "artifact_id": model_artifact_id,
            },
            "gaussian_ply": gaussian,
        },
        "parameters": parameters,
        "outputs": {
            "mesh": mesh,
            "render_manifest": render_artifact,
            "test_render_manifest": test_render_artifact,
        },
        "counts": {"train_views": int(render.get("num_views", -1))},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=manifest_path.parent, prefix=".mesh-manifest-", suffix=".json"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)

manifest = load_complete(manifest_path, "removed mesh-stage manifest")
if manifest.get("kind") != "paintmesh-pgsr-removal-mesh":
    raise SystemExit(f"unexpected mesh-stage manifest kind: {manifest_path}")
if manifest.get("artifact_id") != artifact_id:
    raise SystemExit(
        "mesh-stage contract differs from the requested model or extraction settings"
    )
if manifest.get("parameters") != parameters:
    raise SystemExit("mesh-stage parameters do not match the current invocation")
print(f"Mesh-stage contract: {manifest_path}")
PY
}

invalidate_manifest() {
    local manifest="$1"
    local kind="$2"

    run_inpaint - "${manifest}" "${kind}" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "kind": sys.argv[2],
    "complete": False,
    "status": "in_progress",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
finally:
    temporary.unlink(missing_ok=True)
PY
}

link_tracker_checkpoint() {
    local checkpoint_name="$1"
    local source_path="${CKPT_ROOT}/${checkpoint_name}"
    local checkpoint_root="${TRACKER_SOURCE_ROOT}/ckpt"
    local target_path="${checkpoint_root}/${checkpoint_name}"

    require_file "${source_path}"
    mkdir -p "${checkpoint_root}"
    if [[ -L "${target_path}" && ! -e "${target_path}" ]]; then
        fail "tracker checkpoint symlink is broken: ${target_path}"
    fi
    if [[ -e "${target_path}" ]]; then
        [[ "$(realpath -- "${target_path}")" == "$(realpath -- "${source_path}")" ]] || \
            fail "tracker checkpoint points to a different CKPT_ROOT: ${target_path}"
    else
        local relative_source
        relative_source="$(realpath --relative-to="${checkpoint_root}" "${source_path}")"
        ln -s "${relative_source}" "${target_path}"
    fi
    require_file "${target_path}"
}

link_removed_geometry() {
    local link_path="${REMOVED_MESH_ROOT}/geometry.ply"
    local relative_target

    require_file "${REMOVED_RAW_MESH}"
    mkdir -p "${REMOVED_MESH_ROOT}"
    relative_target="$(realpath --relative-to="${REMOVED_MESH_ROOT}" "${REMOVED_RAW_MESH}")"
    if [[ -L "${link_path}" ]]; then
        [[ "$(readlink -- "${link_path}")" == "${relative_target}" ]] || \
            fail "removed mesh geometry link points to an unexpected target: ${link_path}"
    elif [[ -e "${link_path}" ]]; then
        fail "cannot create geometry link over an existing path: ${link_path}"
    else
        ln -s "${relative_target}" "${link_path}"
    fi
}

removal_contract() {
    local mode="$1"

    run_inpaint - \
        "${mode}" \
        "${REMOVAL_MANIFEST}" \
        "${WORKSPACE_MANIFEST}" \
        "${REMOVED_MODEL_MANIFEST}" \
        "${REMOVED_MESH_STAGE_MANIFEST}" \
        "${REMOVED_SEMANTIC_MANIFEST}" \
        "${FINAL_REMOVED_PLY}" \
        "${REMOVED_GS_PLY}" \
        "${REMOVED_RAW_MESH}" \
        "${TARGET_IDS}" \
        "${SURROUNDING_IDS}" \
        "${DISTILL_ITERATION}" \
        "${REMOVAL_THRESHOLD}" \
        "${MESH_NEIGHBORS}" \
        "${MESH_CHUNK_SIZE}" \
        "${MESH_WORKERS}" \
        "${MESH_OPACITY_MIN}" \
        "${MESH_SUPPORT_SIGMA}" \
        "${MESH_NORMAL_POWER}" \
        "${MESH_MIN_CONFIDENCE}" \
        "${MESH_MIN_MARGIN}" \
        "${MESH_UNKNOWN_ID}" \
        "${WRITE_COLORED_MESH}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from plyfile import PlyData

(
    mode,
    output_path,
    workspace_manifest,
    model_manifest,
    mesh_stage_manifest,
    semantic_manifest,
    work_ply,
    published_ply,
    mesh_path,
) = (sys.argv[1], *map(Path, sys.argv[2:10]))
if mode not in {"write", "validate"}:
    raise SystemExit(f"unsupported removal contract mode: {mode}")
target_ids = [int(value) for value in sys.argv[10].split(",")]
surrounding_ids = (
    [] if sys.argv[11] == "none" else [int(value) for value in sys.argv[11].split(",")]
)
iteration = int(sys.argv[12])
threshold = float(sys.argv[13])

def parse_bool(raw):
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"invalid boolean: {raw}")

semantic_parameters = {
    "neighbors_requested": int(sys.argv[14]),
    "chunk_size": int(sys.argv[15]),
    "workers": int(sys.argv[16]),
    "opacity_min": float(sys.argv[17]),
    "support_sigma": float(sys.argv[18]),
    "normal_power": float(sys.argv[19]),
    "min_confidence": float(sys.argv[20]),
    "min_margin": float(sys.argv[21]),
    "unknown_id": int(sys.argv[22]),
    "write_colored_ply": parse_bool(sys.argv[23]),
}

def load_complete(path, label):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise SystemExit(f"{label} is incomplete: {path}")
    return payload

def artifact(path):
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file() or stat.st_size <= 0:
        raise SystemExit(f"artifact is missing or empty: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }

def verify_record(record, path, label):
    current = artifact(path)
    for key, value in current.items():
        if record.get(key) != value:
            raise SystemExit(
                f"{label} changed: {key}={record.get(key)!r}, expected {value!r}"
            )
    return current

manifests = {}
for name, path in (
    ("workspace", workspace_manifest),
    ("model", model_manifest),
    ("mesh_stage", mesh_stage_manifest),
    ("semantic", semantic_manifest),
):
    manifests[name] = load_complete(path, f"{name} manifest")

for name in ("workspace", "model", "mesh_stage"):
    artifact_id = manifests[name].get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise SystemExit(f"{name} manifest has no artifact_id")

model_parameters = manifests["model"].get("parameters", {})
expected_model_parameters = {
    "iteration": iteration,
    "target_ids": target_ids,
    "surrounding_ids": surrounding_ids,
    "removal_threshold": threshold,
}
if model_parameters != expected_model_parameters:
    raise SystemExit(
        f"removed model parameters differ: {model_parameters!r}, "
        f"expected {expected_model_parameters!r}"
    )

actual_semantic_parameters = manifests["semantic"].get("parameters", {})
for key, value in semantic_parameters.items():
    if actual_semantic_parameters.get(key) != value:
        raise SystemExit(
            f"semantic mesh parameter {key}={actual_semantic_parameters.get(key)!r}, "
            f"expected {value!r}"
        )

semantic_inputs = manifests["semantic"].get("inputs", {})
for input_name, input_record in semantic_inputs.items():
    recorded_path = input_record.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise SystemExit(f"semantic input {input_name} has no path")
    verify_record(input_record, Path(recorded_path), f"semantic input {input_name}")
verify_record(semantic_inputs.get("gaussian_ply", {}), published_ply, "semantic Gaussian")
verify_record(semantic_inputs.get("mesh", {}), mesh_path, "semantic mesh geometry")

semantic_counts = manifests["semantic"].get("counts", {})
semantic_outputs = manifests["semantic"].get("outputs", {})
sidecar_counts = {
    "gaussian_label": int(semantic_counts["gaussians"]),
    "gaussian_confidence": int(semantic_counts["gaussians"]),
    "vertex_label": int(semantic_counts["mesh_vertices"]),
    "vertex_confidence": int(semantic_counts["mesh_vertices"]),
    "face_label": int(semantic_counts["mesh_triangles"]),
    "face_confidence": int(semantic_counts["mesh_triangles"]),
}
for output_name, expected_count in sidecar_counts.items():
    record = semantic_outputs.get(output_name, {})
    filename = record.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise SystemExit(f"semantic output {output_name} has an unsafe file name")
    output_file = semantic_manifest.parent / filename
    current = artifact(output_file)
    if record.get("size_bytes") != current["size_bytes"]:
        raise SystemExit(f"semantic output size mismatch: {output_file}")
    array = np.load(output_file, mmap_mode="r")
    if tuple(array.shape) != (expected_count,):
        raise SystemExit(
            f"semantic output shape mismatch: {output_file}: {array.shape}, "
            f"expected {(expected_count,)}"
        )
    if record.get("shape") != [expected_count] or record.get("dtype") != str(array.dtype):
        raise SystemExit(f"semantic output manifest metadata mismatch: {output_file}")
    del array

for output_name in ("palette",):
    record = semantic_outputs.get(output_name, {})
    filename = record.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise SystemExit(f"semantic output {output_name} has an unsafe file name")
    current = artifact(semantic_manifest.parent / filename)
    if record.get("size_bytes") != current["size_bytes"]:
        raise SystemExit(f"semantic output size mismatch: {current['path']}")

work_artifact = artifact(work_ply)
published_artifact = artifact(published_ply)
mesh_artifact = artifact(mesh_path)
mesh_parameters = manifests["mesh_stage"].get("parameters")
parameters = {
    "distill_iteration": iteration,
    "removal_threshold": threshold,
    "mesh_extraction": mesh_parameters,
    "semantic_lifting": semantic_parameters,
}

gaussian_count = len(PlyData.read(published_ply, mmap=True)["vertex"].data)
semantic_payload = manifests["semantic"]
counts = {
    "remaining_gaussians": gaussian_count,
    "mesh_vertices": int(semantic_payload["counts"]["mesh_vertices"]),
    "mesh_triangles": int(semantic_payload["counts"]["mesh_triangles"]),
}
inputs = {
    "workspace_manifest": {
        "path": str(workspace_manifest.resolve(strict=True)),
        "artifact_id": manifests["workspace"]["artifact_id"],
    },
    "model_manifest": {
        "path": str(model_manifest.resolve(strict=True)),
        "artifact_id": manifests["model"]["artifact_id"],
    },
    "mesh_stage_manifest": {
        "path": str(mesh_stage_manifest.resolve(strict=True)),
        "artifact_id": manifests["mesh_stage"]["artifact_id"],
    },
    "work_removed_ply": work_artifact,
}
outputs = {
    "removed_3dgs": published_artifact,
    "removed_mesh": mesh_artifact,
    "semantic_mesh_root": str(semantic_manifest.parent.resolve(strict=True)),
}
identity_payload = {
    "kind": "paintmesh-object-removal",
    "schema_version": 1,
    "target_ids": target_ids,
    "surrounding_ids": surrounding_ids,
    "parameters": parameters,
    "counts": counts,
    "inputs": inputs,
    "outputs": outputs,
}
artifact_id = hashlib.sha256(
    json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

payload = {
    "schema_version": 1,
    "kind": "paintmesh-object-removal",
    "complete": True,
    "status": "complete",
    "artifact_id": artifact_id,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "target_ids": target_ids,
    "surrounding_ids": surrounding_ids,
    "parameters": parameters,
    "counts": counts,
    "inputs": inputs,
    "outputs": outputs,
}

if mode == "write":
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=".removal-",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

recorded = load_complete(output_path, "removal manifest")
if recorded.get("kind") != "paintmesh-object-removal":
    raise SystemExit(f"unexpected removal manifest kind: {output_path}")
if recorded.get("artifact_id") != artifact_id:
    raise SystemExit(
        "removal manifest differs from the requested IDs, inputs, or parameters"
    )
if semantic_parameters["write_colored_ply"]:
    colored_record = semantic_outputs.get("semantic_mesh", {})
    colored_name = colored_record.get("file")
    if not isinstance(colored_name, str) or Path(colored_name).name != colored_name:
        raise SystemExit("semantic_mesh output has an unsafe file name")
    colored = semantic_manifest.parent / "semantic_mesh.ply"
    if not colored.is_file() or colored.stat().st_size <= 0:
        raise SystemExit(f"colored semantic mesh is missing: {colored}")
    if colored.resolve() != (semantic_manifest.parent / colored_name).resolve():
        raise SystemExit("semantic_mesh output path does not match its manifest")
    if colored_record.get("size_bytes") != colored.stat().st_size:
        raise SystemExit(f"semantic_mesh output size mismatch: {colored}")
elif (
    "semantic_mesh" in semantic_outputs
    or (semantic_manifest.parent / "semantic_mesh.ply").exists()
):
    raise SystemExit("colored semantic mesh exists although WRITE_COLORED_MESH is false")
print(f"Removal contract: {output_path}")
PY
}

run_tracker() {
    local -a environment=(
        "PYTHONPATH=${TRACKER_SOURCE_ROOT}"
        "TRACKER_IMAGE_SEQUENCE=${TRACKER_ARCHIVE}"
        "TRACKER_ASSETS_ROOT=${TRACKER_ASSETS_ROOT}"
        "TRACKING_RESULTS_ROOT=${TRACKING_RESULTS_ROOT}"
        "GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME:-127.0.0.1}"
        "GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT:-7860}"
        "GRADIO_SHARE=${GRADIO_SHARE:-false}"
        "PYTHONUNBUFFERED=1"
    )
    if [[ -n "${GPU}" ]]; then
        environment+=("CUDA_VISIBLE_DEVICES=${GPU}")
    fi

    (
        cd "${TRACKER_SOURCE_ROOT}"
        env "${environment[@]}" \
            "${CONDA_BIN}" run --no-capture-output \
            -n "${PAINTMESH_ENV}" python -u app.py
    )
}

trap 'echo "Error: paintmesh run_remove failed at line ${LINENO}." >&2' ERR

require_positive_integer "${DISTILL_ITERATION}" "DISTILL_ITERATION"
require_positive_integer "${NUM_CLUSTERS}" "NUM_CLUSTERS"
require_positive_integer "${MESH_NEIGHBORS}" "MESH_NEIGHBORS"
require_positive_integer "${MESH_CHUNK_SIZE}" "MESH_CHUNK_SIZE"
if [[ "${MESH_WORKERS}" != "-1" ]]; then
    require_positive_integer "${MESH_WORKERS}" "MESH_WORKERS"
fi
[[ "${MESH_UNKNOWN_ID}" =~ ^[0-9]+$ ]] || \
    fail "MESH_UNKNOWN_ID must be a canonical non-negative integer"
require_boolean "${RENDER_VIDEO}"
require_boolean "${RENDER_OBJECT_VIDEOS}"
require_boolean "${RENDER_REMOVAL_TRAIN}"
require_boolean "${RENDER_REMOVAL_TEST}"
require_boolean "${RENDER_EDGS_TEST}"
require_boolean "${WRITE_DEBUG_PLY}"
require_boolean "${LAUNCH_REFINER}"
require_boolean "${WRITE_COLORED_MESH}"
require_boolean "${USE_DEPTH_FILTER}"

require_dir "${EDGS_ROOT}"
require_dir "${INPAINT_ROOT}"
require_dir "${TRACKER_SOURCE_ROOT}"
require_dir "${SCENE_ROOT}"
if [[ "${EDGS_IMAGES}" == /* ]]; then
    require_dir "${EDGS_IMAGES}"
else
    require_dir "${SCENE_ROOT}/${EDGS_IMAGES}"
fi
require_dir "${SEMANTIC_GS_ROOT}"
require_dir "${SEMANTIC_MESH_ROOT}"
require_file "${EDGS_MODEL_ROOT}/config.yaml"
require_file "${SEMANTIC_GS_ROOT}/cfg_args"
require_file "${SEMANTIC_PLY}"
require_file "${CLASSIFIER_PATH}"
require_file "${SCENE_INFO}"
require_file "${BRIDGE_MANIFEST}"
require_file "${BASE_SEMANTIC_MANIFEST}"
require_file "${SEMANTIC_MESH_ROOT}/gaussian_instance_id.npy"
command -v "${CONDA_BIN}" >/dev/null || \
    fail "conda executable not found: ${CONDA_BIN}"

mkdir -p "${REMOVAL_ROOT}" "${CONFIG_ROOT}" "${TRACKER_RUN_ROOT}"

echo "PaintMesh environment : ${PAINTMESH_ENV}"
echo "Dataset               : ${SCENE_ROOT}"
echo "Semantic 3DGS          : ${SEMANTIC_GS_ROOT}"
echo "Semantic mesh          : ${SEMANTIC_MESH_ROOT}"
echo "Removal root           : ${REMOVAL_ROOT}"
echo "Target IDs             : ${TARGET_IDS}"
echo "Surrounding IDs        : ${SURROUNDING_IDS}"
echo "Resolution             : ${RESOLUTION}"
echo "Distill iteration      : ${DISTILL_ITERATION}"
echo "Stages                 : ${START_STAGE}..${END_STAGE}"

run_inpaint -c \
    'import numpy, plyfile, scipy, torch; print("paintmesh removal imports: OK")'
run_edgs -c \
    'import diff_plane_rasterization, omegaconf, open3d, torch; print("paintmesh EDGS imports: OK")'
validate_numeric_configuration
validate_input_contract

if should_run 1; then
    echo "[1/5] Initializing isolated removal configuration and work model"
    run_inpaint tools/init_configs.py \
        --dataset_name "${DATASET_NAME}" \
        --scene "${SCENE}" \
        --target_id "${TARGET_IDS}" \
        --target_surronding_id "${SURROUNDING_IDS}" \
        --removal_thresh "${REMOVAL_THRESHOLD}" \
        --output_root "${CONFIG_ROOT}"
else
    echo "[1/5] Reusing isolated removal configuration"
fi
require_file "${REMOVAL_CONFIG}"
require_file "${INPAINT_CONFIG}"
validate_config_contract

# Preparing the workspace is cheap, idempotent, and validates that a resumed
# run is still bound to the same semantic reconstruction.
run_inpaint tools/prepare_removal_workspace.py \
    --semantic-model "${SEMANTIC_GS_ROOT}" \
    --iteration "${DISTILL_ITERATION}" \
    --bridge-manifest "${BRIDGE_MANIFEST}" \
    --semantic-manifest "${BASE_SEMANTIC_MANIFEST}" \
    --output "${WORK_MODEL_ROOT}"
require_complete_manifest "${WORKSPACE_MANIFEST}"

if (( END_STAGE >= 2 )); then
    if should_run 2; then
        echo "[2/5] Removing selected Gaussians"
        if [[ -f "${FINAL_REMOVED_PLY}" && -f "${REMOVED_MODEL_MANIFEST}" ]]; then
            if is_true "${RENDER_VIDEO}" || \
               is_true "${RENDER_OBJECT_VIDEOS}" || \
               is_true "${RENDER_REMOVAL_TRAIN}" || \
               is_true "${RENDER_REMOVAL_TEST}" || \
               is_true "${WRITE_DEBUG_PLY}"; then
                fail "Stage 2 diagnostics cannot be added to an immutable published run; choose a new RUN_NAME or REMOVAL_ROOT"
            fi
            echo "      Reusing a published removed Gaussian checkpoint"
        else
            removal_arguments=(
                edit_object_removal.py
                --source_path "${SCENE_ROOT}"
                --model_path "${WORK_MODEL_ROOT}"
                --reference_model_path "${SEMANTIC_GS_ROOT}"
                --iteration "${DISTILL_ITERATION}"
                --resolution "${RESOLUTION}"
                --config_file "${REMOVAL_CONFIG}"
            )
            if is_true "${RENDER_REMOVAL_TRAIN}"; then
                require_dir "${SEMANTIC_GS_ROOT}/train/ours_${DISTILL_ITERATION}/depth"
            fi
            if is_true "${RENDER_REMOVAL_TEST}"; then
                require_dir "${SEMANTIC_GS_ROOT}/test/ours_${DISTILL_ITERATION}/depth"
            fi
            if ! is_true "${RENDER_REMOVAL_TRAIN}"; then
                removal_arguments+=(--skip_train)
            fi
            if ! is_true "${RENDER_REMOVAL_TEST}"; then
                removal_arguments+=(--skip_test)
            fi
            if is_true "${RENDER_VIDEO}"; then
                removal_arguments+=(--render_video)
            fi
            if is_true "${RENDER_OBJECT_VIDEOS}"; then
                removal_arguments+=(--render_object_videos)
            fi
            if ! is_true "${WRITE_DEBUG_PLY}"; then
                removal_arguments+=(--skip_debug_ply)
            fi
            run_inpaint "${removal_arguments[@]}"
        fi
    else
        echo "[2/5] Reusing target-removed 3DGS"
        require_complete_manifest "${REMOVED_MODEL_MANIFEST}"
    fi

    require_file "${ALL_SELECTED_REMOVED_PLY}"
    require_file "${FINAL_REMOVED_PLY}"
    run_inpaint tools/publish_removed_edgs_model.py \
        --removed-ply "${FINAL_REMOVED_PLY}" \
        --classifier "${CLASSIFIER_PATH}" \
        --edgs-config "${EDGS_MODEL_ROOT}/config.yaml" \
        --cfg-args "${SEMANTIC_GS_ROOT}/cfg_args" \
        --iteration "${DISTILL_ITERATION}" \
        --target-ids "${TARGET_IDS}" \
        --surrounding-ids "${SURROUNDING_IDS}" \
        --bridge-manifest "${BRIDGE_MANIFEST}" \
        --semantic-manifest "${BASE_SEMANTIC_MANIFEST}" \
        --removal-threshold "${REMOVAL_THRESHOLD}" \
        --output "${REMOVED_GS_ROOT}"
    require_complete_manifest "${REMOVED_MODEL_MANIFEST}"
    require_file "${REMOVED_GS_PLY}"
    require_file "${REMOVED_CLASSIFIER}"
fi

if should_run 3; then
    echo "[3/5] Extracting and labeling the target-removed PGSR mesh"
    invalidate_manifest "${REMOVAL_MANIFEST}" "paintmesh-object-removal"
    mesh_stage_is_reusable=false
    if [[ -f "${REMOVED_RAW_MESH}" && \
          -f "${REMOVED_RENDER_MANIFEST}" && \
          -f "${REMOVED_MESH_STAGE_MANIFEST}" ]] && \
       mesh_stage_contract validate; then
        mesh_stage_is_reusable=true
    fi
    if is_true "${mesh_stage_is_reusable}"; then
        echo "      Reusing contract-matched PGSR TSDF mesh: ${REMOVED_RAW_MESH}"
    else
        invalidate_manifest \
            "${REMOVED_MESH_STAGE_MANIFEST}" \
            "paintmesh-pgsr-removal-mesh"
        mesh_arguments=(
            render.py
            --model-path "${REMOVED_GS_ROOT}"
            --iteration "${DISTILL_ITERATION}"
            --renderer pgsr
            --source-path "${SCENE_ROOT}"
            --images "${EDGS_IMAGES}"
            --resolution "${RESOLUTION}"
            --extract-mesh
            --max-depth "${MAX_DEPTH}"
            --voxel-size "${VOXEL_SIZE}"
            --num-clusters "${NUM_CLUSTERS}"
        )
        if ! is_true "${RENDER_EDGS_TEST}"; then
            mesh_arguments+=(--skip-test)
        fi
        if is_true "${USE_DEPTH_FILTER}"; then
            mesh_arguments+=(--use-depth-filter)
        fi
        run_edgs "${mesh_arguments[@]}"
        require_file "${REMOVED_RAW_MESH}"
        require_file "${REMOVED_RENDER_MANIFEST}"
        mesh_stage_contract write
    fi
    require_file "${REMOVED_RAW_MESH}"
    mesh_stage_contract validate

    semantic_mesh_arguments=(
        tools/lift_gaussian_semantics_to_mesh.py
        --gaussian-ply "${REMOVED_GS_PLY}"
        --classifier "${REMOVED_CLASSIFIER}"
        --scene-info "${SCENE_INFO}"
        --mesh "${REMOVED_RAW_MESH}"
        --output-dir "${REMOVED_MESH_ROOT}"
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
    if is_true "${WRITE_COLORED_MESH}"; then
        semantic_mesh_arguments+=(--write-colored-ply)
    fi
    run_inpaint "${semantic_mesh_arguments[@]}"
    require_complete_manifest "${REMOVED_SEMANTIC_MANIFEST}"
    link_removed_geometry
    removal_contract write
    removal_contract validate
elif (( START_STAGE > 3 && END_STAGE >= 4 )); then
    echo "[3/5] Reusing target-removed mesh"
    mesh_stage_contract validate
    removal_contract validate
    link_removed_geometry
fi

if should_run 4; then
    echo "[4/5] Generating isolated virtual views for mask refinement"
    invalidate_manifest \
        "${TRACKING_SESSION_MANIFEST}" \
        "paintmesh-tracking-session"
    run_inpaint tools/virtual_pose.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${WORK_MODEL_ROOT}" \
        --iteration "${DISTILL_ITERATION}" \
        --resolution "${RESOLUTION}" \
        --config_file "${REMOVAL_CONFIG}" \
        --tracker_archive "${TRACKER_ARCHIVE}" \
        --camera_manifest "${VIRTUAL_CAMERA_MANIFEST}"
    validate_tracker_archive
    begin_tracking_session
else
    echo "[4/5] Skipping virtual-view generation"
fi

if should_run 5; then
    if is_true "${LAUNCH_REFINER}"; then
        echo "[5/5] Launching isolated interactive mask refinement"
        validate_tracker_archive
        if tracking_session_is_complete >/dev/null 2>&1; then
            echo "      Reusing 30 archive-matched refined masks"
        elif validate_tracking_masks >/dev/null 2>&1; then
            # A previous Gradio run may have produced all valid masks before
            # `conda run` translated Ctrl+C into exit status 1.  Commit the
            # artifacts first so resuming Stage 5 never relaunches needlessly.
            echo "      Committed 30 session-matched refined masks"
        else
            require_dir "${CKPT_ROOT}"
            link_tracker_checkpoint "sam_vit_b_01ec64.pth"
            link_tracker_checkpoint "R50_DeAOTL_PRE_YTB_DAV.pth"
            link_tracker_checkpoint "groundingdino_swint_ogc.pth"
            mkdir -p "${TRACKER_ASSETS_ROOT}" "${TRACKING_RESULTS_ROOT}"

            if ! tracking_session_is_active >/dev/null 2>&1; then
                begin_tracking_session
            else
                echo "      Resuming the archive-matched tracking session"
            fi
            echo "      Local interface: http://${GRADIO_SERVER_NAME:-127.0.0.1}:${GRADIO_SERVER_PORT:-7860}"
            echo "      Export the 30 masks, then stop the interface with Ctrl+C."
            tracker_status=0
            run_tracker || tracker_status=$?
            case "${tracker_status}" in
                0|1|130) ;;
                *) fail "interactive tracker exited with status ${tracker_status}" ;;
            esac
            if ! validate_tracking_masks; then
                fail "interactive tracker exited with status ${tracker_status} and did not produce 30 valid masks"
            fi
            if (( tracker_status == 1 )); then
                echo "      Accepted Conda's Ctrl+C status 1 after full mask validation"
            fi
        fi
    else
        echo "[5/5] Interactive mask refinement disabled (LAUNCH_REFINER=false)"
    fi
else
    echo "[5/5] Skipping interactive mask refinement"
fi

if should_run 5 && ! is_true "${LAUNCH_REFINER}"; then
    echo "PaintMesh core removal completed; optional Stage 5 was disabled."
else
    echo "PaintMesh object removal completed through stage ${END_STAGE}."
fi
if [[ -f "${REMOVED_GS_PLY}" ]]; then
    echo "Removed 3DGS : ${REMOVED_GS_ROOT}"
fi
if [[ -f "${REMOVED_MESH_ROOT}/semantic_manifest.json" ]]; then
    echo "Removed mesh : ${REMOVED_MESH_ROOT}"
fi
if tracking_session_is_complete >/dev/null 2>&1; then
    echo "Refined masks: ${TRACKING_MASK_ROOT}"
fi
