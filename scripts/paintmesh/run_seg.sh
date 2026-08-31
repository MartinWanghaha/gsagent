#!/usr/bin/env bash

set -Eeuo pipefail

# EDGS-PGSR + Inpaint360GS semantic-instance pipeline.
#
# Usage:
#   scripts/paintmesh/run_seg \
#       [dataset_name] [scene] [resolution] [start_stage]
#
# Examples:
#   scripts/paintmesh/run_seg "mip-nerf/360_v2" kitchen 8 1
#   EDGS_MODEL_ROOT=/absolute/path/to/an/edgs/run \
#       scripts/paintmesh/run_seg "mip-nerf/360_v2" kitchen 8 2
#
# start_stage:
#   1 = train EDGS-PGSR and extract the base TSDF mesh
#   2 = generate per-view CropFormer masks
#   3 = associate instance IDs across views through the EDGS Gaussians
#   4 = distill object embeddings into the EDGS Gaussians
#   5 = render the semantic/object-aware 3DGS
#   6 = lift Gaussian instance embeddings onto the PGSR mesh
#
# Runtime configuration is provided through environment variables. Important
# overrides include PAINTMESH_ENV, DATA_ROOT, OUTPUT_ROOT, EDGS_MODEL_ROOT,
# BASE_ITERATION, DISTILL_ITERATION, GPU, PGSR_DEBUG and WRITE_COLORED_MESH.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INVOCATION_ROOT="$(pwd -P)"
EDGS_ROOT="${REPO_ROOT}/submodules/EDGS"
INPAINT_ROOT="${REPO_ROOT}/submodules/Inpaint360GS"

resolve_from_invocation() {
    local path="$1"

    if [[ "${path}" == /* ]]; then
        realpath -m -- "${path}"
    else
        realpath -m -- "${INVOCATION_ROOT}/${path}"
    fi
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
START_STAGE="${4:-${START_STAGE:-1}}"

BASE_ITERATION="${BASE_ITERATION:-30000}"
DISTILL_ITERATION="${DISTILL_ITERATION:-2000}"
EDGS_IMAGES="${EDGS_IMAGES:-images}"
NO_DENSIFY="${NO_DENSIFY:-true}"
PGSR_DEBUG="${PGSR_DEBUG:-false}"
PGSR_DEBUG_INTERVAL="${PGSR_DEBUG_INTERVAL:-200}"
PGSR_DEBUG_FROM_ITER="${PGSR_DEBUG_FROM_ITER:-auto}"
PGSR_DEBUG_OUTPUT_DIR="${PGSR_DEBUG_OUTPUT_DIR:-debug}"
PGSR_DEBUG_JPEG_QUALITY="${PGSR_DEBUG_JPEG_QUALITY:-95}"
SEG_THRESHOLD="${SEG_THRESHOLD:-0.5}"
ASSOCIATION_PATCH="${ASSOCIATION_PATCH:-16}"
MAX_DEPTH="${MAX_DEPTH:-5.0}"
VOXEL_SIZE="${VOXEL_SIZE:-0.002}"
NUM_CLUSTERS="${NUM_CLUSTERS:-1}"
USE_DEPTH_FILTER="${USE_DEPTH_FILTER:-false}"
RENDER_VIDEO="${RENDER_VIDEO:-false}"
WRITE_COLORED_MESH="${WRITE_COLORED_MESH:-true}"

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

SCENE_ROOT="${DATA_ROOT}/${DATASET_NAME}/${SCENE}"
DISTILL_CONFIG="$(resolve_from_invocation "${DISTILL_CONFIG:-${INPAINT_ROOT}/config/object_distill/train_distill.json}")"
SEG_WEIGHT_ROOT="${INPAINT_ROOT}/seg/weight"

EDGS_PLY="${EDGS_MODEL_ROOT}/point_cloud/iteration_${BASE_ITERATION}/point_cloud.ply"
EDGS_MESH="${EDGS_MODEL_ROOT}/mesh/ours_${BASE_ITERATION}/tsdf_fusion_post.ply"
EDGS_DEBUG_ROOT="${EDGS_MODEL_ROOT}/${PGSR_DEBUG_OUTPUT_DIR}"
SEMANTIC_ITERATION_ROOT="${SEMANTIC_GS_ROOT}/point_cloud/iteration_${DISTILL_ITERATION}"
SEMANTIC_PLY="${SEMANTIC_ITERATION_ROOT}/point_cloud.ply"
CLASSIFIER_PATH="${SEMANTIC_ITERATION_ROOT}/classifier.pth"
SCENE_INFO="${SCENE_ROOT}/associated_hqsam/scene.json"

usage() {
    cat <<'EOF'
Usage:
  scripts/paintmesh/run_seg \
      [dataset_name] [scene] [resolution] [start_stage]

Examples:
  scripts/paintmesh/run_seg "mip-nerf/360_v2" kitchen 8 1
  EDGS_MODEL_ROOT=/absolute/path/to/an/edgs/run \
      scripts/paintmesh/run_seg "mip-nerf/360_v2" kitchen 8 2

start_stage:
  1 = train EDGS-PGSR and extract the base TSDF mesh
  2 = generate per-view CropFormer masks
  3 = associate instance IDs across views through the EDGS Gaussians
  4 = distill object embeddings into the EDGS Gaussians
  5 = render the semantic/object-aware 3DGS
  6 = lift Gaussian instance embeddings onto the PGSR mesh

Optional Stage 1 PGSR training visualization:
  PGSR_DEBUG=true                 write PGSR-style 2x4 JPEG montages
  PGSR_DEBUG_INTERVAL=200         capture interval in absolute GS steps
  PGSR_DEBUG_FROM_ITER=auto       auto follows the multi-view loss threshold
  PGSR_DEBUG_OUTPUT_DIR=debug     directory relative to EDGS_MODEL_ROOT
  PGSR_DEBUG_JPEG_QUALITY=95      JPEG quality in [1, 100]

This visualization is not gs.pipe.debug.  It records training images and does
not enable CUDA rasterizer crash snapshots.
EOF
}

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

require_positive_integer() {
    local value="$1"
    local name="$2"

    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || \
        fail "${name} must be a positive integer; got '${value}'"
}

require_nonnegative_integer() {
    local value="$1"
    local name="$2"

    [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || \
        fail "${name} must be a canonical non-negative integer; got '${value}'"
}

require_integer_range() {
    local value="$1"
    local name="$2"
    local minimum="$3"
    local maximum="$4"

    require_nonnegative_integer "${value}" "${name}"
    (( 10#${value} >= minimum && 10#${value} <= maximum )) || \
        fail "${name} must be in [${minimum}, ${maximum}]; got '${value}'"
}

require_relative_subdirectory() {
    local value="$1"
    local name="$2"
    local segment
    local -a segments

    [[ -n "${value}" && "${value}" != /* && "${value}" != */ ]] || \
        fail "${name} must be a non-empty path relative to EDGS_MODEL_ROOT"
    [[ "${value}" =~ ^[A-Za-z0-9._/-]+$ ]] || \
        fail "${name} may contain only letters, digits, '.', '_', '-', and '/'"
    IFS='/' read -r -a segments <<< "${value}"
    for segment in "${segments[@]}"; do
        [[ -n "${segment}" && "${segment}" != "." && "${segment}" != ".." ]] || \
            fail "${name} cannot contain empty, '.' or '..' path segments"
    done
}

require_directory_within() {
    local directory="$1"
    local root="$2"
    local description="$3"
    local resolved_directory
    local resolved_root

    require_dir "${directory}"
    resolved_directory="$(realpath -- "${directory}")"
    resolved_root="$(realpath -- "${root}")"
    [[ "${resolved_directory}" == "${resolved_root}"/* ]] || \
        fail "${description} resolves outside ${root}: ${resolved_directory}"
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

run_edgs() {
    run_python "${EDGS_ROOT}" "${EDGS_ROOT}" "$@"
}

run_inpaint() {
    run_python \
        "${INPAINT_ROOT}" \
        "${INPAINT_ROOT}:${INPAINT_ROOT}/seg/detectron2" \
        "$@"
}

link_checkpoint_if_missing() {
    local checkpoint_name="$1"
    local source_path="${CKPT_ROOT}/${checkpoint_name}"
    local target_path="${SEG_WEIGHT_ROOT}/${checkpoint_name}"

    require_file "${source_path}"
    if [[ -L "${target_path}" && ! -e "${target_path}" ]]; then
        fail "checkpoint symlink is broken: ${target_path}"
    fi
    if [[ ! -e "${target_path}" ]]; then
        ln -s "${source_path}" "${target_path}"
    fi
    require_file "${target_path}"
}

require_colmap_model() {
    local sparse_root="$1"

    if [[ -f "${sparse_root}/cameras.bin" && \
          -f "${sparse_root}/images.bin" && \
          -f "${sparse_root}/points3D.bin" ]]; then
        return
    fi
    if [[ -f "${sparse_root}/cameras.txt" && \
          -f "${sparse_root}/images.txt" && \
          -f "${sparse_root}/points3D.txt" ]]; then
        return
    fi
    fail "COLMAP model must contain a complete cameras/images/points3D .bin set or a complete .txt set under ${sparse_root}"
}

validate_mask_pairs() {
    local image_root="$1"
    local mask_root="$2"
    local description="$3"

    run_inpaint - "${image_root}" "${mask_root}" "${description}" <<'PY'
from pathlib import Path
import sys

import numpy as np
from PIL import Image

image_root = Path(sys.argv[1])
mask_root = Path(sys.argv[2])
description = sys.argv[3]
extensions = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

def index_by_stem(paths, kind):
    indexed = {}
    duplicates = {}
    for path in paths:
        if path.stem in indexed:
            duplicates.setdefault(path.stem, [indexed[path.stem]]).append(path)
        else:
            indexed[path.stem] = path
    if duplicates:
        details = ", ".join(
            f"{stem}={[str(path) for path in choices]}"
            for stem, choices in sorted(duplicates.items())
        )
        raise SystemExit(f"{description}: duplicate {kind} stems: {details}")
    return indexed


images = index_by_stem(
    (
        path
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ),
    "image",
)
masks = index_by_stem(
    (path for path in mask_root.glob("*.png") if path.is_file()),
    "mask",
)

if not images:
    raise SystemExit(f"{description}: no input images found in {image_root}")
missing = sorted(images.keys() - masks.keys())
stale = sorted(masks.keys() - images.keys())
if missing or stale:
    raise SystemExit(
        f"{description}: image/mask stem mismatch; "
        f"missing={missing[:8]}, stale={stale[:8]}"
    )

for stem, image_path in images.items():
    mask_path = masks[stem]
    with Image.open(image_path) as image:
        expected_size = image.size
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image)
        actual_size = mask_image.size
    if actual_size != expected_size:
        raise SystemExit(
            f"{description}: size mismatch for {stem}: "
            f"image={expected_size}, mask={actual_size}"
        )
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise SystemExit(
            f"{description}: {mask_path} must be a 2D integer label image; "
            f"got shape={mask.shape}, dtype={mask.dtype}"
        )

print(f"{description}: validated {len(images)} image/mask pairs")
PY
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

validate_bridge_contract() {
    local manifest="$1"

    run_inpaint - \
        "${manifest}" \
        "${SCENE_ROOT}" \
        "${EDGS_IMAGES}" \
        "${RESOLUTION}" \
        "${BASE_ITERATION}" <<'PY'
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
expected_source = Path(sys.argv[2]).resolve()
expected_images = sys.argv[3]
expected_resolution = int(sys.argv[4])
expected_iteration = int(sys.argv[5])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))

actual_source = Path(payload["dataset"]["source_path"]).resolve()
checks = {
    "dataset source": (actual_source, expected_source),
    "image directory": (payload["dataset"]["images"], expected_images),
    "resolution": (int(payload["dataset"]["resolution"]), expected_resolution),
    "iteration": (int(payload["edgs"]["iteration"]), expected_iteration),
    "renderer backend": (payload["edgs"]["renderer_backend"], "pgsr"),
}
mismatches = [
    f"{name}: actual={actual!r}, expected={expected!r}"
    for name, (actual, expected) in checks.items()
    if actual != expected
]
if mismatches:
    raise SystemExit("EDGS bridge does not match this run:\n  " + "\n  ".join(mismatches))
print("EDGS bridge contract: OK")
PY
}

validate_pgsr_debug_contract() {
    local config_path="$1"

    run_edgs - \
        "${config_path}" \
        "${PGSR_DEBUG_INTERVAL}" \
        "${PGSR_DEBUG_FROM_ITER}" \
        "${PGSR_DEBUG_OUTPUT_DIR}" \
        "${PGSR_DEBUG_JPEG_QUALITY}" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

path = Path(sys.argv[1])
expected_interval = int(sys.argv[2])
expected_from_iter = None if sys.argv[3] == "auto" else int(sys.argv[3])
expected_output_dir = sys.argv[4]
expected_quality = int(sys.argv[5])

config = OmegaConf.load(path)
try:
    node = OmegaConf.to_container(config.gs.opt.pgsr_debug, resolve=True)
except Exception as error:
    raise SystemExit(
        f"saved EDGS config has no valid gs.opt.pgsr_debug section: {path}"
    ) from error
if not isinstance(node, dict):
    raise SystemExit(f"saved gs.opt.pgsr_debug is not a mapping: {path}")

expected = {
    "enabled": True,
    "interval": expected_interval,
    "from_iter": expected_from_iter,
    "output_dir": expected_output_dir,
    "jpeg_quality": expected_quality,
}
mismatches = [
    f"{name}: saved={node.get(name)!r}, requested={value!r}"
    for name, value in expected.items()
    if name not in node
    or node.get(name) != value
    or type(node.get(name)) is not type(value)
]
if mismatches:
    raise SystemExit(
        "reused PGSR debug history does not match this request:\n  "
        + "\n  ".join(mismatches)
    )
print(
    "PGSR debug contract: enabled, "
    f"interval={expected_interval}, from_iter={expected_from_iter}, "
    f"output_dir={expected_output_dir}, jpeg_quality={expected_quality}"
)
PY
}

validate_semantic_contract() {
    local bridge_manifest="$1"
    local semantic_ply="$2"
    local classifier_path="$3"
    local scene_info="$4"

    run_inpaint - \
        "${bridge_manifest}" \
        "${semantic_ply}" \
        "${classifier_path}" \
        "${scene_info}" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData
import torch

bridge_path, semantic_path, classifier_path, scene_path = map(Path, sys.argv[1:])
bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
scene = json.loads(scene_path.read_text(encoding="utf-8"))
base_path = Path(bridge["edgs"]["point_cloud_path"])

base = PlyData.read(base_path, mmap=True)["vertex"].data
semantic = PlyData.read(semantic_path, mmap=True)["vertex"].data
expected_count = int(bridge["gaussian_ply"]["elements"]["vertex"]["count"])
if len(base) != expected_count or len(semantic) != expected_count:
    raise SystemExit(
        "Semantic checkpoint is not bound to this EDGS reconstruction: "
        f"base={len(base)}, semantic={len(semantic)}, expected={expected_count}."
    )

semantic_names = set(semantic.dtype.names or ())
expected_objects = {f"obj_dc_{index}" for index in range(16)}
actual_objects = {name for name in semantic_names if name.startswith("obj_dc_")}
if actual_objects != expected_objects:
    raise SystemExit(
        "Semantic PLY must contain exactly obj_dc_0..obj_dc_15; "
        f"found {sorted(actual_objects)}."
    )

# Distillation freezes geometry. Exact XYZ equality binds the semantic PLY to
# the bridge more strongly than a point-count check while remaining mmap-safe.
chunk_size = 262_144
for start in range(0, expected_count, chunk_size):
    stop = min(start + chunk_size, expected_count)
    for field in ("x", "y", "z"):
        if not np.array_equal(base[field][start:stop], semantic[field][start:stop]):
            raise SystemExit(
                "Semantic checkpoint geometry differs from the EDGS source at "
                f"field={field}, rows=[{start}, {stop})."
            )

try:
    state = torch.load(classifier_path, map_location="cpu", weights_only=True)
except TypeError:
    state = torch.load(classifier_path, map_location="cpu")
for wrapper in ("state_dict", "classifier", "model"):
    if isinstance(state, dict) and isinstance(state.get(wrapper), dict):
        state = state[wrapper]
        break
if not isinstance(state, dict):
    raise SystemExit("Classifier checkpoint does not contain a state dict.")
weights = [
    value
    for key, value in state.items()
    if isinstance(value, torch.Tensor) and str(key).split(".")[-1] == "weight"
]
biases = [
    value
    for key, value in state.items()
    if isinstance(value, torch.Tensor) and str(key).split(".")[-1] == "bias"
]
if len(weights) != 1 or len(biases) != 1:
    raise SystemExit("Classifier must contain exactly one weight and one bias tensor.")
weight = weights[0].squeeze(-1).squeeze(-1)
num_classes = int(scene["num_classes"])
if tuple(weight.shape) != (num_classes, 16) or biases[0].numel() != num_classes:
    raise SystemExit(
        "Classifier/scene contract mismatch: "
        f"weight={tuple(weight.shape)}, bias={biases[0].numel()}, "
        f"classes={num_classes}."
    )
print(f"Semantic checkpoint contract: {expected_count} frozen Gaussians, {num_classes} classes")
PY
}

trap 'echo "Error: paintmesh run_seg failed at line ${LINENO}." >&2' ERR

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
        fail "start_stage must be an integer from 1 to 6; got '${START_STAGE}'"
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
        fail "resolution must be one of 1, 2, 4, or 8; got '${RESOLUTION}'"
        ;;
esac

require_positive_integer "${BASE_ITERATION}" "BASE_ITERATION"
require_positive_integer "${DISTILL_ITERATION}" "DISTILL_ITERATION"
require_positive_integer "${NUM_CLUSTERS}" "NUM_CLUSTERS"
require_positive_integer "${MESH_NEIGHBORS}" "MESH_NEIGHBORS"
require_positive_integer "${MESH_CHUNK_SIZE}" "MESH_CHUNK_SIZE"
require_boolean "${NO_DENSIFY}"
require_boolean "${PGSR_DEBUG}"
require_boolean "${USE_DEPTH_FILTER}"
require_boolean "${RENDER_VIDEO}"
require_boolean "${WRITE_COLORED_MESH}"
if is_true "${NO_DENSIFY}"; then
    NO_DENSIFY_OVERRIDE=true
else
    NO_DENSIFY_OVERRIDE=false
fi
if is_true "${PGSR_DEBUG}"; then
    PGSR_DEBUG_OVERRIDE=true
else
    PGSR_DEBUG_OVERRIDE=false
fi
require_positive_integer "${PGSR_DEBUG_INTERVAL}" "PGSR_DEBUG_INTERVAL"
if [[ "${PGSR_DEBUG_FROM_ITER}" != "auto" ]]; then
    require_nonnegative_integer "${PGSR_DEBUG_FROM_ITER}" "PGSR_DEBUG_FROM_ITER"
fi
require_integer_range \
    "${PGSR_DEBUG_JPEG_QUALITY}" "PGSR_DEBUG_JPEG_QUALITY" 1 100
require_relative_subdirectory \
    "${PGSR_DEBUG_OUTPUT_DIR}" "PGSR_DEBUG_OUTPUT_DIR"

require_dir "${EDGS_ROOT}"
require_dir "${INPAINT_ROOT}"
require_dir "${SCENE_ROOT}"
require_dir "${SCENE_ROOT}/${EDGS_IMAGES}"
command -v "${CONDA_BIN}" >/dev/null || fail "conda executable not found: ${CONDA_BIN}"

if (( START_STAGE <= 5 )); then
    require_dir "${SCENE_ROOT}/${SEGMENTATION_IMAGE_FOLDER}"
    require_dir "${SCENE_ROOT}/sparse/0"
    require_colmap_model "${SCENE_ROOT}/sparse/0"
fi
if (( START_STAGE <= 4 )); then
    require_file "${DISTILL_CONFIG}"
fi

mkdir -p \
    "${PIPELINE_ROOT}" \
    "${SEMANTIC_GS_ROOT}" \
    "${SEMANTIC_MESH_ROOT}"

if (( START_STAGE <= 2 )); then
    require_dir "${CKPT_ROOT}"
    mkdir -p "${SEG_WEIGHT_ROOT}"
    link_checkpoint_if_missing "CropFormer_hornet_3x_03823a.pth"
fi

echo "PaintMesh environment : ${PAINTMESH_ENV}"
echo "Dataset               : ${SCENE_ROOT}"
echo "Pipeline root          : ${PIPELINE_ROOT}"
echo "EDGS model             : ${EDGS_MODEL_ROOT}"
echo "Semantic 3DGS          : ${SEMANTIC_GS_ROOT}"
echo "Semantic mesh          : ${SEMANTIC_MESH_ROOT}"
echo "Resolution             : ${RESOLUTION}"
echo "Base iteration         : ${BASE_ITERATION}"
echo "Distill iteration      : ${DISTILL_ITERATION}"
echo "Start stage            : ${START_STAGE}"
if is_true "${PGSR_DEBUG}"; then
    echo "PGSR training debug    : ${EDGS_DEBUG_ROOT} (every ${PGSR_DEBUG_INTERVAL} steps, after ${PGSR_DEBUG_FROM_ITER})"
else
    echo "PGSR training debug    : disabled"
fi

run_inpaint -c \
    'import cv2, numpy, plyfile, scipy, torch; print("paintmesh Inpaint360GS imports: OK")'
run_edgs -c \
    'import diff_plane_rasterization, omegaconf, open3d, torch; print("paintmesh EDGS imports: OK")'

if (( START_STAGE <= 4 )); then
    run_inpaint - "${DISTILL_CONFIG}" "${DISTILL_ITERATION}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected = int(sys.argv[2])
configured = int(json.loads(path.read_text(encoding="utf-8"))["iterations"])
if configured != expected:
    raise SystemExit(
        f"DISTILL_ITERATION={expected} disagrees with {path}: iterations={configured}"
    )
print(f"Distillation config iteration: {configured}")
PY
fi

if (( START_STAGE <= 1 )); then
    echo "[1/6] Preparing the EDGS-PGSR reconstruction and base TSDF mesh"
    if [[ -f "${EDGS_PLY}" ]]; then
        require_file "${EDGS_MODEL_ROOT}/config.yaml"
        echo "      Reusing Gaussian checkpoint: ${EDGS_PLY}"
        if is_true "${PGSR_DEBUG}"; then
            if [[ -d "${EDGS_DEBUG_ROOT}" ]]; then
                validate_pgsr_debug_contract \
                    "${EDGS_MODEL_ROOT}/config.yaml"
                require_directory_within \
                    "${EDGS_DEBUG_ROOT}" \
                    "${EDGS_MODEL_ROOT}" \
                    "PGSR debug directory"
                echo "      Reusing existing training debug directory: ${EDGS_DEBUG_ROOT}"
            else
                fail "PGSR_DEBUG=true cannot be fulfilled from a reused Gaussian checkpoint without ${EDGS_DEBUG_ROOT}; use a new EDGS_MODEL_ROOT to retrain, or set PGSR_DEBUG=false"
            fi
        fi
    else
        if [[ -e "${EDGS_MODEL_ROOT}/config.yaml" || -e "${EDGS_MODEL_ROOT}/point_cloud" ]]; then
            fail "incomplete EDGS training output has no iteration ${BASE_ITERATION}; use a new EDGS_MODEL_ROOT or finish that run: ${EDGS_MODEL_ROOT}"
        fi
        training_arguments=(
            train.py
            gs=pgsr
            "train.gs_epochs=${BASE_ITERATION}"
            "train.no_densify=${NO_DENSIFY_OVERRIDE}"
            "gs.opt.iterations=${BASE_ITERATION}"
            "gs.opt.position_lr_max_steps=${BASE_ITERATION}"
            "gs.opt.pgsr_debug.enabled=${PGSR_DEBUG_OVERRIDE}"
            "gs.opt.pgsr_debug.interval=${PGSR_DEBUG_INTERVAL}"
            "gs.opt.pgsr_debug.output_dir=${PGSR_DEBUG_OUTPUT_DIR}"
            "gs.opt.pgsr_debug.jpeg_quality=${PGSR_DEBUG_JPEG_QUALITY}"
            "gs.dataset.source_path=${SCENE_ROOT}"
            "gs.dataset.model_path=${EDGS_MODEL_ROOT}"
            "gs.dataset.images=${EDGS_IMAGES}"
            "gs.dataset.resolution=${RESOLUTION}"
            gs.dataset.eval=true
            "gs.opt.save_iterations=[${BASE_ITERATION}]"
            init_wC.use=true
            wandb.mode=disabled
        )
        if [[ "${PGSR_DEBUG_FROM_ITER}" != "auto" ]]; then
            training_arguments+=(
                "gs.opt.pgsr_debug.from_iter=${PGSR_DEBUG_FROM_ITER}"
            )
        fi
        run_edgs "${training_arguments[@]}"
        if is_true "${PGSR_DEBUG}"; then
            require_dir "${EDGS_DEBUG_ROOT}"
        fi
    fi

    if [[ -f "${EDGS_MESH}" ]]; then
        echo "      Reusing TSDF mesh: ${EDGS_MESH}"
    else
        mesh_arguments=(
            render.py
            -m "${EDGS_MODEL_ROOT}"
            --iteration "${BASE_ITERATION}"
            --renderer pgsr
            --extract-mesh
            --max-depth "${MAX_DEPTH}"
            --voxel-size "${VOXEL_SIZE}"
            --num-clusters "${NUM_CLUSTERS}"
        )
        if is_true "${USE_DEPTH_FILTER}"; then
            mesh_arguments+=(--use-depth-filter)
        fi
        run_edgs "${mesh_arguments[@]}"
    fi
else
    echo "[1/6] Reusing the existing EDGS-PGSR reconstruction"
fi

require_file "${EDGS_MODEL_ROOT}/config.yaml"
require_file "${EDGS_PLY}"
require_file "${EDGS_MESH}"

# The bridge protects the immutable EDGS output from Inpaint360GS cfg_args
# writes and makes the selected base iteration explicit.
run_inpaint tools/build_edgs_bridge.py \
    --edgs-model "${EDGS_MODEL_ROOT}" \
    --iteration "${BASE_ITERATION}" \
    --output "${EDGS_BRIDGE_ROOT}"
require_complete_manifest "${EDGS_BRIDGE_ROOT}/bridge_manifest.json"
validate_bridge_contract "${EDGS_BRIDGE_ROOT}/bridge_manifest.json"

if (( START_STAGE <= 2 )); then
    echo "[2/6] Generating per-view CropFormer instance masks"
    run_inpaint seg/raw_mask_sam.py \
        --dataset_path "${DATA_ROOT}/${DATASET_NAME}" \
        --scene_name "${SCENE}" \
        --image_folder "${SEGMENTATION_IMAGE_FOLDER}" \
        --method hqsam \
        --threshold "${SEG_THRESHOLD}"
else
    echo "[2/6] Skipping per-view CropFormer masks"
fi
if (( START_STAGE <= 3 )); then
    validate_mask_pairs \
        "${SCENE_ROOT}/${SEGMENTATION_IMAGE_FOLDER}" \
        "${SCENE_ROOT}/raw_hqsam" \
        "raw CropFormer masks"
fi

if (( START_STAGE <= 3 )); then
    echo "[3/6] Associating instance IDs through the EDGS Gaussians"
    run_inpaint seg/mask_associate.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${EDGS_BRIDGE_ROOT}" \
        --iteration "${BASE_ITERATION}" \
        --images "${EDGS_IMAGES}" \
        --resolution "${RESOLUTION}" \
        --mask_generator hqsam \
        --patch "${ASSOCIATION_PATCH}" \
        --eval

    run_inpaint tools/add_label_num_hqsam.py \
        --source_path "${SCENE_ROOT}" \
        --resolution "${RESOLUTION}" \
        --mask_generator hqsam
else
    echo "[3/6] Skipping cross-view instance association"
fi
require_file "${SCENE_INFO}"
if (( START_STAGE <= 5 )); then
    validate_mask_pairs \
        "${SCENE_ROOT}/${SEGMENTATION_IMAGE_FOLDER}" \
        "${SCENE_ROOT}/associated_hqsam" \
        "associated instance masks"
fi

if (( START_STAGE <= 4 )); then
    echo "[4/6] Distilling instance embeddings into the EDGS Gaussians"
    run_inpaint seg/distillation.py \
        --source_path "${SCENE_ROOT}" \
        --model_path "${SEMANTIC_GS_ROOT}" \
        --vanilla_3dgs_path "${EDGS_BRIDGE_ROOT}" \
        --images "${EDGS_IMAGES}" \
        --resolution "${RESOLUTION}" \
        --object_path associated_hqsam \
        --config_file "${DISTILL_CONFIG}" \
        --test_iterations "${DISTILL_ITERATION}" \
        --save_iterations "${DISTILL_ITERATION}" \
        --checkpoint_iterations "${DISTILL_ITERATION}" \
        --eval
else
    echo "[4/6] Skipping Gaussian object-feature distillation"
fi
require_file "${SEMANTIC_GS_ROOT}/cfg_args"
require_file "${SEMANTIC_PLY}"
require_file "${CLASSIFIER_PATH}"
validate_semantic_contract \
    "${EDGS_BRIDGE_ROOT}/bridge_manifest.json" \
    "${SEMANTIC_PLY}" \
    "${CLASSIFIER_PATH}" \
    "${SCENE_INFO}"

if (( START_STAGE <= 5 )); then
    echo "[5/6] Rendering the semantic/object-aware 3DGS"
    semantic_render_arguments=(
        render.py
        --model_path "${SEMANTIC_GS_ROOT}"
        --iteration "${DISTILL_ITERATION}"
        --skip_fused_ply
    )
    if is_true "${RENDER_VIDEO}"; then
        semantic_render_arguments+=(--render_video)
    fi
    run_inpaint "${semantic_render_arguments[@]}"
else
    echo "[5/6] Skipping semantic 3DGS rendering"
fi

if (( START_STAGE <= 6 )); then
    echo "[6/6] Lifting Gaussian instance embeddings onto the PGSR mesh"
    semantic_mesh_arguments=(
        tools/lift_gaussian_semantics_to_mesh.py
        --gaussian-ply "${SEMANTIC_PLY}"
        --classifier "${CLASSIFIER_PATH}"
        --scene-info "${SCENE_INFO}"
        --mesh "${EDGS_MESH}"
        --output-dir "${SEMANTIC_MESH_ROOT}"
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
else
    echo "[6/6] Skipping semantic mesh export"
fi

require_complete_manifest "${SEMANTIC_MESH_ROOT}/semantic_manifest.json"
require_file "${SEMANTIC_MESH_ROOT}/gaussian_instance_id.npy"
require_file "${SEMANTIC_MESH_ROOT}/gaussian_confidence.npy"
require_file "${SEMANTIC_MESH_ROOT}/vertex_instance_id.npy"
require_file "${SEMANTIC_MESH_ROOT}/vertex_confidence.npy"
require_file "${SEMANTIC_MESH_ROOT}/face_instance_id.npy"
require_file "${SEMANTIC_MESH_ROOT}/face_confidence.npy"
if is_true "${WRITE_COLORED_MESH}"; then
    require_file "${SEMANTIC_MESH_ROOT}/semantic_mesh.ply"
fi

echo "PaintMesh semantic segmentation completed."
echo "Semantic 3DGS: ${SEMANTIC_ITERATION_ROOT}"
echo "Semantic mesh: ${SEMANTIC_MESH_ROOT}"
