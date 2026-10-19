#!/usr/bin/env bash

# The eight user-selected removal runs, all at resolution 4.
# Usage: bash scripts/paintmesh/run_inpaint_table1_batch.sh [all|0|1|--dry-run]
# GPU overrides the assigned GPU for every selected scene.
# OVERWRITE_INPAINT=true replaces each scene's previous inpaint run just before
# starting that scene. Removal, tracking and segmentation inputs are preserved.
# MASK_DILATION_CONFIG points to a reviewed per-scene JSON policy and takes
# precedence over the global MASK_DILATION value.
set -Eeuo pipefail

main() {
    local script_dir repo_root selection dry_run=false scene target_id gpu job
    local overwrite="${OVERWRITE_INPAINT:-false}"
    case "${overwrite}" in
        true|false) ;;
        *) echo "OVERWRITE_INPAINT must be true or false" >&2; return 2 ;;
    esac
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd -- "${script_dir}/../.." && pwd)"
    cd "${repo_root}"
    selection="${1:-all}"
    if (( $# > 1 )); then
        echo "Usage: $0 [all|0|1|--dry-run]" >&2
        return 2
    fi
    case "${selection}" in
        all|0|1) ;;
        --dry-run) dry_run=true; selection=all ;;
        *) echo "Usage: $0 [all|0|1|--dry-run]" >&2; return 2 ;;
    esac

    local dilation_policy="${MASK_DILATION_CONFIG:-}" policy_rows key value mask_dilation
    local -A scene_dilations=()
    if [[ -n "${dilation_policy}" ]]; then
        dilation_policy="$(realpath -e -- "${dilation_policy}")"
        policy_rows="$(python3 - "${dilation_policy}" "${RECURSIVE_GUIDE:-false}" <<'PY'
import json
import sys
from pathlib import Path
policy = json.loads(Path(sys.argv[1]).read_text())
if policy.get("schema_version") != 1 or policy.get("kind") != "paintmesh-lama-dilation-policy":
    raise SystemExit("Invalid mask dilation policy schema")
if policy.get("dataset") != "Inpaint360GS/inpaint360" or policy.get("resolution") != 4:
    raise SystemExit("Policy dataset/resolution differs from batch")
if policy.get("recursive_guide") is not False or sys.argv[2].lower() not in ("false", "0", "no", "off"):
    raise SystemExit("Reviewed policy requires RECURSIVE_GUIDE=false")
for scene in ("bag", "car", "cone_red", "cone_yellow", "cube", "doppelherz", "redbull", "truck"):
    value = policy.get("scenes", {}).get(scene, {}).get("mask_dilation")
    if type(value) is not int or not 0 <= value <= 256:
        raise SystemExit(f"Missing/invalid mask dilation for {scene}: {value!r}")
    print(f"{scene}\t{value}")
PY
)"
        while IFS=$'\t' read -r key value; do
            scene_dilations["${key}"]="${value}"
        done <<< "${policy_rows}"
        echo "Per-scene mask dilation policy: ${dilation_policy}"
    fi

    local log_dir="${BATCH_LOG_DIR:-${repo_root}/logs/inpaint_table1_$(date +%Y%m%d_%H%M%S)_$$}"
    if [[ "${dry_run}" == false ]]; then
        mkdir -p "${log_dir}"
        echo "Per-scene logs: ${log_dir}"
    fi
    local -a command
    local removal_root inpaint_root
    for job in bag:36:1 car:7:1 cone_red:24:1 cone_yellow:8:1 cube:25:1 doppelherz:18:1 redbull:8:0 truck:50:0; do
        IFS=: read -r scene target_id gpu <<< "${job}"
        [[ "${selection}" == all || "${selection}" == "${gpu}" ]] || continue
        gpu="${GPU:-${gpu}}"
        mask_dilation="${scene_dilations[${scene}]:-${MASK_DILATION:-10}}"
        removal_root="${repo_root}/output/paintmesh/Inpaint360GS/inpaint360/${scene}/removal/table1_hemi90"
        inpaint_root="${removal_root}/inpaint/edgs_rgbdn_joint_woDP"
        command=(env
            "PAINTMESH_ENV=${PAINTMESH_ENV:-paintmesh}" "LAMA_ENV=${LAMA_ENV:-lama}"
            "GPU=${gpu}" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
            "DATA_ROOT=${repo_root}/data"
            "PIPELINE_ROOT=${repo_root}/output/paintmesh/Inpaint360GS/inpaint360/${scene}"
            "REMOVAL_ROOT=${removal_root}"
            "INPAINT_RUN_ROOT=${inpaint_root}"
            INPAINT_RUN_NAME=edgs_rgbdn_joint_woDP INPAINT_PIPELINE=edgs-pgsr
            "MASK_DILATION=${mask_dilation}" "RECURSIVE_GUIDE=${RECURSIVE_GUIDE:-false}"
            "EDGS_INPAINT_CONFIG=${script_dir}/configs/edgs_inpaint.yaml"
            VIRTUAL_CAMERA_COUNT=90 DISTILL_ITERATION=2000
            EDGS_INIT_USE_DEPTH=false EDGS_INIT_USE_NORMAL=false
            EDGS_TRAIN_USE_DEPTH=true EDGS_TRAIN_USE_NORMAL=true
            EDGS_INPAINT_ITERATIONS=5000 EDGS_GEOMETRY_FROM_ITER=100 EDGS_GEOMETRY_RAMP_ITERS=2000
            LOCAL_GEOMETRY_REFINE=false SUPPORT_DENSITY_MODE=legacy END_STAGE=8
            bash "${script_dir}/run_inpaint.sh"
            Inpaint360GS/inpaint360 "${scene}" 4 "${target_id}" none 1)
        if [[ "${dry_run}" == true ]]; then
            if [[ "${overwrite}" == true ]]; then
                echo "Would replace inpaint output: ${inpaint_root}"
            fi
            printf '%q ' "${command[@]}"
            printf '\n'
        else
            if [[ "${overwrite}" == true ]]; then
                if [[ -L "${inpaint_root}" || "$(realpath -m -- "${inpaint_root}")" != "${inpaint_root}" ]]; then
                    echo "Refusing to replace an inpaint path containing symlinks: ${inpaint_root}" >&2
                    return 2
                fi
                local required
                for required in removal_manifest.json removed_3dgs/model_manifest.json \
                    tracker/tracking_session.json tracker/virtual_cameras.json; do
                    if [[ ! -f "${removal_root}/${required}" ]]; then
                        echo "Required removal input missing: ${removal_root}/${required}" >&2
                        return 2
                    fi
                done
                echo "Replacing previous inpaint output: ${inpaint_root}"
                rm -rf -- "${inpaint_root}"
            fi
            echo "Starting scene=${scene}, target=${target_id}, GPU=${gpu}, stages=1..8, MASK_DILATION=${mask_dilation}, RECURSIVE_GUIDE=${RECURSIVE_GUIDE:-false}"
            if "${command[@]}" < /dev/null 2>&1 | tee "${log_dir}/${scene}.log"; then
                echo "Completed scene: ${scene}"
            else
                local status=$?
                echo "Batch stopped at ${scene} (exit ${status}); log: ${log_dir}/${scene}.log" >&2
                return "${status}"
            fi
        fi
    done
}

main "$@"; exit $?
