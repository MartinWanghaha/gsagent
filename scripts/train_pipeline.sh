#!/bin/bash
# Usage: bash train_pipeline.sh [options] <scene_folder> <output_base>
# Options:
#   --skip 0        skip annotation generation (step 0, LocateAnything)
#   --skip 1        skip EDGS (step 1)
#   --skip 2        skip ReferSplat (step 2)
#   --skip 0,1,2    skip multiple steps
#   --cuda 0        use GPU 0 (default: 0)
set -e

CONDA_BASE=/home/zhouyingchengliao/miniconda3
ENV_NAME=gsagent
PYTHON=${CONDA_BASE}/envs/${ENV_NAME}/bin/python

EDGS_DIR=/home/zhouyingchengliao/project/wyc/gsagent/submodules/EDGS
REFERSPLAT_DIR=/home/zhouyingchengliao/project/wyc/gsagent/submodules/ReferSplat

SKIP=""
CUDA_ID=7

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip) SKIP=$2; shift 2 ;;
        --cuda) CUDA_ID=$2; shift 2 ;;
        *) break ;;
    esac
done

SCENE_PATH=$(realpath ${1:?Usage: $0 [--skip 1|2|1,2] [--cuda N] <scene_folder> <output_base>})
OUTPUT_BASE=$(realpath -m ${2:?Usage: $0 [--skip 1|2|1,2] [--cuda N] <scene_folder> <output_base>})

EDGS_OUT=${OUTPUT_BASE}/edgs
REFERSPLAT_OUT=${OUTPUT_BASE}/refersplat
EDGS_ITERS=30000
EDGS_CKPT=${EDGS_OUT}/chkpnt${EDGS_ITERS}.pth

export CUDA_VISIBLE_DEVICES=${CUDA_ID}

SCRIPTS_DIR=/home/zhouyingchengliao/project/wyc/gsagent/scripts

if [[ ",${SKIP}," != *",0,"* ]]; then
    echo "=== [0/2] Annotation generation (LocateAnything, cuda=${CUDA_ID}) ==="
    ${PYTHON} ${SCRIPTS_DIR}/generate_annotations.py ${SCENE_PATH} --device cuda:0
else
    echo "=== [0/2] Annotation generation skipped ==="
fi

if [[ ",${SKIP}," != *",1,"* ]]; then
    echo "=== [1/3] EDGS training (cuda=${CUDA_ID}) ==="
    cd ${EDGS_DIR}
    ${PYTHON} train.py \
        train.gs_epochs=${EDGS_ITERS} \
        train.no_densify=True \
        gs.dataset.source_path=${SCENE_PATH} \
        gs.dataset.model_path=${EDGS_OUT} \
        init_wC.matches_per_ref=15000 \
        init_wC.nns_per_ref=3 \
        init_wC.num_refs=180 \
        device="cuda:0" \
        wandb.mode=disabled
else
    echo "=== [1/3] EDGS training skipped ==="
fi

if [[ ",${SKIP}," != *",2,"* ]]; then
    echo "=== [2/3] ReferSplat training (cuda=${CUDA_ID}) ==="
    cd ${REFERSPLAT_DIR}
    ${PYTHON} train.py \
        -s ${SCENE_PATH} \
        -m ${REFERSPLAT_OUT} \
        --start_checkpoint ${EDGS_CKPT} \
        --total_iters 45000
else
    echo "=== [2/3] ReferSplat training skipped ==="
fi

echo "=== Done. Output: ${OUTPUT_BASE} ==="
