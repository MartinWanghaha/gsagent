#!/usr/bin/env bash
# Seven independent runs: reuse depth_anchor PLY; original appearance+geometry.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PAINTMESH_PYTHON="${PAINTMESH_PYTHON:-/home/member/.conda/envs/paintmesh/bin/python}"
EXPERIMENT_PARENT="${REPO_ROOT}/output/paintmesh/Inpaint360GS/inpaint360/bag/removal/table1_adaptive180_sweep/inpaint"
BASELINE_RUN="${BASELINE_RUN:-${EXPERIMENT_PARENT}/original_supervision_depth_anchor_init_20261022_215217/depth_anchor}"
COMPARISON_ROOT="${COMPARISON_ROOT:-${EXPERIMENT_PARENT}/edgs_depth_local_prune_perceptual/comparison_paper_bag}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-${EXPERIMENT_PARENT}/original_supervision_depth_anchor_iterations_$(date +%Y%m%d_%H%M%S)}"
cd -- "${REPO_ROOT}"
exec "${PAINTMESH_PYTHON}" "${SCRIPT_DIR}/run_iteration_ablation.py" \
  --baseline-run "${BASELINE_RUN}" \
  --iterations 2000 4000 7000 9000 10000 15000 20000 \
  --gpu "${GPU:-1}" --python "${PAINTMESH_PYTHON}" \
  --paper-comparison "${COMPARISON_ROOT}/comparison.json" \
  --report "${COMPARISON_ROOT}/comparison.md" \
  --output-root "${RUN_OUTPUT_ROOT}" --detach "$@"
