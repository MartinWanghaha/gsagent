#!/usr/bin/env bash
# Bag: official appearance, then cumulative geometry, with clone/split/prune.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
paintmesh_python="${PAINTMESH_PYTHON:-/home/member/.conda/envs/paintmesh/bin/python}"
baseline_run="$repo_root/output/paintmesh/Inpaint360GS/inpaint360/bag/removal/table1_adaptive180_sweep/inpaint/edgs_depth_local_prune_perceptual"
previous_suite="${PRUNING_REFERENCE:-$("$paintmesh_python" -c 'import json; print(json.load(open("logs/official_geometry_ablation_gpu1.latest.json"))["output"])')}"
exec env PYTHONNOUSERSITE=1 "$paintmesh_python" scripts/paintmesh/run_geometry_ablation.py \
  --baseline-run "$baseline_run" \
  --reference-run "${baseline_run}_official_appearance_geo" \
  --adaptive-density --pruning-reference "$previous_suite" \
  --gpu "${GPU:-1}" --detach "$@"
