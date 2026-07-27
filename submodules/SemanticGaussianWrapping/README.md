# Semantic Gaussian Wrapping

Semantic Gaussian Wrapping is a complete 3D Gaussian Splatting training and
surface-reconstruction project. It combines Gaga-style view-consistent instance
evidence with a Gaussian-Wrapping-style surface field, but semantics are not
used as hard geometric labels. Instead, confidence-gated semantics select a
soft mixture of five geometry policies: planar, curved, thin, freeform, and
fuzzy.

The research target is a Pareto improvement: better novel-view rendering and
better geometry from the same learned Gaussian state. The implementation
therefore protects the RGB gradient during joint optimization and reports
render and mesh metrics separately; it does not hide one objective behind a
single weighted score. Quantitative gains are deliberately not claimed without
scene-level experiments against the included RGB-only and semantic ablations.

## What is implemented

- COLMAP and Blender readers with pixel-aligned RGB, alpha, instance ID,
  confidence, and semantic-boundary observations.
- A Gaussian attribute registry that atomically handles xyz, covariance, SH,
  opacity, semantic embedding, geometry posterior, evidence buffers, optimizer
  state, checkpoints, and PLY IO during every clone/split/prune operation.
- A custom tile-based CUDA rasterizer with native forward and backward that
  composites RGB, 16D semantics, expected depth, alpha, normals, and dominant
  Gaussian IDs in one ordered pass. A differentiable PyTorch implementation
  provides a correctness oracle and CPU fallback.
- A four-phase trainer: RGB bootstrap, semantic lift with geometry
  stop-gradient, joint geometry optimization, and surface/mesh refinement with
  a photometric Pareto guard.
- Unified adaptive density control driven by absolute image-space gradient,
  RGB error, semantic error, boundary evidence, and geometric inconsistency.
  Soft geometry policies control split direction, scale behaviour, and pruning
  protection; per-step growth caps prevent residual normalization from causing
  runaway densification.
- Candidate-first semantic topology routing: only Gaussians that pass the
  absolute density gates are decoded, in exact FP32 chunks, before regional
  growth balancing. Unrenormalized foreground top-k probabilities, background,
  omitted probability mass, and confidence remain explicit; routing never
  collapses the model to hard labels.
- Candidate-first surface routing: a bounded, multi-scale cKDTree shortlist is
  re-ranked with the exact anisotropic support metric. One optimization step
  then shares the compact live attributes and geometry-policy outputs across
  global feedback-mesh probes and all soft region fields at each Gaussian
  center/pivot, without repeating KNN work for top-k memberships.
- Region-Conditioned Gaussian Wrapping (RC-GW): one renderer-consistent opacity
  field defines geometry for every group, while sparse Gaga posteriors allocate
  bounded local Delaunay charts and contact halos. Overlapping charts share
  canonical edge roots, so semantics increase local topology capacity without
  creating competing per-label surfaces.
- Region-balanced RGB and depth-normal objectives prevent small instances from
  disappearing inside scene-wide averages. They remain auxiliary Pareto tasks:
  any gradient component that conflicts with the global RGB/SSIM objective is
  removed before it reaches Gaussian parameters.
- A shared, refreshable neighbor index used by evidence projection,
  regularization, and the surface field. Surface queries retrieve scale-aware
  Gaussian support and re-rank it with the true anisotropic Mahalanobis metric.
- Separate direct and propagated confidence buffers, coherence-gated evidence
  diffusion, and certainty-aware geometry-expert supervision. Propagation can
  guide unobserved support but can never masquerade as a camera observation.
- A controlled late surface-topology window and donor-funded prune-and-replace
  policy. Clone/split costs are accounted exactly, protected seams/thin
  structures are not used as donors, and the global Gaussian cap is invariant.
- Cleaned feedback meshes extracted asynchronously from immutable inference
  snapshots. Only complete finite cache generations are installed; legacy or
  fragmented caches are regenerated rather than replayed on resume.
- One shared semantic surface field for differentiable training regularization
  and asynchronous feedback, plus a separate RC-GW offline exporter. The
  exporter integrates opacity to arbitrary 3D points with the CUDA renderer,
  builds Gaussian-adaptive pivots, and meshes semantic spatial charts against
  that single global field; it never treats raw Gaussian-mixture density or a
  decoder label as the final surface.
- Render metrics (PSNR/SSIM/LPIPS/L1) and mesh metrics (accuracy, completeness,
  Chamfer-L1, precision, recall, and F-score).

The detailed invariants are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Installation

The project requires Python 3.9+, PyTorch 2.1+, and, for the fast renderer, a
CUDA toolkit compatible with the installed PyTorch build.

```bash
cd submodules/SemanticGaussianWrapping
python install.py
```

If the build machine cannot see its deployment GPU, the installer compiles a
portable Turing/Ampere/PTX fallback instead of asking PyTorch to infer an empty
architecture list. For a smaller, device-specific binary, pass the target
compute capability explicitly, for example:

```bash
python install.py --cuda-arch-list "8.6"
```

Existing `TORCH_CUDA_ARCH_LIST` values are honored unless this flag is given.
Use `--skip-extension` for a reference-backend-only installation.

To install the Python dependencies without compiling CUDA, use:

```bash
pip install -r requirements.txt
```

The renderer automatically uses its PyTorch backend on CPU or when the CUDA
extension is unavailable. This is suitable for tests, not full-scene training.

## Dataset layout

An undistorted COLMAP dataset uses the standard 3DGS layout. Gaga observations
may be PNG, TIFF, NPY, NPZ, or PT tensors and must share each image's stem.

```text
scene/
├── images/
│   ├── 00001.jpg
│   └── ...
├── sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── sam_mask/
│   ├── info.json
│   ├── 00001.png
│   └── ...
├── confidence/                 # optional, [0,1]
│   └── 00001.npy
└── boundary/                   # optional; derived from IDs if absent
    └── 00001.png
```

`info.json` and associated IDs produced by Gaga are consumed directly. Raw IDs
are compacted to a scene-local label space for training, while the inverse map
is retained for export. Background and missing observations have zero geometry
confidence, so an RGB-only dataset reduces to the base 3DGS path.

For uncalibrated images, first place them in `scene/input` and run:

```bash
python convert.py -s /path/to/scene --resize
```

## Training

```bash
python train.py \
  -s /path/to/scene \
  -m output/scene \
  --config configs/default.yaml
```

Nested configuration values can be changed without editing the YAML:

```bash
python train.py -s /path/to/scene -m output/scene \
  --set optimization.iterations=30000 \
  --set data.semantic_confidence=confidence \
  --set data.semantic_boundary=boundary
```

For the repository's Mip-NeRF 360 layout, the batch launcher discovers and
validates COLMAP scenes plus aligned Gaga observations before starting each
training process:

```bash
python scripts/semantic_gaussian_wrapping/train_semantic_gaussian_wrapping_mipnerf360.py \
  --scene counter --gpu 0 --eval
```

Run this command from the `gsagent` root. Repeat `--scene`, use a comma-separated
list, or pass `--scene all`; `--dry-run` prints the exact translated `train.py`
command. Missing or partial semantic masks, metadata, foreground IDs, or image
alignment fail before scene construction; this entry point has no RGB-only
fallback. Use `--resume` to select the latest compatible checkpoint in the
base scene directory. To resume an isolated run created by
`--force`, select it explicitly instead of falling back to the base directory:

```bash
python scripts/semantic_gaussian_wrapping/train_semantic_gaussian_wrapping_mipnerf360.py \
  --scene counter --gpu 0 --resume --resume-run counter_rerun_002
```

`--resume-run` requires one concrete `--scene`, accepts that run's directory
name under `--output-root` (or its absolute path), and selects its latest
compatible checkpoint. The launcher also decodes observation files, checks their
aspect-ratio alignment (full-resolution masks may accompany downsampled images)
and Gaga metadata/ID coverage, and validates the fully resolved curriculum before
launching CUDA training. Re-run Gaga association with `--force` if stale 8-bit
masks cannot represent the `info.json` ID domain. The launcher also warns when a
single per-view dense semantic-loss logit tensor already exceeds 1 GiB.
Wrapper-owned settings must use their dedicated flags rather than `--set`. A
custom short fresh-run `--iterations` schedule therefore also needs explicitly
ordered `phases.*` and `surface.topology_{from,until}` overrides; resume keeps
the checkpoint's phase definition immutable and, unless `--iterations` is
explicit, preserves its original target iteration. `--force` never mixes old and new artifacts: it preserves the
existing scene directory and atomically reserves an isolated
`<scene>_rerun_NNN` output.

Resume with `--checkpoint output/scene/chkpnt12000.pth`. Only native schema-v3
training checkpoints from this architecture are accepted; older runs require a
fresh experiment. A checkpoint owns its
resolved experiment definition and restores Gaussian/optimizer state, semantic
heads and evidence, density statistics, mesh-feedback cache, camera sampling
stack, and random-number states. On resume only `optimization.iterations`,
`logging.*`, and the execution-only
`semantic.region_decode_chunk_size` may be changed with `--set`; a relocated but
identical dataset also requires `--allow-source-relocation`. Outputs include
standard `point_cloud/iteration_*` PLY snapshots, self-contained checkpoints,
the resolved YAML configuration, and JSONL training diagnostics. Checkpoints
are serialized beside the destination and atomically renamed only after a
complete write, so an interrupted save cannot replace a valid resume point.

For a low-frequency wall-time diagnosis, set `logging.profile_interval=N` with
`N > 0`. Sampled JSONL records add `time_render_ms`, `time_surface_ms`,
`time_backward_ms`, `time_topology_ms`, and `time_step_ms`. CUDA is synchronized
only at those sampled phase boundaries; the default value `0` adds no CUDA
synchronization. The measured step window excludes JSONL and checkpoint I/O.
`time_backward_ms` includes loss assembly, while `time_topology_ms` includes
density observation, optimizer updates, and any scheduled topology mutation.
The live cKDTree router uses `surface.scipy_workers` (default `4`), while the
asynchronous mesh extractor is independently capped by
`surface.mesh_feedback_scipy_workers` (default `1`) so it cannot occupy every
host core. Both are execution-only resume overrides and leave candidate
budgets, field samples, and mesh resolution unchanged.

The default curriculum is:

1. iterations 1–6999: RGB/SSIM and ordinary photometric densification;
2. 7000–11999: learn semantic embeddings and the scene decoder without
   allowing the semantic loss to move geometry;
3. 12000–23999: enable confidence-gated soft geometry policies and semantic
   topology control;
4. 24000–30000: enable shared-field and asynchronous cleaned-mesh feedback,
   plus bounded zero-net-growth surface prune-and-replace; conflicting
   auxiliary geometry gradients are projected away from the RGB gradient.

Mesh-v4 refresh policy is fully declared in the `surface` configuration. The
freshness controls are `mesh_feedback_max_candidate_age`,
`mesh_feedback_max_topology_events`, `mesh_feedback_max_churn_ratio`, and
`mesh_feedback_retry_interval`; accepted targets use
`mesh_feedback_blend_iterations`. Quality gating uses
`mesh_feedback_gate_{probes,min_score,sdf_p90,normal,semantic}`. Eligible live
support is restricted by `mesh_feedback_min_{opacity,confidence,expert_certainty}`,
and robust local correspondence uses `mesh_feedback_match_{k,radius,semantic}`,
`mesh_feedback_robust_delta`, and `mesh_feedback_min_matches`. Defaults retain
the quality-first extraction/routing sizes: resolution `96`, feedback samples
`8192`, and surface candidate budget `2048`.

The training step is ordered as render/shared query, backward, optimizer,
atomic density mutation, then mesh scheduling. A worker therefore always
captures the post-commit topology stamp. Completion creates a candidate rather
than changing the active target; publication requires a deterministic global
quality gate, at least 50% valid local correspondence coverage, and no material
regression against the active mesh. The current camera is used only for the
per-step visible-support batch. Active targets fade with age, and accepted
candidates enter through a smoothstep transition, so neither cache expiry nor
worker completion can create a one-step loss jump.

Both directions are local and dimensionless. Visible, opaque, semantically
confident, expert-certain Gaussian candidates are projected to nearby mesh
triangles with semantic/radius gating; mesh-to-field samples are then drawn
from those matched faces. Point-to-plane and SDF residuals are divided by a
detached Gaussian/mesh local scale and passed through a bounded Geman–McClure
penalty. Unmatched support becomes sparse mesh-coverage evidence for the
bounded prune-and-replace controller, not a long-range attraction gradient.
The extractor uses a surface-only immutable snapshot (no RGB SH or optimizer
state), and `snapshot_device=auto` keeps scenes above 500k Gaussians on CPU to
avoid competing with live CUDA backward memory.

Photometric densification keeps the standard 3DGS viewport-gradient contract:
the renderer exposes an NDC proxy whose x/y gradients include the `W/2` and
`H/2` pixel conversion used to calibrate `gradient_threshold`. Statistics are
collected from the first training view, while topology changes begin strictly
after `density.from_iter`. Opacity pruning remains active at every topology
step; screen/world-size pruning is opt-in (`density.enable_size_pruning`) and,
when requested, starts only after the first opacity-reset interval.

Semantic region balancing is candidate-first and exact. Absolute gates first
produce a compact candidate index set, and decoding is skipped when no growth
capacity remains. Only candidate embeddings are passed through the unchanged
semantic decoder. FP32 softmax retains the top-k foreground probabilities
without renormalizing them; background and omitted foreground mass are stored
separately. Regional budgets use confidence-weighted probability mass, and a
Gaussian may support several quotas while the topology transaction still
selects it at most once. The temporary logits scale as `O(B * C)` rather than
`O(N * C)`, where `B` is `semantic.region_decode_chunk_size`, `C` is the scene
class count, and `N` is the total Gaussian count.
The same chunk contract computes the dense image cross-entropy with activation
checkpointing and decodes density residuals under `no_grad`, so it never retains
an `H * W * C` logit tensor. This is an exact memory schedule, not a sampled or
reduced semantic objective.

## Rendering and evaluation

```bash
python render.py -m output/scene --iteration -1
python render.py -m output/scene --iteration 24000 --skip_train \
  --view-index 0 --view-index 7 --view-name DSC_0123
python metrics.py -m output/scene --split test \
  --iteration 7000 --iteration 24000
python metrics.py -m output/scene --split test --all-iterations
```

Each view writes RGB, ground truth, semantic IDs/features, expected depth,
alpha, and normals. View filters are repeatable, form a union, and retain the
original split-local numeric filename, so a targeted re-render never masquerades
as view zero. Rendering loads checkpoints through CPU, discards optimizer,
density, RNG, and mesh-cache state, and transfers only Gaussian registry and
decoder tensors to the target device. Pixel semantic argmax is exact FP32 but
streamed in bounded chunks. Metrics default to the latest render; explicit
iterations or `--all-iterations` evaluate several `ours_*` directories in one
process and store non-overwriting records under
`results/<split>/ours_<iteration>.json` plus a selection summary.

## Mesh extraction and evaluation

```bash
# Run from the gsagent repository root. The latest complete checkpoint is used.
python scripts/semantic_gaussian_wrapping/extract_semantic_gaussian_wrapping_mesh.py \
  outputs/semantic_gaussian_wrapping_mipnerf360/counter_rerun_003 \
  --gpu 0 --max-gaussians 500000

# Select a checkpoint, bound each local Delaunay problem, and optionally simplify.
python scripts/semantic_gaussian_wrapping/extract_semantic_gaussian_wrapping_mesh.py \
  outputs/semantic_gaussian_wrapping_mipnerf360/counter_rerun_003 \
  --iteration 30000 --gpu 0 \
  --max-gaussians 500000 --max-chart-gaussians 12000 \
  --target-faces 500000 --force
```

The offline exporter has one quality path and uses calibrated training cameras
only. For each camera, the CUDA rasterizer prepares projection, tile ordering,
and Gaussian precision once; point queries then integrate the same
front-to-back opacity convention used by RGB rendering. The conservative global
field is `occupancy_threshold - min_view(alpha_to_point)`. It is the only
surface oracle: semantic IDs never alter its value or create a separate
per-region zero set.

Sparse top-k Gaga posteriors instead build a coverage-conserving region atlas.
Low-confidence support enters a residual region, large groups are split into
bounded spatial charts, and overlap/contact halos keep supported boundaries
connected. Each selected Gaussian contributes covariance-aware
`[-sigma, center, +sigma]` normal pivots. Every chart runs local Delaunay and
marching tetrahedra against the global opacity field. Canonical pivot-edge keys
weld chart overlap before roots are queried; each unique edge is refined once
by bounded binary search using its few controlling views. Region-aware cleanup
then removes only small floaters while preserving the main component of every
represented group.

The result stores semantic ID, embedding, normal, and uncertainty per vertex,
and region ownership per face. Output is always binary little-endian PLY. A
required sibling `<mesh>.ply.json` uses schema 2 and records the checkpoint,
resolved RC-GW policy, atlas/topology counts, and optional metrics. The public
launcher considers an output complete only when this pair is valid, and each
file is replaced atomically. There is no tangent-SDF, global-Delaunay-then-label
filter, marching-cubes, raw-density, OBJ, or legacy extraction fallback.

`--max-gaussians` bounds the semantic-balanced anchor set; each anchor produces
three pivots. `--max-chart-gaussians` bounds one SciPy Delaunay problem without
reducing scene-wide coverage. `--view-stride` and `--camera-scale` reduce
renderer evidence cost for diagnosis, while quality runs should retain the
checkpoint defaults. Explicit CLI options override `mesh_export`. With
`--reference`, the same command reports accuracy, completeness, Chamfer,
precision, recall, and F-score; the evaluator accepts ASCII or binary PLY, OBJ,
and point-cloud references.

For a controlled dual-metric matrix, run the complete eight-variant matrix with
one scene/seed, the same evaluation cameras, and one reference mesh:

```bash
python scripts/semantic_gaussian_wrapping/run_ablation_matrix_mipnerf360.py \
  --scene counter --gpu 0 \
  --mesh-reference '/data/mesh_gt/{scene}.ply'

# Print every command without creating files.
python scripts/semantic_gaussian_wrapping/run_ablation_matrix_mipnerf360.py \
  --scene counter --mesh-reference '/data/mesh_gt/{scene}.ply' --dry-run

# Resume only from each variant checkpoint's authoritative configuration.
python scripts/semantic_gaussian_wrapping/run_ablation_matrix_mipnerf360.py \
  --scene counter --resume --gpu 0 \
  --mesh-reference '/data/mesh_gt/{scene}.ply'
```

The matrix contains the RGB and semantic-render controls, `full`, plus
factorized `full_no_mesh_feedback`, `full_no_surface_topology`,
`full_no_confidence_propagation`, `full_no_expert_certainty`, and
`full_no_prune_replace` variants, isolated under one directory per variant.
Fresh runs refuse accidental reuse; resume never passes
the ablation config or fresh data flags back into training. Image and mesh
metrics remain separate in each run, while a non-overwriting
`ablation_matrix*.json` joins them for comparison. Use `--steps train` when no
reference geometry exists; `--skip-mesh-metrics` is explicitly extraction-only
and is not a dual-metric benchmark. The runner validates every reference path
before launching any training. Mip-NeRF 360 ships images and COLMAP sparse
points, not a ground-truth surface mesh; those sparse points and any predicted
Gaussian/mesh PLY must not be substituted for mesh ground truth.

## Verifying the checkout

```bash
PYTHONPATH=. pytest -q
python -m compileall -q .
```

CUDA parity tests run when a GPU is available. On CPU they are skipped while
the differentiable reference renderer and all data/topology/mesh tests still
run.

## Research and licensing

This repository is research software. The rasterizer is a derivative of the
Graphdeco Gaussian Splatting rasterizer; attribution and component terms are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). See [LICENSE.md](LICENSE.md)
before redistribution or commercial use.
