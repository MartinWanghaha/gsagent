# SemanticPriorField

**Semantic Prior Field Splatting (SPF)** is a complete 3D Gaussian Splatting
pipeline in which view-consistent instance semantics actively *guide* the
spatial distribution of the Gaussians — orientation, density, size, spherical
harmonics — and the surface extraction, targeting a simultaneous improvement
of novel-view PSNR and mesh quality.

It contains the full standard 3DGS stack: COLMAP data loading, RaDeGS and
median-depth ("Ours") CUDA rasterizers with a native 16D semantic channel,
adaptive density control, depth-normal / multiview / normal-field / MILo
regularization, pivot-based Marching-Tetrahedra mesh extraction with binary
search refinement, texturing and evaluation. Geometry and appearance inherit
[Gaussian Wrapping](https://github.com/diego1401/GaussianWrapping) (oriented
Gaussians as stochastic surface elements); semantics inherit
[Gaga](https://github.com/weijielyu/Gaga) (16D identity embedding + scene
1x1 classifier, supervised by cross-view associated masks).

## The Semantic Prior Field

The core abstraction is a live, periodically refreshed field derived from
the jointly trained semantic embedding:

```text
associated masks (static, with confidence / valid / ignore)
      │  cross-entropy supervision
      ▼
16D per-Gaussian embedding + 1x1 classifier (live, trained jointly)
      │  argmax + margin, no_grad, refreshed every 500 iterations
      ▼
per-Gaussian labels & confidences ──► per-instance geometric proxies
                                       (plane / quadric / thin / freeform,
                                        fitted online by RANSAC + fit quality)
      │
      ├─► Regularizers        orientation prior, selective flattening,
      │                       SH region consistency, SH outlier decay
      ├─► Density control     boundary splitting, identity pruning,
      │                       per-instance densify-threshold multipliers
      └─► Mesh extraction     identity carried through Marching Tetrahedra,
                              semantic bad-edge filtering
```

Every prior is **soft, confidence-gated and per-instance**: an instance
whose proxy fits poorly automatically degrades to "no prior", so the worst
case is the unregularized baseline. All guidance lives in PyTorch — the CUDA
rasterizers are untouched beyond the additive 16D semantic channel.

### Channels

| Channel | Flag | Mechanism | PSNR | Mesh |
|---|---|---|---|---|
| Boundary weighting | `--sp_boundary` | depth-normal + multiview NCC down-weighted near instance boundaries | + | + |
| Identity pruning | `--sp_prune` | flat label posterior ∩ never argmax-contributor → prune | + | + |
| Boundary splitting | `--sp_split` | high semantic error → split along dominant tangent | + | + |
| Orientation prior | `--sp_orient` | learned normal aligned to plane/quadric proxy normal | ≈ | ++ |
| Selective flattening | `--sp_flatten` | min-scale loss only on planar/quadric instances, thin exempt | ≈/+ | + |
| SH regularization | `--sp_sh` | same-instance SH consistency + outlier decay | + | + |
| Budget reallocation | `--sp_budget` | planar densify less (×1.5 threshold), thin densify more (×0.7) | + | + |
| Rasterizer stats | `--sp_stats` | per-Gaussian semantic conflict from the SPF rasterizer backward drives splitting (replaces episodic camera sweeps) | + | + |
| Loss balancing | `--balance_semantic` | Kendall uncertainty weighting between photometric and semantic CE (default off, untested) | ? | ? |

Mesh-side semantic edge filtering (`--filter_semantic_edges`) is available
but **default-off**: it fragments meshes in cluttered fine-grained-instance
scenes (docs/experiments.md AB-MESH). Vertex identity export
(`--use_semantics`) is on by default in the launcher.

Each channel is independently switchable (`--no-sp_<name>`) for ablations.
With `--semantic_prior` off, the pipeline is exactly the underlying
Gaussian Wrapping (+ optional passive Gaga semantics).

Defaults are experiment-driven — see [docs/experiments.md](docs/experiments.md).
Headline result (counter, 6k-iteration isolation A/B): conflict-stats
splitting beats both no-splitting (+0.10 dB PSNR, −8% depth-normal error)
and camera-sweep splitting (+0.07 dB at identical split budget, zero
attribution overhead).

## Installation

```bash
cd submodules/SemanticPriorField
conda create -n semantic_prior_field python=3.9 -y
conda activate semantic_prior_field

export CPATH=/usr/local/cuda-12.1/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.1/bin:$PATH

python install.py --cuda_version 12.1
```

`install.py` builds the original Gaussian Wrapping rasterizers and the two
semantic extensions (`diff_gaussian_rasterization_gw_semantic`,
`diff_gaussian_rasterization_gw_ours_semantic`).

## Input contract

The image dataset uses the same COLMAP layout as Gaussian Wrapping.
Associated instance masks are supplied in one directory and resolved by
camera image stem:

```text
dataset/
  images/
  sparse/0/
gaga_masks/
  000001.png          # uint16 instance IDs, 0 = background
  000002.png
  info.json           # {"num_mask": N, "ignore_label": 65535, ...}
  confidence/         # optional per-pixel confidence (robust association)
  valid/              # optional validity masks
```

Masks may be indexed PNG/TIFF/NPY files; RGB masks use Gaga's 24-bit ID
encoding. The 65535 ignore label and the `confidence/` / `valid/` sidecars
produced by robust association are consumed automatically. Resizing is
always nearest-neighbour.

## Usage

### End to end

```bash
python semantic_prior_field/scripts/train_and_extract_spf.py \
    -s <PATH_TO_COLMAP_DATASET> \
    -m <OUTPUT_DIR> \
    --semantic_masks <ASSOCIATED_MASK_DIR> \
    --rasterizer ours -r 2
```

Runs joint training with the Semantic Prior Field, pivot-based mesh
extraction with semantic edge filtering, and texture refinement.

### Step by step

```bash
# 1. Joint geometry + semantics + Semantic Prior Field
python semantic_prior_field/train.py \
    -s <DATASET> -m <OUTPUT_DIR> \
    --semantic_masks <MASK_DIR> \
    --semantic_prior \
    --rasterizer ours --exposure_compensation --data_device cpu

# 2. Mesh with identity carried through Marching Tetrahedra
python semantic_prior_field/pivot_based_mesh_extraction.py \
    -s <DATASET> -m <OUTPUT_DIR> \
    --rasterizer ours --sdf_mode ours --isosurface_value 0.0 \
    --n_binary_steps 10 --use_valid_mask --filter_large_edges \
    --use_semantics --filter_semantic_edges

# 3. Texture
python semantic_prior_field/texture_mesh.py \
    -s <DATASET> -m <OUTPUT_DIR> --rasterizer ours \
    --mesh <OUTPUT_DIR>/mesh_ours_2pivots.ply
```

Outputs next to the mesh: `*.semantic.npz` (per-vertex labels + confidences,
optionally 16D embeddings with `--export_vertex_semantics`) and
`*_semantic.ply` (instance-colored mesh).

Ablations — any subset of channels:

```bash
python semantic_prior_field/train.py ... --semantic_prior \
    --no-sp_flatten --no-sp_sh        # keep pruning/splitting/boundary only
```

Schedules and weights live in
[`semantic_prior_field/configs/semantic_prior/default.yaml`](semantic_prior_field/configs/semantic_prior/default.yaml).

### Training timeline (default config)

| Iteration | Event |
|---|---|
| 0 | photometric + joint semantic CE (embedding-only gradients) |
| 500–15000 | standard densification; from 7000 with per-instance threshold multipliers |
| 7000 | depth-normal + multiview on, **boundary-weighted** |
| 7000 | first Semantic Prior Field refresh (every 500, invalidated on topology changes) |
| 15000, 20000 | identity-stability pruning |
| 20001 | normal field on + **orientation prior**, selective flattening, SH terms |
| 22500–26500 | semantic boundary splitting (offset from normal-field spoke splitting at 22000–26000) |
| 30000 | save; extraction with semantic pivots + bad-edge filtering |

### Passive semantics (no geometry guidance)

All Gaga-style stages remain available unchanged when the prior field is
not wanted:

```bash
# Frozen-geometry lift into an already trained model
python semantic_prior_field/semantic_lift.py \
  -s <DATASET> -m <GEOMETRY_OUTPUT> \
  --semantic_masks <MASKS> --semantic_output <SEMANTIC_OUTPUT> \
  --rasterizer radegs

# Render / evaluate / transfer semantics
python semantic_prior_field/semantic_render.py -s <DATASET> -m <SEMANTIC_OUTPUT> \
  --semantic_checkpoint <...>/semantic/semantic_chkpnt10000.pth --split test
python semantic_prior_field/semantic_eval.py --render_dir <RENDER_OUTPUT/test> \
  --semantic_masks <MASKS> --semantic_checkpoint <...>.pth
python semantic_prior_field/semantic_mesh.py --mesh <MESH.ply> \
  --semantic_ply <...>/point_cloud.ply --semantic_checkpoint <...>.pth \
  --output <SEMANTIC_MESH.ply>
```

Joint training without guidance: pass `--semantic_masks` but not
`--semantic_prior`.

## Output contract

The semantic PLY is a full Gaussian Wrapping PLY plus `obj_dc_0..15`.
Nothing is discarded: SH color, opacity, scale, rotation, mip filter,
occupancy, normal features and exposure/appearance state keep the original
formats. Semantic sidecars (`semantic/semantic_chkpnt<iter>.pth`) carry the
head, an exact copy of the embeddings, optimizer state and metadata.

## Diagnostics

Every training run records its intermediate state under
`<model_path>/diagnostics/` (on by default; `--no-diagnostics` disables):

```text
diagnostics/
  scalars.jsonl        every active loss term + Gaussian count, every 10 iters
  events.jsonl         every densify/split/prune decision with before/after counts
  prior_field/         per-refresh instance tables: proxy type, fit residual,
                       weight, member counts, labelled fraction
  images/iter_XXXXXX/  current training view: render/gt, depth, normal, alpha,
                       depth-normal error, semantic pred/gt/PCA/error map,
                       mask confidence, boundary weights
  snapshots/           per-Gaussian .npz every 5000 iters: xyz, opacity, scale,
                       labels, confidence, prior type/weight, densify
                       multiplier, SH high-order energy
```

Intervals: `--diag_scalar_interval 10 --diag_image_interval 1000
--diag_snapshot_interval 5000` (snapshot 0 = off). Loading for analysis:

```python
import pandas as pd, numpy as np
scalars = pd.read_json("diagnostics/scalars.jsonl", lines=True)
events  = pd.read_json("diagnostics/events.jsonl", lines=True)
snap    = np.load("diagnostics/snapshots/iter_030000.npz")
```

Diagnostics are failure-isolated: an error in any dump prints a warning and
never interrupts training.

## Evaluation

The full Gaussian Wrapping evaluation stack is preserved: `metrics.py`
(PSNR/SSIM/LPIPS), `evaluate_dtu_mesh.py` (Chamfer), `eval/` (TnT F1), plus
`semantic_eval.py` (Hungarian-matched segmentation IoU).

Acceptance criterion for every channel: PSNR and Chamfer must never degrade
together (Pareto rule); with all channels off, results are bit-identical to
the base pipeline.

## Tests

```bash
python -m pytest tests/ -q
```

`tests/test_semantic_prior_field.py` covers proxy fitting (plane RANSAC,
quadric normals, thin-structure protection), label derivation parity with
the pixel classifier, boundary weight maps, the semantic edge predicate,
pivot expansion layout, SH regularizers and threshold multipliers — all on
CPU. CUDA rasterizer parity tests are in
`tests/test_cuda_semantic_rasterizers.py`.

## Project structure

```text
semantic_prior_field/
├── train.py                          # joint training + SPF wiring
├── scene/                            # data loading, GaussianModel (16D semantics native)
├── gaussian_renderer/                # RaDeGS / Ours (+ semantic channel), SOF, Mini-Splatting stats
├── semantic/
│   ├── prior_field.py                # ★ SPF: labels, proxies, boundary maps
│   ├── observations.py               # associated-mask store (confidence/valid/ignore)
│   └── head.py / losses.py           # 1x1 classifier, CE, spatial consistency
├── regularization/
│   ├── regularizer/semantic_prior.py # ★ orientation / flatten / SH consumers
│   └── regularizer/{normal_field,multiview,mesh_in_the_loop,...}.py
├── densification/
│   ├── semantic_error.py             # ★ boundary splitting + identity pruning
│   └── normal_error.py
├── extraction/
│   ├── mesh.py                       # MT (+ semantic interpolation & edge filter)
│   ├── semantic.py                   # ★ sidecar loading, pivot expansion
│   └── pivots.py
├── pivot_based_mesh_extraction.py    # extraction CLI (+ --use_semantics)
├── configs/semantic_prior/default.yaml
└── scripts/train_and_extract_spf.py  # ★ end-to-end launcher
```

★ = Semantic Prior Field additions.

## Acknowledgements

Built on [Gaussian Wrapping](https://github.com/diego1401/GaussianWrapping)
(Gomez, Guédon et al.), [Gaga](https://github.com/weijielyu/Gaga)
(Lyu et al.) and their upstream lineage (3DGS, Mip-Splatting, RaDe-GS, GOF,
PGSR, Mini-Splatting2, MILo, GGGS). See `LICENSE.md`, `LICENSE_GAGA.md` and
`THIRD_PARTY_NOTICES.md`.
