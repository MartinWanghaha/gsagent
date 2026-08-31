# PaintMesh semantic reconstruction, removal, and inpainting pipeline

English | [简体中文](README.zh-CN.md) | [中文原理说明](PRINCIPLES.zh-CN.md)

`run_seg.sh` combines the modified EDGS-PGSR reconstruction with the
Inpaint360GS object-feature distillation pipeline. It produces an object-aware
3DGS checkpoint and lifts the same scene-local instance IDs onto the PGSR TSDF
mesh.

`run_remove.sh` consumes that immutable semantic result, removes selected
instances in an isolated Inpaint360GS workspace, publishes a reusable removed
3DGS, and reconstructs a matching PGSR/TSDF mesh from the removed Gaussians.

`run_inpaint.sh` then consumes the camera-bound tracker masks, performs
run-local LaMa RGB/depth completion and Inpaint360GS optimization, and rebuilds
both the completed EDGS-PGSR 3DGS and its semantic TSDF mesh.

The labels produced by CropFormer and Inpaint360GS are scene-local instance
IDs. They are not category names such as `chair` or `table` unless an external
instance-to-category mapping is provided.

## Environment

All EDGS, Inpaint360GS and mesh-lifting stages run in the `paintmesh` conda
environment. EDGS and Inpaint360GS are launched in separate processes with
isolated `PYTHONPATH` values because both repositories contain modules named
`scene`, `utils` and `gaussian_renderer`.

Path overrides such as `DATA_ROOT`, `OUTPUT_ROOT`, `PIPELINE_ROOT` and
`EDGS_MODEL_ROOT` are resolved against the directory from which the runner was
invoked before any subprocess changes its working directory. Absolute paths
remain the clearest choice for reusable runs.

Required checkpoints are expected in the repository-level `ckpt/` directory:

```text
ckpt/CropFormer_hornet_3x_03823a.pth
```

Interactive removal-mask refinement additionally requires:

```text
ckpt/sam_vit_b_01ec64.pth
ckpt/R50_DeAOTL_PRE_YTB_DAV.pth
ckpt/groundingdino_swint_ogc.pth
```

## Run from scratch

```bash
scripts/paintmesh/run_seg "mip-nerf/360_v2" kitchen 8 1
```

Arguments are:

```text
dataset_name scene resolution start_stage
```

`resolution` accepts `1`, `2`, `4`, or `8`. Value `1` uses `images/`; the
other values use `images_<resolution>/` for 2D segmentation while EDGS and
Inpaint360GS use the same `round()`-based output dimensions. The corresponding
image directory must already exist.

The stages are:

1. EDGS-PGSR reconstruction and TSDF extraction.
2. Per-view CropFormer masks.
3. Cross-view instance association through the EDGS Gaussians.
4. Inpaint360GS object-feature distillation.
5. Semantic/object-aware 3DGS rendering.
6. Gaussian-to-mesh semantic lifting.

Stage 1 reuses the final Gaussian checkpoint and TSDF mesh independently, so
a failed mesh extraction can be resumed without retraining EDGS.

## Optional PGSR training visualization

Stage 1 can save the same 2-by-4 diagnostic montage used by upstream PGSR:

```bash
PGSR_DEBUG=true GPU=0 \
  scripts/paintmesh/run_seg "mip-nerf/360_v2" kitchen 8 1
```

The feature is disabled by default and writes JPEG files to:

```text
output/paintmesh/<dataset>/<scene>/edgs/debug/
└── 07200_<camera_name>.jpg
```

Each image contains, from left to right:

```text
GT | rendered RGB | rendered normal | rendered plane distance
reprojection weight | plane depth | depth normal | image-gradient weight
```

The defaults reproduce PGSR's schedule: the multi-view threshold is strict
(`step > 7000`) and the interval is 200, so the first normal frame is step
7200. The following environment variables control it:

```bash
PGSR_DEBUG=true
PGSR_DEBUG_INTERVAL=200
PGSR_DEBUG_FROM_ITER=auto       # use pgsr_loss.multi_view_from_iter
PGSR_DEBUG_OUTPUT_DIR=debug     # relative to EDGS_MODEL_ROOT
PGSR_DEBUG_JPEG_QUALITY=95
```

For example, to delay capture until after step 10000 and then write every 500
iterations, use `PGSR_DEBUG_FROM_ITER=10000 PGSR_DEBUG_INTERVAL=500`. A
montage is only written after the PGSR multi-view loss is active and when that
training step has a valid neighboring view and actual reprojection weights;
this avoids silently replacing the geometry panel with invented data.

This option is a training visualization and is deliberately unrelated to
`gs.pipe.debug`, which controls low-level CUDA crash snapshots. It reuses the
active training render and selected neighbor, so it adds CPU image conversion
and one JPEG write at capture steps but no extra reference-view render.

If the final EDGS checkpoint already exists, Stage 1 reuses it and cannot
retroactively reconstruct the random training views. Select a new
`EDGS_MODEL_ROOT` to generate a fresh debug history. To avoid claiming success
without the requested artifact, the runner stops when `PGSR_DEBUG=true` reuses
a checkpoint that has no matching debug directory. It also checks the saved
`config.yaml`; `enabled`, interval, start threshold, output directory and JPEG
quality must match the requested history. Existing debug images are never
deleted when the option is disabled.

The same feature is available when invoking EDGS directly:

```bash
cd submodules/EDGS
conda run --no-capture-output -n paintmesh python train.py \
  gs=pgsr \
  gs.opt.pgsr_debug.enabled=true \
  gs.opt.pgsr_debug.interval=200 \
  gs.dataset.source_path=/path/to/scene \
  gs.dataset.model_path=/path/to/output \
  train.gs_epochs=30000 \
  wandb.mode=disabled
```

## Reuse an existing EDGS run

The EDGS run must contain `config.yaml`, the selected Gaussian iteration and
the corresponding PGSR mesh.

```bash
EDGS_MODEL_ROOT=/home/martin/code/gsagent/output/kitchen_pgsr_r2 \
  scripts/paintmesh/run_seg "mip-nerf/360_v2" kitchen 8 2
```

Do not reuse a classifier distilled from another Gaussian model. The bridge
pins the EDGS iteration with hashes and point counts; the semantic exporter
checks the object-channel/classifier schema and records every resolved input
path, size and modification time in its manifest. Before rendering or mesh
lifting, the runner also compares the frozen semantic PLY XYZ values against
the bridged EDGS PLY. Keep the generated PLY and `classifier.pth` together;
two independently distilled classifiers over identical geometry still do not
carry a universal scene ID.

## Important overrides

```bash
PAINTMESH_ENV=paintmesh          # conda environment
GPU=0                            # optional CUDA_VISIBLE_DEVICES value
BASE_ITERATION=30000             # EDGS checkpoint/mesh iteration
DISTILL_ITERATION=2000           # must match train_distill.json when stage 4 runs
NO_DENSIFY=true
PGSR_DEBUG=false
PGSR_DEBUG_INTERVAL=200
MAX_DEPTH=5.0
VOXEL_SIZE=0.002
USE_DEPTH_FILTER=false
RENDER_VIDEO=false
WRITE_COLORED_MESH=true
MESH_NEIGHBORS=8
MESH_CHUNK_SIZE=32768
```

`WRITE_COLORED_MESH=true` writes another full mesh. The kitchen reference mesh
is approximately 867 MB, so ensure that the output filesystem has sufficient
space. The `.npy` label arrays and `semantic_manifest.json` remain the
authoritative semantic result.

## Outputs

By default, outputs are stored under:

```text
output/paintmesh/<dataset>/<scene>/
├── edgs/
├── edgs_bridge/
├── semantic_3dgs/
└── semantic_mesh/
```

The semantic 3DGS contract is:

```text
semantic_3dgs/point_cloud/iteration_<D>/point_cloud.ply
semantic_3dgs/point_cloud/iteration_<D>/classifier.pth
<scene>/associated_hqsam/scene.json
```

The semantic mesh contract is:

```text
semantic_mesh/semantic_mesh.ply                 # optional colored geometry
semantic_mesh/gaussian_instance_id.npy
semantic_mesh/gaussian_confidence.npy
semantic_mesh/vertex_instance_id.npy
semantic_mesh/vertex_confidence.npy
semantic_mesh/face_instance_id.npy
semantic_mesh/face_confidence.npy
semantic_mesh/palette.json
semantic_mesh/semantic_manifest.json
```

Label `0` is background. Label `65535` is reserved for vertices or faces with
insufficient support or ambiguous predictions.

Stage 6 uses the manifest as an atomic completion marker and can be rerun
safely. For example, after changing only mesh-lifting thresholds:

```bash
MESH_MIN_CONFIDENCE=0.2 MESH_MIN_MARGIN=0.05 \
  EDGS_MODEL_ROOT=/home/martin/code/gsagent/output/kitchen_pgsr_r2 \
  scripts/paintmesh/run_seg "mip-nerf/360_v2" kitchen 8 6
```

## Remove objects from the semantic 3DGS and mesh

Run `run_seg` through Stage 6 first, then select scene-local instance IDs from
the numbered previews under `images_<resolution>_num/`. ID `0` is background
and cannot be removed.

Remove instance `14`:

```bash
GPU=0 scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

Temporarily remove instances `10` and `24` as occluders while permanently
removing `14`:

```bash
GPU=0 scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 "10,24" 1
```

Arguments are:

```text
dataset_name scene resolution target_ids surrounding_ids start_stage
```

`target_ids` is mandatory and accepts a positive integer or a comma-separated
list. `surrounding_ids` accepts the same format or `none`. Target and
surrounding sets must be disjoint and every ID must exist in the semantic
Gaussian sidecar produced by `run_seg`. `resolution` accepts `1`, `2`, `4`,
or `8` and must match the resolution used for that `run_seg` output.

The removal stages are:

1. Create run-local removal/inpainting configs and a read-only semantic-model
   workspace.
2. Remove Gaussians and publish an EDGS-loadable target-removed 3DGS.
3. Re-render the removed 3DGS with PGSR, extract a TSDF mesh, and lift the
   remaining instance semantics onto it.
4. Render 30 virtual views and package them in a run-local tracker archive.
5. Optionally launch Segment-and-Track-Anything for manual mask refinement.

Stages 4–5 are auxiliary mask-refinement outputs. They do not modify the
removed 3DGS or mesh produced by Stages 2–3.

The default executes all stages and launches the local Gradio interface. After
exporting all 30 masks, stop the interface with `Ctrl+C`; the runner accepts
that interrupt only when all masks exist. To generate only the removed 3DGS
and mesh without starting the interactive stages:

```bash
END_STAGE=3 LAUNCH_REFINER=false \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

Resume only mesh extraction and semantic lifting:

```bash
END_STAGE=3 LAUNCH_REFINER=false \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 3
```

Important removal overrides are:

```bash
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
MAX_DEPTH=5.0
VOXEL_SIZE=0.002
NUM_CLUSTERS=1
REMOVAL_ROOT=/absolute/output/path
```

`WRITE_DEBUG_PLY=false` avoids two large diagnostic point clouds written by
upstream Inpaint360GS. `RENDER_VIDEO` and `RENDER_OBJECT_VIDEOS` are also
disabled by default because Stage 3 already renders the authoritative removed
model. These settings do not affect which Gaussians are removed.

`RENDER_REMOVAL_TRAIN` and `RENDER_REMOVAL_TEST` write optional Stage 2
diagnostic images into `work_model/{train,test}/`. Their depth visualization
is normalized with the matching original semantic-3DGS reference depth, so the
isolated work model never needs to duplicate those large arrays.

Inside the Stage 5 Gradio page:

1. Open the **Image-Seq** tab and click the run-local `images.zip` example.
2. Extract the sequence, mark the target on frame `00000`, and initialize the
   tracker.
3. Click **Start Tracking** and wait until all 30 base masks are exported.
4. Stop the server with `Ctrl+C`; the runner then verifies the exact
   `00000.png`–`00029.png` set, dimensions, and archive-bound session manifest.

Files such as `*_new.png` are tracker diagnostics and are not counted as final
frame masks. A Stage 5 restart reuses a valid complete session or resumes an
`in_progress` session bound to the same `images.zip`. Rerun Stage 4 first when
you intentionally want to invalidate those masks and start refinement again.

Removal directories are immutable with respect to the selected IDs,
`REMOVAL_THRESHOLD`, semantic checkpoint, and EDGS bridge. For a parameter
variant, choose a new run name or output root instead of reusing the old one:

```bash
RUN_NAME=target_14_t080 REMOVAL_THRESHOLD=0.8 END_STAGE=3 \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

Likewise, diagnostic rendering flags must be selected on the first Stage 2 run
for that directory. The runner refuses to pretend they were generated when it
reuses an immutable published checkpoint.

The default output is target-specific, so different removals cannot overwrite
each other:

```text
output/paintmesh/<dataset>/<scene>/removal/target_<ids>/
├── config/
├── work_model/
│   ├── workspace_manifest.json
│   ├── point_cloud -> ../../../semantic_3dgs/point_cloud
│   ├── point_cloud_object_removal/
│   └── virtual/
├── removed_3dgs/
│   ├── config.yaml
│   ├── cfg_args
│   ├── model_manifest.json
│   ├── point_cloud/iteration_<D>/
│   │   ├── point_cloud.ply
│   │   └── classifier.pth
│   └── mesh/ours_<D>/
│       ├── tsdf_fusion_post.ply
│       └── mesh_manifest.json
├── removed_mesh/
│   ├── geometry.ply -> ../removed_3dgs/mesh/ours_<D>/tsdf_fusion_post.ply
│   ├── semantic_mesh.ply
│   ├── vertex_instance_id.npy
│   ├── face_instance_id.npy
│   └── semantic_manifest.json
├── tracker/
│   ├── images.zip
│   ├── virtual_cameras.json
│   ├── assets/
│   ├── results/
│   └── tracking_session.json
└── removal_manifest.json
```

With surrounding IDs, the directory name is
`target_<ids>__surrounding_<ids>` unless `RUN_NAME` or `REMOVAL_ROOT` is set.

When surrounding IDs are supplied, `work_model/point_cloud_object_removal/
iteration_<D>/` is the temporary background with target and surrounding
objects removed. The published `removed_3dgs` comes from
`iteration_<D>_removal_target/`, where surrounding objects have been restored.

The removed mesh is reconstructed from the published removed Gaussians rather
than made by deleting labeled faces from the old mesh. This keeps its geometry
consistent with Inpaint360GS's probability threshold and convex-hull removal.
The manifests bind the result to the exact EDGS bridge, semantic checkpoint,
instance IDs, threshold, and iteration; a mismatched reuse fails instead of
silently mixing scenes or classifiers.

## Inpaint the removed region and rebuild both 3DGS and mesh

`run_inpaint` consumes one completed `run_remove` result. It never writes into
the original dataset, the shared LaMa `data/`/`output/` directories, or the
published removed model.

Before the first inpaint run, Stage 4 of `run_remove` must have generated
`tracker/virtual_cameras.json`, and Stage 5 must have committed exactly 30
refined masks. Existing tracking sessions created before the exact-camera
manifest was introduced should be regenerated:

```bash
PAINTMESH_ENV=paintmesh GPU=0 END_STAGE=5 \
  scripts/paintmesh/run_remove \
  "mip-nerf/360_v2" kitchen 8 14 none 4
```

After exporting the masks and stopping the tracker with `Ctrl+C`, run the full
inpaint pipeline:

```bash
PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 END_STAGE=8 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

The main Inpaint360GS, 3DGS, PGSR, semantic-lifting, and artifact-validation
steps all run in `PAINTMESH_ENV=paintmesh`. The repository's current
`paintmesh` environment does not include LaMa's pinned `webdataset`,
`pytorch_lightning`, and `easydict` stack, so only the two LaMa prediction
processes default to `LAMA_ENV=lama`. If those dependencies are installed and
validated in `paintmesh`, use `LAMA_ENV=paintmesh` to run every process in the
same environment.

Arguments use the same removal identity as `run_remove`:

```text
dataset_name scene resolution target_ids surrounding_ids start_stage
```

The inpaint stages are:

1. Validate the immutable removal result, camera-bound complete tracker
   session, and create a run-local workspace with relative links.
2. Convert the indexed tracker labels to cleaned/dilated binary masks and
   prepare run-local RGB/depth LaMa inputs.
3. Complete RGB and depth, preserve all pixels outside the mask, and validate
   the exact 30-frame shape/finite/hash contract.
4. Use the saved full-precision virtual cameras to back-project the completed
   RGB-D frames into support point clouds.
5. Optimize the inpainted object-aware 3DGS from the selected support frame.
6. Publish an EDGS-loadable model at the final inpaint iteration. The
   published Gaussian PLY is an independent regular-file copy, not a symbolic
   link; `config.yaml`, `cfg_args`, and `classifier.pth` remain managed links.
7. Re-render that model with PGSR and reconstruct a new TSDF mesh.
8. Reclassify the new Gaussians/mesh, create semantic sidecars, and commit the
   final manifest.

The default support frame matches upstream Inpaint360GS (`00004`). It is
configurable without introducing a global path:

```bash
FUSION_SEED_FRAME=4 PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

Resume from an already completed LaMa stage:

```bash
PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 END_STAGE=8 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 4
```

Stages before `start_stage` are verified before reuse. A manifest with changed
inputs, masks, camera poses, model checkpoint, IDs, or parameters is rejected;
use a new `INPAINT_RUN_NAME` for variants:

```bash
INPAINT_RUN_NAME=dilate_16 MASK_DILATION=16 \
  PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

Useful overrides are:

```bash
PAINTMESH_ENV=paintmesh
LAMA_ENV=lama
GPU=0
END_STAGE=8
DISTILL_ITERATION=2000
FINETUNE_ITERATION=5000       # must equal the run-local inpaint config
FUSION_SEED_FRAME=4           # 0 through 29
MASK_MIN_AREA=50
MASK_DILATION=10
RECURSIVE_GUIDE=false
RENDER_INPAINT_VIDEO=false
RENDER_INPAINT_TRAIN=false
RENDER_INPAINT_TEST=false
RENDER_EDGS_TEST=false
WRITE_HOLE_PLY=false
WRITE_COLORED_MESH=true
MAX_DEPTH=5.0
VOXEL_SIZE=0.002
NUM_CLUSTERS=1
INPAINT_RUN_NAME=default
REMOVAL_ROOT=/absolute/removal/run
INPAINT_RUN_ROOT=/absolute/inpaint/run
```

`WRITE_HOLE_PLY=false` avoids 30 unused diagnostic point clouds. The support
point clouds required by optimization are always written. `RENDER_INPAINT_*`
controls only upstream diagnostics; Stage 7 remains the authoritative PGSR
render and mesh reconstruction.

The default output layout is:

```text
output/paintmesh/<dataset>/<scene>/removal/target_<ids>/inpaint/default/
├── work_model/
│   ├── workspace_manifest.json
│   ├── point_cloud -> removal semantic checkpoint
│   ├── point_cloud_object_removal -> removal workspace
│   └── virtual/cameras.json -> tracker/virtual_cameras.json
├── lama/
│   ├── input/{color,depth}/
│   └── output/{color,depth}/
├── fused/
│   └── mask/00000.ply ... 00029.ply
├── inpainted_3dgs/
│   ├── config.yaml
│   ├── cfg_args
│   ├── model_manifest.json
│   ├── point_cloud/iteration_<F>/
│   │   ├── point_cloud.ply                  # independent regular-file copy
│   │   └── classifier.pth
│   └── mesh/ours_<F>/
│       ├── tsdf_fusion_post.ply
│       └── mesh_manifest.json
├── inpainted_mesh/
│   ├── geometry.ply -> ../inpainted_3dgs/mesh/ours_<F>/tsdf_fusion_post.ply
│   ├── semantic_mesh.ply
│   ├── gaussian_instance_id.npy
│   ├── vertex_instance_id.npy
│   ├── face_instance_id.npy
│   └── semantic_manifest.json
├── manifests/
└── inpaint_manifest.json
```

The inpainted mesh is reconstructed from the final inpainted Gaussian scene;
it is not made by patching or relabeling the removed mesh. Target-ID residuals
are recorded as quality statistics in `inpaint_manifest.json` but do not turn
a geometrically valid reconstruction into a false pipeline failure.

## Exact stage commands, inputs, and outputs

The stable entry points are `run_seg`, `run_remove`, and `run_inpaint`; the
extensionless files forward all arguments to their corresponding `.sh`
implementation. Prefer those entry points for production and resume runs.
Besides launching the commands shown below, the runners validate schemas,
hashes, frame sets, and parameters and atomically commit several manifests.
The lower-level commands are documented for debugging and reproducing an
individual algorithmic step.

Run `scripts/paintmesh/run_seg --help`, `run_remove --help`, or
`run_inpaint --help` for the concise CLI reference.

### Command context used below

The examples below describe the same kitchen run used throughout this file.
Run this block from the repository root before copying a lower-level command:

```bash
REPO_ROOT=/home/martin/code/gsagent
EDGS_ROOT="${REPO_ROOT}/submodules/EDGS"
INPAINT_ROOT="${REPO_ROOT}/submodules/Inpaint360GS"
LAMA_ROOT="${INPAINT_ROOT}/LaMa"

PAINTMESH_ENV=paintmesh
LAMA_ENV=lama
GPU=0

DATASET_NAME="mip-nerf/360_v2"
SCENE=kitchen
RESOLUTION=8
TARGET_IDS=14
SURROUNDING_IDS=none
BASE_ITERATION=30000
DISTILL_ITERATION=2000
FINETUNE_ITERATION=5000
FUSION_SEED_FRAME=4
FUSION_SEED_NAME=00004

DATA_ROOT="${REPO_ROOT}/data"
CKPT_ROOT="${REPO_ROOT}/ckpt"
SCENE_ROOT="${DATA_ROOT}/${DATASET_NAME}/${SCENE}"
PIPELINE_ROOT="${REPO_ROOT}/output/paintmesh/${DATASET_NAME}/${SCENE}"
EDGS_MODEL_ROOT="${PIPELINE_ROOT}/edgs"
EDGS_BRIDGE_ROOT="${PIPELINE_ROOT}/edgs_bridge"
SEMANTIC_GS_ROOT="${PIPELINE_ROOT}/semantic_3dgs"
SEMANTIC_MESH_ROOT="${PIPELINE_ROOT}/semantic_mesh"

# This example uses one target and no surrounding IDs. Change the generated
# run directory when the ID sets or RUN_NAME differ.
REMOVAL_ROOT="${PIPELINE_ROOT}/removal/target_14"
REMOVAL_WORK_MODEL="${REMOVAL_ROOT}/work_model"
REMOVED_GS_ROOT="${REMOVAL_ROOT}/removed_3dgs"
REMOVED_MESH_ROOT="${REMOVAL_ROOT}/removed_mesh"

INPAINT_RUN_ROOT="${REMOVAL_ROOT}/inpaint/default"
WORK_MODEL="${INPAINT_RUN_ROOT}/work_model"
MANIFEST_ROOT="${INPAINT_RUN_ROOT}/manifests"
LAMA_INPUT_ROOT="${INPAINT_RUN_ROOT}/lama/input"
LAMA_OUTPUT_ROOT="${INPAINT_RUN_ROOT}/lama/output"
FUSED_ROOT="${INPAINT_RUN_ROOT}/fused/mask"
HOLE_ROOT="${INPAINT_RUN_ROOT}/fused/hole"

run_edgs_py() {
  (
    cd "${EDGS_ROOT}"
    env PYTHONPATH="${EDGS_ROOT}" CUDA_VISIBLE_DEVICES="${GPU}" \
      conda run --no-capture-output -n "${PAINTMESH_ENV}" python "$@"
  )
}

run_inpaint_py() {
  (
    cd "${INPAINT_ROOT}"
    env \
      PYTHONPATH="${INPAINT_ROOT}:${INPAINT_ROOT}/seg/detectron2" \
      CUDA_VISIBLE_DEVICES="${GPU}" \
      conda run --no-capture-output -n "${PAINTMESH_ENV}" python "$@"
  )
}

run_lama_py() {
  local lama_prefix
  lama_prefix="$(conda run --no-capture-output -n "${LAMA_ENV}" \
    python -c 'import sys; print(sys.prefix)')"
  (
    cd "${LAMA_ROOT}"
    env \
      PYTHONPATH="${LAMA_ROOT}" \
      TORCH_HOME="${LAMA_ROOT}" \
      LD_LIBRARY_PATH="${lama_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      CUDA_VISIBLE_DEVICES="${GPU}" \
      conda run --no-capture-output -n "${LAMA_ENV}" python "$@"
  )
}
```

If `GPU` is intentionally unset, omit `CUDA_VISIBLE_DEVICES=...`; that is what
the runners do. A `resolution` of `1` changes the segmentation folder below
from `images_8` to `images`.

### `run_seg`: semantic 3DGS and semantic base mesh

Full invocation:

```bash
PAINTMESH_ENV=paintmesh GPU=0 \
  scripts/paintmesh/run_seg \
  "${DATASET_NAME}" "${SCENE}" "${RESOLUTION}" 1
```

The arguments are `dataset_name scene resolution start_stage`. This runner
has no `END_STAGE`: `start_stage=N` executes Stages N through 6. To debug one
algorithmic stage in isolation, use its lower-level command below and then
resume the runner from the next stage.

| Stage | Primary inputs | Authoritative outputs |
|---:|---|---|
| 1 | scene images, COLMAP model, EDGS/PGSR configuration | `edgs/config.yaml`, iteration-30000 PLY, PGSR renders, `tsdf_fusion_post.ply` |
| bridge | Stage 1 model and mesh | `edgs_bridge/cfg_args`, managed links, `bridge_manifest.json` |
| 2 | resolution-specific images, CropFormer checkpoint | scene-local `raw_hqsam/*.png` and color previews |
| 3 | raw masks, cameras, bridged Gaussians | `associated_hqsam/*.png`, `scene.json`, numbered previews |
| 4 | associated masks, bridge checkpoint | semantic PLY with 16 `obj_dc_*` channels and `classifier.pth` |
| 5 | semantic PLY/classifier and cameras | object-ID/RGB/depth renders under `semantic_3dgs/{train,test}` |
| 6 | semantic PLY/classifier, scene metadata, base TSDF mesh | semantic `.npy` sidecars, palette, optional colored mesh, manifest |

#### Stage 1: train EDGS-PGSR and extract the base TSDF mesh

```bash
run_edgs_py train.py \
  gs=pgsr \
  "train.gs_epochs=${BASE_ITERATION}" \
  train.no_densify=true \
  "gs.opt.iterations=${BASE_ITERATION}" \
  "gs.opt.position_lr_max_steps=${BASE_ITERATION}" \
  gs.opt.pgsr_debug.enabled=false \
  gs.opt.pgsr_debug.interval=200 \
  gs.opt.pgsr_debug.output_dir=debug \
  gs.opt.pgsr_debug.jpeg_quality=95 \
  "gs.dataset.source_path=${SCENE_ROOT}" \
  "gs.dataset.model_path=${EDGS_MODEL_ROOT}" \
  gs.dataset.images=images \
  "gs.dataset.resolution=${RESOLUTION}" \
  gs.dataset.eval=true \
  "gs.opt.save_iterations=[${BASE_ITERATION}]" \
  init_wC.use=true \
  wandb.mode=disabled

run_edgs_py render.py \
  -m "${EDGS_MODEL_ROOT}" \
  --iteration "${BASE_ITERATION}" \
  --renderer pgsr \
  --extract-mesh \
  --max-depth 5.0 \
  --voxel-size 0.002 \
  --num-clusters 1
```

Inputs are `${SCENE_ROOT}/images/` and `${SCENE_ROOT}/sparse/0/`. Outputs are
the final Gaussian PLY, train/test render directories, and
`${EDGS_MODEL_ROOT}/mesh/ours_${BASE_ITERATION}/tsdf_fusion_post.ply`. Add
`--use-depth-filter` to the render command when `USE_DEPTH_FILTER=true`.
Debug-training overrides are listed earlier in this README.

#### Shared bridge step

The runner executes this idempotent binding/validation step on every run,
including `start_stage=6`:

```bash
run_inpaint_py tools/build_edgs_bridge.py \
  --edgs-model "${EDGS_MODEL_ROOT}" \
  --iteration "${BASE_ITERATION}" \
  --output "${EDGS_BRIDGE_ROOT}"
```

It consumes the Stage 1 config, PLY, and mesh, and creates `cfg_args`, managed
relative links, and `bridge_manifest.json` under `edgs_bridge/`.

#### Stage 2: generate per-view CropFormer masks

```bash
run_inpaint_py seg/raw_mask_sam.py \
  --dataset_path "${DATA_ROOT}/${DATASET_NAME}" \
  --scene_name "${SCENE}" \
  --image_folder images_8 \
  --method hqsam \
  --threshold 0.5
```

The input is `${SCENE_ROOT}/images_8/` plus
`${CKPT_ROOT}/CropFormer_hornet_3x_03823a.pth`. The runner exposes the
checkpoint at `seg/weight/` with a managed link. Outputs are scene-local
`raw_hqsam/` integer-ID masks and `raw_hqsam_color/` previews. This command
reprocesses the complete image set; after an OOM, remove only demonstrably
partial/stale masks or rerun the command and let it overwrite matching names.

#### Stage 3: associate IDs across views and label previews

```bash
run_inpaint_py seg/mask_associate.py \
  --source_path "${SCENE_ROOT}" \
  --model_path "${EDGS_BRIDGE_ROOT}" \
  --iteration "${BASE_ITERATION}" \
  --images images \
  --resolution "${RESOLUTION}" \
  --mask_generator hqsam \
  --patch 16 \
  --eval

run_inpaint_py tools/add_label_num_hqsam.py \
  --source_path "${SCENE_ROOT}" \
  --resolution "${RESOLUTION}" \
  --mask_generator hqsam
```

Inputs are the raw masks, COLMAP cameras, and bridged Gaussian model. Outputs
are `associated_hqsam/*.png`, `associated_hqsam/scene.json`, color masks, and
`images_8_num/` previews (`images_num/` at resolution 1). Choose removal IDs
from these numbered previews.

#### Stage 4: distill 16-D object features into Gaussians

```bash
run_inpaint_py seg/distillation.py \
  --source_path "${SCENE_ROOT}" \
  --model_path "${SEMANTIC_GS_ROOT}" \
  --vanilla_3dgs_path "${EDGS_BRIDGE_ROOT}" \
  --images images \
  --resolution "${RESOLUTION}" \
  --object_path associated_hqsam \
  --config_file "${INPAINT_ROOT}/config/object_distill/train_distill.json" \
  --test_iterations "${DISTILL_ITERATION}" \
  --save_iterations "${DISTILL_ITERATION}" \
  --checkpoint_iterations "${DISTILL_ITERATION}" \
  --eval
```

Inputs are the bridge model and associated masks. Outputs are
`semantic_3dgs/cfg_args`, `chkpnt<D>.pth`, and
`point_cloud/iteration_<D>/{point_cloud.ply,classifier.pth}`. The runner
checks identical XYZ ordering against the bridge, exactly 16 object channels,
and classifier shape `[num_classes, 16]`.

#### Stage 5: render the semantic 3DGS

```bash
run_inpaint_py render.py \
  --model_path "${SEMANTIC_GS_ROOT}" \
  --iteration "${DISTILL_ITERATION}" \
  --skip_fused_ply
```

Add `--render_video` when `RENDER_VIDEO=true`. Inputs are the Stage 4 model,
classifier, associated masks, and cameras. Outputs include RGB, depth,
`objects_pred/`, and colored object maps in
`semantic_3dgs/{train,test}/ours_<D>/`; the optional video is written below
`semantic_3dgs/video/ours_<D>/`.

#### Stage 6: lift Gaussian semantics onto the base mesh

```bash
run_inpaint_py tools/lift_gaussian_semantics_to_mesh.py \
  --gaussian-ply \
    "${SEMANTIC_GS_ROOT}/point_cloud/iteration_${DISTILL_ITERATION}/point_cloud.ply" \
  --classifier \
    "${SEMANTIC_GS_ROOT}/point_cloud/iteration_${DISTILL_ITERATION}/classifier.pth" \
  --scene-info "${SCENE_ROOT}/associated_hqsam/scene.json" \
  --mesh \
    "${EDGS_MODEL_ROOT}/mesh/ours_${BASE_ITERATION}/tsdf_fusion_post.ply" \
  --output-dir "${SEMANTIC_MESH_ROOT}" \
  --neighbors 8 \
  --chunk-size 32768 \
  --workers -1 \
  --opacity-min 0.01 \
  --support-sigma 3.0 \
  --normal-power 2.0 \
  --min-confidence 0.10 \
  --min-margin 0.02 \
  --unknown-id 65535 \
  --overwrite \
  --write-colored-ply
```

Inputs are the semantic PLY/classifier, `scene.json`, and base mesh. Outputs
are Gaussian/vertex/face ID and confidence arrays, `palette.json`, optional
`semantic_mesh.ply`, and the atomic `semantic_manifest.json`. Omit
`--write-colored-ply` when `WRITE_COLORED_MESH=false`.

### `run_remove`: remove selected objects and reconstruct their complement

Full invocation:

```bash
PAINTMESH_ENV=paintmesh GPU=0 END_STAGE=5 \
  scripts/paintmesh/run_remove \
  "${DATASET_NAME}" "${SCENE}" "${RESOLUTION}" \
  "${TARGET_IDS}" "${SURROUNDING_IDS}" 1
```

The arguments are `dataset_name scene resolution target_ids surrounding_ids
start_stage`. `target_ids` is required. Each ID set is a positive integer or
comma-separated list; `surrounding_ids` may be `none`. Use the same ID sets in
`run_remove` and `run_inpaint`.

Unlike `run_seg`, this runner supports `END_STAGE`. These commands execute one
stage while preserving all runner-level contract checks:

```bash
# Stage 1
END_STAGE=1 scripts/paintmesh/run_remove \
  "${DATASET_NAME}" "${SCENE}" "${RESOLUTION}" \
  "${TARGET_IDS}" "${SURROUNDING_IDS}" 1

# Stage 2
END_STAGE=2 scripts/paintmesh/run_remove \
  "${DATASET_NAME}" "${SCENE}" "${RESOLUTION}" \
  "${TARGET_IDS}" "${SURROUNDING_IDS}" 2

# Stage 3
END_STAGE=3 LAUNCH_REFINER=false scripts/paintmesh/run_remove \
  "${DATASET_NAME}" "${SCENE}" "${RESOLUTION}" \
  "${TARGET_IDS}" "${SURROUNDING_IDS}" 3

# Stage 4
END_STAGE=4 LAUNCH_REFINER=false scripts/paintmesh/run_remove \
  "${DATASET_NAME}" "${SCENE}" "${RESOLUTION}" \
  "${TARGET_IDS}" "${SURROUNDING_IDS}" 4

# Stage 5
END_STAGE=5 scripts/paintmesh/run_remove \
  "${DATASET_NAME}" "${SCENE}" "${RESOLUTION}" \
  "${TARGET_IDS}" "${SURROUNDING_IDS}" 5
```

Prefix each command with `PAINTMESH_ENV=paintmesh GPU=0` as needed.

| Stage | Primary inputs | Authoritative outputs |
|---:|---|---|
| 1 | bridge manifest, semantic checkpoint/manifest, selected IDs | run-local configs, linked workspace, `workspace_manifest.json` |
| 2 | workspace, semantic PLY/classifier, removal config | removed work PLY, published `removed_3dgs`, model manifest |
| 3 | published removed model, dataset cameras, `scene.json` | PGSR renders/TSDF mesh, semantic sidecars, removal manifest |
| 4 | work model, removed result, removal config | 30 full/removal virtual RGB-D views, `images.zip`, camera manifest, in-progress session |
| 5 | tracker archive/cameras/session and three tracker checkpoints | 30 base masks and a complete `tracking_session.json` |

#### Stage 1: initialize configs and the isolated removal workspace

The first command runs only when Stage 1 is selected. The second is an
idempotent binding step and is also run when later stages are resumed.

```bash
run_inpaint_py tools/init_configs.py \
  --dataset_name "${DATASET_NAME}" \
  --scene "${SCENE}" \
  --target_id "${TARGET_IDS}" \
  --target_surronding_id "${SURROUNDING_IDS}" \
  --removal_thresh 0.7 \
  --output_root "${REMOVAL_ROOT}/config"

run_inpaint_py tools/prepare_removal_workspace.py \
  --semantic-model "${SEMANTIC_GS_ROOT}" \
  --iteration "${DISTILL_ITERATION}" \
  --bridge-manifest "${EDGS_BRIDGE_ROOT}/bridge_manifest.json" \
  --semantic-manifest "${SEMANTIC_MESH_ROOT}/semantic_manifest.json" \
  --output "${REMOVAL_WORK_MODEL}"
```

Inputs are the complete `run_seg` result and selected IDs. Outputs are
`config/object_{removal,inpaint}/<dataset>/<scene>.json`, managed workspace
links, and `work_model/workspace_manifest.json`. The misspelling in
`--target_surronding_id` is the upstream-compatible CLI spelling.

#### Stage 2: remove Gaussians and publish the removed model

```bash
REMOVAL_CONFIG="${REMOVAL_ROOT}/config/object_removal/${DATASET_NAME}/${SCENE}.json"

run_inpaint_py edit_object_removal.py \
  --source_path "${SCENE_ROOT}" \
  --model_path "${REMOVAL_WORK_MODEL}" \
  --reference_model_path "${SEMANTIC_GS_ROOT}" \
  --iteration "${DISTILL_ITERATION}" \
  --resolution "${RESOLUTION}" \
  --config_file "${REMOVAL_CONFIG}" \
  --skip_train \
  --skip_test \
  --skip_debug_ply

FINAL_REMOVED_PLY="${REMOVAL_WORK_MODEL}/point_cloud_object_removal/iteration_${DISTILL_ITERATION}/point_cloud.ply"

run_inpaint_py tools/publish_removed_edgs_model.py \
  --removed-ply "${FINAL_REMOVED_PLY}" \
  --classifier \
    "${SEMANTIC_GS_ROOT}/point_cloud/iteration_${DISTILL_ITERATION}/classifier.pth" \
  --edgs-config "${EDGS_MODEL_ROOT}/config.yaml" \
  --cfg-args "${SEMANTIC_GS_ROOT}/cfg_args" \
  --iteration "${DISTILL_ITERATION}" \
  --target-ids "${TARGET_IDS}" \
  --surrounding-ids "${SURROUNDING_IDS}" \
  --bridge-manifest "${EDGS_BRIDGE_ROOT}/bridge_manifest.json" \
  --semantic-manifest "${SEMANTIC_MESH_ROOT}/semantic_manifest.json" \
  --removal-threshold 0.7 \
  --output "${REMOVED_GS_ROOT}"
```

With surrounding IDs, the published PLY instead comes from
`iteration_<D>_removal_target/point_cloud.ply`; that checkpoint restores the
temporary occluders while keeping targets removed. Inputs are the workspace,
removal config, semantic checkpoint, and manifests. Outputs are per-object and
combined work PLYs plus the EDGS-loadable `removed_3dgs/` model and
`model_manifest.json`. Enable optional train/test/video/debug outputs on the
first immutable Stage 2 run; the runner refuses to claim them later when it
only reuses an existing published model.

#### Stage 3: rebuild the removed PGSR mesh and lift remaining semantics

```bash
run_edgs_py render.py \
  --model-path "${REMOVED_GS_ROOT}" \
  --iteration "${DISTILL_ITERATION}" \
  --renderer pgsr \
  --source-path "${SCENE_ROOT}" \
  --images images \
  --resolution "${RESOLUTION}" \
  --extract-mesh \
  --max-depth 5.0 \
  --voxel-size 0.002 \
  --num-clusters 1 \
  --skip-test

run_inpaint_py tools/lift_gaussian_semantics_to_mesh.py \
  --gaussian-ply \
    "${REMOVED_GS_ROOT}/point_cloud/iteration_${DISTILL_ITERATION}/point_cloud.ply" \
  --classifier \
    "${REMOVED_GS_ROOT}/point_cloud/iteration_${DISTILL_ITERATION}/classifier.pth" \
  --scene-info "${SCENE_ROOT}/associated_hqsam/scene.json" \
  --mesh \
    "${REMOVED_GS_ROOT}/mesh/ours_${DISTILL_ITERATION}/tsdf_fusion_post.ply" \
  --output-dir "${REMOVED_MESH_ROOT}" \
  --neighbors 8 \
  --chunk-size 32768 \
  --workers -1 \
  --opacity-min 0.01 \
  --support-sigma 3.0 \
  --normal-power 2.0 \
  --min-confidence 0.10 \
  --min-margin 0.02 \
  --unknown-id 65535 \
  --overwrite \
  --write-colored-ply
```

Omit `--skip-test` when `RENDER_EDGS_TEST=true`, add
`--use-depth-filter` when requested, and omit `--write-colored-ply` when
disabled. Inputs are the published removed model, cameras, and scene metadata.
Outputs are PGSR renders, the raw/postprocessed TSDF meshes, semantic mesh
sidecars, `removed_mesh/geometry.ply`, mesh/semantic manifests, and the
top-level `removal_manifest.json`. The last manifest operations are embedded
in `run_remove.sh`; use the stage runner, not only the two commands above, to
produce the complete contract.

#### Stage 4: render and package 30 exact virtual views

```bash
run_inpaint_py tools/virtual_pose.py \
  --source_path "${SCENE_ROOT}" \
  --model_path "${REMOVAL_WORK_MODEL}" \
  --iteration "${DISTILL_ITERATION}" \
  --resolution "${RESOLUTION}" \
  --config_file "${REMOVAL_CONFIG}" \
  --tracker_archive "${REMOVAL_ROOT}/tracker/images.zip" \
  --camera_manifest "${REMOVAL_ROOT}/tracker/virtual_cameras.json"
```

Inputs are the full and removed work checkpoints, cameras, and removal config.
Outputs are 30 full-scene views below `virtual/ours_<D>/`, 30 removed views
below `virtual/ours_object_removal/iteration_<D>/`, a root-level
`00000.png`-through-`00029.png` archive, and a full-precision camera manifest.
The runner invalidates old masks before this command, validates the archive,
and creates a camera/archive-bound `tracking_session.json` with
`status=in_progress`.

#### Stage 5: refine and track the removal mask interactively

The runner links these inputs from `${CKPT_ROOT}` into the tracker checkout:

```text
sam_vit_b_01ec64.pth
R50_DeAOTL_PRE_YTB_DAV.pth
groundingdino_swint_ogc.pth
```

The underlying server command is:

```bash
TRACKER_ROOT="${INPAINT_ROOT}/Segment-and-Track-Anything"
(
  cd "${TRACKER_ROOT}"
  env \
    PYTHONPATH="${TRACKER_ROOT}" \
    TRACKER_IMAGE_SEQUENCE="${REMOVAL_ROOT}/tracker/images.zip" \
    TRACKER_ASSETS_ROOT="${REMOVAL_ROOT}/tracker/assets" \
    TRACKING_RESULTS_ROOT="${REMOVAL_ROOT}/tracker/results" \
    GRADIO_SERVER_NAME=127.0.0.1 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_SHARE=false \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    conda run --no-capture-output -n "${PAINTMESH_ENV}" python -u app.py
)
```

Open `http://127.0.0.1:7860`, select the **Image-Seq** tab, load/extract the
run-local `images.zip`, mark the target on frame `00000`, initialize, and
start tracking. Wait until all 30 base masks exist, then press `Ctrl+C`. The
runner accepts exit status 0, 1, or 130 only after validating exact names,
dimensions, timestamps, and hashes. Authoritative outputs are
`tracker/results/images/images_masks/00000.png` through `00029.png` and a
complete `tracking_session.json`; videos, GIFs, masked frames, and ZIP files
are auxiliary. To intentionally discard a prior mask session, rerun Stage 4.

### `run_inpaint`: complete RGB-D, optimize 3DGS, and rebuild the mesh

Full invocation:

```bash
PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 END_STAGE=8 \
  scripts/paintmesh/run_inpaint \
  "${DATASET_NAME}" "${SCENE}" "${RESOLUTION}" \
  "${TARGET_IDS}" "${SURROUNDING_IDS}" 1
```

The arguments are `dataset_name scene resolution target_ids surrounding_ids
start_stage`. `END_STAGE` selects the last stage. The removal identity must
match `run_remove`. Stage 1 always refreshes or validates the isolated
workspace; earlier data/manifest stages are validated before later stages are
allowed to reuse their outputs.

Use this pattern to run one stage with all runner-level checks:

```bash
# Replace N with 1..8.
PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 END_STAGE=N \
  scripts/paintmesh/run_inpaint \
  "${DATASET_NAME}" "${SCENE}" "${RESOLUTION}" \
  "${TARGET_IDS}" "${SURROUNDING_IDS}" N
```

| Stage | Primary inputs | Authoritative outputs |
|---:|---|---|
| 1 | complete removal/tracker/camera contracts and 30 masks | isolated `work_model` plus workspace manifest |
| 2 | tracking masks, removed RGB/depth, reference depth | run-local LaMa RGB/depth inputs and input manifest |
| 3 | LaMa inputs and `big-lama` checkpoint | completed RGB/depth plus completion manifest |
| 4 | completed RGB-D, masks, exact virtual cameras | 30 support PLYs and fusion manifest |
| 5 | removal workspace, support PLYs, completion data/config | optimized inpaint work PLY at final iteration |
| 6 | work PLY, classifier, upstream manifests/config | EDGS-loadable published model and model manifest |
| 7 | published model, dataset cameras, PGSR settings | renders, TSDF mesh, mesh manifest |
| 8 | published PLY/classifier, mesh, scene metadata | semantic sidecars/mesh and final inpaint manifest |

The final iteration is read from
`${REMOVAL_ROOT}/config/object_inpaint/${DATASET_NAME}/${SCENE}.json`.
`FINETUNE_ITERATION`, when supplied, is a consistency assertion and must equal
that JSON value.

#### Stage 1: prepare the isolated inpaint workspace

```bash
run_inpaint_py tools/prepare_inpaint_workspace.py \
  --removal-workspace "${REMOVAL_WORK_MODEL}" \
  --removal-manifest "${REMOVAL_ROOT}/removal_manifest.json" \
  --tracking-session "${REMOVAL_ROOT}/tracker/tracking_session.json" \
  --camera-manifest "${REMOVAL_ROOT}/tracker/virtual_cameras.json" \
  --tracking-masks \
    "${REMOVAL_ROOT}/tracker/results/images/images_masks" \
  --source-iteration "${DISTILL_ITERATION}" \
  --output "${WORK_MODEL}"
```

Inputs are the complete removal manifest, camera-bound tracker session,
camera manifest, removal workspace, and 30 exact masks. Outputs are managed
relative links/copies below `work_model/` and
`work_model/workspace_manifest.json`. This command also runs as a cheap
refresh/validation step when `start_stage` is greater than 1.

#### Stage 2: prepare isolated LaMa color/depth inputs

```bash
run_inpaint_py tools/prepare_paintmesh_lama_data.py prepare \
  --tracking-masks "${WORK_MODEL}/tracking_masks" \
  --removed-rgb \
    "${WORK_MODEL}/virtual/ours_object_removal/iteration_${DISTILL_ITERATION}/renders" \
  --removed-depth \
    "${WORK_MODEL}/virtual/ours_object_removal/iteration_${DISTILL_ITERATION}/depth" \
  --reference-depth \
    "${WORK_MODEL}/virtual/ours_${DISTILL_ITERATION}/depth" \
  --color-input "${LAMA_INPUT_ROOT}/color" \
  --depth-input "${LAMA_INPUT_ROOT}/depth" \
  --manifest "${MANIFEST_ROOT}/lama_input_manifest.json" \
  --frames 30 \
  --min-area 50 \
  --dilation 10
```

Inputs are masks plus removed/reference RGB-D virtual renders. Outputs are 30
color images and masks, 30 depth arrays and masks, preserved original depths,
and `lama_input_manifest.json`. `MASK_MIN_AREA` and `MASK_DILATION` replace the
two numeric values in the runner.

#### Stage 3: complete and validate RGB/depth with LaMa

```bash
run_lama_py bin/predict_color.py \
  --input-dir "${LAMA_INPUT_ROOT}/color" \
  --output-dir "${LAMA_OUTPUT_ROOT}/color" \
  --model-path "${CKPT_ROOT}/big-lama"

run_lama_py bin/predict_depth.py \
  --input-dir "${LAMA_INPUT_ROOT}/depth" \
  --output-dir "${LAMA_OUTPUT_ROOT}/depth" \
  --model-path "${CKPT_ROOT}/big-lama"

run_inpaint_py tools/prepare_paintmesh_lama_data.py validate-output \
  --color-input "${LAMA_INPUT_ROOT}/color" \
  --depth-input "${LAMA_INPUT_ROOT}/depth" \
  --color-output "${LAMA_OUTPUT_ROOT}/color" \
  --depth-output "${LAMA_OUTPUT_ROOT}/depth" \
  --model-path "${CKPT_ROOT}/big-lama" \
  --input-manifest "${MANIFEST_ROOT}/lama_input_manifest.json" \
  --manifest "${MANIFEST_ROOT}/lama_completion_manifest.json" \
  --frames 30
```

When `RECURSIVE_GUIDE=true`, append `--recursive-guide` to the color and
validation commands. Inputs are Stage 2 data and
`ckpt/big-lama/{config.yaml,models/best.ckpt}`. Outputs are exactly 30 color
PNGs, 30 finite depth arrays, and `lama_completion_manifest.json`. A matched
completion manifest causes validation/reuse instead of another prediction.

#### Stage 4: back-project completed RGB-D into support point clouds

```bash
INPAINT_CONFIG="${REMOVAL_ROOT}/config/object_inpaint/${DATASET_NAME}/${SCENE}.json"

run_inpaint_py edit_object_removal_plyfusion.py \
  --source_path "${SCENE_ROOT}" \
  --model_path "${WORK_MODEL}" \
  --resolution "${RESOLUTION}" \
  --iteration "${DISTILL_ITERATION}" \
  --source_iteration "${DISTILL_ITERATION}" \
  --config_file "${INPAINT_CONFIG}" \
  --inpainted_color_dir "${LAMA_OUTPUT_ROOT}/color" \
  --inpaint_mask_dir "${LAMA_INPUT_ROOT}/color" \
  --completed_depth_dir "${LAMA_OUTPUT_ROOT}/depth" \
  --fused_output_dir "${FUSED_ROOT}" \
  --hole_output_dir "${HOLE_ROOT}" \
  --camera_manifest "${REMOVAL_ROOT}/tracker/virtual_cameras.json" \
  --lama_manifest "${MANIFEST_ROOT}/lama_completion_manifest.json" \
  --manifest "${MANIFEST_ROOT}/fusion_manifest.json" \
  --skip_hole_ply
```

Inputs are the completed color/depth, masks, exact cameras, and config.
Outputs are `fused/mask/00000.ply` through `00029.ply` and
`fusion_manifest.json`. Remove `--skip_hole_ply` when `WRITE_HOLE_PLY=true` to
also write `fused/hole/*.ply`. The default Stage 5 support file is
`fused/mask/00004.ply`.

#### Stage 5: optimize the object-aware inpainted 3DGS

```bash
run_inpaint_py edit_object_inpaint.py \
  --source_path "${SCENE_ROOT}" \
  --model_path "${WORK_MODEL}" \
  --images images \
  --resolution "${RESOLUTION}" \
  --iteration "${DISTILL_ITERATION}" \
  --source_iteration "${DISTILL_ITERATION}" \
  --config_file "${INPAINT_CONFIG}" \
  --supp_ply "${FUSED_ROOT}/${FUSION_SEED_NAME}.ply" \
  --fusion_dir "${FUSED_ROOT}" \
  --fusion_seed_frame "${FUSION_SEED_FRAME}" \
  --inpaint_output_ply \
    "${WORK_MODEL}/point_cloud_object_inpaint_virtual/iteration_${FINETUNE_ITERATION}/point_cloud.ply" \
  --inpainted_color_dir "${LAMA_OUTPUT_ROOT}/color" \
  --inpaint_mask_dir "${LAMA_INPUT_ROOT}/color" \
  --camera_manifest "${REMOVAL_ROOT}/tracker/virtual_cameras.json" \
  --removal_source_iteration "${DISTILL_ITERATION}" \
  --skip_surrounding_filter \
  --skip_train \
  --skip_test
```

Remove the corresponding skip flag for `RENDER_INPAINT_TRAIN=true` or
`RENDER_INPAINT_TEST=true`; add `--render_video` for
`RENDER_INPAINT_VIDEO=true`. Inputs are the prepared workspace, all support
PLYs, chosen support frame, completed RGB, masks, cameras, and run-local
config. The authoritative output is
`work_model/point_cloud_object_inpaint_virtual/iteration_<F>/point_cloud.ply`,
which includes the 16 semantic `obj_dc_*` properties.

#### Stage 6: publish an EDGS-loadable inpainted model

The runner reads the exact EDGS config path from
`removed_3dgs/model_manifest.json`; the example resolves to the scene EDGS
config:

```bash
run_inpaint_py tools/publish_inpainted_edgs_model.py \
  --inpainted-ply \
    "${WORK_MODEL}/point_cloud_object_inpaint_virtual/iteration_${FINETUNE_ITERATION}/point_cloud.ply" \
  --classifier \
    "${WORK_MODEL}/point_cloud/iteration_${DISTILL_ITERATION}/classifier.pth" \
  --edgs-config "${EDGS_MODEL_ROOT}/config.yaml" \
  --cfg-args "${WORK_MODEL}/cfg_args" \
  --source-iteration "${DISTILL_ITERATION}" \
  --output-iteration "${FINETUNE_ITERATION}" \
  --target-ids "${TARGET_IDS}" \
  --surrounding-ids "${SURROUNDING_IDS}" \
  --removed-model-manifest "${REMOVED_GS_ROOT}/model_manifest.json" \
  --removal-manifest "${REMOVAL_ROOT}/removal_manifest.json" \
  --workspace-manifest "${WORK_MODEL}/workspace_manifest.json" \
  --tracking-session "${REMOVAL_ROOT}/tracker/tracking_session.json" \
  --lama-manifest "${MANIFEST_ROOT}/lama_completion_manifest.json" \
  --fusion-manifest "${MANIFEST_ROOT}/fusion_manifest.json" \
  --fusion-seed-frame "${FUSION_SEED_FRAME}" \
  --inpaint-config "${INPAINT_CONFIG}" \
  --output "${INPAINT_RUN_ROOT}/inpainted_3dgs"
```

Inputs are the Stage 5 PLY, classifier/config, and every upstream completion
manifest. Outputs are `config.yaml`, `cfg_args`, `model_manifest.json`, and
`point_cloud/iteration_<F>/{point_cloud.ply,classifier.pth}`. The published
`point_cloud.ply` is an atomically written, inode-independent regular copy of
the Stage 5 PLY. It deliberately retains the 16 semantic feature properties;
the other small shared artifacts remain managed links. The model manifest
records `point_cloud_storage.type=regular_copy` and its SHA-256.

#### Stage 7: PGSR rendering and inpainted TSDF reconstruction

```bash
INPAINTED_GS_ROOT="${INPAINT_RUN_ROOT}/inpainted_3dgs"
INPAINTED_PLY="${INPAINTED_GS_ROOT}/point_cloud/iteration_${FINETUNE_ITERATION}/point_cloud.ply"
INPAINTED_RAW_MESH="${INPAINTED_GS_ROOT}/mesh/ours_${FINETUNE_ITERATION}/tsdf_fusion_post.ply"

run_edgs_py render.py \
  --model-path "${INPAINTED_GS_ROOT}" \
  --source-path "${SCENE_ROOT}" \
  --images images \
  --resolution "${RESOLUTION}" \
  --iteration "${FINETUNE_ITERATION}" \
  --renderer pgsr \
  --extract-mesh \
  --max-depth 5.0 \
  --voxel-size 0.002 \
  --num-clusters 1 \
  --skip-test

run_inpaint_py tools/publish_inpaint_mesh.py \
  --model-manifest "${INPAINTED_GS_ROOT}/model_manifest.json" \
  --gaussian-ply "${INPAINTED_PLY}" \
  --mesh "${INPAINTED_RAW_MESH}" \
  --train-render-manifest \
    "${INPAINTED_GS_ROOT}/train/ours_${FINETUNE_ITERATION}/render_manifest.json" \
  --output \
    "${INPAINTED_GS_ROOT}/mesh/ours_${FINETUNE_ITERATION}/mesh_manifest.json" \
  --iteration "${FINETUNE_ITERATION}" \
  --source-path "${SCENE_ROOT}" \
  --images images \
  --resolution "${RESOLUTION}" \
  --max-depth 5.0 \
  --voxel-size 0.002 \
  --num-clusters 1
```

For `RENDER_EDGS_TEST=true`, remove `--skip-test` and add
`--test-render-manifest <.../test/ours_F/render_manifest.json>` to the publish
command. Add `--use-depth-filter` to both relevant commands when enabled.
Inputs are the published Stage 6 model and dataset cameras. Outputs are train
(and optional test) renders, `tsdf_fusion.ply`, `tsdf_fusion_post.ply`, and
`mesh_manifest.json`. A manifest-matched mesh/render set is validated and
reused.

#### Stage 8: lift semantics and commit the final inpaint result

```bash
INPAINTED_MESH_ROOT="${INPAINT_RUN_ROOT}/inpainted_mesh"

run_inpaint_py tools/lift_gaussian_semantics_to_mesh.py \
  --gaussian-ply "${INPAINTED_PLY}" \
  --classifier \
    "${INPAINTED_GS_ROOT}/point_cloud/iteration_${FINETUNE_ITERATION}/classifier.pth" \
  --scene-info "${SCENE_ROOT}/associated_hqsam/scene.json" \
  --mesh "${INPAINTED_RAW_MESH}" \
  --output-dir "${INPAINTED_MESH_ROOT}" \
  --neighbors 8 \
  --chunk-size 32768 \
  --workers -1 \
  --opacity-min 0.01 \
  --support-sigma 3.0 \
  --normal-power 2.0 \
  --min-confidence 0.10 \
  --min-margin 0.02 \
  --unknown-id 65535 \
  --overwrite \
  --write-colored-ply

# The runner atomically creates inpainted_mesh/geometry.ply as a relative link
# to INPAINTED_RAW_MESH before executing this final commit command.
run_inpaint_py tools/finalize_inpaint_result.py \
  --model-manifest "${INPAINTED_GS_ROOT}/model_manifest.json" \
  --mesh-manifest \
    "${INPAINTED_GS_ROOT}/mesh/ours_${FINETUNE_ITERATION}/mesh_manifest.json" \
  --semantic-manifest "${INPAINTED_MESH_ROOT}/semantic_manifest.json" \
  --removal-manifest "${REMOVAL_ROOT}/removal_manifest.json" \
  --workspace-manifest "${WORK_MODEL}/workspace_manifest.json" \
  --lama-manifest "${MANIFEST_ROOT}/lama_completion_manifest.json" \
  --fusion-manifest "${MANIFEST_ROOT}/fusion_manifest.json" \
  --gaussian-ply "${INPAINTED_PLY}" \
  --mesh "${INPAINTED_RAW_MESH}" \
  --geometry "${INPAINTED_MESH_ROOT}/geometry.ply" \
  --output "${INPAINT_RUN_ROOT}/inpaint_manifest.json"
```

Inputs are the published Gaussian model/classifier, new mesh, scene metadata,
and all upstream manifests. Outputs are Gaussian/vertex/face label and
confidence arrays, `palette.json`, optional `semantic_mesh.ply`, a geometry
link, `semantic_manifest.json`, and the final atomic `inpaint_manifest.json`.

### Resume, variants, and a complete rerun

`start_stage` means “validate earlier contracts and begin work here”; it is
not an overwrite switch. Published directories are immutable with respect to
the inputs and parameters recorded by their manifests. If a changed Stage 5
PLY, mask, camera, checkpoint, or setting is pointed at an existing publish
directory, the expected error is:

```text
belongs to different parameters or inputs; choose a new output directory
```

For a variant, keep the old run and select a new name:

```bash
INPAINT_RUN_NAME=dilate_16 MASK_DILATION=16 \
  PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 END_STAGE=8 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

To completely rerun and replace the logical `default` result, move the entire
precise run directory aside, then start at Stage 1. Do not delete only one
manifest, because that can leave later products bound to an older PLY:

```bash
RUN_ROOT="/home/martin/code/gsagent/output/paintmesh/mip-nerf/360_v2/kitchen/removal/target_14/inpaint/default"
BACKUP="${RUN_ROOT}_before_full_rerun_$(date +%Y%m%d_%H%M%S)"

mv -- "${RUN_ROOT}" "${BACKUP}"

PAINTMESH_ENV=paintmesh LAMA_ENV=lama GPU=0 END_STAGE=8 \
  scripts/paintmesh/run_inpaint \
  "mip-nerf/360_v2" kitchen 8 14 none 1
```

Use the same whole-directory rule for a deliberate full `run_remove` reset:
move only the exact `${PIPELINE_ROOT}/removal/<run-name>` directory, never the
scene pipeline root. Rerunning `run_remove` Stage 4 intentionally invalidates
the old tracker binding and is the supported way to start a fresh mask
annotation session.
