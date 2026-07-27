# GaussianWrappingGaga Mip-NeRF 360 launchers

These launchers use associated Gaga masks produced by:

```bash
python scripts/gaga/preprocess_gaga_masks_mipnerf360.py ...
python scripts/gaga/associate_gaga_masks_mipnerf360.py ...
```

Masks are read from:

```text
data/mip-nerf/360_v2/<scene>/sam_mask/
data/mip-nerf/360_v2/<scene>/entityseg_mask/
```

## Training

Recommended stable two-stage training:

```bash
python scripts/GaussianWrappingGaga/train_gaussian_wrapping_gaga_mipnerf360.py \
  --scene counter \
  --mode two-stage \
  --mask-method entityseg \
  --rasterizer radegs \
  --gpu 0
```

This first trains complete Gaussian Wrapping geometry for 30,000 iterations,
then freezes geometry and lifts 16D Gaga semantics for 10,000 iterations.

Synchronous joint training:

```bash
python scripts/GaussianWrappingGaga/train_gaussian_wrapping_gaga_mipnerf360.py \
  --scene counter \
  --mode joint \
  --mask-method entityseg \
  --lambda-semantic 1.0 \
  --lambda-semantic-3d 0.01 \
  --gpu 0
```

Joint mode performs RGB/geometry and semantic optimization in the same loop.
Semantic CUDA backward remains embedding-only, while Gaussian Wrapping losses
continue to update geometry and appearance. Densification, split, clone and
prune carry the 16D semantic field with the parent Gaussian.

Both modes automatically render the final semantic checkpoint after training.
The default render resolution is `-r 2`, matching the Gaga Mip-NeRF 360
convention. Semantic classification is performed in pixel chunks, so scenes
with hundreds of associated instances do not allocate a full
`num_classes × height × width` CUDA logits tensor.

```text
outputs/gaussian_wrapping_gaga_mipnerf360/
  <scene>/<mode>/<mask-method>/
    train/ours_<semantic-iteration>/
      renders/ gt/
      objects_feature16/ objects_pred/ objects_test/
      gt_objects/ gt_objects_color/
      ground_truth/ depth/ expected_depth/ median_depth/
      normal/ alpha/ semantic_labels/ semantic_color/
    test/ours_<semantic-iteration>/
      renders/ gt/
      objects_feature16/ objects_pred/ objects_test/
      ground_truth/ depth/ expected_depth/ median_depth/
      normal/ alpha/ semantic_labels/ semantic_color/
    render_manifest.json
```

The first group of directories is Gaga-compatible. `objects_test` and
`gt_objects` are lossless uint16 label PNGs. The remaining PNG directories
retain Gaussian Wrapping's geometry diagnostics. Use
`--render-output-profile full` to additionally export numerical depth, normal,
coordinates, renderer auxiliaries, 16D features, and chunked float16 logits.
Use `--no-render-after-train` to disable the post-training stage or
`--render-resolution` and `--render-class-chunk-size` to tune it.

Repeat `--scene`, pass comma-separated scenes, or use `--scene all`.
Additional native `train.py` flags may be supplied after `--`.

## Mesh extraction

```bash
python scripts/GaussianWrappingGaga/extract_gaussian_wrapping_gaga_mesh.py \
  --scene counter \
  --mode two-stage \
  --mask-method entityseg \
  --gpu 0
```

The command exports:

```text
outputs/gaussian_wrapping_gaga_mipnerf360/
  <scene>/<mode>/<mask-method>/
    geometry/ or model/
    semantic/                 # two-stage only
    run_manifest.json
    mesh/
      <rasterizer>_iteration_<N>.ply
      <rasterizer>_iteration_<N>_semantic.ply
      <rasterizer>_iteration_<N>_semantic.semantic.npy
      <rasterizer>_iteration_<N>_semantic.semantic_distance.npy
      <rasterizer>_iteration_<N>_semantic.semantic.json
```

Add `--texture` for Gaussian Wrapping texture refinement. Mesh simplification
is available through `--target-faces` or `--simplify-ratio`. Use
`--no-semantic-mesh` when only geometry is required.

Both launchers support `--dry-run`, `--force`, `--keep-going`, custom roots and
explicit iteration selection.

## Native semantic extension resolution

The renderer first imports an extension installed in the active environment,
then automatically falls back to an in-place build under
`submodules/GaussianWrappingGaga/submodules/`. The training launcher validates
the selected extension before starting geometry training.

If neither form exists, build it explicitly with the same Python used for
training:

```bash
python -m pip install --no-build-isolation \
  submodules/GaussianWrappingGaga/submodules/diff-gaussian-rasterization-semantic

python -m pip install --no-build-isolation \
  submodules/GaussianWrappingGaga/submodules/diff-gaussian-rasterization_ours-semantic
```
