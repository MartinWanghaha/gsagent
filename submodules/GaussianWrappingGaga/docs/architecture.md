# Architecture

## Design boundary

GaussianWrappingGaga owns a complete copy of Gaussian Wrapping. The original
rendering extensions remain installed under their original package names.
Turning semantics off therefore selects the original ABI and execution path;
turning semantics on selects a separate extension. No runtime monkey-patching
or global feature buffer is used.

## Data flow

```text
COLMAP images/cameras ──> Scene ───────────────┐
                                               │
Gaga associated masks ──> ObservationStore     │
                                               v
GaussianModel[N] ──> RaDeGS/Ours projection and tile sorting
  xyz, SH, opacity, scale, rotation             │
  Gaussian Wrapping auxiliary state             ├─> RGB/depth/normal/alpha
  semantic embedding [N,16]                     │
                                               v
                                   native 16D CUDA compositing
                                               │
                                               v
                                      Conv1x1(16,C)
                                               │
                                      CE + sampled 3D loss
```

The semantic field is premultiplied:

`E(p) = sum_i T_i * alpha_i * e_i`.

The CUDA kernel consumes the geometry buffer, binning buffer and image tile
ranges from the same rasterization invocation. Consequently RGB and semantics
share visibility, ellipse footprint, opacity, depth order and termination.

## Gradient policy

The auxiliary semantic backward computes gradients only for the 16D embedding.
Conics and opacities are const inputs to this pass. This prevents a noisy or
temporarily inconsistent mask from deforming an already reconstructed surface
during geometry-first lifting. During joint training the normal renderer
backward still carries all RGB, depth, normal and regularization gradients to
geometry.

## Gaussian lifecycle

`GaussianModel` treats `semantic_features` as a normal per-Gaussian optimizer
group. Every point-count mutation has a corresponding semantic mutation:

- initialization creates a small random 16D vector;
- clone copies the parent vector;
- split repeats the parent vector for all children;
- prune uses the identical keep mask;
- optimizer replacement and concatenation preserve Adam state;
- PLY stores `obj_dc_0..obj_dc_15`;
- capture/restore and the versioned sidecar preserve training state;
- legacy non-semantic PLY/checkpoints remain loadable.

## Renderer matrix

| Mode | Original package | Semantic package | Geometry outputs |
|---|---|---|---|
| RaDeGS | `diff_gaussian_rasterization` | `diff_gaussian_rasterization_gw_semantic` | RGB, expected/median depth and coordinate, alpha, normal |
| Ours | `diff_gaussian_rasterization_ours` | `diff_gaussian_rasterization_gw_ours_semantic` | RGB, median depth, alpha, normal |

Both semantic packages add `[16,H,W]` without removing any renderer output.

## Checkpoint split

The full PLY is the interoperable model artifact. The semantic sidecar contains
the scene-level decoder, an exact copy of the per-Gaussian semantic tensor,
optimizer state, class count, renderer, compositing convention and format
version. Count and dimension mismatches fail loudly.

## Mesh semantics

Geometry and texture are produced by the unchanged Gaussian Wrapping surface
pipeline. The semantic exporter decodes each Gaussian once, builds a cKDTree on
Gaussian centers, and assigns the nearest Gaussian class to each mesh vertex.
It preserves every existing vertex and face field, adds `semantic_id`, writes
deterministic RGB colors, and stores labels and transfer distances as sidecars.
