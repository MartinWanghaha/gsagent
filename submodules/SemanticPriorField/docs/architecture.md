# Architecture

## Design boundary

SemanticPriorField owns a complete copy of Gaussian Wrapping. The original
rendering extensions remain installed under their original package names.
Turning semantics off therefore selects the original ABI and execution path;
turning semantics on selects a separate extension. No runtime monkey-patching
or global feature buffer is used.

On top of the passive semantic channel, the **Semantic Prior Field** adds
active geometry guidance. It is strictly additive: with `--semantic_prior`
off, training is the base pipeline; with individual `--sp_*` channels off,
each guidance mechanism is removed independently.

## Data flow

```text
COLMAP images/cameras ──> Scene ───────────────┐
                                               │
associated masks ───────> ObservationStore     │
  (confidence/valid/ignore sidecars)           v
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

The semantic field is premultiplied: `E(p) = sum_i T_i * alpha_i * e_i`.
RGB and semantics share visibility, ellipse footprint, opacity, depth order
and termination.

## Semantic Prior Field

```text
embedding [N,16] + head ──(argmax + top1-top2 margin, no_grad)──> labels[N], conf[N]
labels ──(per instance: RANSAC plane → quadric → thin heuristics)──> proxies
                                                                       │
   ┌───────────────────────────────────────────────────────────────────┤
   v                          v                          v
regularizers            density control            mesh extraction
- orientation prior     - boundary splitting       - per-pivot identity
- selective flatten       (semantic-error gidx       through Marching
- SH consistency          scatter, tangent split)    Tetrahedra
- SH outlier decay      - identity pruning         - semantic bad-edge
                          (flat posterior ∩          filtering
                           non-maximal)            - vertex label export
                        - threshold multipliers
```

Cache discipline follows the normal-field/MILo pattern: the field refreshes
every `refresh_interval` iterations and is invalidated whenever the Gaussian
set changes (densify, prune, split). All consumers guard on `field.valid`
and on the Gaussian-count match.

Proxy family selection is data-driven, not taxonomy-driven: thin (needle
anisotropy or filament PCA) → planar (RANSAC inlier ratio) → quadric
(algebraic fit residual) → freeform (no orientation prior). Prior weights
are `confidence x instance_confidence x exp(-fit_residual / sigma)`.

Boundary weighting is mask-only (independent of the embedding): per-view
maps down-weight the depth-normal consistency and multiview NCC/geo losses
within a dilated band around label discontinuities, and are cached per view.

## Gradient policy

The auxiliary semantic backward computes gradients only for the 16D
embedding; conics and opacities are const inputs to that pass. The Semantic
Prior Field influences geometry exclusively through explicit PyTorch losses
and density-control decisions — never through raw semantic-CE gradients on
alpha/conic. Every influence channel is therefore interpretable, gated and
individually ablatable.

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

Semantic-error splitting and identity pruning reuse this lifecycle
(`densify_and_clone_from_mask` / `prune_points`), so the embedding, normal
features, occupancy and mip filter stay consistent through every mutation.

## Renderer matrix

| Mode | Original package | Semantic package | Geometry outputs |
|---|---|---|---|
| RaDeGS | `diff_gaussian_rasterization` | `diff_gaussian_rasterization_gw_semantic` | RGB, expected/median depth and coordinate, alpha, normal |
| Ours | `diff_gaussian_rasterization_ours` | `diff_gaussian_rasterization_gw_ours_semantic` | RGB, median depth, alpha, normal |
| Ours + stats | `diff_gaussian_rasterization_ours` | `diff_gaussian_rasterization_spf` | same as Ours + per-Gaussian stats buffers |

Both semantic packages add `[16,H,W]` without removing any renderer output.

## SPF rasterizer stats channel

`diff_gaussian_rasterization_spf` extends the Ours semantic backward with two
per-Gaussian accumulation buffers, delivered through the raster settings'
`stats_sink` after each backward:

```text
semantic_abs_grad[i]     = sum_p  alpha_i(p) T_i(p) ||dL/dE(p)||_2   (unsigned)
semantic_contribution[i] = sum_p  alpha_i(p) T_i(p)                  (visibility mass)
semantic_grad[i]         = the signed embedding gradient (N, 16)
```

The **conflict score** `(abs - ||signed||) / contribution` is zero when a
Gaussian's semantic supervision agrees across its footprint and large when
different pixels pull it toward different classes — i.e. exactly for
boundary-straddling Gaussians and identity-unstable floaters. Validation
(`tests/exp_spf_stats_validation.py`): forward is bit-identical to the Ours
semantic backend, the contribution identity holds to 1e-5, aligned gradients
produce zero conflict, top-conflict Gaussians localize ~30x closer to the
instance boundary than chance, and the measured overhead is ~0%.

During training (`--sp_stats`, ours rasterizer) the accumulated conflict
replaces the episodic full-camera semantic-error sweep as the boundary
splitting signal: continuous, contribution-weighted, and free. The
accumulator resets on every topology change.

## Checkpoint split

The full PLY is the interoperable model artifact. The semantic sidecar
contains the scene-level decoder, an exact copy of the per-Gaussian semantic
tensor, optimizer state, class count, renderer, compositing convention and
format version. Count and dimension mismatches fail loudly. The prior field
itself is transient state and is never checkpointed — it is recomputed from
the embedding on resume.

## Mesh semantics

The primary path carries identity through extraction itself: per-Gaussian
labels/embeddings are expanded to the Gaussian-major pivot layout, Marching
Tetrahedra interpolates embeddings with the same SDF weights as vertex
positions, categorical labels follow the endpoint nearest the surface, and
faces whose edges confidently bridge two different foreground instances are
filtered (`--filter_semantic_edges`). Vertex labels, confidences and
optional embeddings are exported as `.semantic.npz` plus an
instance-colored `_semantic.ply`.

The legacy cKDTree transfer (`semantic_mesh.py`) remains available for
meshes extracted without semantics.
