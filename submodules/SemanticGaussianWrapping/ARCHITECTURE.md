# Semantic Gaussian Wrapping Architecture

This document is the implementation contract for the project.  The public
layout intentionally follows the original 3D Gaussian Splatting repository.

## Runtime data contract

`scene.Camera` owns aligned RGB and optional semantic observations:

- `original_image`: float tensor `[3,H,W]` in `[0,1]`.
- `gt_mask`: optional float alpha tensor `[1,H,W]` in `[0,1]`.
- `semantic_ids`: long tensor `[H,W]`, `-1` for ignored pixels.
- `semantic_confidence`: float tensor `[H,W]` in `[0,1]`.
- `semantic_boundary`: float tensor `[H,W]` in `[0,1]`.

Every resize or crop must be shared by all observations and the camera
intrinsics. Ground-truth RGB is composed over the current renderer background
at loss/evaluation time, so random-background training remains correct.

## Gaussian state contract

`scene.GaussianModel` delegates all per-Gaussian state mutation to a
`GaussianAttributeRegistry`.  Clone, split, prune, checkpoint, and PLY IO must
operate on every registered attribute atomically.  Core attributes are:

- `xyz [N,3]`, `features_dc [N,1,3]`, `features_rest [N,K,3]`
- `opacity [N,1]`, `scaling [N,3]`, `rotation [N,4]`
- `semantic_embedding [N,D]`, `geometry_logits [N,5]`
- evidence buffers `semantic_confidence [N,1]`,
  `propagated_semantic_confidence [N,1]`, `boundary_score [N,1]`,
  `geometry_error [N,1]`, and `observation_count [N,1]`

The five geometry experts are `planar`, `curved`, `thin`, `freeform`, and
`fuzzy`.  They are soft policies, never hard class-to-shape rules.

## Renderer contract

`gaussian_renderer.render()` returns a dictionary containing:

- `render [3,H,W]`, `semantic [D,H,W]`
- `expected_depth [1,H,W]`, `alpha [1,H,W]`, `normal [3,H,W]`
- `dominant_index [H,W]`, `viewspace_points`, `visibility_filter`, `radii`

The CUDA implementation alpha-composites RGB, semantic features, depth, and
normal in one front-to-back pass and supplies a native backward for all
continuous Gaussian inputs. Projection/binning choices and dominant IDs remain
discrete. A differentiable PyTorch reference backend implements the same
contract for CPU tests and numerical CUDA parity checks.

## Shared neighborhood contract

Evidence projection, manifold regularization, and surface evaluation share one
`GaussianNeighborIndex`; topology changes invalidate it and scheduled refreshes
track moving centers. Center k-nearest-neighbor lookup is used where appropriate.
The surface field instead builds a fixed-memory, multi-scale cKDTree shortlist
and performs exact anisotropic Mahalanobis re-ranking inside it. Center-nearest
Gaussians guarantee valid output while scale buckets retain farther, larger
support. `surface.support_candidate_budget` controls the quality/cost tradeoff;
the quality-first default is `2048`, and the exact backend streams all
Gaussians in bounded blocks as a correctness oracle.

## Semantic surface field contract

`semantic.surface_field.SemanticSurfaceField.query(points)` returns:

- `occupancy [P]`, `sdf [P]`, `normal [P,3]`
- `semantic [P,D]`, `geometry_posterior [P,5]`, `uncertainty [P]`

Training regularizers and asynchronous feedback-mesh sampling call this field.
Its far-field SDF is a log-density residual, not a clipped finite-support proxy;
this gives the bounded training-time extractor a meaningful empty-space signal.
The final offline exporter intentionally does not treat this Gaussian-mixture
residual as a calibrated scene SDF. A single oversized or weakly observed
Gaussian can be useful for rendering while producing a catastrophic global
density isosurface, so final geometry is conditioned on rendered multi-view
observations instead.

`query_point_regions(points, region_ids[P,K])` evaluates K soft regional
occupancy/SDF/normal fields per point. `query_partitioned(global_points,
regional_points, region_ids)` serves global mesh probes and regional Gaussian
probes from one candidate context. Missing regional support is explicitly
invalid and never substitutes the global field. For final mesh ownership,
`query_region_ownership` aggregates sparse memberships in bounded point chunks
instead of retaining a scene-sized `vertices × regions` field.

For one optimizer step, Gaussian center/positive/negative probes and cached
mesh vertices form one partitioned query. Discrete shortlist routing produces
one candidate context; live Gaussian attributes plus geometry-policy outputs
are gathered once. Global reduction runs only for mesh probes, while all top-k
regional reductions for a Gaussian reuse its same spatial candidates.

The global RGB/SSIM objective is the protected PSNR task. Region-balanced RGB
is an auxiliary task rather than part of that protected scalar: the Pareto
gradient guard retains its aligned component and removes only the component
that conflicts with global photometric descent. Small Gaga regions therefore
receive useful appearance gradients without redefining the global image
objective.

## Training lifecycle

1. `bootstrap`: RGB/SSIM and ordinary 3DGS densification.
2. `semantic_lift`: geometry stop-gradient; learn semantic embeddings/decoder.
3. `joint_geometry`: confidence-gated semantic geometry policies and unified
   density control.
4. `surface_refine`: field/mesh consistency with a photometric Pareto guard.

Low-confidence semantic observations must reduce to baseline 3DGS behavior.
Direct camera confidence and propagated confidence remain separate. Only the
direct buffer seeds coherence-gated propagation; spatial support, semantic
cosine, normal agreement, and boundary barriers determine the inferred buffer,
which has its own ceiling and decay. Geometry-expert entropy is reduced only
where the detached geometry target is itself confident, while a target-matched
batch balance term prevents single-expert collapse.
Depth-derived normals are valid only where the center-difference neighborhood
has reliable rendered foreground alpha; silhouette/background depth steps must
not become geometry evidence.

## Topology lifecycle

The unified density controller is the only component allowed to clone, split,
or prune Gaussians. Candidates must pass an absolute photometric/semantic/
boundary/geometric gate; normalized scores only rank eligible candidates.
Global and fractional per-step growth caps bound topology changes. The registry
then updates every attribute and Adam moment together.

After ordinary densification ends, the surface phase owns a separate bounded
observation/topology window. Near the global cap and in zero-net-growth surface
steps, low-utility prune donors finance refinement atomically: a clone costs one
slot and a split costs `children - 1`. Donor churn and net growth have separate
budgets; confident semantic seams and thin structures are protected. Every
decision and completed mutation asserts `N <= max_gaussians`.

Regional growth balancing follows a candidate-first semantic routing contract.
Absolute gates and available capacity are evaluated before semantic decoding;
the density controller then supplies only the compact candidate indices to the
pointwise scene decoder. Candidate embeddings are gathered per chunk, decoded
outside autocast in FP32 over all classes, and reduced immediately to sparse
foreground top-k probabilities. These probabilities are not renormalized:
background and omitted foreground mass remain explicit. Regional budgets use
confidence-weighted probability mass; overlapping membership may fund several
regional quotas, but stable deduplication permits each Gaussian to enter the
atomic topology transaction only once. Transient logit storage falls from
`O(N * C)` to `O(B * C)`, where `B` is
`semantic.region_decode_chunk_size`.
Pixel semantic supervision uses the same exact decoder chunks with activation
recomputation; its objective and gradients equal dense cross-entropy while peak
class-logit memory remains `O(B * C)` instead of `O(H * W * C)`. The per-pixel
density residual is decoded chunkwise without autograd retention.

The absolute photometric gate is defined in the standard Graphdeco normalized
viewport proxy coordinates, not raw pixel-offset coordinates. The renderer
owns the conversion (`x * W/2`, `y * H/2`) so the density controller remains
resolution-independent and the conventional `2e-4` threshold retains its
meaning. Accumulation starts at iteration one, mutation starts strictly after
the warm-up boundary, and large-footprint pruning is an explicitly enabled
late-stage policy rather than a bootstrap default.

Mesh construction samples the shared field on conforming adaptive blocks and
uses a face-neighbor halo to close mixed-resolution lattice faces. Semantic
compatibility and a learned contact graph preserve legitimate touching
instances instead of blindly disconnecting different labels. Simplification
preserves high-confidence semantic seams and visible small components.

Training mesh feedback is derived asynchronously from one immutable,
surface-only Gaussian snapshot after optimizer and topology commit. At most
one refresh is in flight, training continues against the previous complete
cache, and a cleaned finite result first enters an untrusted candidate slot.
Feedback topology is extracted as global geometry without hard semantic IDs;
small connected components are removed unconditionally before quality gating.
The last optimizer step never launches a refresh that cannot contribute a loss.

The mesh-v4 acceptance contract is represented explicitly under
`surface.mesh_feedback_*`. Freshness is bounded by candidate age, intervening
topology events and churn ratio; failed work observes a retry interval, while
accepted feedback is blended over a configured number of iterations. A bounded
probe gate checks aggregate score, SDF p90, normal agreement and semantic
agreement. Local candidate-to-live correspondences additionally require
opacity, semantic confidence and geometry-expert certainty, then use a
`k`-neighbor radius/semantic gate, a robust residual delta and a minimum match
count plus a 50% hard coverage floor. Publication probes a deterministic global
support reservoir; current-camera visibility is reserved for the training
batch. Accepted candidates blend in with smoothstep weights, while an aging
active target fades continuously. Triangle targets own their projector, so a
retired/rejected cache also releases its GPU tensors and cKDTree. These values
belong to the authoritative checkpoint configuration so a resume cannot
silently change mesh acceptance policy.

The bidirectional loss is candidate-first. Gaussian-to-mesh projection uses a
bounded face-centroid broad phase and exact point-to-triangle narrow phase;
mesh-to-field samples come only from matched local faces. SDF and point-to-plane
residuals use detached local thickness/edge spacing and bounded Geman–McClure
penalties. Coverage misses are accumulated sparsely by Gaussian index and enter
the unified density score at topology boundaries, instead of pulling distant
Gaussians toward stale geometry.

## Checkpoint and extraction contract

Native checkpoints contain the resolved configuration, Gaussian and optimizer
state, semantic decoder/evidence, density accumulators, lagged mesh cache,
camera stack, label mapping, and Python/NumPy/PyTorch RNG states. Resume treats
that configuration as authoritative; only total iterations, logging options,
and the execution-memory policy `semantic.region_decode_chunk_size` are mutable.
The training, Gaussian-registry, geometry-evidence, and mesh-feedback schema
versions are exact contracts: missing, legacy, future, or coercible string/float
versions fail before model mutation and require a fresh run.
Dataset identity is checked unless an explicit relocation flag is provided.
Checkpoint publication uses a same-directory temporary archive plus atomic
rename before the optional PLY export; a failed PLY cannot become a false
latest checkpoint. Offline inference memory-maps checkpoints through CPU,
selects only the Gaussian registry and semantic decoder, drops every
optimizer/training cache, and then transfers that compact inference snapshot
to the requested device. Latest resolution prefers native checkpoints and
falls back to PLY snapshots only when no checkpoint exists.

## Offline Region-Conditioned Gaussian Wrapping contract

Final mesh extraction requires a native checkpoint, its semantic decoder, and
calibrated training cameras. It never consumes test views, PLY-only snapshots,
a user-provided field factory, or an alternate density-isosurface path. The
checkpoint owns the Gaussian registry, camera calibration, and `mesh_export`
policy. The strict CLI exposes only the scene-wide Gaussian budget, local chart
budget, deterministic view stride, camera scale, and optional face target.

RC-GW separates geometric evidence from semantic topology:

- `RendererOpacityField` is the single geometric oracle. The CUDA rasterizer
  prepares projection, tile order, conic/precision state, and Gaussian
  visibility once per selected camera. Arbitrary 3D queries reuse that context
  and integrate opacity with the renderer's front-to-back ordering. For a point
  supported by enough cameras, the scalar is
  `occupancy_threshold - min_view(alpha_to_point)`.
- `RegionAtlas` is a topology allocator, not a second field. Exact chunked
  decoder posteriors retain top-k membership and tail mass. Confident groups,
  a residual group, boundary anchors, spatial overlap, and explicit contact
  halos cover the selected Gaussian set. Large groups are partitioned so no
  local Delaunay chart exceeds `max_chart_gaussians`.
- `GaussianPivotSet` places one covariance-aware
  `[-sigma, center, +sigma]` triplet per selected Gaussian. Sigma is normalized
  by robust same-region spacing, retaining thin structures without allowing an
  oversized splat to set chart scale.
- Each chart computes Delaunay tetrahedra locally and applies marching
  tetrahedra to the same global field values. Semantic ownership determines
  which anchors may compete locally; it never changes a sign, deletes a
  crossing only because labels differ, or defines an independent zero set.
- Chart surfaces are represented by canonical global pivot-edge keys. Overlap
  and contact charts therefore weld before geometry is evaluated. Every unique
  crossing edge is refined once with bounded binary search; the controlling
  views recorded at its endpoints are reused, so refinement does not rescan all
  cameras at every step.
- Root attributes interpolate semantic embeddings, confidence, normals, and
  uncertainty. Cleanup is performed per represented region, removing small
  floaters while preserving each region's main supported component before
  optional topology-aware simplification.

The result is one renderer-consistent mesh assembled from bounded semantic
charts, with shared roots at chart overlap and supported contacts. A complete
publication is binary little-endian PLY plus a required schema-2 `.ply.json`
sidecar containing checkpoint identity, resolved RC-GW policy, atlas/topology
counts, and optional mesh metrics. Each file is atomically replaced. There is
no tangent-distance calibration, global-Delaunay-then-semantic-delete stage,
marching cubes, raw-density surface, OBJ output, or legacy fallback.
