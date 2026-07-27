# Experiment log

Decisions in this project are experiment-driven: every optimization channel
must earn its default. This file records the validation and A/B runs behind
the current defaults. Pareto rule: a channel is kept only if PSNR and the
geometry metrics never degrade together.

## E-RAST: SPF rasterizer stats-channel correctness (2026-07-26)

Script: `tests/exp_spf_stats_validation.py` (synthetic two-instance scene,
3000 Gaussians, 160x160).

| Check | Result |
|---|---|
| E1 forward parity vs ours-semantic (color/semantic/depth/alpha) | bit-identical (max diff 0) |
| E2 backward parity (embedding gradients) | 1.9e-6 (atomicAdd ordering only) |
| E3 contribution identity `stat_contribution == grad_e[:, c]` for one-hot dL/dE | 7.6e-6 |
| E4 aligned gradient => zero conflict | 6.7e-6 |
| E5 conflict localization: median abs(x) of top-100 conflict Gaussians vs all visible | **0.015 vs 0.489 (~33x)** |
| E6 fw+bw wall time vs ours backend | -5.5% (zero overhead, within noise) |

Conclusion: the stats channel is mathematically correct, free, and its
conflict score localizes boundary-straddling Gaussians ~33x better than
chance.

## AB-SPLIT: boundary-splitting signal (2026-07-26)

Setup: counter (MipNeRF-360), r8, `--eval`, 6000 iterations, config
`semantic_prior/fast_ab.yaml`, orientation/flatten/SH channels disabled to
isolate splitting; identity pruning at 2500, splits at ~3000/4000/5000.
Single RTX 4070 Ti SUPER, runs sequential on an idle GPU.

| Variant | Test PSNR@6000 | depth-normal loss | L1 | Gaussians | split budget | train time |
|---|---|---|---|---|---|---|
| v0 `--no-sp_split` | 28.026 | 0.00390 | 0.0229 | 589.9k | 0 | 650s |
| v1 split via camera sweep (`--no-sp_stats`) | 28.060 | 0.00359 | 0.0220 | 633.4k | 75.9k | 696s |
| v2 split via conflict stats (`--sp_stats`) | **28.129** | **0.00357** | **0.0199** | 635.3k | 75.8k | **599s** |

Readings:

- Splitting as a channel is Pareto-positive: both v1 and v2 beat v0 on PSNR
  and on the depth-normal consistency proxy simultaneously.
- v1 and v2 spent an almost identical split budget (~76k new Gaussians), so
  v2's +0.069 dB over v1 is pure **selection quality**: the accumulated
  conflict score picks better split candidates than the episodic
  camera-sweep attribution (argmax-contributor, single snapshot).
- v2 also removes the sweep overhead entirely (~15 s per split event at r8;
  the full-schedule default has 5 events at r2, where sweeps are ~4x more
  expensive).
- All three v2 split events fired with `source: stats` (logged in
  `events.jsonl`).

Decision: **keep** `--sp_stats` as default; the sweep path remains only as
the automatic fallback (radegs rasterizer, or stats not yet accumulated).

## AB-MESH: mesh extraction comparison (2026-07-27)

Same checkpoints as AB-SPLIT (6000 iterations, r8 — proxy scale; full-scale
numbers require r2/30k runs). Extraction: sdf_mode ours, 8 binary steps,
valid mask, large-edge filtering, postprocess (largest connected component).

| Mesh | Vertices raw | Vertices post | Faces raw | Faces post |
|---|---|---|---|---|
| v0 no-split | 524,837 | 398,300 | 890,102 | 760,782 |
| v2 conflict-split | 568,596 | 410,517 | 958,786 | 783,910 |
| v2 + semantic edge filter (conf 0.5) | 568,596 | **191,739** | — | — |
| v2 + semantic edge filter + length gate (0.5x) | 568,596 | **207,425** | — | — |

Readings:

- v2's post mesh keeps ~3% more surface than v0 (more detail retained after
  floater removal), consistent with the depth-normal advantage. Definitive
  mesh quality (Chamfer/F1) requires GT scenes (DTU/TnT) — open item.
- **Semantic edge filtering is harmful in this configuration** and is now
  default-off. Root cause: with 877 fine-grained instances, label-crossing
  MT edges occur along every legitimate instance adjacency, not only on
  bridges; dropping those faces fragments the largest connected component,
  which the postprocess then decimates (410k -> 192k vertices). A length
  gate (drop only long crossing edges) recovers only marginally (207k).
  The identity export itself (per-vertex labels npz, colored mesh, MT
  interpolation) works end-to-end (878 classes, 1.27M labelled pivots) and
  stays on.
- Rescue directions if bridging artifacts matter in a target scene:
  per-object extraction mode (`--per_object` design), filtering
  tetrahedra (not faces) whose four pivots are confidently
  multi-instance, or skipping the largest-component postprocess when the
  filter is active.

Decision: **keep** `--use_semantics` (identity on the mesh); **drop**
`--filter_semantic_edges` from defaults (available as an explicit opt-in
for scenes with few, well-separated instances).

## Open items (not yet validated)

- `--balance_semantic` (Kendall uncertainty weighting): implemented,
  default off, no A/B yet.
- Boundary weighting / identity pruning / budget multipliers were active in
  every AB-SPLIT variant equally; their isolated contributions still need
  their own ablation round at full scale.
- Full-scale (r2, 30k) confirmation of the AB-SPLIT ranking.
