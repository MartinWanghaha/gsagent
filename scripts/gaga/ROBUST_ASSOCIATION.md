# Robust Gaga mask association

`associate_gaga_masks_mipnerf360.py` provides two backends:

- `robust` (default): global, confidence-aware multi-view association.
- `legacy`: Gaga's original greedy Gaussian bank.

## Recommended command

Write a side-by-side result first:

```bash
python scripts/gaga/associate_gaga_masks_mipnerf360.py \
  --scene counter \
  --seg-method entityseg \
  --point-cloud outputs/gaussian_wrapping_mipnerf360/counter \
  --association-algorithm robust \
  --output-name entityseg_mask_robust \
  --gpu 0 \
  --visualize
```

After comparison, replace the standard Gaga input while retaining an automatic
backup of the old directory:

```bash
python scripts/gaga/associate_gaga_masks_mipnerf360.py \
  --scene counter \
  --seg-method entityseg \
  --point-cloud outputs/gaussian_wrapping_mipnerf360/counter \
  --association-algorithm robust \
  --force \
  --gpu 0
```

## Algorithm

The robust backend treats each raw mask as a noisy observation.

1. It separates a reliable mask core from an uncertain boundary.
2. It projects scene-aligned Gaussian centers and retains depth-weighted,
   patch-local front evidence.
3. It creates a sparse mask graph over geometrically neighboring cameras.
4. Edge scores combine weighted Gaussian Jaccard, bidirectional coverage,
   color appearance, 3D proximity, and observation quality.
5. Hungarian matching supplies high-precision pairwise edges.
6. Constrained global clustering prevents accidental same-view identity
   collapse. A conservative second pass can merge adjacent, appearance-consistent
   fragments.
7. Tentative tracks are resolved by area, observation quality, Gaussian support,
   and propagation from confirmed neighboring views. View count alone never
   decides whether a track is retained.
8. Confirmed tracks vote for a probabilistic global Gaussian label field.
9. Every raw mask keeps one track by default. A real split is allowed only when
   multiple strong 3D labels agree with RGB SLIC superpixels; alternative
   components must satisfy seed purity, connectivity, and minimum-area checks.
   There is no nearest-neighbor pixel propagation.
10. Export is rescanned, unused classes are removed, and IDs are compacted to
    `1..num_mask`. Empty classes, per-frame ignore fraction, region purity, and
    label jump rate are mandatory QA gates; failed output is not published.

This first implementation uses a memory-bounded center/depth visibility backend.
Its module boundaries allow a top-K alpha-contribution CUDA backend to replace
visibility without changing graph, clustering, refinement, or output formats.

## Output

```text
<scene>/<output-name>/
├── <image>.png                    # uint16 global IDs
├── info.json                      # Gaga-compatible metadata
├── confidence/<image>.png         # uint8 confidence
├── valid/<image>.png              # supervision-valid pixels
├── visualization/<image>.png      # optional
└── association/
    ├── manifest.json
    ├── observations.jsonl
    ├── tracks.json
    ├── graph_edges.npz
    ├── diagnostics.json
    └── qa.json
```

Labels use:

- `0`: background
- `1..num_mask`: global instances
- `65535`: unresolved/ignore

GaussianWrappingGaga consumes the confidence and valid sidecars. The bundled
Gaga lift path recognizes the ignore label.

## Important controls

- `--match-threshold`: minimum robust graph match score (default `0.28`).
- `--match-margin`: reject ambiguous pairwise matches.
- `--min-track-views`: views required for a confirmed instance.
- `--tentative-min-area-fraction`, `--tentative-min-quality`, and
  `--tentative-min-gaussians`: local evidence needed to promote a tentative track.
- `--tentative-propagation-threshold` and
  `--tentative-min-neighbor-views`: neighbor evidence needed to propagate it.
- `--core-radius`: reliable-core erosion radius.
- `--boundary-weight`: supervision weight at uncertain raw boundaries.
- `--split-fraction`: minimum 3D consensus mass for splitting one raw mask.
- `--split-min-seed-points`, `--split-seed-purity`, and
  `--split-min-area-pixels`: spatial split gates.
- `--superpixel-size` and `--superpixel-compactness`: RGB SLIC controls.
- `--gaussian-label-margin`: reject ambiguous Gaussian identities.
- `--qa-max-ignore-fraction`, `--qa-min-region-purity`, and
  `--qa-max-label-jump-rate`: mandatory export gates.

Refinement and export are streamed one full-resolution view at a time, so RAM
does not grow with the number of cameras. The exporter builds a staging
directory and publishes it atomically. With `--force`, an existing output is
renamed to a timestamped backup before the new result is installed.
