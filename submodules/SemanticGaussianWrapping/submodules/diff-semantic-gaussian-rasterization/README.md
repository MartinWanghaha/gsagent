# diff-semantic-gaussian-rasterization

An independently named Gaussian rasterizer for Semantic Gaussian Wrapping. One
front-to-back alpha pass produces RGB, a fixed 16-dimensional semantic
embedding, expected camera-space depth, alpha, camera-space normal, dominant
Gaussian index, and per-Gaussian screen radius.

Calibrated off-center cameras are supported through the `cx` and `cy`
settings. Omitting them preserves the standard centered convention
(`width / 2`, `height / 2`) exactly.

The package exposes the familiar `GaussianRasterizationSettings` and
`GaussianRasterizer` API. `backend="auto"` uses the CUDA extension for CUDA
float32 tensors when installed and otherwise selects the memory-bounded,
fully-differentiable PyTorch reference implementation. The reference backend is
also the executable derivative specification used by gradient parity tests.

CUDA backward is native: a reverse front-to-back recurrence jointly
differentiates RGB, semantic embeddings, expected depth, alpha and normalized
surface normals, then a compact per-Gaussian projection kernel propagates into
3D/screen means, anisotropic scales and quaternions.  Sorting, finite footprint
support, alpha cutoffs, normal-axis selection and dominant IDs remain discrete.
Autograd state is bounded by projected Gaussians plus compact tile overlaps;
there is no pixel-by-Gaussian PyTorch replay graph.

The CUDA package also exposes forward-only renderer-consistent point
integration. It projects and sorts a camera's Gaussians once, then reuses that
state across memory-bounded point chunks:

```python
context = rasterizer.prepare_point_integration(
    means3D, opacities, scales, rotations, query_chunk_size=65_536
)
result = context.query(points3D)
# result.alpha, result.transmittance, result.inside, result.visibility
```

`context.radii` and `context.gaussian_visibility` expose the projected evidence
for view selection. Point integration intentionally has no CPU/reference
execution path: mesh extraction must use the exact CUDA field rather than a
numerically different approximation.

Build the optional extension with:

```bash
python -m pip install --no-build-isolation ./submodules/diff-semantic-gaussian-rasterization
```

When no GPU is visible during compilation, the build uses
`7.5;8.0;8.6+PTX` rather than relying on PyTorch's empty device-derived list.
Set `TORCH_CUDA_ARCH_LIST`, such as `TORCH_CUDA_ARCH_LIST="8.9"`, to target the
deployment GPU explicitly. `SGW_HEADLESS_CUDA_ARCH_LIST` changes only the
headless fallback and has lower precedence than `TORCH_CUDA_ARCH_LIST`.

The implementation is derived conceptually from Graphdeco's differentiable
Gaussian rasterizer and retains its research-only license and attribution.
