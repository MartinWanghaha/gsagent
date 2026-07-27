# Gaussian Wrapping Gaga semantic RaDeGS rasterizer

This independent extension preserves the complete Gaussian Wrapping RaDeGS
forward/backward and integration API and adds a native 16-channel semantic
output. It is installed as `diff_gaussian_rasterization_gw_semantic`, so the
original non-semantic extension remains ABI-isolated.

The semantic compositor reuses the projected conics, sorted tile point lists
and per-tile ranges of the RGB/depth/normal pass. Forward returns the
premultiplied-alpha semantic field. Its auxiliary backward intentionally emits
only `grad_semantic_features`; the original renderer backward remains
unchanged for all geometry and appearance outputs.

```python
color, semantic, radii, coord, median_coord, depth, median_depth, alpha, normal = (
    rasterizer(
        means3D=means3D,
        means2D=means2D,
        semantic_features=features,  # [N, 16]
        ...
    )
)
```

---

# Differential Gaussian Rasterization

**NOTE**: this is a modified version to support depth & alpha rendering (both forward and backward) from the [original repository](https://github.com/graphdeco-inria/diff-gaussian-rasterization). 

```python
rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
    means3D=means3D,
    means2D=means2D,
    shs=shs,
    colors_precomp=colors_precomp,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=cov3D_precomp,
)
```


Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>
