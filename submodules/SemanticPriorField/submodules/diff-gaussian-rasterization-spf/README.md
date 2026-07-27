# Gaussian Wrapping Gaga semantic median-depth rasterizer

This independent extension preserves the complete Gaussian Wrapping Ours
forward/backward, depth sampling and integration API and adds a native
16-channel semantic output. It is installed as
`diff_gaussian_rasterization_spf`, leaving the original package
and ABI untouched.

The auxiliary CUDA compositor shares the renderer's projected conics, sorted
point list and tile ranges, and returns premultiplied-alpha semantic features.
Its backward is embedding-only by design.

```python
color, semantic, radii, median_depth, alpha, normal = rasterizer(
    means3D=means3D,
    means2D=means2D,
    semantic_features=features,  # [N, 16]
    ...
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
