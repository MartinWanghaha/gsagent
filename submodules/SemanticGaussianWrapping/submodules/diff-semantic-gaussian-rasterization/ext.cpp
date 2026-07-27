/*
 * Copyright (C) 2023, Inria, GRAPHDECO research group.
 * Copyright (C) 2026, Semantic Gaussian Wrapping contributors.
 *
 * This interface follows the Gaussian Splatting extension layout.  The
 * semantic multi-attribute implementation is an independent derivative and is
 * distributed under the terms in LICENSE.md.
 */

#include <torch/extension.h>

#include "rasterize.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "rasterize_forward",
      &semantic_gaussian_rasterize_forward,
      "Front-to-back semantic Gaussian rasterization (CUDA)");
  module.def(
      "rasterize_backward",
      &semantic_gaussian_rasterize_backward,
      "Analytic semantic Gaussian rasterization backward pass (CUDA)");
  module.def(
      "prepare_point_integration",
      &semantic_gaussian_prepare_point_integration,
      "Prepare renderer-consistent Gaussian point integration state (CUDA)");
  module.def(
      "integrate_points_forward",
      &semantic_gaussian_integrate_points_forward,
      "Integrate Gaussian opacity up to arbitrary 3D query points (CUDA)");
}
