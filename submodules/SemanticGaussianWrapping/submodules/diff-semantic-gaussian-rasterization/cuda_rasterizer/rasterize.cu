/*
 * Copyright (C) 2023, Inria, GRAPHDECO research group.
 * Copyright (C) 2026, Semantic Gaussian Wrapping contributors.
 *
 * Inspired by the EWA projection and front-to-back compositing formulation in
 * diff-gaussian-rasterization.  This is an independent, deliberately compact
 * implementation for the RGB + semantic + depth + alpha + normal contract.
 * See LICENSE.md for the research-only license inherited from Graphdeco.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cmath>
#include <limits>
#include <vector>

#include "../rasterize.h"

namespace {

constexpr int kSemanticDim = 16;
constexpr int kTileWidth = 16;
constexpr int kTileHeight = 16;
constexpr float kAlphaCap = 0.99f;
constexpr float kAlphaThreshold = 1.0f / 255.0f;
// Match the reference Graphdeco rasterizer. Sparse COLMAP clouds commonly
// contain points arbitrarily close to a camera plane; projecting them creates
// billion-pixel footprints and overflows the analytic projection derivative.
constexpr float kNearPlane = 0.2f;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);        \
  CHECK_CONTIGUOUS(x);  \
  CHECK_FLOAT(x)

struct Projection {
  float mean_x;
  float mean_y;
  float depth;
  float conic_a;
  float conic_b;
  float conic_c;
  float normal_x;
  float normal_y;
  float normal_z;
  float radius;
  bool visible;
};

// A single-tangent dual number is used only by the per-Gaussian projection
// backward kernel.  Replaying the compact projection twelve times (one input
// tangent at a time) is intentionally compute-heavy but register- and
// memory-bounded.  It also keeps the CUDA derivative tied directly to the
// forward equations without constructing an H x W PyTorch autograd graph.
struct Dual {
  float value;
  float tangent;

  __device__ Dual() : value(0.0f), tangent(0.0f) {}
  __device__ Dual(float value_, float tangent_ = 0.0f) : value(value_), tangent(tangent_) {}
};

__device__ inline Dual operator+(const Dual& lhs, const Dual& rhs) {
  return Dual(lhs.value + rhs.value, lhs.tangent + rhs.tangent);
}

__device__ inline Dual operator-(const Dual& lhs, const Dual& rhs) {
  return Dual(lhs.value - rhs.value, lhs.tangent - rhs.tangent);
}

__device__ inline Dual operator-(const Dual& value) {
  return Dual(-value.value, -value.tangent);
}

__device__ inline Dual operator*(const Dual& lhs, const Dual& rhs) {
  return Dual(
      lhs.value * rhs.value,
      lhs.tangent * rhs.value + lhs.value * rhs.tangent);
}

__device__ inline Dual operator/(const Dual& lhs, const Dual& rhs) {
  const float inverse = 1.0f / rhs.value;
  return Dual(
      lhs.value * inverse,
      (lhs.tangent - lhs.value * inverse * rhs.tangent) * inverse);
}

__device__ inline Dual dual_floor_max(const Dual& value, float floor) {
  return value.value > floor ? value : Dual(floor, 0.0f);
}

__device__ inline Dual dual_clamp(const Dual& value, float minimum, float maximum) {
  if (value.value < minimum) return Dual(minimum, 0.0f);
  if (value.value > maximum) return Dual(maximum, 0.0f);
  return value;
}

__device__ inline Dual dual_rsqrt(const Dual& value, float floor) {
  const Dual clamped = dual_floor_max(value, floor);
  const float inverse_sqrt = rsqrtf(clamped.value);
  return Dual(
      inverse_sqrt,
      -0.5f * clamped.tangent * inverse_sqrt * inverse_sqrt * inverse_sqrt);
}

struct ProjectionDual {
  Dual mean_x;
  Dual mean_y;
  Dual depth;
  Dual conic_a;
  Dual conic_b;
  Dual conic_c;
  Dual normal_x;
  Dual normal_y;
  Dual normal_z;
};

__device__ inline void quaternion_matrix_dual(const Dual quaternion[4], Dual matrix[9]) {
  Dual norm_squared;
#pragma unroll
  for (int component = 0; component < 4; ++component) {
    norm_squared = norm_squared + quaternion[component] * quaternion[component];
  }
  const Dual inverse_norm = dual_rsqrt(norm_squared, 1e-24f);
  const Dual w = quaternion[0] * inverse_norm;
  const Dual x = quaternion[1] * inverse_norm;
  const Dual y = quaternion[2] * inverse_norm;
  const Dual z = quaternion[3] * inverse_norm;
  matrix[0] = Dual(1.0f) - Dual(2.0f) * (y * y + z * z);
  matrix[1] = Dual(2.0f) * (x * y - w * z);
  matrix[2] = Dual(2.0f) * (x * z + w * y);
  matrix[3] = Dual(2.0f) * (x * y + w * z);
  matrix[4] = Dual(1.0f) - Dual(2.0f) * (x * x + z * z);
  matrix[5] = Dual(2.0f) * (y * z - w * x);
  matrix[6] = Dual(2.0f) * (x * z - w * y);
  matrix[7] = Dual(2.0f) * (y * z + w * x);
  matrix[8] = Dual(1.0f) - Dual(2.0f) * (x * x + y * y);
}

// seed layout: xyz (0..2), screen offset xy (3..4), scale (5..7),
// quaternion wxyz (8..11).
__device__ inline ProjectionDual project_one_dual(
    int gaussian,
    int seed,
    const float* means3d,
    const float* means2d,
    const float* scales,
    const float* rotations,
    const float* view,
    int height,
    int width,
    float tanfovx,
    float tanfovy,
    float cx,
    float cy,
    float scale_modifier,
    float antialias_sigma) {
  Dual world[3];
  Dual screen_offset[2];
  Dual scale[3];
  Dual quaternion[4];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    world[axis] = Dual(means3d[3 * gaussian + axis], seed == axis ? 1.0f : 0.0f);
    scale[axis] = Dual(scales[3 * gaussian + axis], seed == 5 + axis ? 1.0f : 0.0f);
  }
#pragma unroll
  for (int axis = 0; axis < 2; ++axis) {
    screen_offset[axis] = Dual(means2d[3 * gaussian + axis], seed == 3 + axis ? 1.0f : 0.0f);
  }
#pragma unroll
  for (int component = 0; component < 4; ++component) {
    quaternion[component] = Dual(
        rotations[4 * gaussian + component],
        seed == 8 + component ? 1.0f : 0.0f);
  }

  Dual camera[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    camera[axis] =
        world[0] * Dual(view[axis]) +
        world[1] * Dual(view[4 + axis]) +
        world[2] * Dual(view[8 + axis]) +
        Dual(view[12 + axis]);
  }
  const float focal_x = 0.5f * static_cast<float>(width) / tanfovx;
  const float focal_y = 0.5f * static_cast<float>(height) / tanfovy;
  ProjectionDual projected;
  projected.depth = camera[2];
  projected.mean_x =
      Dual(focal_x) * camera[0] / camera[2] + Dual(cx) + screen_offset[0];
  projected.mean_y =
      Dual(focal_y) * camera[1] / camera[2] + Dual(cy) + screen_offset[1];

  Dual rotation[9];
  quaternion_matrix_dual(quaternion, rotation);
  Dual scale_squared[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    const Dual modified = scale[axis] * Dual(scale_modifier);
    scale_squared[axis] = modified * modified;
  }
  Dual covariance_world[9];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      Dual value;
#pragma unroll
      for (int axis = 0; axis < 3; ++axis) {
        value = value + rotation[3 * row + axis] * scale_squared[axis] * rotation[3 * column + axis];
      }
      covariance_world[3 * row + column] = value;
    }
  }
  Dual covariance_camera[9];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      Dual value;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          value = value + Dual(view[4 * i + row]) * covariance_world[3 * i + j] * Dual(view[4 * j + column]);
        }
      }
      covariance_camera[3 * row + column] = value;
    }
  }
  const Dual inverse_z = Dual(1.0f) / camera[2];
  // Match Graphdeco's EWA projection domain.  The projected mean remains the
  // true pinhole projection above, while the covariance Jacobian is evaluated
  // on a slightly enlarged, clamped frustum.  Without this clamp an
  // off-frustum Gaussian can acquire a million-pixel footprint even though it
  // contributes only near an image edge.
  const Dual covariance_x = dual_clamp(
      camera[0] * inverse_z,
      -1.3f * tanfovx,
      1.3f * tanfovx) * camera[2];
  const Dual covariance_y = dual_clamp(
      camera[1] * inverse_z,
      -1.3f * tanfovy,
      1.3f * tanfovy) * camera[2];
  const Dual jacobian[6] = {
      Dual(focal_x) * inverse_z,
      Dual(0.0f),
      -Dual(focal_x) * covariance_x * inverse_z * inverse_z,
      Dual(0.0f),
      Dual(focal_y) * inverse_z,
      -Dual(focal_y) * covariance_y * inverse_z * inverse_z};
  Dual covariance_2d[4];
#pragma unroll
  for (int row = 0; row < 2; ++row) {
#pragma unroll
    for (int column = 0; column < 2; ++column) {
      Dual value;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          value = value + jacobian[3 * row + i] * covariance_camera[3 * i + j] * jacobian[3 * column + j];
        }
      }
      covariance_2d[2 * row + column] = value;
    }
  }
  const float variance_floor = antialias_sigma * antialias_sigma;
  covariance_2d[0] = covariance_2d[0] + Dual(variance_floor);
  covariance_2d[3] = covariance_2d[3] + Dual(variance_floor);
  const Dual determinant = dual_floor_max(
      covariance_2d[0] * covariance_2d[3] - covariance_2d[1] * covariance_2d[1],
      1e-12f);
  projected.conic_a = covariance_2d[3] / determinant;
  projected.conic_b = -covariance_2d[1] / determinant;
  projected.conic_c = covariance_2d[0] / determinant;

  // Match the detached argmin and sign choice in the reference projection.
  int normal_axis = 0;
  if (scale_squared[1].value < scale_squared[normal_axis].value) normal_axis = 1;
  if (scale_squared[2].value < scale_squared[normal_axis].value) normal_axis = 2;
  const Dual normal_world[3] = {
      rotation[normal_axis],
      rotation[3 + normal_axis],
      rotation[6 + normal_axis]};
  Dual normal_camera[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    normal_camera[axis] =
        normal_world[0] * Dual(view[axis]) +
        normal_world[1] * Dual(view[4 + axis]) +
        normal_world[2] * Dual(view[8 + axis]);
  }
  Dual normal_norm_squared;
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    normal_norm_squared = normal_norm_squared + normal_camera[axis] * normal_camera[axis];
  }
  const Dual inverse_normal = dual_rsqrt(normal_norm_squared, 1e-24f);
  const float orientation = normal_camera[2].value > 0.0f ? -1.0f : 1.0f;
  projected.normal_x = normal_camera[0] * inverse_normal * Dual(orientation);
  projected.normal_y = normal_camera[1] * inverse_normal * Dual(orientation);
  projected.normal_z = normal_camera[2] * inverse_normal * Dual(orientation);
  return projected;
}

__device__ inline void quaternion_matrix(const float* quaternion, float matrix[9]) {
  float w = quaternion[0];
  float x = quaternion[1];
  float y = quaternion[2];
  float z = quaternion[3];
  const float inverse_norm = rsqrtf(fmaxf(w * w + x * x + y * y + z * z, 1e-24f));
  w *= inverse_norm;
  x *= inverse_norm;
  y *= inverse_norm;
  z *= inverse_norm;
  matrix[0] = 1.0f - 2.0f * (y * y + z * z);
  matrix[1] = 2.0f * (x * y - w * z);
  matrix[2] = 2.0f * (x * z + w * y);
  matrix[3] = 2.0f * (x * y + w * z);
  matrix[4] = 1.0f - 2.0f * (x * x + z * z);
  matrix[5] = 2.0f * (y * z - w * x);
  matrix[6] = 2.0f * (x * z - w * y);
  matrix[7] = 2.0f * (y * z + w * x);
  matrix[8] = 1.0f - 2.0f * (x * x + y * y);
}

__device__ inline Projection project_one(
    int gaussian,
    const float* means3d,
    const float* means2d,
    const float* scales,
    const float* rotations,
    const float* view,
    int height,
    int width,
    float tanfovx,
    float tanfovy,
    float cx,
    float cy,
    float scale_modifier,
    float antialias_sigma) {
  Projection projected{};
  const float world_x = means3d[3 * gaussian + 0];
  const float world_y = means3d[3 * gaussian + 1];
  const float world_z = means3d[3 * gaussian + 2];
  const float camera_x = world_x * view[0] + world_y * view[4] + world_z * view[8] + view[12];
  const float camera_y = world_x * view[1] + world_y * view[5] + world_z * view[9] + view[13];
  const float camera_z = world_x * view[2] + world_y * view[6] + world_z * view[10] + view[14];
  projected.depth = camera_z;
  if (camera_z <= kNearPlane) {
    projected.visible = false;
    return projected;
  }

  const float focal_x = 0.5f * static_cast<float>(width) / tanfovx;
  const float focal_y = 0.5f * static_cast<float>(height) / tanfovy;
  const float offset_x = means2d == nullptr ? 0.0f : means2d[3 * gaussian + 0];
  const float offset_y = means2d == nullptr ? 0.0f : means2d[3 * gaussian + 1];
  projected.mean_x = focal_x * camera_x / camera_z + cx + offset_x;
  projected.mean_y = focal_y * camera_y / camera_z + cy + offset_y;

  float rotation[9];
  quaternion_matrix(rotations + 4 * gaussian, rotation);
  float scale_squared[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    const float value = scales[3 * gaussian + axis] * scale_modifier;
    scale_squared[axis] = value * value;
  }

  // Sigma_world = R diag(s^2) R^T.
  float covariance_world[9];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      float value = 0.0f;
#pragma unroll
      for (int axis = 0; axis < 3; ++axis) {
        value += rotation[3 * row + axis] * scale_squared[axis] * rotation[3 * column + axis];
      }
      covariance_world[3 * row + column] = value;
    }
  }

  // For p_cam = p_world @ B, Sigma_cam = B^T Sigma_world B.
  float covariance_camera[9];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      float value = 0.0f;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          value += view[4 * i + row] * covariance_world[3 * i + j] * view[4 * j + column];
        }
      }
      covariance_camera[3 * row + column] = value;
    }
  }

  const float inverse_z = 1.0f / camera_z;
  const float covariance_x =
      fminf(1.3f * tanfovx, fmaxf(-1.3f * tanfovx, camera_x * inverse_z)) * camera_z;
  const float covariance_y =
      fminf(1.3f * tanfovy, fmaxf(-1.3f * tanfovy, camera_y * inverse_z)) * camera_z;
  const float jacobian[6] = {
      focal_x * inverse_z,
      0.0f,
      -focal_x * covariance_x * inverse_z * inverse_z,
      0.0f,
      focal_y * inverse_z,
      -focal_y * covariance_y * inverse_z * inverse_z};
  float covariance_2d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int row = 0; row < 2; ++row) {
#pragma unroll
    for (int column = 0; column < 2; ++column) {
      float value = 0.0f;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          value += jacobian[3 * row + i] * covariance_camera[3 * i + j] * jacobian[3 * column + j];
        }
      }
      covariance_2d[2 * row + column] = value;
    }
  }
  covariance_2d[0] += antialias_sigma * antialias_sigma;
  covariance_2d[3] += antialias_sigma * antialias_sigma;
  const float determinant = fmaxf(covariance_2d[0] * covariance_2d[3] - covariance_2d[1] * covariance_2d[1], 1e-12f);
  projected.conic_a = covariance_2d[3] / determinant;
  projected.conic_b = -covariance_2d[1] / determinant;
  projected.conic_c = covariance_2d[0] / determinant;
  const float half_trace = 0.5f * (covariance_2d[0] + covariance_2d[3]);
  const float max_eigenvalue = fmaxf(half_trace + sqrtf(fmaxf(half_trace * half_trace - determinant, 0.0f)), 0.0f);
  projected.radius = ceilf(3.0f * sqrtf(max_eigenvalue));
  projected.visible =
      isfinite(projected.radius) &&
      projected.mean_x + projected.radius >= 0.0f &&
      projected.mean_x - projected.radius < static_cast<float>(width) &&
      projected.mean_y + projected.radius >= 0.0f &&
      projected.mean_y - projected.radius < static_cast<float>(height);

  int normal_axis = 0;
  if (scale_squared[1] < scale_squared[normal_axis]) normal_axis = 1;
  if (scale_squared[2] < scale_squared[normal_axis]) normal_axis = 2;
  const float normal_world_x = rotation[normal_axis];
  const float normal_world_y = rotation[3 + normal_axis];
  const float normal_world_z = rotation[6 + normal_axis];
  float normal_x = normal_world_x * view[0] + normal_world_y * view[4] + normal_world_z * view[8];
  float normal_y = normal_world_x * view[1] + normal_world_y * view[5] + normal_world_z * view[9];
  float normal_z = normal_world_x * view[2] + normal_world_y * view[6] + normal_world_z * view[10];
  const float inverse_normal_length = rsqrtf(fmaxf(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z, 1e-24f));
  normal_x *= inverse_normal_length;
  normal_y *= inverse_normal_length;
  normal_z *= inverse_normal_length;
  if (normal_z > 0.0f) {
    normal_x = -normal_x;
    normal_y = -normal_y;
    normal_z = -normal_z;
  }
  projected.normal_x = normal_x;
  projected.normal_y = normal_y;
  projected.normal_z = normal_z;
  return projected;
}

// Store the inverse camera-space covariance and its product with the Gaussian
// center.  A point query with unit camera ray r can then recover the exact
// one-dimensional Gaussian along that ray:
//
//   inverse_variance = r^T precision r
//   peak_distance    = r^T precision mean / inverse_variance
//
// This is the renderer-consistent longitudinal term used by Gaussian opacity
// fields, without the first-order screen-space ray-plane approximation.
__device__ inline void camera_precision(
    int gaussian,
    const float* means3d,
    const float* scales,
    const float* rotations,
    const float* view,
    float scale_modifier,
    float* compact_precision,
    float* precision_mean) {
  float rotation[9];
  quaternion_matrix(rotations + 4 * gaussian, rotation);

  float inverse_scale_squared[3];
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    const float scale = fmaxf(fabsf(scales[3 * gaussian + axis] * scale_modifier), 1e-8f);
    inverse_scale_squared[axis] = 1.0f / (scale * scale);
  }

  float precision_world[9];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      float value = 0.0f;
#pragma unroll
      for (int axis = 0; axis < 3; ++axis) {
        value +=
            rotation[3 * row + axis] *
            inverse_scale_squared[axis] *
            rotation[3 * column + axis];
      }
      precision_world[3 * row + column] = value;
    }
  }

  float precision_camera[9];
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      float value = 0.0f;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          value +=
              view[4 * i + row] *
              precision_world[3 * i + j] *
              view[4 * j + column];
        }
      }
      precision_camera[3 * row + column] = value;
    }
  }

  compact_precision[6 * gaussian + 0] = precision_camera[0];
  compact_precision[6 * gaussian + 1] = precision_camera[1];
  compact_precision[6 * gaussian + 2] = precision_camera[2];
  compact_precision[6 * gaussian + 3] = precision_camera[4];
  compact_precision[6 * gaussian + 4] = precision_camera[5];
  compact_precision[6 * gaussian + 5] = precision_camera[8];

  const float world_x = means3d[3 * gaussian + 0];
  const float world_y = means3d[3 * gaussian + 1];
  const float world_z = means3d[3 * gaussian + 2];
  const float camera_mean[3] = {
      world_x * view[0] + world_y * view[4] + world_z * view[8] + view[12],
      world_x * view[1] + world_y * view[5] + world_z * view[9] + view[13],
      world_x * view[2] + world_y * view[6] + world_z * view[10] + view[14]};
#pragma unroll
  for (int row = 0; row < 3; ++row) {
    precision_mean[3 * gaussian + row] =
        precision_camera[3 * row + 0] * camera_mean[0] +
        precision_camera[3 * row + 1] * camera_mean[1] +
        precision_camera[3 * row + 2] * camera_mean[2];
  }
}

__device__ inline void tile_bounds(
    float mean_x,
    float mean_y,
    float radius,
    int tiles_x,
    int tiles_y,
    int& minimum_x,
    int& minimum_y,
    int& maximum_x,
    int& maximum_y) {
  minimum_x = max(0, min(tiles_x, static_cast<int>(floorf((mean_x - radius) / kTileWidth))));
  minimum_y = max(0, min(tiles_y, static_cast<int>(floorf((mean_y - radius) / kTileHeight))));
  maximum_x = max(0, min(tiles_x, static_cast<int>(floorf((mean_x + radius) / kTileWidth)) + 1));
  maximum_y = max(0, min(tiles_y, static_cast<int>(floorf((mean_y + radius) / kTileHeight)) + 1));
}

__global__ void project_and_count_kernel(
    int count,
    const float* means3d,
    const float* means2d,
    const float* scales,
    const float* rotations,
    const float* viewmatrix,
    int height,
    int width,
    float tanfovx,
    float tanfovy,
    float cx,
    float cy,
    float scale_modifier,
    float antialias_sigma,
    float* projected_means,
    float* conics,
    float* depths,
    float* normals,
    float* radii,
    int64_t* tile_counts) {
  const int gaussian = blockIdx.x * blockDim.x + threadIdx.x;
  if (gaussian >= count) return;
  const Projection projected = project_one(
      gaussian,
      means3d,
      means2d,
      scales,
      rotations,
      viewmatrix,
      height,
      width,
      tanfovx,
      tanfovy,
      cx,
      cy,
      scale_modifier,
      antialias_sigma);
  projected_means[2 * gaussian] = projected.mean_x;
  projected_means[2 * gaussian + 1] = projected.mean_y;
  conics[3 * gaussian] = projected.conic_a;
  conics[3 * gaussian + 1] = projected.conic_b;
  conics[3 * gaussian + 2] = projected.conic_c;
  depths[gaussian] = projected.depth;
  normals[3 * gaussian] = projected.normal_x;
  normals[3 * gaussian + 1] = projected.normal_y;
  normals[3 * gaussian + 2] = projected.normal_z;
  radii[gaussian] = projected.visible ? projected.radius : 0.0f;
  if (!projected.visible) {
    tile_counts[gaussian] = 0;
    return;
  }
  const int tiles_x = (width + kTileWidth - 1) / kTileWidth;
  const int tiles_y = (height + kTileHeight - 1) / kTileHeight;
  int minimum_x, minimum_y, maximum_x, maximum_y;
  tile_bounds(
      projected.mean_x,
      projected.mean_y,
      projected.radius,
      tiles_x,
      tiles_y,
      minimum_x,
      minimum_y,
      maximum_x,
      maximum_y);
  tile_counts[gaussian] = static_cast<int64_t>(maximum_x - minimum_x) * (maximum_y - minimum_y);
}

__global__ void project_integration_gaussians_kernel(
    int count,
    const float* means3d,
    const float* scales,
    const float* rotations,
    const float* viewmatrix,
    int height,
    int width,
    float tanfovx,
    float tanfovy,
    float cx,
    float cy,
    float scale_modifier,
    float antialias_sigma,
    float* projected_means,
    float* conics,
    float* depths,
    float* precisions,
    float* precision_means,
    float* radii,
    int64_t* tile_counts) {
  const int gaussian = blockIdx.x * blockDim.x + threadIdx.x;
  if (gaussian >= count) return;

  const Projection projected = project_one(
      gaussian,
      means3d,
      nullptr,
      scales,
      rotations,
      viewmatrix,
      height,
      width,
      tanfovx,
      tanfovy,
      cx,
      cy,
      scale_modifier,
      antialias_sigma);
  projected_means[2 * gaussian] = projected.mean_x;
  projected_means[2 * gaussian + 1] = projected.mean_y;
  conics[3 * gaussian] = projected.conic_a;
  conics[3 * gaussian + 1] = projected.conic_b;
  conics[3 * gaussian + 2] = projected.conic_c;
  depths[gaussian] = projected.depth;
  radii[gaussian] = projected.visible ? projected.radius : 0.0f;

  camera_precision(
      gaussian,
      means3d,
      scales,
      rotations,
      viewmatrix,
      scale_modifier,
      precisions,
      precision_means);

  if (!projected.visible) {
    tile_counts[gaussian] = 0;
    return;
  }
  const int tiles_x = (width + kTileWidth - 1) / kTileWidth;
  const int tiles_y = (height + kTileHeight - 1) / kTileHeight;
  int minimum_x, minimum_y, maximum_x, maximum_y;
  tile_bounds(
      projected.mean_x,
      projected.mean_y,
      projected.radius,
      tiles_x,
      tiles_y,
      minimum_x,
      minimum_y,
      maximum_x,
      maximum_y);
  tile_counts[gaussian] =
      static_cast<int64_t>(maximum_x - minimum_x) * (maximum_y - minimum_y);
}

__global__ void duplicate_with_keys_kernel(
    int count,
    const float* projected_means,
    const float* depths,
    const float* radii,
    const int64_t* offsets,
    int tiles_x,
    int tiles_y,
    int64_t* keys,
    int32_t* gaussian_ids) {
  const int gaussian = blockIdx.x * blockDim.x + threadIdx.x;
  if (gaussian >= count || radii[gaussian] <= 0.0f) return;
  int minimum_x, minimum_y, maximum_x, maximum_y;
  tile_bounds(
      projected_means[2 * gaussian],
      projected_means[2 * gaussian + 1],
      radii[gaussian],
      tiles_x,
      tiles_y,
      minimum_x,
      minimum_y,
      maximum_x,
      maximum_y);
  int64_t output = offsets[gaussian];
  const uint32_t depth_bits = __float_as_uint(depths[gaussian]);
  for (int tile_y = minimum_y; tile_y < maximum_y; ++tile_y) {
    for (int tile_x = minimum_x; tile_x < maximum_x; ++tile_x) {
      const uint32_t tile = static_cast<uint32_t>(tile_y * tiles_x + tile_x);
      keys[output] = static_cast<int64_t>((static_cast<uint64_t>(tile) << 32) | depth_bits);
      gaussian_ids[output] = gaussian;
      ++output;
    }
  }
}

__global__ void identify_tile_ranges_kernel(
    int64_t reference_count,
    const int64_t* keys,
    int64_t* ranges) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= reference_count) return;
  const uint32_t tile = static_cast<uint64_t>(keys[index]) >> 32;
  if (index == 0 || (static_cast<uint64_t>(keys[index - 1]) >> 32) != tile) {
    ranges[2 * tile] = index;
  }
  if (index + 1 == reference_count || (static_cast<uint64_t>(keys[index + 1]) >> 32) != tile) {
    ranges[2 * tile + 1] = index + 1;
  }
}

__global__ void integrate_points_kernel(
    int query_count,
    const float* query_points,
    const float* projected_means,
    const float* conics,
    const float* precisions,
    const float* precision_means,
    const float* opacities,
    const float* radii,
    const int32_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    const float* viewmatrix,
    int height,
    int width,
    float tanfovx,
    float tanfovy,
    float cx,
    float cy,
    float* output_transmittance,
    bool* output_inside) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= query_count) return;

  const float world_x = query_points[3 * query + 0];
  const float world_y = query_points[3 * query + 1];
  const float world_z = query_points[3 * query + 2];
  const float camera_x =
      world_x * viewmatrix[0] +
      world_y * viewmatrix[4] +
      world_z * viewmatrix[8] +
      viewmatrix[12];
  const float camera_y =
      world_x * viewmatrix[1] +
      world_y * viewmatrix[5] +
      world_z * viewmatrix[9] +
      viewmatrix[13];
  const float camera_z =
      world_x * viewmatrix[2] +
      world_y * viewmatrix[6] +
      world_z * viewmatrix[10] +
      viewmatrix[14];
  if (!isfinite(camera_x) ||
      !isfinite(camera_y) ||
      !isfinite(camera_z) ||
      camera_z <= kNearPlane) {
    output_transmittance[query] = 1.0f;
    output_inside[query] = false;
    return;
  }

  const float focal_x = 0.5f * static_cast<float>(width) / tanfovx;
  const float focal_y = 0.5f * static_cast<float>(height) / tanfovy;
  const float sample_x = focal_x * camera_x / camera_z + cx;
  const float sample_y = focal_y * camera_y / camera_z + cy;
  if (!isfinite(sample_x) ||
      !isfinite(sample_y) ||
      sample_x < 0.0f ||
      sample_x >= static_cast<float>(width) ||
      sample_y < 0.0f ||
      sample_y >= static_cast<float>(height)) {
    output_transmittance[query] = 1.0f;
    output_inside[query] = false;
    return;
  }

  const float query_distance = sqrtf(
      camera_x * camera_x +
      camera_y * camera_y +
      camera_z * camera_z);
  if (!isfinite(query_distance) || query_distance <= 1e-8f) {
    output_transmittance[query] = 1.0f;
    output_inside[query] = false;
    return;
  }
  const float inverse_distance = 1.0f / query_distance;
  const float ray_x = camera_x * inverse_distance;
  const float ray_y = camera_y * inverse_distance;
  const float ray_z = camera_z * inverse_distance;

  const int tiles_x = (width + kTileWidth - 1) / kTileWidth;
  const int tile =
      (static_cast<int>(sample_y) / kTileHeight) * tiles_x +
      static_cast<int>(sample_x) / kTileWidth;
  const int64_t range_start = tile_ranges[2 * tile];
  const int64_t range_end = tile_ranges[2 * tile + 1];
  float transmittance = 1.0f;

  for (int64_t reference = range_start; reference < range_end; ++reference) {
    const int gaussian = sorted_gaussian_ids[reference];
    const float dx = sample_x - projected_means[2 * gaussian];
    const float dy = sample_y - projected_means[2 * gaussian + 1];
    if (fabsf(dx) > radii[gaussian] || fabsf(dy) > radii[gaussian]) continue;

    const float exponent = -0.5f * (
        conics[3 * gaussian] * dx * dx +
        2.0f * conics[3 * gaussian + 1] * dx * dy +
        conics[3 * gaussian + 2] * dy * dy);
    if (exponent > 0.0f) continue;
    const float alpha = fminf(
        kAlphaCap,
        fmaxf(0.0f, opacities[gaussian] * expf(exponent)));
    if (alpha < kAlphaThreshold) continue;

    const float p00 = precisions[6 * gaussian + 0];
    const float p01 = precisions[6 * gaussian + 1];
    const float p02 = precisions[6 * gaussian + 2];
    const float p11 = precisions[6 * gaussian + 3];
    const float p12 = precisions[6 * gaussian + 4];
    const float p22 = precisions[6 * gaussian + 5];
    const float precision_ray_x = p00 * ray_x + p01 * ray_y + p02 * ray_z;
    const float precision_ray_y = p01 * ray_x + p11 * ray_y + p12 * ray_z;
    const float precision_ray_z = p02 * ray_x + p12 * ray_y + p22 * ray_z;
    const float inverse_variance =
        ray_x * precision_ray_x +
        ray_y * precision_ray_y +
        ray_z * precision_ray_z;
    if (!isfinite(inverse_variance) || inverse_variance <= 1e-12f) continue;

    const float projected_mean =
        ray_x * precision_means[3 * gaussian + 0] +
        ray_y * precision_means[3 * gaussian + 1] +
        ray_z * precision_means[3 * gaussian + 2];
    const float peak_distance = projected_mean / inverse_variance;
    if (!isfinite(peak_distance) || peak_distance <= 0.0f) continue;

    float attenuation = 1.0f - alpha;
    if (query_distance < peak_distance) {
      const float normalized_delta =
          (peak_distance - query_distance) * sqrtf(inverse_variance);
      const float longitudinal_weight =
          expf(-0.5f * normalized_delta * normalized_delta);
      attenuation = 1.0f - alpha * longitudinal_weight;
    }
    transmittance *= fmaxf(attenuation, 0.0f);
    if (transmittance <= 1e-6f) {
      transmittance = 0.0f;
      break;
    }
  }

  output_transmittance[query] = transmittance;
  output_inside[query] = true;
}

__global__ void render_kernel(
    const float* projected_means,
    const float* conics,
    const float* depths,
    const float* projected_normals,
    const float* colors,
    const float* semantics,
    const float* opacities,
    const int32_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    const float* background,
    const float* radii,
    int height,
    int width,
    float* output_color,
    float* output_semantic,
    float* output_depth,
    float* output_alpha,
    float* output_normal,
    int64_t* dominant_index) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixel_count = height * width;
  if (pixel >= pixel_count) return;
  const int pixel_x = pixel % width;
  const int pixel_y = pixel / width;
  const float sample_x = static_cast<float>(pixel_x) + 0.5f;
  const float sample_y = static_cast<float>(pixel_y) + 0.5f;
  const int tiles_x = (width + kTileWidth - 1) / kTileWidth;
  const int tile = (pixel_y / kTileHeight) * tiles_x + pixel_x / kTileWidth;
  const int64_t range_start = tile_ranges[2 * tile];
  const int64_t range_end = tile_ranges[2 * tile + 1];

  float transmittance = 1.0f;
  float color[3] = {0.0f, 0.0f, 0.0f};
  float semantic[kSemanticDim] = {0.0f};
  float depth = 0.0f;
  float normal[3] = {0.0f, 0.0f, 0.0f};
  float largest_weight = 0.0f;
  int64_t dominant = -1;

  for (int64_t reference = range_start; reference < range_end; ++reference) {
    const int gaussian = sorted_gaussian_ids[reference];
    const float dx = sample_x - projected_means[2 * gaussian];
    const float dy = sample_y - projected_means[2 * gaussian + 1];
    if (fabsf(dx) > radii[gaussian] || fabsf(dy) > radii[gaussian]) continue;
    const float exponent = -0.5f * (
        conics[3 * gaussian] * dx * dx +
        2.0f * conics[3 * gaussian + 1] * dx * dy +
        conics[3 * gaussian + 2] * dy * dy);
    if (exponent > 0.0f) continue;
    const float alpha = fminf(kAlphaCap, fmaxf(0.0f, opacities[gaussian] * expf(exponent)));
    if (alpha < kAlphaThreshold) continue;
    const float weight = transmittance * alpha;
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
      color[channel] += weight * colors[3 * gaussian + channel];
    }
#pragma unroll
    for (int channel = 0; channel < kSemanticDim; ++channel) {
      semantic[channel] += weight * semantics[kSemanticDim * gaussian + channel];
    }
    depth += weight * depths[gaussian];
    normal[0] += weight * projected_normals[3 * gaussian];
    normal[1] += weight * projected_normals[3 * gaussian + 1];
    normal[2] += weight * projected_normals[3 * gaussian + 2];
    if (weight > largest_weight) {
      largest_weight = weight;
      dominant = gaussian;
    }
    transmittance *= 1.0f - alpha;
  }

  const float alpha = 1.0f - transmittance;
  output_alpha[pixel] = alpha;
#pragma unroll
  for (int channel = 0; channel < 3; ++channel) {
    output_color[channel * pixel_count + pixel] = color[channel] + transmittance * background[channel];
  }
  if (alpha > 1e-8f) {
    const float inverse_alpha = 1.0f / alpha;
    output_depth[pixel] = depth * inverse_alpha;
#pragma unroll
    for (int channel = 0; channel < kSemanticDim; ++channel) {
      output_semantic[channel * pixel_count + pixel] = semantic[channel] * inverse_alpha;
    }
    const float normal_length = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    const float inverse_normal = normal_length > 1e-8f ? 1.0f / normal_length : 0.0f;
    output_normal[pixel] = normal[0] * inverse_normal;
    output_normal[pixel_count + pixel] = normal[1] * inverse_normal;
    output_normal[2 * pixel_count + pixel] = normal[2] * inverse_normal;
    dominant_index[pixel] = dominant;
  } else {
    output_depth[pixel] = 0.0f;
#pragma unroll
    for (int channel = 0; channel < kSemanticDim; ++channel) {
      output_semantic[channel * pixel_count + pixel] = 0.0f;
    }
    output_normal[pixel] = 0.0f;
    output_normal[pixel_count + pixel] = 0.0f;
    output_normal[2 * pixel_count + pixel] = 0.0f;
    dominant_index[pixel] = -1;
  }
}

__device__ inline bool evaluate_alpha(
    int gaussian,
    float sample_x,
    float sample_y,
    const float* projected_means,
    const float* conics,
    const float* radii,
    const float* opacities,
    float& dx,
    float& dy,
    float& exponent,
    float& footprint,
    float& raw_alpha,
    float& alpha) {
  dx = sample_x - projected_means[2 * gaussian];
  dy = sample_y - projected_means[2 * gaussian + 1];
  if (fabsf(dx) > radii[gaussian] || fabsf(dy) > radii[gaussian]) return false;
  exponent = -0.5f * (
      conics[3 * gaussian] * dx * dx +
      2.0f * conics[3 * gaussian + 1] * dx * dy +
      conics[3 * gaussian + 2] * dy * dy);
  if (exponent > 0.0f) return false;
  footprint = expf(exponent);
  raw_alpha = opacities[gaussian] * footprint;
  alpha = fminf(kAlphaCap, fmaxf(0.0f, raw_alpha));
  return alpha >= kAlphaThreshold;
}

__global__ void render_backward_kernel(
    const float* projected_means,
    const float* conics,
    const float* depths,
    const float* projected_normals,
    const float* colors,
    const float* semantics,
    const float* opacities,
    const int32_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    const float* background,
    const float* radii,
    const float* output_semantic,
    const float* output_depth,
    const float* output_alpha,
    const float* output_normal,
    const float* grad_color,
    const float* grad_semantic,
    const float* grad_depth,
    const float* grad_alpha,
    const float* grad_normal,
    int height,
    int width,
    float* grad_projected_means,
    float* grad_conics,
    float* grad_projected_depths,
    float* grad_projected_normals,
    float* grad_colors,
    float* grad_semantics,
    float* grad_opacities) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixel_count = height * width;
  if (pixel >= pixel_count) return;
  const int pixel_x = pixel % width;
  const int pixel_y = pixel / width;
  const float sample_x = static_cast<float>(pixel_x) + 0.5f;
  const float sample_y = static_cast<float>(pixel_y) + 0.5f;
  const int tiles_x = (width + kTileWidth - 1) / kTileWidth;
  const int tile = (pixel_y / kTileHeight) * tiles_x + pixel_x / kTileWidth;
  const int64_t range_start = tile_ranges[2 * tile];
  const int64_t range_end = tile_ranges[2 * tile + 1];

  // The reverse recurrence needs T before each splat.  Accumulating log(T)
  // prevents an opaque ray from losing all earlier-prefix information when
  // 1 - output_alpha rounds to zero in float32.
  float final_log_transmittance = 0.0f;
  float accumulated_normal[3] = {0.0f, 0.0f, 0.0f};
  float prefix_transmittance = 1.0f;
  for (int64_t reference = range_start; reference < range_end; ++reference) {
    const int gaussian = sorted_gaussian_ids[reference];
    float dx, dy, exponent, footprint, raw_alpha, alpha;
    if (!evaluate_alpha(
            gaussian,
            sample_x,
            sample_y,
            projected_means,
            conics,
            radii,
            opacities,
            dx,
            dy,
            exponent,
            footprint,
            raw_alpha,
            alpha)) {
      continue;
    }
    const float weight = prefix_transmittance * alpha;
    accumulated_normal[0] += weight * projected_normals[3 * gaussian];
    accumulated_normal[1] += weight * projected_normals[3 * gaussian + 1];
    accumulated_normal[2] += weight * projected_normals[3 * gaussian + 2];
    prefix_transmittance *= 1.0f - alpha;
    final_log_transmittance += log1pf(-alpha);
  }

  const float alpha_total = output_alpha[pixel];
  const float upstream_depth = grad_depth[pixel];
  const float upstream_alpha = grad_alpha[pixel];
  float grad_semantic_acc[kSemanticDim] = {0.0f};
  float grad_depth_acc = 0.0f;
  float grad_alpha_total = upstream_alpha;
  if (alpha_total > 1e-8f) {
    const float inverse_alpha = 1.0f / alpha_total;
#pragma unroll
    for (int channel = 0; channel < kSemanticDim; ++channel) {
      const float upstream = grad_semantic[channel * pixel_count + pixel];
      grad_semantic_acc[channel] = upstream * inverse_alpha;
      grad_alpha_total -= upstream * output_semantic[channel * pixel_count + pixel] * inverse_alpha;
    }
    grad_depth_acc = upstream_depth * inverse_alpha;
    grad_alpha_total -= upstream_depth * output_depth[pixel] * inverse_alpha;
  }

  float grad_normal_acc[3] = {0.0f, 0.0f, 0.0f};
  const float normal_length = sqrtf(
      accumulated_normal[0] * accumulated_normal[0] +
      accumulated_normal[1] * accumulated_normal[1] +
      accumulated_normal[2] * accumulated_normal[2]);
  if (alpha_total > 1e-8f && normal_length > 1e-8f) {
    float normal_dot = 0.0f;
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
      normal_dot += grad_normal[channel * pixel_count + pixel] * output_normal[channel * pixel_count + pixel];
    }
    const float inverse_normal_length = 1.0f / normal_length;
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
      grad_normal_acc[channel] =
          (grad_normal[channel * pixel_count + pixel] -
           normal_dot * output_normal[channel * pixel_count + pixel]) *
          inverse_normal_length;
    }
  }

  float transmittance_adjoint = -grad_alpha_total;
#pragma unroll
  for (int channel = 0; channel < 3; ++channel) {
    transmittance_adjoint += grad_color[channel * pixel_count + pixel] * background[channel];
  }

  float log_transmittance_after = final_log_transmittance;
  for (int64_t reference = range_end; reference-- > range_start;) {
    const int gaussian = sorted_gaussian_ids[reference];
    float dx, dy, exponent, footprint, raw_alpha, alpha;
    if (!evaluate_alpha(
            gaussian,
            sample_x,
            sample_y,
            projected_means,
            conics,
            radii,
            opacities,
            dx,
            dy,
            exponent,
            footprint,
            raw_alpha,
            alpha)) {
      continue;
    }
    const float log_one_minus_alpha = log1pf(-alpha);
    const float log_transmittance_before = log_transmittance_after - log_one_minus_alpha;
    const float transmittance_before = expf(log_transmittance_before);
    const float weight = transmittance_before * alpha;

    float attribute_adjoint = grad_depth_acc * depths[gaussian];
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
      attribute_adjoint += grad_color[channel * pixel_count + pixel] * colors[3 * gaussian + channel];
      attribute_adjoint += grad_normal_acc[channel] * projected_normals[3 * gaussian + channel];
      atomicAdd(
          grad_colors + 3 * gaussian + channel,
          weight * grad_color[channel * pixel_count + pixel]);
      atomicAdd(
          grad_projected_normals + 3 * gaussian + channel,
          weight * grad_normal_acc[channel]);
    }
#pragma unroll
    for (int channel = 0; channel < kSemanticDim; ++channel) {
      attribute_adjoint += grad_semantic_acc[channel] * semantics[kSemanticDim * gaussian + channel];
      atomicAdd(
          grad_semantics + kSemanticDim * gaussian + channel,
          weight * grad_semantic_acc[channel]);
    }
    atomicAdd(grad_projected_depths + gaussian, weight * grad_depth_acc);

    const float grad_splat_alpha =
        transmittance_before * (attribute_adjoint - transmittance_adjoint);
    transmittance_adjoint =
        alpha * attribute_adjoint + (1.0f - alpha) * transmittance_adjoint;

    // clamp(raw_alpha, 0, .99) and the alpha threshold are piecewise discrete.
    // Gradients flow only through the open, non-saturated interval.
    if (raw_alpha > 0.0f && raw_alpha < kAlphaCap) {
      atomicAdd(grad_opacities + gaussian, grad_splat_alpha * footprint);
      const float grad_exponent = grad_splat_alpha * raw_alpha;
      atomicAdd(
          grad_projected_means + 2 * gaussian,
          grad_exponent * (conics[3 * gaussian] * dx + conics[3 * gaussian + 1] * dy));
      atomicAdd(
          grad_projected_means + 2 * gaussian + 1,
          grad_exponent * (conics[3 * gaussian + 1] * dx + conics[3 * gaussian + 2] * dy));
      atomicAdd(grad_conics + 3 * gaussian, grad_exponent * (-0.5f * dx * dx));
      atomicAdd(grad_conics + 3 * gaussian + 1, grad_exponent * (-dx * dy));
      atomicAdd(grad_conics + 3 * gaussian + 2, grad_exponent * (-0.5f * dy * dy));
    }
    log_transmittance_after = log_transmittance_before;
  }
}

__global__ void projection_backward_kernel(
    int count,
    const float* means3d,
    const float* means2d,
    const float* scales,
    const float* rotations,
    const float* viewmatrix,
    const float* radii,
    const float* grad_projected_means,
    const float* grad_conics,
    const float* grad_projected_depths,
    const float* grad_projected_normals,
    int height,
    int width,
    float tanfovx,
    float tanfovy,
    float cx,
    float cy,
    float scale_modifier,
    float antialias_sigma,
    float* grad_means3d,
    float* grad_means2d,
    float* grad_scales,
    float* grad_rotations) {
  const int gaussian = blockIdx.x * blockDim.x + threadIdx.x;
  if (gaussian >= count || radii[gaussian] <= 0.0f) return;

#pragma unroll 1
  for (int seed = 0; seed < 12; ++seed) {
    const ProjectionDual projected = project_one_dual(
        gaussian,
        seed,
        means3d,
        means2d,
        scales,
        rotations,
        viewmatrix,
        height,
        width,
        tanfovx,
        tanfovy,
        cx,
        cy,
        scale_modifier,
        antialias_sigma);
    float gradient =
        grad_projected_means[2 * gaussian] * projected.mean_x.tangent +
        grad_projected_means[2 * gaussian + 1] * projected.mean_y.tangent +
        grad_conics[3 * gaussian] * projected.conic_a.tangent +
        grad_conics[3 * gaussian + 1] * projected.conic_b.tangent +
        grad_conics[3 * gaussian + 2] * projected.conic_c.tangent +
        grad_projected_depths[gaussian] * projected.depth.tangent +
        grad_projected_normals[3 * gaussian] * projected.normal_x.tangent +
        grad_projected_normals[3 * gaussian + 1] * projected.normal_y.tangent +
        grad_projected_normals[3 * gaussian + 2] * projected.normal_z.tangent;
    if (seed < 3) {
      grad_means3d[3 * gaussian + seed] = gradient;
    } else if (seed < 5) {
      grad_means2d[3 * gaussian + seed - 3] = gradient;
    } else if (seed < 8) {
      grad_scales[3 * gaussian + seed - 5] = gradient;
    } else {
      grad_rotations[4 * gaussian + seed - 8] = gradient;
    }
  }
}

struct ProjectionState {
  torch::Tensor projected_means;
  torch::Tensor conics;
  torch::Tensor depths;
  torch::Tensor projected_normals;
  torch::Tensor precisions;
  torch::Tensor precision_means;
  torch::Tensor radii;
  torch::Tensor sorted_gaussian_ids;
  torch::Tensor tile_ranges;
};

ProjectionState prepare_projection_state(
    const torch::Tensor& means3d,
    const torch::Tensor* means2d,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& viewmatrix,
    int image_height,
    int image_width,
    float tanfovx,
    float tanfovy,
    float cx,
    float cy,
    float scale_modifier,
    float antialias_sigma,
    bool point_integration) {
  const int count = static_cast<int>(means3d.size(0));
  auto float_options = means3d.options();
  auto long_options = means3d.options().dtype(torch::kLong);
  ProjectionState state{
      torch::zeros({count, 2}, float_options),
      torch::zeros({count, 3}, float_options),
      torch::zeros({count}, float_options),
      point_integration
          ? torch::empty({0, 3}, float_options)
          : torch::zeros({count, 3}, float_options),
      point_integration
          ? torch::zeros({count, 6}, float_options)
          : torch::empty({0, 6}, float_options),
      point_integration
          ? torch::zeros({count, 3}, float_options)
          : torch::empty({0, 3}, float_options),
      torch::zeros({count}, float_options),
      torch::Tensor(),
      torch::Tensor()};
  auto tile_counts = torch::zeros({count}, long_options);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  if (count > 0) {
    if (point_integration) {
      project_integration_gaussians_kernel<<<
          (count + threads - 1) / threads,
          threads,
          0,
          stream>>>(
          count,
          means3d.data_ptr<float>(),
          scales.data_ptr<float>(),
          rotations.data_ptr<float>(),
          viewmatrix.data_ptr<float>(),
          image_height,
          image_width,
          tanfovx,
          tanfovy,
          cx,
          cy,
          scale_modifier,
          antialias_sigma,
          state.projected_means.data_ptr<float>(),
          state.conics.data_ptr<float>(),
          state.depths.data_ptr<float>(),
          state.precisions.data_ptr<float>(),
          state.precision_means.data_ptr<float>(),
          state.radii.data_ptr<float>(),
          tile_counts.data_ptr<int64_t>());
    } else {
      project_and_count_kernel<<<
          (count + threads - 1) / threads,
          threads,
          0,
          stream>>>(
          count,
          means3d.data_ptr<float>(),
          means2d->data_ptr<float>(),
          scales.data_ptr<float>(),
          rotations.data_ptr<float>(),
          viewmatrix.data_ptr<float>(),
          image_height,
          image_width,
          tanfovx,
          tanfovy,
          cx,
          cy,
          scale_modifier,
          antialias_sigma,
          state.projected_means.data_ptr<float>(),
          state.conics.data_ptr<float>(),
          state.depths.data_ptr<float>(),
          state.projected_normals.data_ptr<float>(),
          state.radii.data_ptr<float>(),
          tile_counts.data_ptr<int64_t>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto offsets = torch::zeros({count}, long_options);
  int64_t reference_count = 0;
  if (count > 0) {
    auto inclusive_offsets = at::cumsum(tile_counts, 0, at::kLong);
    offsets = inclusive_offsets - tile_counts;
    reference_count = inclusive_offsets[count - 1].item<int64_t>();
  }
  auto keys = torch::empty({reference_count}, long_options);
  state.sorted_gaussian_ids =
      torch::empty({reference_count}, means3d.options().dtype(torch::kInt));
  const int tiles_x = (image_width + kTileWidth - 1) / kTileWidth;
  const int tiles_y = (image_height + kTileHeight - 1) / kTileHeight;
  if (reference_count > 0) {
    duplicate_with_keys_kernel<<<
        (count + threads - 1) / threads,
        threads,
        0,
        stream>>>(
        count,
        state.projected_means.data_ptr<float>(),
        state.depths.data_ptr<float>(),
        state.radii.data_ptr<float>(),
        offsets.data_ptr<int64_t>(),
        tiles_x,
        tiles_y,
        keys.data_ptr<int64_t>(),
        state.sorted_gaussian_ids.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    thrust::device_ptr<int64_t> key_begin(keys.data_ptr<int64_t>());
    thrust::device_ptr<int32_t> id_begin(
        state.sorted_gaussian_ids.data_ptr<int32_t>());
    thrust::sort_by_key(
        thrust::cuda::par.on(stream),
        key_begin,
        key_begin + reference_count,
        id_begin);
  }

  state.tile_ranges = torch::zeros({tiles_x * tiles_y, 2}, long_options);
  if (reference_count > 0) {
    identify_tile_ranges_kernel<<<
        (reference_count + threads - 1) / threads,
        threads,
        0,
        stream>>>(
        reference_count,
        keys.data_ptr<int64_t>(),
        state.tile_ranges.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return state;
}

}  // namespace

std::vector<torch::Tensor> semantic_gaussian_rasterize_forward(
    const torch::Tensor& means3d,
    const torch::Tensor& means2d,
    const torch::Tensor& colors,
    const torch::Tensor& semantics,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& background,
    int64_t image_height,
    int64_t image_width,
    double tanfovx,
    double tanfovy,
    double cx,
    double cy,
    double scale_modifier,
    double antialias_sigma) {
  CHECK_INPUT(means3d);
  CHECK_INPUT(means2d);
  CHECK_INPUT(colors);
  CHECK_INPUT(semantics);
  CHECK_INPUT(opacities);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(background);
  TORCH_CHECK(means3d.dim() == 2 && means3d.size(1) == 3, "means3d must be [N,3]");
  const int count = static_cast<int>(means3d.size(0));
  TORCH_CHECK(means2d.sizes() == means3d.sizes(), "means2d must be [N,3]");
  TORCH_CHECK(colors.dim() == 2 && colors.size(0) == count && colors.size(1) == 3, "colors must be [N,3]");
  TORCH_CHECK(semantics.dim() == 2 && semantics.size(0) == count && semantics.size(1) == kSemanticDim, "semantics must be [N,16]");
  TORCH_CHECK(opacities.numel() == count, "opacities must contain N elements");
  TORCH_CHECK(scales.sizes() == means3d.sizes(), "scales must be [N,3]");
  TORCH_CHECK(rotations.dim() == 2 && rotations.size(0) == count && rotations.size(1) == 4, "rotations must be [N,4]");
  TORCH_CHECK(viewmatrix.sizes() == torch::IntArrayRef({4, 4}), "viewmatrix must be [4,4]");
  TORCH_CHECK(background.numel() == 3, "background must contain 3 elements");
  TORCH_CHECK(image_height > 0 && image_width > 0, "image dimensions must be positive");
  TORCH_CHECK(std::isfinite(cx) && std::isfinite(cy), "principal point must be finite");
  TORCH_CHECK(means3d.size(0) <= std::numeric_limits<int>::max(), "too many Gaussians for the CUDA backend");
  TORCH_CHECK(
      image_height <= std::numeric_limits<int>::max() / image_width,
      "image contains too many pixels for the CUDA backend");

  const c10::cuda::CUDAGuard device_guard(means3d.device());
  auto float_options = means3d.options();
  auto long_options = means3d.options().dtype(torch::kLong);
  auto output_color = torch::zeros({3, image_height, image_width}, float_options);
  auto output_semantic = torch::zeros({kSemanticDim, image_height, image_width}, float_options);
  auto output_depth = torch::zeros({1, image_height, image_width}, float_options);
  auto output_alpha = torch::zeros({1, image_height, image_width}, float_options);
  auto output_normal = torch::zeros({3, image_height, image_width}, float_options);
  auto dominant = torch::full({image_height, image_width}, -1, long_options);
  auto projection = prepare_projection_state(
      means3d,
      &means2d,
      scales,
      rotations,
      viewmatrix,
      static_cast<int>(image_height),
      static_cast<int>(image_width),
      static_cast<float>(tanfovx),
      static_cast<float>(tanfovy),
      static_cast<float>(cx),
      static_cast<float>(cy),
      static_cast<float>(scale_modifier),
      static_cast<float>(antialias_sigma),
      false);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  const int pixel_count = static_cast<int>(image_height * image_width);
  render_kernel<<<(pixel_count + threads - 1) / threads, threads, 0, stream>>>(
      projection.projected_means.data_ptr<float>(),
      projection.conics.data_ptr<float>(),
      projection.depths.data_ptr<float>(),
      projection.projected_normals.data_ptr<float>(),
      colors.data_ptr<float>(),
      semantics.data_ptr<float>(),
      opacities.data_ptr<float>(),
      projection.sorted_gaussian_ids.data_ptr<int32_t>(),
      projection.tile_ranges.data_ptr<int64_t>(),
      background.data_ptr<float>(),
      projection.radii.data_ptr<float>(),
      static_cast<int>(image_height),
      static_cast<int>(image_width),
      output_color.data_ptr<float>(),
      output_semantic.data_ptr<float>(),
      output_depth.data_ptr<float>(),
      output_alpha.data_ptr<float>(),
      output_normal.data_ptr<float>(),
      dominant.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // The final six tensors are opaque autograd state.  Python exposes only the
  // first seven public results and retains these O(N + tile-overlap) buffers
  // for the native backward pass.
  return {
      output_color,
      output_semantic,
      output_depth,
      output_alpha,
      output_normal,
      projection.radii,
      dominant,
      projection.projected_means,
      projection.conics,
      projection.depths,
      projection.projected_normals,
      projection.sorted_gaussian_ids,
      projection.tile_ranges};
}

std::vector<torch::Tensor> semantic_gaussian_prepare_point_integration(
    const torch::Tensor& means3d,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& viewmatrix,
    int64_t image_height,
    int64_t image_width,
    double tanfovx,
    double tanfovy,
    double cx,
    double cy,
    double scale_modifier,
    double antialias_sigma) {
  CHECK_INPUT(means3d);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(viewmatrix);
  TORCH_CHECK(
      means3d.dim() == 2 && means3d.size(1) == 3,
      "means3d must be [N,3]");
  const int64_t count = means3d.size(0);
  TORCH_CHECK(scales.sizes() == means3d.sizes(), "scales must be [N,3]");
  TORCH_CHECK(
      rotations.dim() == 2 &&
          rotations.size(0) == count &&
          rotations.size(1) == 4,
      "rotations must be [N,4]");
  TORCH_CHECK(
      viewmatrix.sizes() == torch::IntArrayRef({4, 4}),
      "viewmatrix must be [4,4]");
  TORCH_CHECK(
      scales.device() == means3d.device() &&
          rotations.device() == means3d.device() &&
          viewmatrix.device() == means3d.device(),
      "all point-integration inputs must be on the same CUDA device");
  TORCH_CHECK(
      image_height > 0 && image_width > 0,
      "image dimensions must be positive");
  TORCH_CHECK(
      image_height <= std::numeric_limits<int>::max() / image_width,
      "image contains too many pixels for the CUDA backend");
  TORCH_CHECK(
      count <= std::numeric_limits<int>::max(),
      "too many Gaussians for the CUDA backend");
  TORCH_CHECK(
      std::isfinite(tanfovx) &&
          std::isfinite(tanfovy) &&
          tanfovx > 0.0 &&
          tanfovy > 0.0,
      "tanfovx and tanfovy must be finite and positive");
  TORCH_CHECK(
      std::isfinite(cx) && std::isfinite(cy),
      "principal point must be finite");
  TORCH_CHECK(
      std::isfinite(scale_modifier) && scale_modifier > 0.0,
      "scale_modifier must be finite and positive");
  TORCH_CHECK(
      std::isfinite(antialias_sigma) && antialias_sigma >= 0.0,
      "antialias_sigma must be finite and non-negative");

  const c10::cuda::CUDAGuard device_guard(means3d.device());
  auto projection = prepare_projection_state(
      means3d,
      nullptr,
      scales,
      rotations,
      viewmatrix,
      static_cast<int>(image_height),
      static_cast<int>(image_width),
      static_cast<float>(tanfovx),
      static_cast<float>(tanfovy),
      static_cast<float>(cx),
      static_cast<float>(cy),
      static_cast<float>(scale_modifier),
      static_cast<float>(antialias_sigma),
      true);
  return {
      projection.projected_means,
      projection.conics,
      projection.precisions,
      projection.precision_means,
      projection.radii,
      projection.sorted_gaussian_ids,
      projection.tile_ranges};
}

std::vector<torch::Tensor> semantic_gaussian_integrate_points_forward(
    const torch::Tensor& query_points,
    const torch::Tensor& projected_means,
    const torch::Tensor& conics,
    const torch::Tensor& precisions,
    const torch::Tensor& precision_means,
    const torch::Tensor& opacities,
    const torch::Tensor& radii,
    const torch::Tensor& sorted_gaussian_ids,
    const torch::Tensor& tile_ranges,
    const torch::Tensor& viewmatrix,
    int64_t image_height,
    int64_t image_width,
    double tanfovx,
    double tanfovy,
    double cx,
    double cy) {
  CHECK_INPUT(query_points);
  CHECK_INPUT(projected_means);
  CHECK_INPUT(conics);
  CHECK_INPUT(precisions);
  CHECK_INPUT(precision_means);
  CHECK_INPUT(opacities);
  CHECK_INPUT(radii);
  CHECK_INPUT(viewmatrix);
  CHECK_CUDA(sorted_gaussian_ids);
  CHECK_CONTIGUOUS(sorted_gaussian_ids);
  CHECK_CUDA(tile_ranges);
  CHECK_CONTIGUOUS(tile_ranges);
  TORCH_CHECK(
      sorted_gaussian_ids.scalar_type() == at::kInt,
      "sorted_gaussian_ids must be int32");
  TORCH_CHECK(
      tile_ranges.scalar_type() == at::kLong,
      "tile_ranges must be int64");
  TORCH_CHECK(
      query_points.dim() == 2 && query_points.size(1) == 3,
      "query_points must be [Q,3]");
  TORCH_CHECK(
      projected_means.dim() == 2 && projected_means.size(1) == 2,
      "projected_means must be [N,2]");
  const int64_t count = projected_means.size(0);
  TORCH_CHECK(
      conics.sizes() == torch::IntArrayRef({count, 3}),
      "conics must be [N,3]");
  TORCH_CHECK(
      precisions.sizes() == torch::IntArrayRef({count, 6}),
      "precisions must be [N,6]");
  TORCH_CHECK(
      precision_means.sizes() == torch::IntArrayRef({count, 3}),
      "precision_means must be [N,3]");
  TORCH_CHECK(opacities.numel() == count, "opacities must contain N elements");
  TORCH_CHECK(radii.numel() == count, "radii must contain N elements");
  TORCH_CHECK(
      viewmatrix.sizes() == torch::IntArrayRef({4, 4}),
      "viewmatrix must be [4,4]");
  TORCH_CHECK(
      image_height > 0 && image_width > 0,
      "image dimensions must be positive");
  TORCH_CHECK(
      query_points.size(0) <= std::numeric_limits<int>::max(),
      "too many query points for one CUDA launch");
  TORCH_CHECK(
      std::isfinite(tanfovx) &&
          std::isfinite(tanfovy) &&
          tanfovx > 0.0 &&
          tanfovy > 0.0,
      "tanfovx and tanfovy must be finite and positive");
  TORCH_CHECK(
      std::isfinite(cx) && std::isfinite(cy),
      "principal point must be finite");
  const int64_t tiles_x =
      (image_width + static_cast<int64_t>(kTileWidth) - 1) / kTileWidth;
  const int64_t tiles_y =
      (image_height + static_cast<int64_t>(kTileHeight) - 1) / kTileHeight;
  TORCH_CHECK(
      tile_ranges.sizes() == torch::IntArrayRef({tiles_x * tiles_y, 2}),
      "tile_ranges has an incompatible image grid");

  const auto device = query_points.device();
  TORCH_CHECK(
      projected_means.device() == device &&
          conics.device() == device &&
          precisions.device() == device &&
          precision_means.device() == device &&
          opacities.device() == device &&
          radii.device() == device &&
          sorted_gaussian_ids.device() == device &&
          tile_ranges.device() == device &&
          viewmatrix.device() == device,
      "all point-integration tensors must be on the same CUDA device");

  const c10::cuda::CUDAGuard device_guard(device);
  const int query_count = static_cast<int>(query_points.size(0));
  auto transmittance = torch::ones({query_count}, query_points.options());
  auto inside = torch::zeros(
      {query_count},
      query_points.options().dtype(torch::kBool));
  if (query_count > 0) {
    constexpr int threads = 256;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    integrate_points_kernel<<<
        (query_count + threads - 1) / threads,
        threads,
        0,
        stream>>>(
        query_count,
        query_points.data_ptr<float>(),
        projected_means.data_ptr<float>(),
        conics.data_ptr<float>(),
        precisions.data_ptr<float>(),
        precision_means.data_ptr<float>(),
        opacities.data_ptr<float>(),
        radii.data_ptr<float>(),
        sorted_gaussian_ids.data_ptr<int32_t>(),
        tile_ranges.data_ptr<int64_t>(),
        viewmatrix.data_ptr<float>(),
        static_cast<int>(image_height),
        static_cast<int>(image_width),
        static_cast<float>(tanfovx),
        static_cast<float>(tanfovy),
        static_cast<float>(cx),
        static_cast<float>(cy),
        transmittance.data_ptr<float>(),
        inside.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {transmittance, inside};
}

std::vector<torch::Tensor> semantic_gaussian_rasterize_backward(
    const torch::Tensor& means3d,
    const torch::Tensor& means2d,
    const torch::Tensor& colors,
    const torch::Tensor& semantics,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& background,
    const torch::Tensor& projected_means,
    const torch::Tensor& conics,
    const torch::Tensor& depths,
    const torch::Tensor& projected_normals,
    const torch::Tensor& radii,
    const torch::Tensor& sorted_gaussian_ids,
    const torch::Tensor& tile_ranges,
    const torch::Tensor& output_semantic,
    const torch::Tensor& output_depth,
    const torch::Tensor& output_alpha,
    const torch::Tensor& output_normal,
    const torch::Tensor& grad_color,
    const torch::Tensor& grad_semantic,
    const torch::Tensor& grad_depth,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& grad_normal,
    int64_t image_height,
    int64_t image_width,
    double tanfovx,
    double tanfovy,
    double cx,
    double cy,
    double scale_modifier,
    double antialias_sigma) {
  CHECK_INPUT(means3d);
  CHECK_INPUT(means2d);
  CHECK_INPUT(colors);
  CHECK_INPUT(semantics);
  CHECK_INPUT(opacities);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(background);
  CHECK_INPUT(projected_means);
  CHECK_INPUT(conics);
  CHECK_INPUT(depths);
  CHECK_INPUT(projected_normals);
  CHECK_INPUT(radii);
  CHECK_INPUT(output_semantic);
  CHECK_INPUT(output_depth);
  CHECK_INPUT(output_alpha);
  CHECK_INPUT(output_normal);
  CHECK_INPUT(grad_color);
  CHECK_INPUT(grad_semantic);
  CHECK_INPUT(grad_depth);
  CHECK_INPUT(grad_alpha);
  CHECK_INPUT(grad_normal);
  CHECK_CUDA(sorted_gaussian_ids);
  CHECK_CONTIGUOUS(sorted_gaussian_ids);
  CHECK_CUDA(tile_ranges);
  CHECK_CONTIGUOUS(tile_ranges);
  TORCH_CHECK(sorted_gaussian_ids.scalar_type() == at::kInt, "sorted_gaussian_ids must be int32");
  TORCH_CHECK(tile_ranges.scalar_type() == at::kLong, "tile_ranges must be int64");

  TORCH_CHECK(means3d.dim() == 2 && means3d.size(1) == 3, "means3d must be [N,3]");
  const int64_t count64 = means3d.size(0);
  TORCH_CHECK(count64 <= std::numeric_limits<int>::max(), "too many Gaussians for the CUDA backend");
  const int count = static_cast<int>(count64);
  TORCH_CHECK(means2d.sizes() == means3d.sizes(), "means2d must be [N,3]");
  TORCH_CHECK(colors.sizes() == torch::IntArrayRef({count64, 3}), "colors must be [N,3]");
  TORCH_CHECK(semantics.sizes() == torch::IntArrayRef({count64, kSemanticDim}), "semantics must be [N,16]");
  TORCH_CHECK(opacities.numel() == count64, "opacities must contain N elements");
  TORCH_CHECK(scales.sizes() == means3d.sizes(), "scales must be [N,3]");
  TORCH_CHECK(rotations.sizes() == torch::IntArrayRef({count64, 4}), "rotations must be [N,4]");
  TORCH_CHECK(projected_means.sizes() == torch::IntArrayRef({count64, 2}), "projected_means must be [N,2]");
  TORCH_CHECK(conics.sizes() == torch::IntArrayRef({count64, 3}), "conics must be [N,3]");
  TORCH_CHECK(depths.numel() == count64, "depths must contain N elements");
  TORCH_CHECK(projected_normals.sizes() == torch::IntArrayRef({count64, 3}), "projected_normals must be [N,3]");
  TORCH_CHECK(radii.numel() == count64, "radii must contain N elements");
  TORCH_CHECK(viewmatrix.sizes() == torch::IntArrayRef({4, 4}), "viewmatrix must be [4,4]");
  TORCH_CHECK(background.numel() == 3, "background must contain 3 elements");
  TORCH_CHECK(image_height > 0 && image_width > 0, "image dimensions must be positive");
  TORCH_CHECK(std::isfinite(cx) && std::isfinite(cy), "principal point must be finite");
  TORCH_CHECK(
      image_height <= std::numeric_limits<int>::max() / image_width,
      "image contains too many pixels for the CUDA backend");
  const int64_t pixel_count64 = image_height * image_width;
  TORCH_CHECK(output_semantic.numel() == kSemanticDim * pixel_count64, "invalid output_semantic shape");
  TORCH_CHECK(output_depth.numel() == pixel_count64, "invalid output_depth shape");
  TORCH_CHECK(output_alpha.numel() == pixel_count64, "invalid output_alpha shape");
  TORCH_CHECK(output_normal.numel() == 3 * pixel_count64, "invalid output_normal shape");
  TORCH_CHECK(grad_color.numel() == 3 * pixel_count64, "invalid grad_color shape");
  TORCH_CHECK(grad_semantic.numel() == kSemanticDim * pixel_count64, "invalid grad_semantic shape");
  TORCH_CHECK(grad_depth.numel() == pixel_count64, "invalid grad_depth shape");
  TORCH_CHECK(grad_alpha.numel() == pixel_count64, "invalid grad_alpha shape");
  TORCH_CHECK(grad_normal.numel() == 3 * pixel_count64, "invalid grad_normal shape");
  const int tiles_x = (static_cast<int>(image_width) + kTileWidth - 1) / kTileWidth;
  const int tiles_y = (static_cast<int>(image_height) + kTileHeight - 1) / kTileHeight;
  TORCH_CHECK(tile_ranges.sizes() == torch::IntArrayRef({tiles_x * tiles_y, 2}), "invalid tile_ranges shape");

  const c10::cuda::CUDAGuard device_guard(means3d.device());
  auto grad_means3d = torch::zeros_like(means3d);
  auto grad_means2d = torch::zeros_like(means2d);
  auto grad_colors = torch::zeros_like(colors);
  auto grad_semantics = torch::zeros_like(semantics);
  auto grad_opacities = torch::zeros_like(opacities);
  auto grad_scales = torch::zeros_like(scales);
  auto grad_rotations = torch::zeros_like(rotations);
  auto grad_projected_means = torch::zeros_like(projected_means);
  auto grad_conics = torch::zeros_like(conics);
  auto grad_projected_depths = torch::zeros_like(depths);
  auto grad_projected_normals = torch::zeros_like(projected_normals);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  const int pixel_count = static_cast<int>(pixel_count64);
  render_backward_kernel<<<(pixel_count + threads - 1) / threads, threads, 0, stream>>>(
      projected_means.data_ptr<float>(),
      conics.data_ptr<float>(),
      depths.data_ptr<float>(),
      projected_normals.data_ptr<float>(),
      colors.data_ptr<float>(),
      semantics.data_ptr<float>(),
      opacities.data_ptr<float>(),
      sorted_gaussian_ids.data_ptr<int32_t>(),
      tile_ranges.data_ptr<int64_t>(),
      background.data_ptr<float>(),
      radii.data_ptr<float>(),
      output_semantic.data_ptr<float>(),
      output_depth.data_ptr<float>(),
      output_alpha.data_ptr<float>(),
      output_normal.data_ptr<float>(),
      grad_color.data_ptr<float>(),
      grad_semantic.data_ptr<float>(),
      grad_depth.data_ptr<float>(),
      grad_alpha.data_ptr<float>(),
      grad_normal.data_ptr<float>(),
      static_cast<int>(image_height),
      static_cast<int>(image_width),
      grad_projected_means.data_ptr<float>(),
      grad_conics.data_ptr<float>(),
      grad_projected_depths.data_ptr<float>(),
      grad_projected_normals.data_ptr<float>(),
      grad_colors.data_ptr<float>(),
      grad_semantics.data_ptr<float>(),
      grad_opacities.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (count > 0) {
    projection_backward_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
        count,
        means3d.data_ptr<float>(),
        means2d.data_ptr<float>(),
        scales.data_ptr<float>(),
        rotations.data_ptr<float>(),
        viewmatrix.data_ptr<float>(),
        radii.data_ptr<float>(),
        grad_projected_means.data_ptr<float>(),
        grad_conics.data_ptr<float>(),
        grad_projected_depths.data_ptr<float>(),
        grad_projected_normals.data_ptr<float>(),
        static_cast<int>(image_height),
        static_cast<int>(image_width),
        static_cast<float>(tanfovx),
        static_cast<float>(tanfovy),
        static_cast<float>(cx),
        static_cast<float>(cy),
        static_cast<float>(scale_modifier),
        static_cast<float>(antialias_sigma),
        grad_means3d.data_ptr<float>(),
        grad_means2d.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        grad_rotations.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {
      grad_means3d,
      grad_means2d,
      grad_colors,
      grad_semantics,
      grad_opacities,
      grad_scales,
      grad_rotations};
}
