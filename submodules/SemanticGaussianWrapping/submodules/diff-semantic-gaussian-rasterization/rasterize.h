/* Copyright (C) 2026, Semantic Gaussian Wrapping contributors. */
#pragma once

#include <torch/extension.h>

#include <vector>

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
    double antialias_sigma);

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
    double antialias_sigma);

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
    double antialias_sigma);

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
    double cy);
