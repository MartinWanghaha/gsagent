/*
 * Semantic Prior Field auxiliary compositor.
 *
 * The implementation reuses the exact Gaussian tile ranges, ordering and
 * conic/opacity values produced by the Gaussian Wrapping rasterizer.  Semantic
 * gradients intentionally update only the per-Gaussian embeddings; in
 * addition, the backward pass accumulates two per-Gaussian statistics that
 * drive the Semantic Prior Field density control:
 *
 *   stat_abs_grad[i]     = sum_p  alpha_i(p) T_i(p) * || dL/dE(p) ||_2
 *   stat_contribution[i] = sum_p  alpha_i(p) T_i(p)
 *
 * The unsigned accumulation cannot cancel across pixels, so the ratio of
 * stat_abs_grad to the norm of the signed embedding gradient measures how
 * conflicting the semantic supervision of a Gaussian is: boundary-straddling
 * Gaussians receive opposing per-pixel gradients that cancel in the signed
 * sum but not in the unsigned one.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

namespace SEMANTIC
{
	constexpr int NUM_SEMANTIC_CHANNELS = 16;

	void forward(
		const dim3 grid,
		const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int width,
		int height,
		const float2* means2D,
		const float4* conic_opacity,
		const float* semantic_features,
		float* out_semantic);

	void backward(
		const dim3 grid,
		const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int width,
		int height,
		const float2* means2D,
		const float4* conic_opacity,
		const float* grad_out_semantic,
		float* grad_semantic_features,
		float* stat_abs_grad,
		float* stat_contribution);
}
