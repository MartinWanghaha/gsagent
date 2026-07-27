/*
 * Gaussian Wrapping Gaga semantic auxiliary compositor.
 *
 * The implementation reuses the exact Gaussian tile ranges, ordering and
 * conic/opacity values produced by the Gaussian Wrapping rasterizer.  Semantic
 * gradients intentionally update only the per-Gaussian embeddings.
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
		float* grad_semantic_features);
}
