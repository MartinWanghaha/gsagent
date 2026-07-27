/*
 * Gaussian Wrapping Gaga semantic auxiliary compositor.
 *
 * Copyright (C) 2023, Inria, GRAPHDECO research group.
 * Gaga-inspired semantic extension Copyright (C) 2026.
 */

#include "semantic.h"
#include "config.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace
{
constexpr int SEMANTIC_BLOCK_SIZE = BLOCK_X * BLOCK_Y;

template<int CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
semanticForwardCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int width,
	int height,
	const float2* __restrict__ means2D,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ semantic_features,
	float* __restrict__ out_semantic)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix = {
		block.group_index().x * BLOCK_X + block.thread_index().x,
		block.group_index().y * BLOCK_Y + block.thread_index().y
	};
	const bool inside = pix.x < width && pix.y < height;
	bool done = !inside;
	const uint32_t pix_id = width * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = (range.y - range.x + SEMANTIC_BLOCK_SIZE - 1) / SEMANTIC_BLOCK_SIZE;
	int to_do = range.y - range.x;

	__shared__ int collected_id[SEMANTIC_BLOCK_SIZE];
	__shared__ float2 collected_xy[SEMANTIC_BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[SEMANTIC_BLOCK_SIZE];
	__shared__ float collected_semantic[CHANNELS * SEMANTIC_BLOCK_SIZE];

	float transmittance = 1.0f;
	float semantic[CHANNELS] = {0.0f};

	for (int round = 0; round < rounds; ++round, to_do -= SEMANTIC_BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == SEMANTIC_BLOCK_SIZE)
			break;

		const int progress = round * SEMANTIC_BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int gaussian_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = gaussian_id;
			collected_xy[block.thread_rank()] = means2D[gaussian_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[gaussian_id];
			for (int channel = 0; channel < CHANNELS; ++channel)
				collected_semantic[channel * SEMANTIC_BLOCK_SIZE + block.thread_rank()] =
					semantic_features[gaussian_id * CHANNELS + channel];
		}
		block.sync();

		for (int j = 0; !done && j < min(SEMANTIC_BLOCK_SIZE, to_do); ++j)
		{
			const float2 delta = {
				collected_xy[j].x - pixf.x,
				collected_xy[j].y - pixf.y
			};
			const float4 conic = collected_conic_opacity[j];
			const float power = -0.5f * (
				conic.x * delta.x * delta.x
				+ conic.z * delta.y * delta.y
			) - conic.y * delta.x * delta.y;
			if (power > 0.0f)
				continue;

			const float alpha = min(0.99f, conic.w * expf(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			const float next_transmittance = transmittance * (1.0f - alpha);
			if (next_transmittance < 0.0001f)
			{
				done = true;
				continue;
			}

			const float weight = alpha * transmittance;
			for (int channel = 0; channel < CHANNELS; ++channel)
				semantic[channel] += collected_semantic[channel * SEMANTIC_BLOCK_SIZE + j] * weight;
			transmittance = next_transmittance;
		}
	}

	if (inside)
		for (int channel = 0; channel < CHANNELS; ++channel)
			out_semantic[channel * height * width + pix_id] = semantic[channel];
}

template<int CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
semanticBackwardCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int width,
	int height,
	const float2* __restrict__ means2D,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ grad_out_semantic,
	float* __restrict__ grad_semantic_features)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix = {
		block.group_index().x * BLOCK_X + block.thread_index().x,
		block.group_index().y * BLOCK_Y + block.thread_index().y
	};
	const bool inside = pix.x < width && pix.y < height;
	bool done = !inside;
	const uint32_t pix_id = width * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = (range.y - range.x + SEMANTIC_BLOCK_SIZE - 1) / SEMANTIC_BLOCK_SIZE;
	int to_do = range.y - range.x;

	__shared__ int collected_id[SEMANTIC_BLOCK_SIZE];
	__shared__ float2 collected_xy[SEMANTIC_BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[SEMANTIC_BLOCK_SIZE];

	float transmittance = 1.0f;
	float pixel_gradient[CHANNELS] = {0.0f};
	if (inside)
		for (int channel = 0; channel < CHANNELS; ++channel)
			pixel_gradient[channel] =
				grad_out_semantic[channel * height * width + pix_id];

	for (int round = 0; round < rounds; ++round, to_do -= SEMANTIC_BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == SEMANTIC_BLOCK_SIZE)
			break;

		const int progress = round * SEMANTIC_BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int gaussian_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = gaussian_id;
			collected_xy[block.thread_rank()] = means2D[gaussian_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[gaussian_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(SEMANTIC_BLOCK_SIZE, to_do); ++j)
		{
			const float2 delta = {
				collected_xy[j].x - pixf.x,
				collected_xy[j].y - pixf.y
			};
			const float4 conic = collected_conic_opacity[j];
			const float power = -0.5f * (
				conic.x * delta.x * delta.x
				+ conic.z * delta.y * delta.y
			) - conic.y * delta.x * delta.y;
			if (power > 0.0f)
				continue;

			const float alpha = min(0.99f, conic.w * expf(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			const float next_transmittance = transmittance * (1.0f - alpha);
			if (next_transmittance < 0.0001f)
			{
				done = true;
				continue;
			}

			const float weight = alpha * transmittance;
			const int gaussian_id = collected_id[j];
			for (int channel = 0; channel < CHANNELS; ++channel)
				atomicAdd(
					&grad_semantic_features[gaussian_id * CHANNELS + channel],
					weight * pixel_gradient[channel]);
			transmittance = next_transmittance;
		}
	}
}
}

void SEMANTIC::forward(
	const dim3 grid,
	const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* means2D,
	const float4* conic_opacity,
	const float* semantic_features,
	float* out_semantic)
{
	semanticForwardCUDA<NUM_SEMANTIC_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		width,
		height,
		means2D,
		conic_opacity,
		semantic_features,
		out_semantic);
}

void SEMANTIC::backward(
	const dim3 grid,
	const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int width,
	int height,
	const float2* means2D,
	const float4* conic_opacity,
	const float* grad_out_semantic,
	float* grad_semantic_features)
{
	semanticBackwardCUDA<NUM_SEMANTIC_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		width,
		height,
		means2D,
		conic_opacity,
		grad_out_semantic,
		grad_semantic_features);
}
