#include <cuda_runtime.h>
#include <math.h>
#include "ops.cuh"

namespace black_manbo
{
	__global__ void conv2d_kernel(const float* d_input, float* d_output, const float* d_kernel, int width, int height, int kernelSize)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int half = kernelSize / 2;

		extern __shared__ float shared_memory[];
		float* shared_kernel = shared_memory;
		float* shared_input = &shared_memory[kernelSize * kernelSize];


		if(threadIdx.x < kernelSize && threadIdx.y < kernelSize)
			shared_kernel[threadIdx.y * kernelSize + threadIdx.x] = d_kernel[threadIdx.y * kernelSize + threadIdx.x];
		
		int shared_width = blockDim.x + kernelSize - 1;
		int shared_x = threadIdx.x + half;
		int shared_y = threadIdx.y + half;

		if (x < width && y < height) 
			shared_input[shared_y * shared_width + shared_x] = d_input[y * width + x];
		else
			shared_input[shared_y * shared_width + shared_x] = 0.0f;
		
		if (threadIdx.x < half)
		{
			int left_x = max(x - half, 0);
			shared_input[shared_y * shared_width + threadIdx.x] = d_input[y * width + left_x];
			int right_x = min(x + blockDim.x, width - 1);
			shared_input[shared_y * shared_width + shared_x + blockDim.x] = d_input[y * width + right_x];
		}
		if (threadIdx.y < half)
		{
			int top_y = max(y - half, 0);
			shared_input[threadIdx.y * shared_width + shared_x] = d_input[top_y * width + x];
			int bottom_y = min(y + blockDim.y, height - 1);
			shared_input[(shared_y + blockDim.y) * shared_width + shared_x] = d_input[bottom_y * width + x];
		}
		if(threadIdx.x < half && threadIdx.y < half)
		{
			shared_input[threadIdx.y * shared_width + threadIdx.x] = d_input[max(y - half, 0) * width + max(x - half, 0)];
			shared_input[threadIdx.y * shared_width + shared_x + blockDim.x] = d_input[max(y - half, 0) * width + min(x + blockDim.x, width - 1)];
			shared_input[(shared_y + blockDim.y) * shared_width + threadIdx.x] = d_input[min(y + blockDim.y, height - 1) * width + max(x - half, 0)];
			shared_input[(shared_y + blockDim.y) * shared_width + shared_x + blockDim.x] = d_input[min(y + blockDim.y, height - 1) * width + min(x + blockDim.x, width - 1)];
		}
		__syncthreads();

		if(x >= width || y >= height)
			return;
		float sum = 0.0f;
		for(int ky = -half; ky <= half; ky++)
			for(int kx = -half; kx <= half; kx++)
			{
				int shared_ix = shared_x + kx;
					int shared_iy = shared_y + ky;
					sum += shared_input[shared_iy * shared_width + shared_ix] * shared_kernel[(ky + half) * kernelSize + (kx + half)];
			}
		d_output[y * width + x] = sum;
	}

	void conv2d_kernel_launcher(const float* d_input, float* d_output, const float* d_kernel, int width, int height, int kernelSize)
	{
		dim3 block(16, 16);
		dim3 grid((width + 15) / 16, (height + 15) / 16);
		size_t shared_memory_size = (kernelSize * kernelSize + (width + kernelSize - 1) * (height + kernelSize - 1)) * sizeof(float);
		conv2d_kernel<<<grid, block, shared_memory_size>>>(
			d_input, d_output, d_kernel,
			width, height, kernelSize);
		cudaDeviceSynchronize();
	}
}