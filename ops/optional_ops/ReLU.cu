#include <cuda_runtime.h>
#include <math.h>
#include "ops.cuh"
#include "optional_ops/ReLU.cuh"

namespace manbo
{
	__global__ void ReLU::relu_kernel(const float* __restrict__ a, float* __restrict__ b, int N)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx < N)
			b[idx] = fmaxf(0, a[idx]);
	}


	void ReLU::relu_kernel_launcher(float* d_a, float* d_b, int N)
	{
		relu_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, N);
		cudaDeviceSynchronize();
	}
}