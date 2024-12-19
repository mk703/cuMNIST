#include <cuda_runtime.h>
#include <math.h>
#include "ops.cuh"
#include "optional_ops/add.cuh"

namespace manbo
{
	__global__ void Add::add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int N)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx < N)
			c[idx] = a[idx] + b[idx];
	}

	void Add::add_kernel_launcher(float* d_a, float* d_b, float* d_c, int N)
	{
		add_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
		cudaDeviceSynchronize();//等待所有线程结束
	}
}