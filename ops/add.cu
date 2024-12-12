#include <cuda_runtime.h>
#include <math.h>
#include "ops.cuh"

namespace black_manbo
{
	__global__ void add_kernel(float* a, float* b, float* c, int N)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx < N)
			c[idx] = a[idx] + b[idx];
	}

	void add_kernel_launcher(float* d_a, float* d_b, float* d_c, int N)
	{
		add_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
		cudaDeviceSynchronize();
	}
}