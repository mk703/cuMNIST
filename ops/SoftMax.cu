#include <cuda_runtime.h>
#include <math.h>
#include "ops.cuh"

namespace black_manbo
{
	__global__ void softmax_kernel(float* a, float* b, int N)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= N) return;
		float max = a[idx];
		for(int i = 0; i < N; i++)
			if(a[i] > max)
				max = a[i];
		float sum = 0;
		for(int i = 0; i < N; i++)
		{
			b[i] = expf(a[i] - max);
			sum += b[i];
		}
		for(int i = 0; i < N; i++)
			b[i] /= sum;
	}

	void softmax_kernel_launcher(float* d_a, float* d_b, int N)
	{
		softmax_kernel<<<(N + 255) / 256, 256>>>(d_a, d_b, N);
		cudaDeviceSynchronize();
	}
}