#include <cuda_runtime.h>
#include <math.h>
#include "ops.cuh"
#include "optional_ops/SoftMax.cuh"

namespace manbo
{
	__global__ void SoftMax::softmax_kernel(float* __restrict__ a, float* __restrict__ b, float* max, int N)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= N) return;
		//用原子操作更新max
		atomicMax(max, a[idx]);
		__syncthreads();

		for(int i = 0; i < N; i++)
			if(a[i] > &max)
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

	void SoftMax::softmax_kernel_launcher(float* d_a, float* d_b, int N, int M)
	{
		for(int i = 0;i < N / M; i ++)
			softmax_kernel<<<(M + 255) / 256, 256>>>(d_a + i * M, d_b + i * M, M);
		cudaDeviceSynchronize();
	}
}