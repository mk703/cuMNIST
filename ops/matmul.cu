#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "ops.cuh"

//row y M, col x N

namespace black_manbo
{
	__global__ void matmul_kernel(float* a, float* b, float* c, int M, int N, int K)
    {
        extern __shared__ float shared_mem[];
		float* shared_a = shared_mem;
		float* shared_b = shared_mem + blockDim.y * blockDim.x;

		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		float sum = 0.0f;

		for (int i = 0; i < (K + blockDim.x - 1) / blockDim.x; i++)
		{
			if (row < M && i * blockDim.x + threadIdx.x < K)
				shared_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i * blockDim.x + threadIdx.x];
			else
				shared_a[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;

			if (col < N && i * blockDim.y + threadIdx.y < K)
				shared_b[threadIdx.y * blockDim.x + threadIdx.x] = b[(i * blockDim.y + threadIdx.y) * N + col];
			else
				shared_b[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;

			__syncthreads();

			for (int j = 0; j < blockDim.x; j++)
				sum += shared_a[threadIdx.y * blockDim.x + j] * shared_b[j * blockDim.x + threadIdx.x];

			__syncthreads();
		}

		if (row < M && col < N)
			c[row * N + col] = sum;
    }

    void matmul_kernel_launcher(float* d_a, float* d_b, float* d_c, int M, int N, int K)
    {
        dim3 blockDim(16, 16);  
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
        size_t shared_mem_size = (blockDim.y * blockDim.x * 2) * sizeof(float);
        matmul_kernel<<<gridDim, blockDim, shared_mem_size>>>(d_a, d_b, d_c, M, N, K);
        cudaDeviceSynchronize();
	}
}