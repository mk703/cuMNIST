#include <cuda_runtime.h>
#include <math.h>

namespace black_manbo
{
	__global__ void softmax_kernel(float* a, float* b, int N);
	void softmax_kernel_launcher(float* d_a, float* d_b, int N);
	__global__ void relu_kernel(float* a, float* b, int N);
	void relu_kernel_launcher(float* d_a, float* d_b, int N);
	__global__ void add_kernel(float* a, float* b, float* c, int N);
	void add_kernel_launcher(float* d_a, float* d_b, float* d_c, int N);
	__global__ void matmul_kernel(float* a, float* b, float* c, int M, int N, int K);
	void matmul_kernel_launcher(float* d_a, float* d_b, float* d_c, int M, int N, int K);//M * K, K * N, M * N
	__global__ void conv2d_kernel(const float* d_input, float* d_output, const float* d_kernel, int width, int height, int kernelSize);
	void conv2d_kernel_launcher(const float* d_input, float* d_output, const float* d_kernel, int width, int height, int kernelSize);
}