#ifndef CONV2D
#define CONV2D

namespace black_manbo
{
	__global__ void conv2d_kernel(const float* d_input, float* d_output, const float* d_kernel, int width, int height, int kernelSize);
	void conv2d_kernel_launcher(const float* d_input, float* d_output, const float* d_kernel, int width, int height, int kernelSize);
}

#endif