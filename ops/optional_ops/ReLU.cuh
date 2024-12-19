#ifndef RELU
#define RELU

#include <vector>

namespace manbo
{
	class ReLU : public ops
	{
		public:
			ReLU(int N)
			{
				this->N = N;
			}
			void exec()
			{
				relu_kernel_launcher(in[0]->data, out[0]->data, N);
			}
		private:
			__global__ void relu_kernel(const float* __restrict__ a, float* b, int N);
			void relu_kernel_launcher(float* d_a, float* d_b, int N);
			int N;
			
	};
}

#endif