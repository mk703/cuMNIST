#ifndef SOFTMAX
#define SOFTMAX

#include <vector>

namespace manbo
{
	class SoftMax : public ops
	{
		public:
			SoftMax(int N, int M)//一共对N个数进行softmax，每次算M个
			{
				this->N = N;
				this->M = M;
			}
			void exec()
			{
				softmax_kernel_launcher(in[0]->data, out[0]->data, N, M);
			}
		private:
			int N, M;
			__global__ void softmax_kernel(float* __restrict__ a, float* __restrict__ b, float* max, int N);
			void softmax_kernel_launcher(float* d_a, float* d_b, int N, int M);
	};
}

#endif