#ifndef SOFTMAX
#define SOFTMAX

#include <vector>

namespace manbo
{
	class softmax : public ops
	{
		public:
			void init(int N, int M)//执行M次，每次对N个数进行softmax
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
			__global__ void softmax_kernel(float* __restrict__ a, float* __restrict__ b, int N);
			void softmax_kernel_launcher(float* d_a, float* d_b, int N, int M);
	};
}

#endif