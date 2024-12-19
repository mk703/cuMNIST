#ifndef MATMUL
#define MATMUL

#include <vector>

namespace manbo
{
	class MatMul : public ops
	{
		//M * K, K * N, M * N
		public:
			MatMul(int M, int N, int K)
			{
				this->M = M;
				this->N = N;
				this->K = K;
			}
			void exec()
			{
				matmul_kernel_launcher(in[0]->data, in[1]->data, out[0]->data, M, N, K);
			}
		private:
			int M, N, K;
			__global__ void matmul_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int M, int N, int K);
			void matmul_kernel_launcher(float* d_a, float* d_b, float* d_c, int M, int N, int K);
	};
}

#endif
