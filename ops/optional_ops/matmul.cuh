#ifndef MATMUL
#define MATMUL

#include <vector>

namespace black_manbo
{
	class matmul : public ops
	{
		//M * K, K * N, M * N
		public:
			int init(std::vector<cudaEvent_t *> in_ops, int M, int N, int K)
			{
				/* 
				一个stream，一组event, 计算所需张量
				 */
				this->M = M;
				this->N = N;
				this->K = K;
				cudaStreamCreate(&stream);
				for(auto in_op : in_ops)
					start.push_back(in_op);
				cudaMalloc((void **)&out, M * N * sizeof(float));
				return 0;
			}
			int exec(float *in1, float *in2, int times)
			{
				for(int i = 0; i < times; i++)
				{
					for(auto in_op : start)
						cudaStreamWaitEvent(stream, *in_op, 0);
					matmul_kernel_launcher(in1, in2, out, M , N, K);
					cudaEventRecord(stops[0], stream);
				}
			}
			float *out;
		private:
			int M, N, K;
			__global__ void matmul_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int M, int N, int K);
			void matmul_kernel_launcher(float* d_a, float* d_b, float* d_c, int M, int N, int K);
	};
}

#endif
