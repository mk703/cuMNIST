#ifndef SOFTMAX
#define SOFTMAX

#include <vector>

namespace black_manbo
{
	class softmax : public ops
	{
		public:
			int init(std::vector<cudaEvent_t *> in_ops, int N)
			{
				/* 
				一个stream，一组event, 计算所需张量
				 */
				this->N = N;
				cudaStreamCreate(&stream);
				for(auto in_op : in_ops)
					start.push_back(in_op);
				cudaMalloc((void **)&out, N * sizeof(float));
				return 0;
			}
			int exec(float *in, int times)
			{
				for(int i = 0; i < times; i++)
				{
					for(auto in_op : start)
						cudaStreamWaitEvent(stream, *in_op, 0);
					softmax_kernel(in, out, N);
					cudaEventRecord(stops[0], stream);
				}
			}
			float *out;
		private:
			int N;
			__global__ void softmax_kernel(float* __restrict__ a, float* __restrict__ b, int N);
			void softmax_kernel_launcher(float* d_a, float* d_b, int N, int M);
	};
}

#endif