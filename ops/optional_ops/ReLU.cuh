#ifndef RELU
#define RELU

#include <vector>

namespace black_manbo
{
	class relu : public ops
	{
		public:
			int init(std::vector<cudaEvent_t *> in_ops, int N)
			{
				/* 
				一个stream，一组event用来开始, 计算所需张量
				 */
				this->N = N;
				cudaStreamCreate(&stream);
				for(auto in_op : in_ops)
					start.push_back(in_op);
				cudaMalloc((void **)&out, N * sizeof(float));
				return 0;
			}
			int exec(float *in1, int times)
			{
				for(int i = 0; i < times; i++)
				{
					for(auto in_op : start)
						cudaStreamWaitEvent(stream, *in_op, 0);
					relu_kernel_launcher(in1, out, N);
					cudaEventRecord(stops[0], stream);
				}
			}
			float *out;
		private:
			__global__ void relu_kernel(const float* __restrict__ a, float* b, int N);
			void relu_kernel_launcher(float* d_a, float* d_b, int N);
			int N;
			
	};
}

namespace black_manbo
{
	
}

#endif