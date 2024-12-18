#ifndef ADD
#define ADD

#include <vector>

namespace black_manbo
{
	class add : public ops
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
			int exec(float *in1, float *in2, int times)
			{
				for(int i = 0; i < times; i++)
				{
					for(auto in_op : start)
						cudaStreamWaitEvent(stream, *in_op, 0);
					add_kernel_launcher(in1, in2, out, N);
					cudaEventRecord(stops[0], stream);
				}
			}
			float *out;
		private:
			int N;
			__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int N);
			void add_kernel_launcher(float* d_a, float* d_b, float* d_c, int N);
	};
}

#endif