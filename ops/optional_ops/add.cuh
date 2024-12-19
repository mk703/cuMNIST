#ifndef ADD
#define ADD

#include <vector>

namespace manbo
{
	class Add : public ops
	{
		public:
			Add(int N)
			{
				this->N = N;
			}
			void exec()
			{
				add_kernel_launcher(in[0]->data, in[1]->data, out[0]->data, N);
			}
		private:
			int N;
			__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int N);
			void add_kernel_launcher(float* d_a, float* d_b, float* d_c, int N);
	};
}

#endif