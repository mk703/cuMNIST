#include <cuda_runtime.h>
#include <math.h>
#include "ops.cuh"
#ifndef COPYOUT
#define COPYOUT

#include <vector>

namespace manbo
{
	class copyout : public ops
	{
		public:
			void init(float * out, int N)
			{
				this->in = in;
				this->N = N;
			}
			void exec()
			{
				cudaMemcpyAsync(out[0]->data, in + times * N, N * sizeof(float), cudaMemcpyHostToDevice, stream);
				times++;
			}
			float *in;
		private:
			int N;
			int times;
	};
}

#endif