#include <cuda_runtime.h>
#include <math.h>
#include "ops.cuh"
#ifndef COPYIN
#define COPYIN

#include <vector>

namespace manbo
{
	class copyin : public ops
	{
		public:
			void init(float * in, int in_size, int N)
			{
				this->in = in;
				this->in_size = in_size;
				this->N = N;
			}
			void exec()
			{
				cudaMemcpyAsync(out[0]->data, in + times * N, N * sizeof(float), cudaMemcpyHostToDevice, stream);
				times++;
			}
			float *in;
		private:
			int N, in_size;
			int times;
	};
}

#endif