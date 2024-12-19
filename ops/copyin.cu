#include <cuda_runtime.h>
#include <math.h>
#include "ops.cuh"
#ifndef COPYIN
#define COPYIN

#include <vector>

namespace manbo
{
	class CopyH2D : public ops
	{
		public:
			CopyH2D(float * in, int in_size, int N)//需要保证in_size是N的整数倍
			{
				this->in = in;
				this->in_size = in_size;
				this->N = N;
			}
			bool execable()
			{
				if(times * N >= in_size || out[0]->data == NULL)
					return false;
				return true;
			}
			void exec()
			{
				cudaMemcpy(out[0]->data, in + times * N, N * sizeof(float), cudaMemcpyHostToDevice);
				times++;
			}
			float *in;
		private:
			int N, in_size;
			int times;
	};
}

#endif