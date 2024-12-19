#include <cuda_runtime.h>
#include <math.h>
#include "ops.cuh"
#ifndef COPYOUT
#define COPYOUT

#include <vector>

namespace manbo
{
	class CopyD2H : public ops
	{
		public:
			CopyD2H(float * out, int out_size, int N)//需要保证out_size是N的整数倍
			{
				this->out = out;
				this->out_size = out_size;
				this->N = N;
			}
			bool execable()
			{
				if(times * N >= out_size || in[0]->data == NULL)
					return false;
				return true;
			}
			void exec()
			{
				cudaMemcpy(out + times * N, in[0]->data, N * sizeof(float), cudaMemcpyDeviceToHost);
				times++;
			}
			float *out;//在主机上的输出
		private:
			int N, out_size;
			int times;
	};
}

#endif