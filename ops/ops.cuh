#include <cuda_runtime.h>
#include "add.cuh"
#include "matmul.cuh"
#include "SoftMax.cuh"
#include "conv2d.cuh"
#include "ReLU.cuh"

namespace black_manbo
{
	class ops
	{
		public:
			int free()
			{
				cudaStreamDestroy(stream);
				cudaFree(stream);
				for(auto stop : stops)
					cudaEventDestroy(stop);
				for(auto stop : stops)
					cudaFree(stop);
				stops.clear();
				return 0;
			}
			std::vector<cudaEvent_t *>start;
			std::vector<cudaEvent_t>stops;
			cudaStream_t stream;
		protected:
	};
}