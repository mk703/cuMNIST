#include <vector>
#include <queue>
#include "ops.cuh"

namespace manbo
{
	class Scheduler
	{
		public:
			Scheduler();
			~Scheduler();
			void run();
			Graph *graph;
		private:
			
	};
	class Graph
	{
		public:

			bool add_op(ops op);
			bool add_tensor(ops in_op, std::vector<ops *> out_op, int size, bool is_const = false);
			bool add_tensor(float *data, std::vector<ops *> out_op, int size, bool is_const = true);
			bool free();
			std::vector<ops> op_set;
			std::vector<tensor> tensor_set;
	};
} // namespace manbo