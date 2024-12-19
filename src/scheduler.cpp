#include <vector>
#include <queue>
#include "ops.cuh"
#include "manbo.hpp"

namespace manbo
{
	void Scheduler::run()
	{
		std::queue<exec_pack> q;
		for(auto op : graph->op_set)
			if(op.execable())
			{
				exec_pack pack = op.pack();
				q.push(pack);
			}
		while(!q.empty())
		{
			exec_pack pack = q.front();
			q.pop();
			pack.exec();
			for(auto out_op : (pack.op)->out)
				for(auto op : out_op->out_ops)
					if(op->execable())
					{
						exec_pack pack = op->pack();
						q.push(pack);
					}
		}
	}	
} // namespace manbo


