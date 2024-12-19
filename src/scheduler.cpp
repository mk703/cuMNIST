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
			for(auto out_tensor : (pack.op)->out)
				for(auto op : out_tensor->out_ops)
					if(op->execable())
					{
						exec_pack pack = op->pack();
						q.push(pack);
					}
			//额外检查copyin是否可以执行
			if(graph->op_set[0].execable())
			{
				exec_pack pack = graph->op_set[0].pack();
				q.push(pack);
			}
		}
	}	
}


