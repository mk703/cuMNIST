#include <cuda_runtime.h>
#include <stdexcept>

namespace manbo
{
	class ops
	{
		public:
			void attribute_in_tensor(tensor *in)
			{
				in->out_ops.push_back(this);
				this->in.push_back(in);
			}
			void attribute_out_tensor(tensor *out)
			{
				out->in_ops = this;
				this->out.push_back(out);
			}
			bool execable()
			{
				for(auto in_op : in)
					if(!in_op->read_ready())
						return false;
				for(auto out_op : out)
					if(!out_op->write_ready())
						return false;
				return true;
			}
			exec_pack pack()
			{
				if(!execable())
					throw std::runtime_error("Operation is not executable");
				for(auto out_tensor : out)
					out_tensor->init_data();
				exec_pack pack(this, in, out);
				return pack;
			}
			void init();
			void exec();
			std::vector<tensor *>in;
			std::vector<tensor *>out;
			cudaStream_t stream;
		protected:
	};
	class tensor
	{
		public:
			float *data = NULL;
			ops *in_ops;
			std::vector<ops *>out_ops;
			int size;
			int left_dpd;//依赖该张量的操作，但是还没有使用该张量的操作数
			bool is_const;//常量，被存下来的参数
			tensor(float *data, ops *in_ops, std::vector<ops *>out_ops, int size, bool is_const = false)
			{
				this->data = data;
				this->in_ops = in_ops;
				this->out_ops = out_ops;
				this->size = size;
				this->is_const = is_const;
			}
			bool read_ready()
			{
				return data != NULL;
			}
			bool write_ready()
			{
				return left_dpd == 0;
			}
			bool clear_data()
			{
				if(data != NULL)
				{
					cudaFree(data);
					data = NULL;
				}else
					return false;
				return true;
			}
			bool init_data()
			{
				if(data == NULL)
				{
					cudaMalloc(&data, size);
					left_dpd = out_ops.size();
				}else
					return false;
				return true;
			}
	};
	class exec_pack
	{
		public:
			exec_pack(ops *op, std::vector<tensor *> in, std::vector<tensor *> out)
			{
				this->op = op;
				this->in = in;
				this->out = out;
			}
			int exec()
			{
				/*
				1.执行
				2.检查是否可以释放输入张量
				*/
				op->exec();
				for(auto in_tensor : in)
				{
					if(in_tensor->is_const)
						continue;
					in_tensor->left_dpd--;
					if(in_tensor->write_ready())
						in_tensor->clear_data();
				}
			}
			friend class Scheduler;
		protected:
			ops *op;
			std::vector<tensor *> in;
			std::vector<tensor *> out;
	};
}