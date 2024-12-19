#include <vector>
#include <queue>
#include "ops.cuh"
#include "manbo.hpp"

namespace manbo
{
	bool is_pointer_on_gpu(const void* ptr)
	{
		cudaPointerAttributes attributes;
		cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
		if (err != cudaSuccess)
			// 如果指针无效或其他错误，返回false
			return false;
		return attributes.type == cudaMemoryTypeDevice;
	}
	bool Graph::add_op(ops op)
	{
		op_set.push_back(op);
		return true;
	}
	bool Graph::add_tensor(ops in_op, std::vector<ops *> out_op, int size, bool is_const = false)
	{
		tensor t(NULL, &in_op, out_op, size, is_const);
		tensor_set.push_back(t);
		in_op.attribute_in_tensor(&t);
		for(auto op : out_op)
			op->attribute_out_tensor(&t);
		return true;
	}
	bool Graph::add_tensor(float *data, std::vector<ops *> out_op, int size, bool is_const = true)
	{
		//判断data是否在GPU上，如果不在，需要cudaMalloc，cudaMemcpy
		float* d_data = data;
		if (!is_pointer_on_gpu(data))
		{
			// 如果data不在GPU上，分配GPU内存并复制数据
			cudaMalloc(&d_data, size * sizeof(float));
			cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
		}
		tensor t(d_data, NULL, out_op, size, is_const);
		tensor_set.push_back(t);
		for(auto op : out_op)
			op->attribute_out_tensor(&t);
		return true;
	}
	bool Graph::free()
	{
		return true;
	}
} // namespace manbo
