#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include "manbo.hpp"
#include "optional_ops.h"
#include "ops.cuh"
#include "utils/cnpy.h"

#define BATCH_SIZE 100
#define INPUT_SIZE 784
#define UP_PROJECTION_SIZE 3072
#define DOWN_PROJECTION_SIZE 128
#define OUTPUT_SIZE 10

int M[3] = {INPUT_SIZE, UP_PROJECTION_SIZE, DOWN_PROJECTION_SIZE};
int N[3] = {UP_PROJECTION_SIZE, DOWN_PROJECTION_SIZE, OUTPUT_SIZE};
int data_size[4] = {INPUT_SIZE, UP_PROJECTION_SIZE, DOWN_PROJECTION_SIZE, OUTPUT_SIZE};

void read(char* str_idx, float *x_test)
{
    const std::string filePath = "../data/x_test.bin";
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
        std::cerr << "无法打开文件: " << filePath << std::endl;

    std::vector<float> data;
    float buffer;
    while (file.read(reinterpret_cast<char*>(&buffer), sizeof(float))) 
        data.push_back(buffer);
    file.close();

    int idx = std::stoi(str_idx);
    for(int i = idx * INPUT_SIZE; i < (idx + BATCH_SIZE) * INPUT_SIZE; i++)
        x_test[i - idx * INPUT_SIZE] = data[i];
}

cnpy::NpyArray arr[3][2];

int main(int argc, char **argv)
{
    for(int i = 0; i < 3; i++)
    {
        arr[i][0] = cnpy::npy_load("../model/dense_bias_" + std::to_string(i) + ".npy");
        arr[i][1] = cnpy::npy_load("../model/dense_kernel_" + std::to_string(i) + ".npy");
        //cudaMemcpy(d_dense_kernel[i], arr.data<float>(), M[i] * N[i] * sizeof(float), cudaMemcpyHostToDevice);
    }

    float *x_test = new float[BATCH_SIZE * INPUT_SIZE];
    float *result = new float[BATCH_SIZE * OUTPUT_SIZE];
    read(argv[1], x_test);
    
    manbo::Graph graph;
    graph.add_op(manbo::CopyH2D(x_test, BATCH_SIZE * INPUT_SIZE, INPUT_SIZE));//0
    graph.add_op(manbo::MatMul(1, UP_PROJECTION_SIZE, INPUT_SIZE));
    graph.add_op(manbo::Add(UP_PROJECTION_SIZE));
    graph.add_op(manbo::ReLU(UP_PROJECTION_SIZE));//3
    graph.add_op(manbo::MatMul(1, DOWN_PROJECTION_SIZE, UP_PROJECTION_SIZE));
    graph.add_op(manbo::Add(DOWN_PROJECTION_SIZE));
    graph.add_op(manbo::MatMul(1, OUTPUT_SIZE, DOWN_PROJECTION_SIZE));//6
    graph.add_op(manbo::Add(OUTPUT_SIZE));
    graph.add_op(manbo::SoftMax(OUTPUT_SIZE, 1));
    graph.add_op(manbo::CopyD2H(result, BATCH_SIZE * OUTPUT_SIZE, OUTPUT_SIZE));//9
    //tensor(float *data,ops *in_ops, std::vector<ops *>out_ops, int size, bool is_const = false)
    graph.add_tensor(graph.op_set[0], {&graph.op_set[1]}, data_size[0]);
    graph.add_tensor(graph.op_set[1], {&graph.op_set[2]}, data_size[1]);
    graph.add_tensor(graph.op_set[2], {&graph.op_set[3]}, data_size[1]);
    graph.add_tensor(graph.op_set[3], {&graph.op_set[4]}, data_size[1]);
    graph.add_tensor(graph.op_set[4], {&graph.op_set[5]}, data_size[2]);
    graph.add_tensor(graph.op_set[5], {&graph.op_set[6]}, data_size[2]);
    graph.add_tensor(graph.op_set[6], {&graph.op_set[7]}, data_size[2]);
    graph.add_tensor(graph.op_set[7], {&graph.op_set[8]}, data_size[3]);
    graph.add_tensor(graph.op_set[8], {&graph.op_set[9]}, data_size[3]);

    graph.add_tensor(arr[0][0].data<float>(), {&graph.op_set[2]}, data_size[1], true);
    graph.add_tensor(arr[0][1].data<float>(), {&graph.op_set[1]}, M[0] * N[0], true);
    graph.add_tensor(arr[1][0].data<float>(), {&graph.op_set[5]}, data_size[2], true);
    graph.add_tensor(arr[1][1].data<float>(), {&graph.op_set[4]}, M[1] * N[1], true);
    graph.add_tensor(arr[2][0].data<float>(), {&graph.op_set[8]}, data_size[3], true);
    graph.add_tensor(arr[2][1].data<float>(), {&graph.op_set[7]}, M[2] * N[2], true);

    manbo::Scheduler scheduler;
    scheduler.graph = &graph;
    scheduler.run();

    for(int i = 0; i < BATCH_SIZE; i++)
    {
        int max_idx = 0;
        for(int j = 0; j < OUTPUT_SIZE; j++)
        {
            std::cout << result[i * OUTPUT_SIZE + j] << " ";
            if(result[i * OUTPUT_SIZE + j] > result[i * OUTPUT_SIZE + max_idx])
                max_idx = j;
        }
            
        std::cout << "Prediction: " << max_idx << std::endl;
    }
    return 0;
}
