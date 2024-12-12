#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include "ops.cuh"
#include "utils/cnpy.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BATCH_SIZE 1
#define INPUT_SIZE 784
#define UP_PROJECTION_SIZE 3072
#define DOWN_PROJECTION_SIZE 128
#define OUTPUT_SIZE 10

int M[3] = {INPUT_SIZE, UP_PROJECTION_SIZE, DOWN_PROJECTION_SIZE};
int N[3] = {UP_PROJECTION_SIZE, DOWN_PROJECTION_SIZE, OUTPUT_SIZE};
int data_size[4] = {BATCH_SIZE * INPUT_SIZE, BATCH_SIZE * UP_PROJECTION_SIZE, BATCH_SIZE * DOWN_PROJECTION_SIZE, BATCH_SIZE * OUTPUT_SIZE};

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

int main(int argc, char **argv)
{
    float *d_dense_bias[3], *d_dense_kernel[3], *data[4], *tmp, *tmp2;

    for(int i = 0; i < 3; i++)
    {
        cudaMalloc((void **)&d_dense_bias[i], N[i] * sizeof(float));
        cudaMalloc((void **)&d_dense_kernel[i], M[i] * N[i] * sizeof(float));
        cnpy::NpyArray arr = cnpy::npy_load("../model/dense_bias_" + std::to_string(i) + ".npy");
        cudaMemcpy(d_dense_bias[i], arr.data<float>(), N[i] * sizeof(float), cudaMemcpyHostToDevice);
        arr = cnpy::npy_load("../model/dense_kernel_" + std::to_string(i) + ".npy");
        cudaMemcpy(d_dense_kernel[i], arr.data<float>(), M[i] * N[i] * sizeof(float), cudaMemcpyHostToDevice);
    }
    for(int i = 0; i < 4;i++)
        cudaMalloc((void **)&data[i], data_size[i] * sizeof(float));

    float *x_test = new float[BATCH_SIZE * INPUT_SIZE];
    std::cout << argv[1] << std:: endl;
    read(argv[1], x_test);
    cudaMemcpy(data[0], x_test, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    for(int i = 0;i < 28;i++)
    {
        for(int j = 0;j < 28;j++)
            std::cout << ((x_test[i * 28 + j] > 0.5) ? 1 : 0) << " ";
        std::cout << std::endl;
    }
    
    black_manbo::matmul_kernel_launcher(data[0], d_dense_kernel[0], data[1], BATCH_SIZE, UP_PROJECTION_SIZE, INPUT_SIZE);
    black_manbo::add_kernel_launcher(data[1], d_dense_bias[0], data[1], BATCH_SIZE * UP_PROJECTION_SIZE);
    black_manbo::relu_kernel_launcher(data[1], data[1], BATCH_SIZE * UP_PROJECTION_SIZE);
    black_manbo::matmul_kernel_launcher(data[1], d_dense_kernel[1], data[2], BATCH_SIZE, DOWN_PROJECTION_SIZE, UP_PROJECTION_SIZE);
    black_manbo::add_kernel_launcher(data[2], d_dense_bias[1], data[2], BATCH_SIZE * DOWN_PROJECTION_SIZE);
    black_manbo::matmul_kernel_launcher(data[2], d_dense_kernel[2], data[3], BATCH_SIZE, OUTPUT_SIZE, DOWN_PROJECTION_SIZE);
    black_manbo::add_kernel_launcher(data[3], d_dense_bias[2], data[3], BATCH_SIZE * OUTPUT_SIZE);
    black_manbo::softmax_kernel_launcher(data[3], data[3], BATCH_SIZE * OUTPUT_SIZE);

    float * result = new float[BATCH_SIZE * OUTPUT_SIZE];
    cudaMemcpy(result, data[3], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

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


    for(int i = 0; i < 3; i++)
    {
        cudaFree(d_dense_bias[i]);
        cudaFree(d_dense_kernel[i]);
    }
    for(int i = 0; i < 4; i++)
        cudaFree(data[i]);
    return 0;
}
