cmake_minimum_required(VERSION 3.18)
project(cuMNIST LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 CUDA
enable_language(CUDA)

# 设置编译选项
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 包含路径
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/ops
    /usr/include           # 添加 zlib 头文件路径
)

file(GLOB OPS_SOURCES ${CMAKE_SOURCE_DIR}/ops/*.cu)
# 源文件列表
set(SOURCES
    main.cpp
    ${OPS_SOURCES}
    utils/cnpy.cpp
)

# 查找 CUDA 包
find_package(CUDA REQUIRED)

# 查找 zlib 包
find_package(ZLIB REQUIRED)

# 添加可执行文件
add_executable(cuMNIST ${SOURCES})

# 链接 CUDA 库
target_link_libraries(cuMNIST ${CUDA_CUDART_LIBRARIES})

# 链接 zlib 库
target_link_libraries(cuMNIST ZLIB::ZLIB)