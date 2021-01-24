cmake_minimum_required(VERSION 3.0.0)
project(cuda-ldpc VERSION 0.1.0)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11") set(CMAKE_CXX_FLAGS
# "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "-ccbin gcc-7")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.0/bin/nvcc")
enable_language(CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


include(CTest)
enable_testing()

set(SOURCES main.cu encoder.cu decoder.cu simu.cu)

add_executable(${PROJECT_NAME} ${SOURCES})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
