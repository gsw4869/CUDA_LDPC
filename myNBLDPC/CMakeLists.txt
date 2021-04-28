cmake_minimum_required(VERSION 3.10.0)
project(cuda-ldpc)
include(CheckLanguage)
include(CTest)

set(SOURCES src/main.cu src/LDPC_Decoder.cu src/LDPC_Encoder.cu src/Simulation.cu src/GF.cu src/Decode_GPU.cu src/struct.cu)


add_executable(${PROJECT_NAME} ${SOURCES})

# set(CUDA_NVCC_FLAGS "-ccbin gcc-7") set(CMAKE_CUDA_COMPILER
# "/usr/local/cuda-11.0/bin/nvcc")
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
else()
  message(STATUS "No CUDA detected. GPU wont be used.")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# pthread conflict with OPENMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

target_include_directories(${PROJECT_NAME}
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


