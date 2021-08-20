cmake_minimum_required(VERSION 3.0.0)
project(cuda-ldpc)
include(CheckLanguage)
include(CTest)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11") set(CMAKE_CXX_FLAGS
# "${CMAKE_CXX_FLAGS} -std=c++11")
# set(CUDA_NVCC_FLAGS "-ccbin gcc-7")


enable_testing()

set(SOURCES main.cu LDPC_Decoder.cu LDPC_Encoder.cu Simulation.cu)

add_executable(${PROJECT_NAME} ${SOURCES})

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.0/bin/nvcc")
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
else()
  message(STATUS "No CUDA detected. GPU wont be used.")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
