#include "define.cuh"
#include "struct.cuh"


void LDPC_Decoder_GPU(int* D, float* Channel_Out, cudaDeviceProp prop, int* Address_Variablenode, int* Weight_Checknode, int* Weight_Variablenode, LDPCCode *LDPC);

__global__ void Variablenode_Kernel(float* Memory_RQ, int* D, float* Channel_Out, int* Address_Variablenode, int* Weight_Variablenode);

__global__ void Variablenode_Shared_Kernel(float* Memory_RQ, int* D, float* Channel_Out, int* Address_Variablenode, int* Weight_Variablenode);

__global__ void Checknode_Kernel(float* Memory_RQ,int* Weight_Checknode);

__global__ void Checknode_Shared_Kernel(float* Memory_RQ, int* Weight_Checknode);

__device__ void sortQ(float* MinQ, float* SubMinQ, float* Q, int Weight);