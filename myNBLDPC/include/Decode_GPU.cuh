#ifndef _Decode_GPU_CUH_
#define _Decode_GPU_CUH_

#include "define.cuh"
#include "struct.cuh"
#include "LDPC_Decoder.cuh"
#include "float.h"

__device__ int GFAdd_GPU(int ele1, int ele2, const unsigned *TableAdd_GPU);
__device__ int GFMultiply_GPU(int ele1, int ele2, const unsigned *TableMultiply);
__device__ int GFInverse_GPU(int ele, const unsigned *TableInverse);

__device__ int ConstructConf_GPU(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, int Nm, int Nc, int &sumNonele, float &sumNonLLR, int &diff, int begin, int except, int end, int row, float *EMS_L_c2v, const int *Variblenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int *sort_Entr_v2c, float *sort_L_v2c);

__global__ void Checknode_EMS(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, int EMS_Nm, int EMS_Nc, const int *Checknode_weight, const int *Variblenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int *sort_Entr_v2c, float *sort_L_v2c, float *Checknode_L_c2v, int Checknode_num);

int Decoding_EMS_GPU(const LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput, const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const int *Checknode_weight, const int *Variablenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int &iter_number);

#endif
