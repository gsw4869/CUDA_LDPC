#ifndef _Decode_GPU_CUH_
#define _Decode_GPU_CUH_

#include "define.cuh"
#include "struct.cuh"
#include "LDPC_Decoder.cuh"
#include "float.h"

__device__ int GFAdd_GPU(int ele1, int ele2, unsigned *TableAdd_GPU);
__device__ int GFMultiply_GPU(int ele1, int ele2, unsigned *TableMultiply);
__device__ int GFInverse_GPU(int ele, unsigned *TableInverse);

__device__ int ConstructConf_GPU(unsigned *TabelMultiply_GPU, unsigned *TableAdd_GPU, int Nm, int Nc, int &sumNonele, float &sumNonLLR, int &diff, int begin, int except, int end, int row, float *EMS_L_c2v, int *Variblenode_linkCNs, int *Checknode_linkVNs, int *Checknode_linkVNs_GF, int *sort_Entr_v2c, int *sort_L_v2c);

#endif
