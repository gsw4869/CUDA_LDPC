#ifndef _Decode_GPU_CUH_
#define _Decode_GPU_CUH_

#include "define.h"
#include "struct.h"
#include "LDPC_Decoder.h"
#include "float.h"

__device__ int GFAdd_GPU(int ele1, int ele2, const unsigned *TableAdd_GPU);
__device__ int GFMultiply_GPU(int ele1, int ele2, const unsigned *TableMultiply);
__device__ int GFInverse_GPU(int ele, const unsigned *TableInverse);

__global__ void Checknode_EMS(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, int EMS_Nm, int EMS_Nc, const int *Checknode_weight, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int *sort_Entr_v2c, float *sort_L_v2c, float *Checknode_L_c2v, int Checknode_num);

__global__ void Variablenode_EMS(const int *Variablenode_weight, const int *Variablenode_linkCNs, int *sort_Entr_v2c, float *sort_L_v2c, float *Checknode_L_c2v, const float *L_ch, float *LLR, int *DecodeOutput, int Variablenode_num);

int Decoding_EMS_GPU(const LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput, const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const int *Variablenode_weight, const int *Checknode_weight, const int *Variablenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int &iter_number);

int Decoding_TMM_GPU(const LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput, const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const unsigned *TableInverse_GPU, const int *Variablenode_weight, const int *Checknode_weight, const int *Variablenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int &iter_number);

__global__ void Variablenode_Update_EMS(const int *Variablenode_weight, const int *Variablenode_linkCNs, int *sort_Entr_v2c, float *sort_L_v2c, float *Checknode_L_c2v, const float *L_ch, float *LLR, int Variablenode_num);

__global__ void Variablenode_TMM(const int *Variablenode_weight, const int *Variablenode_linkCNs, float *sort_L_v2c, float *Checknode_L_c2v, float *LLR, int *DecodeOutput, int Variablenode_num);

__global__ void Variablenode_Update_TMM(const int *Variablenode_weight, const int *Variablenode_linkCNs, float *sort_L_v2c, float *Checknode_L_c2v, float *LLR, int Variablenode_num);

__global__ void Checknode_TMM(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const unsigned *TableInverse_GPU, const int *Checknode_weight, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, float *sort_L_v2c, float *Checknode_L_c2v, int Checknode_num);

__device__ int d_TMM_Get_Zn_GPU(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const unsigned *TableInverse_GPU, const int *Checknode_weight, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, float *sort_L_v2c, float *Checknode_L_c2v, int *TMM_Zn, int row, int &TMM_Syndrome);

__device__ int d_TMM_Get_deltaU_GPU(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const unsigned *TableInverse_GPU, const int *Checknode_weight, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, float *sort_L_v2c, float *Checknode_L_c2v, int *TMM_Zn, float *TMM_deltaU, int row);

__device__ int TMM_Get_Min_GPU(const int *Checknode_weight, int *TMM_Zn, float *TMM_deltaU, float *TMM_Min1, float *TMM_Min2, int *TMM_Min1_Col, int row);

__device__ int TMM_ConstructConf_GPU(const unsigned *TableAdd_GPU, float *TMM_deltaU, float *TMM_Min1, float *TMM_Min2, int *TMM_Min1_Col, float *TMM_I, int *TMM_Path, float *TMM_E);

#endif
