#ifndef _LDPC_DECODER_CUH
#define _LDPC_DECODER_CUH

#include "define.cuh"
#include "struct.cuh"

void Demodulate(LDPCCode *H, AWGNChannel *AWGN, CComplex *CONSTELLATION, VN *Variablenode, CComplex *CComplex_sym_Channelout);

int ConstructConf(CN *Checknode, VN *Variablenode, int Nm, int Nc, int &sumNonele, float &sumNonLLR, int &diff, int begin, int except, int end, int row, float *EMS_L_c2v);

void LDPC_Decoder_GPU(int *D, float *Channel_Out, cudaDeviceProp prop, int *Address_Variablenode, int *Weight_Checknode, int *Weight_Variablenode, LDPCCode *LDPC);

int Decoding_EMS(LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput);

int index_in_VN(CN *Checknode, int CNnum, int index_in_linkVNS, VN *Variablenode);

int index_in_CN(VN *Variablenode, int VNnum, int index_in_linkCNS, CN *Checknode);

int SortLLRVector(int GF, float *Entr_v2c, int *index);

int DecideLLRVector(float *LLR, int GF);

#endif
