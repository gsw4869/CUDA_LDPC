#ifndef _LDPC_ENCODER_CUH
#define _LDPC_ENCODER_CUH

#include "define.cuh"
#include "struct.cuh"

void BitToSym(LDPCCode* H,int* CodeWord_sym,int* CodeWord_bit);

void Modulate(LDPCCode* H,CComplex* CONSTELLATION,CComplex* CComplex_sym,int* CodeWord_sym);

void AWGNChannel_CPU(LDPCCode* H, AWGNChannel* AWGN, CComplex* CComplex_sym_Channelout,CComplex* CComplex_sym);

float RandomModule(int* seed);

#endif