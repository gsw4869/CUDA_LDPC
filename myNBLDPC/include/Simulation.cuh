#ifndef _SIMULATION_CUH_
#define _SIMULATION_CUH_

#include "define.cuh"
#include "struct.cuh"
#include "GF.cuh"

void Simulation_CPU(LDPCCode *H, AWGNChannel *AWGN, Simulation *SIM, CComplex *CONSTELLATION, VN *Variablenode, CN *Checknode, CComplex *CComplex_sym, int *CodeWord_sym, int *DecodeOutput);

void Simulation_GPU(LDPCCode *H, AWGNChannel *AWGN, Simulation *SIM, CComplex *CONSTELLATION, VN *Variablenode, CN *Checknode, CComplex *CComplex_sym, int *CodeWord_sym, int *DecodeOutput, unsigned *TableMultiply_GPU, unsigned *TableAdd_GPU, int *Checknode_weight, int *Variablenode_linkCNs, int *Checknode_linkVNs, int *Checknode_linkVNs_GF);

void WriteLogo(AWGNChannel *AWGN, Simulation *SIM);

int Statistic(Simulation *SIM, int *CodeWord_Frames, int *D, LDPCCode *LDPC);

CComplex *Get_CONSTELLATION(LDPCCode *H);

void Get_H(LDPCCode *H, VN *Variablenode, CN *Checknode);

void Transform_H(int *H, int *Weight_Checknode, int *Weight_Variablenode, int *Address_Variablenode);

#endif
