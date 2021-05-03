#ifndef _SIMULATION_CUH_
#define _SIMULATION_CUH_

#include "define.h"
#include "struct.h"
#include "GF.h"

void Simulation_CPU(const LDPCCode *H, AWGNChannel *AWGN, Simulation *SIM, const CComplex *CONSTELLATION, VN *Variablenode, CN *Checknode, const CComplex *CComplex_sym, const int *CodeWord_sym);

void Simulation_GPU(const LDPCCode *H, AWGNChannel *AWGN, Simulation *SIM, const CComplex *CONSTELLATION, VN *Variablenode, CN *Checknode, const CComplex *CComplex_sym, int *CodeWord_sym, const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const int *Variablenode_weight, const int *Checknode_weight, const int *Variablenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF);

void WriteLogo(AWGNChannel *AWGN, Simulation *SIM);

int Statistic(Simulation *SIM, const int *CodeWord_Frames, int *D, const LDPCCode *LDPC);

CComplex *Get_CONSTELLATION(LDPCCode *H);

void Get_H(LDPCCode *H, VN *Variablenode, CN *Checknode);

void Transform_H(int *H, int *Weight_Checknode, int *Weight_Variablenode, int *Address_Variablenode);

#endif
