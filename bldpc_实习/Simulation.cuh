#include "define.cuh"
#include "struct.cuh"

void Simulation_GPU(AWGNChannel* AWGN, float* sigma_GPU, Simulation* SIM, int* Address_Variablenode, int* Weight_Checknode, int* Weight_Variablenode);

void WriteLogo(AWGNChannel* AWGN, Simulation* SIM);

int Statistic(Simulation* SIM, int* CodeWord_Frames, int* D, LDPCCode *LDPC);

void Get_H(int* H, int* Weight_Checknode, int* Weight_Variablenode);

void Transform_H(int* H, int* Weight_Checknode, int* Weight_Variablenode, int* Address_Variablenode);
