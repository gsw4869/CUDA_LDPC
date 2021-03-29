#include "define.cuh"
#include "struct.cuh"

void Simulation_GPU(Simulation* SIM, VN* Variablenode, CN* Checknode, float* Channel_Out);

void WriteLogo(AWGNChannel* AWGN, Simulation* SIM);

int Statistic(Simulation* SIM, int* CodeWord_Frames, int* D, LDPCCode *LDPC);

void Get_H(LDPCCode* H,VN* Variablenode,CN* Checknode);

void Transform_H(int* H, int* Weight_Checknode, int* Weight_Variablenode, int* Address_Variablenode);
