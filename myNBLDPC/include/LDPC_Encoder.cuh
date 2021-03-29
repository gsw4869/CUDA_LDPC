#include "define.cuh"
#include "struct.cuh"

void AWGNChannel_CPU(LDPCCode* H, AWGNChannel* AWGN, float* Channel_Out,int* CodeWord);

float RandomModule(int* seed);
