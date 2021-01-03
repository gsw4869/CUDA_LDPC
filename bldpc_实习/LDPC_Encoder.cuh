#include "define.cuh"
#include "struct.cuh"

__global__ void BPSK(float* BPSK_Out, int* CodeWord);

void AWGNChannel_CPU(AWGNChannel* AWGN, float* Channel_Out, int* CodeWord);

float RandomModule(int* seed);
