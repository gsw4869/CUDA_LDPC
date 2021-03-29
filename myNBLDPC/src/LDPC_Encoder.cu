#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LDPC_Encoder.cuh"
#include "struct.cuh"


/*
* CodeWord：原始码组
* Channel_Out：经过BPSK调制的输出信号
*/
void AWGNChannel_CPU(LDPCCode* H, AWGNChannel* AWGN, float* Channel_Out,int* CodeWord)
{
	int index0;
	float u1, u2, temp;
	for (index0 = 0; index0 < H->length; index0++)
	{
		u1 = RandomModule(AWGN->seed);
		u2 = RandomModule(AWGN->seed);

		temp = (float)sqrt((float)(-2) * log((float)1 - u1));
		*(Channel_Out + index0) = (AWGN->sigma) * sin(2 * PI * u2) * temp + 1.0 - 2 * (*(CodeWord + index0));//产生高斯白噪声信号(https://www.cnblogs.com/tsingke/p/6194737.html)
	}
}



float RandomModule(int* seed)
{
	float temp = 0.0;
	seed[0] = (seed[0] * 249) % 61967;
	seed[1] = (seed[1] * 251) % 63443;
	seed[2] = (seed[2] * 252) % 63599;
	temp = (((float)seed[0]) / ((float)61967)) + (((float)seed[1]) / ((float)63443))
		+ (((float)seed[2]) / ((float)63599));
	temp -= (int)temp;
	return (temp);
}
