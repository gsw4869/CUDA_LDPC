#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LDPC_Encoder.cuh"
#include "struct.cuh"

/*
* CodeWord：原始码组
* BPSK_Out：经过BPSK调制的输出信号
*/
__global__ void BPSK(float* BPSK_Out, int* CodeWord)
{
	int offset;
	offset = threadIdx.x + blockDim.x * blockIdx.x;

	if (offset < Num_Frames_OneTime * CW_Len)
	{
		*(BPSK_Out + offset) = 1.0 - 2*(*(CodeWord + offset));
	}
}

/*
* CodeWord：原始码组
* Channel_Out：经过BPSK调制的输出信号
*/
void AWGNChannel_CPU(AWGNChannel* AWGN, float* Channel_Out,int* CodeWord)
{
	int index0, index1;
	float u1, u2, temp;

	for (index0 = 0; index0 < Num_Frames_OneTime; index0++)
	{
		for (index1 = 0; index1 < CW_Len; index1++)
		{
			u1 = RandomModule(AWGN->seed);
			u2 = RandomModule(AWGN->seed);

			temp = (float)sqrt((float)(-2) * log((float)1 - u1));
			*(Channel_Out + index1 * Num_Frames_OneTime + index0) = (AWGN->sigma) * sin(2 * PI * u2) * temp + 1.0 - 2 * (*(CodeWord + index1 * Num_Frames_OneTime + index0));//产生高斯白噪声信号(https://www.cnblogs.com/tsingke/p/6194737.html)
		}
	}
}

/*

*/
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
