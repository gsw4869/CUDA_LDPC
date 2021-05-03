#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LDPC_Encoder.h"
#include "struct.h"

void BitToSym(LDPCCode *H, int *CodeWord_sym, int *CodeWord_bit)
{
	for (int s = 0; s < H->Variablenode_num; s++)
	{
		CodeWord_sym[s] = 0;
		for (int i = 0; i < H->q_bit; i++)
		{
			CodeWord_sym[s] = 2 * CodeWord_sym[s] + CodeWord_bit[H->q_bit * s + H->q_bit - 1 - i];
		}
	}
}

void Modulate(const LDPCCode *H, CComplex *CONSTELLATION, CComplex *CComplex_sym, int *CodeWord_sym)
{
	if (n_QAM != 2)
	{
		for (int s = 0; s < H->Variablenode_num; s++)
		{
			CComplex_sym[s].Real = CONSTELLATION[CodeWord_sym[s]].Real;
			CComplex_sym[s].Image = CONSTELLATION[CodeWord_sym[s]].Image;
		}
	}
	else
	{
		for (int s = 0; s < H->bit_length; s++)
		{
			CComplex_sym[s].Real = CONSTELLATION[CodeWord_sym[s]].Real;
			CComplex_sym[s].Image = CONSTELLATION[CodeWord_sym[s]].Image;
		}
	}
}
/*
* CodeWord：原始码组
* Channel_Out：经过BPSK调制的输出信号
*/
void AWGNChannel_CPU(const LDPCCode *H, AWGNChannel *AWGN, CComplex *CComplex_sym_Channelout, const CComplex *CComplex_sym)
{
	int index0, len;
	float u1, u2, temp;
	if (n_QAM != 2)
	{
		len = H->Variablenode_num;
	}
	else
	{
		len = H->bit_length;
	}
	for (index0 = 0; index0 < len; index0++)
	{

		u1 = RandomModule(AWGN->seed);
		u2 = RandomModule(AWGN->seed);

		temp = (float)sqrt((float)(-2) * log((float)1 - u1));
		CComplex_sym_Channelout[index0].Real = (AWGN->sigma) * cos(2 * PI * u2) * temp + CComplex_sym[index0].Real; //产生高斯白噪声信号(https://www.cnblogs.com/tsingke/p/6194737.html)

		u1 = RandomModule(AWGN->seed);
		u2 = RandomModule(AWGN->seed);

		temp = (float)sqrt((float)(-2) * log((float)1 - u1));
		CComplex_sym_Channelout[index0].Image = (AWGN->sigma) * cos(2 * PI * u2) * temp + CComplex_sym[index0].Image;
	}
}

float RandomModule(int *seed)
{
	float temp = 0.0;
	seed[0] = (seed[0] * 249) % 61967;
	seed[1] = (seed[1] * 251) % 63443;
	seed[2] = (seed[2] * 252) % 63599;
	temp = (((float)seed[0]) / ((float)61967)) + (((float)seed[1]) / ((float)63443)) + (((float)seed[2]) / ((float)63599));
	temp -= (int)temp;
	return (temp);
}
