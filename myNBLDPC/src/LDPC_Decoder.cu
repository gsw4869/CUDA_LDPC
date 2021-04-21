#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <conio.h>
#include <string.h>
#include <memory.h>
#include <time.h>
//#include <direct.h>
#include "define.cuh"
#include "struct.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LDPC_Decoder.cuh"
#include "GF.cuh"
#include "float.h"

void BubleSort(float a[], int n, int index[])
{
	int i, j;
	float x;
	for (i = 0; i < n; i++)
	{
		for (j = 1; j < n - i; j++)
		{
			if (a[j - 1] < a[j])
			{
				x = a[j];
				a[j] = a[j - 1];
				a[j - 1] = x;
				x = index[j];
				index[j] = index[j - 1];
				index[j - 1] = x;
			}
		}
	}
}

int SortLLRVector(int GF, float *Entr_v2c, int *index)
{
	BubleSort(Entr_v2c, GF, index);
	return 1;
}

int DecideLLRVector(float *LLR, int GF)
{
	float max = 0;
	int alpha_i;
	for (int q = 0; q < GF - 1; q++)
	{
		if (LLR[q] > max)
		{
			max = LLR[q];
			alpha_i = q + 1;
		}
	}
	if (max <= 0)
	{
		return 0;
	}
	else
	{
		return alpha_i;
	}
}

int index_in_VN(CN *Checknode, int CNnum, int index_in_linkVNS, VN *Variablenode)
{
	for (int i = 0; i < Variablenode[Checknode[CNnum].linkVNs[index_in_linkVNS]].weight; i++)
	{
		if (Variablenode[Checknode[CNnum].linkVNs[index_in_linkVNS]].linkCNs[i] == CNnum)
		{
			return i;
		}
	}
	printf("index_in_VN error\n");
	exit(0);
}

int index_in_CN(VN *Variablenode, int VNnum, int index_in_linkCNS, CN *Checknode)
{
	for (int i = 0; i < Checknode[Variablenode[VNnum].linkCNs[index_in_linkCNS]].weight; i++)
	{
		if (Checknode[Variablenode[VNnum].linkCNs[index_in_linkCNS]].linkVNs[i] == VNnum)
		{
			return i;
		}
	}
	printf("index_in_CN error\n");
	exit(0);
}

void Demodulate(LDPCCode *H, AWGNChannel *AWGN, CComplex *CONSTELLATION, VN *Variablenode, CComplex *CComplex_sym_Channelout)
{
	float *RX_LLR_BIT = (float *)malloc(H->bit_length * sizeof(float));
	if (n_QAM == 2)
	{
		// RX_MOD_SYM --> RX_LLR_BIT --> RX_LLR_SYM
		// only support bpsk now
		int p_i = 0;
		for (int b = 0; b < H->bit_length; b++)
		{

			RX_LLR_BIT[b] = -2 * CComplex_sym_Channelout[b - p_i].Real / (AWGN->sigma * AWGN->sigma);
		}
		// RX_LLE_BIT --> RX_LLR_SYM
		for (int s = 0; s < H->Variablenode_num; s++)
		{
			for (int q = 1; q < H->GF; q++)
			{
				Variablenode[s].L_ch[q - 1] = 0;
				for (int b_p_s = 0; b_p_s < H->q_bit; b_p_s++)
				{
					if ((q & (1 << b_p_s)) != 0)
					{
						Variablenode[s].L_ch[q - 1] += RX_LLR_BIT[s * H->q_bit + b_p_s];
					}
				}
			}
		}
	}
	else
	{
		for (int s = 0; s < H->Variablenode_num; s++)
		{
			for (int q = 1; q < H->GF; q++)
			{
				Variablenode[s].L_ch[q - 1] = ((2 * CComplex_sym_Channelout[s].Real - CONSTELLATION[0].Real - CONSTELLATION[q].Real) * (CONSTELLATION[q].Real - CONSTELLATION[0].Real) + (2 * CComplex_sym_Channelout[s].Image - CONSTELLATION[0].Image - CONSTELLATION[q].Image) * (CONSTELLATION[q].Image - CONSTELLATION[0].Image)) / (2 * AWGN->sigma * AWGN->sigma);
			}
		}
	}
}
int Decoding_EMS(LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput)
{
	for (int col = 0; col < H->Variablenode_num; col++)
	{
		for (int d = 0; d < Variablenode[col].weight; d++)
		{
			for (int q = 0; q < H->GF; q++)
			{
				Variablenode[col].Entr_v2c[d][q] = Variablenode[col].L_ch[q];
			}
		}
	}
	for (int row = 0; row < H->Checknode_num; row++)
	{
		for (int d = 0; d < Checknode[row].weight; d++)
		{
			for (int q = 0; q < H->GF - 1; q++)
			{
				Checknode[row].L_c2v[d][q] = 0;
			}
		}
	}

	int iter_number = 0;
	bool decode_correct = true;
	while (iter_number++ < maxIT)
	{
		// printf("it_time: %d\n",iter_number);
		for (int col = 0; col < H->Variablenode_num; col++)
		{
			for (int d = 0; d < Variablenode[col].weight; d++)
			{
				for (int q = 0; q < H->GF - 1; q++)
				{
					Variablenode[col].LLR[q] = Variablenode[col].L_ch[q];
				}
			}
		}
		for (int col = 0; col < H->Variablenode_num; col++)
		{
			for (int d = 0; d < Variablenode[col].weight; d++)
			{
				for (int q = 0; q < H->GF - 1; q++)
				{
					Variablenode[col].LLR[q] += Checknode[Variablenode[col].linkCNs[d]].L_c2v[index_in_CN(Variablenode, col, d, Checknode)][q];
				}
			}
			DecodeOutput[col] = DecideLLRVector(Variablenode[col].LLR, H->GF);
			// printf("%d ",DecodeOutput[col]);
		}
		// printf("\n");
		// exit(0);

		decode_correct = true;
		int sum_temp = 0;
		for (int row = 0; row < H->Checknode_num; row++)
		{
			for (int i = 0; i < Checknode[row].weight; i++)
			{
				sum_temp = GFAdd(sum_temp, GFMultiply(DecodeOutput[Checknode[row].linkVNs[i]], Checknode[row].linkVNs_GF[i]));
			}
			if (sum_temp)
			{
				decode_correct = false;
				break;
			}
		}
		if (decode_correct)
		{
			return 1;
		}

		// message from var to check
		for (int col = 0; col < H->Variablenode_num; col++)
		{
			for (int dv = 0; dv < Variablenode[col].weight; dv++)
			{
				for (int q = 0; q < H->GF - 1; q++)
				{
					Variablenode[col].Entr_v2c[dv][q] = Variablenode[col].LLR[q] - Checknode[Variablenode[col].linkCNs[dv]].L_c2v[index_in_CN(Variablenode, col, dv, Checknode)][q];
				}
			}
		}

		int *index = (int *)malloc((H->GF) * sizeof(int));
		for (int col = 0; col < H->Variablenode_num; col++)
		{
			memcpy(Variablenode[col].sort_L_v2c[0], Variablenode[col].Entr_v2c[0], Variablenode[col].weight * H->GF * sizeof(float));

			for (int dv = 0; dv < Variablenode[col].weight; dv++)
			{
				for (int i = 0; i < H->GF - 1; i++)
				{
					index[i] = i + 1;
				}
				index[H->GF - 1] = 0;
				SortLLRVector(H->GF, Variablenode[col].sort_L_v2c[dv], index);
				for (int i = 0; i < H->GF; i++)
				{
					Variablenode[col].sort_Entr_v2c[dv][i] = index[i];
				}
			}
		}

		float *EMS_L_c2v = (float *)malloc(H->GF * sizeof(float));

		// message from check to var
		for (int row = 0; row < H->Checknode_num; row++)
		{

			for (int dc = 0; dc < Checknode[row].weight; dc++)
			{
				// reset the sum store vector to the munimum
				for (int q = 0; q < H->GF; q++)
				{
					EMS_L_c2v[q] = -DBL_MAX;
				}

				// recursly exhaustly
				int sumNonele, diff;
				float sumNonLLR;
				// conf(q, 1)
				sumNonele = 0;
				sumNonLLR = 0;
				diff = 0;
				ConstructConf(Checknode, Variablenode, H->GF, 1, sumNonele, sumNonLLR, diff, 0, dc, Checknode[row].weight - 1, row, EMS_L_c2v);

				// conf(nm, nc)
				sumNonele = 0;
				sumNonLLR = 0;
				diff = 0;
				ConstructConf(Checknode, Variablenode, EMS_Nm, EMS_Nc, sumNonele, sumNonLLR, diff, 0, dc, Checknode[row].weight - 1, row, EMS_L_c2v);

				// calculate each c2v LLR
				int v = 0;
				for (int k = 1; k < H->GF; k++)
				{
					v = GFMultiply(k, Checknode[row].linkVNs_GF[dc]);
					Checknode[row].L_c2v[dc][k - 1] = (EMS_L_c2v[v] - EMS_L_c2v[0]) / 1.2;
				}
			}
		}
		free(EMS_L_c2v);
	}
	return 0;
}

int ConstructConf(CN *Checknode, VN *Variablenode, int Nm, int Nc, int &sumNonele, float &sumNonLLR, int &diff, int begin, int except, int end, int row, float *EMS_L_c2v)
{
	int index;
	if (begin > end)
	{
		if (sumNonLLR > EMS_L_c2v[sumNonele])
		{
			EMS_L_c2v[sumNonele] = sumNonLLR;
		}
	}
	else if (begin == except)
	{
		ConstructConf(Checknode, Variablenode, Nm, Nc, sumNonele, sumNonLLR, diff, begin + 1, except, end, row, EMS_L_c2v);
		return 0;
	}
	else
	{
		index = index_in_VN(Checknode, row, begin, Variablenode);
		for (int k = 0; k < Nm; k++)
		{
			sumNonele = GFAdd(GFMultiply(Variablenode[Checknode[row].linkVNs[begin]].sort_Entr_v2c[index][k], Checknode[row].linkVNs_GF[begin]), sumNonele);
			sumNonLLR = sumNonLLR + Variablenode[Checknode[row].linkVNs[begin]].sort_L_v2c[index][k];
			diff += (k != 0) ? 1 : 0;
			if (diff <= Nc)
			{
				ConstructConf(Checknode, Variablenode, Nm, Nc, sumNonele, sumNonLLR, diff, begin + 1, except, end, row, EMS_L_c2v);
				sumNonele = GFAdd(GFMultiply(Variablenode[Checknode[row].linkVNs[begin]].sort_Entr_v2c[index][k], Checknode[row].linkVNs_GF[begin]), sumNonele);
				sumNonLLR = sumNonLLR - Variablenode[Checknode[row].linkVNs[begin]].sort_L_v2c[index][k];
				diff -= (k != 0) ? 1 : 0;
			}
			else
			{
				sumNonele = GFAdd(GFMultiply(Variablenode[Checknode[row].linkVNs[begin]].sort_Entr_v2c[index][k], Checknode[row].linkVNs_GF[begin]), sumNonele);
				sumNonLLR = sumNonLLR - Variablenode[Checknode[row].linkVNs[begin]].sort_L_v2c[index][k];
				diff -= (k != 0) ? 1 : 0;
				break;
			}
		}
	}
	return 0;
}
