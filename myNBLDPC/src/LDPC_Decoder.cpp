#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <conio.h>
#include <string.h>
#include <memory.h>
#include <time.h>
//#include <direct.h>
#include "define.h"
#include "struct.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LDPC_Decoder.h"
#include "GF.h"
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

void d_BubleSort(float a[], int n, int index[])
{
	int i, j;
	float x;
	for (i = 0; i < n; i++)
	{
		for (j = 1; j < n - i; j++)
		{
			if (a[j - 1] > a[j])
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

int d_SortLLRVector(int GF, float *Entr_v2c, int *index)
{
	d_BubleSort(Entr_v2c, GF, index);
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
int d_DecideLLRVector(float *LLR, int GF)
{
	float min = DBL_MAX;
	int alpha_i;
	for (int q = 0; q < GFQ; q++)
	{
		if (LLR[q] < min)
		{
			min = LLR[q];
			alpha_i = q;
		}
	}
	return alpha_i;
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

void Demodulate(const LDPCCode *H, AWGNChannel *AWGN, const CComplex *CONSTELLATION, VN *Variablenode, CComplex *CComplex_sym_Channelout)
{
	float *RX_LLR_BIT = (float *)malloc(H->bit_length * sizeof(float));
	if (n_QAM == 2)
	{
		// RX_MOD_SYM --> RX_LLR_BIT --> RX_LLR_SYM
		// only support bpsk now
		for (int b = 0; b < H->bit_length; b++)
		{

			RX_LLR_BIT[b] = -2 * CComplex_sym_Channelout[b].Real / (AWGN->sigma * AWGN->sigma);
		}
		// RX_LLE_BIT --> RX_LLR_SYM
		for (int s = 0; s < H->Variablenode_num; s++)
		{
			for (int q = 1; q < GFQ; q++)
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
			for (int q = 1; q < GFQ; q++)
			{
				Variablenode[s].L_ch[q - 1] = ((2 * CComplex_sym_Channelout[s].Real - CONSTELLATION[0].Real - CONSTELLATION[q].Real) * (CONSTELLATION[q].Real - CONSTELLATION[0].Real) + (2 * CComplex_sym_Channelout[s].Image - CONSTELLATION[0].Image - CONSTELLATION[q].Image) * (CONSTELLATION[q].Image - CONSTELLATION[0].Image)) / (2 * AWGN->sigma * AWGN->sigma);
			}
		}
	}
	free(RX_LLR_BIT);
}
int Decoding_EMS(const LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput, int &iter_number)
{
	for (int col = 0; col < H->Variablenode_num; col++)
	{
		for (int d = 0; d < Variablenode[col].weight; d++)
		{
			for (int q = 0; q < GFQ; q++)
			{
				Variablenode[col].sort_L_v2c[d][q] = Variablenode[col].L_ch[q];
			}
		}
	}
	for (int row = 0; row < H->Checknode_num; row++)
	{
		for (int d = 0; d < Checknode[row].weight; d++)
		{
			for (int q = 0; q < GFQ; q++)
			{
				Checknode[row].L_c2v[d][q] = 0;
			}
		}
	}
	float *EMS_L_c2v = (float *)malloc(GFQ * sizeof(float));
	int *index = (int *)malloc((GFQ) * sizeof(int));

	iter_number = 0;
	bool decode_correct = true;
	while (iter_number < maxIT)
	{
		iter_number++;
		for (int col = 0; col < H->Variablenode_num; col++)
		{
			memcpy(Variablenode[col].LLR, Variablenode[col].L_ch, (GFQ - 1) * sizeof(float));
		}
		for (int col = 0; col < H->Variablenode_num; col++)
		{
			for (int d = 0; d < Variablenode[col].weight; d++)
			{
				for (int q = 0; q < GFQ - 1; q++)
				{
					Variablenode[col].LLR[q] += Checknode[Variablenode[col].linkCNs[d]].L_c2v[index_in_CN(Variablenode, col, d, Checknode)][q];
				}
			}
			DecodeOutput[col] = DecideLLRVector(Variablenode[col].LLR, GFQ);
		}

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
			free(EMS_L_c2v);
			free(index);
			iter_number--;
			return 1;
		}

		// message from var to check
		for (int col = 0; col < H->Variablenode_num; col++)
		{
			for (int dv = 0; dv < Variablenode[col].weight; dv++)
			{
				for (int q = 0; q < GFQ - 1; q++)
				{
					Variablenode[col].sort_L_v2c[dv][q] = Variablenode[col].LLR[q] - Checknode[Variablenode[col].linkCNs[dv]].L_c2v[index_in_CN(Variablenode, col, dv, Checknode)][q];
				}
				Variablenode[col].sort_L_v2c[dv][GFQ - 1] = 0;
			}
		}

		for (int col = 0; col < H->Variablenode_num; col++)
		{

			for (int dv = 0; dv < Variablenode[col].weight; dv++)
			{
				for (int i = 0; i < GFQ - 1; i++)
				{
					index[i] = i + 1;
				}
				index[GFQ - 1] = 0;
				SortLLRVector(GFQ, Variablenode[col].sort_L_v2c[dv], index);
				for (int i = 0; i < GFQ; i++)
				{
					Variablenode[col].sort_Entr_v2c[dv][i] = index[i];
				}
			}
		}

		// message from check to var
		for (int row = 0; row < H->Checknode_num; row++)
		{

			for (int dc = 0; dc < Checknode[row].weight; dc++)
			{
				// reset the sum store vector to the munimum
				for (int q = 0; q < GFQ; q++)
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
				ConstructConf(Checknode, Variablenode, GFQ, 1, sumNonele, sumNonLLR, diff, 0, dc, Checknode[row].weight - 1, row, EMS_L_c2v);

				// conf(nm, nc)
				sumNonele = 0;
				sumNonLLR = 0;
				diff = 0;
				if (EMS_Nc == maxdc - 1)
				{
					ConstructConf(Checknode, Variablenode, EMS_Nm, Checknode[row].weight - 1, sumNonele, sumNonLLR, diff, 0, dc, Checknode[row].weight - 1, row, EMS_L_c2v);
				}
				else
				{
					ConstructConf(Checknode, Variablenode, EMS_Nm, EMS_Nc, sumNonele, sumNonLLR, diff, 0, dc, Checknode[row].weight - 1, row, EMS_L_c2v);
				}
				// calculate each c2v LLR
				int v = 0;
				for (int k = 1; k < GFQ; k++)
				{
					v = GFMultiply(k, Checknode[row].linkVNs_GF[dc]);
					Checknode[row].L_c2v[dc][k - 1] = (EMS_L_c2v[v] - EMS_L_c2v[0]) / 1.2;
				}
			}
		}
	}
	free(EMS_L_c2v);
	free(index);
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

int Decoding_TMM(const LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput, int &iter_number)
{
	float max = -DBL_MAX;
	for (int col = 0; col < H->Variablenode_num; col++)
	{
		max = -DBL_MAX;
		for (int q = 0; q < GFQ - 1; q++)
		{
			if (Variablenode[col].L_ch[q] > max)
			{
				max = Variablenode[col].L_ch[q];
			}
		}
		for (int d = 0; d < Variablenode[col].weight; d++)
		{
			for (int q = 0; q < GFQ; q++)
			{
				if (q == 0)
				{
					Variablenode[col].sort_L_v2c[d][q] = max;
					Variablenode[col].LLR[q] = max;
				}
				else
				{
					Variablenode[col].sort_L_v2c[d][q] = max - Variablenode[col].L_ch[q - 1];
					Variablenode[col].LLR[q] = max - Variablenode[col].L_ch[q - 1];
				}
			}
		}
	}

	for (int row = 0; row < H->Checknode_num; row++)
	{
		for (int d = 0; d < Checknode[row].weight; d++)
		{
			for (int q = 0; q < GFQ; q++)
			{
				Checknode[row].L_c2v[d][q] = 0;
			}
		}
	}

	int *TMM_Zn = (int *)malloc(maxdc * sizeof(int));
	float *TMM_deltaU = (float *)malloc(maxdc * GFQ * sizeof(float));
	float *TMM_Min1 = (float *)malloc(GFQ * sizeof(float));
	float *TMM_Min2 = (float *)malloc(GFQ * sizeof(float));
	int *TMM_Min1_Col = (int *)malloc(GFQ * sizeof(int));
	float *TMM_I = (float *)malloc(GFQ * sizeof(float));
	int *TMM_Path = (int *)malloc(GFQ * 2 * sizeof(int));
	float *TMM_E = (float *)malloc(GFQ * sizeof(float));
	float *TMM_Lc2p = (float *)malloc(GFQ * sizeof(float));
	int TMM_Syndrome = 0;

	iter_number = 0;
	bool decode_correct = true;

	while (iter_number < maxIT)
	{
		iter_number++;
		// for (int col = 0; col < H->Variablenode_num; col++)
		// {
		// 	DecodeOutput[col] = d_DecideLLRVector(Variablenode[col].LLR, GFQ);
		// }

		for (int col = 0; col < H->Variablenode_num; col++)
		{
			for (int d = 0; d < Variablenode[col].weight; d++)
			{
				for (int q = 0; q < GFQ; q++)
				{
					Variablenode[col].LLR[q] += Checknode[Variablenode[col].linkCNs[d]].L_c2v[index_in_CN(Variablenode, col, d, Checknode)][q];
				}
			}
			DecodeOutput[col] = d_DecideLLRVector(Variablenode[col].LLR, GFQ);
		}

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
			free(TMM_Zn);
			free(TMM_deltaU);
			free(TMM_Min1);
			free(TMM_Min2);
			free(TMM_Min1_Col);
			free(TMM_I);
			free(TMM_Path);
			free(TMM_E);
			free(TMM_Lc2p);
			iter_number--;
			return 1;
		}

		for (int col = 0; col < H->Variablenode_num; col++)
		{
			for (int dv = 0; dv < Variablenode[col].weight; dv++)
			{
				for (int q = 0; q < GFQ; q++)
				{
					Variablenode[col].sort_L_v2c[dv][q] = Variablenode[col].LLR[q] - Checknode[Variablenode[col].linkCNs[dv]].L_c2v[index_in_CN(Variablenode, col, dv, Checknode)][q];
				}
			}
		}

		for (int row = 0; row < H->Checknode_num; row++)
		{
			TMM_Syndrome = 0;
			// for (int d = 0; d < Checknode[row].weight; d++)
			// {
			// 	for (int q = 0; q < GFQ; q++)
			// 	{
			// 		Variablenode[Checknode[row].linkVNs[d]].sort_L_v2c[index_in_VN(Checknode, row, d, Variablenode)][q] = Variablenode[Checknode[row].linkVNs[d]].LLR[q] - Checknode[row].L_c2v[d][q];
			// 	}
			// }

			d_TMM_Get_Zn(Checknode, Variablenode, TMM_Zn, row, TMM_Syndrome);

			d_TMM_Get_deltaU(Checknode, Variablenode, TMM_Zn, TMM_deltaU, row);

			TMM_Get_Min(Checknode, TMM_Zn, TMM_deltaU, TMM_Min1, TMM_Min2, TMM_Min1_Col, row);

			TMM_ConstructConf(TMM_deltaU, TMM_Min1, TMM_Min2, TMM_Min1_Col, TMM_I, TMM_Path, TMM_E);

			for (int dc = 0; dc < Checknode[row].weight; dc++)
			{
				// choose to output
				TMM_Lc2p[0] = 0;
				for (int eta = 1; eta < GFQ; eta++)
				{
					if (dc != TMM_Path[eta * 2 + 0] && dc != TMM_Path[eta * 2 + 1])
					{
						TMM_Lc2p[eta] = TMM_I[eta];
					}
					else
					{
						TMM_Lc2p[eta] = TMM_E[eta];
					}
				}

				int h_inverse = GFInverse(Checknode[row].linkVNs_GF[dc]);
				int beta_syn = GFAdd(TMM_Syndrome, TMM_Zn[dc]);
				double L0 = TMM_Lc2p[beta_syn];
				for (int eta = 0; eta < GFQ; eta++)
				{
					int beta =
						GFMultiply(h_inverse, GFAdd(eta, beta_syn));
					Checknode[row].L_c2v[dc][beta] = (TMM_Lc2p[eta]) * 0.8;
				}
			}

			// for (int d = 0; d < Checknode[row].weight; d++)
			// {
			// 	for (int q = 0; q < GFQ; q++)
			// 	{
			// 		Variablenode[Checknode[row].linkVNs[d]].LLR[q] = Variablenode[Checknode[row].linkVNs[d]].sort_L_v2c[index_in_VN(Checknode, row, d, Variablenode)][q] + Checknode[row].L_c2v[d][q];
			// 	}
			// }
		}
	}
	free(TMM_Zn);
	free(TMM_deltaU);
	free(TMM_Min1);
	free(TMM_Min2);
	free(TMM_Min1_Col);
	free(TMM_I);
	free(TMM_Path);
	free(TMM_E);
	free(TMM_Lc2p);
	return 0;
}

int Decoding_layered_TMM(const LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput, int &iter_number)
{
	float max = -DBL_MAX;
	for (int col = 0; col < H->Variablenode_num; col++)
	{
		max = -DBL_MAX;
		for (int q = 0; q < GFQ - 1; q++)
		{
			if (Variablenode[col].L_ch[q] > max)
			{
				max = Variablenode[col].L_ch[q];
			}
		}
		for (int d = 0; d < Variablenode[col].weight; d++)
		{
			for (int q = 0; q < GFQ; q++)
			{
				if (q == 0)
				{
					Variablenode[col].sort_L_v2c[d][q] = max;
					Variablenode[col].LLR[q] = max;
				}
				else
				{
					Variablenode[col].sort_L_v2c[d][q] = max - Variablenode[col].L_ch[q - 1];
					Variablenode[col].LLR[q] = max - Variablenode[col].L_ch[q - 1];
				}
			}
		}
	}

	for (int row = 0; row < H->Checknode_num; row++)
	{
		for (int d = 0; d < Checknode[row].weight; d++)
		{
			for (int q = 0; q < GFQ; q++)
			{
				Checknode[row].L_c2v[d][q] = 0;
			}
		}
	}

	int *TMM_Zn = (int *)malloc(maxdc * sizeof(int));
	float *TMM_deltaU = (float *)malloc(maxdc * GFQ * sizeof(float));
	float *TMM_Min1 = (float *)malloc(GFQ * sizeof(float));
	float *TMM_Min2 = (float *)malloc(GFQ * sizeof(float));
	int *TMM_Min1_Col = (int *)malloc(GFQ * sizeof(int));
	float *TMM_I = (float *)malloc(GFQ * sizeof(float));
	int *TMM_Path = (int *)malloc(GFQ * 2 * sizeof(int));
	float *TMM_E = (float *)malloc(GFQ * sizeof(float));
	float *TMM_Lc2p = (float *)malloc(GFQ * sizeof(float));
	int TMM_Syndrome = 0;

	iter_number = 0;
	bool decode_correct = true;

	while (iter_number < maxIT)
	{
		iter_number++;
		for (int col = 0; col < H->Variablenode_num; col++)
		{
			DecodeOutput[col] = d_DecideLLRVector(Variablenode[col].LLR, GFQ);
		}

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
			free(TMM_Zn);
			free(TMM_deltaU);
			free(TMM_Min1);
			free(TMM_Min2);
			free(TMM_Min1_Col);
			free(TMM_I);
			free(TMM_Path);
			free(TMM_E);
			free(TMM_Lc2p);
			iter_number--;
			return 1;
		}

		for (int row = 0; row < H->Checknode_num; row++)
		{
			TMM_Syndrome = 0;
			for (int d = 0; d < Checknode[row].weight; d++)
			{
				for (int q = 0; q < GFQ; q++)
				{
					Variablenode[Checknode[row].linkVNs[d]].sort_L_v2c[index_in_VN(Checknode, row, d, Variablenode)][q] = Variablenode[Checknode[row].linkVNs[d]].LLR[q] - Checknode[row].L_c2v[d][q];
				}
			}

			d_TMM_Get_Zn(Checknode, Variablenode, TMM_Zn, row, TMM_Syndrome);

			d_TMM_Get_deltaU(Checknode, Variablenode, TMM_Zn, TMM_deltaU, row);

			TMM_Get_Min(Checknode, TMM_Zn, TMM_deltaU, TMM_Min1, TMM_Min2, TMM_Min1_Col, row);

			TMM_ConstructConf(TMM_deltaU, TMM_Min1, TMM_Min2, TMM_Min1_Col, TMM_I, TMM_Path, TMM_E);

			for (int dc = 0; dc < Checknode[row].weight; dc++)
			{
				// choose to output
				TMM_Lc2p[0] = 0;
				for (int eta = 1; eta < GFQ; eta++)
				{
					if (dc != TMM_Path[eta * 2 + 0] && dc != TMM_Path[eta * 2 + 1])
					{
						TMM_Lc2p[eta] = TMM_I[eta];
					}
					else
					{
						TMM_Lc2p[eta] = TMM_E[eta];
					}
				}

				int h_inverse = GFInverse(Checknode[row].linkVNs_GF[dc]);
				int beta_syn = GFAdd(TMM_Syndrome, TMM_Zn[dc]);
				double L0 = TMM_Lc2p[beta_syn];
				for (int eta = 0; eta < GFQ; eta++)
				{
					int beta =
						GFMultiply(h_inverse, GFAdd(eta, beta_syn));
					Checknode[row].L_c2v[dc][beta] = (TMM_Lc2p[eta]) * 0.8;
				}
			}

			for (int d = 0; d < Checknode[row].weight; d++)
			{
				for (int q = 0; q < GFQ; q++)
				{
					Variablenode[Checknode[row].linkVNs[d]].LLR[q] = Variablenode[Checknode[row].linkVNs[d]].sort_L_v2c[index_in_VN(Checknode, row, d, Variablenode)][q] + Checknode[row].L_c2v[d][q];
				}
			}
		}
	}
	free(TMM_Zn);
	free(TMM_deltaU);
	free(TMM_Min1);
	free(TMM_Min2);
	free(TMM_Min1_Col);
	free(TMM_I);
	free(TMM_Path);
	free(TMM_E);
	free(TMM_Lc2p);
	return 0;
}

int d_TMM_Get_Zn(CN *Checknode, VN *Variablenode, int *TMM_Zn, int row, int &TMM_Syndrome)
{
	TMM_Syndrome = 0;
	for (int dc = 0; dc < Checknode[row].weight; dc++)
	{
		double min = DBL_MAX;
		int min_ele = 0;
		for (int q = 0; q < GFQ; q++)
		{
			if (Variablenode[Checknode[row].linkVNs[dc]].sort_L_v2c[index_in_VN(Checknode, row, dc, Variablenode)][q] < min)
			{
				min = Variablenode[Checknode[row].linkVNs[dc]].sort_L_v2c[index_in_VN(Checknode, row, dc, Variablenode)][q];
				min_ele = GFMultiply(q, Checknode[row].linkVNs_GF[dc]);
			}
		}
		TMM_Zn[dc] = min_ele;
		TMM_Syndrome = GFAdd(TMM_Syndrome, min_ele);
	}
	return 0;
}

int d_TMM_Get_deltaU(CN *Checknode, VN *Variablenode, int *TMM_Zn, float *TMM_deltaU, int row)
{
	for (int dc = 0; dc < Checknode[row].weight; dc++)
	{

		int h_inverse = GFInverse(Checknode[row].linkVNs_GF[dc]);

		int beta_p = GFMultiply(h_inverse, TMM_Zn[dc]);
		float min = Variablenode[Checknode[row].linkVNs[dc]].sort_L_v2c[index_in_VN(Checknode, row, dc, Variablenode)][beta_p];

		for (int x = 0; x < GFQ; x++)
		{
			int eta = GFAdd(x, TMM_Zn[dc]);
			TMM_deltaU[dc * GFQ + eta] =
				Variablenode[Checknode[row].linkVNs[dc]].sort_L_v2c[index_in_VN(Checknode, row, dc, Variablenode)][GFMultiply(h_inverse, x)] - min;
		}
	}
	return 0;
}

int TMM_Get_Min(CN *Checknode, int *TMM_Zn, float *TMM_deltaU, float *TMM_Min1, float *TMM_Min2, int *TMM_Min1_Col, int row)
{
	// sort
	for (int q = 0; q < GFQ; q++)
	{
		// clear
		TMM_Min1[q] = DBL_MAX;
		TMM_Min2[q] = DBL_MAX;
		// search min and submin
		for (int dc = 0; dc < Checknode[row].weight; dc++)
		{
			if (TMM_deltaU[dc * GFQ + q] < TMM_Min1[q])
			{
				TMM_Min2[q] = TMM_Min1[q];
				TMM_Min1[q] = TMM_deltaU[dc * GFQ + q];
				TMM_Min1_Col[q] = dc;
			}
			else if (TMM_deltaU[dc * GFQ + q] < TMM_Min2[q])
			{
				TMM_Min2[q] = TMM_deltaU[dc * GFQ + q];
			}
		}
	}

	return 0;
}

int TMM_ConstructConf(float *TMM_deltaU, float *TMM_Min1, float *TMM_Min2, int *TMM_Min1_Col, float *TMM_I, int *TMM_Path, float *TMM_E)
{
	// dQ[0]
	TMM_I[0] = 0;
	TMM_Path[0] = TMM_Path[1] = -1;
	TMM_E[0] = 0;

	double deviation1, deviation2;
	for (int i = 1; i < GFQ; i++)
	{
		// 1 deviation
		TMM_I[i] = TMM_deltaU[TMM_Min1_Col[i] * GFQ + i];
		TMM_Path[i * 2 + 0] = TMM_Path[i * 2 + 1] = TMM_Min1_Col[i];
		TMM_E[i] = TMM_Min2[i];

		// 2 deviation
		for (int j = 0; j < GFQ; j++)
		{
			if (j != i)
			{
				int k = GFAdd(i, j);
				if (TMM_Min1_Col[j] != TMM_Min1_Col[k]) // 不在同一列
				{
					deviation1 = TMM_deltaU[TMM_Min1_Col[j] * GFQ + j];
					deviation2 = TMM_deltaU[TMM_Min1_Col[k] * GFQ + k];
					if (deviation1 > deviation2 && deviation1 < TMM_I[i])
					{
						TMM_I[i] = deviation1;
						TMM_Path[i * 2 + 0] = TMM_Min1_Col[j];
						TMM_Path[i * 2 + 1] = TMM_Min1_Col[k];
						TMM_E[i] = TMM_Min1[i];
					}
					else if (deviation1 < deviation2 &&
							 deviation2 < TMM_I[i])
					{
						TMM_I[i] = deviation2;
						TMM_Path[i * 2 + 0] = TMM_Min1_Col[j];
						TMM_Path[i * 2 + 1] = TMM_Min1_Col[k];
						TMM_E[i] = TMM_Min1[i];
					}
				}
			}
		}
	}
	return 0;
}