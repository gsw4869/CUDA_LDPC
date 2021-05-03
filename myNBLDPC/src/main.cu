#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "define.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LDPC_Decoder.h"
#include "LDPC_Encoder.h"
#include "GF.h"
#include "math.h"
#include "Decode_GPU.cuh"

int main()
{
	cudaError_t cudaStatus;
	cudaDeviceProp prop;
	int Num_Device;

	cudaStatus = cudaGetDeviceCount(&Num_Device);
	if (cudaStatus != cudaSuccess)
	{
		printf("There is no GPU beyond 1.0, exit!\n");
		exit(0);
	}
	else
	{
		cudaStatus = cudaGetDeviceProperties(&prop, Num_Device - 1);
		if (cudaStatus != cudaSuccess)
		{
			printf("Cannot get device properties, exit!\n");
			exit(0);
		}
	}
	printf("Device Name : %s.\n", prop.name);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
	printf("maxThreadsPerMultiProcessor : %d.\n",
		   prop.maxThreadsPerMultiProcessor);

	AWGNChannel *AWGN;
	AWGN = (AWGNChannel *)malloc(sizeof(AWGN));
	Simulation *SIM;
	SIM = (Simulation *)malloc(sizeof(Simulation));

	CN *Checknode;	  // LDPC码各分块中校验节点的重量
	VN *Variablenode; // LDPC码各分块中变量节点的重量

	LDPCCode *H;
	H = (LDPCCode *)malloc(sizeof(LDPCCode));

	//	先读取行数和列数,分配空间
	FILE *fp_H;

	if (NULL == (fp_H = fopen(Matrixfile, "r")))
	{
		printf("can not open file: %s\n", Matrixfile);
		exit(0);
	}

	int threadNum =
		THREAD_NUM ? THREAD_NUM : std::thread::hardware_concurrency();

	fscanf(fp_H, "%d", &H->Variablenode_num); // 变量节点个数（行数）
	Variablenode = (VN *)malloc(H->Variablenode_num * threadNum * sizeof(VN));

	fscanf(fp_H, "%d", &H->Checknode_num); // 校验节点个数（列数）
	Checknode = (CN *)malloc(H->Checknode_num * threadNum * sizeof(CN));

	fclose(fp_H);
	//
	Get_H(H, Variablenode, Checknode); //初始化剩下的参数

	GFInitial(GFQ);

	CComplex *CONSTELLATION;
	CONSTELLATION = Get_CONSTELLATION(H);

	CComplex *CComplex_sym;

	int *CodeWord_bit;
	CodeWord_bit = (int *)malloc(H->bit_length * sizeof(int));
	memset(CodeWord_bit, 0, H->bit_length * sizeof(int));

	int *CodeWord_sym;
	CodeWord_sym = (int *)malloc(H->Variablenode_num * sizeof(int));
	memset(CodeWord_sym, 0, H->Variablenode_num * sizeof(int));

	int CodeWord_sym_test[96] = {12, 26, 32, 18, 58, 59, 49, 24, 55, 48, 19, 14, 13, 2, 59, 15, 7, 43, 20, 8, 36, 54, 23, 7, 29, 2, 31, 43, 34, 30, 51, 57, 3, 14, 41, 38, 30, 58, 32, 26, 51, 48, 26, 23, 20, 63, 34, 51, 45, 62, 62, 13, 42, 33, 9, 61, 3, 25, 12, 51, 4, 48, 32, 48, 36, 42, 37, 14, 37, 21, 48, 39, 25, 51, 12, 23, 60, 51, 50, 15, 45, 35, 30, 23, 11, 45, 1, 25, 62, 47, 17, 25, 37, 32, 58, 56};

	unsigned *TableMultiply_GPU;
	cudaMalloc((void **)&TableMultiply_GPU, GFQ * GFQ * sizeof(unsigned));
	cudaMemcpy(TableMultiply_GPU, TableMultiply[0], GFQ * GFQ * sizeof(unsigned), cudaMemcpyHostToDevice); //GPU乘法表

	unsigned *TableAdd_GPU;
	cudaMalloc((void **)&TableAdd_GPU, GFQ * GFQ * sizeof(unsigned));
	cudaMemcpy(TableAdd_GPU, TableAdd[0], GFQ * GFQ * sizeof(unsigned), cudaMemcpyHostToDevice); //GPU加法表

	unsigned *TableInverse_GPU;
	cudaMalloc((void **)&TableInverse_GPU, GFQ * sizeof(unsigned));
	cudaMemcpy(TableInverse_GPU, TableInverse, GFQ * sizeof(unsigned), cudaMemcpyHostToDevice); //GPU除法表

	int *Checknode_weight;
	cudaMalloc((void **)&Checknode_weight, H->Checknode_num * sizeof(int));

	int *Checknode_weight_temp = (int *)malloc(H->Checknode_num * sizeof(int));
	for (int i = 0; i < H->Checknode_num; i++)
	{
		Checknode_weight_temp[i] = Checknode[i].weight;
	}
	cudaStatus = cudaMemcpy(Checknode_weight, Checknode_weight_temp, H->Checknode_num * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot copy Checknode_weight\n");
		exit(0);
	}
	free(Checknode_weight_temp);

	int *Variablenode_weight;
	cudaMalloc((void **)&Variablenode_weight, H->Variablenode_num * sizeof(int));

	int *Variablenode_weight_temp = (int *)malloc(H->Variablenode_num * sizeof(int));
	for (int i = 0; i < H->Variablenode_num; i++)
	{
		Variablenode_weight_temp[i] = Variablenode[i].weight;
	}
	cudaStatus = cudaMemcpy(Variablenode_weight, Variablenode_weight_temp, H->Variablenode_num * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot copy Variablenode_weight\n");
		exit(0);
	}
	free(Variablenode_weight_temp);

	int *Variablenode_linkCNs;
	cudaMalloc((void **)&Variablenode_linkCNs, H->Variablenode_num * maxdv * sizeof(int));

	int *Variablenode_linkCNs_temp = (int *)malloc(H->Variablenode_num * maxdv * sizeof(int));
	for (int i = 0; i < H->Variablenode_num; i++)
	{
		for (int j = 0; j < Variablenode[i].weight; j++)
		{
			Variablenode_linkCNs_temp[i * maxdv + j] = Variablenode[i].linkCNs[j] * GFQ * maxdc + index_in_CN(Variablenode, i, j, Checknode) * GFQ; //直接给到数组里的序号
		}
	}
	cudaStatus = cudaMemcpy(Variablenode_linkCNs, Variablenode_linkCNs_temp, H->Variablenode_num * maxdv * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot copy Variablenode_linkCNs\n");
		exit(0);
	}
	free(Variablenode_linkCNs_temp);

	int *Checknode_linkVNs;
	cudaMalloc((void **)&Checknode_linkVNs, H->Checknode_num * maxdc * sizeof(int));

	int *Checknode_linkVNs_temp = (int *)malloc(H->Checknode_num * maxdc * sizeof(int));
	for (int i = 0; i < H->Checknode_num; i++)
	{
		for (int j = 0; j < Checknode[i].weight; j++)
		{
			Checknode_linkVNs_temp[i * maxdc + j] = Checknode[i].linkVNs[j] * GFQ * maxdv + index_in_VN(Checknode, i, j, Variablenode) * GFQ; //直接给到数组里的序号
		}
	}
	cudaStatus = cudaMemcpy(Checknode_linkVNs, Checknode_linkVNs_temp, H->Checknode_num * maxdc * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot copy Checknode_linkVNs\n");
		exit(0);
	}
	free(Checknode_linkVNs_temp);

	int *Checknode_linkVNs_GF;
	cudaMalloc((void **)&Checknode_linkVNs_GF, H->Checknode_num * maxdc * sizeof(int));

	int *Checknode_linkVNs_GF_temp = (int *)malloc(H->Checknode_num * maxdc * sizeof(int));
	for (int i = 0; i < H->Checknode_num; i++)
	{
		for (int j = 0; j < Checknode[i].weight; j++)
		{
			Checknode_linkVNs_GF_temp[i * maxdc + j] = Checknode[i].linkVNs_GF[j];
		}
	}
	cudaStatus = cudaMemcpy(Checknode_linkVNs_GF, Checknode_linkVNs_GF_temp, H->Checknode_num * maxdc * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot copy Checknode_linkVNs_GF\n");
		exit(0);
	}
	free(Checknode_linkVNs_GF_temp);

	if (n_QAM != 2)
	{
		CComplex_sym = (CComplex *)malloc(H->Variablenode_num * sizeof(CComplex));
		BitToSym(H, CodeWord_sym, CodeWord_bit);
		for (int i = 0; i < H->Variablenode_num; i++)
		{
			CodeWord_sym[i] = CodeWord_sym_test[i];
		}
		Modulate(H, CONSTELLATION, CComplex_sym, CodeWord_sym);
	}
	else
	{
		CComplex_sym = (CComplex *)malloc(H->bit_length * sizeof(CComplex));
		for (int i = 0; i < H->Variablenode_num; i++)
		{
			for (int j = 0; j < H->q_bit; j++)
			{
				CodeWord_bit[i * H->q_bit + j] = (CodeWord_sym_test[i] & (1 << j)) >> j;
			}
		}
		BitToSym(H, CodeWord_sym, CodeWord_bit);
		Modulate(H, CONSTELLATION, CComplex_sym, CodeWord_bit);
	}

	printf("sim start\n");
	for (SIM->SNR = startSNR; SIM->SNR <= stopSNR; SIM->SNR += stepSNR)
	{
		AWGN->seed[0] = ix_define;
		AWGN->seed[1] = iy_define;
		AWGN->seed[2] = iz_define;
		AWGN->sigma = 0;
		if (snrtype == 0)
		{
			AWGN->sigma = (float)sqrt(0.5 / (log(n_QAM) / log(2) * H->rate * (pow(10.0, (SIM->SNR / 10.0))))); //(float)LDPC->msgLen / LDPC->codewordLen;
		}
		else if (snrtype == 1)
		{
			AWGN->sigma = (float)sqrt(0.5 / (log(n_QAM) / log(2) * pow(10.0, (SIM->SNR / 10.0))));
		}
		SIM->num_Frames = 0; // 重新开始统计
		SIM->num_Error_Frames = 0;
		SIM->num_Error_Bits = 0;
		SIM->Total_Iteration = 0;
		SIM->num_False_Frames = 0;
		SIM->num_Alarm_Frames = 0;
		SIM->sumTime = 0;
		SIM->FER = 0;
		SIM->BER = 0;
		SIM->AverageIT = 0;
		SIM->FER_False = 0;
		SIM->FER_Alarm = 0;
		if (!CPU_GPU)
		{
			Simulation_CPU((const LDPCCode *)H, AWGN, SIM, (const CComplex *)CONSTELLATION, Variablenode, Checknode, (const CComplex *)CComplex_sym, (const int *)CodeWord_sym);
		}
		else
		{
			Simulation_GPU((const LDPCCode *)H, AWGN, SIM, (const CComplex *)CONSTELLATION, Variablenode, Checknode, (const CComplex *)CComplex_sym, CodeWord_sym, (const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, (const int *)Variablenode_weight, (const int *)Checknode_weight, (const int *)Variablenode_linkCNs, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF);
		}
	}
	cudaFree(TableMultiply_GPU);
	cudaFree(TableAdd_GPU);
	cudaFree(TableInverse_GPU);
	cudaFree(Checknode_weight);
	cudaFree(Variablenode_linkCNs);
	cudaFree(Checknode_linkVNs);
	cudaFree(Checknode_linkVNs_GF);
	free(AWGN);
	free(SIM);
	freeCN(H, Checknode);
	freeVN(H, Variablenode);
	free(H);
	free(CodeWord_sym);
	free(CodeWord_bit);
	free(CComplex_sym);
	free(CONSTELLATION);

	return 0;
}