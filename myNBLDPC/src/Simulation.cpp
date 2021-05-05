#include "Simulation.h"
#include "LDPC_Encoder.h"
#include "LDPC_Decoder.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#include <fstream>
#include <string>
#include "Decode_GPU.cuh"
#include <mutex>
#include <thread>
#include <vector>

std::mutex mtx;

void decode_once_cpu(const LDPCCode *H, AWGNChannel *AWGN, Simulation *SIM, const CComplex *CONSTELLATION, VN *Variablenode_0, CN *Checknode_0, const CComplex *CComplex_sym, const int *CodeWord_sym, int thread_id)
{

	int iter_number = 0;

	VN *Variablenode = Variablenode_0 + thread_id * H->Variablenode_num;

	CN *Checknode = Checknode_0 + thread_id * H->Checknode_num;

	int *DecodeOutput;
	DecodeOutput = (int *)malloc(H->Variablenode_num * sizeof(int));
	memset(DecodeOutput, 0, H->Variablenode_num * sizeof(int));

	CComplex *CComplex_sym_Channelout;
	if (n_QAM != 2)
	{
		CComplex_sym_Channelout = (CComplex *)malloc(H->Variablenode_num * sizeof(CComplex));
	}
	else
	{
		CComplex_sym_Channelout = (CComplex *)malloc(H->bit_length * sizeof(CComplex));
	}
	std::chrono::_V2::steady_clock::time_point start;
	std::chrono::_V2::steady_clock::time_point end;

	while (SIM->num_Error_Frames < leastErrorFrames || SIM->num_Frames < leastTestFrames)
	{

		mtx.lock();

		AWGNChannel_CPU(H, AWGN, CComplex_sym_Channelout, (const CComplex *)CComplex_sym);

		mtx.unlock();

		Demodulate(H, AWGN, (const CComplex *)CONSTELLATION, Variablenode, CComplex_sym_Channelout);

		start = std::chrono::steady_clock::now();

		if (decoder_method == 0)
		{
			Decoding_EMS(H, Variablenode, Checknode, EMS_NM, EMS_NC, DecodeOutput, iter_number);
		}
		else if (decoder_method == 1)
		{

			Decoding_TMM(H, Variablenode, Checknode, EMS_NM, EMS_NC, DecodeOutput, iter_number);
		}
		else if (decoder_method == 2)
		{
			Decoding_EMS(H, Variablenode, Checknode, GFQ, maxdc - 1, DecodeOutput, iter_number);
		}

		end = std::chrono::steady_clock::now();

		mtx.lock();

		SIM->num_Frames += 1;
		SIM->sumTime += (end - start).count() / 1000000000.0;

		SIM->Total_Iteration += iter_number;

		Statistic(SIM, (const int *)CodeWord_sym, DecodeOutput, H);

		mtx.unlock();
	}
	free(DecodeOutput);
	free(CComplex_sym_Channelout);
}

void decode_once_gpu(const LDPCCode *H, AWGNChannel *AWGN, Simulation *SIM, const CComplex *CONSTELLATION, VN *Variablenode_0, CN *Checknode_0, const CComplex *CComplex_sym, const int *CodeWord_sym, const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const int *Variablenode_weight, const int *Checknode_weight, const int *Variablenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int thread_id)
{

	int iter_number = 0;

	VN *Variablenode = Variablenode_0 + thread_id * H->Variablenode_num;

	CN *Checknode = Checknode_0 + thread_id * H->Checknode_num;

	int *DecodeOutput;
	DecodeOutput = (int *)malloc(H->Variablenode_num * sizeof(int));
	memset(DecodeOutput, 0, H->Variablenode_num * sizeof(int));

	CComplex *CComplex_sym_Channelout;
	if (n_QAM != 2)
	{
		CComplex_sym_Channelout = (CComplex *)malloc(H->Variablenode_num * sizeof(CComplex));
	}
	else
	{
		CComplex_sym_Channelout = (CComplex *)malloc(H->bit_length * sizeof(CComplex));
	}
	std::chrono::_V2::steady_clock::time_point start = std::chrono::steady_clock::now();
	std::chrono::_V2::steady_clock::time_point end;
	SIM->sumTime = 0;

	while (SIM->num_Error_Frames < leastErrorFrames || SIM->num_Frames < leastTestFrames)
	{

		mtx.lock();

		AWGNChannel_CPU(H, AWGN, CComplex_sym_Channelout, CComplex_sym);

		mtx.unlock();

		Demodulate(H, AWGN, CONSTELLATION, Variablenode, CComplex_sym_Channelout);

		start = std::chrono::steady_clock::now();

		if (decoder_method == 0)
		{
			Decoding_EMS_GPU(H, Variablenode, Checknode, EMS_NM, EMS_NC, DecodeOutput, (const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, (const int *)Variablenode_weight, (const int *)Checknode_weight, (const int *)Variablenode_linkCNs, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, iter_number);
		}
		else if (decoder_method == 1)
		{
			printf("unfinished\n");
			exit(0);
		}
		else if (decoder_method == 2)
		{
			Decoding_EMS_GPU(H, Variablenode, Checknode, GFQ, maxdc - 1, DecodeOutput, (const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, (const int *)Variablenode_weight, (const int *)Checknode_weight, (const int *)Variablenode_linkCNs, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, iter_number);
		}
		end = std::chrono::steady_clock::now();

		mtx.lock();

		SIM->num_Frames += 1;

		SIM->sumTime += (end - start).count() / 1000000000.0;

		SIM->Total_Iteration += iter_number;

		Statistic(SIM, CodeWord_sym, DecodeOutput, H);

		mtx.unlock();
	}
	free(DecodeOutput);
	free(CComplex_sym_Channelout);
}

/*
* 仿真函数
* AWGN:AWGNChannel类变量，包含噪声种子等
* 
*/
void Simulation_CPU(const LDPCCode *H, AWGNChannel *AWGN, Simulation *SIM, const CComplex *CONSTELLATION, VN *Variablenode, CN *Checknode, const CComplex *CComplex_sym, const int *CodeWord_sym)
{
	// printf(" %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4esec\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->sumTime / SIM->num_Frames);
	// printf("%d %d %d %f\n", AWGN->seed[0], AWGN->seed[1], AWGN->seed[2], AWGN->sigma);

	int threadNum =
		THREAD_NUM ? THREAD_NUM : std::thread::hardware_concurrency();
	std::vector<std::thread> threads_(threadNum);

	for (int i = 0; i < threadNum; i++)
	{
		threads_[i] = std::thread(
			decode_once_cpu, (const LDPCCode *)H, AWGN, SIM, (const CComplex *)CONSTELLATION, Variablenode, Checknode, (const CComplex *)CComplex_sym, (const int *)CodeWord_sym, i);
	}
	for (int i = 0; i < threadNum; i++)
	{
		threads_[i].join();
	}
	for (int i = 0; i < threadNum; i++)
	{
		threads_[i].~thread();
	}

	if (SIM->num_Error_Frames >= leastErrorFrames && SIM->num_Frames >= leastTestFrames)
	{
		SIM->BER = ((double)SIM->num_Error_Bits / (double)(SIM->num_Frames)) / (double)(H->Variablenode_num);
		SIM->FER = (double)SIM->num_Error_Frames / (double)SIM->num_Frames;
		SIM->AverageIT = (double)SIM->Total_Iteration / (double)SIM->num_Frames;
		// SIM->FER_Alarm = (double)SIM->num_Alarm_Frames / (double)SIM->num_Frames;
		// SIM->FER_False = (double)SIM->num_False_Frames / (double)SIM->num_Frames;
		printf(" %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4esec\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->sumTime / SIM->num_Frames / threadNum);
		FILE *fp_H;
		if (NULL == (fp_H = fopen("results.txt", "a")))
		{
			printf("can not open file: results.txt\n");
			exit(0);
		}
		fprintf(fp_H, " %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4esec\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->sumTime / SIM->num_Frames / threadNum);
		fclose(fp_H);
	}
}

/*
* 仿真函数
* AWGN:AWGNChannel类变量，包含噪声种子等
* 
*/
void Simulation_GPU(const LDPCCode *H, AWGNChannel *AWGN, Simulation *SIM, const CComplex *CONSTELLATION, VN *Variablenode, CN *Checknode, const CComplex *CComplex_sym, int *CodeWord_sym, const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const int *Variablenode_weight, const int *Checknode_weight, const int *Variablenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF)
{
	int threadNum =
		THREAD_NUM ? THREAD_NUM : std::thread::hardware_concurrency();
	std::vector<std::thread> threads_(threadNum);
	for (int i = 0; i < threadNum; i++)
	{
		threads_[i] = std::thread(
			decode_once_gpu, (const LDPCCode *)H, AWGN, SIM, (const CComplex *)CONSTELLATION, Variablenode, Checknode, (const CComplex *)CComplex_sym, (const int *)CodeWord_sym, (const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, (const int *)Variablenode_weight, (const int *)Checknode_weight, (const int *)Variablenode_linkCNs, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, i);
	}
	for (int i = 0; i < threadNum; i++)
	{
		threads_[i].join();
	}
	for (int i = 0; i < threadNum; i++)
	{
		threads_[i].~thread();
	}

	if (SIM->num_Error_Frames >= leastErrorFrames && SIM->num_Frames >= leastTestFrames)
	{
		SIM->BER = ((double)SIM->num_Error_Bits / (double)(SIM->num_Frames)) / (double)(H->Variablenode_num);
		SIM->FER = (double)SIM->num_Error_Frames / (double)SIM->num_Frames;
		SIM->AverageIT = (double)SIM->Total_Iteration / (double)SIM->num_Frames;
		// SIM->FER_Alarm = (double)SIM->num_Alarm_Frames / (double)SIM->num_Frames;
		// SIM->FER_False = (double)SIM->num_False_Frames / (double)SIM->num_Frames;
		printf(" %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4esec\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->sumTime / SIM->num_Frames / threadNum);
		FILE *fp_H;
		if (NULL == (fp_H = fopen("results.txt", "a")))
		{
			printf("can not open file: results.txt\n");
			exit(0);
		}
		fprintf(fp_H, " %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4esec\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->sumTime / SIM->num_Frames / threadNum);
		fclose(fp_H);
	}
}

/*
* 统计函数，统计仿真结果
*/
int Statistic(Simulation *SIM, const int *CodeWord_Frames, int *D, const LDPCCode *H)
{
	int index1;
	int Error_msgBit = 0;

	int threadNum =
		THREAD_NUM ? THREAD_NUM : std::thread::hardware_concurrency();

	for (index1 = 0; index1 < H->Variablenode_num; index1++)
	{
		Error_msgBit = (D[index1] != CodeWord_Frames[index1]) ? Error_msgBit + 1 : Error_msgBit;
	}
	SIM->num_Error_Bits += Error_msgBit;
	SIM->num_Error_Frames = (Error_msgBit != 0) ? SIM->num_Error_Frames + 1 : SIM->num_Error_Frames;
	// SIM->num_Error_Frames = (Error_msgBit!= 0 || D[index0 + CW_Len * Num_Frames_OneTime] == 0) ? SIM->num_Error_Frames + 1 : SIM->num_Error_Frames;
	// SIM->num_Alarm_Frames = (Error_msgBit[index0] == 0 && D[index0 + CW_Len * Num_Frames_OneTime] == 0) ? SIM->num_Alarm_Frames + 1 : SIM->num_Alarm_Frames;
	// SIM->num_False_Frames = (Error_msgBit[index0] != 0 && D[index0 + CW_Len * Num_Frames_OneTime] == 1) ? SIM->num_False_Frames + 1 : SIM->num_False_Frames;

	if (SIM->num_Frames % displayStep == 0)
	{
		SIM->BER = ((double)SIM->num_Error_Bits / (double)(SIM->num_Frames)) / (double)(H->Variablenode_num);
		SIM->FER = (double)SIM->num_Error_Frames / (double)SIM->num_Frames;
		SIM->AverageIT = (double)SIM->Total_Iteration / (double)SIM->num_Frames;
		// SIM->FER_Alarm = (double)SIM->num_Alarm_Frames / (double)SIM->num_Frames;
		// SIM->FER_False = (double)SIM->num_False_Frames / (double)SIM->num_Frames;
		printf(" %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4esec\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->sumTime / SIM->num_Frames / threadNum);
		FILE *fp_H;
		if (NULL == (fp_H = fopen("results.txt", "a")))
		{
			printf("can not open file: results.txt\n");
			exit(0);
		}
		fprintf(fp_H, " %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4esec\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->sumTime / SIM->num_Frames / threadNum);
		fclose(fp_H);
	}

	if (SIM->num_Error_Frames >= leastErrorFrames && SIM->num_Frames >= leastTestFrames)
	{
		SIM->BER = ((double)SIM->num_Error_Bits / (double)(SIM->num_Frames)) / (double)(H->Variablenode_num);
		SIM->FER = (double)SIM->num_Error_Frames / (double)SIM->num_Frames;
		SIM->AverageIT = (double)SIM->Total_Iteration / (double)SIM->num_Frames;
		// SIM->FER_Alarm = (double)SIM->num_Alarm_Frames / (double)SIM->num_Frames;
		// SIM->FER_False = (double)SIM->num_False_Frames / (double)SIM->num_Frames;
		// printf(" %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4esec\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->sumTime / SIM->num_Frames);
		FILE *fp_H;
		if (NULL == (fp_H = fopen("results.txt", "a")))
		{
			printf("can not open file: results.txt\n");
			exit(0);
		}
		// fprintf(fp_H, " %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4esec\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->sumTime / SIM->num_Frames);
		fclose(fp_H);
		return 1;
	}
	return 0;
}

CComplex *Get_CONSTELLATION(LDPCCode *H)
{
	CComplex *CONSTELLATION = new CComplex[GFQ];

	char temp[100];
	FILE *fp_H;
	if (NULL == (fp_H = fopen(Constellationfile, "r")))
	{
		printf("can not open file: %s\n", Constellationfile);
		exit(0);
	}

	int index;
	for (int k = 0; k < n_QAM; k++)
	{
		fscanf(fp_H, "%s", temp);
		fscanf(fp_H, "%d", &index);
		fscanf(fp_H, "%s", temp);
		fscanf(fp_H, "%f", &CONSTELLATION[index].Real); // GF域
		fscanf(fp_H, "%s", temp);
		fscanf(fp_H, "%f", &CONSTELLATION[index].Image); // GF域
	}
	fclose(fp_H);

	return CONSTELLATION;
}

/*
H:校验矩阵
Weight_Checknode:按顺序记录每个校验节点的重量
Weight_Variablenode:按顺序记录每个变量节点的重量
Address_Variablenode:变量节点相连的校验节点的序号
Address_Checknode:校验节点相连的变量节点的序号
*/
void Get_H(LDPCCode *H, VN *Variablenode, CN *Checknode)
{
	int index1;

	FILE *fp_H;

	if (NULL == (fp_H = fopen(Matrixfile, "r")))
	{
		printf("can not open file: %s\n", Matrixfile);
		exit(0);
	}

	fscanf(fp_H, "%d", &H->Variablenode_num); // 变量节点个数（行数）
	// Variablenode=(VN *)malloc(H->Variablenode_num*sizeof(VN));

	fscanf(fp_H, "%d", &H->Checknode_num); // 校验节点个数（列数）
	// Checknode=(CN *)malloc(H->Checknode_num*sizeof(CN));

	H->rate = (float)(H->Variablenode_num - H->Checknode_num) / H->Variablenode_num;

	fscanf(fp_H, "%d", &H->GF); // GF域

	switch (GFQ)
	{
	case 4:
		H->q_bit = 2;
		break;
	case 8:
		H->q_bit = 3;
		break;
	case 16:
		H->q_bit = 4;
		break;
	case 32:
		H->q_bit = 5;
		break;
	case 64:
		H->q_bit = 6;
		break;
	case 128:
		H->q_bit = 7;
		break;
	case 256:
		H->q_bit = 8;
		break;
	default:
		printf("error");
		exit(0);
	}

	H->bit_length = H->Variablenode_num * H->q_bit;

	fscanf(fp_H, "%d", &H->maxWeight_variablenode); //变量节点相连的校验节点的个数

	fscanf(fp_H, "%d", &H->maxWeight_checknode); //校验节点相连的变量节点的个数
	int threadNum =
		THREAD_NUM ? THREAD_NUM : std::thread::hardware_concurrency();

	for (int i = 0; i < H->Variablenode_num; i++)
	{
		fscanf(fp_H, "%d", &index1);
		for (int j = 0; j < threadNum; j++)
		{
			Variablenode[j * H->Variablenode_num + i].weight = index1;
			Variablenode[j * H->Variablenode_num + i].linkCNs = (int *)malloc(Variablenode[i].weight * sizeof(int));
			Variablenode[j * H->Variablenode_num + i].linkCNs_GF = (int *)malloc(Variablenode[i].weight * sizeof(int));
			Variablenode[j * H->Variablenode_num + i].L_ch = (float *)malloc((GFQ) * sizeof(float));
			Variablenode[j * H->Variablenode_num + i].LLR = (float *)malloc((GFQ) * sizeof(float));
			Variablenode[j * H->Variablenode_num + i].sort_L_v2c = malloc_2_float(Variablenode[i].weight, GFQ);
			Variablenode[j * H->Variablenode_num + i].sort_Entr_v2c = malloc_2(Variablenode[i].weight, GFQ);
		}
	}

	for (int i = 0; i < H->Checknode_num; i++)
	{
		fscanf(fp_H, "%d", &index1);
		for (int j = 0; j < threadNum; j++)
		{
			Checknode[j * H->Checknode_num + i].weight = index1;
			Checknode[j * H->Checknode_num + i].linkVNs = (int *)malloc(Checknode[i].weight * sizeof(int));
			Checknode[j * H->Checknode_num + i].linkVNs_GF = (int *)malloc(Checknode[i].weight * sizeof(int));
			Checknode[j * H->Checknode_num + i].L_c2v = malloc_2_float(Checknode[i].weight, GFQ);
		}
	}

	for (int i = 0; i < H->Variablenode_num; i++)
	{
		for (int j = 0; j < Variablenode[i].weight; j++)
		{
			fscanf(fp_H, "%d", &index1);
			for (int t = 0; t < threadNum; t++)
			{
				Variablenode[t * H->Variablenode_num + i].linkCNs[j] = index1 - 1;
			}
			fscanf(fp_H, "%d", &index1);
			for (int t = 0; t < threadNum; t++)
			{
				Variablenode[t * H->Variablenode_num + i].linkCNs_GF[j] = index1;
			}
		}
	}

	for (int i = 0; i < H->Checknode_num; i++)
	{
		for (int j = 0; j < Checknode[i].weight; j++)
		{
			fscanf(fp_H, "%d", &index1);
			for (int t = 0; t < threadNum; t++)
			{
				Checknode[t * H->Checknode_num + i].linkVNs[j] = index1 - 1;
			}
			fscanf(fp_H, "%d", &index1);
			for (int t = 0; t < threadNum; t++)
			{
				Checknode[t * H->Checknode_num + i].linkVNs_GF[j] = index1;
			}
		}
	}

	fclose(fp_H);
}