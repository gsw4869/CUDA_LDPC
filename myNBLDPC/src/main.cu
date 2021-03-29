#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "define.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LDPC_Decoder.cuh"
#include "LDPC_Encoder.cuh"

int main()
{
	AWGNChannel* AWGN;
	AWGN=(AWGNChannel* )malloc(sizeof(AWGN));
	Simulation* SIM;
	SIM=(Simulation* )malloc(sizeof(Simulation));

	CN* Checknode;			// LDPC码各分块中校验节点的重量
	VN* Variablenode;		// LDPC码各分块中变量节点的重量
	
	LDPCCode* H;
	H=(LDPCCode* )malloc(sizeof(LDPCCode));
	
	Get_H(H,Variablenode,Checknode);


	int* CodeWord;
	CodeWord=(int* )malloc(H->length*sizeof(int));
	memset(CodeWord,0,H->length*sizeof(int));


	float* Channel_Out;
	Channel_Out=(float* )malloc(H->length*sizeof(float));

	for (SIM->SNR = startSNR; SIM->SNR <= stopSNR; SIM->SNR += stepSNR)
	{
		AWGN->seed[0]=ix_define;
		AWGN->seed[1]=iy_define;
		AWGN->seed[2]=iz_define;
		AWGN->sigma=0;

		if (snrtype == 0)
		{
			AWGN->sigma = (float)sqrt(0.5 / (H->rate * (pow(10.0, (SIM->SNR / 10.0)))));//(float)LDPC->msgLen / LDPC->codewordLen;
		}
		else if (snrtype == 1)
		{
			AWGN->sigma = (float)sqrt(0.5 / (pow(10.0, (SIM->SNR / 10.0))));
		}
		SIM->num_Frames = 0;					// 重新开始统计
		SIM->num_Error_Frames = 0;
		SIM->num_Error_Bits = 0;
		SIM->Total_Iteration = 0;
		SIM->num_False_Frames = 0;
		SIM->num_Alarm_Frames = 0;

		// BPSK(H,BPSK_Out,CodeWord);
		AWGNChannel_CPU(H,AWGN,Channel_Out,CodeWord);

		Simulation_GPU(SIM, Variablenode, Checknode, Channel_Out);

		Statistic(SIM,CodeWord,CodeWord,H);

		// for(int i=0;i<H->Variablenode_num;i++)
		// {
		// 	printf("%f ",Channel_Out[i]);
		// }
		// printf("\n");
		// exit(0);
	}

	return 0;
}