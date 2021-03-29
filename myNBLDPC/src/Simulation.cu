#include "Simulation.cuh"
#include "LDPC_Encoder.cuh"
#include "LDPC_Decoder.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>

/*
* 仿真函数
* AWGN:AWGNChannel类变量，包含噪声种子等
* 
*/
void Simulation_GPU(Simulation* SIM, VN* Variablenode, CN* Checknode, float* Channel_Out)
{	
	// while (SIM->num_Frames<50)
	// {
	// 	SIM->num_Frames += 1;
	// }
	SIM->num_Frames = 40960;
}

/*
* 统计函数，统计仿真结果
*/
int Statistic(Simulation* SIM, int* CodeWord_Frames, int* D,LDPCCode *H)
{
	int index1;
	int Error_msgBit=0;	

	
	for (index1 = 0; index1 < H->length; index1++)
	{
		Error_msgBit = (D[index1] != CodeWord_Frames[index1]) ? Error_msgBit + 1 : Error_msgBit;
	}
	SIM->num_Error_Bits += Error_msgBit;
	SIM->num_Error_Frames = (Error_msgBit!= 0) ? SIM->num_Error_Frames + 1 : SIM->num_Error_Frames;
	// SIM->num_Error_Frames = (Error_msgBit!= 0 || D[index0 + CW_Len * Num_Frames_OneTime] == 0) ? SIM->num_Error_Frames + 1 : SIM->num_Error_Frames;
	// SIM->num_Alarm_Frames = (Error_msgBit[index0] == 0 && D[index0 + CW_Len * Num_Frames_OneTime] == 0) ? SIM->num_Alarm_Frames + 1 : SIM->num_Alarm_Frames;
	// SIM->num_False_Frames = (Error_msgBit[index0] != 0 && D[index0 + CW_Len * Num_Frames_OneTime] == 1) ? SIM->num_False_Frames + 1 : SIM->num_False_Frames;
	SIM->Total_Iteration += H->iteraTime;
	
	if (SIM->num_Frames % displayStep == 0)
	{
		SIM->BER = ((double)SIM->num_Error_Bits / (double)(SIM->num_Frames)) / (double)(H->length);
		SIM->FER = (double)SIM->num_Error_Frames / (double)SIM->num_Frames;
		SIM->AverageIT = (double)SIM->Total_Iteration / (double)SIM->num_Frames;
		// SIM->FER_Alarm = (double)SIM->num_Alarm_Frames / (double)SIM->num_Frames;
		// SIM->FER_False = (double)SIM->num_False_Frames / (double)SIM->num_Frames;
		printf(" %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4e %6.4e\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->FER_False, SIM->FER_Alarm);
	}

	if (SIM->num_Error_Frames >= leastErrorFrames && SIM->num_Frames >= leastTestFrames)
	{
		SIM->BER = ((double)SIM->num_Error_Bits / (double)(SIM->num_Frames)) / (double)(H->length);
		SIM->FER = (double)SIM->num_Error_Frames / (double)SIM->num_Frames;
		SIM->AverageIT = (double)SIM->Total_Iteration / (double)SIM->num_Frames;
		// SIM->FER_Alarm = (double)SIM->num_Alarm_Frames / (double)SIM->num_Frames;
		// SIM->FER_False = (double)SIM->num_False_Frames / (double)SIM->num_Frames;
		printf(" %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4e %6.4e\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->FER_False, SIM->FER_Alarm);
		return 1;
	}
	return 0;
}

/*
H:校验矩阵
Weight_Checknode:按顺序记录每个校验节点的重量
Weight_Variablenode:按顺序记录每个变量节点的重量
Address_Variablenode:变量节点相连的校验节点的序号
Address_Checknode:校验节点相连的变量节点的序号
*/
void Get_H(LDPCCode* H,VN* Variablenode,CN* Checknode)
{
	int index1;
	char file[100]="Tanner_74_9_Z128_GF16.txt";
	FILE* fp_H;
	
	if (NULL == (fp_H = fopen(file, "r")))
	{
		printf("can not open file: %s\n", file);
		exit(0);
	}

	fscanf(fp_H, "%d", &H->Variablenode_num);// 变量节点个数（行数）
	Variablenode=(VN *)malloc(H->Variablenode_num*sizeof(VN));

	fscanf(fp_H, "%d", &H->Checknode_num);// 校验节点个数（列数）
	Checknode=(CN *)malloc(H->Checknode_num*sizeof(CN));

	H->rate=(float)(H->Variablenode_num-H->Checknode_num)/H->Variablenode_num;
    H->length=H->Variablenode_num;
	fscanf(fp_H, "%d", &index1);// GF域

	fscanf(fp_H, "%d", &H->maxWeight_variablenode);//最大行重

	fscanf(fp_H, "%d", &H->maxWeight_checknode);//最大列重


	for(int i=0;i<H->Variablenode_num;i++)
	{
		fscanf(fp_H, "%d", &index1);
		Variablenode[i].weight=index1;
		Variablenode[i].linkCNs=(int *)malloc(Variablenode[i].weight*sizeof(int));
		Variablenode[i].linkCNs_GF=(int *)malloc(Variablenode[i].weight*sizeof(int));
	}

	
	for(int i=0;i<H->Checknode_num;i++)
	{
		fscanf(fp_H, "%d", &index1);
		Checknode[i].weight=index1;
		Checknode[i].linkVNs=(int *)malloc(Checknode[i].weight*sizeof(int));
		Checknode[i].linkVNs_GF=(int *)malloc(Checknode[i].weight*sizeof(int));
	}
	
	for(int i=0;i<H->Variablenode_num;i++)
	{
		for(int j=0;j<Variablenode[i].weight;j++)
		{
			fscanf(fp_H, "%d", &index1);
			Variablenode[i].linkCNs[j]=index1;
			fscanf(fp_H, "%d", &index1);
			Variablenode[i].linkCNs_GF[j]=index1;

		}
	}

	for(int i=0;i<H->Checknode_num;i++)
	{
		for(int j=0;j<Checknode[i].weight;j++)
		{
			fscanf(fp_H, "%d", &index1);
			Checknode[i].linkVNs[j]=index1;
			fscanf(fp_H, "%d", &index1);
			Checknode[i].linkVNs_GF[j]=index1;

		}
	}

	fclose(fp_H);

	// for(int i=0;i<H->Checknode_num;i++)
	// {
	// 	for(int j=0;j<Checknode[i].weight;j++)
	// 	{
	// 		printf("%d ",Checknode[i].linkVNs_GF[j]);

	// 	}
	// 	printf("\n");
	// }

}