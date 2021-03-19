#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "define.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LDPC_Decoder.cuh"

int main()
{
	AWGNChannel* AWGN;
	Simulation* SIM;


	int* Weight_Checknode;			// LDPC码各分块中校验节点的重量,长度为J_define+1,最后一个为最大重量.分配在CPU上.
	int* Weight_Variablenode;		// LDPC码各分块中变量节点的重量,长度为L_define+1,最后一个为最大重量.分配在CPU上.
	int* H;					// LDPC码分块式校验矩阵,先行后列,长度为J_define*L_define*Z_define.分配在CPU上.
	int* Address_Variablenode;		// 变量节点相连接的校验节点的序号(注意现在是Num_Frames_OneTime_define个帧穿插在一起同时译码),长度为L_define*J_define*Z_define.分配在CPU上.
	int* Address_Variablenode_GPU;
	float* sigma_GPU;
	cudaError_t cudaStatus;

	

	AWGN = (AWGNChannel*)malloc(sizeof(AWGNChannel));
	if (AWGN == NULL)
	{
		printf("Can not malloc AWGN in main on Host!\n");
		//getch();
		exit(0);
	}

	SIM = (Simulation*)malloc(sizeof(Simulation));
	if (SIM == NULL)
	{
		printf("Can not malloc SIM in main on Host!\n");
		//getch();
		exit(0);
	}

	Weight_Checknode = (int*)malloc((J + 1) * sizeof(int));// LDPC码各分块中校验节点的重量,长度为J_define+1,最后一个为最大重量.分配在CPU上.
	if (Weight_Checknode == NULL)
	{
		printf("Can not malloc Weight_Checknode in main on Host!\n");
		//getch();
		exit(0);
	}

	Weight_Variablenode = (int*)malloc((L + 1) * sizeof(int));// LDPC码各分块中变量节点的重量,长度为L_define+1,最后一个为最大重量.分配在CPU上.
	if (Weight_Variablenode == NULL)
	{
		printf("Can not malloc Weight_Variablenode in main on Host!\n");
		//getch();
		exit(0);
	}

	H = (int*)malloc(J * L * sizeof(int));// LDPC码分块式校验矩阵,先行后列,长度为J_define*L_define.分配在CPU上.
	if (H == NULL)
	{
		printf("Can not malloc Block_H in main on Host!\n");
		//getch();
		exit(0);
	}

	Address_Variablenode = (int*)malloc(J * L * Z * sizeof(int));// 变量节点相连接的校验节点的序号(注意现在是Num_Frames_OneTime_define个帧穿插在一起同时译码),长度为L_define*J_define*Z_define.分配在CPU上.
	if (Address_Variablenode == NULL)
	{
		printf("Can not malloc Address_Variablenode in main on Host!\n");
		//getch();
		exit(0);
	}

	cudaStatus = cudaMalloc((void**)&Address_Variablenode_GPU, J * L * Z * sizeof(int));	// 分配在GPU的global memory中
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc Address_Variablenode_GPU in main on device, exit!\n");
		//getch();
		exit(0);
	}

	cudaStatus = cudaMalloc((void**)&sigma_GPU, sizeof(float));	// 分配在GPU的global memory中
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc sigma_GPU in main on device, exit!\n");
		//getch();
		exit(0);
	}

	memset(Weight_Checknode, 0, (J + 1) * sizeof(int));
	memset(Weight_Variablenode, 0, (L + 1) * sizeof(int));
	memset(H, 0, J * L* sizeof(int));
	memset(Address_Variablenode, -1, J* L* Z * sizeof(int));
	

	// 从define.cuh中读取所要的参数,并从外部的QC-LDPC码分块式校验矩阵中读取相应的偏移量和节点重量值,存于相应的CPU内存中
	Get_H(H, Weight_Checknode, Weight_Variablenode);
	Transform_H(H, Weight_Checknode, Weight_Variablenode, Address_Variablenode);
	cudaStatus = cudaMemcpy(Address_Variablenode_GPU, Address_Variablenode, J * L * Z * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot copy Address_Variablenode to Address_Variablenode_GPU in main, exit!\n");
		//getch();
		exit(0);
	}

	*(AWGN->seed + 0) = ix_define;			// 每一个SNR下要赋一次初值
	*(AWGN->seed + 1) = iy_define;			// 每一个SNR下要赋一次初值
	*(AWGN->seed + 2) = iz_define;			// 每一个SNR下要赋一次初值
	AWGN->sigma = 0.0;						// 每一个SNR下要赋一次值

	// 将部分重要参数显示在屏幕上,并写入相应的文档中
	WriteLogo(AWGN, SIM);

	for (SIM->SNR = startSNR; SIM->SNR <= stopSNR; SIM->SNR += stepSNR)
	{
		/*开始仿真之前,对参数进行初始化*/
		*(AWGN->seed + 0) = ix_define;			// 保证每一个SNR下的仿真环境完全相同
		*(AWGN->seed + 1) = iy_define;
		*(AWGN->seed + 2) = iz_define;
		if (snrtype == 0)
		{
			AWGN->sigma = (float)sqrt(0.5 / (rate * (pow(10.0, (SIM->SNR / 10.0)))));//(float)LDPC->msgLen / LDPC->codewordLen;
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

		//cudaThreadSynchronize();

		/*开始本信噪比点的仿真*/
		if (CPU_GPU == 0)
		{
			//SNR_Simulation_CPU(LDPC, AWGN, SIM, Address_Variablenode, Weight_Checknode, Weight_Variablenode);
		}
		else if (CPU_GPU == 1)
		{
			cudaStatus = cudaMemcpy(sigma_GPU, &(AWGN->sigma), sizeof(float), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess)
			{
				printf("Cannot copy sigma to sigma_GPU in main, exit!\n");
				//getch();
				exit(0);
			}

			Simulation_GPU(AWGN, sigma_GPU, SIM, Address_Variablenode_GPU, Weight_Checknode, Weight_Variablenode);
		}
		/*对CPU和GPU进行同步,防止只执行一个信噪比点即跳出循环*/
		cudaThreadSynchronize();
	}


	free(AWGN);
	free(SIM);
	free(Weight_Checknode);
	free(Weight_Variablenode);
	free(H);
	free(Address_Variablenode);
	cudaFree(sigma_GPU);

	cudaThreadExit();

	printf("\ntask finish\n");
	printf("\nPress any key to stop\n");
	getchar();
	return 0;
}