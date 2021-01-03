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


	int* Weight_Checknode;			// LDPC����ֿ���У��ڵ������,����ΪJ_define+1,���һ��Ϊ�������.������CPU��.
	int* Weight_Variablenode;		// LDPC����ֿ��б����ڵ������,����ΪL_define+1,���һ��Ϊ�������.������CPU��.
	int* H;					// LDPC��ֿ�ʽУ�����,���к���,����ΪJ_define*L_define*Z_define.������CPU��.
	int* Address_Variablenode;		// �����ڵ������ӵ�У��ڵ�����(ע��������Num_Frames_OneTime_define��֡������һ��ͬʱ����),����ΪL_define*J_define*Z_define.������CPU��.
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

	Weight_Checknode = (int*)malloc((J + 1) * sizeof(int));// LDPC����ֿ���У��ڵ������,����ΪJ_define+1,���һ��Ϊ�������.������CPU��.
	if (Weight_Checknode == NULL)
	{
		printf("Can not malloc Weight_Checknode in main on Host!\n");
		//getch();
		exit(0);
	}

	Weight_Variablenode = (int*)malloc((L + 1) * sizeof(int));// LDPC����ֿ��б����ڵ������,����ΪL_define+1,���һ��Ϊ�������.������CPU��.
	if (Weight_Variablenode == NULL)
	{
		printf("Can not malloc Weight_Variablenode in main on Host!\n");
		//getch();
		exit(0);
	}

	H = (int*)malloc(J * L * sizeof(int));// LDPC��ֿ�ʽУ�����,���к���,����ΪJ_define*L_define.������CPU��.
	if (H == NULL)
	{
		printf("Can not malloc Block_H in main on Host!\n");
		//getch();
		exit(0);
	}

	Address_Variablenode = (int*)malloc(J * L * Z * sizeof(int));// �����ڵ������ӵ�У��ڵ�����(ע��������Num_Frames_OneTime_define��֡������һ��ͬʱ����),����ΪL_define*J_define*Z_define.������CPU��.
	if (Address_Variablenode == NULL)
	{
		printf("Can not malloc Address_Variablenode in main on Host!\n");
		//getch();
		exit(0);
	}

	cudaStatus = cudaMalloc((void**)&Address_Variablenode_GPU, J * L * Z * sizeof(int));	// ������GPU��global memory��
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc Address_Variablenode_GPU in main on device, exit!\n");
		//getch();
		exit(0);
	}

	cudaStatus = cudaMalloc((void**)&sigma_GPU, sizeof(float));	// ������GPU��global memory��
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
	

	// ��define.cuh�ж�ȡ��Ҫ�Ĳ���,�����ⲿ��QC-LDPC��ֿ�ʽУ������ж�ȡ��Ӧ��ƫ�����ͽڵ�����ֵ,������Ӧ��CPU�ڴ���
	Get_H(H, Weight_Checknode, Weight_Variablenode);
	Transform_H(H, Weight_Checknode, Weight_Variablenode, Address_Variablenode);
	cudaStatus = cudaMemcpy(Address_Variablenode_GPU, Address_Variablenode, J * L * Z * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot copy Address_Variablenode to Address_Variablenode_GPU in main, exit!\n");
		//getch();
		exit(0);
	}

	*(AWGN->seed + 0) = ix_define;			// ÿһ��SNR��Ҫ��һ�γ�ֵ
	*(AWGN->seed + 1) = iy_define;			// ÿһ��SNR��Ҫ��һ�γ�ֵ
	*(AWGN->seed + 2) = iz_define;			// ÿһ��SNR��Ҫ��һ�γ�ֵ
	AWGN->sigma = 0.0;						// ÿһ��SNR��Ҫ��һ��ֵ

	// ��������Ҫ������ʾ����Ļ��,��д����Ӧ���ĵ���
	WriteLogo(AWGN, SIM);

	for (SIM->SNR = startSNR; SIM->SNR <= stopSNR; SIM->SNR += stepSNR)
	{
		/*��ʼ����֮ǰ,�Բ������г�ʼ��*/
		*(AWGN->seed + 0) = ix_define;			// ��֤ÿһ��SNR�µķ��滷����ȫ��ͬ
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

		SIM->num_Frames = 0;					// ���¿�ʼͳ��
		SIM->num_Error_Frames = 0;
		SIM->num_Error_Bits = 0;
		SIM->Total_Iteration = 0;
		SIM->num_False_Frames = 0;
		SIM->num_Alarm_Frames = 0;

		//cudaThreadSynchronize();

		/*��ʼ������ȵ�ķ���*/
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
		/*��CPU��GPU����ͬ��,��ִֹֻ��һ������ȵ㼴����ѭ��*/
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