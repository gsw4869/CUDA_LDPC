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

/*
* D:һ�����
* Channel_Out:����AWGN�ŵ����ź�
* Weight_Checknode:У��ڵ�����
* Weight_Variablenode:�����ڵ�����
* Address_Variablenode:ÿ�������ڵ�����ӦУ��ڵ��memory_rq�ĵ�ַ
* LDPC:��������
*/
void LDPC_Decoder_GPU(int* D, float* Channel_Out, cudaDeviceProp prop, int* Address_Variablenode, int* Weight_Checknode, int* Weight_Variablenode, LDPCCode *LDPC)
{
	cudaError_t cudaStatus;
	int index0, index1, Length;
	int ThreadPerBlock, Num_Block;
	float* Memory_RQ;
	int* Weight_Checknode_GPU, *Weight_Variablenode_GPU;
	int* D_GPU;
	cudaEvent_t GPU_start;			// GPU����ͳ�Ʋ���
	cudaEvent_t GPU_stop;
	cudaEventCreate(&GPU_start);
	cudaEventCreate(&GPU_stop);

	Length = (Message_CW == 0) ? msgLen : CW_Len;

	cudaStatus = cudaMalloc((void**)&Memory_RQ, parLen * Weight_Checknode[J] * Num_Frames_OneTime * sizeof(float));	// ������GPU��global memory��
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc Memory_RQ in LDPC_Decoder_GPU on device, exit!\n");
		//getch();
		exit(0);
	}
	cudaStatus = cudaMalloc((void**)&D_GPU, (CW_Len + 1) * Num_Frames_OneTime * sizeof(int));		// ������GPU��global memory��
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc D_GPU in LDPC_Decoder_GPU on device, exit!\n");
		//getch();
		exit(0);
	}
	cudaStatus = cudaMalloc((void**)&Weight_Checknode_GPU, (J + 1) * sizeof(int));		// ������GPU��global memory��
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc Weight_Checknode_GPU in LDPC_Decoder_GPU on device, exit!\n");
		//getch();
		exit(0);
	}
	cudaStatus = cudaMalloc((void**)&Weight_Variablenode_GPU, (L + 1) * sizeof(int));		// ������GPU��global memory��
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc Weight_Checknode_GPU in LDPC_Decoder_GPU on device, exit!\n");
		//getch();
		exit(0);
	}
	cudaStatus = cudaMemcpy(Weight_Checknode_GPU, Weight_Checknode, (J + 1) * sizeof(int), cudaMemcpyHostToDevice);//J�� L�У����һ���?�������?
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot copy Weight_Checknode to Weight_Checknode_GPU in LDPC_Decoder_GPU, exit!\n");
		//getch();
		exit(0);
	}
	cudaStatus = cudaMemcpy(Weight_Variablenode_GPU, Weight_Variablenode, (L + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot copy Weight_Variablenode to Weight_Variablenode_GPU in LDPC_Decoder_GPU, exit!\n");
		//getch();
		exit(0);
	}

	// ��ʼ��
	cudaStatus = cudaMemset(Memory_RQ, 0, parLen * Weight_Checknode[J] * Num_Frames_OneTime * sizeof(float));	// �洢������,Ϊ��һ�ε�����׼��,parlenУ��λ���ȣ�J*Z��
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot memset Memory_RQ in LDPC_Decoder_GPU on device, exit!\n");
		//getch();
		exit(0);
	}

	LDPC->iteraTime = 0;

	cudaThreadSynchronize();

	while (LDPC->iteraTime < maxIT)
	{
		LDPC->iteraTime = LDPC->iteraTime + 1;		
		if ((Z * Num_Frames_OneTime) % prop.maxThreadsPerBlock == 0)	// ��ʱ����prop.maxThreadsPerBlock�ֳɵ�ÿ���߳̿������һ���п�����п�Ľڵ�,һ����֮����Թ�������
		{
			cudaThreadSynchronize();
			ThreadPerBlock = prop.maxThreadsPerBlock;
			Num_Block = ((Num_Frames_OneTime * CW_Len) % ThreadPerBlock == 0) ? (Num_Frames_OneTime * CW_Len) / ThreadPerBlock : ((Num_Frames_OneTime * CW_Len) / ThreadPerBlock) + 1;
			Variablenode_Shared_Kernel<<<Num_Block, ThreadPerBlock>>>(Memory_RQ, D_GPU, Channel_Out, Address_Variablenode, Weight_Variablenode_GPU);//�����ڵ���㣬�õ�L

			ThreadPerBlock = prop.maxThreadsPerBlock;
			Num_Block = ((Num_Frames_OneTime * parLen) % ThreadPerBlock == 0) ? (Num_Frames_OneTime * parLen) / ThreadPerBlock : ((Num_Frames_OneTime * parLen) / ThreadPerBlock) + 1;

			if (decoder_method == 0)
			{
				Checknode_Shared_Kernel<<<Num_Block, ThreadPerBlock>>>(Memory_RQ, Weight_Checknode_GPU);
			}

			cudaThreadSynchronize();
		}
		else
		{
			cudaThreadSynchronize();
			ThreadPerBlock = prop.maxThreadsPerBlock;
			Num_Block = ((Num_Frames_OneTime * CW_Len) % ThreadPerBlock == 0) ? (Num_Frames_OneTime * CW_Len) / ThreadPerBlock : ((Num_Frames_OneTime * CW_Len) / ThreadPerBlock) + 1;
			Variablenode_Kernel<<<Num_Block, ThreadPerBlock>>>(Memory_RQ, D_GPU, Channel_Out, Address_Variablenode, Weight_Variablenode_GPU);//�����ڵ���㣬�õ�L

			cudaThreadSynchronize();

			ThreadPerBlock = prop.maxThreadsPerBlock;
			Num_Block = ((Num_Frames_OneTime * parLen) % ThreadPerBlock == 0) ? (Num_Frames_OneTime * parLen) / ThreadPerBlock : ((Num_Frames_OneTime * parLen) / ThreadPerBlock) + 1;
			if (decoder_method == 0)
			{
				Checknode_Kernel<<<Num_Block, ThreadPerBlock>>>(Memory_RQ, Weight_Checknode_GPU);
			}
			
			cudaThreadSynchronize();
		}
		cudaThreadSynchronize();

		memset(D + CW_Len * Num_Frames_OneTime, 0, Num_Frames_OneTime * sizeof(int));
		cudaMemcpy(D, D_GPU, CW_Len * Num_Frames_OneTime * sizeof(int), cudaMemcpyDeviceToHost);

		for (index0 = 0; index0 < Length; index0++)
		{
			for (index1 = 0; index1 < Num_Frames_OneTime; index1++)
			{
				D[index1 + CW_Len * Num_Frames_OneTime] += D[index0 * Num_Frames_OneTime + index1];//��ÿһ֡�������ֽڼ��������
			}
		}
		index0 = 0;
		for (index1 = 0; index1 < Num_Frames_OneTime; index1++)
		{
			D[index1 + CW_Len * Num_Frames_OneTime] = (D[index1 + CW_Len * Num_Frames_OneTime] == 0) ? 1 : 0;//ȫ�����м�����Ϊȫ0����ȷ���Ϊ1
			index0 += D[index1 + CW_Len * Num_Frames_OneTime];//ͳ�ƶԵ�֡��
		}
		if (index0 == Num_Frames_OneTime)
		{
			break;//����֡��������
		}
		cudaThreadSynchronize();

	}
	cudaEventDestroy(GPU_start);
	cudaEventDestroy(GPU_stop);

	cudaFree(Memory_RQ);
	cudaFree(Weight_Checknode_GPU);
	cudaFree(Weight_Variablenode_GPU);
	cudaFree(D_GPU);
}

/*
* Memory_RQ:���ڴ洢�����ڵ��У��ڵ����ʱ�õ���R��Qֵ
* D:�������
* Weight_Variablenode:�����ڵ�����
* Address_Variablenode:ÿ�������ڵ�����ӦУ��ڵ��memory_rq�ĵ�ַ
*/
__global__ void Variablenode_Kernel(float* Memory_RQ, int* D, float* Channel_Out, int* Address_Variablenode, int* Weight_Variablenode)
{
	int offset, num_Variablenode, num_Frames, num_VariablenodeZ;
	float R[15];
	float Add_result;
	int Ad[15];
	int Weight;

	offset = threadIdx.x + blockIdx.x * blockDim.x;			// �̺߳�
	num_Variablenode = offset / Num_Frames_OneTime;		// �����ڵ���ţ�16��֡�ĵ�һ�������ڵ�-16��֡�ĵڶ����ڵ�-��������16��֡�����һ���ڵ㣩
	num_Frames = offset % Num_Frames_OneTime;		// ֡��
	num_VariablenodeZ = num_Variablenode / Z;					// �ֿ�ʽУ������ж�Ӧ���п��,����offset / (Z*Num_Frames_OneTime_define)��1��zά�����z���ڵ㣩
	num_Variablenode = num_Variablenode * Weight_Variablenode[L];//ת������Address_Variablenode_GPU���λ�ã�ÿ�������ڵ��Ӧ�����ӹ�ϵ��Address_Variablenode_GPUÿһ����һ�������ڵ�����е����ӣ�

	

	if (offset < CW_Len * Num_Frames_OneTime)//memory�������ǣ�֡1�����ڵ�1���ӵĽڵ㡪��֡2�ڵ��1���ӵĽڵ㡪��֡3����������������������
	{
		Weight = Weight_Variablenode[num_VariablenodeZ];
		for (int i = 0; i < Weight; i++)
		{
			Ad[i] = Address_Variablenode[num_Variablenode + i] * Num_Frames_OneTime + num_Frames;
		}
		for (int i = 0; i < Weight; i++)
		{
			R[i] = Memory_RQ[(Ad[i])];
		}
		for (int i = 0; i < Weight; i++)
		{
			Add_result += R[i];
		}
		Add_result += Channel_Out[offset];
		D[offset] = (Add_result < 0) ? 1 : 0;//����R����Q�������ڵ�;
		for (int i = 0; i < Weight;i++)
		{
			Memory_RQ[Ad[i]] = Add_result - R[i];

		}
	}
}
/*
* Memory_RQ:���ڴ洢�����ڵ��У��ڵ����ʱ�õ���R��Qֵ
* D:�������
* Weight_Variablenode:�����ڵ�����
* Address_Variablenode:ÿ�������ڵ�����ӦУ��ڵ��memory_rq�ĵ�ַ
*/
__global__ void Variablenode_Shared_Kernel(float* Memory_RQ, int* D, float* Channel_Out, int* Address_Variablenode, int* Weight_Variablenode)
{
	int offset, num_Variablenode, num_Frames, num_VariablenodeZ;
	float R[15];
	float Add_result;
	int Ad[15];
	__shared__ int Weight;

	offset = threadIdx.x + blockIdx.x * blockDim.x;			// �̺߳�
	num_Variablenode = offset / Num_Frames_OneTime;		// �����ڵ���ţ�16��֡�ĵ�һ�������ڵ�-16��֡�ĵڶ����ڵ�-��������16��֡�����һ���ڵ㣩
	num_Frames = offset % Num_Frames_OneTime;		// ֡��
	num_VariablenodeZ = num_Variablenode / Z;					// �ֿ�ʽУ������ж�Ӧ���п��,����offset / (Z*Num_Frames_OneTime_define)��1��zά�����z���ڵ㣩
	num_Variablenode = num_Variablenode * Weight_Variablenode[L];//ת������Address_Variablenode_GPU���λ�ã�ÿ�������ڵ��Ӧ�����ӹ�ϵ��Address_Variablenode_GPUÿһ����һ�������ڵ�����е����ӣ�

	if (threadIdx.x == 0 && num_VariablenodeZ < L)
	{
		Weight = Weight_Variablenode[num_VariablenodeZ];//ֻ��Ҫ��һ��ֵ
	}
	__syncthreads();

	if (offset < CW_Len * Num_Frames_OneTime)//memory�������ǣ�֡1�����ڵ�1���ӵĽڵ㡪��֡2�ڵ��1���ӵĽڵ㡪��֡3����������������������
	{
		// �����ַ��ʱ����Ҫ��������
		for (int i = 0; i < Weight; i++)
		{
			Ad[i] = Address_Variablenode[num_Variablenode + i] * Num_Frames_OneTime + num_Frames;
		}
		for (int i = 0; i < Weight; i++)
		{
			R[i] = Memory_RQ[(Ad[i])];
		}
		for (int i = 0; i < Weight; i++)
		{
			Add_result += R[i];
		}
		Add_result += Channel_Out[offset];
		D[offset] = (Add_result < 0) ? 1 : 0;//����R����Q�������ڵ�;
		for (int i = 0; i < Weight; i++)
		{
			Memory_RQ[Ad[i]] = Add_result - R[i];

		}
	}
}
__global__ void Checknode_Kernel(float* Memory_RQ, int* Weight_Checknode)
{
	int offset, num_Checknode, num_Frames, num_ChecknodeZ;
	__shared__ int Weight;
	float Q[25], Q0[25];
	int Sign[26];
	float MinQ, SubMinQ;
	int Index_minQ;

	offset = threadIdx.x + blockIdx.x * blockDim.x;
	num_Checknode = offset / Num_Frames_OneTime;													// У��ڵ����
	num_Frames = offset % Num_Frames_OneTime;													// ֡��
	num_Frames = num_Frames + num_Checknode * Num_Frames_OneTime * Weight_Checknode[J];	// ��ǰ֡�ĸ�У��ڵ��0��Qֵ�Ĵ�ŵ�ַ
	num_ChecknodeZ = num_Checknode / Z;																	// ��ǰУ��ڵ����ڵ��п��

	

	if (offset < Num_Frames_OneTime * parLen)//q����memory_rq��һ��
	{
		Weight = Weight_Checknode[num_ChecknodeZ];
		for (int i = 0; i < Weight; i++)
		{
			Q[i] = Memory_RQ[num_Frames + i * Num_Frames_OneTime];
		}
		for (int i = 0; i < Weight; i++)
		{
			Sign[i] = (Q[i] < 0) ? -1 : 1;
			Q[i] = (Q[i] < 0) ? -Q[i] : Q[i];
			Q0[i] = Q[i];
		}
		Sign[25] = 1;
		for (int i = 0; i < Weight; i++)
		{
			Sign[25] *= Sign[i];
		}
		sortQ(&MinQ, &SubMinQ, Q, Weight);
		for (int i = 0; i < Weight; i++)
		{
			if (Q0[i] == MinQ)
			{
				Index_minQ = i;
				break;
			}
		}
		for (int i = 0; i < Weight; i++)
		{
			if (i != Index_minQ)
			{
				Memory_RQ[num_Frames + i * Num_Frames_OneTime] = Sign[25] * Sign[i] * MinQ;
			}
			else Memory_RQ[num_Frames + i * Num_Frames_OneTime] = Sign[25] * Sign[i] * SubMinQ;
		}
	}
}
__global__ void Checknode_Shared_Kernel(float* Memory_RQ, int* Weight_Checknode)
{
	int offset, num_Checknode, num_Frames, num_ChecknodeZ;
	__shared__ int Weight;
	float Q[25],Q0[25];
	int Sign[26];
	float MinQ, SubMinQ;
	int Index_minQ;

	offset = threadIdx.x + blockIdx.x * blockDim.x;
	num_Checknode = offset / Num_Frames_OneTime;													// У��ڵ����
	num_Frames = offset % Num_Frames_OneTime;													// ֡��
	num_Frames = num_Frames + num_Checknode * Num_Frames_OneTime * Weight_Checknode[J];	// ��ǰ֡�ĸ�У��ڵ��0��Qֵ�Ĵ�ŵ�ַ
	num_ChecknodeZ = num_Checknode / Z;																	// ��ǰУ��ڵ����ڵ��п��

	if (threadIdx.x == 0 && num_ChecknodeZ < J)		// �õ�0���߳��ҵ����߳̿��������̶߳�Ӧ������
	{
		Weight = Weight_Checknode[num_ChecknodeZ];
	}
	__syncthreads();

	if (offset < Num_Frames_OneTime * parLen)//q����memory_rq��һ��
	{
		for (int i = 0; i < Weight; i++)
		{
			Q[i] = Memory_RQ[num_Frames + i * Num_Frames_OneTime];
		}
		for (int i = 0; i < Weight; i++)
		{
			Sign[i]= (Q[i] < 0) ? -1 : 1;
			Q[i] = (Q[i] < 0) ? -Q[i] : Q[i];
			Q0[i] = Q[i];
		}
		Sign[25] = 1;
		for (int i = 0; i < Weight; i++)
		{
			Sign[25] *= Sign[i];
		}
		sortQ(&MinQ, &SubMinQ, Q, Weight);
		for (int i = 0; i < Weight;i++)
		{
			if (Q0[i] == MinQ)
			{
				Index_minQ = i;
				break;
			}
		}
		for (int i = 0; i < Weight; i++)
		{
			if (i != Index_minQ)
			{
				Memory_RQ[num_Frames + i * Num_Frames_OneTime] = Sign[25] * Sign[i] * MinQ;
			}
			else Memory_RQ[num_Frames + i * Num_Frames_OneTime] = Sign[25] * Sign[i] * SubMinQ;
		}
	}
}

__device__ void sortQ(float* MinQ, float* SubMinQ, float* Q,int Weight)
{
	float tmp;
	for (int i = 0; i < 2; i++) {

		for (int j = 0; j < Weight-1; j++) 
		{

			if (Q[j] < Q[j + 1]) 
			{

				tmp = Q[j];

				Q[j] = Q[j + 1];

				Q[j + 1] = tmp;

			}

		}

	}
	*MinQ = Q[Weight - 1];
	*SubMinQ = Q[Weight - 2];
}