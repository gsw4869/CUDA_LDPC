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
* D:一码输出
* Channel_Out:经过AWGN信道的信号
* Weight_Checknode:校验节点重量
* Weight_Variablenode:变量节点重量
* Address_Variablenode:每个变量节点所对应校验节点的memory_rq的地址
* LDPC:迭代次数
*/
void LDPC_Decoder_GPU(int* D, float* Channel_Out, cudaDeviceProp prop, int* Address_Variablenode, int* Weight_Checknode, int* Weight_Variablenode, LDPCCode *LDPC)
{
	cudaError_t cudaStatus;
	int index0, index1, Length;
	int ThreadPerBlock, Num_Block;
	float* Memory_RQ;
	int* Weight_Checknode_GPU, *Weight_Variablenode_GPU;
	int* D_GPU;
	cudaEvent_t GPU_start;			// GPU速率统计参数
	cudaEvent_t GPU_stop;
	cudaEventCreate(&GPU_start);
	cudaEventCreate(&GPU_stop);

	Length = (Message_CW == 0) ? msgLen : CW_Len;

	cudaStatus = cudaMalloc((void**)&Memory_RQ, parLen * Weight_Checknode[J] * Num_Frames_OneTime * sizeof(float));	// 锟斤拷锟斤拷锟斤拷GPU锟斤拷global memory锟斤拷
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc Memory_RQ in LDPC_Decoder_GPU on device, exit!\n");
		//getch();
		exit(0);
	}
	cudaStatus = cudaMalloc((void**)&D_GPU, (CW_Len + 1) * Num_Frames_OneTime * sizeof(int));		// 锟斤拷锟斤拷锟斤拷GPU锟斤拷global memory锟斤拷
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc D_GPU in LDPC_Decoder_GPU on device, exit!\n");
		//getch();
		exit(0);
	}
	cudaStatus = cudaMalloc((void**)&Weight_Checknode_GPU, (J + 1) * sizeof(int));		// 锟斤拷锟斤拷锟斤拷GPU锟斤拷global memory锟斤拷
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc Weight_Checknode_GPU in LDPC_Decoder_GPU on device, exit!\n");
		//getch();
		exit(0);
	}
	cudaStatus = cudaMalloc((void**)&Weight_Variablenode_GPU, (L + 1) * sizeof(int));		// 锟斤拷锟斤拷锟斤拷GPU锟斤拷global memory锟斤拷
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc Weight_Checknode_GPU in LDPC_Decoder_GPU on device, exit!\n");
		//getch();
		exit(0);
	}
	cudaStatus = cudaMemcpy(Weight_Checknode_GPU, Weight_Checknode, (J + 1) * sizeof(int), cudaMemcpyHostToDevice);//J锟斤拷 L锟叫ｏ拷锟斤拷锟揭伙拷锟轿?锟斤拷锟斤拷锟斤拷锟?
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

	// 初始化
	cudaStatus = cudaMemset(Memory_RQ, 0, parLen * Weight_Checknode[J] * Num_Frames_OneTime * sizeof(float));	// 锟芥储锟斤拷锟斤拷锟斤拷,为锟斤拷一锟轿碉拷锟斤拷锟斤拷准锟斤拷,parlen校锟斤拷位锟斤拷锟饺ｏ拷J*Z锟斤拷
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
		if ((Z * Num_Frames_OneTime) % prop.maxThreadsPerBlock == 0)	// 此时根据prop.maxThreadsPerBlock分成的每个线程块均属于一个列块或者行块的节点,一个块之间可以共享重量
		{
			cudaThreadSynchronize();
			ThreadPerBlock = prop.maxThreadsPerBlock;
			Num_Block = ((Num_Frames_OneTime * CW_Len) % ThreadPerBlock == 0) ? (Num_Frames_OneTime * CW_Len) / ThreadPerBlock : ((Num_Frames_OneTime * CW_Len) / ThreadPerBlock) + 1;
			Variablenode_Shared_Kernel<<<Num_Block, ThreadPerBlock>>>(Memory_RQ, D_GPU, Channel_Out, Address_Variablenode, Weight_Variablenode_GPU);//变量节点计算，得到L

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
			Variablenode_Kernel<<<Num_Block, ThreadPerBlock>>>(Memory_RQ, D_GPU, Channel_Out, Address_Variablenode, Weight_Variablenode_GPU);//变量节点计算，得到L

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
				D[index1 + CW_Len * Num_Frames_OneTime] += D[index0 * Num_Frames_OneTime + index1];//把每一帧的所有字节加起来求和
			}
		}
		index0 = 0;
		for (index1 = 0; index1 < Num_Frames_OneTime; index1++)
		{
			D[index1 + CW_Len * Num_Frames_OneTime] = (D[index1 + CW_Len * Num_Frames_OneTime] == 0) ? 1 : 0;//全零序列加起来为全0，正确结果为1
			index0 += D[index1 + CW_Len * Num_Frames_OneTime];//统计对的帧数
		}
		if (index0 == Num_Frames_OneTime)
		{
			break;//所有帧都解码完
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
* Memory_RQ:用于存储变量节点和校验节点计算时得到的R与Q值
* D:译码输出
* Weight_Variablenode:变量节点重量
* Address_Variablenode:每个变量节点所对应校验节点的memory_rq的地址
*/
__global__ void Variablenode_Kernel(float* Memory_RQ, int* D, float* Channel_Out, int* Address_Variablenode, int* Weight_Variablenode)
{
	int offset, num_Variablenode, num_Frames, num_VariablenodeZ;
	float R[15];
	float Add_result;
	int Ad[15];
	int Weight;

	offset = threadIdx.x + blockIdx.x * blockDim.x;			// 线程号
	num_Variablenode = offset / Num_Frames_OneTime;		// 变量节点序号（16个帧的第一个变量节点-16个帧的第二个节点-————16个帧的最后一个节点）
	num_Frames = offset % Num_Frames_OneTime;		// 帧号
	num_VariablenodeZ = num_Variablenode / Z;					// 分块式校验矩阵中对应的列块号,等于offset / (Z*Num_Frames_OneTime_define)（1个z维矩阵块z个节点）
	num_Variablenode = num_Variablenode * Weight_Variablenode[L];//转换到在Address_Variablenode_GPU里的位置（每个变量节点对应的连接关系，Address_Variablenode_GPU每一块是一个变量节点和所有的连接）

	

	if (offset < CW_Len * Num_Frames_OneTime)//memory数组里是（帧1变量节点1连接的节点——帧2节点的1连接的节点——帧3——————————）
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
		D[offset] = (Add_result < 0) ? 1 : 0;//根据R计算Q，变量节点;
		for (int i = 0; i < Weight;i++)
		{
			Memory_RQ[Ad[i]] = Add_result - R[i];

		}
	}
}
/*
* Memory_RQ:用于存储变量节点和校验节点计算时得到的R与Q值
* D:译码输出
* Weight_Variablenode:变量节点重量
* Address_Variablenode:每个变量节点所对应校验节点的memory_rq的地址
*/
__global__ void Variablenode_Shared_Kernel(float* Memory_RQ, int* D, float* Channel_Out, int* Address_Variablenode, int* Weight_Variablenode)
{
	int offset, num_Variablenode, num_Frames, num_VariablenodeZ;
	float R[15];
	float Add_result;
	int Ad[15];
	__shared__ int Weight;

	offset = threadIdx.x + blockIdx.x * blockDim.x;			// 线程号
	num_Variablenode = offset / Num_Frames_OneTime;		// 变量节点序号（16个帧的第一个变量节点-16个帧的第二个节点-————16个帧的最后一个节点）
	num_Frames = offset % Num_Frames_OneTime;		// 帧号
	num_VariablenodeZ = num_Variablenode / Z;					// 分块式校验矩阵中对应的列块号,等于offset / (Z*Num_Frames_OneTime_define)（1个z维矩阵块z个节点）
	num_Variablenode = num_Variablenode * Weight_Variablenode[L];//转换到在Address_Variablenode_GPU里的位置（每个变量节点对应的连接关系，Address_Variablenode_GPU每一块是一个变量节点和所有的连接）

	if (threadIdx.x == 0 && num_VariablenodeZ < L)
	{
		Weight = Weight_Variablenode[num_VariablenodeZ];//只需要赋一次值
	}
	__syncthreads();

	if (offset < CW_Len * Num_Frames_OneTime)//memory数组里是（帧1变量节点1连接的节点——帧2节点的1连接的节点——帧3——————————）
	{
		// 计算地址的时候不需要根据重量
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
		D[offset] = (Add_result < 0) ? 1 : 0;//根据R计算Q，变量节点;
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
	num_Checknode = offset / Num_Frames_OneTime;													// 校验节点序号
	num_Frames = offset % Num_Frames_OneTime;													// 帧号
	num_Frames = num_Frames + num_Checknode * Num_Frames_OneTime * Weight_Checknode[J];	// 当前帧的该校验节点第0个Q值的存放地址
	num_ChecknodeZ = num_Checknode / Z;																	// 当前校验节点所在的列块号

	

	if (offset < Num_Frames_OneTime * parLen)//q就是memory_rq的一行
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
	num_Checknode = offset / Num_Frames_OneTime;													// 校验节点序号
	num_Frames = offset % Num_Frames_OneTime;													// 帧号
	num_Frames = num_Frames + num_Checknode * Num_Frames_OneTime * Weight_Checknode[J];	// 当前帧的该校验节点第0个Q值的存放地址
	num_ChecknodeZ = num_Checknode / Z;																	// 当前校验节点所在的列块号

	if (threadIdx.x == 0 && num_ChecknodeZ < J)		// 用第0个线程找到该线程块中所有线程对应的重量
	{
		Weight = Weight_Checknode[num_ChecknodeZ];
	}
	__syncthreads();

	if (offset < Num_Frames_OneTime * parLen)//q就是memory_rq的一行
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