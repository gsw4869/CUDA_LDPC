#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "decoder.cuh"

/**
 * @description:排序函数，得到最小的和次小的
 * @param {*}MinQ:最小的 SubMinQ:次小的
 * @return {*}
 */
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

/**
 * @description:变量节点计算 
 * @param {*}
 * @return {*}
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