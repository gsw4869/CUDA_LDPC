#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "decoder.cuh"

/**
 * @description:���������õ���С�ĺʹ�С��
 * @param {*}MinQ:��С�� SubMinQ:��С��
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
 * @description:�����ڵ���� 
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