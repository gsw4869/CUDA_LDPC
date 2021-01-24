#include "simu.cuh"

void Get_H(int* H, int* Weight_Checknode, int* Weight_Variablenode)
{
	int index0, index1;
	char file[100];
	FILE* fp_H;
	sprintf(file,"J%d_L%d_Z%d_BlockH.txt",J,L,Z);
	if (NULL == (fp_H = fopen(file, "r")))
	{
		printf("can not open file: %s\n", file);
		getchar();
		exit(0);
	}

	for (index0 = 0; index0 < L *J; index0++)
	{
		fscanf(fp_H, "%d", &index1);
		*(H + index0) = index1;
	}
	fclose(fp_H);

	for (index0 = 0; index0 < J; index0++)
	{
		for (index1 = 0; index1 < L; index1++)
		{
			Weight_Checknode[index0] = (H[index0 * L + index1] != -1) ? Weight_Checknode[index0] + 1 : Weight_Checknode[index0];
			//ͳ�ƽڵ�����
		}
		// �õ��������
		Weight_Checknode[J] = (Weight_Checknode[index0] > Weight_Checknode[J]) ? Weight_Checknode[index0] : Weight_Checknode[J];//��????????????
	}


	for (index0 = 0; index0 < L; index0++)
	{
		for (index1 = 0; index1 < J; index1++)
		{
			Weight_Variablenode[index0] = (H[index1 * L + index0] != -1) ? Weight_Variablenode[index0] + 1 : Weight_Variablenode[index0];
			//ͳ�ƽڵ�����
		}
		// �õ��������
		Weight_Variablenode[L] = (Weight_Variablenode[index0] > Weight_Variablenode[L]) ? Weight_Variablenode[index0] : Weight_Variablenode[L];//???????????????
	}
}


/**
 * @description:���ڵõ�ÿ�������ڵ��Ӧ��У��ڵ���R��Q�����еĵ�ַ
 *				��Ϊֻ��Ҫ��H��1�ĸ��������ڴ�ռ����ڴ�����е�Q����R���� 
 * @param {*} H:H���� 
 *Weight_Checknode:ÿ��У��ڵ������ 
 *Weight_Variablenode:ÿ�������ڵ������ 
 *Address_Variablenode:ÿ�������ڵ�����ӦУ��ڵ���R��Q�еĵ�ַ
 * У��ڵ㲻��Ҫ����ΪУ��ڵ��Ӧ�ľ���ÿһ��memory_rq����ַ������һ��� 
 * @return {*}
 */
void Transform_H(int* H, int* Weight_Checknode, int* Weight_Variablenode, int* Address_Variablenode)
{
	int index0, index1, index2, index3, index4, position;
	for (index0 = 0; index0 < L; index0++)		// index0Ϊ��ǰ������
	{
		index2 = 0;
		for (index1 = 0; index1 < J; index1++)	// index1Ϊ��ǰ������
		{
			if (H[index1 * L + index0] != -1)
			{
				position = 0;	// �����ڵ�������У��ڵ���,�ñ����ڵ�����λ��(ÿһ��1��Ӧ��λ��)
				for (index3 = 0; index3 < index0; index3++)
				{
					position = (H[index1 * L + index3] != -1) ? position + 1 : position;//ÿһ�еڼ����ȫ������
				}
				for (index3 = 0; index3 < Z; index3++)//index3(��)��ÿ��һ����ͳ����һ�������ڵ�����ӹ�ϵ
				{
					index4 = (((Z - H[index1 * L + index0]) % Z + index3) >= Z) ? (Z - H[index1 * L + index0]) % Z + index3 - Z : index3;
					//index4:zά�����ÿһ�е�1��zά������ڼ��У��������ʽ�ӣ�
					Address_Variablenode[(index0 * Z + index3) * Weight_Variablenode[L] + index2] = (index1 * Z + index4) * Weight_Checknode[J] + position;//��¼ÿ�������ڵ㣨ÿһ�У�����Щ�����ӣ����Ϊ���к��У�
				}
				index2++;//index2��ÿһ�еڼ����ȫ������
			}
		}
	}
}