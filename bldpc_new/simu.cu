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
			//统计节点重量
		}
		// 得到最大重量
		Weight_Checknode[J] = (Weight_Checknode[index0] > Weight_Checknode[J]) ? Weight_Checknode[index0] : Weight_Checknode[J];//У????????????
	}


	for (index0 = 0; index0 < L; index0++)
	{
		for (index1 = 0; index1 < J; index1++)
		{
			Weight_Variablenode[index0] = (H[index1 * L + index0] != -1) ? Weight_Variablenode[index0] + 1 : Weight_Variablenode[index0];
			//统计节点重量
		}
		// 得到最大重量
		Weight_Variablenode[L] = (Weight_Variablenode[index0] > Weight_Variablenode[L]) ? Weight_Variablenode[index0] : Weight_Variablenode[L];//???????????????
	}
}


/**
 * @description:用于得到每个变量节点对应的校验节点在R、Q数组中的地址
 *				因为只需要“H中1的个数”的内存空间用于存放所有的Q或者R即可 
 * @param {*} H:H矩阵 
 *Weight_Checknode:每个校验节点的重量 
 *Weight_Variablenode:每个变量节点的重量 
 *Address_Variablenode:每个变量节点所对应校验节点在R、Q中的地址
 * 校验节点不需要是因为校验节点对应的就是每一行memory_rq，地址是连在一起的 
 * @return {*}
 */
void Transform_H(int* H, int* Weight_Checknode, int* Weight_Variablenode, int* Address_Variablenode)
{
	int index0, index1, index2, index3, index4, position;
	for (index0 = 0; index0 < L; index0++)		// index0为当前所在列
	{
		index2 = 0;
		for (index1 = 0; index1 < J; index1++)	// index1为当前所在行
		{
			if (H[index1 * L + index0] != -1)
			{
				position = 0;	// 变量节点相连的校验节点中,该变量节点的相对位置(每一行1对应的位置)
				for (index3 = 0; index3 < index0; index3++)
				{
					position = (H[index1 * L + index3] != -1) ? position + 1 : position;//每一行第几块非全零矩阵块
				}
				for (index3 = 0; index3 < Z; index3++)//index3(列)，每加一就是统计下一个变量节点的连接关系
				{
					index4 = (((Z - H[index1 * L + index0]) % Z + index3) >= Z) ? (Z - H[index1 * L + index0]) % Z + index3 - Z : index3;
					//index4:z维方块里，每一列的1在z维方块里第几行（结合下面式子）
					Address_Variablenode[(index0 * Z + index3) * Weight_Variablenode[L] + index2] = (index1 * Z + index4) * Weight_Checknode[J] + position;//记录每个变量节点（每一列）和哪些点连接（序号为先行后列）
				}
				index2++;//index2是每一列第几块非全零矩阵块
			}
		}
	}
}