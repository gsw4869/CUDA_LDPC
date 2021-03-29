#include "Simulation.cuh"
#include "LDPC_Encoder.cuh"
#include "LDPC_Decoder.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>


int **malloc_2(int xDim, int yDim) {
    int **a = (int **)malloc(xDim * sizeof(int *));
    a[0] = (int *)malloc(xDim * yDim * sizeof(int));
    memset(a[0], 0, xDim * yDim * sizeof(int));
    for (int i = 1; i < xDim; i++) {
        a[i] = a[i - 1] + yDim;
    }
    assert(a != NULL);
    return a;
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