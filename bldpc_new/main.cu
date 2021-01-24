#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "includelib.cuh"

int main()
{	
	int H[J*L];
	int Weight_Checknode[J+1];
	int Weight_Variablenode[L+1];
	Get_H(H,Weight_Checknode,Weight_Variablenode);
	// int seed1[3]={173,173,173};
	// int code[CW_Len];
	// float res[CW_Len];
	// AWGNChannel* AWGN;
	// AWGN = (AWGNChannel*)malloc(sizeof(AWGNChannel));
	// AWGN->seed[0]=173;
	// AWGN->seed[1]=173;
	// AWGN->seed[2]=173;
	// AWGN->sigma=10;
	// for(int i=0;i<CW_Len;i++)
	// {
	// 	code[i]=rand()%2;
	// }
	// AWGNChannel_SIMU(AWGN,res,code);
	for(int i=0;i<J*L;i++)
	{
		printf("%d ",H[i]);
	}
	return 0;
}