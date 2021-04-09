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
#include "GF.cuh"

int index_in_VN(CN* Checknode,int CNnum,int index_in_linkVNS,VN* Variablenode)
{
    for(int i=0;i<Variablenode[Checknode[CNnum].linkVNs[index_in_linkVNS]].weight;i++)
    {
        if(Variablenode[Checknode[CNnum].linkVNs[index_in_linkVNS]].linkCNs[i]==CNnum)
        {
            return i;
        }
    }
    printf("index_in_VN error\n");
    exit(0);
} 

int index_in_CN(VN* Variablenode,int VNnum,int index_in_linkCNS,CN* Checknode)
{
    for(int i=0;i<Checknode[Variablenode[VNnum].linkCNs[index_in_linkCNS]].weight;i++)
    {
        if(Checknode[Variablenode[VNnum].linkCNs[index_in_linkCNS]].linkVNs[i]==VNnum)
        {
            return i;
        }
    }
    printf("index_in_CN error\n");
    exit(0);
} 


void Demodulate(LDPCCode* H,AWGNChannel* AWGN,CComplex* CONSTELLATION,VN* Variablenode,CComplex* CComplex_sym_Channelout)
{
    int p_i = 0;
    for(int s = 0; s < H->Variablenode_num; s ++)
    {
            for(int q = 1; q < H->GF; q ++)
            {
                Variablenode[s].LLR[q - 1] = ( (2 * CComplex_sym_Channelout[s - p_i].Real - CONSTELLATION[0].Real - CONSTELLATION[q].Real ) * (CONSTELLATION[q].Real - CONSTELLATION[0].Real) 
                    + (2 * CComplex_sym_Channelout[s - p_i].Image - CONSTELLATION[0].Image - CONSTELLATION[q].Image ) * (CONSTELLATION[q].Image - CONSTELLATION[0].Image) ) / (2 * AWGN->sigma * AWGN->sigma);
            }
    }
}

int ConstructConf(CN *Checknode,VN *Variablenode,int Nm, int Nc, int& sumNonele, double& sumNonLLR, int& diff, int begin, int except, int end, int row)
{
    int index=index_in_VN(Checknode,row,except,Variablenode);
	if (begin > end)
	{
		if (sumNonLLR > Checknode[row].L_c2v[except][sumNonele])
		{
			Checknode[row].L_c2v[except][sumNonele] = sumNonLLR;
		}
	}
	else if (begin == except)
	{
		ConstructConf(Checknode, Variablenode, Nm, Nc, sumNonele, sumNonLLR, diff, begin + 1, except, end, row);
		return 0;
	}
	else
	{
		for (int k = 0; k < Nm; k++)
		{
			sumNonele = GFAdd(GFMultiply(Variablenode[Checknode[row].linkVNs[begin]].sort_Entr_v2c[index][begin], Checknode->linkVNs_GF[begin]), sumNonele);
			sumNonLLR = sumNonLLR + Variablenode[Checknode[row].linkVNs[begin]].sort_Entr_v2c[index][begin];
			diff += (k != 0) ? 1 : 0;
			if (diff <= Nc)
			{
				ConstructConf(Checknode, Variablenode, Nm, Nc, sumNonele, sumNonLLR, diff, begin + 1, except, end, row);
				sumNonele = GFAdd(GFMultiply(Variablenode[Checknode[row].linkVNs[begin]].sort_Entr_v2c[index][begin], Checknode->linkVNs_GF[begin]), sumNonele);
				sumNonLLR = sumNonLLR - Variablenode[Checknode[row].linkVNs[begin]].sort_Entr_v2c[index][begin];
				diff -= (k != 0) ? 1 : 0;
			}
			else
			{
				sumNonele = GFAdd(GFMultiply(Variablenode[Checknode[row].linkVNs[begin]].sort_Entr_v2c[index][begin], Checknode->linkVNs_GF[begin]), sumNonele);
				sumNonLLR = sumNonLLR - Variablenode[Checknode[row].linkVNs[begin]].sort_Entr_v2c[index][begin];
				diff -= (k != 0) ? 1 : 0;
				break;
			}
		}
	}
	return 0;
}
