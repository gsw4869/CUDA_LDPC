#include "Decode_GPU.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>

__device__ int GFAdd_GPU(int ele1, int ele2, unsigned *TableAdd_GPU)
{
    return TableAdd_GPU[GFQ * ele1 + ele2];
}

__device__ int GFMultiply_GPU(int ele1, int ele2, unsigned *TableMultiply_GPU)
{
    return TableMultiply_GPU[GFQ * ele1 + ele2];
}

__device__ int GFInverse_GPU(int ele, unsigned *TableInverse_GPU)
{
    if (ele == 0)
    {
        printf("Div 0 Error!\n");
    }
    return TableInverse_GPU[ele];
}

__device__ int index_in_VN_GPU(int *Checknode_linkVNs, int Checknode_num, int index_in_linkVNs, int *Variablenode_linkCNs)
{
    for (int i = 0; i < maxdv; i++)
    {
        if (Variablenode_linkCNs[maxdv * Checknode_linkVNs[maxdc * Checknode_num + index_in_linkVNs] + i] == Checknode_num)
        {
            return i;
        }
    }
    printf("index_in_VN_GPU error\n");
}

int Decoding_EMS_GPU(LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput)
{

    for (int col = 0; col < H->Variablenode_num; col++)
    {
        for (int d = 0; d < Variablenode[col].weight; d++)
        {
            for (int q = 0; q < H->GF; q++)
            {
                Variablenode[col].Entr_v2c[d][q] = Variablenode[col].L_ch[q];
            }
        }
    }
    for (int row = 0; row < H->Checknode_num; row++)
    {
        for (int d = 0; d < Checknode[row].weight; d++)
        {
            for (int q = 0; q < H->GF - 1; q++)
            {
                Checknode[row].L_c2v[d][q] = 0;
            }
        }
    }

    int iter_number = 0;
    bool decode_correct = true;
    while (iter_number++ < maxIT)
    {
        // printf("it_time: %d\n",iter_number);
        for (int col = 0; col < H->Variablenode_num; col++)
        {
            for (int d = 0; d < Variablenode[col].weight; d++)
            {
                for (int q = 0; q < H->GF - 1; q++)
                {
                    Variablenode[col].LLR[q] = Variablenode[col].L_ch[q];
                }
            }
        }
        for (int col = 0; col < H->Variablenode_num; col++)
        {
            for (int d = 0; d < Variablenode[col].weight; d++)
            {
                for (int q = 0; q < H->GF - 1; q++)
                {
                    Variablenode[col].LLR[q] += Checknode[Variablenode[col].linkCNs[d]].L_c2v[index_in_CN(Variablenode, col, d, Checknode)][q];
                }
            }
            DecodeOutput[col] = DecideLLRVector(Variablenode[col].LLR, H->GF);
            // printf("%d ", DecodeOutput[col]);
        }
        // printf("\n");

        decode_correct = true;
        int sum_temp = 0;
        for (int row = 0; row < H->Checknode_num; row++)
        {
            for (int i = 0; i < Checknode[row].weight; i++)
            {
                sum_temp = GFAdd(sum_temp, GFMultiply(DecodeOutput[Checknode[row].linkVNs[i]], Checknode[row].linkVNs_GF[i]));
            }
            if (sum_temp)
            {
                decode_correct = false;
                break;
            }
        }
        if (decode_correct)
        {
            return 1;
        }

        // message from var to check
        for (int col = 0; col < H->Variablenode_num; col++)
        {
            for (int dv = 0; dv < Variablenode[col].weight; dv++)
            {
                for (int q = 0; q < H->GF - 1; q++)
                {
                    Variablenode[col].Entr_v2c[dv][q] = Variablenode[col].LLR[q] - Checknode[Variablenode[col].linkCNs[dv]].L_c2v[index_in_CN(Variablenode, col, dv, Checknode)][q];
                }
            }
        }

        int *index = (int *)malloc((H->GF) * sizeof(int));
        for (int col = 0; col < H->Variablenode_num; col++)
        {
            memcpy(Variablenode[col].sort_L_v2c[0], Variablenode[col].Entr_v2c[0], Variablenode[col].weight * H->GF * sizeof(float));

            for (int dv = 0; dv < Variablenode[col].weight; dv++)
            {
                for (int i = 0; i < H->GF - 1; i++)
                {
                    index[i] = i + 1;
                }
                index[H->GF - 1] = 0;
                SortLLRVector(H->GF, Variablenode[col].sort_L_v2c[dv], index);
                for (int i = 0; i < H->GF; i++)
                {
                    Variablenode[col].sort_Entr_v2c[dv][i] = index[i];
                }
            }
        }

        float *EMS_L_c2v = (float *)malloc(H->GF * sizeof(float));

        // message from check to var
        for (int row = 0; row < H->Checknode_num; row++)
        {
        }
        free(EMS_L_c2v);
    }
    return 0;
}

/*
Checknode_weight:每一个校验节点的重量
L_c2v:Q个信息，Q个信息，Q个信息，一共校验节点数量*Q个
Variblenode_linkCNs:最大重量dv，每dv个元素代表连接的dv个校验节点的序号
Checknode_linkVNS:最大重量dc，每dc个元素代表连接的dc个变量节点的序号
Checknode_linkVNS_GF:最大重量dc，每dc个元素代表连接的dc个变量节点的多元域值
sort_Entr_v2c:每个变量节点重量dv，q,q,q一共dv个，然后再乘以变量节点个数[变量节点个数][变量节点重量][q]
sort_L_v2c:和sort_Entr_v2c对应的LLR
*/
__global__ void Checknode_EMS(unsigned *TableMultiply_GPU, unsigned *TableAdd_GPU, int EMS_Nm, int EMS_Nc, int *Checknode_weight, float *L_c2v, int *Variblenode_linkCNs, int *Checknode_linkVNs, int *Checknode_linkVNs_GF, int *sort_Entr_v2c, int *sort_L_v2c)
{
    int offset;
    offset = threadIdx.x + blockDim.x * blockIdx.x;
    float EMS_L_c2v[GFQ];
    for (int dc = 0; dc < Checknode_weight[offset]; dc++)
    {
        // reset the sum store vector to the munimum
        for (int q = 0; q < GFQ; q++)
        {
            EMS_L_c2v[q] = -DBL_MAX;
        }

        // recursly exhaustly
        int sumNonele, diff;
        float sumNonLLR;
        // conf(q, 1)
        sumNonele = 0;
        sumNonLLR = 0;
        diff = 0;
        ConstructConf_GPU(TableMultiply_GPU, TableAdd_GPU, GFQ, 1, sumNonele, sumNonLLR, diff, 0, dc, Checknode_weight[offset] - 1, offset, EMS_L_c2v, Variblenode_linkCNs, Checknode_linkVNs, Checknode_linkVNs, sort_Entr_v2c, sort_L_v2c);

        // conf(nm, nc)
        sumNonele = 0;
        sumNonLLR = 0;
        diff = 0;
        ConstructConf_GPU(TableMultiply_GPU, TableAdd_GPU, EMS_Nm, EMS_Nc, sumNonele, sumNonLLR, diff, 0, dc, Checknode_weight[offset] - 1, offset, EMS_L_c2v, Variblenode_linkCNs, Checknode_linkVNs, Checknode_linkVNs, sort_Entr_v2c, sort_L_v2c);

        // calculate each c2v LLR
        // int v = 0;
        // for (int k = 1; k < GFQ; k++)
        // {
        // 	v = GFMultiply_GPU(k, Checknode[row].linkVNs_GF[dc]);
        // 	Checknode[row].L_c2v[dc][k - 1] = (EMS_L_c2v[v] - EMS_L_c2v[0]) / 1.2;
        // }
    }
}
__device__ int ConstructConf_GPU(unsigned *TableMultiply_GPU, unsigned *TableAdd_GPU, int Nm, int Nc, int &sumNonele, float &sumNonLLR, int &diff, int begin, int except, int end, int row, float *EMS_L_c2v, int *Variblenode_linkCNs, int *Checknode_linkVNs, int *Checknode_linkVNs_GF, int *sort_Entr_v2c, int *sort_L_v2c)
{
    int index;
    if (begin > end)
    {
        if (sumNonLLR > EMS_L_c2v[sumNonele])
        {
            EMS_L_c2v[sumNonele] = sumNonLLR;
        }
    }
    else if (begin == except)
    {
        ConstructConf_GPU(TableMultiply_GPU, TableAdd_GPU, Nm, Nc, sumNonele, sumNonLLR, diff, begin + 1, except, end, row, EMS_L_c2v, Variblenode_linkCNs, Checknode_linkVNs, Checknode_linkVNs, sort_Entr_v2c, sort_L_v2c);
        return 0;
    }
    else
    {
        index = index_in_VN_GPU(Checknode_linkVNs, row, begin, Variblenode_linkCNs);
        for (int k = 0; k < Nm; k++)
        {
            sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k], Checknode_linkVNs_GF[row * maxdc + begin], TableMultiply_GPU), sumNonele, TableAdd_GPU);
            sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k];
            diff += (k != 0) ? 1 : 0;
            if (diff <= Nc)
            {
                ConstructConf_GPU(TableMultiply_GPU, TableAdd_GPU, Nm, Nc, sumNonele, sumNonLLR, diff, begin + 1, except, end, row, EMS_L_c2v, Variblenode_linkCNs, Checknode_linkVNs, Checknode_linkVNs, sort_Entr_v2c, sort_L_v2c);
                sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k], Checknode_linkVNs_GF[row * maxdc + begin], TableMultiply_GPU), sumNonele, TableAdd_GPU);
                sumNonLLR = sumNonLLR - sort_L_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k];
                diff -= (k != 0) ? 1 : 0;
            }
            else
            {
                sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k], Checknode_linkVNs_GF[row * maxdc + begin], TableMultiply_GPU), sumNonele, TableAdd_GPU);
                sumNonLLR = sumNonLLR - sort_L_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k];
                diff -= (k != 0) ? 1 : 0;
                break;
            }
        }
    }
    return 0;
}
