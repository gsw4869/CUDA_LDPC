#include "Decode_GPU.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>

__device__ int GFAdd_GPU(int ele1, int ele2, unsigned *TableAdd_GPU)
{
    if (ele1 >= GFQ | ele2 >= GFQ)
    {
        printf("error");
    }
    return TableAdd_GPU[GFQ * ele1 + ele2];
}

__device__ int GFMultiply_GPU(int ele1, int ele2, unsigned *TableMultiply_GPU)
{
    if (ele1 >= GFQ | ele2 >= GFQ)
    {
        printf("error");
    }
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

int Decoding_EMS_GPU(LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput, unsigned *TableMultiply_GPU, unsigned *TableAdd_GPU, int *Checknode_weight, int *Variablenode_linkCNs, int *Checknode_linkVNs, int *Checknode_linkVNs_GF)
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
    int *sort_Entr_v2c_temp = (int *)malloc(H->Variablenode_num * maxdv * GFQ * sizeof(int));
    memset(sort_Entr_v2c_temp, 0, H->Variablenode_num * maxdv * GFQ * sizeof(int));
    int *sort_Entr_v2c;
    cudaMalloc((void **)&sort_Entr_v2c, H->Variablenode_num * maxdv * GFQ * sizeof(int));

    float *sort_L_v2c_temp = (float *)malloc(H->Variablenode_num * maxdv * GFQ * sizeof(float));
    memset(sort_L_v2c_temp, 0, H->Variablenode_num * maxdv * GFQ * sizeof(float));
    float *sort_L_v2c;
    cudaMalloc((void **)&sort_L_v2c, H->Variablenode_num * maxdv * GFQ * sizeof(float));

    float *Checknode_L_c2v_temp = (float *)malloc(H->Checknode_num * maxdc * GFQ * sizeof(float));
    memset(Checknode_L_c2v_temp, 0, H->Checknode_num * maxdc * GFQ * sizeof(float));
    float *Checknode_L_c2v;
    cudaMalloc((void **)&Checknode_L_c2v, H->Checknode_num * maxdc * GFQ * sizeof(float));

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

                    sort_Entr_v2c_temp[col * maxdv * GFQ + dv * GFQ + i] = index[i];
                    sort_L_v2c_temp[col * maxdv * GFQ + dv * GFQ + i] = Variablenode[col].sort_L_v2c[dv][i];
                }
            }
        }
        cudaError_t cudaStatus;
        cudaStatus = cudaMemcpy(sort_Entr_v2c, sort_Entr_v2c_temp, H->Variablenode_num * maxdv * GFQ * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("Cannot copy sort_Entr_v2c\n");
            exit(0);
        }
        cudaStatus = cudaMemcpy(sort_L_v2c, sort_L_v2c_temp, H->Variablenode_num * maxdv * GFQ * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            printf("Cannot copy sort_L_v2c\n");
            exit(0);
        }
        // // message from check to var

        Checknode_EMS<<<((H->Checknode_num % 128) ? (H->Checknode_num / 128 + 1) : (H->Checknode_num / 128)), 128>>>(TableMultiply_GPU, TableAdd_GPU, EMS_Nm, EMS_Nc, Checknode_weight, Variablenode_linkCNs, Checknode_linkVNs, Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c, Checknode_L_c2v, H->Checknode_num);
        // Checknode_EMS<<<1, 1>>>(TableMultiply_GPU, TableAdd_GPU, EMS_Nm, EMS_Nc, Checknode_weight, Variablenode_linkCNs, Checknode_linkVNs, Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c, Checknode_L_c2v, H->Checknode_num);

        cudaStatus = cudaMemcpy(Checknode_L_c2v_temp, Checknode_L_c2v, H->Checknode_num * maxdc * GFQ * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("Cannot copy Checknode_L_c2v\n");
            exit(0);
        }

        for (int i = 0; i < H->Checknode_num; i++)
        {
            for (int j = 0; j < Checknode[i].weight; j++)
            {
                for (int q = 0; q < GFQ - 1; q++)
                {
                    Checknode[i].L_c2v[j][q] = Checknode_L_c2v_temp[i * maxdc * GFQ + j * GFQ + q];
                }
            }
        }
    }
    cudaFree(sort_Entr_v2c);
    cudaFree(sort_L_v2c);
    cudaFree(Checknode_L_c2v);
    free(sort_Entr_v2c_temp);
    free(sort_L_v2c_temp);
    free(Checknode_L_c2v_temp);
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
Checknode_L_c2v:每个校验节点重量dc，q一共dc个，然后再乘以变量节点个数[校验节点个数][校验节点重量][q]
*/
__global__ void Checknode_EMS(unsigned *TableMultiply_GPU, unsigned *TableAdd_GPU, int EMS_Nm, int EMS_Nc, int *Checknode_weight, int *Variblenode_linkCNs, int *Checknode_linkVNs, int *Checknode_linkVNs_GF, int *sort_Entr_v2c, float *sort_L_v2c, float *Checknode_L_c2v, int Checknode_num)
{
    int offset;
    offset = threadIdx.x + blockDim.x * blockIdx.x;
    if (offset < Checknode_num)
    {
        float EMS_L_c2v[GFQ];
        for (int dc = 0; dc < maxdc; dc++)
        {
            if (dc < Checknode_weight[offset])
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
                ConstructConf_GPU(TableMultiply_GPU, TableAdd_GPU, GFQ, 1, sumNonele, sumNonLLR, diff, 0, dc, Checknode_weight[offset] - 1, offset, EMS_L_c2v, Variblenode_linkCNs, Checknode_linkVNs, Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c);

                // conf(nm, nc)
                // sumNonele = 0;
                // sumNonLLR = 0;
                // diff = 0;
                // ConstructConf_GPU(TableMultiply_GPU, TableAdd_GPU, EMS_Nm, EMS_Nc, sumNonele, sumNonLLR, diff, 0, dc, Checknode_weight[offset] - 1, offset, EMS_L_c2v, Variblenode_linkCNs, Checknode_linkVNs, Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c);

                // calculate each c2v LLR
                int v = 0;
                Checknode_L_c2v[offset * maxdc * GFQ + dc * GFQ + GFQ - 1] = 0;
                for (int k = 1; k < GFQ; k++)
                {
                    v = GFMultiply_GPU(k, Checknode_linkVNs_GF[offset * maxdc + dc], TableMultiply_GPU);
                    Checknode_L_c2v[offset * maxdc * GFQ + dc * GFQ + k - 1] = (EMS_L_c2v[v] - EMS_L_c2v[0]) / 1.2;
                }
            }
            else
            {
                for (int k = 0; k < GFQ; k++)
                {
                    Checknode_L_c2v[offset * maxdc * GFQ + dc * GFQ + k] = 0;
                }
            }
        }
    }
}
__device__ int ConstructConf_GPU(unsigned *TableMultiply_GPU, unsigned *TableAdd_GPU, int Nm, int Nc, int &sumNonele, float &sumNonLLR, int &diff, int begin, int except, int end, int row, float *EMS_L_c2v, int *Variblenode_linkCNs, int *Checknode_linkVNs, int *Checknode_linkVNs_GF, int *sort_Entr_v2c, float *sort_L_v2c)
{
    // if (begin > end)
    // {
    //     if (sumNonLLR > EMS_L_c2v[sumNonele])
    //     {
    //         EMS_L_c2v[sumNonele] = sumNonLLR;
    //     }
    // }
    // else if (begin == except)
    // {
    //     ConstructConf_GPU(TableMultiply_GPU, TableAdd_GPU, Nm, Nc, sumNonele, sumNonLLR, diff, begin + 1, except, end, row, EMS_L_c2v, Variblenode_linkCNs, Checknode_linkVNs, Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c);
    //     return 0;
    // }
    // else
    // {
    //     int index = index_in_VN_GPU(Checknode_linkVNs, row, begin, Variblenode_linkCNs);
    //     for (int k = 0; k < Nm; k++)
    //     {

    //         sumNonele = GFAdd_GPU(GFMultiply_GPU(23, 45, TableMultiply_GPU), sumNonele, TableAdd_GPU);
    //         sumNonLLR = sumNonLLR + 0.3;

    //         // sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k], Checknode_linkVNs_GF[row * maxdc + begin], TableMultiply_GPU), sumNonele, TableAdd_GPU);
    //         // sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k];
    //         diff += (k != 0) ? 1 : 0;
    //         if (diff <= Nc)
    //         {
    //             ConstructConf_GPU(TableMultiply_GPU, TableAdd_GPU, Nm, Nc, sumNonele, sumNonLLR, diff, begin + 1, except, end, row, EMS_L_c2v, Variblenode_linkCNs, Checknode_linkVNs, Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c);
    //             // sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k], Checknode_linkVNs_GF[row * maxdc + begin], TableMultiply_GPU), sumNonele, TableAdd_GPU);
    //             // sumNonLLR = sumNonLLR - sort_L_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k];

    //             sumNonele = GFAdd_GPU(GFMultiply_GPU(21, 25, TableMultiply_GPU), sumNonele, TableAdd_GPU);
    //             sumNonLLR = sumNonLLR - 0.3;

    //             diff -= (k != 0) ? 1 : 0;
    //         }
    //         else
    //         {
    //             // sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k], Checknode_linkVNs_GF[row * maxdc + begin], TableMultiply_GPU), sumNonele, TableAdd_GPU);
    //             // sumNonLLR = sumNonLLR - sort_L_v2c[Checknode_linkVNs[row * maxdc + begin] * maxdv * GFQ + index * GFQ + k];

    //             sumNonele = GFAdd_GPU(GFMultiply_GPU(34, 42, TableMultiply_GPU), sumNonele, TableAdd_GPU);
    //             sumNonLLR = sumNonLLR - 0.3;

    //             diff -= (k != 0) ? 1 : 0;
    //             break;
    //         }
    //     }
    // }
    // return 0;
    for (int i = 0; i < 4; i++)
    {
        if (i == except)
        {
            continue;
        }
        int index = index_in_VN_GPU(Checknode_linkVNs, row, i, Variblenode_linkCNs);
        sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[row * maxdc + i] * maxdv * GFQ + index * GFQ], Checknode_linkVNs_GF[row * maxdc + i], TableMultiply_GPU), sumNonele, TableAdd_GPU);
        sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[row * maxdc + i] * maxdv * GFQ + index * GFQ];
    }
    if (sumNonLLR > EMS_L_c2v[sumNonele])
    {
        EMS_L_c2v[sumNonele] = sumNonLLR;
    }
    for (int i = 0; i < 4; i++)
    {
        if (i == except)
        {
            continue;
        }
        for (int k = 1; k < GFQ; k++)
        {

            int index = index_in_VN_GPU(Checknode_linkVNs, row, i, Variblenode_linkCNs);
            sumNonele = 0;
            sumNonLLR = 0;
            sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[row * maxdc + i] * maxdv * GFQ + index * GFQ + k], Checknode_linkVNs_GF[row * maxdc + i], TableMultiply_GPU), sumNonele, TableAdd_GPU);
            sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[row * maxdc + i] * maxdv * GFQ + index * GFQ + k];

            for (int j = 0; j < 4; j++)
            {
                if (j == i | j == except)
                {
                    continue;
                }
                int index = index_in_VN_GPU(Checknode_linkVNs, row, j, Variblenode_linkCNs);
                sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[row * maxdc + j] * maxdv * GFQ + index * GFQ], Checknode_linkVNs_GF[row * maxdc + j], TableMultiply_GPU), sumNonele, TableAdd_GPU);
                sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[row * maxdc + j] * maxdv * GFQ + index * GFQ];
            }
            if (sumNonLLR > EMS_L_c2v[sumNonele])
            {
                EMS_L_c2v[sumNonele] = sumNonLLR;
            }
        }
    }
}

void GPUArray_initial(LDPCCode *H, VN *Variablenode, CN *Checknode, int *Checknode_weight, int *Variablenode_linkCNs, int *Checknode_linkVNs, int *Checknode_linkVNs_GF)
{
    // int *Checknode_weight;
    cudaError_t cudaStatus;

    cudaMalloc((void **)&Checknode_weight, H->Checknode_num * sizeof(int));

    int *Checknode_weight_temp = (int *)malloc(H->Checknode_num * sizeof(int));
    for (int i = 0; i < H->Checknode_num; i++)
    {
        Checknode_weight_temp[i] = Checknode[i].weight;
    }
    cudaStatus = cudaMemcpy(Checknode_weight, Checknode_weight_temp, H->Checknode_num * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Cannot copy Checknode_weight\n");
        exit(0);
    }
    free(Checknode_weight_temp);

    // int *Variablenode_linkCNs;
    cudaMalloc((void **)&Variablenode_linkCNs, H->Variablenode_num * maxdv * sizeof(int));

    int *Variablenode_linkCNs_temp = (int *)malloc(H->Variablenode_num * maxdv * sizeof(int));
    for (int i = 0; i < H->Variablenode_num; i++)
    {
        for (int j = 0; j < Variablenode[i].weight; j++)
        {
            Variablenode_linkCNs_temp[i * maxdv + j] = Variablenode[i].linkCNs[j];
        }
    }
    cudaStatus = cudaMemcpy(Variablenode_linkCNs, Variablenode_linkCNs_temp, H->Variablenode_num * maxdv * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Cannot copy Variablenode_linkCNs\n");
        exit(0);
    }
    free(Variablenode_linkCNs_temp);

    // int *Checknode_linkVNs;
    cudaMalloc((void **)&Checknode_linkVNs, H->Checknode_num * maxdc * sizeof(int));

    int *Checknode_linkVNs_temp = (int *)malloc(H->Checknode_num * maxdc * sizeof(int));
    for (int i = 0; i < H->Checknode_num; i++)
    {
        for (int j = 0; j < Checknode[i].weight; j++)
        {
            Checknode_linkVNs_temp[i * maxdc + j] = Checknode[i].linkVNs[j];
        }
    }
    cudaStatus = cudaMemcpy(Checknode_linkVNs, Checknode_linkVNs_temp, H->Checknode_num * maxdc * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Cannot copy Checknode_linkVNs\n");
        exit(0);
    }
    free(Checknode_linkVNs_temp);

    // int *Checknode_linkVNs_GF;
    cudaMalloc((void **)&Checknode_linkVNs_GF, H->Checknode_num * maxdc * sizeof(int));

    int *Checknode_linkVNs_GF_temp = (int *)malloc(H->Checknode_num * maxdc * sizeof(int));
    for (int i = 0; i < H->Checknode_num; i++)
    {
        for (int j = 0; j < Checknode[i].weight; j++)
        {
            Checknode_linkVNs_GF_temp[i * maxdc + j] = Checknode[i].linkVNs_GF[j];
        }
    }
    cudaStatus = cudaMemcpy(Checknode_linkVNs_GF, Checknode_linkVNs_GF_temp, H->Checknode_num * maxdc * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Cannot copy Checknode_linkVNs_GF\n");
        exit(0);
    }
    free(Checknode_linkVNs_GF_temp);
}