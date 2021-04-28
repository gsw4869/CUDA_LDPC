#include "Decode_GPU.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>

__device__ int GFAdd_GPU(int ele1, int ele2, const unsigned *TableAdd_GPU)
{
    return ele1 ^ ele2;
}

__device__ int GFMultiply_GPU(int ele1, int ele2, const unsigned *TableMultiply_GPU)
{
    return TableMultiply_GPU[GFQ * ele1 + ele2];
}

__device__ int GFInverse_GPU(int ele, const unsigned *TableInverse_GPU)
{
    if (ele == 0)
    {
        printf("Div 0 Error!\n");
    }
    return TableInverse_GPU[ele];
}

int Decoding_EMS_GPU(const LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput, const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const int *Checknode_weight, const int *Variablenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int &iter_number)
{

    for (int col = 0; col < H->Variablenode_num; col++)
    {
        for (int d = 0; d < Variablenode[col].weight; d++)
        {
            memcpy(Variablenode[col].sort_L_v2c[d], Variablenode[col].L_ch, (GFQ - 1) * sizeof(float));
        }
    }

    for (int row = 0; row < H->Checknode_num; row++)
    {
        for (int d = 0; d < Checknode[row].weight; d++)
        {
            memset(Checknode[row].L_c2v[d], 0, (GFQ - 1) * sizeof(float));
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

    int *index = (int *)malloc((GFQ) * sizeof(int));

    iter_number = 0;
    bool decode_correct = true;
    while (iter_number++ < maxIT - 1)
    {
        for (int col = 0; col < H->Variablenode_num; col++)
        {
            memcpy(Variablenode[col].LLR, Variablenode[col].L_ch, (GFQ - 1) * sizeof(float));
        }
        for (int col = 0; col < H->Variablenode_num; col++)
        {
            for (int d = 0; d < Variablenode[col].weight; d++)
            {
                for (int q = 0; q < GFQ - 1; q++)
                {
                    Variablenode[col].LLR[q] += Checknode[Variablenode[col].linkCNs[d]].L_c2v[index_in_CN(Variablenode, col, d, Checknode)][q];
                }
            }
            DecodeOutput[col] = DecideLLRVector(Variablenode[col].LLR, GFQ);
        }

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
            cudaFree(sort_Entr_v2c);
            cudaFree(sort_L_v2c);
            cudaFree(Checknode_L_c2v);
            free(index);
            free(sort_Entr_v2c_temp);
            free(sort_L_v2c_temp);
            free(Checknode_L_c2v_temp);
            return 1;
        }

        // message from var to check
        for (int col = 0; col < H->Variablenode_num; col++)
        {
            for (int dv = 0; dv < Variablenode[col].weight; dv++)
            {
                for (int q = 0; q < GFQ - 1; q++)
                {
                    Variablenode[col].sort_L_v2c[dv][q] = Variablenode[col].LLR[q] - Checknode[Variablenode[col].linkCNs[dv]].L_c2v[index_in_CN(Variablenode, col, dv, Checknode)][q];
                }
                Variablenode[col].sort_L_v2c[dv][GFQ - 1] = 0;
            }
        }

        for (int col = 0; col < H->Variablenode_num; col++)
        {
            for (int dv = 0; dv < Variablenode[col].weight; dv++)
            {
                for (int i = 0; i < GFQ - 1; i++)
                {
                    index[i] = i + 1;
                }
                index[GFQ - 1] = 0;
                SortLLRVector(GFQ, Variablenode[col].sort_L_v2c[dv], index);
                for (int i = 0; i < GFQ; i++)
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

        Checknode_EMS<<<((H->Checknode_num % 128) ? (H->Checknode_num / 128 + 1) : (H->Checknode_num / 128)), 128>>>((const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, EMS_Nm, EMS_Nc, (const int *)Checknode_weight, (const int *)Variablenode_linkCNs, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c, Checknode_L_c2v, H->Checknode_num);
        // Checknode_EMS<<<1, 1>>>((const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, EMS_Nm, EMS_Nc, (const int *)Checknode_weight, (const int *)Variablenode_linkCNs, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c, Checknode_L_c2v, H->Checknode_num);

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
    free(index);
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
__global__ void Checknode_EMS(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, int EMS_Nm, int EMS_Nc, const int *Checknode_weight, const int *Variblenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int *sort_Entr_v2c, float *sort_L_v2c, float *Checknode_L_c2v, int Checknode_num)
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
                int sumNonele;
                float sumNonLLR;
                // conf(q, 1)
                sumNonele = 0;
                sumNonLLR = 0;
                // ConstructConf_GPU((const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, GFQ, 1, sumNonele, sumNonLLR, diff, 0, dc, Checknode_weight[offset] - 1, offset, EMS_L_c2v, (const int *)Variblenode_linkCNs, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c);

                for (int i = 0; i < Checknode_weight[offset]; i++)
                {
                    if (i == dc)
                    {
                        continue;
                    }

                    sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[offset * maxdc + i]], Checknode_linkVNs_GF[offset * maxdc + i], TableMultiply_GPU), sumNonele, TableAdd_GPU);
                    sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[offset * maxdc + i]];
                }
                if (sumNonLLR > EMS_L_c2v[sumNonele])
                {
                    EMS_L_c2v[sumNonele] = sumNonLLR;
                }
                int sumNonele_all_max = sumNonele;
                float sumNonLLR_all_max = sumNonLLR;
                for (int i = 0; i < Checknode_weight[offset]; i++)
                {
                    if (i == dc)
                    {
                        continue;
                    }

                    sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[offset * maxdc + i]], Checknode_linkVNs_GF[offset * maxdc + i], TableMultiply_GPU), sumNonele_all_max, TableAdd_GPU);
                    sumNonLLR = sumNonLLR_all_max - sort_L_v2c[Checknode_linkVNs[offset * maxdc + i]];

                    for (int k = 1; k < GFQ; k++)
                    {

                        int sumNonele1 = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[offset * maxdc + i] + k], Checknode_linkVNs_GF[offset * maxdc + i], TableMultiply_GPU), sumNonele, TableAdd_GPU);
                        float sumNonLLR1 = sumNonLLR + sort_L_v2c[Checknode_linkVNs[offset * maxdc + i] + k];

                        if (sumNonLLR1 > EMS_L_c2v[sumNonele1])
                        {
                            EMS_L_c2v[sumNonele1] = sumNonLLR1;
                        }
                    }
                }

                // conf(nm, nc)
                // sumNonele = 0;
                // sumNonLLR = 0;
                // diff = 0;
                // ConstructConf_GPU((const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, EMS_Nm, EMS_Nc, sumNonele, sumNonLLR, diff, 0, dc, Checknode_weight[offset] - 1, offset, EMS_L_c2v, (const int *)Variblenode_linkCNs, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c);

                int *conf_index = (int *)malloc((Checknode_weight[offset] - 1) * sizeof(int));
                memset(conf_index, 0, (Checknode_weight[offset] - 1) * sizeof(int));

                int flag = 0;

                while (!flag)
                {
                    sumNonele = 0;
                    sumNonLLR = 0;
                    for (int i = 0; i < Checknode_weight[offset] - 1; i++)
                    {
                        conf_index[i] += 1; // move confset[i] to smaller one

                        if (i == Checknode_weight[offset] - 2 && conf_index[i] == EMS_Nm)
                        { // reaches end
                            flag = 1;
                            break;
                        }
                        else if (conf_index[i] >= EMS_Nm)
                        {
                            conf_index[i] = 0;
                            // continue to modify next VN
                        }
                        else
                        {
                            break; // don't modify next VN
                        }
                    }
                    if (!flag)
                    {
                        int k = 0;
                        for (int i = 0; i < Checknode_weight[offset]; i++)
                        {
                            if (i == dc)
                            {
                                continue;
                            }

                            sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[offset * maxdc + i] + conf_index[k]], Checknode_linkVNs_GF[offset * maxdc + i], TableMultiply_GPU), sumNonele, TableAdd_GPU);
                            sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[offset * maxdc + i] + conf_index[k]];
                            k++;
                        }
                        if (sumNonLLR > EMS_L_c2v[sumNonele])
                        {
                            EMS_L_c2v[sumNonele] = sumNonLLR;
                        }
                    }
                }
                free(conf_index);
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
__device__ int ConstructConf_GPU(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, int Nm, int Nc, int &sumNonele, float &sumNonLLR, int &diff, int begin, int except, int end, int row, float *EMS_L_c2v, const int *Variblenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int *sort_Entr_v2c, float *sort_L_v2c)
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
}