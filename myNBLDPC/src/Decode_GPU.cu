#include "Decode_GPU.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>

__device__ void BubleSort_GPU(float a[], int n, int index[])
{
    int i, j;
    float x;
    for (i = 0; i < n; i++)
    {
        for (j = 1; j < n - i; j++)
        {
            if (a[j - 1] < a[j])
            {
                x = a[j];
                a[j] = a[j - 1];
                a[j - 1] = x;
                x = index[j];
                index[j] = index[j - 1];
                index[j - 1] = x;
            }
        }
    }
}

__device__ int SortLLRVector_GPU(int GF, float *Entr_v2c, int *index)
{
    BubleSort_GPU(Entr_v2c, GF, index);
    return 1;
}

__device__ int DecideLLRVector_GPU(float *LLR, int GF)
{
    float max = 0;
    int alpha_i;
    for (int q = 0; q < GF - 1; q++)
    {
        if (LLR[q] > max)
        {
            max = LLR[q];
            alpha_i = q + 1;
        }
    }
    if (max <= 0)
    {
        return 0;
    }
    else
    {
        return alpha_i;
    }
}
__device__ void d_BubleSort_GPU(float a[], int n, int index[])
{
    int i, j;
    float x;
    for (i = 0; i < n; i++)
    {
        for (j = 1; j < n - i; j++)
        {
            if (a[j - 1] > a[j])
            {
                x = a[j];
                a[j] = a[j - 1];
                a[j - 1] = x;
                x = index[j];
                index[j] = index[j - 1];
                index[j - 1] = x;
            }
        }
    }
}

__device__ int d_SortLLRVector_GPU(int GF, float *Entr_v2c, int *index)
{
    d_BubleSort_GPU(Entr_v2c, GF, index);
    return 1;
}

__device__ int d_DecideLLRVector_GPU(float *LLR, int GF)
{
    float min = DBL_MAX;
    int alpha_i;
    for (int q = 0; q < GFQ; q++)
    {
        if (LLR[q] < min)
        {
            min = LLR[q];
            alpha_i = q;
        }
    }
    return alpha_i;
}

__device__ int GetCombCount(int n, int m)
{
    long int i;
    long int a, b, c, s; // s = a/(b*c)
    a = b = c = 1;
    for (i = 1; i <= n; i++)
        a *= i;
    for (i = 1; i <= m; i++)
        b *= i;
    for (i = 1; i <= n - m; i++)
        c *= i;
    s = a / (b * c);
    return s;
}
__device__ void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}
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

int Decoding_EMS_GPU(const LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput, const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const int *Variablenode_weight, const int *Checknode_weight, const int *Variablenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int &iter_number)
{
    cudaError_t cudaStatus;
    // int *sort_Entr_v2c_temp = (int *)malloc(H->Variablenode_num * maxdv * GFQ * sizeof(int));
    // memset(sort_Entr_v2c_temp, 0, H->Variablenode_num * maxdv * GFQ * sizeof(int));
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

    // int *index = (int *)malloc((GFQ) * sizeof(int));

    float *L_ch_temp = (float *)malloc(H->Variablenode_num * (GFQ - 1) * sizeof(float));
    memset(L_ch_temp, 0, H->Variablenode_num * (GFQ - 1) * sizeof(float));

    float *L_ch;
    cudaMalloc((void **)&L_ch, H->Variablenode_num * (GFQ - 1) * sizeof(float));

    float *LLR_temp = (float *)malloc(H->Variablenode_num * (GFQ - 1) * sizeof(float));
    float *LLR;
    cudaMalloc((void **)&LLR, H->Variablenode_num * (GFQ - 1) * sizeof(float));

    for (int col = 0; col < H->Variablenode_num; col++)
    {
        for (int d = 0; d < Variablenode[col].weight; d++)
        {
            Variablenode[col].L_ch[GFQ - 1] = 0;
            for (int q = 0; q < GFQ; q++)
            {
                sort_L_v2c_temp[col * maxdv * GFQ + d * GFQ + q] = Variablenode[col].L_ch[q];
                Variablenode[col].sort_L_v2c[d][q] = Variablenode[col].L_ch[q];
            }
        }
        for (int q = 0; q < GFQ; q++)
        {
            L_ch_temp[col * (GFQ - 1) + q] = Variablenode[col].L_ch[q];
        }
    }
    cudaStatus = cudaMemcpy(L_ch, L_ch_temp, H->Variablenode_num * (GFQ - 1) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Cannot copy L_ch\n");
        exit(0);
    }

    cudaStatus = cudaMemcpy(sort_L_v2c, sort_L_v2c_temp, H->Variablenode_num * maxdv * GFQ * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Cannot copy sort_L_v2c\n");
        exit(0);
    }

    cudaStatus = cudaMemcpy(Checknode_L_c2v, Checknode_L_c2v_temp, H->Checknode_num * maxdc * GFQ * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Cannot copy Checknode_L_c2v\n");
        exit(0);
    }

    for (int row = 0; row < H->Checknode_num; row++)
    {
        for (int d = 0; d < Checknode[row].weight; d++)
        {
            memset(Checknode[row].L_c2v[d], 0, (GFQ - 1) * sizeof(float));
        }
    }

    iter_number = 0;
    bool decode_correct = true;
    int *DecodeOutput_GPU;
    cudaMalloc((void **)&DecodeOutput_GPU, H->Variablenode_num * sizeof(int));

    while (iter_number < maxIT)
    {
        iter_number++;
        Variablenode_EMS<<<H->Variablenode_num,1>>>((const int *)Variablenode_weight, (const int *)Variablenode_linkCNs, sort_Entr_v2c, sort_L_v2c, Checknode_L_c2v, (const float *)L_ch, LLR, DecodeOutput_GPU, H->Variablenode_num);

        cudaStatus = cudaMemcpy(DecodeOutput, DecodeOutput_GPU, H->Variablenode_num * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("Cannot copy DecodeOutput\n");
            exit(0);
        }
        // for (int i = 0; i < H->Variablenode_num; i++)
        // {
        //     for (int q = 0; q < GFQ - 1; q++)
        //     {
        //         Variablenode[i].LLR[q] = LLR_temp[i * (GFQ - 1) + q];
        //     }
        //     DecodeOutput[i] = DecideLLRVector(Variablenode[i].LLR, GFQ);
        // }

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
            cudaFree(LLR);
            cudaFree(L_ch);
            cudaFree(DecodeOutput_GPU);
            free(L_ch_temp);
            free(LLR_temp);
            // free(index);
            // free(sort_Entr_v2c_temp);
            free(sort_L_v2c_temp);
            free(Checknode_L_c2v_temp);
            iter_number--;
            return 1;
        }
        // message from var to check
        // for (int col = 0; col < H->Variablenode_num; col++)
        // {
        //     for (int dv = 0; dv < Variablenode[col].weight; dv++)
        //     {
        //         for (int q = 0; q < GFQ - 1; q++)
        //         {
        //             Variablenode[col].sort_L_v2c[dv][q] = Variablenode[col].LLR[q] - Checknode[Variablenode[col].linkCNs[dv]].L_c2v[index_in_CN(Variablenode, col, dv, Checknode)][q];
        //         }
        //         Variablenode[col].sort_L_v2c[dv][GFQ - 1] = 0;
        //     }
        // }

        // for (int col = 0; col < H->Variablenode_num; col++)
        // {
        //     for (int dv = 0; dv < Variablenode[col].weight; dv++)
        //     {
        //         for (int i = 0; i < GFQ - 1; i++)
        //         {
        //             index[i] = i + 1;
        //         }
        //         index[GFQ - 1] = 0;
        //         SortLLRVector(GFQ, Variablenode[col].sort_L_v2c[dv], index);
        //         for (int i = 0; i < GFQ; i++)
        //         {
        //             Variablenode[col].sort_Entr_v2c[dv][i] = index[i];

        //             sort_Entr_v2c_temp[col * maxdv * GFQ + dv * GFQ + i] = index[i];
        //             sort_L_v2c_temp[col * maxdv * GFQ + dv * GFQ + i] = Variablenode[col].sort_L_v2c[dv][i];
        //         }
        //     }
        // }

        Variablenode_Update_EMS<<<H->Variablenode_num,1>>>((const int *)Variablenode_weight, (const int *)Variablenode_linkCNs, sort_Entr_v2c, sort_L_v2c, Checknode_L_c2v, (const float *)L_ch, LLR, H->Variablenode_num);

        // cudaStatus = cudaMemcpy(sort_Entr_v2c, sort_Entr_v2c_temp, H->Variablenode_num * maxdv * GFQ * sizeof(int), cudaMemcpyHostToDevice);
        // if (cudaStatus != cudaSuccess)
        // {
        //     printf("Cannot copy sort_Entr_v2c\n");
        //     exit(0);
        // }
        // cudaStatus = cudaMemcpy(sort_L_v2c, sort_L_v2c_temp, H->Variablenode_num * maxdv * GFQ * sizeof(float), cudaMemcpyHostToDevice);
        // if (cudaStatus != cudaSuccess)
        // {
        //     printf("Cannot copy sort_L_v2c\n");
        //     exit(0);
        // }
        // // message from check to var

        Checknode_EMS<<<H->Checknode_num,1>>>((const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, EMS_Nm, EMS_Nc, (const int *)Checknode_weight, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c, Checknode_L_c2v, H->Checknode_num);
        // Checknode_EMS<<<1, 1>>>((const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, EMS_Nm, EMS_Nc, (const int *)Checknode_weight, (const int *)Variablenode_linkCNs, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, sort_Entr_v2c, sort_L_v2c, Checknode_L_c2v, H->Checknode_num);

        // cudaStatus = cudaMemcpy(Checknode_L_c2v_temp, Checknode_L_c2v, H->Checknode_num * maxdc * GFQ * sizeof(float), cudaMemcpyDeviceToHost);
        // if (cudaStatus != cudaSuccess)
        // {
        //     printf("Cannot copy Checknode_L_c2v D2V\n");
        //     exit(0);
        // }

        // for (int i = 0; i < H->Checknode_num; i++)
        // {
        //     for (int j = 0; j < Checknode[i].weight; j++)
        //     {
        //         for (int q = 0; q < GFQ - 1; q++)
        //         {
        //             Checknode[i].L_c2v[j][q] = Checknode_L_c2v_temp[i * maxdc * GFQ + j * GFQ + q];
        //         }
        //     }
        // }
    }

    cudaFree(sort_Entr_v2c);
    cudaFree(sort_L_v2c);
    cudaFree(Checknode_L_c2v);
    cudaFree(LLR);
    cudaFree(L_ch);
    free(L_ch_temp);
    free(LLR_temp);
    cudaFree(DecodeOutput_GPU);
    // free(index);
    // free(sort_Entr_v2c_temp);
    free(sort_L_v2c_temp);
    free(Checknode_L_c2v_temp);
    return 0;
}

__global__ void Variablenode_EMS(const int *Variablenode_weight, const int *Variablenode_linkCNs, int *sort_Entr_v2c, float *sort_L_v2c, float *Checknode_L_c2v, const float *L_ch, float *LLR, int *DecodeOutput, int Variablenode_num)
{

    int offset;
    offset = threadIdx.x + blockDim.x * blockIdx.x;
    if (offset < Variablenode_num)
    {
        for (int q = 0; q < GFQ - 1; q++)
        {
            LLR[offset * (GFQ - 1) + q] = L_ch[offset * (GFQ - 1) + q];
        }
        for (int d = 0; d < Variablenode_weight[offset]; d++)
        {
            for (int q = 0; q < GFQ - 1; q++)
            {
                LLR[offset * (GFQ - 1) + q] += Checknode_L_c2v[Variablenode_linkCNs[offset * maxdv + d] + q];
            }
        }
        DecodeOutput[offset] = DecideLLRVector_GPU(LLR + offset * (GFQ - 1), GFQ);
    }
}

__global__ void Variablenode_Update_EMS(const int *Variablenode_weight, const int *Variablenode_linkCNs, int *sort_Entr_v2c, float *sort_L_v2c, float *Checknode_L_c2v, const float *L_ch, float *LLR, int Variablenode_num)
{

    int offset;
    offset = threadIdx.x + blockDim.x * blockIdx.x;
    int *index = (int *)malloc(GFQ * sizeof(int));
    if (offset < Variablenode_num)
    {

        for (int dv = 0; dv < Variablenode_weight[offset]; dv++)
        {
            for (int q = 0; q < GFQ - 1; q++)
            {
                sort_L_v2c[offset * maxdv * GFQ + dv * GFQ + q] = LLR[offset * (GFQ - 1) + q] - Checknode_L_c2v[Variablenode_linkCNs[offset * maxdv + dv] + q];
            }
            sort_L_v2c[offset * maxdv * GFQ + dv * GFQ + GFQ - 1] = 0;
        }
        for (int dv = 0; dv < Variablenode_weight[offset]; dv++)
        {
            for (int i = 0; i < GFQ - 1; i++)
            {
                index[i] = i + 1;
            }
            index[GFQ - 1] = 0;
            SortLLRVector_GPU(GFQ, sort_L_v2c + offset * maxdv * GFQ + dv * GFQ, index);
            for (int i = 0; i < GFQ; i++)
            {
                sort_Entr_v2c[offset * maxdv * GFQ + dv * GFQ + i] = index[i];
            }
        }
    }
    free(index);
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
__global__ void Checknode_EMS(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, int EMS_Nm, int EMS_Nc, const int *Checknode_weight, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int *sort_Entr_v2c, float *sort_L_v2c, float *Checknode_L_c2v, int Checknode_num)
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
                int *bit = new int[Checknode_weight[offset] - 1];
                int EMS_Nc_temp;
                if (EMS_Nc == maxdc - 1)
                {
                    EMS_Nc_temp = Checknode_weight[offset] - 1;
                }
                else
                {
                    EMS_Nc_temp = EMS_Nc;
                }
                for (int choose_n = 2; choose_n <= EMS_Nc_temp; choose_n++)
                {

                    for (int k = 0; k < Checknode_weight[offset] - 1; k++)
                    {
                        if (k < choose_n)
                            bit[k] = 1;
                        else
                            bit[k] = 0;
                    }

                    int i, j, beg, end;
                    int len = Checknode_weight[offset] - 1;
                    int N = GetCombCount(Checknode_weight[offset] - 1, choose_n); //C(n,count)  C(5,3)

                    int *conf_index = (int *)malloc(choose_n * sizeof(int));
                    memset(conf_index, 0, (choose_n) * sizeof(int));

                    int flag = 0;

                    while (!flag)
                    {
                        sumNonele = 0;
                        sumNonLLR = 0;
                        for (int i = 0; i < choose_n; i++)
                        {
                            conf_index[i] += 1; // move confset[i] to smaller one

                            if (i == choose_n - 1 && conf_index[i] == EMS_Nm)
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
                            int t = 0;
                            for (int i = 0; i < Checknode_weight[offset]; i++)
                            {
                                if (i == dc)
                                {
                                    continue;
                                }
                                if (bit[t] == 1)
                                {
                                    sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[offset * maxdc + i] + conf_index[k]], Checknode_linkVNs_GF[offset * maxdc + i], TableMultiply_GPU), sumNonele, TableAdd_GPU);
                                    sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[offset * maxdc + i] + conf_index[k]];
                                    k++;
                                }
                                else
                                {
                                    sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[offset * maxdc + i]], Checknode_linkVNs_GF[offset * maxdc + i], TableMultiply_GPU), sumNonele, TableAdd_GPU);
                                    sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[offset * maxdc + i]];
                                }
                                t++;
                            }
                            if (sumNonLLR > EMS_L_c2v[sumNonele])
                            {
                                EMS_L_c2v[sumNonele] = sumNonLLR;
                            }
                        }
                    }
                    for (j = 1; j < N; j++)
                    {
                        for (i = len - 1; i > 0; i--)
                        {
                            if (bit[i] == 0 && bit[i - 1] == 1)
                            {
                                swap(bit[i], bit[i - 1]);

                                //from index: [i to len-1] , make all bit 1 in the right
                                beg = i;
                                end = len - 1;
                                while (1)
                                {
                                    while (bit[beg] == 1)
                                    {
                                        beg++;
                                        if (beg >= len)
                                            break;
                                    }
                                    while (bit[end] == 0)
                                    {
                                        end--;
                                        if (end < i)
                                            break;
                                    }

                                    if (beg < end)
                                        swap(bit[beg], bit[end]);
                                    else
                                        break;

                                } //end of "while"
                                break;
                            } //end of "if"
                        }
                        flag = 0;
                        memset(conf_index, 0, (choose_n) * sizeof(int));

                        while (!flag)
                        {
                            sumNonele = 0;
                            sumNonLLR = 0;
                            for (int i = 0; i < choose_n; i++)
                            {
                                conf_index[i] += 1; // move confset[i] to smaller one

                                if (i == choose_n - 1 && conf_index[i] == EMS_Nm)
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
                                int t = 0;
                                for (int i = 0; i < Checknode_weight[offset]; i++)
                                {
                                    if (i == dc)
                                    {
                                        continue;
                                    }
                                    if (bit[t] == 1)
                                    {
                                        sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[offset * maxdc + i] + conf_index[k]], Checknode_linkVNs_GF[offset * maxdc + i], TableMultiply_GPU), sumNonele, TableAdd_GPU);
                                        sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[offset * maxdc + i] + conf_index[k]];
                                        k++;
                                    }
                                    else
                                    {
                                        sumNonele = GFAdd_GPU(GFMultiply_GPU(sort_Entr_v2c[Checknode_linkVNs[offset * maxdc + i]], Checknode_linkVNs_GF[offset * maxdc + i], TableMultiply_GPU), sumNonele, TableAdd_GPU);
                                        sumNonLLR = sumNonLLR + sort_L_v2c[Checknode_linkVNs[offset * maxdc + i]];
                                    }
                                    t++;
                                }
                                if (sumNonLLR > EMS_L_c2v[sumNonele])
                                {
                                    EMS_L_c2v[sumNonele] = sumNonLLR;
                                }
                            }
                        }
                    }
                    free(conf_index);
                }

                free(bit);
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

int Decoding_TMM_GPU(const LDPCCode *H, VN *Variablenode, CN *Checknode, int EMS_Nm, int EMS_Nc, int *DecodeOutput, const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const unsigned *TableInverse_GPU, const int *Variablenode_weight, const int *Checknode_weight, const int *Variablenode_linkCNs, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, int &iter_number)
{
    cudaError_t cudaStatus;

    float *sort_L_v2c_temp = (float *)malloc(H->Variablenode_num * maxdv * GFQ * sizeof(float));
    memset(sort_L_v2c_temp, 0, H->Variablenode_num * maxdv * GFQ * sizeof(float));
    float *sort_L_v2c;
    cudaMalloc((void **)&sort_L_v2c, H->Variablenode_num * maxdv * GFQ * sizeof(float));

    float *Checknode_L_c2v_temp = (float *)malloc(H->Checknode_num * maxdc * GFQ * sizeof(float));
    memset(Checknode_L_c2v_temp, 0, H->Checknode_num * maxdc * GFQ * sizeof(float));

    float *Checknode_L_c2v;
    cudaMalloc((void **)&Checknode_L_c2v, H->Checknode_num * maxdc * GFQ * sizeof(float));

    float *LLR_temp = (float *)malloc(H->Variablenode_num * GFQ * sizeof(float));
    memset(LLR_temp, 0, H->Variablenode_num * GFQ * sizeof(float));
    float *LLR;
    cudaMalloc((void **)&LLR, H->Variablenode_num * GFQ * sizeof(float));

    float max = -DBL_MAX;
    for (int col = 0; col < H->Variablenode_num; col++)
    {
        max = -DBL_MAX;
        for (int q = 0; q < GFQ - 1; q++)
        {
            if (Variablenode[col].L_ch[q] > max)
            {
                max = Variablenode[col].L_ch[q];
            }
        }
        for (int d = 0; d < Variablenode[col].weight; d++)
        {
            for (int q = 0; q < GFQ; q++)
            {
                if (q == 0)
                {
                    Variablenode[col].sort_L_v2c[d][q] = max;
                    Variablenode[col].LLR[q] = max;
                    sort_L_v2c_temp[col * maxdv * GFQ + d * GFQ + q] = max;
                    LLR_temp[col * GFQ + q] = max;
                }
                else
                {
                    sort_L_v2c_temp[col * maxdv * GFQ + d * GFQ + q] = max - Variablenode[col].L_ch[q - 1];
                    Variablenode[col].sort_L_v2c[d][q] = max - Variablenode[col].L_ch[q - 1];
                    Variablenode[col].LLR[q] = max - Variablenode[col].L_ch[q - 1];
                    LLR_temp[col * GFQ + q] = max - Variablenode[col].L_ch[q - 1];
                }
            }
        }
    }
    cudaStatus = cudaMemcpy(LLR, LLR_temp, H->Variablenode_num * GFQ * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Cannot copy LLR\n");
        exit(0);
    }

    cudaStatus = cudaMemcpy(sort_L_v2c, sort_L_v2c_temp, H->Variablenode_num * maxdv * GFQ * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Cannot copy sort_L_v2c\n");
        exit(0);
    }

    cudaStatus = cudaMemcpy(Checknode_L_c2v, Checknode_L_c2v_temp, H->Checknode_num * maxdc * GFQ * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        printf("Cannot copy Checknode_L_c2v\n");
        exit(0);
    }

    for (int row = 0; row < H->Checknode_num; row++)
    {
        for (int d = 0; d < Checknode[row].weight; d++)
        {
            memset(Checknode[row].L_c2v[d], 0, GFQ * sizeof(float));
        }
    }

    iter_number = 0;
    bool decode_correct = true;
    int *DecodeOutput_GPU;
    cudaMalloc((void **)&DecodeOutput_GPU, H->Variablenode_num * sizeof(int));

    while (iter_number < maxIT)
    {
        iter_number++;
        Variablenode_TMM<<<H->Variablenode_num,1>>>((const int *)Variablenode_weight, (const int *)Variablenode_linkCNs, sort_L_v2c, Checknode_L_c2v, LLR, DecodeOutput_GPU, H->Variablenode_num);

        cudaStatus = cudaMemcpy(DecodeOutput, DecodeOutput_GPU, H->Variablenode_num * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            printf("Cannot copy DecodeOutput\n");
            exit(0);
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
            cudaFree(sort_L_v2c);
            cudaFree(Checknode_L_c2v);
            cudaFree(LLR);
            cudaFree(DecodeOutput_GPU);
            free(LLR_temp);
            free(sort_L_v2c_temp);
            free(Checknode_L_c2v_temp);
            iter_number--;
            return 1;
        }

        Variablenode_Update_TMM<<<H->Variablenode_num,1>>>((const int *)Variablenode_weight, (const int *)Variablenode_linkCNs, sort_L_v2c, Checknode_L_c2v, LLR, H->Variablenode_num);

        Checknode_TMM<<<H->Checknode_num,1>>>((const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, (const unsigned *)TableInverse_GPU, (const int *)Checknode_weight, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, sort_L_v2c, Checknode_L_c2v, H->Checknode_num);
    }
    cudaFree(sort_L_v2c);
    cudaFree(Checknode_L_c2v);
    cudaFree(LLR);
    free(LLR_temp);
    cudaFree(DecodeOutput_GPU);
    free(sort_L_v2c_temp);
    free(Checknode_L_c2v_temp);
    return 0;
}

__global__ void Variablenode_TMM(const int *Variablenode_weight, const int *Variablenode_linkCNs, float *sort_L_v2c, float *Checknode_L_c2v, float *LLR, int *DecodeOutput, int Variablenode_num)
{

    int offset;
    offset = threadIdx.x + blockDim.x * blockIdx.x;
    if (offset < Variablenode_num)
    {
        for (int d = 0; d < Variablenode_weight[offset]; d++)
        {
            for (int q = 0; q < GFQ; q++)
            {
                LLR[offset * GFQ + q] += Checknode_L_c2v[Variablenode_linkCNs[offset * maxdv + d] + q];
            }
        }
        DecodeOutput[offset] = d_DecideLLRVector_GPU(LLR + offset * GFQ, GFQ);
    }
}

__global__ void Variablenode_Update_TMM(const int *Variablenode_weight, const int *Variablenode_linkCNs, float *sort_L_v2c, float *Checknode_L_c2v, float *LLR, int Variablenode_num)
{

    int offset;
    offset = threadIdx.x + blockDim.x * blockIdx.x;
    if (offset < Variablenode_num)
    {
        for (int dv = 0; dv < Variablenode_weight[offset]; dv++)
        {
            for (int q = 0; q < GFQ; q++)
            {
                sort_L_v2c[offset * maxdv * GFQ + dv * GFQ + q] = LLR[offset * GFQ + q] - Checknode_L_c2v[Variablenode_linkCNs[offset * maxdv + dv] + q];
            }
        }
    }
}

__global__ void Checknode_TMM(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const unsigned *TableInverse_GPU, const int *Checknode_weight, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, float *sort_L_v2c, float *Checknode_L_c2v, int Checknode_num)
{
    int offset;
    offset = threadIdx.x + blockDim.x * blockIdx.x;
    if (offset < Checknode_num)
    {
        int *TMM_Zn = (int *)malloc(maxdc * sizeof(int));
        float *TMM_deltaU = (float *)malloc(maxdc * GFQ * sizeof(float));
        float *TMM_Min1 = (float *)malloc(GFQ * sizeof(float));
        float *TMM_Min2 = (float *)malloc(GFQ * sizeof(float));
        int *TMM_Min1_Col = (int *)malloc(GFQ * sizeof(int));
        float *TMM_I = (float *)malloc(GFQ * sizeof(float));
        int *TMM_Path = (int *)malloc(GFQ * 2 * sizeof(int));
        float *TMM_E = (float *)malloc(GFQ * sizeof(float));
        float *TMM_Lc2p = (float *)malloc(GFQ * sizeof(float));
        int TMM_Syndrome = 0;
        d_TMM_Get_Zn_GPU((const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, (const unsigned *)TableInverse_GPU, (const int *)Checknode_weight, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, sort_L_v2c, Checknode_L_c2v, TMM_Zn, offset, TMM_Syndrome);

        d_TMM_Get_deltaU_GPU((const unsigned *)TableMultiply_GPU, (const unsigned *)TableAdd_GPU, (const unsigned *)TableInverse_GPU, (const int *)Checknode_weight, (const int *)Checknode_linkVNs, (const int *)Checknode_linkVNs_GF, sort_L_v2c, Checknode_L_c2v, TMM_Zn, TMM_deltaU, offset);

        TMM_Get_Min_GPU(Checknode_weight, TMM_Zn, TMM_deltaU, TMM_Min1, TMM_Min2, TMM_Min1_Col, offset);

        TMM_ConstructConf_GPU((const unsigned *)TableInverse_GPU, TMM_deltaU, TMM_Min1, TMM_Min2, TMM_Min1_Col, TMM_I, TMM_Path, TMM_E);

        for (int dc = 0; dc < Checknode_weight[offset]; dc++)
        {
            // choose to output
            TMM_Lc2p[0] = 0;
            for (int eta = 1; eta < GFQ; eta++)
            {
                if (dc != TMM_Path[eta * 2 + 0] && dc != TMM_Path[eta * 2 + 1])
                {
                    TMM_Lc2p[eta] = TMM_I[eta];
                }
                else
                {
                    TMM_Lc2p[eta] = TMM_E[eta];
                }
            }

            int h_inverse = GFInverse_GPU(Checknode_linkVNs_GF[offset * maxdc + dc], TableInverse_GPU);
            int beta_syn = GFAdd_GPU(TMM_Syndrome, TMM_Zn[dc], TableAdd_GPU);
            double L0 = TMM_Lc2p[beta_syn];
            for (int eta = 0; eta < GFQ; eta++)
            {
                int beta =
                    GFMultiply_GPU(h_inverse, GFAdd_GPU(eta, beta_syn, TableAdd_GPU), TableMultiply_GPU);
                Checknode_L_c2v[offset * maxdc * GFQ + dc * GFQ + beta] = (TMM_Lc2p[eta]) * 0.8;
            }
        }
        free(TMM_Zn);
        free(TMM_deltaU);
        free(TMM_Min1);
        free(TMM_Min2);
        free(TMM_Min1_Col);
        free(TMM_I);
        free(TMM_Path);
        free(TMM_E);
        free(TMM_Lc2p);
    }
}

__device__ int d_TMM_Get_Zn_GPU(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const unsigned *TableInverse_GPU, const int *Checknode_weight, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, float *sort_L_v2c, float *Checknode_L_c2v, int *TMM_Zn, int row, int &TMM_Syndrome)
{
    TMM_Syndrome = 0;
    for (int dc = 0; dc < Checknode_weight[row]; dc++)
    {
        double min = DBL_MAX;
        int min_ele = 0;
        for (int q = 0; q < GFQ; q++)
        {
            if (sort_L_v2c[Checknode_linkVNs[row * maxdc + dc] + q] < min)
            {
                min = sort_L_v2c[Checknode_linkVNs[row * maxdc + dc] + q];
                min_ele = GFMultiply_GPU(q, Checknode_linkVNs_GF[row * maxdc + dc], TableMultiply_GPU);
            }
        }
        TMM_Zn[dc] = min_ele;
        TMM_Syndrome = GFAdd_GPU(TMM_Syndrome, min_ele, TableAdd_GPU);
    }
    return 0;
}

__device__ int d_TMM_Get_deltaU_GPU(const unsigned *TableMultiply_GPU, const unsigned *TableAdd_GPU, const unsigned *TableInverse_GPU, const int *Checknode_weight, const int *Checknode_linkVNs, const int *Checknode_linkVNs_GF, float *sort_L_v2c, float *Checknode_L_c2v, int *TMM_Zn, float *TMM_deltaU, int row)
{
    for (int dc = 0; dc < Checknode_weight[row]; dc++)
    {

        int h_inverse = GFInverse_GPU(Checknode_linkVNs_GF[row * maxdc + dc], TableInverse_GPU);

        int beta_p = GFMultiply_GPU(h_inverse, TMM_Zn[dc], TableMultiply_GPU);
        float min = sort_L_v2c[Checknode_linkVNs[row * maxdc + dc] + beta_p];

        for (int x = 0; x < GFQ; x++)
        {
            int eta = GFAdd_GPU(x, TMM_Zn[dc], TableAdd_GPU);
            TMM_deltaU[dc * GFQ + eta] =
                sort_L_v2c[Checknode_linkVNs[row * maxdc + dc] + GFMultiply_GPU(h_inverse, x, TableMultiply_GPU)] - min;
        }
    }
    return 0;
}

__device__ int TMM_Get_Min_GPU(const int *Checknode_weight, int *TMM_Zn, float *TMM_deltaU, float *TMM_Min1, float *TMM_Min2, int *TMM_Min1_Col, int row)
{
    // sort
    for (int q = 0; q < GFQ; q++)
    {
        // clear
        TMM_Min1[q] = DBL_MAX;
        TMM_Min2[q] = DBL_MAX;
        // search min and submin
        for (int dc = 0; dc < Checknode_weight[row]; dc++)
        {
            if (TMM_deltaU[dc * GFQ + q] < TMM_Min1[q])
            {
                TMM_Min2[q] = TMM_Min1[q];
                TMM_Min1[q] = TMM_deltaU[dc * GFQ + q];
                TMM_Min1_Col[q] = dc;
            }
            else if (TMM_deltaU[dc * GFQ + q] < TMM_Min2[q])
            {
                TMM_Min2[q] = TMM_deltaU[dc * GFQ + q];
            }
        }
    }

    return 0;
}

__device__ int TMM_ConstructConf_GPU(const unsigned *TableAdd_GPU, float *TMM_deltaU, float *TMM_Min1, float *TMM_Min2, int *TMM_Min1_Col, float *TMM_I, int *TMM_Path, float *TMM_E)
{
    // dQ[0]
    TMM_I[0] = 0;
    TMM_Path[0] = TMM_Path[1] = -1;
    TMM_E[0] = 0;

    double deviation1, deviation2;
    for (int i = 1; i < GFQ; i++)
    {
        // 1 deviation
        TMM_I[i] = TMM_deltaU[TMM_Min1_Col[i] * GFQ + i];
        TMM_Path[i * 2 + 0] = TMM_Path[i * 2 + 1] = TMM_Min1_Col[i];
        TMM_E[i] = TMM_Min2[i];

        // 2 deviation
        for (int j = 0; j < GFQ; j++)
        {
            if (j != i)
            {
                int k = GFAdd_GPU(i, j, TableAdd_GPU);
                if (TMM_Min1_Col[j] != TMM_Min1_Col[k]) // 不在同一列
                {
                    deviation1 = TMM_deltaU[TMM_Min1_Col[j] * GFQ + j];
                    deviation2 = TMM_deltaU[TMM_Min1_Col[k] * GFQ + k];
                    if (deviation1 > deviation2 && deviation1 < TMM_I[i])
                    {
                        TMM_I[i] = deviation1;
                        TMM_Path[i * 2 + 0] = TMM_Min1_Col[j];
                        TMM_Path[i * 2 + 1] = TMM_Min1_Col[k];
                        TMM_E[i] = TMM_Min1[i];
                    }
                    else if (deviation1 < deviation2 &&
                             deviation2 < TMM_I[i])
                    {
                        TMM_I[i] = deviation2;
                        TMM_Path[i * 2 + 0] = TMM_Min1_Col[j];
                        TMM_Path[i * 2 + 1] = TMM_Min1_Col[k];
                        TMM_E[i] = TMM_Min1[i];
                    }
                }
            }
        }
    }
    return 0;
}