
#include "define.h"
#include "struct.h"
#include "GF.h"

void freeVN(const LDPCCode *H, VN *A)
{
    int threadNum =
        THREAD_NUM ? THREAD_NUM : std::thread::hardware_concurrency();
    for (int i = 0; i < H->Variablenode_num; i++)
    {
        for (int j = 0; j < threadNum; j++)
        {
            free(A[j * H->Variablenode_num + i].linkCNs);
            free(A[j * H->Variablenode_num + i].linkCNs_GF);
            free(A[j * H->Variablenode_num + i].LLR);
            free(A[j * H->Variablenode_num + i].L_ch);
            free(A[j * H->Variablenode_num + i].sort_L_v2c[0]);
            free(A[j * H->Variablenode_num + i].sort_L_v2c);
            free(A[j * H->Variablenode_num + i].sort_Entr_v2c[0]);
            free(A[j * H->Variablenode_num + i].sort_Entr_v2c);
        }
    }
    free(A);
}

void freeCN(const LDPCCode *H, CN *A)
{
    int threadNum =
        THREAD_NUM ? THREAD_NUM : std::thread::hardware_concurrency();
    for (int i = 0; i < H->Checknode_num; i++)
    {
        for (int j = 0; j < threadNum; j++)
        {
            free(A[j * H->Checknode_num + i].linkVNs);
            free(A[j * H->Checknode_num + i].linkVNs_GF);
            free(A[j * H->Checknode_num + i].L_c2v[0]);
            free(A[j * H->Checknode_num + i].L_c2v);
        }
    }
    free(A);
}