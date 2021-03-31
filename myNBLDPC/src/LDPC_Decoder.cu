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